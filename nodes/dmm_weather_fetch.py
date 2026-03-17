"""
DMMWeatherFetch — Real-time weather from Open-Meteo or NWS.
NO API KEY NEEDED for either source.

Open-Meteo: https://open-meteo.com — global, free, no key
NWS: https://api.weather.gov — US only, free, no key

All heavy imports (requests, urllib, json) are inside execute()
to prevent boot crashes if requests isn't installed.
"""

# Only stdlib at module scope — learned this the hard way
import time
import random


class DMMWeatherFetch:
    """Fetches live weather data. No API keys required."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_weather"
    RETURN_TYPES = ("DMM_WEATHER", "STRING",)
    RETURN_NAMES = ("weather_data", "weather_summary",)
    OUTPUT_NODE = False

    # WMO Weather Code → condition string mapping
    # Used by Open-Meteo (they return integer codes, not text)
    _WMO_CODES = {
        0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing Rime Fog",
        51: "Drizzle Light", 53: "Drizzle Moderate", 55: "Drizzle Dense",
        56: "Freezing Drizzle Light", 57: "Freezing Drizzle Dense",
        61: "Rain Slight", 63: "Rain Moderate", 65: "Rain Heavy",
        66: "Freezing Rain Light", 67: "Freezing Rain Heavy",
        71: "Snow Light", 73: "Snow Moderate", 75: "Snow Heavy",
        77: "Snow Grains",
        80: "Rain Showers Slight", 81: "Rain Showers Moderate",
        82: "Rain Showers Violent",
        85: "Snow Showers Slight", 86: "Snow Showers Heavy",
        95: "Thunderstorm", 96: "Thunderstorm Slight Hail",
        99: "Thunderstorm Heavy Hail",
    }

    # Simplified condition grouping for creative mapping
    _WMO_GROUPS = {
        0: "Clear", 1: "Clear", 2: "Clouds", 3: "Clouds",
        45: "Fog", 48: "Fog",
        51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
        56: "Drizzle", 57: "Drizzle",
        61: "Rain", 63: "Rain", 65: "Rain",
        66: "Rain", 67: "Rain",
        71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
        80: "Rain", 81: "Rain", 82: "Rain",
        85: "Snow", 86: "Snow",
        95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "source": (["open_meteo", "nws_gov", "demo_storm",
                            "demo_heatwave", "demo_fog", "demo_random"],),
            },
        }

    def fetch_weather(self, config, source):
        if source.startswith("demo_"):
            data = self._demo_weather(source, config)
        elif source == "open_meteo":
            data = self._fetch_open_meteo(config)
        elif source == "nws_gov":
            data = self._fetch_nws(config)
        else:
            data = self._demo_weather("demo_random", config)

        summary = self._build_summary(data, config["city"])
        return (data, summary)

    def _http_get_json(self, url, headers=None, timeout=12):
        """
        HTTP GET with graceful fallback.
        Tries requests first (better error handling), falls back to urllib.
        This is inside a method, not at module scope — safe pattern.
        """
        # Try requests first
        try:
            import requests
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass
        except Exception as e:
            print(f"[DMM Weather] requests failed: {e}")
            # Fall through to urllib

        # Fallback: urllib (always available, no pip needed)
        try:
            import urllib.request
            import json
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            print(f"[DMM Weather] urllib also failed: {e}")
            return None

    def _fetch_open_meteo(self, config):
        """
        Open-Meteo Current Weather API — NO KEY NEEDED.
        Endpoint: https://api.open-meteo.com/v1/forecast?...&current=...
        """
        params = (
            f"latitude={config['lat']}&longitude={config['lon']}"
            f"&current=temperature_2m,relative_humidity_2m,"
            f"apparent_temperature,precipitation,weather_code,"
            f"cloud_cover,wind_speed_10m,wind_direction_10m,"
            f"wind_gusts_10m,surface_pressure"
            f"&temperature_unit=fahrenheit"
            f"&wind_speed_unit=mph"
            f"&precipitation_unit=inch"
        )
        url = f"https://api.open-meteo.com/v1/forecast?{params}"

        raw = self._http_get_json(url)
        if raw is None or "current" not in raw:
            print("[DMM Weather] Open-Meteo failed, falling back to demo")
            return self._demo_weather("demo_random", config)

        c = raw["current"]
        wmo_code = c.get("weather_code", 0)
        condition = self._WMO_GROUPS.get(wmo_code, "Clear")
        description = self._WMO_CODES.get(wmo_code, "Unknown")

        return {
            "temp_f": c.get("temperature_2m", 72),
            "feels_like_f": c.get("apparent_temperature", 72),
            "humidity": c.get("relative_humidity_2m", 50),
            "pressure_hpa": c.get("surface_pressure", 1013),
            "wind_speed_mph": c.get("wind_speed_10m", 5),
            "wind_deg": c.get("wind_direction_10m", 0),
            "wind_gust_mph": c.get("wind_gusts_10m", 0),
            "clouds_pct": c.get("cloud_cover", 50),
            "precipitation_inch": c.get("precipitation", 0),
            "rain_1h_mm": c.get("precipitation", 0) * 25.4,  # inch to mm
            "snow_1h_mm": 0,
            "wmo_code": wmo_code,
            "condition": condition,
            "description": description,
            "visibility_m": 10000,  # Open-Meteo current doesn't give visibility
            "source": "open_meteo",
            "live": True,
        }

    def _fetch_nws(self, config):
        """
        NWS Current Observation — NO KEY NEEDED, US only.
        Two-step: points → station → latest observation
        """
        headers = {"User-Agent": "ComfyUI-DataMediaMachine/2.0 (jeffrey@brick.dev)"}

        # Step 1: gridpoint lookup
        point_url = f"https://api.weather.gov/points/{config['lat']},{config['lon']}"
        point = self._http_get_json(point_url, headers=headers)
        if point is None:
            print("[DMM Weather] NWS point lookup failed")
            return self._demo_weather("demo_random", config)

        try:
            obs_stations_url = point["properties"]["observationStations"]
        except (KeyError, TypeError):
            print("[DMM Weather] NWS response missing observationStations")
            return self._demo_weather("demo_random", config)

        # Step 2: get nearest station
        stations = self._http_get_json(obs_stations_url, headers=headers)
        if stations is None or not stations.get("features"):
            return self._demo_weather("demo_random", config)
        station_id = stations["features"][0]["properties"]["stationIdentifier"]

        # Step 3: latest observation
        obs_url = f"https://api.weather.gov/stations/{station_id}/observations/latest"
        obs_resp = self._http_get_json(obs_url, headers=headers)
        if obs_resp is None:
            return self._demo_weather("demo_random", config)

        try:
            obs = obs_resp["properties"]
        except (KeyError, TypeError):
            return self._demo_weather("demo_random", config)

        # NWS returns metric, convert
        temp_c = obs.get("temperature", {}).get("value")
        temp_f = (temp_c * 9 / 5 + 32) if temp_c is not None else 72

        wind_ms = obs.get("windSpeed", {}).get("value") or 0
        wind_mph = wind_ms * 2.237

        gust_ms = obs.get("windGust", {}).get("value") or 0
        gust_mph = gust_ms * 2.237

        vis_m = obs.get("visibility", {}).get("value") or 10000
        humidity = obs.get("relativeHumidity", {}).get("value") or 50
        pressure_pa = obs.get("barometricPressure", {}).get("value") or 101325
        text_desc = obs.get("textDescription", "Unknown")

        # Map NWS text to our condition groups
        text_lower = text_desc.lower()
        if "thunder" in text_lower:
            condition = "Thunderstorm"
        elif "snow" in text_lower or "blizzard" in text_lower:
            condition = "Snow"
        elif "rain" in text_lower or "shower" in text_lower:
            condition = "Rain"
        elif "drizzle" in text_lower:
            condition = "Drizzle"
        elif "fog" in text_lower or "mist" in text_lower:
            condition = "Fog"
        elif "cloud" in text_lower or "overcast" in text_lower:
            condition = "Clouds"
        else:
            condition = "Clear"

        return {
            "temp_f": round(temp_f, 1),
            "feels_like_f": round(temp_f, 1),
            "humidity": round(humidity, 1),
            "pressure_hpa": round(pressure_pa / 100, 1),
            "wind_speed_mph": round(wind_mph, 1),
            "wind_deg": obs.get("windDirection", {}).get("value") or 0,
            "wind_gust_mph": round(gust_mph, 1),
            "clouds_pct": 50,  # NWS doesn't give cloud %
            "precipitation_inch": 0,
            "rain_1h_mm": 0,
            "snow_1h_mm": 0,
            "wmo_code": -1,
            "condition": condition,
            "description": text_desc,
            "visibility_m": vis_m,
            "source": "nws_gov",
            "live": True,
        }

    def _demo_weather(self, mode, config):
        """Synthetic weather for offline testing."""
        rng = random.Random(config.get("seed", 42))

        presets = {
            "demo_storm": dict(
                temp_f=58, humidity=92, wind_speed_mph=35, wind_gust_mph=55,
                clouds_pct=100, visibility_m=800, condition="Thunderstorm",
                description="Thunderstorm Heavy Hail", rain_1h_mm=22,
                pressure_hpa=998, wmo_code=99,
            ),
            "demo_heatwave": dict(
                temp_f=108, humidity=12, wind_speed_mph=8, wind_gust_mph=15,
                clouds_pct=5, visibility_m=16000, condition="Clear",
                description="Clear", rain_1h_mm=0,
                pressure_hpa=1018, wmo_code=0,
            ),
            "demo_fog": dict(
                temp_f=55, humidity=98, wind_speed_mph=2, wind_gust_mph=3,
                clouds_pct=100, visibility_m=100, condition="Fog",
                description="Fog", rain_1h_mm=0,
                pressure_hpa=1022, wmo_code=45,
            ),
        }

        if mode in presets:
            base = presets[mode]
        else:
            wmo = rng.choice([0, 2, 3, 45, 61, 65, 71, 80, 95])
            base = dict(
                temp_f=round(rng.uniform(32, 105), 1),
                humidity=rng.randint(10, 100),
                wind_speed_mph=round(rng.uniform(0, 40), 1),
                wind_gust_mph=round(rng.uniform(0, 60), 1),
                clouds_pct=rng.randint(0, 100),
                visibility_m=rng.randint(100, 16000),
                condition=self._WMO_GROUPS.get(wmo, "Clear"),
                description=self._WMO_CODES.get(wmo, "Unknown"),
                rain_1h_mm=round(rng.uniform(0, 15), 1) if wmo >= 61 else 0,
                pressure_hpa=round(rng.uniform(990, 1035), 1),
                wmo_code=wmo,
            )

        return {
            **base,
            "feels_like_f": base["temp_f"] + round(rng.uniform(-5, 5), 1),
            "wind_deg": rng.randint(0, 360),
            "precipitation_inch": base.get("rain_1h_mm", 0) / 25.4,
            "snow_1h_mm": round(rng.uniform(0, 5), 1) if base["condition"] == "Snow" else 0,
            "source": mode,
            "live": False,
        }

    def _build_summary(self, data, city):
        live_tag = " [LIVE]" if data.get("live") else " [DEMO]"
        return (
            f"{city}{live_tag}: {data['description']} | "
            f"{data['temp_f']:.0f}\u00b0F | "
            f"Humidity {data['humidity']}% | "
            f"Wind {data['wind_speed_mph']:.0f} mph | "
            f"Clouds {data['clouds_pct']}%"
        )
