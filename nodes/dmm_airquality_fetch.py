"""
DMMAirQualityFetch — Air quality from Open-Meteo (model) or AirNow (sensor).

Tier 1 (default, no key): Open-Meteo model-based AQ
  Endpoint: https://air-quality-api.open-meteo.com/v1/air-quality

Tier 2 (optional, free key): EPA AirNow real sensor data
  Endpoint: https://www.airnowapi.org/aq/observation/latLong/current/
  Key stored in: config.json → airnow_api_key
  Register free at: https://docs.airnowapi.org/

v3.5 changes:
  - Added AirNow sensor source (real EPA monitor data)
  - Auto-loads API key from config.json if present
  - Falls back to Open-Meteo if key missing or AirNow fails

Creative use: AQI drives haze/clarity in visuals, UV drives lighting intensity.
"""

import time
import random


class DMMAirQualityFetch:
    """Fetches live air quality data. No API keys required."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_air_quality"
    RETURN_TYPES = ("DMM_AIRQUALITY", "STRING",)
    RETURN_NAMES = ("aq_data", "aq_summary",)
    OUTPUT_NODE = False

    _AQI_LABELS = [
        (0, 50, "Good", "clean, crisp atmosphere"),
        (51, 100, "Moderate", "slight atmospheric haze"),
        (101, 150, "Unhealthy for Sensitive Groups", "visible haze, muted distances"),
        (151, 200, "Unhealthy", "thick haze, orange-brown sky tones"),
        (201, 300, "Very Unhealthy", "dense smog, apocalyptic orange sky"),
        (301, 500, "Hazardous", "toxic atmosphere, zero visibility, dystopian"),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "source": (["open_meteo", "airnow_sensor", "demo_clean",
                            "demo_smoggy", "demo_hazardous", "demo_random"],),
            },
        }

    def fetch_air_quality(self, config, source):
        if source.startswith("demo_"):
            data = self._demo_aq(source, config)
        elif source == "airnow_sensor":
            data = self._fetch_airnow(config)
        elif source == "open_meteo":
            data = self._fetch_open_meteo_aq(config)
        else:
            data = self._demo_aq("demo_random", config)

        summary = self._build_summary(data, config["city"])
        return (data, summary)

    def _http_get_json(self, url, timeout=12):
        """Same safe HTTP pattern as weather node."""
        try:
            import requests
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass
        except Exception as e:
            print(f"[DMM AirQ] requests failed: {e}")

        try:
            import urllib.request
            import json
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[DMM AirQ] urllib also failed: {e}")
            return None

    def _fetch_open_meteo_aq(self, config):
        """
        Open-Meteo Air Quality API — NO KEY NEEDED.
        Returns current hour's data from the hourly arrays.
        """
        params = (
            f"latitude={config['lat']}&longitude={config['lon']}"
            f"&current=us_aqi,pm10,pm2_5,carbon_monoxide,"
            f"nitrogen_dioxide,sulphur_dioxide,ozone,"
            f"uv_index,european_aqi"
        )
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?{params}"

        raw = self._http_get_json(url)
        if raw is None or "current" not in raw:
            print("[DMM AirQ] Open-Meteo AQ failed, falling back to demo")
            return self._demo_aq("demo_random", config)

        c = raw["current"]
        us_aqi = c.get("us_aqi", 50)
        label, creative_desc = self._aqi_to_label(us_aqi)

        return {
            "us_aqi": us_aqi,
            "eu_aqi": c.get("european_aqi", 50),
            "pm25": c.get("pm2_5", 10),
            "pm10": c.get("pm10", 20),
            "ozone": c.get("ozone", 50),
            "no2": c.get("nitrogen_dioxide", 10),
            "so2": c.get("sulphur_dioxide", 5),
            "co": c.get("carbon_monoxide", 200),
            "uv_index": c.get("uv_index", 3),
            "aqi_label": label,
            "creative_desc": creative_desc,
            "source": "open_meteo",
            "live": True,
        }

    def _load_api_key(self, key_name):
        """Load an API key from config.json next to the media_machine package."""
        import os, json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )
        if not os.path.exists(config_path):
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get(key_name)
        except Exception:
            return None

    def _fetch_airnow(self, config):
        """
        EPA AirNow — REAL SENSOR DATA. Requires free API key in config.json.
        Falls back to Open-Meteo if key missing or request fails.
        """
        api_key = self._load_api_key("airnow_api_key")
        if not api_key:
            print("[DMM AirQ] No AirNow API key in config.json, falling back to Open-Meteo")
            return self._fetch_open_meteo_aq(config)

        url = (
            f"https://www.airnowapi.org/aq/observation/latLong/current/"
            f"?format=application/json"
            f"&latitude={config['lat']}&longitude={config['lon']}"
            f"&distance=25"
            f"&API_KEY={api_key}"
        )

        raw = self._http_get_json(url)
        if raw is None or not isinstance(raw, list) or len(raw) == 0:
            print("[DMM AirQ] AirNow failed, falling back to Open-Meteo")
            return self._fetch_open_meteo_aq(config)

        # Parse AirNow response — each entry is one pollutant
        # e.g. [{"ParameterName": "O3", "AQI": 22, ...}, {"ParameterName": "PM2.5", ...}]
        sensor_data = {}
        reporting_area = ""
        for entry in raw:
            param = entry.get("ParameterName", "")
            aqi_val = entry.get("AQI", -1)
            reporting_area = entry.get("ReportingArea", reporting_area)
            if param == "O3":
                sensor_data["ozone_aqi"] = aqi_val
            elif param == "PM2.5":
                sensor_data["pm25_aqi"] = aqi_val
            elif param == "PM10":
                sensor_data["pm10_aqi"] = aqi_val

        # Use the highest AQI as overall
        aqi_values = [v for v in sensor_data.values() if v >= 0]
        us_aqi = max(aqi_values) if aqi_values else 50
        label, creative_desc = self._aqi_to_label(us_aqi)

        return {
            "us_aqi": us_aqi,
            "eu_aqi": int(us_aqi * 0.8),  # rough conversion
            "pm25": sensor_data.get("pm25_aqi", 0),
            "pm10": sensor_data.get("pm10_aqi", 0),
            "ozone": sensor_data.get("ozone_aqi", 0),
            "no2": 0,   # AirNow basic doesn't return these
            "so2": 0,
            "co": 0,
            "uv_index": 0,
            "aqi_label": label,
            "creative_desc": creative_desc,
            "reporting_area": reporting_area,
            "data_type": "sensor",
            "source": "airnow_sensor",
            "live": True,
        }

    def _aqi_to_label(self, aqi):
        for lo, hi, label, desc in self._AQI_LABELS:
            if lo <= aqi <= hi:
                return label, desc
        return "Unknown", "undefined atmospheric quality"

    def _demo_aq(self, mode, config):
        rng = random.Random(config.get("seed", 42))

        presets = {
            "demo_clean": dict(us_aqi=25, pm25=5, pm10=10, uv_index=6, ozone=30),
            "demo_smoggy": dict(us_aqi=155, pm25=65, pm10=90, uv_index=8, ozone=100),
            "demo_hazardous": dict(us_aqi=320, pm25=250, pm10=350, uv_index=11, ozone=180),
        }

        if mode in presets:
            base = presets[mode]
        else:
            base = dict(
                us_aqi=rng.randint(10, 200),
                pm25=round(rng.uniform(2, 80), 1),
                pm10=round(rng.uniform(5, 120), 1),
                uv_index=round(rng.uniform(0, 11), 1),
                ozone=round(rng.uniform(20, 150), 1),
            )

        label, creative_desc = self._aqi_to_label(base["us_aqi"])

        return {
            **base,
            "eu_aqi": int(base["us_aqi"] * 0.8),
            "no2": round(rng.uniform(5, 60), 1),
            "so2": round(rng.uniform(1, 30), 1),
            "co": round(rng.uniform(100, 800), 1),
            "aqi_label": label,
            "creative_desc": creative_desc,
            "source": mode,
            "live": False,
        }

    def _build_summary(self, data, city):
        live_tag = " [LIVE]" if data.get("live") else " [DEMO]"
        return (
            f"{city} Air Quality{live_tag}: "
            f"US AQI {data['us_aqi']} ({data['aqi_label']}) | "
            f"PM2.5: {data['pm25']} | UV: {data['uv_index']}"
        )
