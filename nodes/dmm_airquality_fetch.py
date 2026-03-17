"""
DMMAirQualityFetch — Real-time air quality from Open-Meteo.
NO API KEY NEEDED.

Endpoint: https://air-quality-api.open-meteo.com/v1/air-quality
Returns: PM2.5, PM10, US AQI, European AQI, UV index, etc.

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
                "source": (["open_meteo", "demo_clean", "demo_smoggy",
                            "demo_hazardous", "demo_random"],),
            },
        }

    def fetch_air_quality(self, config, source):
        if source.startswith("demo_"):
            data = self._demo_aq(source, config)
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
