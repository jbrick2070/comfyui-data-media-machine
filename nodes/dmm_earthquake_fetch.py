"""
DMMEarthquakeFetch — Real-time earthquake data from USGS.
NO API KEY NEEDED.

Endpoint: https://earthquake.usgs.gov/fdsnws/event/1/query
Returns recent quakes within a radius of the config coordinates.

Creative use: Magnitude → visual intensity, depth → bass rumble,
recent quake count → tension/unease in narrative.
"""

import time
import random


class DMMEarthquakeFetch:
    """Fetches recent earthquakes near configured location. No key required."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_earthquakes"
    RETURN_TYPES = ("DMM_QUAKE", "STRING",)
    RETURN_NAMES = ("quake_data", "quake_summary",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "source": (["usgs_live", "demo_quiet", "demo_active",
                            "demo_big_one"],),
                "lookback_hours": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 168,
                    "step": 1,
                    "tooltip": "How many hours back to search for quakes"
                }),
                "radius_km": ("INT", {
                    "default": 150,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "tooltip": "Search radius in km from your coordinates"
                }),
                "min_magnitude": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 9.0,
                    "step": 0.5,
                    "tooltip": "Minimum magnitude to include"
                }),
            },
        }

    def fetch_earthquakes(self, config, source, lookback_hours,
                           radius_km, min_magnitude):
        if source.startswith("demo_"):
            data = self._demo_quake(source, config)
        elif source == "usgs_live":
            data = self._fetch_usgs(config, lookback_hours,
                                     radius_km, min_magnitude)
        else:
            data = self._demo_quake("demo_quiet", config)

        summary = self._build_summary(data, config["city"])
        return (data, summary)

    def _http_get_json(self, url, timeout=15):
        try:
            import requests
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass
        except Exception as e:
            print(f"[DMM Quake] requests failed: {e}")

        try:
            import urllib.request
            import json
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[DMM Quake] urllib also failed: {e}")
            return None

    def _fetch_usgs(self, config, lookback_hours, radius_km, min_mag):
        """
        USGS FDSN Event Query — NO KEY NEEDED.
        GeoJSON format, filter by lat/lon radius and time window.
        """
        import datetime
        now = datetime.datetime.utcnow()
        start = now - datetime.timedelta(hours=lookback_hours)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S")

        url = (
            f"https://earthquake.usgs.gov/fdsnws/event/1/query"
            f"?format=geojson"
            f"&latitude={config['lat']}&longitude={config['lon']}"
            f"&maxradiuskm={radius_km}"
            f"&starttime={start_str}"
            f"&minmagnitude={min_mag}"
            f"&orderby=time"
            f"&limit=20"
        )

        raw = self._http_get_json(url)
        if raw is None or "features" not in raw:
            print("[DMM Quake] USGS failed, falling back to demo")
            return self._demo_quake("demo_quiet", config)

        quakes = []
        for f in raw["features"]:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [0, 0, 0])
            quakes.append({
                "magnitude": props.get("mag", 0),
                "place": props.get("place", "Unknown location"),
                "time_epoch": props.get("time", 0),
                "depth_km": coords[2] if len(coords) > 2 else 0,
                "type": props.get("type", "earthquake"),
                "felt": props.get("felt"),  # number of felt reports
                "alert": props.get("alert"),  # green/yellow/orange/red
                "tsunami": props.get("tsunami", 0),
                "sig": props.get("sig", 0),  # significance 0-1000
                "lat": coords[1] if len(coords) > 1 else 0,
                "lon": coords[0] if len(coords) > 0 else 0,
            })

        # Compute summary stats
        mags = [q["magnitude"] for q in quakes if q["magnitude"]]
        max_mag = max(mags) if mags else 0
        avg_mag = (sum(mags) / len(mags)) if mags else 0

        # Seismic mood
        if max_mag >= 5.0:
            seismic_mood = "major seismic event, high alert"
        elif max_mag >= 3.0:
            seismic_mood = "noticeable seismic activity, light tremors felt"
        elif len(quakes) > 10:
            seismic_mood = "seismic swarm, frequent micro-quakes, restless earth"
        elif len(quakes) > 3:
            seismic_mood = "mild seismic background, earth murmuring"
        elif len(quakes) > 0:
            seismic_mood = "occasional micro-quakes, barely perceptible"
        else:
            seismic_mood = "seismically quiet, stable ground"

        return {
            "quakes": quakes[:10],  # top 10 most recent
            "count": len(quakes),
            "max_magnitude": round(max_mag, 1),
            "avg_magnitude": round(avg_mag, 1),
            "seismic_mood": seismic_mood,
            "lookback_hours": lookback_hours,
            "radius_km": radius_km,
            "source": "usgs_live",
            "live": True,
        }

    def _demo_quake(self, mode, config):
        rng = random.Random(config.get("seed", 42))

        if mode == "demo_quiet":
            quakes = [{
                "magnitude": 1.2, "place": "8km NW of Inglewood, CA",
                "depth_km": 7, "type": "earthquake", "felt": None,
                "alert": None, "tsunami": 0, "sig": 15,
                "lat": 34.02, "lon": -118.37, "time_epoch": 0,
            }]
            mood = "seismically quiet, stable ground"
        elif mode == "demo_active":
            quakes = []
            for i in range(8):
                quakes.append({
                    "magnitude": round(rng.uniform(1.0, 3.5), 1),
                    "place": rng.choice([
                        "5km S of Pasadena, CA", "12km W of Malibu, CA",
                        "3km N of Compton, CA", "15km E of Northridge, CA",
                        "7km SW of Beverly Hills, CA",
                    ]),
                    "depth_km": round(rng.uniform(2, 20), 1),
                    "type": "earthquake", "felt": rng.choice([None, 3, 12]),
                    "alert": None, "tsunami": 0,
                    "sig": rng.randint(10, 100),
                    "lat": 34.0 + rng.uniform(-0.3, 0.3),
                    "lon": -118.3 + rng.uniform(-0.3, 0.3),
                    "time_epoch": 0,
                })
            mood = "seismic swarm, frequent micro-quakes, restless earth"
        elif mode == "demo_big_one":
            quakes = [{
                "magnitude": 6.4,
                "place": "2km NE of Northridge, CA",
                "depth_km": 12, "type": "earthquake", "felt": 14500,
                "alert": "orange", "tsunami": 0, "sig": 850,
                "lat": 34.24, "lon": -118.54, "time_epoch": 0,
            }]
            for i in range(12):  # aftershocks
                quakes.append({
                    "magnitude": round(rng.uniform(2.0, 4.5), 1),
                    "place": f"Aftershock #{i+1} near Northridge",
                    "depth_km": round(rng.uniform(3, 15), 1),
                    "type": "earthquake", "felt": rng.choice([None, 5, 30, 200]),
                    "alert": None, "tsunami": 0,
                    "sig": rng.randint(20, 300),
                    "lat": 34.24 + rng.uniform(-0.1, 0.1),
                    "lon": -118.54 + rng.uniform(-0.1, 0.1),
                    "time_epoch": 0,
                })
            mood = "major seismic event, high alert"
        else:
            quakes = []
            mood = "seismically quiet, stable ground"

        mags = [q["magnitude"] for q in quakes]
        return {
            "quakes": quakes[:10],
            "count": len(quakes),
            "max_magnitude": max(mags) if mags else 0,
            "avg_magnitude": round(sum(mags) / len(mags), 1) if mags else 0,
            "seismic_mood": mood,
            "lookback_hours": 24,
            "radius_km": 150,
            "source": mode,
            "live": False,
        }

    def _build_summary(self, data, city):
        live_tag = " [LIVE]" if data.get("live") else " [DEMO]"
        return (
            f"{city} Seismic{live_tag}: "
            f"{data['count']} quakes in {data['lookback_hours']}h | "
            f"Max: M{data['max_magnitude']} | "
            f"{data['seismic_mood']}"
        )
