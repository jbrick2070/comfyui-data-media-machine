"""
DMMEnergyGrid — California ISO real-time grid demand data.
NO API KEY NEEDED.

Endpoint: https://oasis.caiso.com/oasisapi/SingleZip
Returns hourly demand forecasts (MW) by TAC area — LADWP, SCE, SDGE, etc.

v3.5: Initial implementation.
  - Fetches Day-Ahead Market (DAM) system load forecast
  - Filters to LADWP + SCE-TAC for LA-area demand
  - Computes grid_stress (0.0-1.0) from current hour vs daily peak
  - Returns current MW, daily peak, grid mood for narrative

Creative use: Grid stress → city pulse intensity, peak demand →
visual saturation, low demand (3am) → eerie calm.

Author: Jeffrey A. Brick
"""

import io
import time
import random
import zipfile
import csv
import datetime


class DMMEnergyGrid:
    """Fetches California ISO grid demand. No API key required."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_grid"
    RETURN_TYPES = ("DMM_ENERGY", "STRING",)
    RETURN_NAMES = ("energy_data", "energy_summary",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "source": (["caiso_live", "demo_peak", "demo_overnight",
                            "demo_blackout"],),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-fetch

    def fetch_grid(self, config, source):
        if source.startswith("demo_"):
            data = self._demo_grid(source, config)
        elif source == "caiso_live":
            data = self._fetch_caiso(config)
        else:
            data = self._demo_grid("demo_peak", config)

        summary = self._build_summary(data, config["city"])
        return (data, summary)

    def _http_get_bytes(self, url, timeout=20):
        """Fetch raw bytes (ZIP file) from CAISO OASIS."""
        try:
            import requests
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "DMM-DataMediaMachine/3.0"
            })
            resp.raise_for_status()
            return resp.content
        except ImportError:
            pass
        except Exception as e:
            print(f"[DMM Energy] requests failed: {e}")

        try:
            import urllib.request
            req = urllib.request.Request(url, headers={
                "User-Agent": "DMM-DataMediaMachine/3.0"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            print(f"[DMM Energy] urllib also failed: {e}")
            return None

    def _fetch_caiso(self, config):
        """
        CAISO OASIS API — NO KEY NEEDED.
        Fetches Day-Ahead Market system load forecast.
        Returns demand in MW by TAC area, hourly.
        """
        now = datetime.datetime.utcnow()
        date_str = now.strftime("%Y%m%d")
        start = f"{date_str}T00:00-0000"
        end = f"{date_str}T23:59-0000"

        url = (
            f"https://oasis.caiso.com/oasisapi/SingleZip"
            f"?queryname=SLD_FCST"
            f"&startdatetime={start}"
            f"&enddatetime={end}"
            f"&market_run_id=DAM"
            f"&resultformat=6"
            f"&version=1"
        )

        raw_bytes = self._http_get_bytes(url)
        if raw_bytes is None:
            print("[DMM Energy] CAISO fetch failed, falling back to demo")
            return self._demo_grid("demo_peak", config)

        # Parse ZIP → CSV
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_name:
                    print("[DMM Energy] No CSV in CAISO ZIP")
                    return self._demo_grid("demo_peak", config)

                with zf.open(csv_name[0]) as f:
                    reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                    rows = list(reader)
        except Exception as e:
            print(f"[DMM Energy] Failed to parse CAISO ZIP/CSV: {e}")
            return self._demo_grid("demo_peak", config)

        # Filter to LA-area TAC zones
        la_zones = {"LADWP", "SCE-TAC"}
        la_rows = [r for r in rows if r.get("TAC_AREA_NAME") in la_zones]

        if not la_rows:
            print("[DMM Energy] No LADWP/SCE data in CAISO response")
            return self._demo_grid("demo_peak", config)

        # Build hourly demand totals (LADWP + SCE combined)
        hourly = {}
        for r in la_rows:
            try:
                hr = int(r.get("OPR_HR", 0))
                mw = float(r.get("MW", 0))
                hourly[hr] = hourly.get(hr, 0) + mw
            except (ValueError, TypeError):
                continue

        if not hourly:
            return self._demo_grid("demo_peak", config)

        # Get current hour in Pacific Time (DST-aware)
        try:
            import zoneinfo
            tz_pt = zoneinfo.ZoneInfo("America/Los_Angeles")
        except ImportError:
            try:
                import pytz
                tz_pt = pytz.timezone("America/Los_Angeles")
            except ImportError:
                tz_pt = None
        if tz_pt:
            local_hour = datetime.datetime.now(tz_pt).hour
        else:
            local_hour = (now.hour - 7) % 24  # fallback: assume PST
        # Find closest hour in data
        current_hr = min(hourly.keys(), key=lambda h: abs(h - local_hour))
        current_mw = hourly[current_hr]
        peak_mw = max(hourly.values())
        trough_mw = min(hourly.values())
        peak_hr = max(hourly, key=hourly.get)
        avg_mw = sum(hourly.values()) / len(hourly)

        # Grid stress: 0.0 (minimum demand) to 1.0 (at peak)
        demand_range = peak_mw - trough_mw
        if demand_range > 0:
            grid_stress = (current_mw - trough_mw) / demand_range
        else:
            grid_stress = 0.5

        # Grid mood
        if grid_stress > 0.85:
            grid_mood = "grid strained, peak demand, city buzzing with energy"
        elif grid_stress > 0.65:
            grid_mood = "high demand, evening rush, lights blazing across the basin"
        elif grid_stress > 0.40:
            grid_mood = "moderate load, daytime hum, city in motion"
        elif grid_stress > 0.20:
            grid_mood = "low demand, quiet hours, city winding down"
        else:
            grid_mood = "minimal load, deep night, city at rest"

        return {
            "current_mw": round(current_mw, 1),
            "peak_mw": round(peak_mw, 1),
            "trough_mw": round(trough_mw, 1),
            "avg_mw": round(avg_mw, 1),
            "peak_hour": peak_hr,
            "current_hour": current_hr,
            "grid_stress": round(grid_stress, 3),
            "grid_mood": grid_mood,
            "zones": list(la_zones),
            "hourly_mw": {str(k): round(v, 1) for k, v in sorted(hourly.items())},
            "source": "caiso_live",
            "live": True,
        }

    def _demo_grid(self, mode, config):
        rng = random.Random(config.get("seed", 42))

        if mode == "demo_peak":
            current_mw = 18500
            peak_mw = 19000
            trough_mw = 11500
            grid_stress = 0.92
            grid_mood = "grid strained, peak demand, city buzzing with energy"
        elif mode == "demo_overnight":
            current_mw = 12000
            peak_mw = 19000
            trough_mw = 11500
            grid_stress = 0.07
            grid_mood = "minimal load, deep night, city at rest"
        elif mode == "demo_blackout":
            current_mw = 20500
            peak_mw = 20500
            trough_mw = 11500
            grid_stress = 1.0
            grid_mood = "grid at breaking point, rolling blackout risk, city stressed"
        else:
            current_mw = round(rng.uniform(11000, 20000), 1)
            peak_mw = 19500
            trough_mw = 11200
            grid_stress = round((current_mw - trough_mw) / (peak_mw - trough_mw), 3)
            grid_mood = "moderate load, daytime hum, city in motion"

        return {
            "current_mw": current_mw,
            "peak_mw": peak_mw,
            "trough_mw": trough_mw,
            "avg_mw": round((peak_mw + trough_mw) / 2, 1),
            "peak_hour": 17,
            "current_hour": datetime.datetime.now().hour,
            "grid_stress": grid_stress,
            "grid_mood": grid_mood,
            "zones": ["LADWP", "SCE-TAC"],
            "hourly_mw": {},
            "source": mode,
            "live": False,
        }

    def _build_summary(self, data, city):
        live_tag = " [LIVE]" if data.get("live") else " [DEMO]"
        return (
            f"{city} Grid{live_tag}: "
            f"{data['current_mw']} MW (peak: {data['peak_mw']} MW) | "
            f"Stress: {data['grid_stress']:.0%} | "
            f"{data['grid_mood']}"
        )
