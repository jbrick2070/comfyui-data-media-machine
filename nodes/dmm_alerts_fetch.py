"""
DMMAlertsFetch — Active NWS weather alerts + CAL FIRE incidents.
NO API KEY NEEDED.

Endpoints:
  NWS: https://api.weather.gov/alerts/active
  CAL FIRE: https://incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList

v3.5 changes:
  - Added CAL FIRE wildfire incident feed (GeoJSON, no key, filters by distance)
  - NWS + CAL FIRE merged into unified alert output when source="nws_live"
  - New source option: "calfire_only" for fire-specific queries

Creative use: Alert severity → narrative urgency, red alerts → visual distortion,
multiple active alerts → layered tension, active fires → smoke/haze overlays.
"""

import time
import random


class DMMAlertsFetch:
    """Fetches active NWS weather alerts. No key required. US only."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_alerts"
    RETURN_TYPES = ("DMM_ALERTS", "STRING",)
    RETURN_NAMES = ("alerts_data", "alerts_summary",)
    OUTPUT_NODE = False

    _SEVERITY_ORDER = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
    _URGENCY_ORDER = {"Immediate": 4, "Expected": 3, "Future": 2, "Past": 1, "Unknown": 0}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "source": (["nws_live", "calfire_only", "demo_clear",
                            "demo_heat_advisory", "demo_fire_warning",
                            "demo_multi_alert"],),
            },
        }

    def fetch_alerts(self, config, source):
        if source.startswith("demo_"):
            data = self._demo_alerts(source, config)
        elif source == "nws_live":
            data = self._fetch_nws_alerts(config)
            # v3.5: Merge CAL FIRE incidents into NWS alerts
            fire_data = self._fetch_calfire(config)
            if fire_data.get("fire_incidents"):
                for fire in fire_data["fire_incidents"]:
                    data["alerts"].append(fire)
                    data["count"] += 1
                data["fire_count"] = fire_data["fire_count"]
                data["nearest_fire"] = fire_data.get("nearest_fire")
                # Re-sort by severity after merge
                data["alerts"].sort(
                    key=lambda a: self._SEVERITY_ORDER.get(a["severity"], 0),
                    reverse=True
                )
                data["alerts"] = data["alerts"][:8]  # cap at 8
                # Update top event if fire is more severe
                if data["alerts"]:
                    data["top_event"] = data["alerts"][0]["event"]
                    data["top_headline"] = data["alerts"][0]["headline"]
            else:
                data["fire_count"] = 0
                data["nearest_fire"] = None
        elif source == "calfire_only":
            data = self._fetch_calfire_standalone(config)
        else:
            data = self._demo_alerts("demo_clear", config)

        summary = self._build_summary(data, config["city"])
        return (data, summary)

    def _http_get_json(self, url, headers=None, timeout=12):
        try:
            import requests
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass
        except Exception as e:
            print(f"[DMM Alerts] requests failed: {e}")
        try:
            import urllib.request
            import json
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[DMM Alerts] urllib also failed: {e}")
            return None

    def _fetch_nws_alerts(self, config):
        """
        NWS Alerts API — NO KEY NEEDED.
        Query by lat/lon point to get active alerts for that area.
        """
        headers = {"User-Agent": "ComfyUI-DataMediaMachine/2.0"}

        # Use the point-based alert query
        url = (
            f"https://api.weather.gov/alerts/active"
            f"?point={config['lat']},{config['lon']}"
            f"&status=actual"
        )

        raw = self._http_get_json(url, headers=headers)
        if raw is None or "features" not in raw:
            print("[DMM Alerts] NWS alerts failed, falling back to demo")
            return self._demo_alerts("demo_clear", config)

        alerts = []
        for f in raw["features"]:
            props = f.get("properties", {})
            alerts.append({
                "event": props.get("event", "Unknown Alert"),
                "headline": props.get("headline", ""),
                "severity": props.get("severity", "Unknown"),
                "urgency": props.get("urgency", "Unknown"),
                "certainty": props.get("certainty", "Unknown"),
                "description": (props.get("description", "") or "")[:500],
                "instruction": (props.get("instruction", "") or "")[:300],
                "area": props.get("areaDesc", ""),
                "sender_name": props.get("senderName", ""),
                "effective": props.get("effective", ""),
                "expires": props.get("expires", ""),
            })

        # Sort by severity
        alerts.sort(
            key=lambda a: self._SEVERITY_ORDER.get(a["severity"], 0),
            reverse=True
        )

        # Determine overall alert level
        if not alerts:
            alert_level = "all_clear"
            alert_mood = "no active alerts, calm and safe"
        else:
            top_sev = alerts[0]["severity"]
            if top_sev == "Extreme":
                alert_level = "extreme"
                alert_mood = "extreme danger, take immediate action"
            elif top_sev == "Severe":
                alert_level = "severe"
                alert_mood = "severe conditions, heightened caution advised"
            elif top_sev == "Moderate":
                alert_level = "moderate"
                alert_mood = "moderate advisory, be aware of conditions"
            else:
                alert_level = "minor"
                alert_mood = "minor advisory, stay informed"

        return {
            "alerts": alerts[:5],  # top 5 most severe
            "count": len(alerts),
            "alert_level": alert_level,
            "alert_mood": alert_mood,
            "top_event": alerts[0]["event"] if alerts else "None",
            "top_headline": alerts[0]["headline"] if alerts else "No active alerts",
            "source": "nws_live",
            "live": True,
        }

    def _fetch_calfire(self, config):
        """
        CAL FIRE Active Incidents — NO KEY NEEDED.
        GeoJSON endpoint, filter by distance to configured lat/lon.
        Returns fire incidents normalized to the same alert schema.
        """
        import math

        url = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList?inactive=false"
        raw = self._http_get_json(url, timeout=10)
        if raw is None or "features" not in raw:
            print("[DMM Alerts] CAL FIRE API failed or empty")
            return {"fire_incidents": [], "fire_count": 0, "nearest_fire": None}

        fires = []
        lat0, lon0 = config["lat"], config["lon"]
        for f in raw["features"]:
            props = f.get("properties", {})
            coords = f.get("geometry", {}).get("coordinates", [0, 0])
            fire_lat = props.get("Latitude", coords[1] if len(coords) > 1 else 0)
            fire_lon = props.get("Longitude", coords[0] if len(coords) > 0 else 0)

            # Distance in miles (haversine approximation)
            dlat = math.radians(fire_lat - lat0)
            dlon = math.radians(fire_lon - lon0)
            a = (math.sin(dlat / 2) ** 2 +
                 math.cos(math.radians(lat0)) * math.cos(math.radians(fire_lat)) *
                 math.sin(dlon / 2) ** 2)
            dist_miles = 3959 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            # Only include fires within 200 miles
            if dist_miles > 200:
                continue

            acres = props.get("AcresBurned", 0) or 0
            contained = props.get("PercentContained", 0) or 0

            # Map fire size to alert severity
            if acres >= 1000 and contained < 50:
                severity = "Extreme"
            elif acres >= 500 or contained < 30:
                severity = "Severe"
            elif acres >= 100:
                severity = "Moderate"
            else:
                severity = "Minor"

            fires.append({
                "event": f"Wildfire: {props.get('Name', 'Unknown')}",
                "headline": (f"{props.get('Name', 'Unknown')} Fire — "
                             f"{int(acres)} acres, {int(contained)}% contained, "
                             f"{int(dist_miles)} mi away"),
                "severity": severity,
                "urgency": "Immediate" if contained < 50 else "Expected",
                "certainty": "Observed",
                "description": (f"{props.get('Type', 'Wildfire')} in "
                                f"{props.get('County', 'Unknown')} County. "
                                f"{props.get('Location', '')}. "
                                f"Admin: {props.get('AdminUnit', '')}"),
                "instruction": "",
                "area": f"{props.get('County', 'Unknown')} County",
                "sender_name": props.get("AdminUnit", "CAL FIRE"),
                "effective": props.get("Started", ""),
                "expires": "",
                "source_type": "calfire",
                "acres_burned": acres,
                "percent_contained": contained,
                "distance_miles": round(dist_miles, 1),
                "fire_lat": fire_lat,
                "fire_lon": fire_lon,
            })

        fires.sort(key=lambda f: f["distance_miles"])
        nearest = fires[0] if fires else None

        return {
            "fire_incidents": fires[:5],  # top 5 nearest
            "fire_count": len(fires),
            "nearest_fire": {
                "name": nearest["event"],
                "distance_miles": nearest["distance_miles"],
                "acres": nearest["acres_burned"],
                "contained": nearest["percent_contained"],
            } if nearest else None,
        }

    def _fetch_calfire_standalone(self, config):
        """CAL FIRE as sole alert source (no NWS merge)."""
        fire_data = self._fetch_calfire(config)
        alerts = fire_data.get("fire_incidents", [])

        if not alerts:
            alert_level, alert_mood = "all_clear", "no active fires in region"
        else:
            top = alerts[0]["severity"]
            if top == "Extreme":
                alert_level = "extreme"
                alert_mood = "active wildfire emergency, extreme danger"
            elif top == "Severe":
                alert_level = "severe"
                alert_mood = "significant wildfire activity, heightened alert"
            else:
                alert_level = "moderate"
                alert_mood = "fire activity in region, monitor conditions"

        return {
            "alerts": alerts,
            "count": len(alerts),
            "alert_level": alert_level,
            "alert_mood": alert_mood,
            "top_event": alerts[0]["event"] if alerts else "None",
            "top_headline": alerts[0]["headline"] if alerts else "No active fires",
            "fire_count": fire_data["fire_count"],
            "nearest_fire": fire_data.get("nearest_fire"),
            "source": "calfire_live",
            "live": True,
        }

    def _demo_alerts(self, mode, config):
        if mode == "demo_clear":
            return {
                "alerts": [],
                "count": 0,
                "alert_level": "all_clear",
                "alert_mood": "no active alerts, calm and safe",
                "top_event": "None",
                "top_headline": "No active alerts for Los Angeles",
                "source": mode,
                "live": False,
            }

        presets = {
            "demo_heat_advisory": [{
                "event": "Excessive Heat Warning",
                "headline": "Excessive Heat Warning issued for Los Angeles County",
                "severity": "Severe",
                "urgency": "Expected",
                "certainty": "Likely",
                "description": "Dangerously hot conditions with temperatures up to 110F expected. "
                               "Drink plenty of fluids, stay in air-conditioned rooms, stay out of the sun.",
                "instruction": "An Excessive Heat Warning means that a period of very hot temperatures "
                               "will occur. Take precautions to avoid heat-related illness.",
                "area": "Los Angeles County",
                "sender_name": "NWS Los Angeles/Oxnard CA",
                "effective": "", "expires": "",
            }],
            "demo_fire_warning": [
                {
                    "event": "Red Flag Warning",
                    "headline": "Red Flag Warning for Los Angeles and Ventura County Mountains",
                    "severity": "Extreme",
                    "urgency": "Immediate",
                    "certainty": "Observed",
                    "description": "Critical fire weather conditions expected. Strong Santa Ana winds "
                                   "with gusts to 60 mph combined with very low humidity.",
                    "instruction": "A Red Flag Warning means critical fire conditions are imminent or occurring.",
                    "area": "Los Angeles County Mountains, Ventura County",
                    "sender_name": "NWS Los Angeles/Oxnard CA",
                    "effective": "", "expires": "",
                },
                {
                    "event": "Wind Advisory",
                    "headline": "Wind Advisory for Santa Ana winds",
                    "severity": "Moderate",
                    "urgency": "Expected",
                    "certainty": "Likely",
                    "description": "Northeast winds 25 to 40 mph with gusts to 55 mph.",
                    "instruction": "Secure outdoor objects. Drive with caution.",
                    "area": "Los Angeles County",
                    "sender_name": "NWS Los Angeles/Oxnard CA",
                    "effective": "", "expires": "",
                },
            ],
            "demo_multi_alert": [
                {
                    "event": "Flash Flood Warning",
                    "headline": "Flash Flood Warning for central Los Angeles County",
                    "severity": "Severe",
                    "urgency": "Immediate",
                    "certainty": "Observed",
                    "description": "Flash flooding is occurring in recently burned areas. "
                                   "Debris flows and mudslides possible.",
                    "instruction": "Move to higher ground now. Do not attempt to cross flooded roads.",
                    "area": "Los Angeles County burn scar areas",
                    "sender_name": "NWS Los Angeles/Oxnard CA",
                    "effective": "", "expires": "",
                },
                {
                    "event": "Air Quality Alert",
                    "headline": "Air Quality Alert due to wildfire smoke",
                    "severity": "Moderate",
                    "urgency": "Expected",
                    "certainty": "Likely",
                    "description": "Wildfire smoke is impacting air quality across the basin.",
                    "instruction": "Sensitive groups should limit outdoor activity.",
                    "area": "South Coast Air Basin",
                    "sender_name": "South Coast AQMD",
                    "effective": "", "expires": "",
                },
            ],
        }

        alerts = presets.get(mode, [])
        alerts.sort(
            key=lambda a: self._SEVERITY_ORDER.get(a["severity"], 0),
            reverse=True
        )

        if not alerts:
            alert_level, alert_mood = "all_clear", "no active alerts"
        else:
            top = alerts[0]["severity"]
            if top == "Extreme":
                alert_level, alert_mood = "extreme", "extreme danger, take action"
            elif top == "Severe":
                alert_level, alert_mood = "severe", "severe conditions, caution"
            else:
                alert_level, alert_mood = "moderate", "advisory in effect"

        return {
            "alerts": alerts,
            "count": len(alerts),
            "alert_level": alert_level,
            "alert_mood": alert_mood,
            "top_event": alerts[0]["event"] if alerts else "None",
            "top_headline": alerts[0]["headline"] if alerts else "All clear",
            "source": mode,
            "live": False,
        }

    def _build_summary(self, data, city):
        live_tag = " [LIVE]" if data.get("live") else " [DEMO]"
        if data["count"] == 0:
            return f"{city} Alerts{live_tag}: All clear. No active advisories."
        return (
            f"{city} Alerts{live_tag}: "
            f"{data['count']} active | "
            f"Level: {data['alert_level'].upper()} | "
            f"{data['top_event']}"
        )
