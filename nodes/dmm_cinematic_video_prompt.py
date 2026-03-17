"""
DMMCinematicVideoPrompt — Data-driven video+audio prompts.

v3.1 changes:
  - Removed dead `rng` variable from both v1 and v2 generate_video_prompt
  - Added logging (was completely silent before)
  - Added IS_CHANGED to v2 (time-dependent prompt output)
  - Added webcam_strict style to inputs and disabled camera movement for it

Two versions coexist for backward compatibility:
  - DMMCinematicVideoPrompt (v1/legacy): 2 outputs (prompt, label).
    Keeps existing v2.5 workflows working.  Text-to-video only.
  - DMMCinematicVideoPromptV2 (v2/v3.0): 3 outputs (prompt, label,
    conditioning_strength).  Supports text-to-video and image-to-video
    with dynamic conditioning based on data intensity.

Both share the same internal data processors via _PromptCore.

Author: Jeffrey A. Brick
"""

import logging
import random
import time
from hashlib import sha256

log = logging.getLogger("DMM.CinematicPrompt")


def _focus_seed_offset(focus):
    """Deterministic seed offset from focus string.
    Uses sha256 instead of hash() which is randomized across Python restarts."""
    return int.from_bytes(sha256(focus.encode()).digest()[:4], "little")


class _PromptCore:
    """Shared logic for both v1 and v2 prompt nodes."""

    _STYLES = {
        "natural": "Photorealistic cinematic video.",
        "noir_cinematic": "Film noir style, high contrast black and white, deep shadows.",
        "documentary_calm": "Documentary style, natural handheld camera, warm tones.",
        "cyberpunk_hud": "Cyberpunk aesthetic, holographic overlays, neon-lit streets.",
        "golden_hour_beauty": "Cinematic 4K, golden hour lighting, shallow depth of field.",
        "dramatic_broadcast": "Broadcast news style, crisp professional lighting, steady shot.",
        "webcam_strict": "Raw live webcam feed, CCTV security footage, static fixed camera, zero camera movement, exact visual match.",
    }

    @staticmethod
    def resolve_time():
        hour = time.localtime().tm_hour
        minute = time.localtime().tm_min
        ts = f"{hour:02d}:{minute:02d}"
        if hour < 6:
            return ts, "pre-dawn darkness, streetlights glowing", "early morning"
        elif hour < 9:
            return ts, "golden morning light, long shadows", "morning"
        elif hour < 12:
            return ts, "bright midday sun", "midday"
        elif hour < 16:
            return ts, "warm afternoon light", "afternoon"
        elif hour < 19:
            return ts, "golden hour, warm orange light", "evening"
        else:
            return ts, "nighttime, city lights and neon glow", "night"

    @staticmethod
    def style_prefix(style):
        return _PromptCore._STYLES.get(style, "Photorealistic cinematic video.")

    @staticmethod
    def resolve_style(visual_style, config):
        if visual_style:
            return visual_style
        if config and "visual_style" in config:
            return config["visual_style"]
        return "natural"

    # ------------------------------------------------------------------ #
    #  DATA PROCESSORS — return dict: display, mood, motion, label, cond
    # ------------------------------------------------------------------ #

    @staticmethod
    def weather(data, city, ts):
        if not data:
            return {"display": f"{city} WEATHER {ts}", "mood": "calm urban",
                    "motion": "gentle breeze, slight movement in foliage",
                    "label": "weather", "conditioning": 0.80}

        temp = data.get("temp_f", 72)
        feels = data.get("feels_like_f", temp)
        desc = data.get("description", "clear").lower()
        wind = data.get("wind_speed_mph", 5)
        gust = data.get("wind_gust_mph", 0)
        humidity = data.get("humidity", 50)
        clouds = data.get("clouds_pct", 0)

        parts = [f"{city.upper()} {ts}", f"{temp:.0f}F FEELS {feels:.0f}F",
                 desc.upper(), f"WIND {wind:.0f}MPH"]
        if gust and gust > wind:
            parts.append(f"GUSTS {gust:.0f}MPH")
        parts.append(f"HUMIDITY {humidity:.0f}%")
        display = " / ".join(parts)

        if "storm" in desc or "thunder" in desc or wind > 30:
            mood = "stormy, dramatic, intense wind and rain"
            motion = (f"violent wind gusting at {gust:.0f} mph bending trees sideways, "
                      f"rain lashing horizontally, debris tumbling across pavement")
            cond = 0.50
        elif "rain" in desc:
            mood = "rainy, wet, reflective surfaces everywhere"
            motion = (f"rain falling steadily, puddles rippling, wipers on cars, "
                      f"wet streets reflecting {int(temp)} degree warmth")
            cond = 0.60
        elif temp > 100:
            mood = "scorching, oppressive heat, empty streets"
            motion = (f"intense heat shimmer rising from {int(temp)} degree pavement, "
                      f"air distortion over every surface, near-zero pedestrian movement")
            cond = 0.55
        elif temp > 90:
            mood = "hot, hazy, sluggish"
            motion = (f"heat haze off {int(temp)} degree asphalt, slow movement, "
                      f"{humidity:.0f} percent humidity thickening the air")
            cond = 0.65
        elif "fog" in desc or humidity > 85:
            mood = "foggy, mysterious, low visibility"
            motion = (f"fog drifting through frame, headlights creating soft halos, "
                      f"{humidity:.0f} percent humidity condensing on surfaces")
            cond = 0.60
        elif clouds > 80:
            mood = "overcast, grey, moody"
            motion = (f"flat diffused light shifting, {clouds:.0f} percent cloud cover "
                      f"pressing down, muted colors")
            cond = 0.70
        elif "clear" in desc and temp > 65:
            mood = "sunny, vivid, lively"
            motion = (f"bright sunlight casting sharp shadows, light {wind:.0f} mph "
                      f"breeze rustling palms, vivid colors")
            cond = 0.85
        else:
            mood = "typical urban"
            motion = (f"gentle ambient movement, {wind:.0f} mph breeze, "
                      f"normal pedestrian and vehicle flow")
            cond = 0.80

        return {"display": display, "mood": mood, "motion": motion,
                "label": "weather", "conditioning": cond}

    @staticmethod
    def earthquake(data, city, ts):
        if not data or data.get("count", 0) == 0:
            mood_text = data.get("seismic_mood", "all quiet") if data else "all quiet"
            return {"display": f"{city.upper()} SEISMIC {ts} / {mood_text.upper()} / 0 EVENTS",
                    "mood": "calm, completely still",
                    "motion": "absolute stillness, no vibration, rock-steady scene",
                    "label": "seismic", "conditioning": 0.85}

        count = data.get("count", 0)
        max_mag = data.get("max_magnitude", 0)
        avg_mag = data.get("avg_magnitude", 0)
        mood_text = data.get("seismic_mood", "").upper()
        lookback = data.get("lookback_hours", 24)
        quakes = data.get("quakes", [])
        top = quakes[0] if quakes else {}
        place = top.get("place", "nearby").upper()
        depth = top.get("depth_km", 0)

        parts = [f"{city.upper()} SEISMIC {ts}", f"{count} EVENTS / {lookback}HRS",
                 f"MAX MAG {max_mag} AT {place}", f"DEPTH {depth:.0f}KM", mood_text]
        display = " / ".join(parts)

        if max_mag >= 4.0:
            mood = "tense, shaking, emergency feel"
            motion = (f"visible camera shake from magnitude {max_mag} tremor, "
                      f"hanging objects swinging, water surfaces rippling")
            cond = 0.50
        elif max_mag >= 2.5:
            mood = "unsettling, cautious"
            motion = (f"subtle vibration in standing water, slight sway in signs, "
                      f"magnitude {max_mag} aftereffects")
            cond = 0.65
        elif count > 5:
            mood = "watchful, subtly tense"
            motion = (f"micro-tremor swarm, barely perceptible vibration, "
                      f"{count} events creating underlying unease")
            cond = 0.70
        else:
            mood = "calm but geologically aware"
            motion = (f"still scene with {count} minor events logged, "
                      f"near-imperceptible ground movement")
            cond = 0.80

        return {"display": display, "mood": mood, "motion": motion,
                "label": "seismic", "conditioning": cond}

    @staticmethod
    def air_quality(data, city, ts):
        if not data:
            return {"display": f"{city.upper()} AIR {ts}", "mood": "hazy urban",
                    "motion": "slight atmospheric haze drifting",
                    "label": "air_quality", "conditioning": 0.75}

        aqi = data.get("us_aqi", 50)
        aqi_label = data.get("aqi_label", "Moderate").upper()
        pm25 = data.get("pm25", 10)
        pm10 = data.get("pm10", 20)
        ozone = data.get("ozone", 50)
        no2 = data.get("no2", 10)
        uv = data.get("uv_index", 3)

        parts = [f"{city.upper()} AIR {ts}", f"AQI {aqi} {aqi_label}",
                 f"PM2.5 {pm25:.0f} / PM10 {pm10:.0f}",
                 f"O3 {ozone:.0f} / NO2 {no2:.0f}", f"UV {uv}"]
        display = " / ".join(parts)

        if aqi > 150:
            mood = "hazardous, thick choking smog, brown-orange tint"
            motion = (f"heavy thick smog creeping across scene, AQI {aqi} reducing "
                      f"visibility to blocks, brown filter, masked pedestrians")
            cond = 0.55
        elif aqi > 100:
            mood = "unhealthy, oppressive haze"
            motion = (f"visible haze at AQI {aqi}, distant objects dissolving, "
                      f"muted desaturated colors, stagnant heavy air")
            cond = 0.65
        elif aqi > 50:
            mood = "slightly hazy but active"
            motion = (f"mild haze at AQI {aqi}, distant details slightly soft, "
                      f"normal activity, UV index {uv} sunlight")
            cond = 0.75
        else:
            mood = "crystal clear, vivid colors"
            motion = (f"pristine air at AQI {aqi}, razor sharp distant details, "
                      f"vivid saturated colors, UV {uv} bright sunlight")
            cond = 0.85

        return {"display": display, "mood": mood, "motion": motion,
                "label": "air_quality", "conditioning": cond}

    @staticmethod
    def transit(data, city, ts):
        if not data:
            return {"display": f"{city.upper()} TRANSIT {ts}", "mood": "busy streets",
                    "motion": "steady vehicle flow passing through frame",
                    "label": "transit", "conditioning": 0.75}

        avg_spd = data.get("avg_speed_mph", 15)
        stopped = data.get("stopped_pct", 0)
        cong = data.get("congestion_pct", 50)
        flow = data.get("flow_desc", "").upper()
        buses = data.get("vehicle_count", 0)
        route_count = data.get("route_count", 0)
        routes = data.get("routes_active", [])
        route_str = " ".join(routes[:3]).upper() if routes else ""

        parts = [f"{city.upper()} METRO {ts}", f"AVG {avg_spd:.0f}MPH / {flow}",
                 f"{buses} VEHICLES / {route_count} ROUTES",
                 f"CONGESTION {cong:.0f}% / {stopped:.0f}% STOPPED"]
        if route_str:
            parts.append(route_str)
        display = " / ".join(parts)

        if cong > 70:
            mood = "gridlocked, frustrated, dense"
            motion = (f"bumper to bumper at {avg_spd:.0f} mph, {stopped:.0f} percent "
                      f"stopped, red brake lights stretching to horizon")
            cond = 0.60
        elif cong > 40:
            mood = "congested, stop and go"
            motion = (f"stop and go at {avg_spd:.0f} mph, {cong:.0f} percent "
                      f"congestion creating wave patterns")
            cond = 0.70
        elif cong > 15:
            mood = "flowing, steady movement"
            motion = (f"smooth flow at {avg_spd:.0f} mph, green lights cascading, "
                      f"{buses} transit vehicles moving steadily")
            cond = 0.80
        else:
            mood = "quiet, sparse, open roads"
            motion = (f"near-empty roads, occasional vehicle at {avg_spd:.0f} mph, "
                      f"wide open lanes, peaceful stillness")
            cond = 0.85

        return {"display": display, "mood": mood, "motion": motion,
                "label": "transit", "conditioning": cond}

    @staticmethod
    def alerts(data, city, ts):
        if not data or data.get("count", 0) == 0:
            return {"display": f"{city.upper()} {ts} / ALL CLEAR / NO ADVISORIES",
                    "mood": "peaceful, calm, safe",
                    "motion": "gentle ambient movement, relaxed city rhythm",
                    "label": "alerts", "conditioning": 0.85}

        alerts_list = data.get("alerts", [])
        top = alerts_list[0] if alerts_list else {}
        event = top.get("event", "ALERT").upper()
        severity = top.get("severity", "Moderate").upper()
        headline = data.get("top_headline", event).upper()
        count = data.get("count", 0)
        level = data.get("alert_level", "ADVISORY").upper()

        parts = [f"{city.upper()} ALERT {ts}", f"{severity}: {event}",
                 headline, f"{count} ACTIVE / {level}"]
        if len(alerts_list) > 1:
            parts.append(f"ALSO: {alerts_list[1]['event'].upper()}")
        display = " / ".join(parts)

        if severity in ("EXTREME", "SEVERE"):
            mood = "urgent, emergency, red alert"
            motion = (f"emergency atmosphere, flashing red and blue reflections, "
                      f"{event} conditions visible, people rushing")
            cond = 0.50
        elif count > 2:
            mood = "tense, multi-warning, heightened awareness"
            motion = (f"{count} concurrent warnings creating visible tension, "
                      f"cautious movement, amber-tinted atmosphere")
            cond = 0.60
        else:
            mood = "cautious, advisory"
            motion = (f"slight unease, {event} advisory conditions, "
                      f"people checking phones, watchful but continuing")
            cond = 0.75

        return {"display": display, "mood": mood, "motion": motion,
                "label": "alerts", "conditioning": cond}


# ====================================================================== #
#  LEGACY v1 — 2 outputs (STRING, STRING) for v2.5 workflow compat
# ====================================================================== #

class DMMCinematicVideoPrompt:
    """Legacy v1: text-to-video prompts only, 2 outputs.  Keeps v2.5 workflows working."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_video_prompt"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("video_prompt", "segment_label",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focus": (["weather", "earthquake", "air_quality", "transit", "alerts"],),
            },
            "optional": {
                "visual_style": ([
                    "natural", "noir_cinematic", "documentary_calm",
                    "cyberpunk_hud", "golden_hour_beauty", "dramatic_broadcast",
                    "webcam_strict",
                ], {"tooltip": "Per-clip override."}),
                "weather_data": ("DMM_WEATHER",),
                "quake_data": ("DMM_QUAKE",),
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "alerts_data": ("DMM_ALERTS",),
                "config": ("DMM_CONFIG",),
            },
        }

    def generate_video_prompt(self, focus, visual_style=None,
                               weather_data=None, quake_data=None,
                               aq_data=None, transit_data=None,
                               alerts_data=None, config=None):
        city = config.get("city", "Los Angeles") if config else "Los Angeles"
        style = _PromptCore.resolve_style(visual_style, config)
        ts, time_light, time_period = _PromptCore.resolve_time()
        sp = _PromptCore.style_prefix(style)

        result = _get_data_result(focus, weather_data, quake_data,
                                  aq_data, transit_data, alerts_data,
                                  city, ts)

        cam_move = "static fixed camera" if style == "webcam_strict" else "the camera slowly moves through the scene"

        prompt = (
            f"{sp} "
            f"A {result['mood']} scene somewhere in {city} during the {time_period}, "
            f"{time_light}, "
            f"a large glowing digital display is visible showing: "
            f"{result['display']}, "
            f"{cam_move}, "
            f"everyday city life happening around the display, "
            f"ambient sounds of {city}"
        )

        log.info("v1 prompt [%s/%s]: %s", focus, style, prompt[:80])
        return (prompt.strip(), result["label"])


# ====================================================================== #
#  v2 — 3 outputs (STRING, STRING, FLOAT) for v3.0 webcam pipeline
# ====================================================================== #

class DMMCinematicVideoPromptV2:
    """v2: supports t2v and i2v modes with dynamic conditioning.  3 outputs."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_video_prompt"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT",)
    RETURN_NAMES = ("video_prompt", "segment_label", "conditioning_strength",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focus": (["weather", "earthquake", "air_quality", "transit", "alerts"],),
            },
            "optional": {
                "mode": (["auto", "text_to_video", "image_to_video"],
                         {"default": "auto",
                          "tooltip": "auto = i2v if webcam_success, else t2v"}),
                "webcam_success": ("BOOLEAN", {"default": False,
                    "tooltip": "From DMMWebcamFetch — routes auto mode"}),
                "visual_style": ([
                    "natural", "noir_cinematic", "documentary_calm",
                    "cyberpunk_hud", "golden_hour_beauty", "dramatic_broadcast",
                    "webcam_strict",
                ], {"tooltip": "Per-clip override."}),
                "weather_data": ("DMM_WEATHER",),
                "quake_data": ("DMM_QUAKE",),
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "alerts_data": ("DMM_ALERTS",),
                "config": ("DMM_CONFIG",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Time-dependent prompt output — always re-run."""
        return float("nan")

    def generate_video_prompt(self, focus,
                               mode="auto", webcam_success=False,
                               visual_style=None,
                               weather_data=None, quake_data=None,
                               aq_data=None, transit_data=None,
                               alerts_data=None, config=None):
        city = config.get("city", "Los Angeles") if config else "Los Angeles"
        style = _PromptCore.resolve_style(visual_style, config)
        ts, time_light, time_period = _PromptCore.resolve_time()
        sp = _PromptCore.style_prefix(style)

        # Resolve mode
        if mode == "auto":
            use_i2v = webcam_success
        elif mode == "image_to_video":
            use_i2v = True
        else:
            use_i2v = False

        result = _get_data_result(focus, weather_data, quake_data,
                                  aq_data, transit_data, alerts_data,
                                  city, ts)

        conditioning = result["conditioning"]

        if use_i2v:
            anim_action = "Static camera footage" if style == "webcam_strict" else "This scene slowly animates"
            prompt = (
                f"{sp} "
                f"{anim_action}, {time_light}, "
                f"{result['motion']}, "
                f"the atmosphere feels {result['mood']}, "
                f"ambient sounds of {city} during the {time_period}"
            )
        else:
            cam_move = "static fixed camera" if style == "webcam_strict" else "the camera slowly moves through the scene"
            prompt = (
                f"{sp} "
                f"A {result['mood']} scene somewhere in {city} during the {time_period}, "
                f"{time_light}, "
                f"a large glowing digital display is visible showing: "
                f"{result['display']}, "
                f"{cam_move}, "
                f"everyday city life happening around the display, "
                f"ambient sounds of {city}"
            )

        mode_str = "i2v" if use_i2v else "t2v"
        log.info("v2 prompt [%s/%s/%s] cond=%.2f: %s",
                 focus, style, mode_str, conditioning, prompt[:80])
        return (prompt.strip(), result["label"], conditioning)


# ====================================================================== #
#  Helper — routes focus to the correct data processor
# ====================================================================== #

def _get_data_result(focus, weather_data, quake_data, aq_data,
                     transit_data, alerts_data, city, ts):
    if focus == "weather":
        return _PromptCore.weather(weather_data, city, ts)
    elif focus == "earthquake":
        return _PromptCore.earthquake(quake_data, city, ts)
    elif focus == "air_quality":
        return _PromptCore.air_quality(aq_data, city, ts)
    elif focus == "transit":
        return _PromptCore.transit(transit_data, city, ts)
    elif focus == "alerts":
        return _PromptCore.alerts(alerts_data, city, ts)
    else:
        return {"display": f"{city} {ts}", "mood": "calm urban",
                "motion": "gentle ambient movement",
                "label": "unknown", "conditioning": 0.75}