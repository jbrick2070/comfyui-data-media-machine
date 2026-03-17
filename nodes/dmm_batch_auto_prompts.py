"""
DMMBatchAutoPrompts — Auto-generates all 5 video prompts from live data feeds.

On every run it fetches the current data state and produces five distinct
cinematic prompts (one per data focus: weather, seismic, air_quality,
transit, alerts) ready to wire directly into DMMBatchVideoGenerator.

Because the seed is salted with the current Unix timestamp by default,
each queue run produces freshly randomized camera moves / wording
even when the underlying live data hasn't changed.

Author: Jeffrey A. Brick
"""

import time
import random


# Focus order — maps directly to prompt_1 … prompt_5
_FOCUSES = ["weather", "earthquake", "air_quality", "transit", "alerts"]


class DMMBatchAutoPrompts:
    """
    Consumes all live data feeds and emits 5 distinct cinematic prompts.

    Designed as a zero-config drop-in for the 5 prompt_* string inputs
    on DMMBatchVideoGenerator.  Wire the data fetch nodes in, hit queue,
    and get fresh prompts every single run.
    """

    CATEGORY = "DataMediaMachine"
    FUNCTION = "build_prompts"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "live_seed")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "visual_style": ([
                    "natural",
                    "noir_cinematic",
                    "documentary_calm",
                    "cyberpunk_hud",
                    "golden_hour_beauty",
                    "dramatic_broadcast",
                    "webcam_strict",
                ], {"default": "natural",
                    "tooltip": "Visual style prefix injected into every prompt"}),
                "time_salt": (["enabled", "disabled"], {
                    "default": "enabled",
                    "tooltip": (
                        "enabled = XOR the config seed with current Unix time so "
                        "camera choices randomise on every queue run. "
                        "disabled = deterministic (same data → same prompts)."
                    ),
                }),
            },
            "optional": {
                "weather_data":  ("DMM_WEATHER",),
                "quake_data":    ("DMM_QUAKE",),
                "aq_data":       ("DMM_AIRQUALITY",),
                "transit_data":  ("DMM_TRANSIT",),
                "alerts_data":   ("DMM_ALERTS",),
            },
        }

    # ---- Camera moves shared with DMMCinematicVideoPrompt ----
    _CAMERAS = [
        "slow pan left",
        "slow tracking shot forward",
        "static wide shot",
        "gentle dolly forward",
        "slow pan right",
    ]

    _STYLE_PREFIXES = {
        "natural":            "",
        "noir_cinematic":     "Film noir style, high contrast black and white with deep shadows, 35mm anamorphic lens.",
        "documentary_calm":   "Documentary style, natural handheld camera, warm natural lighting, 50mm lens.",
        "cyberpunk_hud":      "Cyberpunk aesthetic, holographic HUD overlays, neon-lit rain-slicked streets, wide angle lens.",
        "golden_hour_beauty": "Cinematic 4K, golden hour lighting, warm color grading, shallow depth of field, 85mm lens.",
        "dramatic_broadcast": "Broadcast news style, crisp professional lighting, steady tripod shot, clean composition.",
        "webcam_strict":      "Raw live webcam feed, CCTV security footage, static fixed camera, zero camera movement, exact visual match.",
    }

    # ------------------------------------------------------------------
    def build_prompts(
        self,
        config,
        visual_style="natural",
        time_salt="enabled",
        weather_data=None,
        quake_data=None,
        aq_data=None,
        transit_data=None,
        alerts_data=None,
    ):
        # Compute working seed: optionally XOR with current time
        base_seed = config.get("seed", 42)
        if time_salt == "enabled":
            live_seed = base_seed ^ (int(time.time()) & 0xFFFFFFFF)
        else:
            live_seed = base_seed

        city    = config.get("city", "Los Angeles")
        style   = self._STYLE_PREFIXES.get(visual_style, "")

        # Time-of-day lighting tag (same logic as DMMCinematicVideoPrompt)
        hour = time.localtime().tm_hour
        if hour < 6:
            time_light = "pre-dawn darkness, streetlights casting amber pools"
        elif hour < 9:
            time_light = "golden morning light, long shadows stretching across pavement"
        elif hour < 12:
            time_light = "bright midday sun, harsh overhead light"
        elif hour < 16:
            time_light = "warm afternoon light filtering through haze"
        elif hour < 19:
            time_light = "golden hour, warm orange light painting every surface"
        else:
            time_light = "nighttime city glow, neon and streetlight reflections"

        # Build per-focus RNGs using live_seed so each focus is independent
        rngs = {
            focus: random.Random(live_seed + hash(focus))
            for focus in _FOCUSES
        }

        data_map = {
            "weather":     weather_data,
            "earthquake":  quake_data,
            "air_quality": aq_data,
            "transit":     transit_data,
            "alerts":      alerts_data,
        }

        prompts = []
        print(f"[DMM_AutoPrompts] Generating 5 prompts (live_seed={live_seed}, "
              f"style={visual_style}, time={hour:02d}:xx)")

        for focus in _FOCUSES:
            rng  = rngs[focus]
            data = data_map[focus]

            if focus == "weather":
                prompt, label = self._weather_prompt(data, style, time_light, city, rng, visual_style)
            elif focus == "earthquake":
                prompt, label = self._earthquake_prompt(data, style, time_light, city, rng, visual_style)
            elif focus == "air_quality":
                prompt, label = self._aq_prompt(data, style, time_light, city, rng, visual_style)
            elif focus == "transit":
                prompt, label = self._transit_prompt(data, style, time_light, city, rng, visual_style)
            else:  # alerts
                prompt, label = self._alerts_prompt(data, style, time_light, city, rng, visual_style)

            prompts.append(prompt)
            print(f"[DMM_AutoPrompts]   [{label}] {prompt[:90]}...")

        return (*prompts, live_seed)

    # ==================================================================
    # Prompt generators (identical logic to DMMCinematicVideoPrompt)
    # ==================================================================

    def _weather_prompt(self, data, style, time_light, city, rng, visual_style):
        if visual_style == "webcam_strict":
            camera = "static fixed angle"
        else:
            camera = rng.choice(self._CAMERAS)
            
        if not data:
            return (
                f"{style} {camera}, {city} skyline, {time_light}, "
                "gentle breeze, palm trees swaying, ambient city hum",
                "weather"
            )
        temp  = data.get("temp_f", 72)
        desc  = data.get("description", "clear").lower()
        wind  = data.get("wind_speed_mph", 5)
        hum   = data.get("humidity", 50)

        if "rain" in desc:
            scene = (f"{camera}, {city} streets, steady rainfall, "
                     "rain streaking through streetlights, wet asphalt reflections, "
                     "puddles forming, rain on pavement audio")
        elif "cloud" in desc:
            scene = (f"{camera}, {city} skyline, heavy overcast cloud cover, "
                     "grey diffused light, dark building silhouettes, distant traffic rumble")
        elif "fog" in desc or hum > 85:
            scene = (f"{camera}, {city} streets, dense fog, buildings fading into white haze, "
                     "soft headlight halos, muffled foghorn audio")
        elif wind > 25:
            scene = (f"{camera}, {city} streets, strong wind gusts, "
                     "palm trees bending dramatically, debris on sidewalks, "
                     "wind roar between buildings")
        elif temp > 95:
            scene = (f"{camera}, {city} pavement, extreme heat shimmer, distorted skyline, "
                     f"empty streets, {temp:.0f} degrees, air conditioning hum")
        elif "clear" in desc:
            scene = (f"{camera}, {city} skyline, crystal clear sky, perfect visibility, "
                     f"sharp mountain horizon, motionless palm trees, {temp:.0f} degrees")
        else:
            scene = (f"{camera}, {city} skyline, {desc} sky, "
                     f"{temp:.0f} degrees, {time_light}, ambient city hum")

        return (f"{style} {scene}, {time_light}", "weather")

    # ------------------------------------------------------------------
    def _earthquake_prompt(self, data, style, time_light, city, rng, visual_style):
        if visual_style == "webcam_strict":
            camera = "static fixed angle"
        else:
            camera = rng.choice(self._CAMERAS)
            
        if not data or data.get("count", 0) == 0:
            return (
                f"{style} {camera}, {city} hillside, calm geological layers, "
                "no seismic activity, birds on power lines, quiet ambient audio",
                "seismic"
            )
        count   = data.get("count", 0)
        max_mag = data.get("max_magnitude", 0)
        quakes  = data.get("quakes", [])
        top     = quakes[0] if quakes else {}
        place   = top.get("place", "nearby")

        if max_mag >= 4.0:
            scene = (f"{camera}, seismograph station, needle swinging wildly, "
                     f"magnitude {max_mag}, glasses rattling on shelves, "
                     "deep underground rumble, rising dust")
        elif max_mag >= 2.5:
            scene = (f"{camera}, seismograph close-up, steady needle trace, "
                     f"magnitude {max_mag} from {place}, subtle vibration, "
                     "coffee cup ripple, low frequency hum")
        elif count > 5:
            scene = (f"{camera}, multiple seismograph needles, thin activity lines, "
                     f"{count} micro-tremors, restless instruments, clicking recording equipment")
        else:
            scene = (f"{camera}, seismograph monitoring station, {city}, "
                     f"{count} small events, barely moving needle, calm electronic beeps")

        return (f"{style} {scene}, {time_light}", "seismic")

    # ------------------------------------------------------------------
    def _aq_prompt(self, data, style, time_light, city, rng, visual_style):
        if visual_style == "webcam_strict":
            camera = "static fixed angle"
        else:
            camera = rng.choice(self._CAMERAS)
            
        if not data:
            return (
                f"{style} {camera}, {city} skyline, clear atmosphere, "
                "mountains visible, birds gliding, clean sky",
                "air_quality"
            )
        aqi = data.get("us_aqi", 50)
        uv  = data.get("uv_index", 3)

        if aqi > 150:
            scene = (f"{camera}, {city} skyline, thick toxic smog layer, "
                     f"brown-grey atmospheric haze, low visibility, AQI {aqi}, "
                     "masked pedestrians, oppressive silence")
        elif aqi > 100:
            scene = (f"{camera}, {city} skyline, heavy haze layer, mountains obscured by particulates, "
                     f"AQI {aqi}, pale sun disc through smog, muffled city audio")
        elif aqi > 50:
            scene = (f"{camera}, {city} skyline, moderate haze, slightly blurred distant buildings, "
                     f"AQI {aqi}, normal ambient city audio")
        else:
            scene = (f"{camera}, {city} skyline, pristine clear air, "
                     f"razor sharp building edges, vivid mountain horizon, AQI {aqi}, clean wind audio")

        if uv > 9:
            scene += f", harsh bleaching sunlight, UV index {uv}"
        elif uv > 6:
            scene += f", strong sunlight, sharp shadows, UV index {uv}"

        return (f"{style} {scene}, {time_light}", "air_quality")

    # ------------------------------------------------------------------
    def _transit_prompt(self, data, style, time_light, city, rng, visual_style):
        if visual_style == "webcam_strict":
            camera = "static fixed angle"
        else:
            camera = rng.choice(self._CAMERAS)
            
        if not data:
            return (
                f"{style} {camera}, {city} boulevard, Metro buses passing, "
                "flowing intersection traffic, engine sounds, air brakes audio",
                "transit"
            )
        avg_spd = data.get("avg_speed_mph", 15)
        cong    = data.get("congestion_pct", 50)
        buses   = data.get("vehicle_count", 0)
        routes  = data.get("routes_active", [])
        route_str = ", ".join(routes[:3]) if routes else "various lines"

        if cong > 70:
            scene = (f"{camera}, {city} city streets, severe traffic gridlock, "
                     f"{buses} Metro buses bumper to bumper, {avg_spd:.0f} mph crawl, "
                     "endless brake lights, honking horns audio")
        elif cong > 40:
            scene = (f"{camera}, {city} intersections, moderate stop-and-go traffic, "
                     f"Metro routes {route_str}, {avg_spd:.0f} mph, engine rumble and tire noise audio")
        elif cong > 15:
            scene = (f"{camera}, {city} wide boulevard, smooth flowing traffic, "
                     f"{buses} Metro buses at {avg_spd:.0f} mph, "
                     "cascading green lights, whooshing vehicles audio")
        else:
            scene = (f"{camera}, {city} empty streets, lone Metro bus cruising at {avg_spd:.0f} mph, "
                     "open road, quiet engine hum, distant helicopter audio")

        return (f"{style} {scene}, {time_light}", "transit")

    # ------------------------------------------------------------------
    def _alerts_prompt(self, data, style, time_light, city, rng, visual_style):
        if visual_style == "webcam_strict":
            camera = "static fixed angle"
        else:
            camera = rng.choice(self._CAMERAS)
            
        if not data or data.get("count", 0) == 0:
            return (
                f"{style} {camera}, {city} calm street, no active alerts, "
                "green traffic lights, peaceful pedestrians, gentle ambient audio",
                "alerts"
            )
        alerts   = data.get("alerts", [])
        top      = alerts[0] if alerts else {}
        event    = top.get("event", "Weather Alert")
        severity = top.get("severity", "Moderate")
        count    = data.get("count", 0)

        if severity in ("Extreme", "Severe"):
            scene = (f"{camera}, {city} emergency broadcast screen, {event} alert, "
                     "scrolling red warning text, emergency vehicles with flashing lights, "
                     "wailing sirens audio")
        elif count > 2:
            scene = (f"{camera}, weather station monitor, {count} active warnings for {city}, "
                     f"{event} text prominent, crackling emergency radio audio")
        else:
            scene = (f"{camera}, advisory monitor display, {event} for {city}, "
                     "yellow caution text, soft weather broadcast audio, periodic alert tone")

        return (f"{style} {scene}, {time_light}", "alerts")