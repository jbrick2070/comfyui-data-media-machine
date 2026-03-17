"""
DMMDataToTTS — Generates narration scripts from live data.
Output feeds directly into Kokoro TTS nodes.

No heavy imports — pure string generation.
"""

import random
import json


class DMMDataToTTS:
    """Converts data into radio-drama narration scripts for TTS."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_script"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("narration_text", "voice_id", "script_metadata",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "weather_data": ("DMM_WEATHER",),
                "narrator_style": ([
                    "news_anchor",
                    "noir_detective",
                    "surreal_poet",
                    "radio_dj",
                    "calm_documentary",
                    "old_time_radio",
                    "cyberpunk_dispatch",
                    "haiku_minimalist",
                ],),
            },
            "optional": {
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "kokoro_voice": ("STRING", {
                    "default": "bm_lewis",
                    "tooltip": "Kokoro voice: bm_lewis, af_sarah, bf_emma, am_adam"
                }),
            },
        }

    # Voice suggestions per style
    _VOICE_MAP = {
        "news_anchor": "bm_lewis",
        "noir_detective": "bm_lewis",
        "surreal_poet": "af_sarah",
        "radio_dj": "bm_lewis",
        "calm_documentary": "bf_emma",
        "old_time_radio": "bm_lewis",
        "cyberpunk_dispatch": "am_adam",
        "haiku_minimalist": "af_sarah",
    }

    def generate_script(self, config, weather_data, narrator_style,
                         aq_data=None, transit_data=None,
                         kokoro_voice="bm_lewis"):
        rng = random.Random(config["seed"])
        city = config["city"]

        temp = weather_data.get("temp_f", 72)
        desc = weather_data.get("description", "clear")
        condition = weather_data.get("condition", "Clear")
        wind = weather_data.get("wind_speed_mph", 5)
        humidity = weather_data.get("humidity", 50)
        rain_mm = weather_data.get("rain_1h_mm", 0)
        vis = weather_data.get("visibility_m", 10000)
        is_live = weather_data.get("live", False)

        # Optional data snippets
        aq_line = ""
        if aq_data:
            aqi = aq_data.get("us_aqi", 50)
            label = aq_data.get("aqi_label", "Unknown")
            aq_line = self._aq_narration(narrator_style, aqi, label, rng)

        transit_line = ""
        if transit_data:
            avg_spd = transit_data.get("avg_speed_mph", 15)
            cong = transit_data.get("congestion_pct", 50)
            flow = transit_data.get("flow_desc", "")
            transit_line = self._transit_narration(
                narrator_style, avg_spd, cong, flow, rng)

        # Main narration
        narration = self._build_narration(
            narrator_style, city, temp, desc, condition, wind,
            humidity, rain_mm, vis, aq_line, transit_line, is_live, rng
        )

        # Voice selection
        voice = kokoro_voice or self._VOICE_MAP.get(narrator_style, "bm_lewis")

        metadata = json.dumps({
            "style": narrator_style,
            "voice": voice,
            "city": city,
            "condition": condition,
            "live": is_live,
            "word_count": len(narration.split()),
        })

        return (narration, voice, metadata)

    def _aq_narration(self, style, aqi, label, rng):
        if style == "noir_detective":
            if aqi > 150:
                return "The air itself was poisoned. You could taste it."
            elif aqi > 100:
                return "The smog hung heavy, like a curtain nobody could pull back."
            else:
                return ""
        elif style == "cyberpunk_dispatch":
            return f"Air quality index: {aqi}. Classification: {label}."
        elif style == "surreal_poet":
            if aqi > 100:
                return f"The air remembers {aqi} particles of forgotten industry."
            else:
                return ""
        elif style in ("news_anchor", "old_time_radio", "calm_documentary"):
            if aqi > 100:
                return f"Air quality is {label.lower()} with a US AQI of {aqi}."
            else:
                return ""
        elif style == "radio_dj":
            if aqi > 100:
                return f"Heads up, air quality's at {aqi} today — maybe keep the windows closed!"
            else:
                return ""
        return ""

    def _transit_narration(self, style, avg_spd, congestion, flow, rng):
        if style == "noir_detective":
            if congestion > 70:
                return f"The buses crawl at {avg_spd:.0f} miles an hour. Nobody's going anywhere."
            else:
                return f"Buses running clean at {avg_spd:.0f}. Good night for a ride."
        elif style == "cyberpunk_dispatch":
            return (f"Transit grid: {avg_spd:.0f} MPH average. "
                    f"Congestion index: {congestion}%.")
        elif style == "radio_dj":
            if congestion > 70:
                return f"Metro buses are barely moving out there, {avg_spd:.0f} per hour!"
            else:
                return f"Metro's cruising at {avg_spd:.0f}, smooth sailing!"
        elif style == "old_time_radio":
            return f"The streetcars are averaging {avg_spd:.0f} miles per hour this evening."
        elif style == "surreal_poet":
            return f"The city moves at {avg_spd:.0f} thoughts per hour."
        elif style in ("news_anchor", "calm_documentary"):
            return f"Metro buses averaging {avg_spd:.0f} mph, indicating {flow}."
        return ""

    def _build_narration(self, style, city, temp, desc, condition, wind,
                          humidity, rain_mm, vis, aq_line, transit_line,
                          is_live, rng):
        live_note = " Live conditions." if is_live else ""

        if style == "news_anchor":
            parts = [
                f"Good evening. This is your live data feed for {city}.{live_note}",
                f"Currently {temp:.0f} degrees with {desc.lower()}.",
                f"Winds {'calm' if wind < 5 else f'at {wind:.0f} miles per hour'}.",
                f"Humidity at {humidity:.0f} percent.",
            ]
            if rain_mm > 0:
                parts.append(f"Precipitation recorded at {rain_mm:.1f} millimeters.")
            if vis < 3000:
                parts.append(f"Visibility reduced to {vis} meters.")
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("Stay safe out there.")

        elif style == "noir_detective":
            parts = [
                f"The city doesn't sleep. {city}, {temp:.0f} degrees.",
                f"{'The sky was weeping. ' if rain_mm > 0 else ''}{desc}.",
                f"{'The wind howled through concrete canyons,' if wind > 20 else 'The air was still,'} "
                f"humidity thick at {humidity:.0f} percent.",
            ]
            if vis < 3000:
                parts.append("Couldn't see past two blocks. The kind of night things disappear.")
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("Just another night in the city of angels.")

        elif style == "surreal_poet":
            parts = [
                f"The temperature remembers itself at {temp:.0f}.",
                f"In {city}, the sky speaks in {desc.lower()}.",
                f"The wind carries {wind:.0f} whispers per hour,",
                f"each one {humidity:.0f} percent certain of something it cannot name.",
            ]
            if rain_mm > 0:
                parts.append(f"Rain falls like {rain_mm:.1f} millimeters of forgotten conversation.")
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("The data becomes a dream.")

        elif style == "radio_dj":
            parts = [
                f"What is UP {city}!",
                f"We're sitting at {temp:.0f} degrees, {desc.lower()} outside.",
            ]
            if wind > 15:
                parts.append(f"Wind's kicking at {wind:.0f} miles per hour!")
            if rain_mm > 0:
                parts.append("And yes it IS raining, grab that umbrella!")
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("Now back to the music!")

        elif style == "calm_documentary":
            parts = [
                f"In {city}, the temperature rests at {temp:.0f} degrees Fahrenheit.",
                f"The sky presents {desc.lower()}.",
                f"Wind moves at {wind:.0f} miles per hour.",
            ]
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("The city continues its quiet rhythm.")

        elif style == "old_time_radio":
            parts = [
                f"Good evening, ladies and gentlemen.",
                f"Broadcasting live from the heart of {city}.",
                f"The thermometer reads {temp:.0f} degrees,",
                f"with reports of {desc.lower()} across the metropolitan area.",
                f"Winds from the {'north' if rng.random() > 0.5 else 'west'} at {wind:.0f} miles per hour.",
            ]
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("And now, back to our regularly scheduled program.")

        elif style == "cyberpunk_dispatch":
            parts = [
                f"SYSTEM: {city} environmental scan.",
                f"Temperature: {temp:.0f}F. Condition: {condition}.",
                f"Wind vector: {wind:.0f} MPH. Humidity: {humidity:.0f}.",
                f"Visibility: {vis} meters.",
            ]
            if rain_mm > 0:
                parts.append(f"Precipitation: {rain_mm:.1f} mm per hour.")
            if aq_line:
                parts.append(aq_line)
            if transit_line:
                parts.append(transit_line)
            parts.append("End dispatch.")

        elif style == "haiku_minimalist":
            parts = [
                f"{city}. {temp:.0f} degrees.",
                f"{condition.lower()}.",
                f"Wind, {wind:.0f}.",
                "Silence.",
            ]

        else:
            parts = [f"{city}: {temp:.0f}F, {desc}, wind {wind:.0f} mph."]

        return " ".join(parts).strip()
