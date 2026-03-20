"""
DMMLAPulseNarrative — The big narrative engine.

Takes ALL data streams (weather, air quality, transit, earthquakes, alerts)
and weaves them into a cohesive, informative broadcast script covering
the full gamut of what's happening in LA.

Outputs a radio-broadcast-quality narration ready for Kokoro TTS.

Think: NPR morning report meets old-time radio meets data art.
"""

import random
import json
import time


class DMMLAPulseNarrative:
    """Assembles all data into a comprehensive LA broadcast narrative."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "build_narrative"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("full_narrative", "voice_id", "segment_count", "narrative_metadata",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "weather_data": ("DMM_WEATHER",),
                "broadcast_style": ([
                    "la_morning_report",
                    "noir_city_pulse",
                    "old_time_radio_hour",
                    "cyberpunk_city_scan",
                    "calm_documentary",
                    "surreal_dispatch",
                ],),
            },
            "optional": {
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "quake_data": ("DMM_QUAKE",),
                "alerts_data": ("DMM_ALERTS",),
                "kokoro_voice": ("STRING", {
                    "default": "bm_lewis",
                }),
            },
        }

    # Voice defaults per style
    _VOICE_MAP = {
        "la_morning_report": "bm_lewis",
        "noir_city_pulse": "bm_lewis",
        "old_time_radio_hour": "bm_lewis",
        "cyberpunk_city_scan": "am_adam",
        "calm_documentary": "bf_emma",
        "surreal_dispatch": "af_sarah",
    }

    def build_narrative(self, config, weather_data, broadcast_style,
                         aq_data=None, transit_data=None,
                         quake_data=None, alerts_data=None,
                         kokoro_voice="bm_lewis"):

        rng = random.Random(config["seed"])
        city = config["city"]

        # Determine which segments we have data for
        segments = []

        # 1. OPENING
        segments.append(self._opening(broadcast_style, city, weather_data, rng))

        # 2. ALERTS (highest priority — goes first if present)
        if alerts_data and alerts_data.get("count", 0) > 0:
            segments.append(self._alerts_segment(broadcast_style, alerts_data, rng))

        # 3. WEATHER
        segments.append(self._weather_segment(broadcast_style, weather_data, city, rng))

        # 4. AIR QUALITY
        if aq_data:
            seg = self._aq_segment(broadcast_style, aq_data, rng)
            if seg:
                segments.append(seg)

        # 5. TRANSIT / TRAFFIC
        if transit_data:
            segments.append(self._transit_segment(broadcast_style, transit_data, city, rng))

        # 6. SEISMIC
        if quake_data:
            seg = self._quake_segment(broadcast_style, quake_data, city, rng)
            if seg:
                segments.append(seg)

        # 7. CLOSING
        segments.append(self._closing(broadcast_style, city, weather_data, rng))

        # Join with natural pauses (period-space for TTS pacing)
        full_narrative = " ".join(s.strip() for s in segments if s.strip())

        voice = kokoro_voice or self._VOICE_MAP.get(broadcast_style, "bm_lewis")

        # Count data sources used
        sources_live = sum(1 for d in [weather_data, aq_data, transit_data, quake_data, alerts_data]
                           if d and d.get("live", False))
        sources_total = sum(1 for d in [weather_data, aq_data, transit_data, quake_data, alerts_data]
                            if d is not None)

        metadata = json.dumps({
            "style": broadcast_style,
            "voice": voice,
            "city": city,
            "segments": len(segments),
            "word_count": len(full_narrative.split()),
            "sources_live": sources_live,
            "sources_total": sources_total,
            "has_alerts": bool(alerts_data and alerts_data.get("count", 0) > 0),
            "has_quakes": bool(quake_data and quake_data.get("count", 0) > 0),
        })

        # --- AUTO-LOG TEXT SCRIPT EVERY MINUTE ---
        try:
            log_time = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("la_pulse_script_log.txt", "a", encoding="utf-8") as f:
                f.write(f"[{log_time}] ({broadcast_style})\n{full_narrative}\n\n")
        except Exception as e:
            print(f"[DMM LA Pulse] Could not save text log: {e}")

        return (full_narrative, voice, str(len(segments)), metadata)

    # ---- OPENING ----

    def _opening(self, style, city, weather, rng):
        temp = weather.get("temp_f", 72)
        live = weather.get("live", False)
        live_str = "live" if live else "simulated"

        # Fetch current time and determine the time of day
        current_time = time.strftime("%I:%M %p").lstrip("0")
        hour = time.localtime().tm_hour
        
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        if style == "la_morning_report":
            greetings = [
                f"{greeting}, {city}. It is {current_time}. This is your city pulse report.",
                f"{greeting}, {city}. The time is {current_time}. Here's what's happening across the basin.",
                f"{greeting}. It is {current_time}. You're listening to the {city} Pulse, your {live_str} city data feed.",
            ]
            return rng.choice(greetings)

        elif style == "noir_city_pulse":
            return (
                f"The time is {current_time}. The city breathes at {temp:.0f} degrees. "
                f"{city}. Ten million stories, and this is what the data says right now."
            )

        elif style == "old_time_radio_hour":
            return (
                f"{greeting}, ladies and gentlemen. The time is exactly {current_time}. "
                f"You're tuned to the {city} Information Hour, "
                f"bringing you the latest reports from across the metropolitan area."
            )

        elif style == "cyberpunk_city_scan":
            return (
                f"SYSTEM ONLINE. Temporal sync: {current_time}. Initiating {city} metropolitan scan. "
                f"All sensor arrays active. Data feed: {live_str}."
            )

        elif style == "calm_documentary":
            return (
                f"The time is {current_time}. "
                f"{city} stretches across the basin under a sky that reads {temp:.0f} degrees. "
                f"Here is what the city is telling us right now."
            )

        elif style == "surreal_dispatch":
            return (
                f"The clocks agree that it is {current_time}. "
                f"Somewhere between the mountains and the ocean, "
                f"{city} remembers that it is {temp:.0f} degrees. "
                f"The data begins to speak."
            )

        return f"This is the {city} Pulse at {current_time}."

    # ---- ALERTS ----

    def _alerts_segment(self, style, alerts_data, rng):
        alerts = alerts_data.get("alerts", [])
        if not alerts:
            return ""

        top = alerts[0]
        event = top["event"]
        headline = top.get("headline", "")
        severity = top.get("severity", "Unknown")
        instruction = top.get("instruction", "")
        count = alerts_data.get("count", 0)

        if style == "la_morning_report":
            parts = [f"First, an important alert."]
            parts.append(f"The National Weather Service has issued a {event} for our area.")
            if headline:
                parts.append(headline + ".")
            if instruction:
                parts.append(instruction)
            if count > 1:
                others = [a["event"] for a in alerts[1:3]]
                parts.append(f"Additionally, {', '.join(others)} are also in effect.")
            return " ".join(parts)

        elif style == "noir_city_pulse":
            parts = [f"But first, a warning."]
            parts.append(f"{event}. {severity} level.")
            if headline:
                parts.append(f"The feds say: {headline}.")
            if count > 1:
                parts.append(f"And that's not all. {count} alerts active tonight.")
            return " ".join(parts)

        elif style == "old_time_radio_hour":
            parts = [f"We interrupt with an important bulletin."]
            parts.append(f"The Weather Bureau has issued a {event}.")
            if headline:
                parts.append(headline + ".")
            if instruction:
                parts.append(f"Citizens are advised: {instruction}")
            return " ".join(parts)

        elif style == "cyberpunk_city_scan":
            parts = [f"WARNING. Active alert detected: {event}. Severity: {severity}."]
            if count > 1:
                parts.append(f"Total active alerts: {count}.")
            if instruction:
                parts.append(f"Directive: {instruction}")
            return " ".join(parts)

        elif style == "calm_documentary":
            parts = [f"There is currently a {event} in effect for the area."]
            if headline:
                parts.append(headline + ".")
            return " ".join(parts)

        elif style == "surreal_dispatch":
            parts = [f"The sky has issued a statement. It calls itself: {event}."]
            if severity == "Extreme":
                parts.append("The statement is shouting.")
            return " ".join(parts)

        return f"Alert: {event} in effect."

    # ---- WEATHER ----

    def _weather_segment(self, style, weather, city, rng):
        temp = weather.get("temp_f", 72)
        desc = weather.get("description", "clear").lower()
        wind = weather.get("wind_speed_mph", 5)
        humidity = weather.get("humidity", 50)
        vis = weather.get("visibility_m", 10000)
        gust = weather.get("wind_gust_mph", 0)

        if style == "la_morning_report":
            parts = [f"Current conditions: {temp:.0f} degrees, {desc}."]
            if wind > 10:
                parts.append(f"Winds at {wind:.0f} miles per hour.")
            if gust > 25:
                parts.append(f"Gusts up to {gust:.0f}.")
            parts.append(f"Humidity at {humidity:.0f} percent.")
            if vis < 5000:
                parts.append(f"Visibility is reduced to about {vis} meters. Use caution on the roads.")
            return " ".join(parts)

        elif style == "noir_city_pulse":
            parts = [
                f"{temp:.0f} degrees out there. {desc.capitalize()}.",
            ]
            if wind > 15:
                parts.append(f"Wind's howling at {wind:.0f}.")
            if humidity > 80:
                parts.append("The air is thick enough to chew.")
            elif humidity < 20:
                parts.append("The air is bone dry. Santa Ana weather.")
            if vis < 3000:
                parts.append("Can't see past two blocks.")
            return " ".join(parts)

        elif style == "old_time_radio_hour":
            parts = [
                f"The thermometer reads {temp:.0f} degrees Fahrenheit this evening.",
                f"Skies are reporting {desc}.",
                f"Winds from the {'northeast' if rng.random() > 0.5 else 'northwest'} at {wind:.0f} miles per hour.",
            ]
            return " ".join(parts)

        elif style == "cyberpunk_city_scan":
            return (
                f"Environmental: {temp:.0f}F. Condition: {desc}. "
                f"Wind: {wind:.0f} MPH. Gusts: {gust:.0f}. "
                f"Humidity: {humidity:.0f}%. Visibility: {vis}m."
            )

        elif style == "calm_documentary":
            return (
                f"The temperature sits at {temp:.0f} degrees. "
                f"The sky offers {desc}. "
                f"A wind of {wind:.0f} miles per hour moves through the basin."
            )

        elif style == "surreal_dispatch":
            return (
                f"The temperature has decided to be {temp:.0f}. "
                f"The sky performs {desc}. "
                f"Wind carries {wind:.0f} conversations per hour, "
                f"each {humidity:.0f} percent certain of something unnamed."
            )

        return f"{city}: {temp:.0f}F, {desc}, wind {wind:.0f} mph."

    # ---- AIR QUALITY ----

    def _aq_segment(self, style, aq_data, rng):
        aqi = aq_data.get("us_aqi", 50)
        label = aq_data.get("aqi_label", "Good")
        uv = aq_data.get("uv_index", 3)

        # Skip if unremarkable
        if aqi < 60 and uv < 8:
            if style in ("cyberpunk_city_scan",):
                return f"Air quality nominal. AQI: {aqi}. UV index: {uv}."
            return ""

        if style == "la_morning_report":
            parts = [f"Air quality today: {label}, with a US AQI of {aqi}."]
            if aqi > 100:
                parts.append("Sensitive groups should limit outdoor activity.")
            if uv > 8:
                parts.append(f"UV index is high at {uv}. Wear sunscreen if you're heading out.")
            return " ".join(parts)

        elif style == "noir_city_pulse":
            if aqi > 150:
                return f"The air is poison tonight. AQI at {aqi}. You can taste it."
            elif aqi > 100:
                return f"Smog's thick. AQI reads {aqi}. Not a night for deep breaths."
            return ""

        elif style == "old_time_radio_hour":
            if aqi > 100:
                return f"The air quality bureau reports a reading of {aqi}, which is {label.lower()}."
            return ""

        elif style == "cyberpunk_city_scan":
            return (
                f"Atmospheric contamination: AQI {aqi}, classification {label}. "
                f"UV radiation index: {uv}."
            )

        elif style == "calm_documentary":
            if aqi > 80:
                return f"The air quality index registers {aqi}, categorized as {label.lower()}."
            return ""

        elif style == "surreal_dispatch":
            if aqi > 80:
                return f"The air remembers {aqi} particles of yesterday's ambition."
            return ""

        return ""

    # ---- TRANSIT ----

    def _transit_segment(self, style, transit, city, rng):
        avg_spd = transit.get("avg_speed_mph", 15)
        cong = transit.get("congestion_pct", 50)
        flow = transit.get("flow_desc", "")
        buses = transit.get("vehicle_count", 0)
        routes = transit.get("routes_active", [])
        heading_div = transit.get("heading_diversity", 0.5)

        route_str = ", ".join(routes[:5]) if routes else "various routes"

        if style == "la_morning_report":
            parts = [f"On the roads:"]
            if cong > 70:
                parts.append(f"Heavy traffic across the basin. "
                             f"{buses} Metro buses averaging just {avg_spd:.0f} miles per hour. Congestion is at {cong}%.")
            elif cong > 40:
                parts.append(f"Moderate traffic. {buses} buses moving at {avg_spd:.0f} on routes {route_str}. Congestion is at {cong}%.")
            else:
                parts.append(f"Roads are looking good. {buses} buses flowing at {avg_spd:.0f} miles per hour. Congestion is just {cong}%.")
            if heading_div > 0.7:
                parts.append("Movement patterns are scattered, typical of surface street congestion.")
            return " ".join(parts)

        elif style == "noir_city_pulse":
            if cong > 70:
                return (
                    f"The city's arteries are clogged. "
                    f"{buses} buses crawling at {avg_spd:.0f}. "
                    f"Congestion at {cong}%, going nowhere fast."
                )
            elif cong < 20:
                return f"The roads are empty. {buses} buses doing {avg_spd:.0f} miles an hour, {cong}% congestion. Suspicious."
            else:
                return f"Traffic's moving. {buses} buses doing {avg_spd:.0f}. {cong}% congestion. Normal chaos."

        elif style == "old_time_radio_hour":
            if cong > 60:
                return (
                    f"Motorists are advised that traffic conditions are quite congested at {cong}% capacity. "
                    f"{buses} streetcars report an average speed of merely {avg_spd:.0f} miles per hour."
                )
            else:
                return f"The motorways are moving smoothly at {cong}% congestion. {buses} vehicles averaging {avg_spd:.0f} miles per hour."

        elif style == "cyberpunk_city_scan":
            return (
                f"Transit grid: {buses} active units. "
                f"Average velocity: {avg_spd:.0f} MPH. "
                f"Congestion: {cong}%. "
                f"Heading entropy: {heading_div:.2f}. "
                f"Active routes: {route_str}."
            )

        elif style == "calm_documentary":
            return (
                f"Across the city, {buses} Metro buses trace their routes "
                f"at an average of {avg_spd:.0f} miles per hour. "
                f"The flow suggests {cong}% congestion."
            )

        elif style == "surreal_dispatch":
            return (
                f"{buses} buses dream at {avg_spd:.0f} miles per hour, holding {cong}% congestion. "
                f"Their headings scatter with {heading_div:.0%} uncertainty."
            )

        return f"{buses} buses, {cong}% congestion."

    # ---- EARTHQUAKES ----

    def _quake_segment(self, style, quake, city, rng):
        count = quake.get("count", 0)
        max_mag = quake.get("max_magnitude", 0)
        mood = quake.get("seismic_mood", "quiet")
        quakes = quake.get("quakes", [])
        hours = quake.get("lookback_hours", 24)

        # Skip if nothing interesting
        if count == 0:
            if style == "cyberpunk_city_scan":
                return f"Seismic: no events detected in {hours}h window."
            return ""

        top_quake = quakes[0] if quakes else {}
        place = top_quake.get("place", "nearby")
        depth = top_quake.get("depth_km", 0)

        if style == "la_morning_report":
            if max_mag >= 3.0:
                return (
                    f"Seismic activity to report. "
                    f"A magnitude {max_mag} earthquake was recorded {place}, "
                    f"at a depth of {depth:.0f} kilometers. "
                    f"{count} total events in the last {hours} hours."
                )
            elif count > 5:
                return (
                    f"The seismographs have been busy. "
                    f"{count} small quakes recorded in the last {hours} hours, "
                    f"largest a magnitude {max_mag}. Nothing to worry about, but worth noting."
                )
            else:
                return ""

        elif style == "noir_city_pulse":
            if max_mag >= 3.0:
                return (
                    f"The ground shook. Magnitude {max_mag}, {place}. "
                    f"{depth:.0f} kilometers deep. "
                    f"This city sits on borrowed time and everyone knows it."
                )
            elif count > 5:
                return f"The earth's been restless. {count} tremors. She's talking."
            return ""

        elif style == "old_time_radio_hour":
            if max_mag >= 3.0:
                return (
                    f"The seismological bureau reports a tremor of magnitude {max_mag} "
                    f"was recorded {place}. {count} events noted in the past {hours} hours."
                )
            return ""

        elif style == "cyberpunk_city_scan":
            return (
                f"Seismic array: {count} events in {hours}h. "
                f"Maximum magnitude: {max_mag}. "
                f"Depth: {depth:.0f}km. "
                f"Assessment: {mood}."
            )

        elif style == "calm_documentary":
            if max_mag >= 2.5 or count > 3:
                return (
                    f"Beneath the surface, {count} seismic events have been recorded "
                    f"in the past {hours} hours. The strongest, a magnitude {max_mag}, "
                    f"occurred {place}."
                )
            return ""

        elif style == "surreal_dispatch":
            if count > 0:
                return (
                    f"The earth remembers {count} times in {hours} hours. "
                    f"The strongest memory was {max_mag} on the Richter scale, "
                    f"{depth:.0f} kilometers beneath {place}."
                )
            return ""

        return ""

    # ---- CLOSING ----

    def _closing(self, style, city, weather, rng):
        if style == "la_morning_report":
            closings = [
                f"That's your {city} Pulse for now. Stay safe, stay informed.",
                f"This has been the {city} Pulse. We'll be back with the next update.",
                f"That's the latest from the {city} data feed. Have a good one.",
            ]
            return rng.choice(closings)

        elif style == "noir_city_pulse":
            closings = [
                f"That's the city tonight. Same as always. Different as always.",
                f"The data never lies. But the city does. Stay sharp.",
                f"End of report. The city keeps talking, whether you listen or not.",
            ]
            return rng.choice(closings)

        elif style == "old_time_radio_hour":
            return (
                f"And that concludes our {city} Information Hour. "
                f"We return you now to your regularly scheduled programming. "
                f"Good night."
            )

        elif style == "cyberpunk_city_scan":
            return f"Scan complete. All systems nominal. End transmission."

        elif style == "calm_documentary":
            return f"And so {city} continues. Always moving, always changing."

        elif style == "surreal_dispatch":
            return (
                f"The data has finished dreaming. "
                f"{city} returns to the space between measurements."
            )

        return f"End of {city} Pulse."