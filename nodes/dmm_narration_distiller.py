"""
DMM_NarrationDistiller — Distills 5 data-lane summaries into a broadcast-style
narration script sized for ~24 seconds of TTS speech (~60 words at 2.5 wps).

Inputs:  5 optional STRING slots (weather_summary, quake_summary, aq_summary,
         transit_summary, alerts_summary) — direct wires from fetcher slot 1.
Output:  STRING — natural-language narration ready for KokoroTTS.

v1.0  2026-03-14  Initial release.
"""

import logging
import re
import time

log = logging.getLogger("DMM")


# ---------------------------------------------------------------------------
#  Parsing helpers — each extracts key facts from the pipe-delimited summary
# ---------------------------------------------------------------------------

def _parse_weather(s: str) -> str:
    """'Los Angeles [LIVE]: Clear sky | 68°F | Humidity 35% | Wind 5 mph | ...'"""
    if not s:
        return ""
    parts = [p.strip() for p in s.split("|")]
    # parts[0] has city + tag + description
    desc_match = re.search(r":\s*(.+)", parts[0])
    desc = desc_match.group(1).strip() if desc_match else "conditions unavailable"
    temp = parts[1].strip() if len(parts) > 1 else ""
    wind = parts[3].strip() if len(parts) > 3 else ""
    # Natural sentence
    pieces = []
    if desc:
        pieces.append(desc.rstrip("."))
    if temp:
        pieces.append(temp)
    if wind:
        # Match 'Wind 5 mph' or 'wind: 5 mph' case-insensitively
        m = re.search(r"(\d+)\s*mph", wind, re.IGNORECASE)
        if m and m.group(1) != "0":
            pieces.append(f"winds {m.group(1)} mph")
    return ". ".join(pieces) + "." if pieces else ""


def _parse_quake(s: str) -> str:
    """'Los Angeles Seismic [LIVE]: 3 quakes in 24h | Max: M2.1 | Calm'"""
    if not s:
        return ""
    parts = [p.strip() for p in s.split("|")]
    # Case-insensitive search for quake counts
    count_match = re.search(r"(\d+)\s*(?:quakes?|earthquakes?|tremors?)", parts[0], re.IGNORECASE)
    if count_match:
        n = int(count_match.group(1))
        # Find lookback hours (optional)
        hrs_match = re.search(r"in\s*(\d+)\s*h", parts[0], re.IGNORECASE)
        hrs = hrs_match.group(1) if hrs_match else "24"
        
        if n == 0:
            return "No seismic activity."
        
        # Max magnitude in parts[1]
        mag = parts[1].strip() if len(parts) > 1 else ""
        mood = parts[2].strip() if len(parts) > 2 else ""
        
        line = f"{n} quake{'s' if n != 1 else ''} in the last {hrs} hours"
        if mag:
            # Match M2.1 or Max: M2.1
            m_val = re.search(r"M?(\d+\.?\d*)", mag, re.IGNORECASE)
            if m_val:
                line += f", max magnitude {m_val.group(1)}"
        if mood:
            line += f". {mood}"
        return line.rstrip(".") + "."
    return ""


def _parse_aq(s: str) -> str:
    """'Los Angeles Air Quality [LIVE]: US AQI 42 (Good) | PM2.5: 8.2 | UV: 5'"""
    if not s:
        return ""
    parts = [p.strip() for p in s.split("|")]
    # Match AQI number and label even with "US AQI" or "AQI:" prefixes
    aqi_match = re.search(r"AQI\s*(\d+)\s*(?:\(([^)]+)\))?", parts[0], re.IGNORECASE)
    if aqi_match:
        aqi_val = aqi_match.group(1)
        aqi_label = aqi_match.group(2).lower() if aqi_match.group(2) else "moderate"
        uv = ""
        for p in parts:
            uv_match = re.search(r"UV:\s*(\S+)", p, re.IGNORECASE)
            if uv_match:
                uv = uv_match.group(1)
        line = f"Air quality index {aqi_val}, {aqi_label}"
        if uv:
            line += f". UV index {uv}"
        return line.rstrip(".") + "."
    return ""


def _parse_transit(s: str) -> str:
    """'Los Angeles [LIVE MODEL]: 20 buses | avg 17.6 mph | congestion 30%'"""
    if not s:
        return ""
    parts = [p.strip() for p in s.split("|")]
    buses = ""
    speed = ""
    congestion = ""
    for p in parts:
        # Case-insensitive matches for all fields
        b = re.search(r"(\d+)\s*(?:bus|metro\s*bus|unit)", p, re.IGNORECASE)
        if b:
            buses = b.group(1)
        s_match = re.search(r"(?:avg|speed)\s*([\d.]+)\s*mph", p, re.IGNORECASE)
        if s_match:
            speed = s_match.group(1)
        c = re.search(r"congestion\s*:?\s*(\d+)", p, re.IGNORECASE)
        if c:
            congestion = c.group(1)
            
    pieces = []
    if buses:
        pieces.append(f"{buses} metro buses active")
    if speed:
        pieces.append(f"averaging {float(speed):.0f} miles per hour")
    if congestion:
        pct = int(congestion)
        if pct <= 20:
            pieces.append(f"light traffic at {pct}% congestion")
        elif pct <= 50:
            pieces.append(f"moderate congestion at {pct}%")
        else:
            pieces.append(f"heavy congestion at {pct}%")
    return ", ".join(pieces) + "." if pieces else ""


def _parse_alerts(s: str) -> str:
    """'Los Angeles Alerts [LIVE]: All clear.' or '2 active | Level: WARNING | ...'"""
    if not s:
        return ""
    if any(k in s.lower() for k in ["all clear", "no active", "no advisories"]):
        return "No active advisories."
    parts = [p.strip() for p in s.split("|")]
    count_match = re.search(r"(\d+)\s*active", parts[0], re.IGNORECASE)
    if count_match:
        n = int(count_match.group(1))
        event = ""
        for p in parts:
            # Skip labels and counts, find the event name
            if p.strip() and not re.search(r"(Level|active|count)", p, re.IGNORECASE):
                event = p.strip()
        line = f"{n} active alert{'s' if n != 1 else ''}"
        if event:
            line += f": {event}"
        return line.rstrip(".") + "."
    return ""


# ---------------------------------------------------------------------------
#  Node class
# ---------------------------------------------------------------------------

class DMMNarrationDistiller:
    """Distills 5 data-lane summaries into a ~24-second TTS narration."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "distill"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("narration_text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "weather_summary": ("STRING", {"forceInput": True}),
                "quake_summary": ("STRING", {"forceInput": True}),
                "aq_summary": ("STRING", {"forceInput": True}),
                "transit_summary": ("STRING", {"forceInput": True}),
                "alerts_summary": ("STRING", {"forceInput": True}),
            },
        }

    def distill(self, weather_summary="", quake_summary="", aq_summary="",
                transit_summary="", alerts_summary=""):

        # Parse each lane
        weather = _parse_weather(weather_summary)
        quake = _parse_quake(quake_summary)
        aq = _parse_aq(aq_summary)
        transit = _parse_transit(transit_summary)
        alerts = _parse_alerts(alerts_summary)

        # Time context — include exact time for broadcast authenticity
        now = time.localtime()
        hour_12 = now.tm_hour % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = "AM" if now.tm_hour < 12 else "PM"
        exact_time = f"{hour_12}:{now.tm_min:02d} {am_pm}"

        # Combine all parts directly
        # The DMM_NarrationRefiner (Qwen2.5) will rewrite this into a proper broadcast.
        # We just need to ensure all the raw facts are present and legible.
        intro = f"The current LA time is {exact_time}."
        sections = [s for s in [intro, alerts, weather, quake, aq, transit] if s]
        
        narration = " ".join(sections)

        word_count = len(narration.split())
        est_seconds = word_count / 2.5  # Kokoro at 1.0x ≈ 2.5 wps

        log.info("DMM_NarrationDistiller: %d words, ~%.1f sec estimated",
                 word_count, est_seconds)
        log.info("DMM_NarrationDistiller: %s", narration[:120])

        return (narration,)
