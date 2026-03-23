"""
dmm_event_utils — Local event calendar for TTS narration.

Zero API keys needed.  Uses a curated, date-aware calendar of recurring
LA-area events, venues, cultural happenings, farmers markets, and seasonal
highlights.  Events rotate every 60 seconds so each pipeline run gets
variety without repeating in a loop.

City-aware: defaults to Los Angeles but has entries for other major cities.

v1.0  2026-03-22  Initial release (Ticketmaster API).
v2.0  2026-03-22  Replaced API with local calendar — no keys required.
"""

import logging
import random
import time
from datetime import datetime

log = logging.getLogger("DMM.EventUtils")


# ── LA recurring events by day of week (0=Mon … 6=Sun) ──────────────
_LA_DOW_EVENTS = {
    0: [  # Monday
        ("Monday Night Jazz at LACMA", "LACMA", "evening"),
        ("Open mic night at The Hotel Café", "The Hotel Café, Hollywood", "evening"),
        ("Free yoga in Grand Park", "Grand Park, DTLA", "morning"),
    ],
    1: [  # Tuesday
        ("Taco Tuesday at Grand Central Market", "Grand Central Market, DTLA", "all day"),
        ("Griffith Observatory free public telescope viewing", "Griffith Observatory", "evening"),
        ("Comedy night at The Improv", "The Improv, Hollywood", "evening"),
    ],
    2: [  # Wednesday
        ("Free admission at The Broad", "The Broad, DTLA", "all day"),
        ("Farmers market on Arizona Avenue", "Santa Monica", "morning"),
        ("Wednesday night skating at Moonlight Rollerway", "Moonlight Rollerway, Glendale", "evening"),
    ],
    3: [  # Thursday
        ("First Thursday art walk in DTLA", "Downtown LA Arts District", "evening"),
        ("Live music at The Echo", "The Echo, Echo Park", "evening"),
        ("Original Farmers Market open late", "The Original Farmers Market, Fairfax", "evening"),
    ],
    4: [  # Friday
        ("Friday night jazz at Blue Whale", "Blue Whale, Little Tokyo", "evening"),
        ("Abbot Kinney First Fridays", "Abbot Kinney Blvd, Venice", "evening"),
        ("Night market at ROW DTLA", "ROW DTLA", "evening"),
        ("Free concerts at The Getty Center", "The Getty Center", "evening"),
    ],
    5: [  # Saturday
        ("Hollywood Farmers Market", "Hollywood & Ivar", "morning"),
        ("Smorgasburg LA food market", "ROW DTLA", "all day"),
        ("Free outdoor movies at Street Food Cinema", "various LA parks", "evening"),
        ("Echo Park Pedal Boats", "Echo Park Lake", "all day"),
        ("Third Street Promenade street performers", "Santa Monica", "all day"),
    ],
    6: [  # Sunday
        ("Silver Lake Farmers Market", "Silver Lake", "morning"),
        ("Free day at select LA museums", "various museums", "all day"),
        ("Venice Beach drum circle", "Venice Beach", "afternoon"),
        ("Sunday brunch jazz at Vibrato Grill", "Vibrato Grill, Bel Air", "morning"),
        ("Mar Vista Farmers Market", "Mar Vista", "morning"),
    ],
}

# ── Monthly / seasonal LA events ─────────────────────────────────────
_LA_MONTHLY_EVENTS = {
    1: [  # January
        ("Tournament of Roses Parade viewing spots still buzzing", "Pasadena", "all day"),
        ("Lunar New Year celebrations in Chinatown", "Chinatown, DTLA", "all day"),
        ("Whale watching season along the coast", "Long Beach & San Pedro", "morning"),
    ],
    2: [  # February
        ("Chinese New Year Golden Dragon Parade", "Chinatown, DTLA", "afternoon"),
        ("Valentines Day events along the coast", "Santa Monica Pier", "evening"),
        ("Oscar buzz screenings at local theaters", "various theaters", "evening"),
    ],
    3: [  # March
        ("Cherry blossom season at Lake Balboa", "Lake Balboa Park", "all day"),
        ("St. Patrick's Day celebrations in Hermosa Beach", "Hermosa Beach", "all day"),
        ("LA Marathon season training runs", "various routes", "morning"),
        ("Wildflower super bloom hikes in season", "various trails", "morning"),
    ],
    4: [  # April
        ("Dodgers home season in full swing", "Dodger Stadium", "evening"),
        ("Coachella weekend energy around town", "LA area", "all day"),
        ("Earth Day celebrations at Griffith Park", "Griffith Park", "all day"),
        ("Thai New Year Songkran Festival", "Thai Town, Hollywood", "all day"),
    ],
    5: [  # May
        ("Cinco de Mayo celebrations on Olvera Street", "Olvera Street, DTLA", "all day"),
        ("KCRW Summer Nights series kicks off", "various venues", "evening"),
        ("Memorial Day beach gatherings", "Santa Monica Beach", "all day"),
        ("Fiesta Hermosa arts and crafts fair", "Hermosa Beach", "all day"),
    ],
    6: [  # June
        ("LA Pride Festival in West Hollywood", "West Hollywood", "all day"),
        ("Free summer concerts at Hollywood Bowl", "Hollywood Bowl", "evening"),
        ("Surf City Surf Dog competition", "Huntington Beach", "morning"),
        ("Make Music Day free performances citywide", "various venues", "all day"),
    ],
    7: [  # July
        ("Fourth of July fireworks at the Hollywood Bowl", "Hollywood Bowl", "evening"),
        ("Twilight Concert Series at Santa Monica Pier", "Santa Monica Pier", "evening"),
        ("Outfest LA LGBTQ Film Festival", "various theaters", "evening"),
        ("Marina del Rey summer concerts", "Burton Chace Park", "evening"),
    ],
    8: [  # August
        ("Nisei Week Japanese Festival in Little Tokyo", "Little Tokyo, DTLA", "all day"),
        ("Summer concert series wrapping up at The Getty", "The Getty Center", "evening"),
        ("Perseid meteor shower viewing from Griffith Observatory", "Griffith Observatory", "evening"),
        ("Back to school vibes around UCLA and USC", "Westwood & USC area", "all day"),
    ],
    9: [  # September
        ("LA County Fair in Pomona", "Pomona Fairplex", "all day"),
        ("DTLA Art Walk season", "Downtown LA Arts District", "evening"),
        ("Abbott Kinney Festival", "Abbott Kinney Blvd, Venice", "all day"),
        ("Mexican Independence Day celebrations", "Olvera Street, DTLA", "evening"),
    ],
    10: [  # October
        ("West Hollywood Halloween Carnival", "Santa Monica Blvd, WeHo", "evening"),
        ("Dia de los Muertos preparations at Olvera Street", "Olvera Street, DTLA", "all day"),
        ("Haunted Hayride at Griffith Park", "Griffith Park", "evening"),
        ("LA Dodgers postseason energy around town", "Dodger Stadium area", "evening"),
    ],
    11: [  # November
        ("Dia de los Muertos celebrations at Hollywood Forever", "Hollywood Forever Cemetery", "evening"),
        ("Holiday light displays starting up on Rodeo Drive", "Rodeo Drive, Beverly Hills", "evening"),
        ("Thanksgiving Farmers Market specials", "various markets", "morning"),
        ("LA Auto Show at the Convention Center", "LA Convention Center", "all day"),
    ],
    12: [  # December
        ("Holiday boat parade in Marina del Rey", "Marina del Rey", "evening"),
        ("Griffith Park light festival", "Griffith Park", "evening"),
        ("Ice skating at Pershing Square", "Pershing Square, DTLA", "all day"),
        ("Holiday lights at The Grove", "The Grove", "evening"),
        ("New Years Eve celebrations along Grand Avenue", "DTLA", "evening"),
    ],
}

# ── Date-specific notable LA events ──────────────────────────────────
_LA_DATE_EVENTS = {
    (1, 1): ("Rose Parade day in Pasadena", "Pasadena", "morning"),
    (1, 15): ("Martin Luther King Jr. Day events across LA", "various venues", "all day"),
    (2, 14): ("Valentine's Day sunset walks along the coast", "Santa Monica & Venice", "evening"),
    (3, 17): ("St. Patrick's Day celebrations across LA", "various venues", "all day"),
    (4, 22): ("Earth Day volunteer cleanups at LA beaches", "various beaches", "morning"),
    (5, 5): ("Cinco de Mayo on Olvera Street", "Olvera Street, DTLA", "all day"),
    (5, 25): ("Geek culture events around town", "various venues", "all day"),
    (6, 19): ("Juneteenth celebrations and festivals", "Leimert Park", "all day"),
    (7, 4): ("Fourth of July fireworks all across LA", "citywide", "evening"),
    (9, 16): ("Mexican Independence Day at Olvera Street", "Olvera Street, DTLA", "evening"),
    (10, 31): ("Halloween on the Sunset Strip and WeHo", "West Hollywood", "evening"),
    (11, 1): ("Dia de los Muertos at Hollywood Forever", "Hollywood Forever Cemetery", "evening"),
    (12, 24): ("Christmas Eve luminaria walks", "various neighborhoods", "evening"),
    (12, 31): ("New Years Eve countdown at Grand Park", "Grand Park, DTLA", "evening"),
}

# ── Other city fallbacks (if city isn't LA) ──────────────────────────
_OTHER_CITY_EVENTS = {
    "new york": [
        "Live jazz at Lincoln Center",
        "Free Shakespeare in Central Park",
        "Brooklyn Flea Market happening today",
        "Street performers in Washington Square Park",
    ],
    "san francisco": [
        "Farmers market at the Ferry Building",
        "Free concerts at Stern Grove",
        "Fishermans Wharf street performers active today",
        "Golden Gate Park outdoor activities",
    ],
    "chicago": [
        "Live blues on the Magnificent Mile",
        "Free events at Millennium Park",
        "Chicago Riverwalk dining and music",
    ],
    "seattle": [
        "Pike Place Market buskers performing today",
        "Free First Thursday art walks",
        "Live music in Capitol Hill venues tonight",
    ],
    "miami": [
        "Live music on Ocean Drive",
        "Wynwood Walls open for free viewing",
        "Sunset drum circle at South Beach",
    ],
    "austin": [
        "Live music on Sixth Street tonight",
        "Food truck rallies around South Congress",
        "Barton Springs open for swimming",
    ],
}

# ── Time-of-day intros ───────────────────────────────────────────────
_TIME_INTROS = {
    "morning": [
        "This morning in {city}",
        "Starting the day in {city}",
        "Happening this morning",
    ],
    "afternoon": [
        "This afternoon in {city}",
        "Happening this afternoon",
        "Out and about this afternoon",
    ],
    "evening": [
        "Tonight in {city}",
        "Happening this evening",
        "Out tonight in {city}",
    ],
    "all day": [
        "Happening today in {city}",
        "Going on today",
        "Around {city} today",
    ],
}


def _get_time_of_day():
    """Return current time-of-day bucket."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    else:
        return "evening"


def _match_time(event_time: str, current_tod: str) -> bool:
    """Check if an event's time-of-day matches or is 'all day'."""
    if event_time == "all day":
        return True
    return event_time == current_tod


def get_upcoming_event(city: str = "Los Angeles") -> str:
    """Return a formatted string about a local event happening today.

    Picks from date-specific events first, then monthly/seasonal events,
    then day-of-week recurring events.  Prioritizes events matching the
    current time of day.  Rotates every 60 seconds.

    No API keys required — entirely local calendar data.

    Returns empty string only if zero events match (very unlikely for LA).
    """
    now = datetime.now()
    month = now.month
    day = now.day
    dow = now.weekday()  # 0=Monday
    tod = _get_time_of_day()
    seed = int(time.time()) // 60  # rotate every 60s

    city_lower = city.lower().strip()
    is_la = any(k in city_lower for k in ("los angeles",))

    # ── Non-LA cities get a simple one-liner ─────────────────────────
    if not is_la:
        for cname, events in _OTHER_CITY_EVENTS.items():
            if cname in city_lower:
                rng = random.Random(seed)
                pick = rng.choice(events)
                return f"Around {city} today: {pick}."
        # Unknown city — skip events entirely
        return ""

    # ── LA event pool ────────────────────────────────────────────────
    candidates = []

    # 1) Date-specific (highest priority)
    date_key = (month, day)
    if date_key in _LA_DATE_EVENTS:
        ev = _LA_DATE_EVENTS[date_key]
        candidates.append(ev)

    # 2) Monthly/seasonal
    for ev in _LA_MONTHLY_EVENTS.get(month, []):
        candidates.append(ev)

    # 3) Day-of-week recurring
    for ev in _LA_DOW_EVENTS.get(dow, []):
        candidates.append(ev)

    if not candidates:
        return ""

    # Prefer events matching time-of-day, but fall back to all
    time_matched = [c for c in candidates if _match_time(c[2], tod)]
    pool = time_matched if time_matched else candidates

    rng = random.Random(seed)
    name, venue, ev_tod = rng.choice(pool)

    # Build natural sentence
    intros = _TIME_INTROS.get(ev_tod, _TIME_INTROS["all day"])
    intro = rng.choice(intros).format(city=city)

    parts = [f"{intro}: {name}"]
    if venue and venue.lower() not in name.lower():
        parts[0] += f" at {venue}"
    parts[0] += "."

    # Add a color sentence about the venue or event
    _an = "an" if tod in ("evening", "afternoon") else "a"
    _color_lines = [
        f"A great way to enjoy the {tod} in {city}.",
        f"Always a good time in the neighborhood.",
        f"One of those things that makes {city} special.",
        f"Worth checking out if you are in the area.",
        f"Perfect for {_an} {tod} out in the city.",
    ]
    parts.append(rng.choice(_color_lines))

    return " ".join(parts)
