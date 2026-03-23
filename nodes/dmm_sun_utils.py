"""
dmm_sun_utils — Sunrise/sunset calculator + data-based friendly statements.

Pure math approach using the NOAA solar equations.
Default coordinates: Los Angeles (34.0522°N, 118.2437°W).

Used by TTS narration nodes and the ProceduralClip HUD to display
the date, next sunrise/sunset, and a nice human statement based on
current conditions.

v1.0  2026-03-22  Initial release.
v1.1  2026-03-22  Added moon phase calculator, eclipse lookup, get_moon_info().
"""

import math
import time
from datetime import datetime, timedelta, timezone

# ── LA coordinates ────────────────────────────────────────────────────
LA_LAT = 34.0522
LA_LON = -118.2437
LA_TZ_OFFSET = -7  # PDT (summer); -8 for PST — we auto-detect below


def _tz_offset_hours():
    """Return current UTC offset in hours for the local system clock."""
    now = datetime.now()
    utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
    diff = now - utc_now
    return round(diff.total_seconds() / 3600)


def _sun_times(lat=LA_LAT, lon=LA_LON, date=None):
    """Calculate sunrise and sunset for a given date and location.

    Uses simplified NOAA solar equations.
    Returns (sunrise_dt, sunset_dt) as naive datetime in local time,
    or (None, None) on polar day/night edge cases.
    """
    if date is None:
        date = datetime.now()
    tz_offset = _tz_offset_hours()

    # Day of year
    n = date.timetuple().tm_yday

    # Solar noon approximation
    lng_hour = lon / 15.0

    # Sunrise estimate
    t_rise = n + (6 - lng_hour) / 24.0
    t_set = n + (18 - lng_hour) / 24.0

    results = {}
    for label, t in [("rise", t_rise), ("set", t_set)]:
        # Sun's mean anomaly
        M = 0.9856 * t - 3.289

        # Sun's true longitude
        L = M + 1.916 * math.sin(math.radians(M)) + 0.020 * math.sin(
            math.radians(2 * M)) + 282.634
        L = L % 360

        # Right ascension
        RA = math.degrees(math.atan(0.91764 * math.tan(math.radians(L))))
        RA = RA % 360

        # RA in same quadrant as L
        L_quad = (math.floor(L / 90)) * 90
        RA_quad = (math.floor(RA / 90)) * 90
        RA = RA + (L_quad - RA_quad)
        RA = RA / 15.0  # convert to hours

        # Sun's declination
        sin_dec = 0.39782 * math.sin(math.radians(L))
        cos_dec = math.cos(math.asin(sin_dec))

        # Hour angle
        zenith = 90.833  # official sunrise/sunset
        cos_H = (math.cos(math.radians(zenith)) -
                 sin_dec * math.sin(math.radians(lat))) / (
                 cos_dec * math.cos(math.radians(lat)))

        if cos_H > 1 or cos_H < -1:
            return None, None  # no sunrise/sunset (polar)

        if label == "rise":
            H = 360 - math.degrees(math.acos(cos_H))
        else:
            H = math.degrees(math.acos(cos_H))
        H = H / 15.0

        # Local time
        T = H + RA - 0.06571 * t - 6.622
        UT = (T - lng_hour) % 24
        local_t = (UT + tz_offset) % 24

        hour = int(local_t)
        minute = int((local_t - hour) * 60)
        results[label] = date.replace(hour=hour, minute=minute, second=0,
                                       microsecond=0)

    return results.get("rise"), results.get("set")


# ── Moon phase calculator ─────────────────────────────────────────
# Uses Conway's method — accurate to ~1 day.
# Returns phase age (0-29.53 days) and phase name.

_MOON_PHASE_NAMES = [
    (0.0,   "New Moon"),          # 0
    (1.85,  "Waxing Crescent"),   # 1-6
    (7.38,  "First Quarter"),     # 7
    (8.38,  "Waxing Gibbous"),    # 8-13
    (14.77, "Full Moon"),         # 14-15
    (15.77, "Waning Gibbous"),    # 16-21
    (22.15, "Last Quarter"),      # 22
    (23.15, "Waning Crescent"),   # 23-28
    (29.53, "New Moon"),          # back to 0
]

_MOON_PHASE_EMOJI = {
    "New Moon":         "\U0001F311",  # dark
    "Waxing Crescent":  "\U0001F312",
    "First Quarter":    "\U0001F313",
    "Waxing Gibbous":   "\U0001F314",
    "Full Moon":        "\U0001F315",
    "Waning Gibbous":   "\U0001F316",
    "Last Quarter":     "\U0001F317",
    "Waning Crescent":  "\U0001F318",
}


def _moon_phase_age(date=None):
    """Return the moon's age in days (0-29.53) for the given date.

    Uses a simplified synodic calculation anchored to the known
    new moon of January 6, 2000 00:18 UTC.
    """
    if date is None:
        date = datetime.now()
    # Known new moon epoch: 2000-01-06 18:14 UTC
    epoch = datetime(2000, 1, 6, 18, 14, 0)
    diff = (date - epoch).total_seconds()
    synodic = 29.53058867  # days
    age = (diff / 86400.0) % synodic
    return age


def _phase_name(age):
    """Convert moon age (days) to human-readable phase name."""
    for i in range(len(_MOON_PHASE_NAMES) - 1):
        if age < _MOON_PHASE_NAMES[i + 1][0]:
            return _MOON_PHASE_NAMES[i][1]
    return "New Moon"


def _moon_illumination(age):
    """Approximate moon illumination percentage (0-100) from age."""
    # Cosine model: 0% at new moon (age=0), 100% at full (age=14.77)
    synodic = 29.53058867
    return round((1 - math.cos(2 * math.pi * age / synodic)) / 2 * 100)


# ── Eclipse / notable lunar event lookup ──────────────────────────
# Pre-computed notable events visible from LA through 2028.
# (year, month, day, event_type, description)
_NOTABLE_LUNAR_EVENTS = [
    (2026, 3, 14, "total_lunar", "Total Lunar Eclipse visible from LA"),
    (2026, 8, 12, "partial_solar", "Partial Solar Eclipse (not visible from LA)"),
    (2026, 9, 7,  "total_lunar", "Total Lunar Eclipse visible from LA"),
    (2027, 2, 20, "penumbral_lunar", "Penumbral Lunar Eclipse"),
    (2027, 7, 18, "penumbral_lunar", "Penumbral Lunar Eclipse"),
    (2027, 8, 2,  "total_solar", "Total Solar Eclipse (path across N Africa/Arabia)"),
    (2028, 1, 12, "partial_lunar", "Partial Lunar Eclipse"),
    (2028, 6, 26, "penumbral_lunar", "Penumbral Lunar Eclipse"),
    (2028, 7, 11, "partial_solar", "Partial Solar Eclipse"),
    (2028, 12, 21, "total_lunar", "Total Lunar Eclipse visible from LA"),
]

# Supermoon dates (full moons at perigee) — approximate
_SUPERMOON_DATES = [
    (2026, 5, 26), (2026, 6, 24), (2026, 7, 24),
    (2027, 4, 16), (2027, 5, 15), (2027, 6, 14),
    (2028, 8, 19), (2028, 9, 17), (2028, 10, 17),
]


def get_moon_info(date=None):
    """Return dict with moon phase, illumination, and upcoming events.

    Keys:
        phase_name   — e.g. "Waxing Gibbous"
        phase_emoji  — e.g. "\U0001F314"
        illumination — int 0-100
        age_days     — float 0-29.53
        is_full      — bool (within 1 day of full)
        is_new       — bool (within 1 day of new)
        is_supermoon — bool
        next_eclipse — str or "" (upcoming eclipse within 60 days)
        next_full_days — int days until next full moon
        next_new_days  — int days until next new moon
    """
    if date is None:
        date = datetime.now()

    age = _moon_phase_age(date)
    name = _phase_name(age)
    emoji = _MOON_PHASE_EMOJI.get(name, "")
    illum = _moon_illumination(age)

    synodic = 29.53058867
    # Days until next full (age ~14.77) and new (age ~0/29.53)
    full_target = 14.77
    if age <= full_target:
        next_full = full_target - age
    else:
        next_full = synodic - age + full_target
    next_new = synodic - age if age > 0.5 else -age  # handle near-new
    next_new = max(0, synodic - age) if age > 0.5 else synodic - age
    if next_new < 0:
        next_new += synodic

    is_full = abs(age - full_target) < 1.0
    is_new = age < 1.0 or age > (synodic - 1.0)

    # Supermoon check
    is_supermoon = False
    for sy, sm, sd in _SUPERMOON_DATES:
        sm_date = datetime(sy, sm, sd)
        if abs((date - sm_date).days) <= 1:
            is_supermoon = True
            break

    # Next eclipse within 60 days
    next_eclipse = ""
    for ey, em, ed, etype, edesc in _NOTABLE_LUNAR_EVENTS:
        try:
            edate = datetime(ey, em, ed)
            diff_days = (edate - date).days
            if 0 <= diff_days <= 60:
                next_eclipse = f"{edesc} on {edate.strftime('%b %d')}"
                break
        except ValueError:
            continue

    return {
        "phase_name": name,
        "phase_emoji": emoji,
        "illumination": illum,
        "age_days": round(age, 1),
        "is_full": is_full,
        "is_new": is_new,
        "is_supermoon": is_supermoon,
        "next_eclipse": next_eclipse,
        "next_full_days": int(next_full),
        "next_new_days": int(next_new),
    }


def get_sun_info():
    """Return a dict with formatted sunrise/sunset and 'next' event info.

    Keys:
        sunrise_str   — e.g. "6:48 AM"
        sunset_str    — e.g. "7:12 PM"
        next_event    — "sunrise" or "sunset"
        next_time_str — e.g. "7:12 PM"
        minutes_until — int minutes until next event
        date_str      — e.g. "Sunday, March 22, 2026"
        date_short    — e.g. "Mar 22, 2026"
    """
    now = datetime.now()
    rise, sset = _sun_times(date=now)

    def fmt(dt):
        if dt is None:
            return "—"
        h = dt.hour % 12
        if h == 0:
            h = 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{h}:{dt.minute:02d} {ampm}"

    sunrise_str = fmt(rise)
    sunset_str = fmt(sset)

    # Determine next event
    if rise and sset:
        if now < rise:
            next_event = "sunrise"
            next_dt = rise
        elif now < sset:
            next_event = "sunset"
            next_dt = sset
        else:
            # After today's sunset — next sunrise is tomorrow
            tomorrow = now + timedelta(days=1)
            tom_rise, _ = _sun_times(date=tomorrow)
            next_event = "sunrise"
            next_dt = tom_rise if tom_rise else rise + timedelta(days=1)
    else:
        next_event = "sunrise"
        next_dt = now

    mins_until = max(0, int((next_dt - now).total_seconds() / 60))
    next_time_str = fmt(next_dt)

    date_str = now.strftime("%A, %B %d, %Y").replace(" 0", " ")
    date_short = now.strftime("%b %d, %Y").replace(" 0", " ")

    return {
        "sunrise_str": sunrise_str,
        "sunset_str": sunset_str,
        "next_event": next_event,
        "next_time_str": next_time_str,
        "minutes_until": mins_until,
        "date_str": date_str,
        "date_short": date_short,
    }


# ── LA city facts — interesting "on this day" / local color for TTS ───

# Big pool of LA-specific facts, randomized each run.  Keeps the broadcast
# feeling fresh even on days without a specific calendar event.
_LA_FACTS = [
    "Los Angeles was founded in 1781 by Spanish settlers and its full original name has 34 words.",
    "The LA metro area is home to over 13 million people, making it the second largest in the country.",
    "Los Angeles has more museums per capita than any other city in the United States.",
    "The Griffith Observatory has been a free public observatory since it opened in 1935.",
    "LA's coastline stretches 75 miles, from Malibu all the way down to San Pedro.",
    "The Hollywood sign was originally built in 1923 and read Hollywoodland — a real estate ad.",
    "Dodger Stadium is the largest baseball stadium in the world by seating capacity.",
    "The LA River runs 51 miles from Canoga Park to Long Beach, mostly encased in concrete.",
    "Angels Flight in downtown LA is the world's shortest railway, just 298 feet long.",
    "Los Angeles has hosted the Summer Olympics twice — 1932 and 1984 — and will host again in 2028.",
    "The Watts Towers were built by one man, Simon Rodia, over 33 years using scrap metal and broken tiles.",
    "Olvera Street in downtown LA is considered the birthplace of the city.",
    "The Bradbury Building, built in 1893, is one of the oldest commercial buildings in downtown LA.",
    "Venice Beach was originally designed as a replica of Venice, Italy, complete with canals.",
    "LA gets about 284 sunny days per year, more than almost any other major US city.",
    "The La Brea Tar Pits have been trapping animals for over 50,000 years, right in the middle of the city.",
    "JPL in Pasadena has been the heart of NASA's robotic space exploration since 1958.",
    "The Getty Center sits on a hilltop in Brentwood and admission has been free since it opened in 1997.",
    "LA's Union Station opened in 1939 and is the last of the great American railway stations.",
    "The Miracle Mile on Wilshire Boulevard was one of the first commercial districts designed for cars.",
    "Echo Park Lake has been a neighborhood gathering spot since the 1890s.",
    "The Crenshaw district is the cultural heart of Black Los Angeles.",
    "Koreatown in LA is the most densely populated neighborhood in the entire city.",
    "The original Randy's Donuts giant donut on Manchester Boulevard has been an LA landmark since 1953.",
    "Santa Monica Pier has been standing since 1909 and marks the western end of Route 66.",
    "The Huntington Library in San Marino has one of only 12 remaining copies of the Gutenberg Bible.",
    "El Pueblo de Los Angeles is the historic center where the city was founded 245 years ago.",
    "Mulholland Drive was named after William Mulholland, who engineered the LA Aqueduct in 1913.",
    "The Broad museum in downtown LA houses over 2,000 works of contemporary art.",
    "Griffith Park is one of the largest urban parks in North America at over 4,300 acres.",
    "The Metro Gold Line connects Pasadena to downtown LA along what was once the Pacific Electric route.",
    "Runyon Canyon is one of the most popular hikes in the city, with views from the Hollywood Hills to the ocean.",
    "The Rose Bowl in Pasadena has hosted five Super Bowls and the World Cup final.",
    "Los Angeles has the largest Thai population outside of Thailand.",
    "Boyle Heights has been one of LA's most culturally rich neighborhoods for over a century.",
    "The Staples Center — now Crypto.com Arena — opened in 1999 and hosts over 250 events a year.",
    "Little Tokyo in downtown LA is one of only three official Japantowns in the United States.",
    "The Sepulveda Pass carries more daily traffic than almost any mountain crossing in the world.",
    "Grand Central Market in downtown LA has been feeding Angelenos since 1917.",
    "The original Bob's Big Boy in Burbank is the oldest remaining Big Boy restaurant in America.",
]

# Keyed by (month, day) — LA-specific calendar events.
_LA_DATE_FACTS = {
    (1, 1): ["New Year's Day — the Rose Parade has rolled through Pasadena every year since 1890.",
             "Happy New Year. Los Angeles officially became a city on this date in 1850."],
    (1, 17): ["On this day in 1994, the Northridge earthquake struck Los Angeles at 4:31 AM, magnitude 6.7. The city rebuilt stronger."],
    (2, 14): ["Happy Valentine's Day from the City of Angels."],
    (3, 14): ["Happy Pi Day — 3.14. A good excuse to grab a slice at your favorite LA pie shop."],
    (3, 20): ["It's the spring equinox. The days are getting longer and LA's golden light is about to get even better."],
    (3, 22): ["World Water Day. LA gets about 15 inches of rain a year — every drop matters in this city."],
    (4, 22): ["Happy Earth Day. LA's air quality has come a long way since the smog-filled days of the 1970s."],
    (5, 5): ["Feliz Cinco de Mayo! LA's celebration on Olvera Street is one of the largest in the country."],
    (6, 21): ["Summer solstice — the longest day of the year. LA gets over 14 hours of sunshine today."],
    (7, 4): ["Happy Fourth of July. The fireworks over the Hollywood Bowl are a Los Angeles tradition."],
    (9, 4): ["On this day in 1781, Los Angeles was officially founded by 44 Spanish settlers."],
    (10, 31): ["Happy Halloween. The West Hollywood Halloween Parade is one of the biggest in the world."],
    (11, 1): ["Dia de los Muertos. Olvera Street and Hollywood Forever Cemetery host beautiful celebrations."],
    (12, 21): ["Winter solstice — the shortest day of the year. But even today, LA gets over 9 hours of sunlight."],
    (12, 31): ["New Year's Eve. The Grand Park countdown in downtown LA draws tens of thousands."],
}

# Day-of-week color for broadcasts
_DOW_FACTS = {
    0: "It's Monday — the start of a new week in the City of Angels.",
    1: "It's Tuesday — the workweek is rolling in Los Angeles.",
    2: "It's Wednesday — halfway through the week, LA.",
    3: "It's Thursday — the weekend is almost in sight.",
    4: "It's Friday — the weekend starts now if you want it to.",
    5: "It's Saturday — LA's farmers markets, beaches, and hiking trails are calling.",
    6: "It's Sunday — a good day to slow down and recharge before the week ahead.",
}


def get_date_fact():
    """Return 1-2 interesting LA-focused sentences about today for TTS.

    Combines a day-of-week opener with either a calendar-specific LA fact
    or a random LA city fact.  Rotates every 60 seconds so repeated runs
    on the same day feel fresh.
    """
    import random
    now = datetime.now()
    rng = random.Random(int(time.time()) // 60)

    parts = []

    # Day of week flavor
    dow = _DOW_FACTS.get(now.weekday(), "")
    if dow:
        parts.append(dow)

    # Check for calendar-specific LA fact first
    key = (now.month, now.day)
    cal_facts = _LA_DATE_FACTS.get(key)
    if cal_facts:
        parts.append(rng.choice(cal_facts))
    else:
        # Random LA city fact — always something about the city
        parts.append(rng.choice(_LA_FACTS))

    return " ".join(parts)


# ── Data-based friendly statements ────────────────────────────────────

def get_nice_statement(weather_data=None, aq_data=None, sun_info=None):
    """Generate a friendly, data-driven statement based on current conditions.

    Returns a short sentence like:
        "You better bring sunscreen — UV is high today!"
        "Grab a jacket, it's cooler than you think out there."
        "Beautiful evening ahead — sunset's at 7:12 PM."

    Falls back to generic pleasant closers if no notable data.
    """
    import random
    rng = random.Random(int(time.time()) // 60)  # changes every minute

    statements = []

    if weather_data:
        temp = weather_data.get("temp_f", 72)
        wind = weather_data.get("wind_speed_mph", 5)
        humidity = weather_data.get("humidity", 50)
        rain = weather_data.get("rain_1h_mm", 0)
        desc = weather_data.get("description", "").lower()

        # Temperature-based
        if temp > 95:
            statements.append(
                "Stay hydrated out there — it's absolutely scorching today. "
                "If you can, stick to the shade and save the outdoor plans for the evening. "
                "Your body will thank you for that extra water bottle."
            )
            statements.append(
                "It's a hot one, Los Angeles. The kind of day where the asphalt shimmers "
                "and you start rethinking that midday errand. Maybe save the outdoor run "
                "for after sundown and keep the AC close."
            )
        elif temp > 85:
            statements.append(
                "You better bring sunscreen — it's warm out there and the sun isn't messing around. "
                "A hat and some water go a long way on a day like this. "
                "Enjoy it, but be smart about it."
            )
            statements.append(
                "Perfect pool weather if you can find one. "
                "It's the kind of LA afternoon where you want to be near water, "
                "or at least under a good umbrella with a cold drink in hand."
            )
            statements.append(
                "Don't forget the water bottle today — your future self will appreciate it. "
                "It's beautiful out there, but that sun is doing real work. "
                "Stay cool and stay hydrated, LA."
            )
        elif temp > 72:
            statements.append(
                "Beautiful day to be outside — this is the kind of weather people move to LA for. "
                "Whether it's a walk, a coffee run, or just sitting in the sun for a minute, "
                "get out there and enjoy it."
            )
            statements.append(
                "The kind of day that makes you glad you're in Los Angeles. "
                "Not too hot, not too cool — just right. "
                "Take a moment to appreciate it before the next heatwave rolls in."
            )
            statements.append(
                "Great weather for a walk today. The temperature is sitting in that perfect sweet spot "
                "where you don't need a jacket and you're not breaking a sweat. "
                "Make the most of it."
            )
        elif temp > 58:
            statements.append(
                "Grab a light layer — it's pleasant outside but there's a little coolness in the air. "
                "Perfect for a morning coffee walk or an evening stroll. "
                "The kind of weather that makes you slow down and actually enjoy the commute."
            )
            statements.append(
                "Nice and comfortable out there right now. "
                "A light jacket might be nice if you're heading out early or staying out late, "
                "but otherwise it's smooth sailing, LA."
            )
        elif temp > 45:
            statements.append(
                "Grab a jacket — it's cooler than you think out there. "
                "That morning chill has some bite to it today, "
                "so layer up before you head out. You won't regret it."
            )
            statements.append(
                "Layer up before you step outside — it's brisk out there this morning. "
                "LA doesn't get cold often, but when it does, it catches people off guard. "
                "A warm drink and a good jacket are your best friends today."
            )
        else:
            statements.append(
                "Bundle up, Los Angeles! It's unusually cold out there today. "
                "This is one of those rare days where you actually need a real coat — not just a hoodie. "
                "Stay warm and maybe treat yourself to something hot."
            )
            statements.append(
                "Definitely a hot coffee kind of morning — it's chilly by LA standards. "
                "If you've got a scarf buried somewhere in your closet, today's the day to dig it out. "
                "Stay cozy out there."
            )

        # Rain — only trigger on actual measured precipitation (> 0.1mm),
        # not on condition label alone (OWM sometimes reports "Rain" with 0mm actual)
        if rain > 0.1:
            statements.append(
                "Don't forget your umbrella today — LA rain is rare but real, "
                "and nobody likes getting caught in it unprepared. "
                "Drive carefully out there and give yourself some extra time on the road."
            )
            statements.append(
                "Rainy day in the City of Angels. The streets get slick fast when it rains here, "
                "so take it easy on the road. On the bright side, the city always smells amazing "
                "after a good rain."
            )

        # Wind
        if wind > 25:
            statements.append(
                "Hold onto your hat — it's seriously windy out there today. "
                "If you've got anything lightweight on your balcony, bring it inside. "
                "The gusts are no joke this afternoon."
            )
        elif wind > 15:
            statements.append(
                "It's breezy out there — the kind of wind that messes up your hair "
                "the second you step outside. Not dangerous, just annoying. "
                "At least it keeps the air moving."
            )

        # Humidity
        if humidity > 80:
            statements.append(
                "It's humid out there — the air feels thick and heavy today. "
                "LA doesn't usually do humidity, so when it shows up, you really feel it. "
                "Stay cool and maybe skip the blowout."
            )
        elif humidity < 15:
            statements.append(
                "The air is bone dry today — your skin and your sinuses will notice. "
                "Keep some water nearby and maybe some lip balm too. "
                "Dry air is sneaky — it dehydrates you before you even realize it."
            )

    if aq_data:
        aqi = aq_data.get("us_aqi", 50)
        uv = aq_data.get("uv_index", 3)

        if aqi > 150:
            statements.append(
                "Air quality is rough today — the AQI is elevated enough that you might want to keep it indoors. "
                "If you have to go out, keep it short and avoid heavy exercise outside. "
                "Your lungs will appreciate the caution."
            )
        elif aqi > 100:
            statements.append(
                "Air quality is a bit iffy right now — nothing extreme, but sensitive folks should take it easy. "
                "If you've got asthma or allergies, maybe keep the windows closed today. "
                "The air should clear up as conditions shift."
            )

        if uv >= 10:
            statements.append(
                "UV is extreme today — sunscreen is absolutely non-negotiable. "
                "We're talking SPF 50 minimum, reapply every two hours, and find some shade when you can. "
                "The sun is not playing games today, LA."
            )
            statements.append(
                "You BETTER bring sunscreen — the UV index is off the charts right now. "
                "This is the kind of day where you burn in fifteen minutes without protection. "
                "Hat, shades, sunscreen — the whole kit."
            )
        elif uv >= 7:
            statements.append(
                "You better bring sunscreen — UV is high today and climbing. "
                "Even if it doesn't feel that hot, those rays are working overtime. "
                "Protect your skin and you'll thank yourself later."
            )
            statements.append(
                "SPF 50 kind of day out there. Don't skip it even if it's cloudy — "
                "UV rays don't care about cloud cover. "
                "A little prevention goes a long way, especially in LA."
            )

    if sun_info:
        next_ev = sun_info.get("next_event", "")
        next_t = sun_info.get("next_time_str", "")
        mins = sun_info.get("minutes_until", 0)

        if next_ev == "sunset" and mins < 90:
            statements.append(
                f"Golden hour is coming — sunset's at {next_t} tonight. "
                "If you've got a west-facing window or a rooftop nearby, this is your moment. "
                "LA sunsets don't disappoint."
            )
            statements.append(
                f"Beautiful evening ahead — sunset at {next_t}. "
                "The light's about to get gorgeous over the city. "
                "Grab your phone, grab a friend, and go enjoy the show."
            )
        elif next_ev == "sunrise" and mins < 60:
            statements.append(
                f"Early bird? Sunrise is at {next_t} this morning. "
                "There's something special about watching the light come up over Los Angeles — "
                "the whole city wakes up one neighborhood at a time."
            )
            statements.append(
                f"The sun's almost up — {next_t} is the magic number. "
                "New day, new data, new possibilities. "
                "Get that coffee going and start the day right."
            )

    # Fallback generic nice statements
    if not statements:
        statements = [
            "Have a wonderful day out there, Los Angeles. "
            "Whatever you've got on the schedule, take a minute to look up and appreciate the sky — "
            "it's almost always worth it in this city.",

            "Another day in the City of Angels. Make it count. "
            "Whether you're hustling or resting, the city's got your back today. "
            "Take care of yourself out there.",

            "All systems looking good from here. The data looks clean, the city is humming along, "
            "and you've got a whole day ahead of you. Have a great one, LA.",

            "Los Angeles is doing its thing — and honestly, that's a beautiful thing. "
            "Enjoy the ride today, wherever it takes you. "
            "This city always has something to offer if you're paying attention.",

            "Stay safe and have a really nice day out there. "
            "LA is a big place with a lot going on, but right now, right here, everything's looking good. "
            "Go make the most of it.",
        ]

    return rng.choice(statements)
