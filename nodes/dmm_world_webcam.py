"""
DMM World Webcam Tour
======================
Fetches live satellite imagery from NASA GIBS (free, no API key) for
10 iconic world cities + generates cinematic TTS narration scripts.

Nodes:
  DMM_WorldCityFetch    - NASA satellite image + live weather per city
  DMM_WorldTourNarrator - cinematic TTS script generator

Author: Jeffrey A. Brick (extended DMM node)
"""

import time, datetime, io, os, json

WORLD_CITIES = [
    {"name":"Tokyo",         "lat":35.6762, "lon":139.6503, "tz_offset":9,
     "voice":"en-GB-RyanNeural",
     "flavor":"neon-lit megalopolis of 14 million souls moving in perfect choreography",
     "landmark":"Shibuya, Tokyo Tower, the Yamanote loop","bbox_pad":0.4},
    {"name":"Paris",         "lat":48.8566, "lon":2.3522,   "tz_offset":1,
     "voice":"en-GB-SoniaNeural",
     "flavor":"city of light where philosophy is ordered alongside coffee",
     "landmark":"Eiffel Tower, the Seine, Haussmann boulevards","bbox_pad":0.25},
    {"name":"New York",      "lat":40.7128, "lon":-74.0060, "tz_offset":-5,
     "voice":"en-US-GuyNeural",
     "flavor":"eight million stories compressed into one screaming island",
     "landmark":"Manhattan skyline, Central Park, the bridges","bbox_pad":0.3},
    {"name":"Sydney",        "lat":-33.8688,"lon":151.2093, "tz_offset":11,
     "voice":"en-AU-WilliamNeural",
     "flavor":"where the Pacific wraps itself around the oldest continent on Earth",
     "landmark":"Opera House, Harbour Bridge, the Heads","bbox_pad":0.3},
    {"name":"Cairo",         "lat":30.0444, "lon":31.2357,  "tz_offset":2,
     "voice":"en-US-AriaNeural",
     "flavor":"five thousand years of civilisation compressed into one living city",
     "landmark":"Giza plateau visible from orbit, the Nile delta","bbox_pad":0.4},
    {"name":"Rio de Janeiro","lat":-22.9068,"lon":-43.1729, "tz_offset":-3,
     "voice":"en-US-JennyNeural",
     "flavor":"cidade maravilhosa where jungle presses down to meet the ocean",
     "landmark":"Christ the Redeemer, Copacabana curve, Guanabara Bay","bbox_pad":0.35},
    {"name":"Mumbai",        "lat":19.0760, "lon":72.8777,  "tz_offset":5,
     "voice":"en-IN-NeerjaNeural",
     "flavor":"twenty million dreams compressed into a peninsula that refuses to sink",
     "landmark":"Gateway of India, Marine Drive arc, Dharavi","bbox_pad":0.3},
    {"name":"London",        "lat":51.5074, "lon":-0.1278,  "tz_offset":0,
     "voice":"en-GB-RyanNeural",
     "flavor":"two thousand years of empire reduced to a very polite queue",
     "landmark":"Thames bend, the City, Docklands, Heathrow approach","bbox_pad":0.35},
    {"name":"Los Angeles",   "lat":34.0522, "lon":-118.2437,"tz_offset":-8,
     "voice":"en-US-GuyNeural",
     "flavor":"the factory that manufactures dreams and ships them everywhere else",
     "landmark":"Hollywood Hills, Miracle Mile, LAX contrails","bbox_pad":0.4},
    {"name":"Nairobi",       "lat":-1.2921, "lon":36.8219,  "tz_offset":3,
     "voice":"en-US-AriaNeural",
     "flavor":"city in the sun where savanna presses against glass towers",
     "landmark":"Nairobi National Park border, city grid, Rift Valley edge","bbox_pad":0.35},
]
CITY_NAMES = [c["name"] for c in WORLD_CITIES]

def _nasa_url(lat, lon, pad=0.4, w=512, h=512):
    d = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    return (
        "https://wvs.earthdata.nasa.gov/api/v1/snapshot"
        "?REQUEST=GetSnapshot"
        f"&TIME={d}T00:00:00Z"
        f"&BBOX={lon-pad},{lat-pad},{lon+pad},{lat+pad}"
        "&CRS=EPSG:4326"
        "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor"
        "&FORMAT=image/jpeg"
        f"&WIDTH={w}&HEIGHT={h}&ts={int(time.time())}"
    )

def _fetch_img(url, w, h):
    import requests
    from PIL import Image
    import numpy as np, torch
    r = requests.get(url, timeout=20, headers={"User-Agent":"DMM/2"})
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB").resize((w,h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)/255.0
    return torch.from_numpy(arr).unsqueeze(0)

def _weather(lat, lon):
    import requests
    try:
        u = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
             "&current=temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m"
             "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto")
        d = requests.get(u, timeout=10).json().get("current",{})
        t=d.get("temperature_2m","?"); w=d.get("wind_speed_10m","?")
        h=d.get("relative_humidity_2m","?"); c=d.get("weather_code",0)
        desc = "clear" if c==0 else "partly cloudy" if c<3 else "overcast" if c<50 else "rain" if c<70 else "snow" if c<80 else "storms"
        return f"{t}F, {desc}, wind {w}mph, humidity {h}%"
    except Exception as e:
        return f"weather unavailable"

def _localtime(tz_off):
    local = datetime.datetime.utcnow() + datetime.timedelta(hours=tz_off)
    h = local.hour
    period = "deep night" if h<6 else "early morning" if h<9 else "morning" if h<12 else "midday" if h<14 else "afternoon" if h<17 else "early evening" if h<20 else "night"
    return local.strftime(f"%I:%M %p ({period})")


class DMMWorldCityFetch:
    CATEGORY="DataMediaMachine"; FUNCTION="fetch"
    RETURN_TYPES=("IMAGE","STRING","STRING","STRING","FLOAT","FLOAT")
    RETURN_NAMES=("satellite_image","city_name","weather_summary","local_time","latitude","longitude")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "city":(CITY_NAMES,{"default":"Tokyo"}),
            "image_width":("INT",{"default":512,"min":128,"max":1024,"step":64}),
            "image_height":("INT",{"default":512,"min":128,"max":1024,"step":64}),
            "fetch_weather":(["yes","no"],{"default":"yes"}),
        }}

    @classmethod
    def IS_CHANGED(cls,**kw): return time.time()

    def fetch(self, city, image_width, image_height, fetch_weather):
        info = next(c for c in WORLD_CITIES if c["name"]==city)
        lat,lon,pad = info["lat"],info["lon"],info.get("bbox_pad",0.35)
        url = _nasa_url(lat,lon,pad,image_width,image_height)
        print(f"[WorldCityFetch] {city} -> {url[:80]}...")
        try:
            img = _fetch_img(url, image_width, image_height)
            print(f"[WorldCityFetch] OK {image_width}x{image_height}")
        except Exception as e:
            import torch,numpy as np
            print(f"[WorldCityFetch] FAIL: {e} - placeholder")
            img = torch.from_numpy(np.random.rand(image_height,image_width,3).astype(np.float32)*0.3).unsqueeze(0)
        wx = _weather(lat,lon) if fetch_weather=="yes" else "weather fetch disabled"
        lt = _localtime(info["tz_offset"])
        print(f"[WorldCityFetch] {city}: {lt} | {wx}")
        return (img, city, wx, lt, float(lat), float(lon))


STYLES = {
    "documentary":
        "NARRATOR: {city}. {time}. From orbit, the city reveals itself without disguise. "
        "{flavor}. Current conditions: {weather}. Visible below: {landmark}.",
    "poetic":
        "NARRATOR: We are above {city} now. It is {time} there, and the city breathes. "
        "{flavor}. Below us, {weather}. The landmarks — {landmark} — hold their positions "
        "like old arguments nobody has yet won.",
    "absurdist":
        "NARRATOR: {city} continues to exist, which is frankly impressive. "
        "Conditions: {weather}. The residents — {flavor} — have not noticed us watching from space. "
        "Nor have they noticed {landmark}. They are very busy.",
    "noir":
        "NARRATOR: {city}. {time}. A city that does not sleep "
        "because it never quite figured out how. {weather}. "
        "Somewhere below, {landmark}. {flavor}. Nobody asked me. But I would have built it differently.",
    "nature_doc":
        "NARRATOR: From this altitude, {city} appears almost peaceful. Weather: {weather}. "
        "Here — {landmark} — the human animal has constructed its hive. {flavor}. The hive persists. For now.",
}

class DMMWorldTourNarrator:
    CATEGORY="DataMediaMachine"; FUNCTION="narrate"
    RETURN_TYPES=("STRING","STRING","STRING")
    RETURN_NAMES=("narration_text","voice_id","segment_title")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "city_name":("STRING",{"default":"Tokyo"}),
            "weather_summary":("STRING",{"default":"clear skies"}),
            "local_time":("STRING",{"default":"12:00 PM (midday)"}),
            "style":(list(STYLES.keys()),{"default":"documentary"}),
            "add_global_intro":(["yes","no"],{"default":"no"}),
            "add_global_outro":(["yes","no"],{"default":"no"}),
        }}

    def narrate(self, city_name, weather_summary, local_time, style,
                add_global_intro, add_global_outro):
        info = next((c for c in WORLD_CITIES if c["name"]==city_name),
                    {"flavor":"a city of extraordinary complexity",
                     "landmark":"its streets and skyline","voice":"en-US-GuyNeural"})
        text = STYLES[style].format(
            city=city_name, time=local_time,
            flavor=info["flavor"], weather=weather_summary, landmark=info["landmark"])
        if add_global_intro=="yes":
            text = ("NARRATOR: This is Earth. It has approximately eight billion inhabitants "
                    "and two hundred and forty-seven thousand cities. We will visit ten of them.\n\n") + text
        if add_global_outro=="yes":
            text += "\n\nNARRATOR: The satellite moves on. Below, the city continues. As cities do."
        voice = info.get("voice","en-US-GuyNeural")
        print(f"[WorldTourNarrator] {city_name} | style={style} | {text[:80]}...")
        return (text, voice, f"World Tour: {city_name}")
