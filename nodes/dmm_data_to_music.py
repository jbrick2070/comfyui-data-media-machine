"""
DMMDataToMusic — Maps data streams to musical parameters.
Pure string/math logic, no heavy imports.
"""

import random
import json


class DMMDataToMusic:
    """Converts environmental data into musical parameters."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_music_params"
    RETURN_TYPES = ("STRING", "INT", "STRING", "FLOAT",)
    RETURN_NAMES = ("music_prompt", "bpm", "key_scale", "energy_level",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "weather_data": ("DMM_WEATHER",),
                "genre_preference": ([
                    "ambient", "lo_fi_hiphop", "techno", "jazz",
                    "classical", "synthwave", "drone", "experimental",
                    "trip_hop", "cinematic_score", "auto_from_data",
                ],),
            },
            "optional": {
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
            },
        }

    _TEMP_KEYS = [
        (-20, 20,  "Dm",  "D minor"),
        (20, 40,   "Am",  "A minor"),
        (40, 55,   "Em",  "E minor"),
        (55, 65,   "C",   "C major"),
        (65, 75,   "G",   "G major"),
        (75, 85,   "D",   "D major"),
        (85, 100,  "A",   "A major"),
        (100, 130, "F#m", "F# minor"),
    ]

    _CONDITION_GENRES = {
        "Clear": "synthwave", "Clouds": "ambient", "Rain": "lo_fi_hiphop",
        "Drizzle": "trip_hop", "Thunderstorm": "techno", "Snow": "classical",
        "Fog": "drone",
    }

    _GENRE_DESCS = {
        "ambient": "ambient pad textures, reverb washes, slow evolving drones",
        "lo_fi_hiphop": "lo-fi hip hop beat, vinyl crackle, mellow keys, chill",
        "techno": "driving techno, four on the floor, dark bass, industrial",
        "jazz": "smooth jazz, walking bass, brushed drums, blue notes",
        "classical": "orchestral strings, piano, legato phrasing",
        "synthwave": "analog synths, retro arpeggios, pulsing bass",
        "drone": "deep drone, sustained tones, overtone harmonics, meditative",
        "experimental": "glitch textures, found sounds, deconstructed",
        "trip_hop": "slow breakbeat, moody bass, downtempo, atmospheric",
        "cinematic_score": "cinematic film score, emotional swells, orchestral + electronic",
    }

    def generate_music_params(self, config, weather_data, genre_preference,
                               aq_data=None, transit_data=None):
        intensity = config["intensity"]

        temp = weather_data.get("temp_f", 72)
        wind = weather_data.get("wind_speed_mph", 5)
        condition = weather_data.get("condition", "Clear")
        rain_mm = weather_data.get("rain_1h_mm", 0)

        # BPM from wind + transit
        wind_bpm = int(60 + (wind / 50) * 80)
        if transit_data:
            congestion = transit_data.get("congestion_pct", 50)
            wind_bpm += int((50 - congestion) * 0.3)
        if rain_mm > 5:
            wind_bpm += 10
        bpm = max(50, min(180, int(wind_bpm * intensity)))

        # Key from temperature
        key_short, key_name = "C", "C major"
        for lo, hi, ks, kn in self._TEMP_KEYS:
            if lo <= temp < hi:
                key_short, key_name = ks, kn
                break

        # Energy composite
        wind_e = min(1.0, wind / 40)
        rain_e = min(1.0, rain_mm / 15)
        transit_e = (transit_data.get("congestion_pct", 50) / 100) if transit_data else 0.5
        aq_e = min(1.0, (aq_data.get("us_aqi", 50) / 200)) if aq_data else 0.3
        energy = min(1.0, (wind_e * 0.3 + rain_e * 0.15 + transit_e * 0.3 + aq_e * 0.25) * intensity)

        # Genre
        if genre_preference == "auto_from_data":
            genre = self._CONDITION_GENRES.get(condition, "ambient")
        else:
            genre = genre_preference

        base_desc = self._GENRE_DESCS.get(genre, f"{genre} music")

        # Weather textures
        tex = []
        if rain_mm > 0:
            tex.append("rain sample layered in")
        if condition == "Thunderstorm":
            tex.append("thunder rumble as percussion")
        if wind > 20:
            tex.append("wind noise texture")
        if condition in ("Fog", "Mist"):
            tex.append("foghorn sample, muted reverb")
        if aq_data and aq_data.get("us_aqi", 50) > 150:
            tex.append("industrial haze drone, distorted atmosphere")

        tex_str = ", ".join(tex) if tex else "clean atmosphere"

        prompt = f"{base_desc}, {key_name}, {bpm} BPM, energy {energy:.1f}/1.0, {tex_str}"

        return (prompt, bpm, f"{key_short} ({key_name})", round(energy, 2))
