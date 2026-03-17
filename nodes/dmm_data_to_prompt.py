"""
DMMDataToPrompt — Maps all data streams into generative image/video prompts.

Consumes: weather, air quality, transit
Outputs: positive prompt, negative prompt, style metadata (JSON string)

No heavy imports needed — pure string logic.
"""

import random
import json


class DMMDataToPrompt:
    """Converts live data into rich creative prompts for image/video gen."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_prompt"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "style_tags",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "weather_data": ("DMM_WEATHER",),
                "style_mode": ([
                    "cinematic_realism",
                    "anime_atmospheric",
                    "oil_painting",
                    "retro_vhs",
                    "noir_photography",
                    "synthwave",
                    "lo_fi_illustration",
                    "documentary",
                    "surreal_dreamscape",
                    "abstract_data_viz",
                ],),
                "subject_focus": ([
                    "cityscape",
                    "landscape",
                    "portrait_mood",
                    "abstract",
                    "street_scene",
                    "aerial_view",
                    "interior_mood",
                ],),
            },
            "optional": {
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "custom_prefix": ("STRING", {"default": "", "multiline": True}),
                "custom_suffix": ("STRING", {"default": "", "multiline": True}),
            },
        }

    TEMP_PALETTES = {
        (-20, 20): ("ice blue, frost white, pale violet, crystalline", "frozen, stark, pristine"),
        (20, 40): ("slate grey, cold blue, muted teal", "brisk, austere, contemplative"),
        (40, 60): ("cool grey, sage green, soft lavender", "gentle, melancholic, peaceful"),
        (60, 75): ("warm amber, soft gold, leaf green, sky blue", "pleasant, balanced, inviting"),
        (75, 90): ("burnt orange, golden, coral, warm ochre", "vibrant, hazy, languid"),
        (90, 120): ("scorching red, bleached white, molten gold", "oppressive, relentless, searing"),
    }

    CONDITION_MOODS = {
        "Clear": ("brilliant light, sharp shadows, crystalline atmosphere", "luminous, expansive"),
        "Clouds": ("diffused light, layered grey skies, soft ambient glow", "contemplative, veiled"),
        "Rain": ("wet reflections, rain streaks, glistening surfaces", "melancholic, cleansing"),
        "Drizzle": ("misty droplets, soft blur, dewy surfaces", "delicate, quiet, intimate"),
        "Thunderstorm": ("dramatic lightning, dark churning clouds, electric atmosphere", "chaotic, powerful"),
        "Snow": ("falling snowflakes, white blanket, muted silence", "hushed, ethereal"),
        "Fog": ("dense obscurity, emerging silhouettes, zero horizon", "eerie, isolated, noir"),
    }

    STYLE_TEMPLATES = {
        "cinematic_realism": "photorealistic, cinematic composition, anamorphic lens, film grain, 35mm, shallow depth of field",
        "anime_atmospheric": "anime art style, Makoto Shinkai inspired, detailed backgrounds, volumetric lighting, cel shading",
        "oil_painting": "oil painting, thick impasto brushwork, classical composition, rich pigments, gallery quality",
        "retro_vhs": "VHS aesthetic, scan lines, chromatic aberration, retro 80s color grading, analog distortion",
        "noir_photography": "black and white, film noir, high contrast, dramatic shadows, venetian blind lighting",
        "synthwave": "neon colors, synthwave aesthetic, retrofuturistic, grid lines, chrome reflections",
        "lo_fi_illustration": "lo-fi illustration, soft pastel colors, simple shapes, cozy aesthetic, warm tones",
        "documentary": "documentary photography, natural light, candid moment, photojournalistic, raw, 50mm lens",
        "surreal_dreamscape": "surrealist painting, impossible architecture, melting reality, dreamlike, subconscious",
        "abstract_data_viz": "abstract generative art, data visualization, particle systems, flow fields, algorithmic",
    }

    def generate_prompt(self, config, weather_data, style_mode, subject_focus,
                         aq_data=None, transit_data=None,
                         custom_prefix="", custom_suffix=""):
        rng = random.Random(config["seed"])
        intensity = config["intensity"]

        temp = weather_data.get("temp_f", 72)
        condition = weather_data.get("condition", "Clear")
        wind = weather_data.get("wind_speed_mph", 5)
        humidity = weather_data.get("humidity", 50)
        visibility = weather_data.get("visibility_m", 10000)

        # Temperature palette
        palette_colors, palette_mood = "neutral tones", "ambient"
        for (lo, hi), (colors, mood) in self.TEMP_PALETTES.items():
            if lo <= temp < hi:
                palette_colors, palette_mood = colors, mood
                break

        # Condition atmosphere
        cond_vis, cond_mood = self.CONDITION_MOODS.get(
            condition, ("ambient atmospheric lighting", "undefined"))

        # Wind dynamics
        if wind < 5:
            wind_desc = "perfectly still air, no movement"
        elif wind < 15:
            wind_desc = "gentle breeze, slight motion in leaves and fabric"
        elif wind < 25:
            wind_desc = "moderate wind, visible motion, swaying trees"
        elif wind < 40:
            wind_desc = "strong wind, dramatic motion blur, bending trees"
        else:
            wind_desc = "violent gale, extreme motion, horizontal rain"

        # Visibility → depth
        if visibility < 500:
            depth = "shapes dissolving into obscurity, near-zero visibility"
        elif visibility < 2000:
            depth = "reduced visibility, atmospheric haze"
        elif visibility < 8000:
            depth = "moderate atmospheric depth, slight haze"
        else:
            depth = "crystal clear, infinite depth, sharp distant details"

        # Air quality influence
        aq_desc = ""
        if aq_data:
            aqi = aq_data.get("us_aqi", 50)
            uv = aq_data.get("uv_index", 3)
            aq_desc = aq_data.get("creative_desc", "")
            if uv > 8:
                aq_desc += ", harsh UV light, bleached highlights"
            elif uv < 2:
                aq_desc += ", soft diffused light, low UV"

        # Transit/urban energy
        transit_desc = ""
        if transit_data:
            congestion = transit_data.get("congestion_pct", 50)
            diversity = transit_data.get("heading_diversity", 0.5)
            if congestion > 70:
                transit_desc = "dense urban gridlock energy, frustrated stillness, brake lights"
            elif congestion > 40:
                transit_desc = "moderate urban pulse, stop-and-go rhythm"
            elif congestion > 15:
                transit_desc = "smooth urban movement, flowing city energy"
            else:
                transit_desc = "empty roads, solitary quiet, open urban space"
            if diversity > 0.7:
                transit_desc += ", chaotic multi-directional movement"
            elif diversity < 0.3:
                transit_desc += ", aligned linear flow"

        # Assemble
        style_base = self.STYLE_TEMPLATES.get(style_mode, "high quality")
        parts = [p for p in [
            custom_prefix,
            style_base,
            f"scene: {subject_focus}",
            f"color palette: {palette_colors}",
            f"atmosphere: {cond_vis}",
            f"mood: {palette_mood}, {cond_mood}",
            f"air: {wind_desc}",
            f"depth: {depth}",
            f"air quality: {aq_desc}" if aq_desc else "",
            f"urban energy: {transit_desc}" if transit_desc else "",
            custom_suffix,
        ] if p.strip()]

        if intensity > 1.5:
            flourishes = [
                "heightened emotion", "exaggerated atmosphere",
                "dramatic tension", "surreal edge", "time feels suspended",
            ]
            parts.extend(rng.sample(flourishes, min(3, int(intensity))))

        positive = ", ".join(parts)

        negative = (
            "low quality, blurry, deformed, watermark, text, logo, "
            "oversaturated, bad anatomy, extra limbs, cropped, worst quality"
        )

        tags = json.dumps({
            "style": style_mode,
            "subject": subject_focus,
            "condition": condition,
            "temp_f": temp,
            "intensity": intensity,
            "live_weather": weather_data.get("live", False),
            "live_transit": transit_data.get("live", False) if transit_data else False,
            "live_aq": aq_data.get("live", False) if aq_data else False,
        })

        return (positive, negative, tags)
