"""
DMMDataToVideo — Maps data streams to video generation parameters.
Pure string/math logic, no heavy imports.
"""

import random
import json


class DMMDataToVideo:
    """Converts environmental data into video generation parameters."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_video_params"
    RETURN_TYPES = ("FLOAT", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("motion_amount", "camera_motion", "video_fx_prompt", "video_metadata",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("DMM_CONFIG",),
                "weather_data": ("DMM_WEATHER",),
                "video_model": ([
                    "animatediff", "stable_video_diffusion",
                    "cogvideox", "wan_video", "ltx_video", "generic",
                ],),
            },
            "optional": {
                "aq_data": ("DMM_AIRQUALITY",),
                "transit_data": ("DMM_TRANSIT",),
                "duration_seconds": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 30.0, "step": 0.5,
                }),
            },
        }

    _CAMERA_OPTIONS = {
        "Clear": ["slow pan right", "gentle tilt up", "static wide shot", "slow dolly forward"],
        "Clouds": ["slow pan left", "static with subtle drift", "time-lapse sky"],
        "Rain": ["handheld slight shake", "slow tilt down", "static close-up"],
        "Drizzle": ["gentle drift", "slow zoom in", "rack focus"],
        "Thunderstorm": ["shaky handheld", "quick pan", "Dutch angle drift"],
        "Snow": ["slow descending tilt", "static meditation shot", "gentle orbit"],
        "Fog": ["creeping dolly forward", "static disappearing depth", "slow reveal"],
    }

    def generate_video_params(self, config, weather_data, video_model,
                               aq_data=None, transit_data=None,
                               duration_seconds=4.0):
        rng = random.Random(config["seed"])
        intensity = config["intensity"]

        wind = weather_data.get("wind_speed_mph", 5)
        condition = weather_data.get("condition", "Clear")
        rain_mm = weather_data.get("rain_1h_mm", 0)
        vis = weather_data.get("visibility_m", 10000)

        # Motion amount
        motion = min(1.0, wind / 35)
        if rain_mm > 10:
            motion += 0.3
        elif rain_mm > 0:
            motion += 0.15
        if condition == "Thunderstorm":
            motion += 0.3

        if transit_data:
            cong = transit_data.get("congestion_pct", 50)
            if cong > 70:
                motion *= 0.7
            elif cong < 20:
                motion *= 1.2

        motion = min(1.5, motion * intensity)

        # Camera motion
        cam_pool = self._CAMERA_OPTIONS.get(condition, ["static wide shot"])
        if wind > 25:
            cam_pool = cam_pool + ["dynamic tracking shot", "wind-following pan"]
        camera = rng.choice(cam_pool)

        # FX prompt
        fx = []
        if condition in ("Rain", "Drizzle"):
            fx.append("rain particles falling, wet reflections")
        if condition == "Thunderstorm":
            fx.append("lightning flash, rain sheets, dramatic sky")
        if condition == "Snow":
            fx.append("falling snowflakes, accumulation")
        if condition == "Fog":
            fx.append("volumetric fog, shapes emerging from mist")
        if wind > 20:
            fx.append("wind-blown elements, particle drift")
        if vis < 1000:
            fx.append("extreme atmospheric density")
        elif vis < 5000:
            fx.append("hazy atmosphere, reduced clarity")

        # AQ influence on visuals
        if aq_data:
            aqi = aq_data.get("us_aqi", 50)
            if aqi > 200:
                fx.append("toxic smog, apocalyptic orange-brown sky")
            elif aqi > 150:
                fx.append("thick haze layer, muted colors, brown sky tinge")
            elif aqi > 100:
                fx.append("visible atmospheric haze, slightly muted palette")

        # Transit urban FX
        if transit_data:
            cong = transit_data.get("congestion_pct", 50)
            if cong > 70:
                fx.append("dense traffic, brake lights, urban gridlock")
            elif cong < 20:
                fx.append("smooth flowing traffic, light trails")

        fx_prompt = ", ".join(fx) if fx else "natural atmospheric motion"

        # Model configs (VRAM-conscious for RTX 5080 16GB)
        model_cfgs = {
            "animatediff": {
                "motion_scale": round(motion, 2),
                "frames": int(duration_seconds * 8),
                "context_length": 16,
                "note": "Use --lowvram if OOM at high motion",
            },
            "stable_video_diffusion": {
                "motion_bucket_id": max(1, min(255, int(motion * 180))),
                "fps": 7,
                "frames": int(duration_seconds * 7),
            },
            "cogvideox": {
                "guidance_scale": round(6.0 + motion * 2, 1),
                "num_frames": max(9, min(49, int(duration_seconds * 8))),
            },
            "wan_video": {
                "motion_strength": round(motion, 2),
                "num_frames": max(16, min(81, int(duration_seconds * 16))),
                "guidance_scale": round(5.0 + motion * 2.5, 1),
                "note": "Use --lowvram and NVFP8 for 720p on 16GB",
            },
            "ltx_video": {
                "motion_strength": round(motion, 2),
                "num_frames": max(9, min(97, int(duration_seconds * 24))),
                "note": "LTX-2 runs well with NVFP8 on RTX 5080",
            },
            "generic": {
                "motion_amount": round(motion, 2),
                "duration_sec": duration_seconds,
            },
        }

        metadata = json.dumps({
            "model": video_model,
            "camera": camera,
            "condition": condition,
            "params": model_cfgs.get(video_model, {}),
        }, indent=2)

        return (round(motion, 3), camera, fx_prompt, metadata)
