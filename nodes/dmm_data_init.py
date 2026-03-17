"""
DMMDataInit — Central config for the Data Media Machine.
No API keys needed. All feeds are free and keyless.
Only stdlib imports at module scope.
"""

import time


class DMMDataInit:
    """Central config: coordinates, city, creative intensity, seed."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "configure"
    RETURN_TYPES = ("DMM_CONFIG",)
    RETURN_NAMES = ("config",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latitude": ("FLOAT", {
                    "default": 34.0622,
                    "min": -90.0,
                    "max": 90.0,
                    "step": 0.0001,
                    "tooltip": "Default: LA / Miracle Mile (34.0622)"
                }),
                "longitude": ("FLOAT", {
                    "default": -118.3437,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.0001,
                    "tooltip": "Default: LA / Miracle Mile (-118.3437)"
                }),
                "city_name": ("STRING", {
                    "default": "Los Angeles",
                    "multiline": False,
                }),
                "creative_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "1.0=natural, 2.0=exaggerated, 3.0=surreal"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "0 = use current timestamp"
                }),
                "global_visual_style": ([
                    "natural",
                    "noir_cinematic",
                    "documentary_calm",
                    "cyberpunk_hud",
                    "golden_hour_beauty",
                    "dramatic_broadcast",
                ], {"tooltip": "Visual style applied to ALL video clips. 'natural' = no style overlay."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Return current time so ComfyUI never caches this node.
        Every queue run re-executes DataInit, which invalidates the
        entire feed chain and forces fresh data + new prompts."""
        return time.time()

    def configure(self, latitude, longitude, city_name,
                   creative_intensity, seed, global_visual_style):
        config = {
            "lat": latitude,
            "lon": longitude,
            "city": city_name,
            "intensity": creative_intensity,
            "seed": seed if seed != 0 else int(time.time()) % 2**32,
            "timestamp": time.time(),
            "visual_style": global_visual_style,
        }
        return (config,)
