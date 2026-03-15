"""
ComfyUI Data-Driven Media Machine v3.2
========================================
Real-time weather, air quality & LA Metro transit data
â†’ generative video, audio, TTS narration, music params.

All data feeds are LIVE, FREE, and require ZERO API keys:
  - Open-Meteo (weather + air quality)
  - NWS weather.gov (alerts)
  - LA Metro (real-time bus positions)

Drop this folder into: ComfyUI\\custom_nodes\\comfyui-data-media-machine\\

Lessons applied from radio drama pipeline:
  - Safe per-node imports (one broken node won't kill the pack)
  - Namespaced node names (DMM_ prefix) to avoid collisions
  - All heavy imports inside execute() methods
  - Raw string literals for Windows paths
  - No asyncio event loop touching
  - Graceful fallback on missing deps

Author: Jeffrey A. Brick
Hardware target: Lenovo Legion Pro 7i Gen 10 (RTX 5080 / Win11)
"""

import os
import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- Safe per-node imports ---
# If one node has a bad import, the rest still load.
# This is the pattern we learned the hard way with Coqui on Python 3.12.

_NODE_MODULES = {
    "DMM_DataInit": (".nodes.dmm_data_init", "DMMDataInit"),
    "DMM_WeatherFetch": (".nodes.dmm_weather_fetch", "DMMWeatherFetch"),
    "DMM_AirQualityFetch": (".nodes.dmm_airquality_fetch", "DMMAirQualityFetch"),
    "DMM_MetroFetch": (".nodes.dmm_metro_fetch", "DMMMetroFetch"),
    "DMM_EarthquakeFetch": (".nodes.dmm_earthquake_fetch", "DMMEarthquakeFetch"),
    "DMM_AlertsFetch": (".nodes.dmm_alerts_fetch", "DMMAlertsFetch"),
    "DMM_LAPulseNarrative": (".nodes.dmm_la_pulse", "DMMLAPulseNarrative"),
    "DMM_DataToPrompt": (".nodes.dmm_data_to_prompt", "DMMDataToPrompt"),
    "DMM_DataToTTS": (".nodes.dmm_data_to_tts", "DMMDataToTTS"),
    "DMM_DataToMusic": (".nodes.dmm_data_to_music", "DMMDataToMusic"),
    "DMM_DataToVideo": (".nodes.dmm_data_to_video", "DMMDataToVideo"),
    "DMM_CinematicVideoPrompt": (".nodes.dmm_cinematic_video_prompt", "DMMCinematicVideoPrompt"),
    "DMM_VideoConcat": (".nodes.dmm_video_concat", "DMMVideoConcat"),
    "DMM_BatchVideoGenerator": (".nodes.dmm_batch_video", "DMMBatchVideoGenerator"),
    "DMM_BatchAutoPrompts":    (".nodes.dmm_batch_auto_prompts", "DMMBatchAutoPrompts"),
    "DMM_WorldCityFetch":      (".nodes.dmm_world_webcam", "DMMWorldCityFetch"),
    "DMM_WorldTourNarrator":   (".nodes.dmm_world_webcam", "DMMWorldTourNarrator"),
    # v3.0 webcam pipeline nodes
    "DMM_WebcamFetch":         (".nodes.dmm_webcam_fetch", "DMMWebcamFetch"),
    "DMM_CameraRouter":        (".nodes.dmm_camera_router", "DMMCameraRouter"),
    "DMM_CameraRegistry":      (".nodes.dmm_camera_router", "DMMCameraRegistry"),
    "DMM_FramePrep":           (".nodes.dmm_frame_prep", "DMMFramePrep"),
    "DMM_CinematicVideoPromptV2": (".nodes.dmm_cinematic_video_prompt", "DMMCinematicVideoPromptV2"),
}

# Display names with emoji for the ComfyUI menu
_DISPLAY_NAMES = {
    "DMM_DataInit":        "\U0001f310 DMM: Data Init",
    "DMM_WeatherFetch":    "\U0001f326\ufe0f DMM: Weather Fetch (Live)",
    "DMM_AirQualityFetch": "\U0001f32c\ufe0f DMM: Air Quality (Live)",
    "DMM_MetroFetch":      "\U0001f68d DMM: LA Metro (Live)",
    "DMM_EarthquakeFetch": "\U0001f30b DMM: Earthquakes (Live)",
    "DMM_AlertsFetch":     "\u26a0\ufe0f DMM: NWS Alerts (Live)",
    "DMM_LAPulseNarrative":"\U0001f4e1 DMM: LA Pulse Narrative",
    "DMM_DataToPrompt":    "\U0001f3a8 DMM: Data \u2192 Prompt",
    "DMM_DataToTTS":       "\U0001f399\ufe0f DMM: Data \u2192 TTS Script",
    "DMM_DataToMusic":     "\U0001f3b5 DMM: Data \u2192 Music Params",
    "DMM_DataToVideo":     "\U0001f3ac DMM: Data \u2192 Video Params",
    "DMM_CinematicVideoPrompt": "\U0001f3ac DMM: Cinematic Video Prompt",
    "DMM_VideoConcat": "\U0001f3ac DMM: Video Concat (Stitch)",
    "DMM_BatchVideoGenerator": "\U0001f3ac DMM: Batch Video Generator",
    "DMM_BatchAutoPrompts":    "\U0001f916 DMM: Batch Auto Prompts (Live)",
    # v3.0 webcam pipeline nodes
    "DMM_WebcamFetch":         "\U0001f4f7 DMM: Webcam Fetch (Live)",
    "DMM_CameraRouter":        "\U0001f3af DMM: Camera Router",
    "DMM_CameraRegistry":      "\U0001f4cb DMM: Camera Registry",
    "DMM_FramePrep":           "\U0001f5bc\ufe0f DMM: Frame Prep",
    "DMM_CinematicVideoPromptV2": "\U0001f3ac DMM: Cinematic Video Prompt v2",
}

for node_name, (module_path, class_name) in _NODE_MODULES.items():
    try:
        # Use importlib to import relative to this package
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)
        NODE_CLASS_MAPPINGS[node_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = _DISPLAY_NAMES.get(node_name, node_name)
    except Exception as e:
        print(f"[DMM] WARNING: Failed to load node '{node_name}' from {module_path}: {e}")
        traceback.print_exc()
        # Don't crash the whole pack â€” skip this node and continue

if NODE_CLASS_MAPPINGS:
    print(f"[DMM] Data Media Machine v3.2: loaded {len(NODE_CLASS_MAPPINGS)}/{len(_NODE_MODULES)} nodes")
else:
    print("[DMM] WARNING: No nodes loaded! Check dependencies.")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

