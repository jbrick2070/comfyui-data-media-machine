"""
ComfyUI Data-Driven Media Machine
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
  - Graceful asyncio exception handler (suppresses Windows socket teardown noise)
  - Graceful fallback on missing deps

Author: Jeffrey A. Brick
Hardware target: Lenovo Legion Pro 7i Gen 10 (RTX 5080 / Win11)
"""

DMM_VERSION = "3.6"

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
    # v3.2 TTS narration
    "DMM_NarrationDistiller":  (".nodes.dmm_narration_distiller", "DMMNarrationDistiller"),
    "DMM_AudioMux":            (".nodes.dmm_audio_mux", "DMMAudioMux"),
    # v3.3 audio enhancement / v3.4 crossfade + dark-cache + low-res gen
    "DMM_AudioEnhance":        (".nodes.dmm_audio_enhance", "DMMAudioEnhance"),
    # v3.4 procedural motion graphics
    "DMM_ProceduralClip":      (".nodes.dmm_procedural_clip", "DMMProceduralClip"),
    # v3.4.1 background music (MIDI synth)
    "DMM_BackgroundMusic":     (".nodes.dmm_background_music", "DMMBackgroundMusic"),
    # v3.5 energy grid (CAISO)
    "DMM_EnergyGrid":          (".nodes.dmm_energy_grid", "DMMEnergyGrid"),
    # v3.6 narration AI refiner (Phi-3-mini)
    "DMM_NarrationRefiner":    (".nodes.dmm_narration_refiner", "DMMNarrationRefiner"),
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
    # v3.2 TTS narration
    "DMM_NarrationDistiller":  "\U0001f399\ufe0f DMM: Narration Distiller",
    "DMM_AudioMux":            "\U0001f50a DMM: Audio Mux",
    # v3.3 audio enhancement
    "DMM_AudioEnhance":        "\U0001f3a7 DMM: Audio Enhance (Spatial 48k)",
    # v3.4 procedural motion graphics
    "DMM_ProceduralClip":      "\U0001f3ac DMM: Procedural Clip (LA Style)",
    # v3.4.1 background music
    "DMM_BackgroundMusic":     "\U0001f3b5 DMM: Background Music (MIDI)",
    # v3.5 energy grid
    "DMM_EnergyGrid":          "\u26a1 DMM: Energy Grid (CAISO Live)",
    # v3.6 narration AI refiner
    "DMM_NarrationRefiner":    "\U0001f4dd DMM: Narration Refiner (Phi-3)",
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
    print(f"[DMM] Data Media Machine v{DMM_VERSION}: loaded {len(NODE_CLASS_MAPPINGS)}/{len(_NODE_MODULES)} nodes")
else:
    print("[DMM] WARNING: No nodes loaded! Check dependencies.")


# ---------------------------------------------------------------------------
# Graceful WebSocket disconnect handler (Windows asyncio ProactorEventLoop)
# ---------------------------------------------------------------------------
# On Windows, when the browser/client closes a WebSocket after prompt
# execution, Python's ProactorEventLoop tries to shutdown() an already-closed
# socket, producing ugly ConnectionResetError tracebacks.  This is harmless
# but noisy.  We install a custom exception handler that logs a clean goodbye
# instead of dumping a full traceback.
# ---------------------------------------------------------------------------
def _dmm_asyncio_exception_handler(loop, context):
    """Catch Windows socket teardown noise and log a clean message."""
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        print(f"[DMM] Client disconnected — session complete (v{DMM_VERSION})")
        return  # swallow the ugly traceback
    # For anything else, fall through to the default handler
    loop.default_exception_handler(context)


def _install_exception_handler():
    """Install our handler on the running asyncio loop (best-effort)."""
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop and loop.is_running():
            loop.set_exception_handler(_dmm_asyncio_exception_handler)
            print(f"[DMM] Graceful disconnect handler installed")
        else:
            # Loop not running yet at import time — schedule it
            import threading
            def _deferred_install():
                import time as _time
                _time.sleep(2)  # wait for ComfyUI server to start its loop
                try:
                    loop = asyncio.get_event_loop()
                    loop.set_exception_handler(_dmm_asyncio_exception_handler)
                    print(f"[DMM] Graceful disconnect handler installed (deferred)")
                except Exception:
                    pass  # best-effort, don't crash
            t = threading.Thread(target=_deferred_install, daemon=True)
            t.start()
    except Exception as e:
        print(f"[DMM] Could not install disconnect handler: {e}")


_install_exception_handler()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

