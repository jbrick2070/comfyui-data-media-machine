"""
DMMCameraRouter Ã¢â‚¬â€ Maps data focus types to webcam URLs from a JSON registry.

v3.1 changes:
  - DMMCameraRegistry: added IS_CHANGED using file mtime (auto-reloads on edit)
  - DMMCameraRouter: added camera count to log output
  - Improved _default_registry to include all 5 focus types

v3.2 changes:
  - DMMCameraRouter: added max_radius_miles parameter (default 50.0)
    Filters cameras by distance_miles before selection. Dial to 5 for local,
    50 for all of LA, 200 for no filter.
  - Added "scenic" focus type (ipcamlive cameras)
  - Registry expanded to 299 Caltrans + 3 ipcamlive = 302 cameras

Supports multiple cameras per category with random, round-robin, or fixed
selection.  Returns the selected URL plus fallback URLs for retry logic.

DMMCameraRegistry Ã¢â‚¬â€ Loads camera_registry.json and outputs DMM_CAMERAS.

Author: Jeffrey A. Brick
"""

import json
import logging
import os
import random
import threading
import time
from hashlib import sha256

log = logging.getLogger("DMM.CameraRouter")

# Round-robin state: focus -> index
_rr_state = {}
_rr_lock = threading.Lock()


class DMMCameraRegistry:
    """Loads camera_registry.json and outputs the registry as DMM_CAMERAS."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "load_registry"
    RETURN_TYPES = ("DMM_CAMERAS",)
    RETURN_NAMES = ("camera_registry",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "registry_path": ("STRING", {
                    "default": "camera_registry.json",
                    "tooltip": "Path to camera_registry.json (relative to ComfyUI root or absolute)"
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, registry_path="camera_registry.json"):
        """Re-run if the registry file has been modified (e.g. by healthcheck)."""
        import pathlib
        for candidate in [
            pathlib.Path(registry_path),
            pathlib.Path(__file__).parent.parent / registry_path,
        ]:
            if candidate.exists():
                return str(candidate.stat().st_mtime)
        return float("nan")  # file not found, always re-run

    def load_registry(self, registry_path):
        # Search order: 1) absolute path, 2) relative to this node's package,
        # 3) relative to ComfyUI root.  Uses __file__ for reliable resolution
        # regardless of how ComfyUI was launched.
        if os.path.isabs(registry_path):
            full_path = registry_path
        else:
            # Try relative to the media_machine custom node package first
            node_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(node_dir, registry_path)
            if os.path.exists(candidate):
                full_path = candidate
            else:
                # Fall back to ComfyUI root (4 levels up from this file)
                comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))))
                full_path = os.path.join(comfy_root, registry_path)

        if not os.path.exists(full_path):
            log.error("Camera registry not found: %s", full_path)
            return (self._default_registry(),)

        try:
            with open(full_path, "r", encoding="utf-8-sig") as f:
                registry = json.load(f)
            log.info("Loaded camera registry from %s (%d categories)",
                     full_path, len(registry))
            return (registry,)
        except Exception as e:
            log.error("Failed to load camera registry: %s", e)
            return (self._default_registry(),)

    def _default_registry(self):
        """Minimal fallback registry if the JSON file is missing."""
        return {
            "weather": [],
            "earthquake": [],
            "air_quality": [],
            "transit": [],
            "alerts": [],
            "scenic": [],
        }


class DMMCameraRouter:
    """Maps a data focus type to a webcam URL from the camera registry."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "route_camera"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("camera_urls", "camera_label",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "focus": (["weather", "earthquake", "air_quality", "transit", "alerts", "scenic"],),
                "camera_registry": ("DMM_CAMERAS",),
            },
            "optional": {
                "selection_mode": (["random", "round_robin", "fixed"],
                                   {"default": "random"}),
                "seed": ("INT", {"default": 42}),
                "max_radius_miles": ("FLOAT", {
                    "default": 50.0,
                    "min": 0.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Only consider cameras within this distance from center. 50=all of LA, 5=tight local."
                }),
                "scenic_swap_count": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Exactly how many of the 5 category slots swap to scenic per run. Each swapped slot picks a different scenic cam."
                }),
            },
        }

    def route_camera(self, focus, camera_registry, selection_mode="random",
                     seed=42, max_radius_miles=50.0, scenic_swap_count=2):
        # Scenic rotation: exactly N of the 5 standard slots swap to scenic.
        # All nodes independently compute the SAME swap set so coordination
        # is deterministic without inter-node communication.
        original_focus = focus
        _STANDARD_FOCUSES = ["weather", "earthquake", "air_quality", "transit", "alerts"]
        if scenic_swap_count > 0 and focus in _STANDARD_FOCUSES:
            scenic_cams = camera_registry.get("scenic", [])
            if scenic_cams:
                # Every node computes the same shuffled order using the same seed
                coord_rng = random.Random(seed)
                shuffled = list(_STANDARD_FOCUSES)
                coord_rng.shuffle(shuffled)
                # First N in shuffled order are the swap slots
                swap_set = set(shuffled[:scenic_swap_count])
                if focus in swap_set:
                    log.info("Scenic swap: '%s' slot replaced with scenic cam "
                             "(%d of %d slots)", focus, scenic_swap_count,
                             len(_STANDARD_FOCUSES))
                    focus = "scenic"

        cameras = camera_registry.get(focus, [])

        # v3.5: Skip disabled cameras (set by healthcheck or manual curation)
        cameras = [c for c in cameras if not c.get("disabled", False)]

        # Filter by radius if cameras have distance_miles metadata
        if max_radius_miles > 0:
            filtered = [c for c in cameras
                        if c.get("distance_miles", 0) <= max_radius_miles]
            if filtered:
                cameras = filtered
                log.info("Radius filter: %d/%d cameras within %.1f mi for '%s'",
                         len(cameras), len(camera_registry.get(focus, [])),
                         max_radius_miles, focus)
            else:
                log.warning("No cameras within %.1f mi for '%s', using full pool (%d)",
                            max_radius_miles, focus, len(cameras))

        if not cameras:
            log.warning("No cameras registered for focus '%s'", focus)
            return ("", f"no camera ({focus})")

        if selection_mode == "fixed":
            idx = 0
        elif selection_mode == "round_robin":
            with _rr_lock:
                idx = _rr_state.get(focus, 0)
                _rr_state[focus] = (idx + 1) % len(cameras)
        else:  # random
            # Use original_focus for offset so swapped scenic slots pick different cams
            offset = int.from_bytes(sha256(original_focus.encode()).digest()[:4], "little")
            rng = random.Random(seed + offset)
            idx = rng.randint(0, len(cameras) - 1)

        selected = cameras[idx]
        url = selected.get("url", "")
        label = selected.get("label", f"camera_{idx}")

        # v3.5: Cache-buster for YouTube thumbnail URLs.
        # Google CDNs aggressively cache maxresdefault.jpg; appending a timestamp
        # query param forces edge nodes to serve a fresh copy.
        def _cache_bust(u):
            if "img.youtube.com" in u:
                sep = "&" if "?" in u else "?"
                return f"{u}{sep}t={int(time.time())}"
            return u

        # Build pipe-delimited URL string: selected first, then fallbacks
        # This feeds directly into DMMWebcamFetch.urls which tries each in order
        all_urls = [_cache_bust(url)] + [_cache_bust(c.get("url", ""))
                    for i, c in enumerate(cameras)
                    if i != idx and c.get("url")]
        urls_str = "|".join(u for u in all_urls if u)

        log.info("Selected camera for '%s': %s [%d/%d] (%s) + %d fallbacks (%s mode)",
                 focus, label, idx + 1, len(cameras), url[:60],
                 len(all_urls) - 1, selection_mode)
        return (urls_str, label)
