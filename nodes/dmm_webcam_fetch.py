"""
DMMWebcamFetch — Fetches a live JPEG frame from a public webcam URL.

v3.2 changes:
  - _fallback() now tries a local fallback_images/ dir before black frame
  - Added procedural highway gradient fallback (better LTX-2 conditioning)
  - fallback_images_dir widget added to INPUT_TYPES
  - Fallback image selected by day-of-week seeded random for variety

v3.1 changes:
  - Fixed timeout_sec default mismatch (INPUT_TYPES 5 vs function 10 → now 5)
  - Defensive copy on cache write (prevents cache poisoning if caller retains ref)
  - enumerate() in fetch loop logging (was O(n) url_list.index())
  - Cache hit/miss stats logging

Handles HTTP timeouts, retries across a full fallback URL list,
SSIM-based stale detection (with coordinate cropping to ignore timestamps),
and cache with TTL + max-size limits.  Returns defensive tensor copies
to prevent downstream in-place ops from poisoning the cache.

Author: Jeffrey A. Brick
"""

import io
import logging
import threading
import time
from hashlib import sha256

import numpy as np
import torch

log = logging.getLogger("DMM.WebcamFetch")

# Module-level frame cache: url -> (timestamp, numpy_array)
_frame_cache = {}
_CACHE_MAX_SIZE = 50  # max entries before eviction
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

# v3.4: Dark-camera cache — URLs that returned placeholders recently.
# Avoids re-fetching cameras known to be dark/offline within the TTL window.
# Format: url -> timestamp_when_marked_dark
_dark_camera_cache = {}
_DARK_CAMERA_TTL = 3600  # seconds — skip dark cameras for 1 hour
_dark_cache_lock = threading.Lock()

# v3.5: Domain rate limiter — prevents hammering any single host.
# Tracks last fetch time per domain; enforces minimum delay between requests
# to the same netloc.  Protects against 429s from Caltrans, EarthCam,
# YouTube CDN, ipcamlive, HPWREN, etc.
_domain_last_fetch = {}
_domain_lock = threading.Lock()
_DOMAIN_MIN_DELAY = 2.0  # seconds between requests to the same domain


def _domain_throttle(url):
    """Sleep if needed to respect per-domain rate limit."""
    import random
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    if not domain:
        return
    with _domain_lock:
        now = time.time()
        last = _domain_last_fetch.get(domain, 0)
        elapsed = now - last
        if elapsed < _DOMAIN_MIN_DELAY:
            wait = _DOMAIN_MIN_DELAY - elapsed + random.uniform(0.1, 0.4)
            time.sleep(wait)
            log.debug("Domain throttle: waited %.2fs for %s", wait, domain)
        _domain_last_fetch[domain] = time.time()


class DMMWebcamFetch:
    """Fetches a live JPEG frame from a public webcam, with full fallback support."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "fetch_frame"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "FLOAT",)
    RETURN_NAMES = ("image", "success", "ssim_score",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {
                    "default": "",
                    "tooltip": "Primary URL, or pipe-delimited fallback list: url1|url2|url3"
                }),
            },
            "optional": {
                "timeout_sec": ("INT", {"default": 5, "min": 1, "max": 30,
                    "tooltip": "Per-URL timeout. Keep low to avoid freezing ComfyUI UI."}),
                "retry_count": ("INT", {"default": 1, "min": 0, "max": 3,
                    "tooltip": "Retries per URL before trying next fallback"}),
                "cache_sec": ("INT", {"default": 30, "min": 0, "max": 300,
                    "tooltip": "Seconds to cache a frame before re-fetching"}),
                "ssim_threshold": ("FLOAT", {"default": 0.97, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "SSIM above this = frame flagged as stale, success=False"}),
                "crop_pct": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": "Crop top/bottom % of frame before SSIM calc (ignore timestamps)"}),
                "fallback_width": ("INT", {"default": 768, "min": 64, "max": 2048}),
                "fallback_height": ("INT", {"default": 432, "min": 64, "max": 2048}),
                "fallback_images_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder of local .jpg/.png images to use when all URLs fail. "
                               "Leave blank to use procedural highway fallback."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def fetch_frame(self, urls, timeout_sec=5, retry_count=1,
                    cache_sec=30, ssim_threshold=0.97, crop_pct=0.10,
                    fallback_width=768, fallback_height=432,
                    fallback_images_dir=""):
        import requests
        from PIL import Image

        url_list = [u.strip() for u in urls.split("|") if u.strip()]
        if not url_list:
            log.warning("No webcam URLs provided, returning fallback frame")
            return self._fallback(fallback_width, fallback_height, fallback_images_dir)

        now = time.time()

        # Check cache for primary URL (defensive copy)
        global _cache_hits, _cache_misses
        primary = url_list[0]
        with _cache_lock:
            if primary in _frame_cache:
                cached_time, cached_arr = _frame_cache[primary]
                if now - cached_time < cache_sec:
                    _cache_hits += 1
                    log.info("Cache HIT for %s (%.0fs old, hits=%d misses=%d)",
                             primary[:60], now - cached_time, _cache_hits, _cache_misses)
                    tensor = torch.from_numpy(cached_arr.copy()).unsqueeze(0)
                    return (tensor, True, -1.0)  # -1.0 sentinel = cache hit
            _cache_misses += 1

        # v3.4: Purge expired dark-camera entries
        with _dark_cache_lock:
            expired = [u for u, t in _dark_camera_cache.items() if now - t > _DARK_CAMERA_TTL]
            for u in expired:
                del _dark_camera_cache[u]
            dark_skipped = 0

        # Try each URL in fallback chain
        frame_arr = None
        fetched_url = None
        for url_idx, url in enumerate(url_list):
            # v3.4: Skip known-dark cameras
            with _dark_cache_lock:
                if url in _dark_camera_cache:
                    dark_skipped += 1
                    continue

            for attempt in range(retry_count + 1):
                try:
                    # v3.5: Per-domain rate limiting
                    _domain_throttle(url)
                    log.info("Fetching webcam [%d/%d] attempt %d/%d: %s",
                             url_idx + 1, len(url_list), attempt + 1, retry_count + 1, url[:80])
                    resp = requests.get(url, timeout=timeout_sec, headers={
                        "User-Agent": "DMM-DataMediaMachine/3.0"
                    })
                    resp.raise_for_status()

                    # v3.5: MIME type validation — reject non-image responses.
                    # Some firewalls return 200 OK with an HTML error page that
                    # passes the filesize heuristic but crashes tensor conversion.
                    content_type = resp.headers.get("Content-Type", "")
                    if content_type and not content_type.startswith("image/"):
                        log.warning("Non-image MIME type '%s' from %s — skipping",
                                    content_type, url[:60])
                        frame_arr = None
                        break  # try next URL

                    # Reject tiny responses (< 2KB = likely error page or placeholder)
                    if len(resp.content) < 2048:
                        log.warning("Response too small (%d bytes) from %s — skipping",
                                    len(resp.content), url[:60])
                        frame_arr = None
                        break  # try next URL

                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    frame_arr = np.array(img).astype(np.float32) / 255.0

                    # Reject Caltrans "Temporarily Unavailable" placeholder frames.
                    # Placeholder: white background + dark text = mean brightness > 0.72
                    # and very low std dev (simple 2-color image).
                    # Real camera frames (road, sky, traffic) have mean < 0.65 and higher variance.
                    mean_brightness = frame_arr.mean()
                    std_brightness = frame_arr.std()
                    if mean_brightness > 0.72 and std_brightness < 0.25:
                        log.warning("Placeholder detected (mean=%.2f std=%.2f) — skipping %s",
                                    mean_brightness, std_brightness, url[:60])
                        # v3.4: Mark as dark camera for TTL period
                        with _dark_cache_lock:
                            _dark_camera_cache[url] = now
                        frame_arr = None
                        break  # try next URL

                    fetched_url = url
                    log.info("Frame fetched: %dx%d %.1fKB brightness=%.2f from %s",
                             img.width, img.height, len(resp.content)/1024,
                             mean_brightness, url[:60])
                    break
                except Exception as e:
                    log.warning("Fetch attempt %d failed for %s: %s", attempt + 1, url[:60], e)
                    if attempt < retry_count:
                        time.sleep(0.5)

            if frame_arr is not None:
                break

        if dark_skipped > 0:
            log.info("Dark-camera cache: skipped %d known-dark URLs (TTL=%ds)",
                     dark_skipped, _DARK_CAMERA_TTL)

        if frame_arr is None:
            log.error("All URLs exhausted, returning fallback frame")
            return self._fallback(fallback_width, fallback_height, fallback_images_dir)

        # SSIM stale detection (crop top/bottom to ignore timestamps)
        # v3.5: Luminance gate — skip SSIM for very dark frames (nighttime cams).
        # At night, static dark frames legitimately look identical frame-to-frame;
        # SSIM > threshold would false-positive into procedural fallback.
        ssim_score = 0.0
        is_stale = False
        mean_lum = float(frame_arr.mean())
        with _cache_lock:
            cached_entry = _frame_cache.get(fetched_url)
        if cached_entry is not None:
            _, cached_arr = cached_entry
            if mean_lum < 0.12:
                log.info("Night-mode bypass: mean luminance %.3f < 0.12, skipping SSIM",
                         mean_lum)
                ssim_score = -2.0  # sentinel: SSIM skipped (night mode)
            else:
                ssim_score = self._compute_ssim(cached_arr, frame_arr, crop_pct)
                if ssim_score > ssim_threshold:
                    log.warning("Frame STALE (SSIM=%.4f > %.4f) — forcing t2v fallback",
                                ssim_score, ssim_threshold)
                    is_stale = True

        # Update cache (evict oldest if full)
        self._cache_put(fetched_url, now, frame_arr)

        # Defensive copy for output tensor
        tensor = torch.from_numpy(frame_arr.copy()).unsqueeze(0)
        success = not is_stale
        return (tensor, success, ssim_score)

    def _fallback(self, w, h, fallback_images_dir=""):
        """
        Fallback frame when all live URLs fail.
        Priority:
          1. Random image from fallback_images_dir (if set and populated)
          2. Procedural night-highway gradient (better LTX-2 conditioning than flat grey)
        """
        # --- Option 1: Load from local image bank ---
        if fallback_images_dir and fallback_images_dir.strip():
            arr = self._load_fallback_image(fallback_images_dir.strip(), w, h)
            if arr is not None:
                tensor = torch.from_numpy(arr).unsqueeze(0)
                log.info("Fallback: loaded from local image bank (%dx%d)", w, h)
                return (tensor, False, 0.0)

        # --- Option 2: Procedural night-highway gradient ---
        arr = self._procedural_highway(w, h)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return (tensor, False, 0.0)

    def _load_fallback_image(self, folder, w, h):
        """Load a random image from a local folder. Returns float32 HxWx3 or None."""
        import os, glob, random
        try:
            from PIL import Image as PILImage
        except ImportError:
            return None
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(folder, pat)))
            files.extend(glob.glob(os.path.join(folder, pat.upper())))
        if not files:
            log.warning("fallback_images_dir is set but no images found: %s", folder)
            return None
        # Seed by minute for variety across clips within one run, stability across runs
        rng = random.Random(int(time.time() // 60))
        path = rng.choice(files)
        try:
            img = PILImage.open(path).convert("RGB")
            img = img.resize((w, h), PILImage.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            log.info("Fallback image: %s -> %dx%d", os.path.basename(path), w, h)
            return arr
        except Exception as e:
            log.warning("Failed to load fallback image %s: %s", path, e)
            return None

    def _procedural_highway(self, w, h):
        """
        Generate a procedural night-highway frame.
        Dark asphalt + lane markings + distant headlight glow.
        Gives LTX-2 a structured starting point rather than flat grey.
        """
        arr = np.zeros((h, w, 3), dtype=np.float32)

        # Sky gradient (top 40%): dark blue-grey to near-black
        sky_h = int(h * 0.40)
        for row in range(sky_h):
            t = row / max(sky_h - 1, 1)
            # Dark teal-grey sky
            arr[row, :, 0] = 0.04 + t * 0.08   # R
            arr[row, :, 1] = 0.06 + t * 0.10   # G
            arr[row, :, 2] = 0.10 + t * 0.14   # B

        # Road surface (bottom 60%): dark asphalt with slight texture
        road_start = sky_h
        for row in range(road_start, h):
            t = (row - road_start) / max(h - road_start - 1, 1)
            # Perspective darkening toward horizon
            brightness = 0.14 + (1.0 - t) * 0.06
            arr[row, :, 0] = brightness * 0.85
            arr[row, :, 1] = brightness * 0.88
            arr[row, :, 2] = brightness * 0.90

        # Horizon glow (distant LA city lights)
        horizon_row = sky_h
        glow_h = int(h * 0.07)
        for row in range(max(0, horizon_row - glow_h), min(h, horizon_row + glow_h)):
            d = abs(row - horizon_row) / max(glow_h, 1)
            glow = 0.18 * max(0, 1.0 - d * 2.0)
            arr[row, :, 0] += glow * 0.9
            arr[row, :, 1] += glow * 0.7
            arr[row, :, 2] += glow * 0.4

        # Lane markings (dashed white center lines)
        num_lanes = 4
        lane_w = w // num_lanes
        dash_h = max(3, h // 30)
        gap_h = max(4, h // 20)
        mark_w = max(2, w // 120)

        for lane in range(1, num_lanes):
            cx = lane * lane_w
            y = road_start + int(h * 0.1)
            while y < h - dash_h:
                # Perspective scaling: marks get thicker near bottom
                t = (y - road_start) / max(h - road_start, 1)
                scaled_w = max(1, int(mark_w * (0.5 + t * 1.5)))
                x0 = max(0, cx - scaled_w // 2)
                x1 = min(w, cx + scaled_w // 2 + 1)
                brightness = 0.65 + t * 0.15
                arr[y:y + dash_h, x0:x1, :] = brightness
                y += dash_h + gap_h

        # Distant headlight pair (center-ish, near horizon)
        for cx_offset in [-int(w * 0.06), int(w * 0.06)]:
            cx = w // 2 + cx_offset
            cy = horizon_row + int(h * 0.04)
            rx, ry = max(3, w // 80), max(2, h // 60)
            for dy in range(-ry * 3, ry * 3 + 1):
                for dx in range(-rx * 3, rx * 3 + 1):
                    dist = ((dx / rx) ** 2 + (dy / ry) ** 2) ** 0.5
                    if 0 <= cy + dy < h and 0 <= cx + dx < w:
                        glow = max(0, 0.9 - dist * 0.35)
                        arr[cy + dy, cx + dx, 0] = min(1.0, arr[cy + dy, cx + dx, 0] + glow * 0.95)
                        arr[cy + dy, cx + dx, 1] = min(1.0, arr[cy + dy, cx + dx, 1] + glow * 0.92)
                        arr[cy + dy, cx + dx, 2] = min(1.0, arr[cy + dy, cx + dx, 2] + glow * 0.80)

        arr = np.clip(arr, 0.0, 1.0)
        log.info("Fallback: procedural night-highway generated (%dx%d)", w, h)
        return arr

    def _cache_put(self, url, timestamp, arr):
        """Add to cache with max-size eviction (oldest first)."""
        with _cache_lock:
            if len(_frame_cache) >= _CACHE_MAX_SIZE:
                oldest_url = min(_frame_cache, key=lambda u: _frame_cache[u][0])
                del _frame_cache[oldest_url]
                log.debug("Evicted oldest cache entry: %s", oldest_url[:60])
            _frame_cache[url] = (timestamp, arr.copy())

    def _compute_ssim(self, img1, img2, crop_pct=0.10):
        """SSIM with coordinate-based cropping to ignore timestamp overlays.
        Crops top and bottom crop_pct of the frame before comparison."""
        try:
            if img1.shape != img2.shape:
                return 0.0

            h = img1.shape[0]
            top = int(h * crop_pct)
            bot = h - int(h * crop_pct)

            # Crop to center region (ignore timestamp areas)
            g1 = np.mean(img1[top:bot, :, :], axis=2)
            g2 = np.mean(img2[top:bot, :, :], axis=2)

            mu1 = np.mean(g1)
            mu2 = np.mean(g2)
            sigma1_sq = np.var(g1)
            sigma2_sq = np.var(g2)
            sigma12 = np.mean((g1 - mu1) * (g2 - mu2))

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

            return float(num / den) if den > 0 else 0.0
        except Exception:
            return 0.0
