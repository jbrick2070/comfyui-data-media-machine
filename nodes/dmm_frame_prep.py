"""
DMMFramePrep — Preprocesses webcam frames for LTX-2 image-to-video input.

v3.1 changes:
  - Fixed neon noir grade clipping: 0.03 → 0.0 floor (blackout strips stayed grey)
  - Added total prep timing log
  - Neon noir grade now preserves blackout strips (re-applies after grading)

Pipeline: OpenCV denoise (at native resolution) → Lanczos upscale →
luminance-based style grading.  All CPU-only, zero VRAM.

Denoise runs BEFORE upscale to avoid processing millions of extra pixels.
Night detection uses actual image luminance rather than system clock,
enabling correct behavior for any timezone or deployment.

Author: Jeffrey A. Brick
"""

import logging
import time

import numpy as np
import torch

log = logging.getLogger("DMM.FramePrep")


class DMMFramePrep:
    """Prepares a raw webcam frame for LTX-2 image-to-video generation."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "prep_frame"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "target_width": ("INT", {"default": 768, "min": 64, "max": 2048}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "crop_mode": (["center_crop", "letterbox", "stretch"], {"default": "center_crop"}),
                "auto_style_grade": ("BOOLEAN", {"default": True,
                    "tooltip": "Auto-apply neon noir grading for dark/night frames based on luminance"}),
                "luminance_threshold": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.5, "step": 0.01,
                    "tooltip": "Mean luminance below this triggers night grading"}),
                "denoise_strength": ("INT", {"default": 3, "min": 0, "max": 15,
                    "tooltip": "Median blur kernel size (0=disabled, 3/5/7 typical). Fast JPEG cleanup before upscale."}),
                "blackout_pct": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.25, "step": 0.01,
                    "tooltip": "Paint top/bottom N% of frame black BEFORE encoding. "
                               "Hides burned-in timestamps that LTX-2 would otherwise try to animate."}),
                "weather_data": ("DMM_WEATHER", {
                    "tooltip": "Optional: if connected, daytime storms won't falsely trigger night grading"}),
            },
        }

    def prep_frame(self, image, target_width=768, target_height=512,
                   crop_mode="center_crop", auto_style_grade=True,
                   luminance_threshold=0.25, denoise_strength=3,
                   blackout_pct=0.10, weather_data=None):
        from PIL import Image

        t0 = time.time()

        # Convert tensor to numpy (B, H, W, C)
        try:
            arr = image[0].cpu().numpy()
        except Exception as e:
            log.error("Failed to convert image tensor: %s — returning black frame", e)
            black = np.zeros((target_height, target_width, 3), dtype=np.float32)
            return (torch.from_numpy(black).unsqueeze(0),)
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)

        # --- Step 1: Denoise at NATIVE resolution (fewer pixels = faster) ---
        if denoise_strength > 0:
            arr_uint8 = self._denoise(arr_uint8, denoise_strength)

        # --- Step 1.5: Timestamp blackout (paint top/bottom N% black) ---
        if blackout_pct > 0:
            h = arr_uint8.shape[0]
            strip = max(1, int(h * blackout_pct))
            arr_uint8[:strip, :, :] = 0       # top strip
            arr_uint8[-strip:, :, :] = 0      # bottom strip
            log.info("Blackout: painted top/bottom %d px black (%.0f%% of %d)",
                     strip, blackout_pct * 100, h)

        # --- Step 2: Resize with Lanczos ---
        pil_img = Image.fromarray(arr_uint8)
        src_w, src_h = pil_img.size

        if crop_mode == "stretch":
            pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
        elif crop_mode == "letterbox":
            pil_img = self._letterbox(pil_img, target_width, target_height)
        else:  # center_crop
            pil_img = self._center_crop(pil_img, target_width, target_height)

        result = np.array(pil_img).astype(np.float32) / 255.0
        log.info("Processed %dx%d -> %dx%d (%s)", src_w, src_h, target_width, target_height, crop_mode)

        # --- Step 3: Luminance-based style grading ---
        if auto_style_grade:
            # ITU-R BT.709 luminance
            luminance = np.mean(
                result[:, :, 0] * 0.2126 +
                result[:, :, 1] * 0.7152 +
                result[:, :, 2] * 0.0722
            )

            # Weather override: if weather data says it's daytime but a storm
            # drops luminance below threshold, don't falsely trigger night grading
            is_daytime_storm = False
            if weather_data and luminance < luminance_threshold:
                desc = weather_data.get("description", "").lower()
                hour = time.localtime().tm_hour
                if 6 <= hour < 19 and any(w in desc for w in ["storm", "thunder", "rain", "overcast"]):
                    is_daytime_storm = True
                    log.info("Daytime storm detected (desc='%s', lum=%.3f) — skipping night grade",
                             desc, luminance)

            if luminance < luminance_threshold and not is_daytime_storm:
                result = self._neon_noir_grade(result)
                log.info("Applied neon noir grading (luminance=%.3f < %.3f)",
                         luminance, luminance_threshold)
            else:
                log.debug("Skipping night grade (luminance=%.3f, threshold=%.3f, storm=%s)",
                          luminance, luminance_threshold, is_daytime_storm)

        tensor = torch.from_numpy(result).unsqueeze(0)
        log.info("FramePrep total: %.1fms", (time.time() - t0) * 1000)
        return (tensor,)

    def _center_crop(self, img, tw, th):
        """Resize preserving aspect ratio, then center crop."""
        from PIL import Image
        src_w, src_h = img.size
        if src_h == 0 or src_w == 0:
            log.warning("Zero-dimension source image (%dx%d), returning stretched resize", src_w, src_h)
            return img.resize((tw, th), Image.LANCZOS)
        target_ratio = tw / th
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            new_h = th
            new_w = int(src_w * (th / src_h))
        else:
            new_w = tw
            new_h = int(src_h * (tw / src_w))

        new_w = max(new_w, tw)
        new_h = max(new_h, th)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = max(0, (new_w - tw) // 2)
        top = max(0, (new_h - th) // 2)
        return img.crop((left, top, left + tw, top + th))

    def _letterbox(self, img, tw, th):
        """Resize to fit within target, pad with black bars."""
        from PIL import Image
        src_w, src_h = img.size
        scale = min(tw / src_w, th / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        paste_x = (tw - new_w) // 2
        paste_y = (th - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas

    def _denoise(self, uint8_arr, strength):
        """Fast median blur for JPEG artifact cleanup.  Much faster than
        fastNlMeansDenoisingColored and sufficient since LTX-2 smooths
        remaining artifacts during diffusion."""
        if strength <= 0:
            return uint8_arr
        try:
            import cv2
            # medianBlur requires odd kernel size
            ksize = strength if strength % 2 == 1 else strength + 1
            ksize = max(3, min(ksize, 15))
            denoised = cv2.medianBlur(uint8_arr, ksize)
            log.info("Median blur at native %dx%d, kernel=%d",
                     uint8_arr.shape[1], uint8_arr.shape[0], ksize)
            return denoised
        except ImportError:
            log.warning("OpenCV not available — skipping denoise. "
                        "Install with: pip install opencv-python")
            return uint8_arr
        except Exception as e:
            log.warning("Denoise failed: %s", e)
            return uint8_arr

    def _neon_noir_grade(self, arr):
        """Neon noir color grading for dark/night frames.
        Uses proper BT.709 luminance for shadow/highlight detection."""
        # Compute per-pixel luminance
        lum = arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722

        # Increase contrast
        mean = np.mean(arr)
        contrast = 1.4
        arr = (arr - mean) * contrast + mean

        # Crush blacks (floor at 0.0 to preserve blackout strips)
        arr = np.clip(arr, 0.0, 1.0)

        # Blue push in shadows (luminance-based, not red-channel)
        shadow_mask = lum < 0.3
        for c in range(3):
            if c == 2:  # blue channel
                arr[:, :, c] = np.where(shadow_mask, arr[:, :, c] * 1.3, arr[:, :, c])

        # Warm push in highlights (luminance-based)
        highlight_mask = lum > 0.6
        arr[:, :, 0] = np.where(highlight_mask, arr[:, :, 0] * 1.1, arr[:, :, 0])

        # Slight saturation boost
        gray = np.mean(arr, axis=2, keepdims=True)
        arr = gray + (arr - gray) * 1.2

        return np.clip(arr, 0.0, 1.0)
