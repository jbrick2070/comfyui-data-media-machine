"""
DMM_ProceduralClip — cinematic procedural motion graphics from live narration data.

Generates animated frames using PIL/Pillow with stylized aesthetics inspired by
Claude's native Python video generation technique (Theoretically Media / ComfyUI).

No GPU required — pure CPU frame generation with cinematic post-processing.

Styles:
  la_neon        — dark bg, neon grid, palm silhouettes, data readouts,
                   chromatic aberration, film grain, bloom, animated gauges
  minimal_data   — clean dark panels, typography-focused, subtle grain,
                   breathing dividers, data fade transitions
  retro_terminal — green-on-black terminal / hacker aesthetic,
                   CRT warp, phosphor bloom, flicker

Parses the TTS narration text to extract data points (temp, AQI, quakes,
transit, etc.) and renders them as animated overlays.

Use case: extend total video duration to match TTS narration length
by appending artistic procedural clips after diffusion-generated content.

v3.4: initial implementation
v3.4.1: cinematic upgrade — chromatic aberration, film grain, bloom,
         vignette, animated arc gauges, heat shimmer, breathing glow,
         enhanced particles with trails, organic Perlin-style motion
v3.4.2: procedural art upgrade — cellular automata (Game of Life) bg layer,
         value noise for organic motion, fractal border accents (Mandelbrot),
         kinetic typography with momentum/spring physics,
         inspired by Claude's native Python+FFmpeg video generation paradigm
"""

from __future__ import annotations

import logging
import math
import re
import time

import numpy as np
import torch

log = logging.getLogger("DMM.ProceduralClip")


# ── Color palettes ────────────────────────────────────────────────────
NEON = {
    "bg":       (10, 10, 26),
    "grid":     (25, 25, 60),
    "pink":     (255, 20, 147),
    "cyan":     (0, 229, 255),
    "amber":    (255, 176, 0),
    "purple":   (180, 0, 255),
    "white":    (220, 220, 230),
    "dim":      (80, 80, 120),
    "skyline":  (12, 12, 30),
    "palm":     (6, 6, 18),
    "hot":      (255, 60, 60),
    "cool":     (60, 160, 255),
}


# ── Helpers ───────────────────────────────────────────────────────────
def _font(size: int):
    """Load a monospace font with cross-platform fallbacks."""
    from PIL import ImageFont
    for path in [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\lucon.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def _typewriter(text: str, progress: float) -> str:
    n = int(len(text) * min(max(progress, 0.0), 1.0))
    return text[:n]


def _glow_text(draw, xy, text, font, color, glow_radius=2, intensity=0.33):
    """Draw text with a neon glow halo — variable intensity."""
    x, y = xy
    glow = tuple(max(0, int(c * intensity)) for c in color)
    for dx in range(-glow_radius, glow_radius + 1):
        for dy in range(-glow_radius, glow_radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = abs(dx) + abs(dy)
            if dist <= glow_radius:
                falloff = 1.0 - dist / (glow_radius + 1)
                g = tuple(max(0, int(c * intensity * falloff)) for c in color)
                draw.text((x + dx, y + dy), text, font=font, fill=g)
    draw.text((x, y), text, font=font, fill=color)


def _dim(color, factor):
    """Dim a color by a factor (0-1)."""
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _lerp_color(c1, c2, t):
    """Linear interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _noise_val(x, y, seed=0):
    """Cheap hash-based pseudo noise (0..1). Not Perlin, but fast and tiling."""
    n = int(x * 374761393 + y * 668265263 + seed * 1274126177) & 0x7fffffff
    n = (n ^ (n >> 13)) * 1103515245
    n = (n ^ (n >> 16)) & 0x7fffffff
    return (n & 0xffff) / 65535.0


def _breathing(t, freq=1.0, lo=0.7, hi=1.0):
    """Sinusoidal breathing animation between lo and hi."""
    v = (math.sin(t * freq * math.pi * 2) + 1.0) / 2.0
    return lo + v * (hi - lo)


# ── Value Noise (smooth, Perlin-like) ────────────────────────────────
def _smooth_noise(x, y, seed=0):
    """Smoothed value noise via bilinear interpolation of hash grid."""
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    # Smoothstep
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    n00 = _noise_val(ix, iy, seed)
    n10 = _noise_val(ix + 1, iy, seed)
    n01 = _noise_val(ix, iy + 1, seed)
    n11 = _noise_val(ix + 1, iy + 1, seed)
    nx0 = n00 + (n10 - n00) * fx
    nx1 = n01 + (n11 - n01) * fx
    return nx0 + (nx1 - nx0) * fy


def _fbm(x, y, octaves=4, seed=0):
    """Fractal Brownian Motion — layered value noise for organic textures."""
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    for _ in range(octaves):
        value += amplitude * _smooth_noise(x * frequency, y * frequency, seed)
        amplitude *= 0.5
        frequency *= 2.0
    return value


# ── Cellular Automata (Game of Life) ─────────────────────────────────
class _CellularAutomata:
    """Conway's Game of Life grid — used as living background texture.

    Pre-computed at init, stepped forward each frame. Low-res grid
    mapped to full-res canvas for performance.
    """

    def __init__(self, cols, rows, seed=42):
        rng = np.random.RandomState(seed)
        # Sparse initial state (~18% alive for interesting patterns)
        self.grid = (rng.random((rows, cols)) < 0.18).astype(np.uint8)
        self.rows = rows
        self.cols = cols

    def step(self):
        """Advance one generation via vectorized neighbor count."""
        g = self.grid
        # Count 8 neighbors using rolled arrays
        n = (np.roll(g, 1, 0) + np.roll(g, -1, 0) +
             np.roll(g, 1, 1) + np.roll(g, -1, 1) +
             np.roll(np.roll(g, 1, 0), 1, 1) +
             np.roll(np.roll(g, 1, 0), -1, 1) +
             np.roll(np.roll(g, -1, 0), 1, 1) +
             np.roll(np.roll(g, -1, 0), -1, 1))
        # B3/S23 rules
        self.grid = ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)

    def render(self, w, h, color, intensity=0.08):
        """Render grid as a dim overlay image (numpy HxWx3 uint8).

        Vectorized: uses np.kron to upscale the grid instead of Python loops.
        """
        # Build a small (rows, cols, 3) color image from the grid
        color_arr = np.array(color, dtype=np.float32) * intensity
        small = self.grid[:, :, np.newaxis].astype(np.float32) * color_arr
        # Upscale to canvas size using repeat (fast, no Python loop)
        cell_h = max(1, h // self.rows)
        cell_w = max(1, w // self.cols)
        overlay = np.repeat(np.repeat(small, cell_h, axis=0), cell_w, axis=1)
        # Fit to exact canvas size (repeat may overshoot or undershoot)
        oh, ow = overlay.shape[:2]
        if oh < h or ow < w:
            padded = np.zeros((h, w, 3), dtype=np.float32)
            padded[:min(oh, h), :min(ow, w), :] = overlay[:min(oh, h), :min(ow, w), :]
            overlay = padded
        else:
            overlay = overlay[:h, :w, :]
        return overlay


# ── Mandelbrot fractal accent ────────────────────────────────────────
def _mandelbrot_line(y_norm, x_start, x_end, steps=80, max_iter=20):
    """Compute a horizontal slice of the Mandelbrot set.

    Returns a list of (x_norm, escape_val) pairs for drawing fractal
    border accents — the decorative "data filigree" at edges.
    """
    points = []
    cy = (y_norm - 0.5) * 2.5  # map to complex plane
    for i in range(steps):
        x_norm = x_start + (x_end - x_start) * i / steps
        cx = (x_norm - 0.5) * 3.5 - 0.5  # center on interesting region
        z = complex(0, 0)
        c = complex(cx, cy)
        escape = 0
        for n in range(max_iter):
            if abs(z) > 2.0:
                escape = n / max_iter
                break
            z = z * z + c
        points.append((x_norm, escape))
    return points


# ── Kinetic typography spring physics ────────────────────────────────
def _spring_overshoot(t, damping=5.0, freq=8.0):
    """Damped spring — overshoots then settles to 1.0.

    Creates that satisfying "pop-in" motion for text elements.
    t: 0→1 normalized progress.
    """
    if t <= 0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 1.0 - math.exp(-damping * t) * math.cos(freq * t)


def _elastic_scale(t, magnitude=0.15):
    """Elastic scale factor: starts at 0, overshoots past 1.0, settles."""
    if t <= 0:
        return 0.0
    s = _spring_overshoot(t)
    return s


# ── Node ──────────────────────────────────────────────────────────────
class DMMProceduralClip:
    """Generate cinematic procedural data-visualization motion graphics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narrative": ("STRING", {
                    "multiline": True,
                    "tooltip": "TTS narration text — data points are extracted via regex",
                }),
                "width": ("INT", {"default": 512, "min": 128, "max": 3840, "step": 32}),
                "height": ("INT", {"default": 288, "min": 128, "max": 2160, "step": 32}),
                "duration_sec": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                           "tooltip": "Clip length in seconds"}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "style": (["la_neon", "minimal_data", "retro_terminal"],
                          {"default": "la_neon"}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "DataMediaMachine"

    # ── Entry point ─────────────────────────────────────────────────
    def generate(self, narrative, width, height, duration_sec, fps, style):
        from PIL import Image  # noqa: late import
        import random

        start_time = time.time()
        total_frames = int(duration_sec * fps)
        data = self._parse_narrative(narrative)

        # Scale fonts to output resolution
        s = height / 512.0
        fonts = {
            "title":   _font(int(48 * s)),
            "heading": _font(int(28 * s)),
            "body":    _font(int(20 * s)),
            "small":   _font(int(14 * s)),
            "tiny":    _font(int(10 * s)),
        }

        # Deterministic decorative elements
        rng = random.Random(42)
        particles = [
            {
                "x": rng.random(), "y": rng.random(),
                "speed": rng.uniform(0.3, 1.5),
                "color": rng.choice([NEON["pink"], NEON["cyan"], NEON["amber"], NEON["purple"]]),
                "size": rng.uniform(0.5, 2.0),
                "phase": rng.random() * math.pi * 2,
                "drift": rng.uniform(-0.02, 0.02),
            }
            for _ in range(60)
        ]
        rng2 = random.Random(7)
        skyline = [rng2.uniform(0.08, 0.28) for _ in range(24)]

        # Pre-generate film grain texture (one per ~5 frames for perf)
        grain_seed = rng.randint(0, 99999)

        # Cellular automata — low-res grid for living background texture
        # Step every 3 frames for visible evolution without being too fast
        ca_cols = max(16, width // 24)
        ca_rows = max(12, height // 24)
        ca = _CellularAutomata(ca_cols, ca_rows, seed=42)

        # Pre-compute Mandelbrot border accents (static, computed once)
        fractal_top = _mandelbrot_line(0.35, 0.0, 1.0, steps=width // 4)
        fractal_bot = _mandelbrot_line(0.65, 0.0, 1.0, steps=width // 4)

        # Render all frames
        frames = []
        for f in range(total_frames):
            t = f / max(total_frames - 1, 1)

            # Step cellular automata every 3 frames
            if f % 3 == 0 and f > 0:
                ca.step()

            if style == "la_neon":
                img = self._la_neon(width, height, t, f, fps, data, fonts,
                                    particles, skyline, grain_seed,
                                    ca, fractal_top, fractal_bot)
            elif style == "minimal_data":
                img = self._minimal(width, height, t, f, fps, data, fonts,
                                    grain_seed)
            else:
                img = self._retro(width, height, t, f, fps, data, fonts,
                                  ca=ca)

            arr = np.array(img, dtype=np.float32) / 255.0
            frames.append(torch.from_numpy(arr))

        frame_tensor = torch.stack(frames, dim=0)

        # Silent audio at 48 kHz (TTS narration from AudioMux plays over this)
        n_samples = int(duration_sec * 48000)
        silence = torch.zeros(1, 2, n_samples)  # stereo to match LTX-2 audio VAE output
        audio = {"waveform": silence, "sample_rate": 48000}

        # Wrap as VIDEO via ComfyUI's CreateVideo
        video = self._to_video(frame_tensor, audio, fps)

        elapsed = time.time() - start_time
        log.info("[ProceduralClip] %d frames %dx%d '%s' in %.1fs",
                 total_frames, width, height, style, elapsed)

        return (video,)

    def _to_video(self, images, audio, fps):
        """IMAGE + AUDIO → VIDEO using ComfyUI's CreateVideo node."""
        from nodes import NODE_CLASS_MAPPINGS
        cls = NODE_CLASS_MAPPINGS.get("CreateVideo")
        if cls is None:
            raise RuntimeError("CreateVideo node not found — is ComfyUI-LTXVideo installed?")
        obj = cls()
        fn = getattr(cls, "FUNCTION", "execute")
        result = getattr(obj, fn)(images=images, audio=audio, fps=fps)
        return result.args[0] if hasattr(result, "args") else result[0]

    # ── Narrative parser ────────────────────────────────────────────
    def _parse_narrative(self, text: str) -> dict:
        """Extract data points from narration text via regex."""
        d = {
            "city": "LOS ANGELES", "time_str": "", "period": "",
            "temp": "—", "temp_num": 0, "conditions": "—", "wind": "—",
            "wind_num": 0, "quakes": "0", "quake_max": "—",
            "aqi": "—", "aqi_num": 0,
            "buses": "—", "congestion": "—", "cong_num": 0,
        }

        m = re.search(r"^([A-Za-z\s]+),", text)
        if m:
            d["city"] = m.group(1).strip().upper()

        m = re.search(r"(\d{1,2}:\d{2}\s*[AP]M)", text, re.IGNORECASE)
        if m:
            d["time_str"] = m.group(1).upper()

        m = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
                       r"\s+(morning|afternoon|evening|night)",
                       text, re.IGNORECASE)
        if m:
            d["period"] = f"{m.group(1).capitalize()} {m.group(2).capitalize()}"

        m = re.search(r"(\d+)\s*°?\s*F", text)
        if m:
            d["temp"] = m.group(1) + "°F"
            d["temp_num"] = int(m.group(1))

        m = re.search(r"(?:Clear|Cloudy|Overcast|Rain|Fog|Partly\s*Cloudy|Hazy|Windy)",
                       text, re.IGNORECASE)
        if m:
            d["conditions"] = m.group(0).title()

        m = re.search(r"winds?\s*:?\s*(\d+)\s*mph", text, re.IGNORECASE)
        if m:
            d["wind"] = m.group(1) + " mph"
            d["wind_num"] = int(m.group(1))

        m = re.search(r"(\d+)\s*(?:quakes?|earthquakes?)", text, re.IGNORECASE)
        if m:
            d["quakes"] = m.group(1)

        m = re.search(r"max:?\s*m?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            d["quake_max"] = "M" + m.group(1)

        m = re.search(r"aqi:?\s*(\d+)", text, re.IGNORECASE)
        if m:
            d["aqi"] = m.group(1)
            d["aqi_num"] = int(m.group(1))

        m = re.search(r"(\d+)\s*%?\s*(?:congestion|cong)", text, re.IGNORECASE)
        if m:
            d["congestion"] = m.group(1) + "%"
            d["cong_num"] = int(m.group(1))

        m = re.search(r"(\d+)\s*buses?", text, re.IGNORECASE)
        if m:
            d["buses"] = m.group(1)

        log.info("[ProceduralClip] parsed: %s", d)
        return d

    # ── Post-processing effects ─────────────────────────────────────
    def _apply_vignette(self, img_arr, strength=0.4):
        """Cinematic edge darkening vignette (operates on numpy HxWx3 uint8)."""
        h, w = img_arr.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        max_dist = math.sqrt(cx * cx + cy * cy)

        # Create radial gradient
        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        # Smooth falloff — darken edges
        vignette = 1.0 - strength * (dist / max_dist) ** 1.8
        vignette = np.clip(vignette, 0, 1)

        result = img_arr.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= vignette
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_chromatic_aberration(self, img_arr, offset=2):
        """Subtle RGB channel offset for cinematic lens effect."""
        if offset == 0:
            return img_arr
        h, w = img_arr.shape[:2]
        result = img_arr.copy()
        # Shift red channel left, blue right
        result[:, :offset, 0] = img_arr[:, :offset, 0]
        result[:, offset:, 0] = img_arr[:, :-offset, 0]
        result[:, :w - offset, 2] = img_arr[:, offset:, 2]
        result[:, w - offset:, 2] = img_arr[:, w - offset:, 2]
        return result

    def _apply_film_grain(self, img_arr, intensity=12, frame_idx=0):
        """Add film grain noise — varies per frame for organic feel."""
        rng = np.random.RandomState(seed=(frame_idx * 7 + 31) & 0x7fffffff)
        noise = rng.randint(-intensity, intensity + 1,
                            size=img_arr.shape, dtype=np.int16)
        result = img_arr.astype(np.int16) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_bloom(self, img_arr, threshold=200, radius=8, intensity=0.3):
        """Soft bloom on bright areas — simulates lens diffusion."""
        # Find bright pixels
        bright = img_arr.astype(np.float32)
        mask = np.max(bright, axis=2) > threshold
        bloom_layer = np.zeros_like(bright)
        bloom_layer[mask] = bright[mask]

        # Simple box blur approximation (fast, good enough for real-time)
        from PIL import Image, ImageFilter
        bloom_img = Image.fromarray(bloom_layer.astype(np.uint8))
        bloom_img = bloom_img.filter(ImageFilter.GaussianBlur(radius=radius))
        bloom_arr = np.array(bloom_img, dtype=np.float32)

        result = img_arr.astype(np.float32) + bloom_arr * intensity
        return np.clip(result, 0, 255).astype(np.uint8)

    # ══════════════════════════════════════════════════════════════════
    #  LA NEON style — cinematic upgrade
    # ══════════════════════════════════════════════════════════════════
    def _la_neon(self, w, h, t, frame, fps, data, fonts, particles, sky_h,
                 grain_seed, ca=None, fractal_top=None, fractal_bot=None):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), NEON["bg"])
        draw = ImageDraw.Draw(img)

        # ── BG gradient (deeper, more cinematic) ──
        # Vectorized vertical gradient (no Python per-row loop)
        ratio = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        r_ch = np.clip(8 + ratio * 6, 0, 255).astype(np.uint8)
        g_ch = np.clip(6 + ratio * 8, 0, 255).astype(np.uint8)
        b_ch = np.clip(22 + ratio * 18, 0, 255).astype(np.uint8)
        grad = np.concatenate([
            np.broadcast_to(r_ch[:, :, np.newaxis], (h, w, 1)),
            np.broadcast_to(g_ch[:, :, np.newaxis], (h, w, 1)),
            np.broadcast_to(b_ch[:, :, np.newaxis], (h, w, 1)),
        ], axis=2)
        img = Image.fromarray(grad)
        draw = ImageDraw.Draw(img)

        # ── Cellular automata background (living texture) ──
        if ca is not None:
            ca_fade = _ease_out(min(t / 0.5, 1.0))
            if ca_fade > 0:
                ca_overlay = ca.render(w, h, NEON["cyan"],
                                       intensity=0.04 * ca_fade)
                # Blend into image via numpy
                img_arr = np.array(img, dtype=np.float32)
                img_arr += ca_overlay
                img_arr = np.clip(img_arr, 0, 255)
                img = Image.fromarray(img_arr.astype(np.uint8))
                draw = ImageDraw.Draw(img)

        # ── Subtle horizontal scan lines (vectorized) ──
        img_arr_sl = np.array(img)
        img_arr_sl[::3, :, :] = np.clip(
            img_arr_sl[::3, :, :].astype(np.int16) - np.array([4, 4, 6], dtype=np.int16),
            0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr_sl)
        draw = ImageDraw.Draw(img)

        # ── Perspective grid (bottom 40%) with FBM noise distortion ──
        grid_fade = _ease_out(min(t / 0.25, 1.0))
        grid_pulse = _breathing(frame / fps, freq=0.3, lo=0.7, hi=1.0)
        time_sec = frame / fps
        if grid_fade > 0:
            horizon = int(h * 0.6)
            # Horizontal lines with FBM noise warp
            for i in range(16):
                p = i / 16.0
                y = horizon + int((h - horizon) * p * p)
                # FBM noise adds organic undulation to each grid line
                noise_warp = _fbm(i * 0.5, time_sec * 0.4, octaves=3) * 6 - 3
                wave = math.sin(time_sec * 1.5 + i * 0.4) * 3 + noise_warp
                bright = int(grid_fade * grid_pulse * (30 + 20 * math.sin(time_sec * 2 + i)))
                col = (bright // 4, bright // 3, bright)
                draw.line([(0, int(y + wave)), (w, int(y - wave))], fill=col)
            # Vertical converging lines with subtle sway
            cx = w // 2
            for i in range(20):
                x_bot = int(w * i / 20)
                x_top = cx + int((x_bot - cx) * 0.25)
                # Noise-driven horizontal sway
                sway = _fbm(i * 0.3, time_sec * 0.2, octaves=2) * 4 - 2
                bright = int(grid_fade * grid_pulse * 22)
                draw.line([(int(x_top + sway), horizon), (x_bot, h)],
                          fill=(bright // 4, bright // 3, bright))

        # ── Skyline with lit windows ──
        sky_rise = _ease_out(min(max(t - 0.08, 0) / 0.25, 1.0))
        if sky_rise > 0:
            base_y = h - int(h * 0.07)
            bw = w // len(sky_h)
            for i, bh in enumerate(sky_h):
                bh_px = int(bh * h * sky_rise)
                x0 = i * bw
                x1 = x0 + bw - 2
                y0 = base_y - bh_px
                # Building body — slight gradient
                for by in range(y0, base_y):
                    ratio = (by - y0) / max(bh_px, 1)
                    bc = int(10 + ratio * 4)
                    draw.line([(x0, by), (x1, by)], fill=(bc, bc, bc + 8))
                # Lit windows — some blink
                if bh_px > 20:
                    for wy in range(y0 + 4, base_y - 4, 8):
                        for wx in range(x0 + 3, x1 - 3, 6):
                            hash_v = (wx * 31 + wy * 17 + i * 7) % 11
                            if hash_v < 3:
                                # Warm window light with random flicker
                                flicker = 1.0 if hash_v != 2 else _breathing(
                                    frame / fps, freq=0.5 + (wx % 3) * 0.3,
                                    lo=0.3, hi=1.0)
                                wc = _dim((50, 42, 22), flicker)
                                draw.rectangle([wx, wy, wx + 2, wy + 2], fill=wc)
            # Ground line with subtle glow
            draw.rectangle([0, base_y, w, h], fill=(6, 6, 16))
            glow_bright = int(15 * _breathing(frame / fps, freq=0.2))
            draw.line([(0, base_y), (w, base_y)],
                      fill=(glow_bright, glow_bright, glow_bright + 10), width=1)

        # ── Palm silhouettes with wind sway ──
        if sky_rise > 0.4:
            pf = min((sky_rise - 0.4) * 3, 1.0)
            sway = math.sin(frame / fps * 0.8) * 3
            self._draw_palm(draw, int(w * 0.06), int(h * 0.50), h, pf, sway)
            self._draw_palm(draw, int(w * 0.94), int(h * 0.47), h, pf, -sway)

        # ── Floating particles with FBM noise drift + trails ──
        for p in particles:
            pt = (t * p["speed"]) % 1.0
            # FBM noise-driven organic drift (replaces simple sine)
            noise_dx = _fbm(p["x"] * 4, time_sec * 0.3 + p["phase"],
                            octaves=2) * 0.06 - 0.03
            noise_dy = _fbm(p["y"] * 4 + 100, time_sec * 0.25 + p["phase"],
                            octaves=2) * 0.04 - 0.02
            x = int((p["x"] + noise_dx) % 1.0 * w)
            y = int((p["y"] - pt * 0.3 + noise_dy) % 1.0 * h)
            alpha = 0.3 + 0.5 * math.sin(frame / fps * 2.5 + p["phase"])
            dot_c = _dim(p["color"], alpha)
            sz = max(1, int(p["size"] * h / 512))
            # Particle trail (fading tail)
            trail_len = max(1, int(sz * 2))
            for ti in range(trail_len):
                trail_alpha = alpha * (1.0 - ti / trail_len) * 0.4
                ty_off = y + ti * 2
                if 0 <= ty_off < h:
                    tc = _dim(p["color"], trail_alpha)
                    tsz = max(1, sz - ti)
                    draw.ellipse([x - tsz, ty_off - tsz, x + tsz, ty_off + tsz],
                                 fill=tc)
            # Main particle
            draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=dot_c)

        # ── Title with kinetic spring pop-in + breathing glow ──
        title_raw = min(max(t - 0.12, 0) / 0.2, 1.0)
        title_p = _spring_overshoot(title_raw, damping=4.0, freq=10.0)
        if title_raw > 0:
            city = data["city"]
            display = _typewriter(city, min(title_raw * 1.5, 1.0))
            bbox = fonts["title"].getbbox(city)
            tw = bbox[2] - bbox[0]
            tx = (w - tw) // 2
            # Spring offset — pops up from below then settles
            base_ty = int(h * 0.07)
            spring_offset = int((1.0 - title_p) * h * 0.03)
            ty = base_ty + spring_offset
            glow_i = _breathing(time_sec, freq=0.4, lo=0.25, hi=0.45)
            _glow_text(draw, (tx, ty), display, fonts["title"],
                       NEON["cyan"], glow_radius=4, intensity=glow_i)
            # Underline with gradient — also springs in
            line_w = int(tw * min(title_raw, 1.0))
            line_y = ty + bbox[3] - bbox[1] + int(6 * h / 512)
            for lx in range(line_w):
                lr = lx / max(line_w, 1)
                lc = _lerp_color(NEON["pink"], NEON["cyan"], lr)
                draw.point((tx + lx, line_y), fill=lc)
                draw.point((tx + lx, line_y + 1), fill=lc)

        # ── Subtitle with spring pop-in ──
        sub_raw = min(max(t - 0.22, 0) / 0.15, 1.0)
        sub_spring = _spring_overshoot(sub_raw, damping=5.0, freq=8.0)
        if sub_raw > 0:
            parts = [x for x in [data["period"], data["time_str"]] if x]
            sub = " · ".join(parts) if parts else ""
            if sub:
                display = _typewriter(sub, min(sub_raw * 1.3, 1.0))
                bbox = fonts["heading"].getbbox(sub)
                sw = bbox[2] - bbox[0]
                sx = (w - sw) // 2
                base_sy = int(h * 0.07) + int(58 * h / 512)
                spring_off = int((1.0 - sub_spring) * h * 0.02)
                sy = base_sy + spring_off
                draw.text((sx, sy), display, font=fonts["heading"],
                          fill=_dim(NEON["dim"], min(sub_raw, 1.0)))

        # ── Data panels (2×2 grid) with animated arc gauges ──
        panels = [
            ("WEATHER", f"{data['temp']}  {data['conditions']}",
             f"Wind: {data['wind']}", NEON["cyan"], 0.28,
             data["temp_num"], 120),  # value, max for gauge
            ("AIR QUALITY", f"AQI: {data['aqi']}",
             self._aqi_label(data["aqi_num"]), NEON["amber"], 0.36,
             data["aqi_num"], 300),
            ("TRANSIT", f"{data['buses']} buses active",
             f"Congestion: {data['congestion']}", NEON["pink"], 0.44,
             data["cong_num"], 100),
            ("SEISMIC", f"{data['quakes']} quakes / 24h",
             f"Max: {data['quake_max']}", NEON["purple"], 0.52,
             int(data["quakes"]) if data["quakes"].isdigit() else 0, 20),
        ]

        px_left = int(w * 0.08)
        py_start = int(h * 0.34)
        pw = int(w * 0.39)
        ph = int(h * 0.13)
        col_gap = int(w * 0.06)
        row_gap = int(h * 0.15)

        for idx, (label, line1, line2, color, start_t, gauge_val, gauge_max) in enumerate(panels):
            pp = _ease_out(min(max(t - start_t, 0) / 0.15, 1.0))
            if pp <= 0:
                continue

            col = idx % 2
            row = idx // 2
            bx = px_left + col * (pw + col_gap)
            by = py_start + row * row_gap

            # Panel background — subtle fill
            bg_c = _dim(color, 0.04 * pp)
            draw.rectangle([bx + 1, by + 1, bx + pw - 1, by + ph - 1], fill=bg_c)

            # Panel border with breathing brightness
            border_b = _breathing(frame / fps, freq=0.3 + idx * 0.1, lo=0.5, hi=1.0)
            bc = _dim(color, pp * border_b)
            draw.rectangle([bx, by, bx + pw, by + ph], outline=bc, width=1)

            # Corner accents — L-shaped brackets
            cl = int(pw * 0.08)
            for cx, cy, ddx, ddy in [
                (bx, by, 1, 1), (bx + pw, by, -1, 1),
                (bx, by + ph, 1, -1), (bx + pw, by + ph, -1, -1),
            ]:
                draw.line([(cx, cy), (cx + cl * ddx, cy)], fill=bc, width=2)
                draw.line([(cx, cy), (cx, cy + cl * ddy)], fill=bc, width=2)

            # Label
            draw.text((bx + 8, by + 3),
                      _typewriter(label, pp), font=fonts["small"], fill=color)

            # Data lines
            text_p = min(max((pp - 0.3) / 0.7, 0), 1.0)
            if text_p > 0:
                dy1 = by + int(ph * 0.38)
                dy2 = by + int(ph * 0.68)
                draw.text((bx + 8, dy1),
                          _typewriter(line1, text_p),
                          font=fonts["body"], fill=NEON["white"])
                if line2:
                    draw.text((bx + 8, dy2),
                              _typewriter(line2, text_p),
                              font=fonts["small"], fill=NEON["dim"])

            # ── Animated arc gauge (right side of panel) ──
            if gauge_max > 0 and text_p > 0:
                gauge_r = int(ph * 0.30)
                gauge_cx = bx + pw - gauge_r - int(pw * 0.06)
                gauge_cy = by + int(ph * 0.55)
                gauge_frac = min(gauge_val / gauge_max, 1.0) * text_p
                self._draw_arc_gauge(draw, gauge_cx, gauge_cy, gauge_r,
                                     gauge_frac, color, NEON["dim"])

        # ── Corner brackets (outer frame) ──
        bf = min(t / 0.2, 1.0)
        if bf > 0:
            bl = int(min(w, h) * 0.04)
            bc = _dim(NEON["dim"], bf * _breathing(frame / fps, freq=0.15))
            m = int(min(w, h) * 0.025)
            for cx, cy, ddx, ddy in [
                (m, m, 1, 1), (w - m, m, -1, 1),
                (m, h - m, 1, -1), (w - m, h - m, -1, -1),
            ]:
                draw.line([(cx, cy), (cx + bl * ddx, cy)], fill=bc, width=2)
                draw.line([(cx, cy), (cx, cy + bl * ddy)], fill=bc, width=2)

        # ── Scrolling ticker ──
        ticker_p = max(t - 0.55, 0) / 0.45
        if ticker_p > 0:
            ty = h - int(h * 0.055)
            ticker = (f"   DATA MEDIA MACHINE v3.4  ·  LIVE  ·  "
                      f"{data['city']}  ·  {data['time_str']}  ·  "
                      f"{data['temp']}  ·  AQI {data['aqi']}   ")
            offset = int(frame * 2.5) % max(len(ticker) * 10, 1)
            tc = _dim(NEON["dim"], ticker_p * 0.8)
            draw.text((w - offset, ty), ticker * 5,
                      font=fonts["small"], fill=tc)
            # Thin divider line above ticker
            draw.line([(0, ty - 3), (w, ty - 3)],
                      fill=_dim(NEON["dim"], ticker_p * 0.3), width=1)

        # ── Fractal border filigree (Mandelbrot accent) ──
        fractal_fade = _ease_out(min(max(t - 0.15, 0) / 0.3, 1.0))
        if fractal_fade > 0 and fractal_top is not None:
            # Top edge — decorative fractal ribbon
            for x_norm, escape in fractal_top:
                if escape > 0:
                    px = int(x_norm * w)
                    bright = int(escape * 35 * fractal_fade)
                    # Breathing fractal color
                    frac_c = _dim(NEON["purple"], escape * fractal_fade *
                                  _breathing(time_sec, freq=0.2, lo=0.5, hi=1.0))
                    y_off = int(h * 0.015)
                    for dy in range(max(1, int(escape * 4))):
                        if y_off + dy < h:
                            draw.point((px, y_off + dy), fill=frac_c)
            # Bottom edge — mirrored
            if fractal_bot is not None:
                for x_norm, escape in fractal_bot:
                    if escape > 0:
                        px = int(x_norm * w)
                        frac_c = _dim(NEON["pink"], escape * fractal_fade *
                                      _breathing(time_sec, freq=0.2, lo=0.5, hi=1.0))
                        y_off = h - int(h * 0.07)
                        for dy in range(max(1, int(escape * 4))):
                            if y_off - dy >= 0:
                                draw.point((px, y_off - dy), fill=frac_c)

        # ── Post-processing (numpy) ──
        img_arr = np.array(img)
        img_arr = self._apply_vignette(img_arr, strength=0.45)
        img_arr = self._apply_chromatic_aberration(img_arr,
                                                    offset=max(1, int(2 * w / 768)))
        img_arr = self._apply_film_grain(img_arr, intensity=8, frame_idx=frame)
        img_arr = self._apply_bloom(img_arr, threshold=180, radius=6,
                                     intensity=0.2)

        return Image.fromarray(img_arr)

    # ── Arc gauge helper ────────────────────────────────────────────
    def _draw_arc_gauge(self, draw, cx, cy, r, frac, color, bg_color):
        """Draw a small arc gauge (180° sweep) showing a fraction 0-1."""
        # Background arc
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, 180, 360, fill=_dim(bg_color, 0.3), width=2)
        # Filled arc
        if frac > 0.01:
            end_angle = 180 + int(180 * frac)
            # Color shifts from cool→warm as value increases
            arc_color = _lerp_color(color, NEON["hot"], frac) if frac > 0.6 else color
            draw.arc(bbox, 180, end_angle, fill=arc_color, width=2)

    # ── AQI label helper ────────────────────────────────────────────
    def _aqi_label(self, aqi_num):
        """Human-readable AQI category."""
        if aqi_num <= 0:
            return ""
        elif aqi_num <= 50:
            return "Good"
        elif aqi_num <= 100:
            return "Moderate"
        elif aqi_num <= 150:
            return "Unhealthy (Sensitive)"
        elif aqi_num <= 200:
            return "Unhealthy"
        else:
            return "Hazardous"

    # ── Palm tree helper ────────────────────────────────────────────
    def _draw_palm(self, draw, x, top_y, h, fade, sway=0):
        """Stylized palm tree silhouette with wind sway."""
        color = _dim(NEON["palm"], fade)
        trunk_bot = h - int(h * 0.07)

        # Trunk with sway
        for y in range(top_y, trunk_bot, 2):
            progress = (y - top_y) / max(trunk_bot - top_y, 1)
            off = int((3 + sway * (1 - progress)) * math.sin(progress * 1.5))
            width = max(1, int(3 - progress * 1.5))
            draw.line([(x + off - width, y), (x + off + width, y)], fill=color)

        # Fronds with droop and sway
        frond_len = int(h * 0.10)
        for i in range(7):
            angle = math.radians(-130 + i * (260 / 6))
            sway_angle = angle + math.radians(sway)
            ex = x + int(frond_len * math.cos(sway_angle))
            ey = top_y + int(frond_len * 0.5 * math.sin(sway_angle))
            draw.line([(x, top_y), (ex, ey)], fill=color, width=2)
            # Droop
            ex2 = ex + int(frond_len * 0.3 * math.cos(sway_angle))
            ey2 = ey + int(frond_len * 0.25)
            draw.line([(ex, ey), (ex2, ey2)], fill=color, width=1)

    # ══════════════════════════════════════════════════════════════════
    #  MINIMAL style — enhanced
    # ══════════════════════════════════════════════════════════════════
    def _minimal(self, w, h, t, frame, fps, data, fonts, grain_seed):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), (12, 12, 18))
        draw = ImageDraw.Draw(img)

        # Subtle radial gradient background (vectorized — no Python loops)
        yc, xc = np.ogrid[:h, :w]
        dist = np.sqrt((xc - w / 2.0) ** 2 + (yc - h / 2.0) ** 2)
        max_dist = math.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
        ratio = dist / max_dist
        c_val = np.clip(18 - ratio * 8, 0, 255).astype(np.uint8)
        bg_arr = np.stack([c_val, c_val, np.clip(c_val + 3, 0, 255).astype(np.uint8)], axis=2)
        img = Image.fromarray(bg_arr)
        draw = ImageDraw.Draw(img)

        # Breathing horizontal rule
        ry = int(h * 0.45)
        rule_breath = _breathing(frame / fps, freq=0.25, lo=0.5, hi=0.8)
        rule_w = int(w * 0.6 * _ease_out(min(t / 0.3, 1.0)))
        rx = (w - rule_w) // 2
        rule_c = _dim((80, 80, 110), rule_breath)
        draw.line([(rx, ry), (rx + rule_w, ry)], fill=rule_c, width=1)
        # Thin accent line below
        draw.line([(rx + int(rule_w * 0.2), ry + 4),
                   (rx + int(rule_w * 0.8), ry + 4)],
                  fill=_dim(rule_c, 0.4), width=1)

        lines = [
            (data["city"], fonts["title"], (200, 200, 210), 0.10),
            (f"{data['period']}  ·  {data['time_str']}", fonts["heading"],
             (120, 120, 140), 0.20),
            ("", None, None, 0),
            (f"{data['temp']}  ·  {data['conditions']}  ·  Wind {data['wind']}",
             fonts["body"], (160, 160, 170), 0.30),
            (f"AQI {data['aqi']}  ·  {data['quakes']} quakes  ·  Max {data['quake_max']}",
             fonts["body"], (160, 160, 170), 0.40),
            (f"{data['buses']} buses  ·  {data['congestion']} congestion",
             fonts["body"], (160, 160, 170), 0.50),
        ]

        y = int(h * 0.15)
        for text, font, color, start_t in lines:
            if font is None:
                y += int(h * 0.06)
                continue
            p = _ease_out(min(max(t - start_t, 0) / 0.2, 1.0))
            if p <= 0:
                y += int(h * 0.09)
                continue
            display = _typewriter(text, p)
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            x = (w - tw) // 2
            # Fade-in via alpha simulation
            fc = _dim(color, p)
            draw.text((x, y), display, font=font, fill=fc)
            y += bbox[3] - bbox[1] + int(h * 0.025)

        # DMM watermark with breathing
        wm_p = max(t - 0.6, 0) / 0.4
        if wm_p > 0:
            wm = "DATA MEDIA MACHINE"
            wm_breath = _breathing(frame / fps, freq=0.2, lo=0.3, hi=0.6)
            draw.text((int(w * 0.05), h - int(h * 0.06)),
                      wm, font=fonts["small"],
                      fill=_dim((60, 60, 80), wm_p * wm_breath))

        # Post-processing
        img_arr = np.array(img)
        img_arr = self._apply_vignette(img_arr, strength=0.3)
        img_arr = self._apply_film_grain(img_arr, intensity=5, frame_idx=frame)

        return Image.fromarray(img_arr)

    # ══════════════════════════════════════════════════════════════════
    #  RETRO TERMINAL style — CRT enhanced
    # ══════════════════════════════════════════════════════════════════
    def _retro(self, w, h, t, frame, fps, data, fonts, ca=None):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), (0, 4, 0))
        draw = ImageDraw.Draw(img)

        green = (0, 200, 0)
        dim_g = (0, 70, 0)
        bright_g = (0, 255, 0)

        # ── Cellular automata underlay (Matrix rain effect) ──
        if ca is not None:
            ca_fade = _ease_out(min(t / 0.4, 1.0)) * 0.5
            if ca_fade > 0:
                ca_overlay = ca.render(w, h, green, intensity=0.03 * ca_fade)
                img_arr = np.array(img, dtype=np.float32)
                img_arr += ca_overlay
                img = Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)

        # CRT scan lines with flicker (vectorized)
        flicker = _breathing(frame / fps, freq=30, lo=0.85, hi=1.0)
        c = int(8 * flicker)
        img_arr_crt = np.array(img)
        img_arr_crt[::2, :, 1] = np.clip(
            img_arr_crt[::2, :, 1].astype(np.int16) + c, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr_crt)
        draw = ImageDraw.Draw(img)

        # Phosphor glow line (horizontal sweep)
        sweep_y = int((frame / fps * 0.3) % 1.0 * h)
        for dy in range(-6, 7):
            sy = sweep_y + dy
            if 0 <= sy < h:
                intensity = max(0, 6 - abs(dy))
                draw.line([(0, sy), (w, sy)],
                          fill=(0, intensity * 3, 0))

        lines = [
            f"> SYSTEM: DATA MEDIA MACHINE v3.4",
            f"> LOCATION: {data['city']}",
            f"> TIME: {data['period']} {data['time_str']}",
            f"> {'─' * 36}",
            f"> WEATHER: {data['temp']} {data['conditions']}",
            f">   WIND: {data['wind']}",
            f"> AIR QUALITY: AQI {data['aqi']}",
            f"> SEISMIC: {data['quakes']} EVENTS / MAX {data['quake_max']}",
            f"> TRANSIT: {data['buses']} BUSES / {data['congestion']} CONG",
            f"> {'─' * 36}",
            f"> STATUS: ALL SYSTEMS NOMINAL",
        ]

        total_chars = sum(len(ln) for ln in lines)
        chars_shown = int(t * total_chars * 2.5)
        char_count = 0
        y = int(h * 0.06)
        line_h = int(h * 0.075)

        for ln in lines:
            remaining = chars_shown - char_count
            if remaining <= 0:
                break
            display = ln[:max(0, remaining)]
            if "─" in ln:
                color = dim_g
            elif "STATUS" in ln:
                color = bright_g
            else:
                color = green
            # Slight jitter for authenticity
            jitter_x = int(math.sin(frame / fps * 20 + y * 0.1) * 0.5)
            draw.text((int(w * 0.04) + jitter_x, y), display,
                      font=fonts["body"], fill=color)
            y += line_h
            char_count += len(ln)

        # Blinking cursor with phosphor trail
        if chars_shown >= total_chars:
            if frame % fps < fps // 2:
                draw.text((int(w * 0.04), y), "> █",
                          font=fonts["body"], fill=bright_g)
            else:
                draw.text((int(w * 0.04), y), "> _",
                          font=fonts["body"], fill=dim_g)

        # CRT curvature vignette (vectorized — darken edges)
        img_arr_v = np.array(img, dtype=np.float32)
        edge_w = int(w * 0.06)
        edge_h = int(h * 0.04)
        if edge_w > 0:
            fade_lr = np.linspace(0, 1, edge_w, dtype=np.float32)
            img_arr_v[:, :edge_w, :] *= fade_lr[np.newaxis, :, np.newaxis]
            img_arr_v[:, -edge_w:, :] *= fade_lr[np.newaxis, ::-1, np.newaxis]
        if edge_h > 0:
            fade_tb = np.linspace(0, 1, edge_h, dtype=np.float32)
            img_arr_v[:edge_h, :, :] *= fade_tb[:, np.newaxis, np.newaxis]
            img_arr_v[-edge_h:, :, :] *= fade_tb[::-1, np.newaxis, np.newaxis]
        img = Image.fromarray(np.clip(img_arr_v, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Post-processing — phosphor bloom + grain
        img_arr = np.array(img)
        img_arr = self._apply_bloom(img_arr, threshold=150, radius=4,
                                     intensity=0.25)
        img_arr = self._apply_film_grain(img_arr, intensity=6,
                                          frame_idx=frame)

        return Image.fromarray(img_arr)
