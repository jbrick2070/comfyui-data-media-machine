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

from .dmm_sun_utils import get_sun_info, get_nice_statement, get_moon_info

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


# ── Data-driven atmosphere ────────────────────────────────────────────
# Builds palette overrides + visual effect flags from parsed conditions.
# Controls: bg gradient, particle tint, glow color, weather FX overlays.

def _solar_elevation(data):
    """Return sun elevation as a 0.0–1.0 fraction of its arc above the horizon.

    1.0 = solar noon (maximum brightness), 0.0 = on the horizon (sunrise/sunset).
    Returns -1.0 when the sun is below the horizon (night).

    Uses actual sunrise/sunset from parsed data so the palette tracks the real sky.
    """
    import math
    from datetime import datetime

    now = datetime.now()
    cur = now.hour * 60 + now.minute

    def _parse(s):
        try:
            t = datetime.strptime(s.strip(), "%I:%M %p")
            return t.hour * 60 + t.minute
        except Exception:
            return None

    sr = _parse(data.get("sunrise", ""))
    ss = _parse(data.get("sunset", ""))

    if not sr or not ss or ss <= sr:
        # Fallback: rough elevation from hour alone
        h = now.hour
        if 6 <= h < 18:
            return max(0.0, math.sin(math.pi * (h - 6) / 12))
        return -1.0

    if cur <= sr or cur >= ss:
        return -1.0  # below horizon

    # Sine arc: 0 at sunrise, peaks at solar noon, 0 at sunset
    day_frac = (cur - sr) / (ss - sr)
    return math.sin(math.pi * day_frac)


def _solar_palette(elevation, moon_illum=0):
    """Return base palette dict driven by solar elevation (–1 to 1).

    Blends continuously from deep night → dawn → golden → midday blue
    so the background visually matches the actual sky brightness outside.
    """
    import math

    if elevation < 0:
        # Night — moon illumination adds a faint blue-white ambient
        moon_lift = int(moon_illum * 0.06)   # max +6 brightness at full moon
        return {
            "bg_top":  (4 + moon_lift,  4 + moon_lift,  18 + moon_lift * 2),
            "bg_bot":  (8 + moon_lift,  8 + moon_lift,  26 + moon_lift * 2),
            "accent":  (80, 60, 200), "glow": (40, 20, 120),
            "sky_brightness": 0.03 + moon_illum * 0.0006,
        }

    if elevation < 0.15:
        # Golden hour — sunrise / sunset, warm amber sky, clearly brighter than night
        t = elevation / 0.15
        return {
            "bg_top":  (int(45 + t*50), int(25 + t*30), int(30 + t*35)),
            "bg_bot":  (int(75 + t*45), int(45 + t*35), int(25 + t*30)),
            "accent":  (255, int(140 + t*40), int(50 + t*30)),
            "glow":    (220, int(80 + t*30), int(30 + t*20)),
            "sky_brightness": 0.15 + t * 0.20,
        }

    if elevation < 0.45:
        # Morning / late afternoon — warm golden shifting to blue, medium-bright
        t = (elevation - 0.15) / 0.30
        return {
            "bg_top":  (int(70 + t*20), int(60 + t*30), int(80 + t*40)),
            "bg_bot":  (int(100 + t*10), int(80 + t*25), int(60 + t*40)),
            "accent":  (255, int(180 + t*30), int(80 + t*30)),
            "glow":    (220, int(150 + t*25), int(50 + t*25)),
            "sky_brightness": 0.40 + t * 0.20,
        }

    # Near midday — bright blue sky
    t = min(1.0, (elevation - 0.45) / 0.55)
    return {
        "bg_top":  (int(80 + t*15), int(95 + t*20), int(130 + t*20)),
        "bg_bot":  (int(100 + t*10), int(105 + t*15), int(110 + t*15)),
        "accent":  (255, int(220 + t*15), int(100 + t*15)),
        "glow":    (220, int(185 + t*10), int(75 + t*10)),
        "sky_brightness": 0.60 + t * 0.15,
    }


def _build_atmosphere(data):
    """Build atmosphere dict from parsed narrative data.

    Returns dict with palette overrides and effect flags:
        bg_top, bg_bot   — RGB tuples for vertical gradient
        accent           — primary neon accent color
        glow             — secondary glow color
        particles_tint   — tint multiplier for particles
        fx_rain          — bool: draw falling rain streaks
        fx_fog           — bool: draw fog/mist overlay
        fx_clouds        — bool: draw cloud band layer
        fx_lightning      — bool: occasional flash frames
        fx_heat          — bool: heat shimmer distortion
        fx_wind_streaks  — bool: horizontal wind lines
        sky_brightness   — 0.0 (dark) to 1.0 (bright daytime)
        moon             — moon_info dict or None
    """
    cond = data.get("conditions", "").lower()
    temp = data.get("temp_num", 72)
    wind = data.get("wind_num", 5)

    moon = get_moon_info()
    moon_illum = moon.get("illumination_pct", 0) if moon else 0

    # ── Base palette driven by actual solar elevation ─────────────
    elev = _solar_elevation(data)
    pal = _solar_palette(elev, moon_illum)

    atm = {
        "bg_top": pal["bg_top"],
        "bg_bot": pal["bg_bot"],
        "accent": pal["accent"],
        "glow": pal["glow"],
        "particles_tint": 1.0,
        "fx_rain": False,
        "fx_fog": False,
        "fx_clouds": False,
        "fx_lightning": False,
        "fx_heat": False,
        "fx_wind_streaks": False,
        "sky_brightness": pal["sky_brightness"],
        "moon": moon,
        "solar_elevation": elev,
    }

    # ── Weather condition overrides ─────────────────────────────
    rain_mm = data.get("rain_1h_mm", 0) or 0
    if rain_mm > 0.1:
        # Only show rain FX when there's actual measured precipitation
        atm["fx_rain"] = True
        atm["fx_clouds"] = True
        # Desaturate and cool the palette
        atm["bg_top"] = (10, 12, 22)
        atm["bg_bot"] = (18, 20, 32)
        atm["accent"] = (80, 120, 180)
        atm["glow"] = (40, 80, 140)
        atm["particles_tint"] = 0.5
        atm["sky_brightness"] = min(atm["sky_brightness"], 0.25)

    if "thunder" in cond or "storm" in cond:
        atm["fx_rain"] = True
        atm["fx_lightning"] = True
        atm["fx_clouds"] = True
        atm["bg_top"] = (8, 6, 18)
        atm["bg_bot"] = (14, 12, 26)
        atm["accent"] = (180, 160, 255)
        atm["glow"] = (120, 100, 200)
        atm["sky_brightness"] = min(atm["sky_brightness"], 0.15)

    if "fog" in cond or "mist" in cond or "haz" in cond:
        atm["fx_fog"] = True
        atm["bg_top"] = (20, 22, 28)
        atm["bg_bot"] = (30, 32, 38)
        atm["accent"] = (120, 130, 150)
        atm["glow"] = (80, 90, 110)
        atm["particles_tint"] = 0.3
        atm["sky_brightness"] = min(atm["sky_brightness"], 0.2)

    if "cloud" in cond or "overcast" in cond:
        atm["fx_clouds"] = True
        # Slightly muted palette
        r, g, b = atm["bg_top"]
        atm["bg_top"] = (min(r + 5, 40), min(g + 5, 40), min(b + 8, 50))
        atm["sky_brightness"] = min(atm["sky_brightness"], 0.35)

    if temp > 95:
        atm["fx_heat"] = True

    if wind > 20:
        atm["fx_wind_streaks"] = True

    return atm


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
                "duration_sec": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 120.0, "step": 0.5,
                                           "tooltip": "Clip length in seconds"}),
                "fps": ("INT", {"default": 35, "min": 1, "max": 60}),
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
        atm = _build_atmosphere(data)

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
                                    ca, fractal_top, fractal_bot, atm=atm)
            elif style == "minimal_data":
                img = self._minimal(width, height, t, f, fps, data, fonts,
                                    grain_seed, atm=atm)
            else:
                img = self._retro(width, height, t, f, fps, data, fonts,
                                  ca=ca, atm=atm)

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
        # Fetch live sun info for the HUD (date, sunrise, sunset)
        sun = get_sun_info()

        d = {
            "city": "LOS ANGELES", "time_str": "", "period": "",
            "temp": "—", "temp_num": 0, "conditions": "—", "wind": "—",
            "wind_num": 0, "quakes": "0", "quake_max": "—",
            "aqi": "—", "aqi_num": 0,
            "buses": "—", "congestion": "—", "cong_num": 0,
            # Sun / date / statement fields for HUD
            "date_short": sun["date_short"],
            "date_str": sun["date_str"],
            "sunrise": sun["sunrise_str"],
            "sunset": sun["sunset_str"],
            "next_event": sun["next_event"],
            "next_time": sun["next_time_str"],
            "nice_statement": "",  # set after parsing, so it uses real data
            # Moon data
            "moon_phase": "",
            "moon_emoji": "",
            "moon_illum": 0,
            "moon_eclipse": "",
            "moon_full": False,
            "moon_new": False,
            "moon_supermoon": False,
        }

        # City: extract from narration if present (default is LOS ANGELES).
        # Only accept 2-4 word alpha strings before a comma at the start —
        # prevents greedy matches from grabbing non-city text.
        m = re.match(r"^([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,3})\s*,", text)
        if m:
            d["city"] = m.group(1).strip().upper()

        # Time: 4:39 PM
        m = re.search(r"(\d{1,2}:\d{2}\s*[AP]M)", text, re.IGNORECASE)
        if m:
            d["time_str"] = m.group(1).upper()

        # Period: Thursday afternoon
        m = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
                       r"\s+(morning|afternoon|evening|night)",
                       text, re.IGNORECASE)
        if m:
            d["period"] = f"{m.group(1).capitalize()} {m.group(2).capitalize()}"

        # Temperature: "84 degrees", "84°F", "84 F"
        m = re.search(r"(\d+)\s*(?:°|degrees?|deg)?\s*[FC]", text, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+)\s*(?:°|degrees?|deg)", text, re.IGNORECASE)
        if m:
            d["temp"] = m.group(1) + "°F"
            d["temp_num"] = int(m.group(1))

        # Conditions: Clear, Cloudy, etc.
        m = re.search(r"(?:Clear|Cloudy|Overcast|Rain|Fog|Partly\s*Cloudy|Hazy|Windy|Fair)",
                       text, re.IGNORECASE)
        if m:
            d["conditions"] = m.group(0).title()

        # Wind: "winds 7 mph", "7 mph winds", "wind at 7"
        m = re.search(r"(?:winds?|blowing|gusts?).*?(\d+)\s*mph", text, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+)\s*mph\s*winds?", text, re.IGNORECASE)
        if m:
            d["wind"] = m.group(1) + " mph"
            d["wind_num"] = int(m.group(1))

        # Quakes: "3 quakes", "recorded 3", "3 seismic events"
        m = re.search(r"(\d+)\s*(?:quakes?|earthquakes?|tremors?|seismic events?)", text, re.IGNORECASE)
        if m:
            d["quakes"] = m.group(1)

        # Max Magnitude: "max m1.3", "magnitude 1.3"
        m = re.search(r"(?:max|magnitude).*?m?(\d+\.?\d*)", text, re.IGNORECASE)
        if m:
            d["quake_max"] = "M" + m.group(1)

        # AQI: "AQI 84", "AQI is 84", "air quality index: 84"
        m = re.search(r"(?:aqi|air quality index).*?(\d+)", text, re.IGNORECASE)
        if m:
            d["aqi"] = m.group(1)
            d["aqi_num"] = int(m.group(1))

        # Congestion: "70% congestion", "congestion is at 70%"
        m = re.search(r"(\d+)\s*%?\s*(?:congestion|capacity|traffic volume)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"(?:congestion|capacity).*?(\d+)\s*%?", text, re.IGNORECASE)
        if m:
            d["congestion"] = m.group(1) + "%"
            d["cong_num"] = int(m.group(1))

        # Buses: "20 buses", "20 metro buses", "20 active units"
        m = re.search(r"(\d+)\s*(?:metro\s*)?buses?", text, re.IGNORECASE)
        if not m:
             m = re.search(r"(\d+)\s*(?:active\s*)?units", text, re.IGNORECASE)
        if m:
            d["buses"] = m.group(1)

        # Rain amount — parse from summary ("Rain 0.0mm" or "Rain 2.3mm")
        # Never guess rain from condition name alone; require actual measurement.
        rain_mm = 0.0
        m_rain = re.search(r"Rain\s+([\d.]+)\s*mm", text, re.IGNORECASE)
        if m_rain:
            rain_mm = float(m_rain.group(1))
        d["rain_1h_mm"] = rain_mm

        # Now generate the nice statement using the REAL parsed data
        w_dict = {
            "temp_f": d["temp_num"] if d["temp_num"] else 72,
            "wind_speed_mph": d["wind_num"] if d["wind_num"] else 5,
            "humidity": 50,  # not in narration text — use neutral default
            "rain_1h_mm": rain_mm,
            "description": d["conditions"],
        }
        aq_dict = {
            "us_aqi": d["aqi_num"] if d["aqi_num"] else 50,
            "uv_index": 3,  # not in narration text — use neutral default
        }
        d["nice_statement"] = get_nice_statement(w_dict, aq_dict, sun)

        # Moon data
        moon = get_moon_info()
        d["moon_phase"] = moon["phase_name"]
        d["moon_emoji"] = moon["phase_emoji"]
        d["moon_illum"] = moon["illumination"]
        d["moon_eclipse"] = moon["next_eclipse"]
        d["moon_full"] = moon["is_full"]
        d["moon_new"] = moon["is_new"]
        d["moon_supermoon"] = moon["is_supermoon"]

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
                 grain_seed, ca=None, fractal_top=None, fractal_bot=None,
                 atm=None):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (w, h), NEON["bg"])
        draw = ImageDraw.Draw(img)

        # ── BG gradient (data-driven: time-of-day + weather palette) ──
        bg_top = atm["bg_top"] if atm else (8, 6, 22)
        bg_bot = atm["bg_bot"] if atm else (14, 14, 40)
        ratio = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        r_ch = np.clip(bg_top[0] + ratio * (bg_bot[0] - bg_top[0]), 0, 255).astype(np.uint8)
        g_ch = np.clip(bg_top[1] + ratio * (bg_bot[1] - bg_top[1]), 0, 255).astype(np.uint8)
        b_ch = np.clip(bg_top[2] + ratio * (bg_bot[2] - bg_top[2]), 0, 255).astype(np.uint8)
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
                # Tint the CA layer with the solar-driven accent color
                ca_color = atm["accent"] if atm else NEON["cyan"]
                ca_overlay = ca.render(w, h, ca_color,
                                       intensity=0.04 * ca_fade)
                # Blend into image via numpy
                img_arr = np.array(img, dtype=np.float32)
                img_arr += ca_overlay
                img_arr = np.clip(img_arr, 0, 255)
                img = Image.fromarray(img_arr.astype(np.uint8))
                draw = ImageDraw.Draw(img)

        # ── Weather FX overlays (data-driven) ──────────────────────
        time_sec = frame / fps
        if atm:
            # Cloud bands — translucent horizontal layers drifting slowly
            if atm.get("fx_clouds"):
                img_arr_c = np.array(img, dtype=np.float32)
                for ci in range(3):
                    cy_base = int(h * (0.08 + ci * 0.12))
                    band_h = int(h * 0.06)
                    drift = int(time_sec * (8 + ci * 3)) % w
                    for by in range(max(0, cy_base), min(h, cy_base + band_h)):
                        band_t = (by - cy_base) / max(band_h, 1)
                        # Bell-curve opacity — thick in center, fade at edges
                        opacity = 0.08 * math.exp(-((band_t - 0.5) ** 2) / 0.08)
                        # Noise-modulated density
                        nx = _noise_val(by + ci * 100, drift + ci * 50) * 0.06
                        alpha = min(opacity + nx, 0.15)
                        cloud_c = np.array([60, 65, 80], dtype=np.float32)
                        img_arr_c[by, :, :] = img_arr_c[by, :, :] * (1 - alpha) + cloud_c * alpha
                img = Image.fromarray(np.clip(img_arr_c, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)

            # Fog / mist overlay — full-screen translucent haze
            if atm.get("fx_fog"):
                img_arr_f = np.array(img, dtype=np.float32)
                # Fog thicker at bottom, thinner at top
                fog_grad = np.linspace(0.02, 0.12, h, dtype=np.float32).reshape(-1, 1)
                # Gentle noise modulation
                fog_noise = _breathing(time_sec, freq=0.08, lo=0.8, hi=1.0)
                fog_c = np.array([45, 50, 55], dtype=np.float32)
                for c_idx in range(3):
                    img_arr_f[:, :, c_idx] = (
                        img_arr_f[:, :, c_idx] * (1 - fog_grad * fog_noise)
                        + fog_c[c_idx] * fog_grad * fog_noise
                    )
                img = Image.fromarray(np.clip(img_arr_f, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)

            # Rain streaks — diagonal falling lines
            if atm.get("fx_rain"):
                rain_rng = np.random.RandomState(seed=(frame * 3) & 0x7fffffff)
                n_drops = int(w * 0.08)  # density scales with width
                for _ in range(n_drops):
                    rx = rain_rng.randint(0, w)
                    ry = rain_rng.randint(0, h)
                    length = rain_rng.randint(int(h * 0.03), int(h * 0.08))
                    bright = rain_rng.randint(30, 80)
                    # Diagonal streak (falling slightly left due to wind)
                    x2 = rx - int(length * 0.15)
                    y2 = min(h - 1, ry + length)
                    draw.line([(rx, ry), (x2, y2)],
                              fill=(bright, bright, bright + 20), width=1)

            # Lightning flash — occasional bright frame
            if atm.get("fx_lightning"):
                # Flash every ~4 seconds, lasting 2 frames
                flash_cycle = int(time_sec * fps) % (fps * 4)
                if flash_cycle < 2:
                    img_arr_l = np.array(img, dtype=np.float32)
                    flash_intensity = 0.15 if flash_cycle == 0 else 0.06
                    img_arr_l += flash_intensity * 255
                    img = Image.fromarray(np.clip(img_arr_l, 0, 255).astype(np.uint8))
                    draw = ImageDraw.Draw(img)

            # Wind streaks — horizontal motion blur lines
            if atm.get("fx_wind_streaks"):
                wind_rng = np.random.RandomState(seed=(frame * 7 + 13) & 0x7fffffff)
                for _ in range(int(w * 0.02)):
                    wy = wind_rng.randint(0, h)
                    wx = wind_rng.randint(0, w)
                    wlen = wind_rng.randint(int(w * 0.05), int(w * 0.15))
                    bright = wind_rng.randint(15, 35)
                    draw.line([(wx, wy), (min(w - 1, wx + wlen), wy)],
                              fill=(bright, bright, bright + 5), width=1)

            # Heat shimmer — subtle vertical wave distortion at bottom
            if atm.get("fx_heat"):
                img_arr_h = np.array(img)
                heat_zone = int(h * 0.7)  # bottom 30%
                for hy in range(heat_zone, h):
                    shift = int(math.sin(hy * 0.15 + time_sec * 3) * 2)
                    if shift > 0:
                        img_arr_h[hy, shift:, :] = img_arr_h[hy, :-shift, :]
                    elif shift < 0:
                        img_arr_h[hy, :shift, :] = img_arr_h[hy, -shift:, :]
                img = Image.fromarray(img_arr_h)
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
        if grid_fade > 0:
            # Grid tinted by the solar-driven accent (amber at golden hour, blue at night)
            gr, gg, gb = (atm["accent"] if atm else NEON["cyan"])
            horizon = int(h * 0.6)
            # Horizontal lines with FBM noise warp
            for i in range(16):
                p = i / 16.0
                y = horizon + int((h - horizon) * p * p)
                # FBM noise adds organic undulation to each grid line
                noise_warp = _fbm(i * 0.5, time_sec * 0.4, octaves=3) * 6 - 3
                wave = math.sin(time_sec * 1.5 + i * 0.4) * 3 + noise_warp
                bright = grid_fade * grid_pulse * (0.12 + 0.08 * math.sin(time_sec * 2 + i))
                col = (int(gr * bright), int(gg * bright), int(gb * bright))
                draw.line([(0, int(y + wave)), (w, int(y - wave))], fill=col)
            # Vertical converging lines with subtle sway
            cx = w // 2
            for i in range(20):
                x_bot = int(w * i / 20)
                x_top = cx + int((x_bot - cx) * 0.25)
                # Noise-driven horizontal sway
                sway = _fbm(i * 0.3, time_sec * 0.2, octaves=2) * 4 - 2
                bright = grid_fade * grid_pulse * 0.09
                col = (int(gr * bright), int(gg * bright), int(gb * bright))
                draw.line([(int(x_top + sway), horizon), (x_bot, h)], fill=col)

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
            # Title glow color tracks the solar accent palette
            title_accent = atm["accent"] if atm else NEON["cyan"]
            title_glow   = atm["glow"]   if atm else NEON["pink"]
            _glow_text(draw, (tx, ty), display, fonts["title"],
                       title_accent, glow_radius=4, intensity=glow_i)
            # Underline gradient: glow → accent (shifts with time of day)
            line_w = int(tw * min(title_raw, 1.0))
            line_y = ty + bbox[3] - bbox[1] + int(6 * h / 512)
            for lx in range(line_w):
                lr = lx / max(line_w, 1)
                lc = _lerp_color(title_glow, title_accent, lr)
                draw.point((tx + lx, line_y), fill=lc)
                draw.point((tx + lx, line_y + 1), fill=lc)

        # ── Subtitle with spring pop-in (now includes date) ──
        sub_raw = min(max(t - 0.22, 0) / 0.15, 1.0)
        sub_spring = _spring_overshoot(sub_raw, damping=5.0, freq=8.0)
        if sub_raw > 0:
            parts = [x for x in [data["date_short"], data["period"], data["time_str"]] if x]
            sub = " · ".join(parts) if parts else ""
            if sub:
                display = _typewriter(sub, min(sub_raw * 1.3, 1.0))
                bbox = fonts["heading"].getbbox(sub)
                sw = bbox[2] - bbox[0]
                sh = bbox[3] - bbox[1]
                sx = (w - sw) // 2
                base_sy = int(h * 0.07) + int(58 * h / 512)
                spring_off = int((1.0 - sub_spring) * h * 0.02)
                sy = base_sy + spring_off
                # Dark backdrop behind date/time block for readability
                pad = int(h * 0.012)
                # Panel covers subtitle + sun bar + moon bar (about 55px at 512h)
                panel_h = int(h * 0.11)
                draw.rectangle(
                    [sx - pad, sy - pad, sx + sw + pad, sy + panel_h + pad],
                    fill=(0, 0, 0, 0))  # Solid won't work on RGB; use numpy below
                # Semi-transparent dark panel via numpy compositing
                img_arr_sub = np.array(img, dtype=np.float32)
                y0 = max(0, sy - pad)
                y1 = min(h, sy + panel_h + pad)
                x0 = max(0, sx - pad * 2)
                x1 = min(w, sx + sw + pad * 2)
                img_arr_sub[y0:y1, x0:x1, :] *= 0.35  # darken to 35% brightness
                img = Image.fromarray(np.clip(img_arr_sub, 0, 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)
                # Now draw the bright white text on the darkened panel
                draw.text((sx, sy), display, font=fonts["heading"],
                          fill=_dim((220, 220, 235), min(sub_raw, 1.0)))

        # ── Sunrise / Sunset + Moon bar (below subtitle) ──
        sun_raw = min(max(t - 0.26, 0) / 0.15, 1.0)
        if sun_raw > 0:
            sun_line = f"\u2600 {data['sunrise']}  \u2600\u2193 {data['sunset']}  \u25B6 next {data['next_event']} {data['next_time']}"
            sun_display = _typewriter(sun_line, min(sun_raw * 1.2, 1.0))
            sun_bbox = fonts["small"].getbbox(sun_line)
            sun_w = sun_bbox[2] - sun_bbox[0]
            sun_x = (w - sun_w) // 2
            sun_y = int(h * 0.07) + int(80 * h / 512)
            sun_color = _lerp_color(NEON["amber"], NEON["cyan"],
                                     _breathing(time_sec, freq=0.15, lo=0.0, hi=1.0))
            draw.text((sun_x, sun_y), sun_display, font=fonts["small"],
                      fill=_dim(sun_color, sun_raw * 0.85))

        # ── Moon phase bar (below sun bar) ──
        moon_raw = min(max(t - 0.30, 0) / 0.15, 1.0)
        if moon_raw > 0 and data.get("moon_phase"):
            moon_parts = [
                f"{data['moon_emoji']} {data['moon_phase']}",
                f"{data['moon_illum']}% illuminated",
            ]
            if data.get("moon_supermoon"):
                moon_parts.append("SUPERMOON")
            if data.get("moon_full"):
                moon_parts.append("FULL MOON TONIGHT")
            if data.get("moon_eclipse"):
                moon_parts.append(data["moon_eclipse"])
            moon_line = "  |  ".join(moon_parts)
            moon_display = _typewriter(moon_line, min(moon_raw * 1.2, 1.0))
            moon_bbox = fonts["small"].getbbox(moon_line)
            moon_w = moon_bbox[2] - moon_bbox[0]
            moon_x = (w - moon_w) // 2
            moon_y = int(h * 0.07) + int(94 * h / 512)
            # Moon color: silvery white with breathing
            moon_color = _lerp_color((180, 180, 200), (220, 210, 255),
                                      _breathing(time_sec, freq=0.12, lo=0.0, hi=1.0))
            draw.text((moon_x, moon_y), moon_display, font=fonts["small"],
                      fill=_dim(moon_color, moon_raw * 0.75))

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

        # ── Nice statement (data-based friendly message above ticker) ──
        stmt_p = _ease_out(min(max(t - 0.50, 0) / 0.15, 1.0))
        if stmt_p > 0 and data.get("nice_statement"):
            stmt_text = data["nice_statement"]
            max_text_w = int(w * 0.88)  # 88% of frame width
            # Word-wrap: split into lines that fit within max_text_w
            stmt_lines = []
            words = stmt_text.split()
            current_line = ""
            for word in words:
                test = (current_line + " " + word).strip()
                test_bbox = fonts["body"].getbbox(test)
                if test_bbox[2] - test_bbox[0] > max_text_w and current_line:
                    stmt_lines.append(current_line)
                    current_line = word
                else:
                    current_line = test
            if current_line:
                stmt_lines.append(current_line)
            # Limit to 2 lines max to avoid crowding the ticker area
            stmt_lines = stmt_lines[:2]
            stmt_y_base = h - int(h * 0.18) - int(h * 0.035) * (len(stmt_lines) - 1)
            stmt_color = _lerp_color(NEON["amber"], NEON["white"],
                                      _breathing(time_sec, freq=0.25, lo=0.3, hi=0.7))
            for li, line in enumerate(stmt_lines):
                line_display = _typewriter(line, min(stmt_p * 1.3, 1.0))
                line_bbox = fonts["body"].getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                line_x = (w - line_w) // 2
                line_y = stmt_y_base + int(li * h * 0.035)
                _glow_text(draw, (line_x, line_y), line_display, fonts["body"],
                           stmt_color, glow_radius=2, intensity=0.2 * stmt_p)

        # ── Scrolling ticker ──
        ticker_p = max(t - 0.55, 0) / 0.45
        if ticker_p > 0:
            ty = h - int(h * 0.055)
            moon_ticker = f"  ·  {data.get('moon_emoji','')} {data.get('moon_phase','')}" if data.get("moon_phase") else ""
            ticker = (f"   DATA MEDIA MACHINE v3.7  ·  LIVE  ·  "
                      f"{data['city']}  ·  {data['date_short']}  ·  {data['time_str']}  ·  "
                      f"{data['temp']}  ·  AQI {data['aqi']}  ·  "
                      f"\u2600{data['sunrise']}  \u2600\u2193{data['sunset']}"
                      f"{moon_ticker}   ")
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
        # CA offset: 1px at 512w, 2px at 1080+. Keeping it subtle to preserve text readability.
        ca_offset = 1 if w < 768 else 2
        img_arr = self._apply_chromatic_aberration(img_arr, offset=ca_offset)
        img_arr = self._apply_film_grain(img_arr, intensity=8, frame_idx=frame)
        img_arr = self._apply_bloom(img_arr, threshold=180, radius=6,
                                     intensity=0.2)

        return Image.fromarray(img_arr)

    # ── Arc gauge helper ────────────────────────────────────────────
    def _draw_arc_gauge(self, draw, cx, cy, r, frac, color, bg_color):
        """Draw a small arc gauge (180° sweep) showing a fraction 0-1."""
        # Background arc
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, 180, 360, fill=_dim(bg_color, 0.3), width=4)
        # Filled arc
        if frac > 0.01:
            end_angle = 180 + int(180 * frac)
            # Color shifts from cool→warm as value increases
            arc_color = _lerp_color(color, NEON["hot"], frac) if frac > 0.6 else color
            draw.arc(bbox, 180, end_angle, fill=arc_color, width=4)

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
    def _minimal(self, w, h, t, frame, fps, data, fonts, grain_seed, atm=None):
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
            (f"{data['date_short']}  ·  {data['period']}  ·  {data['time_str']}", fonts["heading"],
             (120, 120, 140), 0.18),
            (f"\u2600 {data['sunrise']}  ·  \u2600\u2193 {data['sunset']}  ·  next {data['next_event']} {data['next_time']}",
             fonts["small"], (180, 150, 80), 0.24),
            (f"{data.get('moon_emoji','')} {data.get('moon_phase','')}  ·  {data.get('moon_illum',0)}% illum"
             + (f"  ·  {data['moon_eclipse']}" if data.get('moon_eclipse') else ""),
             fonts["small"], (160, 160, 190), 0.28),
            ("", None, None, 0),
            (f"{data['temp']}  ·  {data['conditions']}  ·  Wind {data['wind']}",
             fonts["body"], (160, 160, 170), 0.32),
            (f"AQI {data['aqi']}  ·  {data['quakes']} quakes  ·  Max {data['quake_max']}",
             fonts["body"], (160, 160, 170), 0.40),
            (f"{data['buses']} buses  ·  {data['congestion']} congestion",
             fonts["body"], (160, 160, 170), 0.48),
            (data.get("nice_statement", ""), fonts["body"], (200, 180, 120), 0.56),
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
    def _retro(self, w, h, t, frame, fps, data, fonts, ca=None, atm=None):
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
            f"> SYSTEM: DATA MEDIA MACHINE v3.7",
            f"> LOCATION: {data['city']}",
            f"> DATE: {data['date_short']}",
            f"> TIME: {data['period']} {data['time_str']}",
            f"> SOLAR: RISE {data['sunrise']}  SET {data['sunset']}",
            f"> LUNAR: {data.get('moon_phase','?')} {data.get('moon_illum',0)}% ILLUM",
            f"> {'─' * 36}",
            f"> WEATHER: {data['temp']} {data['conditions']}",
            f">   WIND: {data['wind']}",
            f"> AIR QUALITY: AQI {data['aqi']}",
            f"> SEISMIC: {data['quakes']} EVENTS / MAX {data['quake_max']}",
            f"> TRANSIT: {data['buses']} BUSES / {data['congestion']} CONG",
            f"> {'─' * 36}",
            f"> {data.get('nice_statement', 'ALL SYSTEMS NOMINAL').upper()}",
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
