import sys
import math
import numpy as np
from PIL import Image, ImageDraw

NEON = {
    "bg":       (10, 10, 26),
    "grid":     (25, 25, 60),
    "pink":     (255, 20, 147),
    "cyan":     (0, 229, 255),
    "amber":    (255, 176, 0),
    "purple":   (180, 0, 255),
    "white":    (220, 220, 230),
    "dim":      (80, 80, 120),
    "hot":      (255, 60, 60),
}

def _dim(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)

def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def _draw_arc_gauge(draw, cx, cy, r, frac, color, bg_color):
    bbox = [cx - r, cy - r, cx + r, cy + r]
    bg_arc_color = _dim(bg_color, 0.3)
    draw.arc(bbox, 180, 360, fill=bg_arc_color, width=2)
    print(f"Drawing background arc: {bg_arc_color}")

    if frac > 0.01:
        end_angle = 180 + int(180 * frac)
        arc_color = _lerp_color(color, NEON["hot"], frac) if frac > 0.6 else color
        print(f"Drawing filled arc: frac={frac:.2f}, angles: 180 -> {end_angle}, color: {arc_color}")
        draw.arc(bbox, 180, end_angle, fill=arc_color, width=2)

img = Image.new("RGB", (800, 200), NEON["bg"])
draw = ImageDraw.Draw(img)

# Weather gauge: 83 / 120
frac_weather = min(83 / 120, 1.0)
_draw_arc_gauge(draw, 100, 100, 50, frac_weather, NEON["cyan"], NEON["dim"])

# AQI gauge: 84 / 300
frac_aqi = min(84 / 300, 1.0)
_draw_arc_gauge(draw, 300, 100, 50, frac_aqi, NEON["amber"], NEON["dim"])

# Transit gauge: 70 / 100
frac_transit = min(70 / 100, 1.0)
_draw_arc_gauge(draw, 500, 100, 50, frac_transit, NEON["pink"], NEON["dim"])

# Seismic gauge: 3 / 20
frac_seismic = min(3 / 20, 1.0)
_draw_arc_gauge(draw, 700, 100, 50, frac_seismic, NEON["purple"], NEON["dim"])

img.save(r"C:\Users\jeffr\Documents\ComfyUI\comfyui-data-media-machine\test_gauge.png")
print("Saved to test_gauge.png")
