"""
DMMAsciiArt — A native, zero-dependency ASCII art generator for ComfyUI.
Transforms images into classic terminal-style text grids.
Built specifically for real-time VJ setups to avoid third-party node dependency hell.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class DMMAsciiArt:
    """A native ComfyUI node for high-speed ASCII generation."""
    
    CATEGORY = "DataMediaMachine"
    FUNCTION = "generate_ascii"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("ascii_image", "ascii_text",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "columns": ("INT", {"default": 120, "min": 20, "max": 400, "step": 5}),
                "color_theme": (["terminal_green", "cyber_blue", "amber_monochrome", "pure_white"],),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }
        
    def generate_ascii(self, image, columns, color_theme, invert):
        batch_size = image.shape[0]
        out_images = []
        out_texts = []
        
        # Classic ASCII gradient from darkest to lightest
        chars = list(" .':,-~+=*#%@")
        if invert:
            chars.reverse()
            
        themes = {
            "terminal_green": (0, 255, 0),
            "cyber_blue": (0, 200, 255),
            "amber_monochrome": (255, 176, 0),
            "pure_white": (255, 255, 255)
        }
        text_color = themes.get(color_theme, (0, 255, 0))
            
        font = ImageFont.load_default()
        
        try:
            # Pillow < 10.0
            char_w, char_h = font.getsize("A")
        except AttributeError:
            # Pillow >= 10.0
            left, top, right, bottom = font.getbbox("A")
            char_w = right - left
            char_h = bottom - top
            if char_w == 0: char_w = 6
            if char_h == 0: char_h = 11
            
        for i in range(batch_size):
            img_tensor = image[i]
            img_np = (img_tensor.numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            gray_img = img_pil.convert("L")
            
            W, H = gray_img.size
            aspect_ratio = H / float(W)
            font_aspect_correction = char_w / float(char_h)
            
            new_W = columns
            new_H = int(aspect_ratio * columns * font_aspect_correction)
            if new_H < 1: new_H = 1
            
            gray_img = gray_img.resize((new_W, new_H), Image.Resampling.LANCZOS)
            
            pixels = np.array(gray_img)
            ascii_lines = []
            for row in pixels:
                indices = (row / 255.0 * (len(chars) - 1.0001)).astype(int)
                line = "".join([chars[idx] for idx in indices])
                ascii_lines.append(line)
                
            ascii_text = "\n".join(ascii_lines)
            out_texts.append(ascii_text)
            
            out_W = new_W * char_w
            out_H = new_H * char_h
            
            out_img = Image.new("RGB", (out_W, out_H), color=(0, 0, 0))
            draw = ImageDraw.Draw(out_img)
            
            y_text = 0
            for line in ascii_lines:
                draw.text((0, y_text), line, font=font, fill=text_color)
                y_text += char_h
                
            out_np = np.array(out_img).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)
            out_images.append(out_tensor)
            
        final_images = torch.stack(out_images)
        final_text = "\n\n".join(out_texts)
        
        return (final_images, final_text)

NODE_CLASS_MAPPINGS = {
    "DMM_AsciiArt": DMMAsciiArt
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DMM_AsciiArt": "🔤 DMM ASCII Art Filter"
}