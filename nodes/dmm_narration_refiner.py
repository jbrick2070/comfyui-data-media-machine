"""
DMM_NarrationRefiner -- AI-polishes template narration via Qwen2.5.

Takes the ~60-word template narration from NarrationDistiller and rewrites
it to sound like a real broadcast anchor.  All facts are preserved exactly;
only phrasing, flow, and cadence are improved.

Qwen2.5 pattern borrowed from ComfyUI-UCLA-News-Video/nodes/ucla_video_prompt_gen.py:
  - Lazy-loads Qwen/Qwen2.5-3B-Instruct or 7B in float16
  - Unloads VRAM after generation so downstream nodes have full headroom
  - Graceful fallback: returns original text if Qwen fails or is unavailable

Author: Jeffrey A. Brick
"""

import logging
import time

log = logging.getLogger("media_machine.narration_refiner")

# -- Qwen lazy loader (same pattern as UCLA News Video / Goofer) --------------
_qwen_model = None
_qwen_tok = None
_qwen_loaded_id = None


def _get_qwen(model_id):
    """Lazy-load a Qwen2.5-Instruct model in float16 on CUDA."""
    global _qwen_model, _qwen_tok, _qwen_loaded_id

    if _qwen_model is not None and _qwen_loaded_id == model_id:
        return _qwen_model, _qwen_tok

    if _qwen_model is not None:
        _unload_qwen()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("[NarrationRefiner] Loading %s ...", model_id)
        print(f"[DMM] Loading {model_id} (first run downloads model)...")

        _qwen_tok = AutoTokenizer.from_pretrained(model_id)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
        ).to("cuda").eval()

        _qwen_loaded_id = model_id
        log.info("[NarrationRefiner] %s loaded on CUDA", model_id)
        print(f"[DMM] {model_id} loaded on CUDA")
    except Exception as exc:
        log.exception("[NarrationRefiner] %s failed to load: %s", model_id, exc)
        print(f"[DMM] Qwen load failed: {exc}")
        _qwen_model = None
        _qwen_tok = None
        _qwen_loaded_id = None

    return _qwen_model, _qwen_tok


def _unload_qwen():
    """Free VRAM after refinement so TTS/MusicGen/video nodes have headroom."""
    global _qwen_model, _qwen_tok, _qwen_loaded_id
    if _qwen_model is None:
        return
    try:
        import torch
        del _qwen_model, _qwen_tok
        _qwen_model = None
        _qwen_tok = None
        _qwen_loaded_id = None
        torch.cuda.empty_cache()
        log.info("[NarrationRefiner] Qwen unloaded from VRAM")
        print("[DMM] Qwen unloaded from VRAM")
    except Exception as exc:
        log.debug("[NarrationRefiner] Qwen unload: %s", exc)


# -- System prompt for narration rewriting ------------------------------------

_REFINE_SYSTEM = (
    "You are a veteran Los Angeles radio broadcast editor. "
    "Your job is to rewrite data narration scripts so they sound natural, "
    "warm, and professional — like a real late-night radio host reading the "
    "city report.\n\n"
    "Rules you MUST follow:\n"
    "1. Keep ALL facts, numbers, temperatures, times, and data EXACTLY as given. "
    "Do NOT change, round, or omit any data point.\n"
    "2. Output ONLY the rewritten narration — no preamble, no explanation, no quotes.\n"
    "3. Keep the word count close to the original (within +/- 10 words). "
    "The narration must fit in ~24 seconds of speech.\n"
    "4. Use natural broadcast cadence — vary sentence length, add brief pauses "
    "(commas, ellipses) where a real anchor would breathe.\n"
    "5. Do NOT add fictional details, opinions, or editorializing.\n"
    "6. Do NOT use exclamation marks or overly enthusiastic language.\n"
)

_REFINE_USER_TMPL = (
    "Here is the raw data narration to rewrite:\n\n"
    "{narration}\n\n"
    "Rewrite this as a smooth, professional broadcast narration. "
    "Style: {style}."
)

_BAD_OUTPUTS = ["i cannot", "i can't", "i'm sorry", "i apologize", "as an ai",
                "here is", "here's the", "sure,", "certainly"]


def _refine_text(model, tok, narration: str, style: str) -> str:
    """Run Qwen2.5 refinement on narration text. Returns '' on failure."""
    import torch

    msg = _REFINE_USER_TMPL.format(narration=narration, style=style)
    messages = [
        {"role": "system", "content": _REFINE_SYSTEM},
        {"role": "user",   "content": msg},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.6,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    result = tok.decode(new_tok, skip_special_tokens=True).strip()

    # Strip wrapping quotes if Qwen adds them
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1].strip()

    # Validate output
    word_count = len(result.split())
    if word_count < 10:
        log.warning("[NarrationRefiner] Output too short (%d words), using original", word_count)
        return ""
    if any(b in result.lower()[:50] for b in _BAD_OUTPUTS):
        log.warning("[NarrationRefiner] Refusal/preamble detected, using original")
        return ""

    return result


# -- Model size → HuggingFace ID mapping --------------------------------------

_MODEL_MAP = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}


# -- ComfyUI Node -------------------------------------------------------------

class DMMNarrationRefiner:
    """AI-polishes template narration text via Qwen2.5.

    Insert between NarrationDistiller and Kokoro TTS.
    Qwen is lazy-loaded and unloaded after each run to free VRAM.
    Falls back to original text if Qwen is unavailable.
    """

    CATEGORY = "Data Media Machine"
    FUNCTION = "refine"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("narration_text",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "narration_text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Raw narration text from NarrationDistiller to polish",
                }),
            },
            "optional": {
                "refine_mode": (["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Passthrough"], {
                    "default": "Qwen2.5-3B-Instruct",
                    "tooltip": (
                        "Qwen2.5-3B: AI rewrites for broadcast quality (~6 GB VRAM, faster). "
                        "Qwen2.5-7B: higher quality (~14 GB VRAM). "
                        "Passthrough: returns text unchanged."
                    ),
                }),
                "style": (["late_night_radio", "morning_news", "calm_documentary",
                           "weather_channel", "noir_dispatch"], {
                    "default": "late_night_radio",
                    "tooltip": "Broadcast style for the AI rewrite",
                }),
                "unload_after": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Free VRAM after generation. Disable if chaining multiple AI nodes.",
                }),
            },
        }

    def refine(self, narration_text, refine_mode="Qwen2.5-3B-Instruct",
               style="late_night_radio", unload_after=True):
        if not narration_text or not narration_text.strip():
            log.warning("[NarrationRefiner] Empty input, returning empty string")
            return ("",)

        if refine_mode == "Passthrough":
            log.info("[NarrationRefiner] Passthrough mode, returning original (%d words)",
                     len(narration_text.split()))
            return (narration_text,)

        # --- Qwen2.5 refinement ---
        t0 = time.time()
        model_id = _MODEL_MAP.get(refine_mode, "Qwen/Qwen2.5-3B-Instruct")
        model, tok = _get_qwen(model_id)

        if model is None:
            log.warning("[NarrationRefiner] %s unavailable, returning original", refine_mode)
            return (narration_text,)

        style_label = style.replace("_", " ")
        refined = _refine_text(model, tok, narration_text, style_label)

        # Unload Qwen to free VRAM for TTS / MusicGen / video nodes
        if unload_after:
            _unload_qwen()

        if not refined:
            log.info("[NarrationRefiner] Qwen output rejected, using original")
            return (narration_text,)

        elapsed = time.time() - t0
        orig_words = len(narration_text.split())
        new_words = len(refined.split())
        log.info("[NarrationRefiner] Refined in %.1fs: %d -> %d words (style=%s)",
                 elapsed, orig_words, new_words, style)
        log.info("[NarrationRefiner] Original:  %s", narration_text[:120])
        log.info("[NarrationRefiner] Refined:   %s", refined[:120])

        return (refined,)
