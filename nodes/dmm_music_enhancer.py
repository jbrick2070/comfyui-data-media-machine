"""
DMM Music Enhancer â€” MusicGen-melody audio-to-audio music enhancement.

Takes the background music track from BackgroundMusic and runs it through
Meta's MusicGen-melody for music-to-music style transfer via chroma/melody
conditioning. Each run randomly selects one of 12 LA-themed style prompts
and generates music that follows the harmonic contour of the original.

Architecture:
  BackgroundMusic (48kHz stereo) -> MusicEnhancer -> AudioMux -> SaveVideo

Melody conditioning: MusicGen reads the chroma (pitch/harmonic) features
of the input audio and generates new music that tracks those shapes while
adopting the prompted style. The result is blended back with the original
at the specified mix ratio.

Strength widget maps to MusicGen guidance_scale (how strongly the text
prompt drives the output vs the input melody):
  Low  (0.05-0.20): Melody leads â€” output closely follows input harmony
  Mid  (0.30-0.50): Balanced â€” style prompt and melody co-guide output
  High (0.60-0.95): Prompt dominates â€” melody is a loose reference

num_inference_steps widget = seconds of audio to generate (5-30s).
Output is looped (with raised-cosine crossfade) or trimmed to match input.

No external API required. Model loads directly into VRAM (~1.5GB).
Cached after first load.

Requirements:
  - transformers >= 4.35.0
  - Install: pip install transformers

VRAM Budget:
  - MusicGen-melody: ~1.5GB VRAM
  - Much lighter than Stable Audio Open (2-3GB)
  - Safe on RTX 5080 (24GB), RTX 4070+ (12GB+), RTX 3060 (12GB)

Author: Jeffrey A. Brick
Version: 3.5-beta (rewrite: MusicGen-melody, no external API)
"""

import gc
import io
import logging
import random
import time
import wave

import numpy as np
import torch

log = logging.getLogger("DMM.MusicEnhancer")

# MusicGen-melody native sample rate
MUSICGEN_SR = 32000


# ---------------------------------------------------------------------------
# LA-themed style prompt pool
# Each run randomly picks one to guide the music-to-music style transfer.
# References real synthesizer models and instruments native to LA 80s
# culture and LA indigenous culture.
# ---------------------------------------------------------------------------
LA_STYLE_PROMPTS = [
    # 80s LA synthpop â€” Oberheim OB-Xa + Sequential Prophet-5
    "Los Angeles 1984 synthpop, Oberheim OB-Xa polyphonic pad wash, "
    "Sequential Prophet-5 lead arpeggio, Roland TR-808 drum machine, "
    "Sunset Strip neon, cold-wave pulse",

    # West Coast G-funk â€” TR-808 + Moog Minimoog
    "West coast G-funk, Roland TR-808 bass thump, Moog Minimoog sliding bassline, "
    "talk box melody over pentatonic chords, Parliament-Funkadelic groove, "
    "Compton summer heat",

    # 80s LA R&B â€” Yamaha DX7 + Linn LM-1
    "Los Angeles 1986 R&B slow jam, Yamaha DX7 electric piano, Linn LM-1 snare crack, "
    "Fender Rhodes chord stabs, velvet reverb tail, late-night studio session",

    # Synthwave â€” Roland Jupiter-8 + Juno-106
    "Los Angeles synthwave, Roland Jupiter-8 sweeping pads, Roland Juno-106 chorus shimmer, "
    "analog sequencer pulse, Pacific Coast Highway midnight drive, neon rain reflection",

    # Aztec / East LA indigenous â€” huehuetl + teponaztli + conchero
    "East Los Angeles Aztec ceremony fusion, huehuetl heartbeat drum, teponaztli log drum "
    "call-and-response, conchero shell rattle cascade, copal smoke and street murals, "
    "Boyle Heights dusk ritual",

    # Tongva indigenous LA â€” elderberry flute + deer-hoof rattle + clapstick
    "Tongva Gabrielino indigenous soundscape, elderberry flute breathy melody, "
    "deer-hoof rattle shimmer, clapstick rhythm over basket drum pulse, "
    "coastal sage and salt wind, Ballona Creek ceremony",

    # Central Avenue jazz â€” upright bass + saxophone
    "Los Angeles Central Avenue jazz 1948, upright bass walking line, wire brushed snare, "
    "Dexter Gordon tenor saxophone blue note improvisation, Steinway grand chord stabs, "
    "after-hours glow and cigarette smoke",

    # LA gospel â€” Hammond B3 + choir
    "Los Angeles Black church gospel, Hammond B3 organ full drawbar swell, "
    "soulful soprano lead over four-part choir, tambourine on the two and four, "
    "congregation clap-back, sanctified reverb",

    # Lowrider / Chicano soul â€” Fender Strat + marimba + doo-wop
    "East LA lowrider soul, Fender Stratocaster clean-tone chord strum, "
    "marimba counter-melody, smooth doo-wop harmonies, light conga groove, "
    "cruising Whittier Boulevard on a warm Sunday evening",

    # Chumash coastal indigenous â€” bone whistle + gourd rattle + split-stick clapper
    "Chumash coastal California ceremonial, bone whistle ascending melody, "
    "gourd rattle wash, split-stick clapper steady pulse, ocean drum resonance, "
    "Santa Monica Mountains morning mist and wild sage",

    # LA ambient / Blade Runner â€” Roland SH-101 + Prophet-VS
    "Los Angeles late-night ambient, Roland SH-101 sub-bass drone, "
    "Sequential Circuits Prophet-VS granular shimmer pad, freeway overpass texture, "
    "Blade Runner 2019 skyline, slow evolving resonance",

    # Hollywood cinematic score â€” strings + French horn + Steinway
    "Hollywood cinematic score, sweeping string orchestra swell, French horn heroic motif, "
    "Steinway grand piano cascading runs, deep timpani roll, "
    "Bernard Herrmann tension and release, wide-screen grandeur",
]


# ---------------------------------------------------------------------------
# Model cache (singleton â€” loads once, reuses across executions)
# ---------------------------------------------------------------------------
_model_cache = {
    "processor": None,
    "model": None,
    "device": None,
}


def _get_or_load_model(device: torch.device = None):
    """Load MusicGen-melody processor and model, cached after first call.

    Invalidates and reloads if called with a different device than the
    cached model (e.g. user switches --cpu or GPU IDs between runs). (R7 fix)

    Returns (processor, model) tuple, or (None, None) if not available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Invalidate cache on device change (R7)
    if (
        _model_cache["model"] is not None
        and _model_cache["device"] is not None
        and str(_model_cache["device"]) != str(device)
    ):
        log.info(
            "  Device changed (%s -> %s) -- clearing model cache",
            _model_cache["device"], device,
        )
        _model_cache["processor"] = None
        _model_cache["model"] = None
        _model_cache["device"] = None

    if _model_cache["model"] is not None:
        return _model_cache["processor"], _model_cache["model"]

    try:
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

        log.info("  Loading MusicGen-melody (first run only)...")
        t0 = time.time()

        processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            "facebook/musicgen-melody",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        elapsed = time.time() - t0
        log.info("  MusicGen-melody loaded in %.1fs on %s", elapsed, device)

        _model_cache["processor"] = processor
        _model_cache["model"] = model
        _model_cache["device"] = device
        return processor, model

    except ImportError:
        log.warning(
            "transformers not installed. "
            "Run: pip install transformers"
        )
        return None, None
    except Exception as e:
        log.error("Failed to load MusicGen-melody: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Audio resampling utility
# ---------------------------------------------------------------------------
def _resample_tensor(tensor: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
    """Resample audio tensor using torchaudio polyphase resampler. (R2 fix)

    Polyphase resampling avoids the high-frequency aliasing that F.interpolate
    introduces on the 48kHz->32kHz->48kHz round trip.
    Falls back to linear interpolation if torchaudio is not available.

    Handles both (C, T) and (B, C, T) shapes. Returns same shape as input.
    """
    if from_sr == to_sr:
        return tensor

    squeezed = tensor.ndim == 2
    if squeezed:
        tensor = tensor.unsqueeze(0)

    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(
            orig_freq=from_sr, new_freq=to_sr,
        ).to(tensor.device)
        resampled = resampler(tensor.float())
    except ImportError:
        log.warning("torchaudio unavailable -- falling back to linear resampling")
        new_length = int(tensor.shape[-1] * to_sr / from_sr)
        resampled = torch.nn.functional.interpolate(
            tensor.float(),
            size=new_length,
            mode="linear",
            align_corners=False,
        )

    if squeezed:
        resampled = resampled.squeeze(0)

    return resampled


# ---------------------------------------------------------------------------
# Crossfade loop â€” for generated clips shorter than the input timeline
# ---------------------------------------------------------------------------
def crossfade_loop(
    audio: torch.Tensor,
    target_samples: int,
    crossfade_samples: int = None,
    sample_rate: int = 48000,
) -> torch.Tensor:
    """Seamlessly loop audio to reach target_samples using raised-cosine crossfade.

    If MusicGen's generated clip is shorter than the video timeline,
    this loops it with smooth crossfades at boundaries.

    Args:
        audio: Tensor (..., samples)
        target_samples: Desired total length in samples.
        crossfade_samples: Overlap region. Defaults to 2s or 25% of clip.
        sample_rate: Used to calculate default crossfade duration.

    Returns:
        Tensor (..., target_samples) with seamless loops.
    """
    src_len = audio.shape[-1]

    if src_len >= target_samples:
        return audio[..., :target_samples]

    if crossfade_samples is None:
        crossfade_samples = min(sample_rate * 2, src_len // 4)
    # Clamp to src_len // 3 BEFORE computing loop_body_len. (R5 fix)
    # Without this, crossfade_samples >= src_len makes loop_body_len <= 0
    # and the while loop never advances pos, spinning to the safety ceiling.
    crossfade_samples = max(min(crossfade_samples, src_len // 3), 1)

    # Raised-cosine fade curves
    fade = torch.linspace(0.0, 1.0, crossfade_samples, device=audio.device)
    fade = 0.5 * (1.0 - torch.cos(fade * torch.pi))
    fade_in = fade
    fade_out = 1.0 - fade

    loop_body_len = src_len - crossfade_samples
    result = torch.zeros(*audio.shape[:-1], target_samples, device=audio.device)

    pos = 0
    iteration = 0
    while pos < target_samples:
        remaining = target_samples - pos

        if iteration == 0:
            copy_len = min(src_len, remaining)
            result[..., pos:pos + copy_len] = audio[..., :copy_len]
            pos += loop_body_len
        else:
            if pos - crossfade_samples >= 0 and remaining > 0:
                xf_start = pos - crossfade_samples
                xf_end = pos
                result[..., xf_start:xf_end] *= fade_out
                head = audio[..., :crossfade_samples] * fade_in
                result[..., xf_start:xf_end] += head

            body_remaining = min(loop_body_len, remaining)
            copy_src = audio[..., crossfade_samples:crossfade_samples + body_remaining]
            copy_len = copy_src.shape[-1]

            if pos + copy_len > target_samples:
                copy_len = target_samples - pos

            if copy_len > 0:
                result[..., pos:pos + copy_len] = (
                    audio[..., crossfade_samples:crossfade_samples + copy_len]
                )

            pos += loop_body_len

        iteration += 1
        if iteration > (target_samples // max(loop_body_len, 1)) + 2:
            break

    log.info(
        "  Crossfade loop: %d iterations, %.1fs -> %.1fs",
        iteration,
        src_len / sample_rate,
        target_samples / sample_rate,
    )
    return result[..., :target_samples]


# ---------------------------------------------------------------------------
# Dry/wet mix utility (channel + length matching)
# ---------------------------------------------------------------------------
def mix_audio(dry: torch.Tensor, wet: torch.Tensor, mix_val: float) -> torch.Tensor:
    """Blend original (dry) and generated (wet) audio.

    Handles channel count and length mismatches automatically.
    Applies soft clip (tanh) to prevent overs.
    """
    # Ensure both tensors are on the same device
    if dry.device != wet.device:
        wet = wet.to(dry.device)
    # Also ensure matching dtype
    if dry.dtype != wet.dtype:
        wet = wet.to(dry.dtype)

    dry_ch = dry.shape[-2] if dry.ndim >= 2 else 1
    wet_ch = wet.shape[-2] if wet.ndim >= 2 else 1

    if dry_ch != wet_ch:
        log.info("  Channel mismatch: dry=%dch wet=%dch â€” adjusting wet", dry_ch, wet_ch)
        if dry_ch == 2 and wet_ch == 1:
            wet = wet.repeat(*(1,) * (wet.ndim - 2), 2, 1)
        elif dry_ch == 1 and wet_ch == 2:
            wet = wet[:, :1, :] if wet.ndim == 3 else wet[:1, :]
        else:
            log.warning("  Unusual channel mismatch (%d vs %d) â€” truncating wet", dry_ch, wet_ch)
            wet = wet[..., :dry_ch, :]

    dry_len = dry.shape[-1]
    wet_len = wet.shape[-1]
    if dry_len > wet_len:
        pad = torch.zeros(*wet.shape[:-1], dry_len - wet_len, device=wet.device)
        wet = torch.cat([wet, pad], dim=-1)
    elif wet_len > dry_len:
        wet = wet[..., :dry_len]

    mixed = (1.0 - mix_val) * dry + mix_val * wet
    return torch.tanh(mixed)


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancer
# ---------------------------------------------------------------------------
class DMM_MusicEnhancer:
    """Music-to-music enhancement using MusicGen-melody.

    Sits between BackgroundMusic and AudioMux in the DMM pipeline.
    Extracts the harmonic/chroma contour of the input music and generates
    new music conditioned on both the contour and a randomly selected
    LA-themed style prompt. Result is dry/wet blended with the original.

    Widget mapping:
      strength          -> guidance_scale (0.05->1.5, 0.20->2.8, 0.50->5.5)
                          Low = melody-driven | High = style-driven
      mix               -> dry/wet blend (0=original, 1=MusicGen only)
      prompt_override   -> leave blank for random LA prompt
      seed              -> -1 for random each run
      num_inference_steps -> seconds of audio to generate (5-30s)
                            looped or trimmed to match input length
    """

    CATEGORY = "DMM/Audio"
    FUNCTION = "enhance"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Prompt guidance strength. "
                            "Low=melody leads. High=style prompt leads."
                        ),
                    },
                ),
                "mix": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Dry/wet blend. 0=original only, 1=MusicGen only.",
                    },
                ),
            },
            "optional": {
                "prompt_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Leave empty for random LA style prompt each run.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**32 - 1,
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 5,
                        "max": 30,
                        "step": 1,
                        "tooltip": (
                            "Seconds of music to generate per pass (5-30s). "
                            "Output is looped/trimmed to match input clip length."
                        ),
                    },
                ),
            },
        }

    def enhance(
        self,
        audio,
        strength: float = 0.20,
        mix: float = 0.65,
        prompt_override: str = "",
        seed: int = -1,
        num_inference_steps: int = 20,
    ) -> tuple:
        """Main enhancement function called by ComfyUI."""

        # -------------------------------------------------------------------
        # 1. Seed + prompt selection
        # -------------------------------------------------------------------
        if seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        rng = random.Random(actual_seed)

        if prompt_override and prompt_override.strip():
            prompt = prompt_override.strip()
            log.info("DMM_MusicEnhancer: Using custom prompt")
        else:
            prompt = rng.choice(LA_STYLE_PROMPTS)
            log.info("DMM_MusicEnhancer: Random LA style selected")

        # Map strength (0.05-0.95) -> guidance_scale (1.5-9.55)
        # MusicGen default is 3.0; strength=0.20 -> ~3.3 (close to default)
        guidance_scale = 1.5 + strength * 8.5

        log.info("  Prompt: %s", prompt[:80])
        log.info(
            "  Guidance: %.1f (strength=%.2f) | Mix: %.2f | Seed: %d | Duration: %ds",
            guidance_scale, strength, mix, actual_seed, num_inference_steps,
        )

        # -------------------------------------------------------------------
        # 2. Unpack ComfyUI AUDIO dict
        # -------------------------------------------------------------------
        if isinstance(audio, dict):
            if "waveform" not in audio:
                log.warning("Audio dict missing 'waveform' key â€” passing through unchanged")
                return (audio,)
            audio_tensor = audio["waveform"]
            sample_rate = audio.get("sample_rate", 48000)
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio
            sample_rate = 48000
        else:
            log.warning(
                "Unexpected audio type %s â€” passing through unchanged",
                type(audio).__name__,
            )
            return (audio,)

        if audio_tensor is None or audio_tensor.numel() == 0:
            log.warning("Empty audio input â€” passing through unchanged")
            return (audio,)

        original_tensor = audio_tensor.clone()

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, C, T)

        input_duration = audio_tensor.shape[-1] / sample_rate
        log.info(
            "  Input: %.1fs, %dch, %dHz, batch=%d",
            input_duration, audio_tensor.shape[-2], sample_rate, audio_tensor.shape[0],
        )

        # -------------------------------------------------------------------
        # 3. VRAM flush
        # -------------------------------------------------------------------
        log.info("  Flushing VRAM...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            log.info("  VRAM free after flush: %.0f MB", free_mb)

        # -------------------------------------------------------------------
        # 4. Load MusicGen-melody (cached after first call)
        # -------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor, model = _get_or_load_model(device)

        if model is None:
            log.warning(
                "MusicGen-melody not available â€” passing through unchanged\n"
                "  Install: pip install transformers"
            )
            return (
                audio
                if isinstance(audio, dict)
                else {"waveform": original_tensor, "sample_rate": sample_rate},
            )

        # -------------------------------------------------------------------
        # 5. Generate per batch item
        # -------------------------------------------------------------------
        # num_inference_steps is repurposed as generation duration (seconds).
        # MusicGen outputs 50 tokens/sec at its 32kHz / 320-sample frame rate.
        duration_s = max(5, min(30, num_inference_steps))
        max_new_tokens = int(duration_s * 50)

        batch_size = audio_tensor.shape[0]
        enhanced_batches = []
        t0 = time.time()

        for b in range(batch_size):
            clip = audio_tensor[b]  # (C, T)
            orig_samples = clip.shape[-1]

            if batch_size > 1:
                log.info("  Processing batch item %d/%d", b + 1, batch_size)

            try:
                # Resample input to MusicGen's 32kHz for melody conditioning
                clip_32k = _resample_tensor(clip, sample_rate, MUSICGEN_SR)

                # Mono numpy for melody conditioning (processor expects 1-D)
                mono_32k = clip_32k.mean(dim=0).cpu().float().numpy()  # (T,)

                # Build processor inputs
                processor_out = processor(
                    text=[prompt],
                    audio=[mono_32k],
                    sampling_rate=MUSICGEN_SR,
                    padding=True,
                    return_tensors="pt",
                )
                # Move to device AND cast float tensors to model dtype
                # (processor outputs float32 but model may be float16 on CUDA)
                model_dtype = next(model.parameters()).dtype
                inputs = {}
                for k, v in processor_out.items():
                    if hasattr(v, "to"):
                        v = v.to(device)
                        if v.is_floating_point():
                            v = v.to(model_dtype)
                    inputs[k] = v

                # Generate
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                        do_sample=True,
                        temperature=0.95,  # R6: slight reduction prevents tail noise
                    )

                # audio_values: (1, 1, T) at 32kHz
                generated = audio_values[0, 0].float()  # (T,)

                # Resample back to original sample rate
                generated = generated.unsqueeze(0)  # (1, T)
                generated = _resample_tensor(generated, MUSICGEN_SR, sample_rate)
                # generated: (1, T)

                # Expand to match original channel count
                orig_channels = clip.shape[0]
                if orig_channels == 2:
                    generated = generated.repeat(2, 1)  # (2, T)
                # else stays (1, T)

                # Trim or crossfade-loop to match original length
                gen_samples = generated.shape[-1]
                if gen_samples >= orig_samples:
                    generated = generated[..., :orig_samples]
                else:
                    generated = crossfade_loop(
                        generated, orig_samples, sample_rate=sample_rate
                    )

                enhanced_batches.append(generated.unsqueeze(0))  # (1, C, T)

            except Exception as e:
                log.error(
                    "MusicGen generation failed on batch %d: %s â€” using original", b, e
                )
                enhanced_batches.append(clip.unsqueeze(0))

        elapsed = time.time() - t0
        log.info("  MusicGen processing: %.1fs (%d batch items)", elapsed, batch_size)

        # -------------------------------------------------------------------
        # 6. Stack + dry/wet mix
        # -------------------------------------------------------------------
        enhanced_full = torch.cat(enhanced_batches, dim=0)  # (B, C, T)
        result_tensor = mix_audio(original_tensor, enhanced_full, mix)

        log.info(
            "  Output: %.1fs, batch=%d | Enhancement complete",
            result_tensor.shape[-1] / sample_rate,
            result_tensor.shape[0],
        )

        # -------------------------------------------------------------------
        # 7. Repack ComfyUI AUDIO dict
        # -------------------------------------------------------------------
        return ({"waveform": result_tensor, "sample_rate": sample_rate},)


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancerBypass
# ---------------------------------------------------------------------------
class DMM_MusicEnhancerBypass:
    """Simple bypass/passthrough for A/B testing.

    Wire in place of DMM_MusicEnhancer to hear the original mix
    without enhancement.
    """

    CATEGORY = "DMM/Audio"
    FUNCTION = "passthrough"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    def passthrough(self, audio) -> tuple:
        log.info("DMM_MusicEnhancerBypass: Passing audio through unchanged")
        return (audio,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "DMM_MusicEnhancer": DMM_MusicEnhancer,
    "DMM_MusicEnhancerBypass": DMM_MusicEnhancerBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DMM_MusicEnhancer": "DMM Music Enhancer (MusicGen)",
    "DMM_MusicEnhancerBypass": "DMM Music Enhancer Bypass",
}
