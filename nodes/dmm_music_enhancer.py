"""
DMM Music Enhancer — Stable Audio Open native audio-to-audio enhancement.

Takes the background music track from BackgroundMusic and runs it through
Stable Audio Open's latent diffusion for music-to-music style transfer.
Each run randomly selects an LA-themed style prompt (deep house, choir,
noir jazz, etc.) to add cinematic texture — without replacing the original.

Architecture:
  BackgroundMusic (48kHz stereo) -> MusicEnhancer -> AudioMux -> SaveVideo

No external API required. Model loads directly into VRAM as a standard
ComfyUI checkpoint. Cached after first load.

Requirements:
  - Stable Audio Open checkpoint (stabilityai/stable-audio-open-1.0)
  - torch, torchaudio, diffusers, transformers

VRAM Budget:
  - Stable Audio Open: ~2-3GB VRAM
  - LTX-Video will be unloaded by this point in the pipeline
  - Safe on RTX 5080 (24GB) and RTX 4070+ (12GB+)

Author: Jeffrey A. Brick
Version: 3.5-beta (rewrite: native Stable Audio Open, no external API)
"""

import gc
import io
import logging
import os
import random
import time
import wave
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("DMM.MusicEnhancer")


# ---------------------------------------------------------------------------
# LA-themed style prompt pool
# Each run randomly picks one to guide the audio-to-audio style transfer.
# ---------------------------------------------------------------------------
LA_STYLE_PROMPTS = [
    # 80s LA synthpop — Oberheim OB-Xa + Sequential Prophet-5
    "Los Angeles 1984 synthpop, Oberheim OB-Xa polyphonic pad wash, "
    "Sequential Prophet-5 lead arpeggio, Roland TR-808 drum machine, "
    "Sunset Strip neon, cold-wave pulse",

    # West Coast G-funk — TR-808 + Moog Minimoog
    "West coast G-funk, Roland TR-808 bass thump, Moog Minimoog sliding bassline, "
    "talk box melody over pentatonic chords, Parliament-Funkadelic groove, "
    "Compton summer heat",

    # 80s LA R&B — Yamaha DX7 + Linn LM-1
    "Los Angeles 1986 R&B slow jam, Yamaha DX7 electric piano, Linn LM-1 snare crack, "
    "Fender Rhodes chord stabs, velvet reverb tail, late-night studio session",

    # Synthwave — Roland Jupiter-8 + Juno-106
    "Los Angeles synthwave, Roland Jupiter-8 sweeping pads, Roland Juno-106 chorus shimmer, "
    "analog sequencer pulse, Pacific Coast Highway midnight drive, neon rain reflection",

    # Aztec / East LA indigenous — huehuetl + teponaztli + conchero
    "East Los Angeles Aztec ceremony fusion, huehuetl heartbeat drum, teponaztli log drum "
    "call-and-response, conchero shell rattle cascade, copal smoke and street murals, "
    "Boyle Heights dusk ritual",

    # Tongva indigenous LA — elderberry flute + deer-hoof rattle + clapstick
    "Tongva Gabrielino indigenous soundscape, elderberry flute breathy melody, "
    "deer-hoof rattle shimmer, clapstick rhythm over basket drum pulse, "
    "coastal sage and salt wind, Ballona Creek ceremony",

    # Central Avenue jazz — upright bass + saxophone
    "Los Angeles Central Avenue jazz 1948, upright bass walking line, wire brushed snare, "
    "Dexter Gordon tenor saxophone blue note improvisation, Steinway grand chord stabs, "
    "after-hours glow and cigarette smoke",

    # LA gospel — Hammond B3 + choir
    "Los Angeles Black church gospel, Hammond B3 organ full drawbar swell, "
    "soulful soprano lead over four-part choir, tambourine on the two and four, "
    "congregation clap-back, sanctified reverb",

    # Lowrider / Chicano soul — Fender Strat + marimba + doo-wop
    "East LA lowrider soul, Fender Stratocaster clean-tone chord strum, "
    "marimba counter-melody, smooth doo-wop harmonies, light conga groove, "
    "cruising Whittier Boulevard on a warm Sunday evening",

    # Chumash coastal indigenous — bone whistle + gourd rattle + split-stick clapper
    "Chumash coastal California ceremonial, bone whistle ascending melody, "
    "gourd rattle wash, split-stick clapper steady pulse, ocean drum resonance, "
    "Santa Monica Mountains morning mist and wild sage",

    # LA ambient / Blade Runner — Roland SH-101 + Prophet-VS
    "Los Angeles late-night ambient, Roland SH-101 sub-bass drone, "
    "Sequential Circuits Prophet-VS granular shimmer pad, freeway overpass texture, "
    "Blade Runner 2019 skyline, slow evolving resonance",

    # Hollywood cinematic score — strings + French horn + Steinway
    "Hollywood cinematic score, sweeping string orchestra swell, French horn heroic motif, "
    "Steinway grand piano cascading runs, deep timpani roll, "
    "Bernard Herrmann tension and release, wide-screen grandeur",
]


# ---------------------------------------------------------------------------
# Model cache (singleton — loads once, reuses across executions)
# ---------------------------------------------------------------------------
_model_cache = {
    "pipe": None,
    "device": None,
}


def _get_or_load_model(device: torch.device = None):
    """Load Stable Audio Open pipeline, cached after first call.

    Uses diffusers StableAudioPipeline for native PyTorch inference.
    Falls back gracefully if the model isn't installed.
    """
    if _model_cache["pipe"] is not None:
        return _model_cache["pipe"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from diffusers import StableAudioPipeline

        log.info("  Loading Stable Audio Open model (first run only)...")
        t0 = time.time()

        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)

        elapsed = time.time() - t0
        log.info("  Model loaded in %.1fs on %s", elapsed, device)

        _model_cache["pipe"] = pipe
        _model_cache["device"] = device
        return pipe

    except ImportError:
        log.error(
            "diffusers not installed. Run: pip install diffusers transformers accelerate"
        )
        return None
    except Exception as e:
        log.error("Failed to load Stable Audio Open: %s", e)
        return None


# ---------------------------------------------------------------------------
# Audio tensor <-> WAV conversion utilities
# ---------------------------------------------------------------------------
def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int = 48000) -> bytes:
    """Convert a single audio tensor (channels, samples) to WAV bytes.

    Expects shape (channels, samples) or (1, channels, samples).
    Values should be in [-1.0, 1.0] float32.
    """
    if audio_tensor.ndim == 3:
        audio = audio_tensor[0]
    elif audio_tensor.ndim == 2:
        audio = audio_tensor
    else:
        raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")

    channels = audio.shape[0]
    samples = audio.shape[1]

    audio_np = audio.cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    interleaved = np.empty(channels * samples, dtype=np.int16)
    for ch in range(channels):
        interleaved[ch::channels] = audio_int16[ch]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Convert WAV bytes back to audio tensor.

    Returns (tensor, sample_rate) where tensor is (1, channels, samples).
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        max_val = 32767.0
    elif sample_width == 4:
        dtype = np.int32
        max_val = 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio_np = np.frombuffer(raw, dtype=dtype).astype(np.float32) / max_val
    audio_np = audio_np.reshape(-1, channels).T  # (channels, samples)

    tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, channels, samples)
    return tensor, sample_rate


# ---------------------------------------------------------------------------
# Dry/wet mix utility (with channel matching)
# ---------------------------------------------------------------------------
def crossfade_loop(audio: torch.Tensor, target_samples: int,
                    crossfade_samples: int = None,
                    sample_rate: int = 48000) -> torch.Tensor:
    """Seamlessly loop audio to reach target_samples using crossfade.

    If the generated audio (capped at 47s by Stable Audio Open) is shorter
    than the video timeline, this function loops it with smooth crossfades
    at the boundaries to avoid audible cuts.

    Args:
        audio: Tensor of shape (..., samples) — the generated clip.
        target_samples: Desired total length in samples.
        crossfade_samples: Overlap region in samples. Defaults to 2s worth.
        sample_rate: For calculating default crossfade duration.

    Returns:
        Tensor of shape (..., target_samples) with seamless loops.
    """
    src_len = audio.shape[-1]

    # If already long enough, just trim
    if src_len >= target_samples:
        return audio[..., :target_samples]

    # Default crossfade: 2 seconds, but never more than 25% of the clip
    if crossfade_samples is None:
        crossfade_samples = min(sample_rate * 2, src_len // 4)
    crossfade_samples = max(crossfade_samples, 1)  # Safety floor

    # Build fade curves (raised cosine for smooth energy-preserving transition)
    fade = torch.linspace(0.0, 1.0, crossfade_samples, device=audio.device)
    fade = 0.5 * (1.0 - torch.cos(fade * torch.pi))  # Raised cosine
    fade_in = fade
    fade_out = 1.0 - fade

    # The loopable body excludes the crossfade tail (which blends into the head)
    loop_body_len = src_len - crossfade_samples

    # Pre-allocate output
    result = torch.zeros(*audio.shape[:-1], target_samples, device=audio.device)

    pos = 0
    iteration = 0
    while pos < target_samples:
        remaining = target_samples - pos

        if iteration == 0:
            # First pass: copy the full clip (up to what fits)
            copy_len = min(src_len, remaining)
            result[..., pos:pos + copy_len] = audio[..., :copy_len]
            pos += loop_body_len  # Next write starts where crossfade begins
        else:
            # Subsequent passes: crossfade the head into the tail of previous
            if pos - crossfade_samples >= 0 and remaining > 0:
                # Apply crossfade at the join point
                xf_start = pos - crossfade_samples
                xf_end = pos

                # Tail of previous iteration fades out
                result[..., xf_start:xf_end] *= fade_out

                # Head of new iteration fades in
                head = audio[..., :crossfade_samples] * fade_in
                result[..., xf_start:xf_end] += head

            # Copy the rest of the loop body (after crossfade region)
            body_remaining = min(loop_body_len, remaining)
            copy_src = audio[..., crossfade_samples:crossfade_samples + body_remaining]
            copy_len = copy_src.shape[-1]

            if pos + copy_len > target_samples:
                copy_len = target_samples - pos

            if copy_len > 0:
                result[..., pos:pos + copy_len] = audio[..., crossfade_samples:crossfade_samples + copy_len]

            pos += loop_body_len

        iteration += 1

        # Safety valve: prevent infinite loop on degenerate input
        if iteration > (target_samples // max(loop_body_len, 1)) + 2:
            break

    log.info("  Crossfade loop: %d iterations, %.1fs → %.1fs",
             iteration, src_len / sample_rate, target_samples / sample_rate)

    return result[..., :target_samples]


def mix_audio(dry: torch.Tensor, wet: torch.Tensor, mix_val: float) -> torch.Tensor:
    """Blend original (dry) and enhanced (wet) audio.

    Handles channel count and length mismatches automatically.
    """
    # --- Channel matching ---
    dry_ch = dry.shape[-2] if dry.ndim >= 2 else 1
    wet_ch = wet.shape[-2] if wet.ndim >= 2 else 1

    if dry_ch != wet_ch:
        log.info("  Channel mismatch: dry=%dch, wet=%dch — adjusting wet", dry_ch, wet_ch)
        if dry_ch == 2 and wet_ch == 1:
            wet = wet.repeat(*(1,) * (wet.ndim - 2), 2, 1)
        elif dry_ch == 1 and wet_ch == 2:
            if wet.ndim == 3:
                wet = wet[:, :1, :]
            else:
                wet = wet[:1, :]
        else:
            log.warning("  Unusual channel mismatch (%d vs %d), truncating wet", dry_ch, wet_ch)
            wet = wet[..., :dry_ch, :]

    # --- Length matching ---
    dry_len = dry.shape[-1]
    wet_len = wet.shape[-1]
    if dry_len > wet_len:
        pad = torch.zeros(*wet.shape[:-1], dry_len - wet_len, device=wet.device)
        wet = torch.cat([wet, pad], dim=-1)
    elif wet_len > dry_len:
        wet = wet[..., :dry_len]

    mixed = (1.0 - mix_val) * dry + mix_val * wet

    # Soft clip to prevent overs
    mixed = torch.tanh(mixed)

    return mixed


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancer
# ---------------------------------------------------------------------------
class DMM_MusicEnhancer:
    """Music-to-music enhancement using native Stable Audio Open.

    Sits between BackgroundMusic and AudioMux in the DMM pipeline. Takes
    the background music, encodes it into Stable Audio's latent space,
    applies noise at the specified strength, denoises with a randomly
    selected LA-themed style prompt, and decodes back to audio.

    No external API. Model loads directly into VRAM (cached after first run).

    Strength controls how much the style transfer changes the audio:
    - 0.10-0.20: Subtle texture (ambient shimmer, warmth)
    - 0.20-0.35: Moderate enhancement (new instrumentation blended in)
    - 0.35-0.50: Strong transformation (significant new character)
    - 0.50+: Heavy regen (original becomes a loose reference)

    Recommended: 0.20 for production, 0.15 for narration-heavy content.
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
                        "tooltip": "Dry/wet mix. 0=original only, 1=enhanced only",
                    },
                ),
            },
            "optional": {
                "prompt_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Leave empty for random LA style. Set to override with a specific prompt.",
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
                        "max": 100,
                        "step": 5,
                        "tooltip": "More steps = better quality, slower. 20 is a good balance.",
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

        # -----------------------------------------------------------
        # 1. Select style prompt (random from pool or user override)
        # -----------------------------------------------------------
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

        log.info("  Prompt: %s", prompt[:80])
        log.info("  Strength: %.2f | Mix: %.2f | Seed: %d | Steps: %d",
                 strength, mix, actual_seed, num_inference_steps)

        # -----------------------------------------------------------
        # 2. Unpack ComfyUI AUDIO dict
        # -----------------------------------------------------------
        if isinstance(audio, dict):
            if "waveform" not in audio:
                log.warning("Audio dict missing 'waveform' key — passing through unchanged")
                return (audio,)
            audio_tensor = audio["waveform"]
            sample_rate = audio.get("sample_rate", 48000)
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio
            sample_rate = 48000
        else:
            log.warning("Unexpected audio type %s — passing through unchanged", type(audio).__name__)
            return (audio,)

        if audio_tensor is None or audio_tensor.numel() == 0:
            log.warning("Empty audio input — passing through unchanged")
            return (audio,)

        original_tensor = audio_tensor.clone()

        # Ensure 3D: (batch, channels, samples)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        input_duration = audio_tensor.shape[-1] / sample_rate
        log.info("  Input: %.1fs, %d channels, %d Hz, batch=%d",
                 input_duration, audio_tensor.shape[-2], sample_rate, audio_tensor.shape[0])

        # -----------------------------------------------------------
        # 3. VRAM flush before model load
        # -----------------------------------------------------------
        log.info("  Flushing VRAM...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            log.info("  VRAM free after flush: %.0f MB", free_mb)

        # -----------------------------------------------------------
        # 4. Load Stable Audio Open (cached after first call)
        # -----------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = _get_or_load_model(device)

        if pipe is None:
            log.warning("Stable Audio Open not available — passing through unchanged")
            return (audio if isinstance(audio, dict) else {"waveform": original_tensor, "sample_rate": sample_rate},)

        # -----------------------------------------------------------
        # 5. Audio-to-audio style transfer per batch item
        #    - Cap generation at 47s (Stable Audio Open limit)
        #    - Trim or pad output to match original input length
        # -----------------------------------------------------------
        gen_duration = min(input_duration, 47.0)
        batch_size = audio_tensor.shape[0]
        enhanced_batches = []

        generator = torch.Generator(device=device).manual_seed(actual_seed)

        t0 = time.time()
        for b in range(batch_size):
            single_track = audio_tensor[b:b + 1]  # (1, channels, samples)

            if batch_size > 1:
                log.info("  Processing batch item %d/%d", b + 1, batch_size)

            try:
                # Run Stable Audio Open audio-to-audio
                # The pipeline handles encoding to latent, adding noise at
                # strength, denoising with prompt conditioning, and decoding
                result = pipe(
                    prompt=prompt,
                    audio=single_track,
                    audio_end_in_s=gen_duration,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    generator=generator,
                )

                # Extract audio tensor from pipeline output
                if hasattr(result, "audios"):
                    enh = result.audios  # (batch, channels, samples) numpy or tensor
                    if isinstance(enh, np.ndarray):
                        enh = torch.from_numpy(enh)
                    if enh.ndim == 2:
                        enh = enh.unsqueeze(0)
                elif hasattr(result, "audio"):
                    enh = result.audio
                    if isinstance(enh, np.ndarray):
                        enh = torch.from_numpy(enh)
                    if enh.ndim == 2:
                        enh = enh.unsqueeze(0)
                else:
                    log.error("Unexpected pipeline output: %s — using original", type(result))
                    enhanced_batches.append(single_track)
                    continue

                # Match length to original: trim if longer, crossfade-loop if shorter
                orig_samples = single_track.shape[-1]
                enh_samples = enh.shape[-1]
                if enh_samples >= orig_samples:
                    enh = enh[..., :orig_samples]
                else:
                    # Enhanced clip is shorter than input (47s cap hit).
                    # Use crossfade loop for seamless extension.
                    enh = crossfade_loop(enh, orig_samples, sample_rate=sample_rate)

                enhanced_batches.append(enh)

            except Exception as e:
                log.error("Enhancement failed on batch %d: %s — using original", b, e)
                enhanced_batches.append(single_track)
                continue

            # Vary seed for next batch item
            if b < batch_size - 1:
                generator = torch.Generator(device=device).manual_seed(actual_seed + b + 1)

        elapsed = time.time() - t0
        log.info("  Stable Audio processing: %.1fs (%d batch items)", elapsed, batch_size)

        # Stack enhanced batches
        enhanced_full = torch.cat(enhanced_batches, dim=0)

        # -----------------------------------------------------------
        # 6. Dry/wet mix
        # -----------------------------------------------------------
        result_tensor = mix_audio(original_tensor, enhanced_full, mix)

        output_duration = result_tensor.shape[-1] / sample_rate
        log.info("  Output: %.1fs, batch=%d | Enhancement complete", output_duration, result_tensor.shape[0])

        # -----------------------------------------------------------
        # 7. Repack into ComfyUI AUDIO dict
        # -----------------------------------------------------------
        result_audio = {"waveform": result_tensor, "sample_rate": sample_rate}
        return (result_audio,)

    @staticmethod
    def _resample(tensor: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Simple linear resampling."""
        if from_sr == to_sr:
            return tensor

        new_length = int(tensor.shape[-1] * (to_sr / from_sr))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        resampled = torch.nn.functional.interpolate(
            tensor.float(), size=new_length, mode="linear", align_corners=False
        )
        return resampled


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancerBypass
# ---------------------------------------------------------------------------
class DMM_MusicEnhancerBypass:
    """Simple bypass/passthrough node for A/B testing the enhancer.

    Wire this in place of DMM_MusicEnhancer to hear the original mix
    without enhancement, for comparison purposes.
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
    "DMM_MusicEnhancer": "DMM Music Enhancer (Stable Audio)",
    "DMM_MusicEnhancerBypass": "DMM Music Enhancer Bypass",
}