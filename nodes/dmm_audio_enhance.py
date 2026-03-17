"""
DMM_AudioEnhance — Upscales mono 24kHz TTS audio to stereo 48kHz with
faux-spatial widening via Haas effect and subtle stereo decorrelation.

Pipeline position:  KokoroTTS → AudioEnhance → AudioMux → SaveVideo

Processing chain:
  1. Resample 24kHz → 48kHz (sinc interpolation via linear interp fallback)
  2. Mono → Stereo duplication
  3. Haas-effect spatial widening (delay one channel 0.2–0.8 ms)
  4. Subtle stereo decorrelation (low-shelf EQ difference between L/R)
  5. Normalize to -1.0 dBFS peak

Input:   AUDIO  (mono or stereo, any sample rate — typically 24kHz from Kokoro)
Output:  AUDIO  (stereo, 48kHz, spatially widened)

v1.0  2026-03-14  Initial release — real audio upscale experiment.
"""

import logging
import math
import torch

log = logging.getLogger("DMM")


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform from orig_sr to target_sr using linear interpolation.

    For production quality we'd use torchaudio.transforms.Resample, but this
    avoids the torchaudio dependency.  Linear interp at 2× (24k→48k) is clean
    enough for speech — no aliasing risk when upsampling.
    """
    if orig_sr == target_sr:
        return waveform

    ratio = target_sr / orig_sr
    orig_len = waveform.shape[-1]
    new_len = int(orig_len * ratio)

    # Use torch interpolate on the last dimension
    # Needs shape (batch, channels, length) → treat as 1-D signal
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    resampled = torch.nn.functional.interpolate(
        waveform.float(), size=new_len, mode="linear", align_corners=False
    )
    return resampled


def _mono_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Convert mono (B,1,N) to stereo (B,2,N) by duplicating the channel."""
    if waveform.shape[1] >= 2:
        return waveform  # already stereo or more
    return torch.cat([waveform, waveform], dim=1)


def _haas_delay(waveform: torch.Tensor, sample_rate: int,
                delay_ms: float = 0.4) -> torch.Tensor:
    """Apply Haas effect: delay the right channel by delay_ms milliseconds.

    The Haas effect (precedence effect) creates a perception of spatial width
    without altering perceived loudness.  Delays of 0.2–0.8 ms produce a wide
    stereo image for speech without echo artifacts.
    """
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    if delay_samples < 1 or waveform.shape[1] < 2:
        return waveform

    B, C, N = waveform.shape
    # Create delayed right channel — pad front with silence, trim end
    silence = torch.zeros(B, 1, delay_samples, dtype=waveform.dtype,
                          device=waveform.device)
    right_delayed = torch.cat([silence, waveform[:, 1:2, :]], dim=-1)[:, :, :N]

    return torch.cat([waveform[:, 0:1, :], right_delayed], dim=1)


def _stereo_decorrelate(waveform: torch.Tensor, amount: float = 0.15) -> torch.Tensor:
    """Subtle stereo decorrelation via mid-side processing.

    Boosts the 'side' (L-R) component by `amount` to widen the image,
    then recombines.  This is the same technique used in mastering studios
    for stereo width enhancement.

    amount: 0.0 = mono, 0.15 = subtle width, 0.5 = very wide
    """
    if waveform.shape[1] < 2:
        return waveform

    left = waveform[:, 0:1, :]
    right = waveform[:, 1:2, :]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # Boost side signal
    side = side * (1.0 + amount)

    # Recombine
    new_left = mid + side
    new_right = mid - side

    return torch.cat([new_left, new_right], dim=1)


def _normalize(waveform: torch.Tensor, target_dbfs: float = -1.0) -> torch.Tensor:
    """Peak-normalize waveform to target_dbfs.

    -1.0 dBFS leaves a tiny bit of headroom to avoid clipping after
    any downstream processing.
    """
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform  # silence, don't amplify noise

    target_linear = 10.0 ** (target_dbfs / 20.0)
    return waveform * (target_linear / peak)


def _apply_bass_warmth(waveform: torch.Tensor, sample_rate: int,
                       warmth: float = 0.1) -> torch.Tensor:
    """Add subtle low-frequency warmth via a simple one-pole low-pass blend.

    This gives the TTS voice a slightly richer, broadcast-quality tone.
    warmth: 0.0 = none, 0.1 = subtle, 0.3 = noticeable
    """
    if warmth <= 0.0:
        return waveform

    # Simple one-pole LPF at ~200 Hz
    cutoff = 200.0
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    # Process each channel
    B, C, N = waveform.shape
    filtered = torch.zeros_like(waveform)
    filtered[:, :, 0] = waveform[:, :, 0]

    for i in range(1, N):
        filtered[:, :, i] = filtered[:, :, i - 1] + alpha * (
            waveform[:, :, i] - filtered[:, :, i - 1]
        )

    # Blend: original + warmth * low_passed
    return waveform + warmth * filtered


class DMMAudioEnhance:
    """Upscales TTS audio: 24kHz→48kHz, mono→stereo, faux-spatial widening."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "enhance"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "target_sample_rate": ("INT", {
                    "default": 48000, "min": 24000, "max": 96000, "step": 8000,
                    "tooltip": "Target sample rate in Hz (48000 = broadcast standard)"
                }),
                "spatial_width": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stereo spatial width: 0=mono center, 0.3=natural, 1.0=extreme wide"
                }),
                "haas_delay_ms": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Haas effect delay in ms (0.2-0.8 = natural spatial, 0=disabled)"
                }),
                "bass_warmth": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Low-frequency warmth for broadcast tone (0=off, 0.1=subtle)"
                }),
                "normalize_dbfs": ("FLOAT", {
                    "default": -1.0, "min": -12.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Peak normalization target in dBFS (-1.0 = standard broadcast)"
                }),
            },
        }

    def enhance(self, audio, target_sample_rate=48000, spatial_width=0.3,
                haas_delay_ms=0.4, bass_warmth=0.1, normalize_dbfs=-1.0):

        # --- Extract waveform & sample rate ---
        if isinstance(audio, tuple):
            audio = audio[0]

        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            orig_sr = int(audio.get("sample_rate", 24000))
        else:
            waveform = audio
            orig_sr = 24000

        if waveform is None:
            log.warning("DMM_AudioEnhance: no waveform data, passing through")
            return (audio,)

        # Ensure 3D: (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        orig_channels = waveform.shape[1]
        orig_samples = waveform.shape[-1]
        orig_duration = orig_samples / orig_sr

        log.info("DMM_AudioEnhance: input %dHz %dch %d samples (%.1fs)",
                 orig_sr, orig_channels, orig_samples, orig_duration)

        # --- Step 1: Resample to target rate ---
        waveform = _resample(waveform, orig_sr, target_sample_rate)
        log.info("DMM_AudioEnhance: resampled %dHz → %dHz (%d samples)",
                 orig_sr, target_sample_rate, waveform.shape[-1])

        # --- Step 2: Mono → Stereo ---
        waveform = _mono_to_stereo(waveform)
        log.info("DMM_AudioEnhance: %dch → %dch",
                 orig_channels, waveform.shape[1])

        # --- Step 3: Bass warmth (before spatial, so it's centered) ---
        if bass_warmth > 0:
            waveform = _apply_bass_warmth(waveform, target_sample_rate, bass_warmth)
            log.info("DMM_AudioEnhance: bass warmth %.2f applied", bass_warmth)

        # --- Step 4: Haas-effect spatial delay ---
        if haas_delay_ms > 0:
            waveform = _haas_delay(waveform, target_sample_rate, haas_delay_ms)
            log.info("DMM_AudioEnhance: Haas delay %.1fms applied", haas_delay_ms)

        # --- Step 5: Stereo decorrelation (mid-side widening) ---
        if spatial_width > 0:
            waveform = _stereo_decorrelate(waveform, spatial_width)
            log.info("DMM_AudioEnhance: stereo width %.2f applied", spatial_width)

        # --- Step 6: Peak normalize ---
        waveform = _normalize(waveform, normalize_dbfs)
        log.info("DMM_AudioEnhance: normalized to %.1f dBFS", normalize_dbfs)

        # --- Final output ---
        enhanced = {"waveform": waveform, "sample_rate": target_sample_rate}

        final_samples = waveform.shape[-1]
        final_duration = final_samples / target_sample_rate
        log.info("DMM_AudioEnhance: output %dHz %dch %d samples (%.1fs) — "
                 "spatial width=%.2f, Haas=%.1fms, warmth=%.2f",
                 target_sample_rate, waveform.shape[1], final_samples,
                 final_duration, spatial_width, haas_delay_ms, bass_warmth)

        return (enhanced,)
