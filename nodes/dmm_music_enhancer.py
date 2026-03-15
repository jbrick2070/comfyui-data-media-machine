"""
DMM Music Enhancer — ACE-Step 1.5 audio enhancement for the DMM pipeline.

Takes the final mixed audio (narration + background music) from AudioMux
and runs it through ACE-Step's audio-to-audio mode to add cinematic texture,
richer orchestration, and generative depth — without replacing the original.

Architecture:
  AudioMux output (48kHz stereo) -> ACE-Step audio2audio -> Enhanced audio
  -> feeds into SaveVideo node

Requirements:
  - ACE-Step 1.5 installed (git clone https://github.com/ACE-Step/ACE-Step-1.5)
  - OR ACE-Step REST API running on localhost:8001
  - Model: acestep-v15-turbo (8 steps, fast inference, ~4GB VRAM)

VRAM Budget:
  - ACE-Step turbo: ~3-4GB VRAM
  - LTX-Video will be unloaded by this point in the pipeline
  - Safe on RTX 5080 (24GB) and RTX 4070+ (12GB+)

Author: Jeffrey A. Brick
Version: 3.5-beta
"""

import io
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

log = logging.getLogger("DMM.MusicEnhancer")


# ---------------------------------------------------------------------------
# ACE-Step API client (connects to the REST API)
# ---------------------------------------------------------------------------
class ACEStepClient:
    """Client for ACE-Step 1.5 REST API (localhost:8001 by default).

    Uses the audio-to-audio endpoint to enhance existing audio with
    generative music textures while preserving the original structure.
    """

    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url.rstrip("/")
        self.timeout = 120  # seconds — generous for long audio

    def health_check(self) -> bool:
        """Check if ACE-Step API is running."""
        try:
            url = f"{self.api_url}/health"
            if HAS_REQUESTS:
                r = requests.get(url, timeout=5)
                return r.status_code == 200
            else:
                req = urllib.request.Request(url)
                resp = urllib.request.urlopen(req, timeout=5)
                return resp.status == 200
        except Exception:
            return False

    def enhance_audio(
        self,
        audio_bytes: bytes,
        prompt: str,
        strength: float = 0.35,
        duration: float = None,
        seed: int = -1,
    ) -> bytes:
        """Send audio to ACE-Step for enhancement via audio-to-audio.

        Args:
            audio_bytes: WAV file bytes (48kHz stereo)
            prompt: Style prompt, e.g. "LA noir cinematic underscore,
                    subtle strings and ambient synth pads"
            strength: How much to transform (0.0 = no change, 1.0 = full regen).
                      0.25-0.40 is the sweet spot for enhancement.
            duration: Target duration in seconds (None = match input)
            seed: Random seed (-1 = random)

        Returns:
            Enhanced WAV file bytes
        """
        if HAS_REQUESTS:
            return self._enhance_requests(audio_bytes, prompt, strength, duration, seed)
        else:
            return self._enhance_urllib(audio_bytes, prompt, strength, duration, seed)

    def _enhance_requests(self, audio_bytes, prompt, strength, duration, seed):
        url = f"{self.api_url}/generate"
        files = {"audio": ("input.wav", audio_bytes, "audio/wav")}
        data = {
            "prompt": prompt,
            "audio2audio_strength": strength,
            "seed": seed,
            "model": "acestep-v15-turbo",
            "steps": 8,
        }
        if duration is not None:
            data["duration"] = duration

        r = requests.post(url, files=files, data=data, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"ACE-Step API error {r.status_code}: {r.text[:200]}")

        # Response may be JSON with a file path or direct audio bytes
        content_type = r.headers.get("content-type", "")
        if "audio" in content_type:
            return r.content
        else:
            # JSON response with file path or base64
            result = r.json()
            if "audio_path" in result:
                with open(result["audio_path"], "rb") as f:
                    return f.read()
            elif "audio" in result:
                import base64
                return base64.b64decode(result["audio"])
            else:
                raise RuntimeError(f"Unexpected API response: {list(result.keys())}")

    def _enhance_urllib(self, audio_bytes, prompt, strength, duration, seed):
        """Fallback for systems without requests library."""
        url = f"{self.api_url}/generate"

        # Build multipart form data manually
        boundary = "----DMMAceStepBoundary"
        body = b""

        # Add audio file
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="audio"; filename="input.wav"\r\n'
        body += b"Content-Type: audio/wav\r\n\r\n"
        body += audio_bytes
        body += b"\r\n"

        # Add text fields
        fields = {
            "prompt": prompt,
            "audio2audio_strength": str(strength),
            "seed": str(seed),
            "model": "acestep-v15-turbo",
            "steps": "8",
        }
        if duration is not None:
            fields["duration"] = str(duration)

        for key, val in fields.items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
            body += val.encode()
            body += b"\r\n"

        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=self.timeout)

        if resp.status != 200:
            raise RuntimeError(f"ACE-Step API error {resp.status}")
        return resp.read()


# ---------------------------------------------------------------------------
# Audio tensor <-> WAV conversion utilities
# ---------------------------------------------------------------------------
def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int = 48000) -> bytes:
    """Convert ComfyUI audio tensor to WAV bytes.

    ComfyUI audio tensors are typically shape (batch, channels, samples)
    with values in [-1.0, 1.0] float32.
    """
    if audio_tensor.ndim == 3:
        audio = audio_tensor[0]  # Take first batch
    elif audio_tensor.ndim == 2:
        audio = audio_tensor
    else:
        raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")

    channels = audio.shape[0]
    samples = audio.shape[1]

    # Convert to int16
    audio_np = audio.cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Interleave channels for WAV format
    interleaved = np.empty(channels * samples, dtype=np.int16)
    for ch in range(channels):
        interleaved[ch::channels] = audio_int16[ch]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())

    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Convert WAV bytes back to ComfyUI audio tensor.

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

    # De-interleave
    audio_np = audio_np.reshape(-1, channels).T  # (channels, samples)

    tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, channels, samples)
    return tensor, sample_rate


# ---------------------------------------------------------------------------
# Dry/wet mix utility
# ---------------------------------------------------------------------------
def mix_audio(dry: torch.Tensor, wet: torch.Tensor, mix: float) -> torch.Tensor:
    """Blend original (dry) and enhanced (wet) audio.

    mix=0.0 -> 100% original
    mix=1.0 -> 100% enhanced
    mix=0.5 -> equal blend
    """
    # Match lengths (pad shorter with zeros)
    dry_len = dry.shape[-1]
    wet_len = wet.shape[-1]
    if dry_len > wet_len:
        pad = torch.zeros(*wet.shape[:-1], dry_len - wet_len, device=wet.device)
        wet = torch.cat([wet, pad], dim=-1)
    elif wet_len > dry_len:
        wet = wet[..., :dry_len]

    mixed = (1.0 - mix) * dry + mix * wet

    # Soft clip to prevent overs
    mixed = torch.tanh(mixed)

    return mixed


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancer
# ---------------------------------------------------------------------------
class DMM_MusicEnhancer:
    """Enhances the pipeline's mixed audio using ACE-Step 1.5.

    Sits after AudioMux in the DMM pipeline. Takes the narration+music mix
    and adds generative cinematic texture via audio-to-audio transformation.

    The strength parameter controls how much ACE-Step changes the audio:
    - 0.15-0.25: Subtle texture (ambient pads, harmonic shimmer)
    - 0.25-0.40: Moderate enhancement (added instrumentation, richer feel)
    - 0.40-0.60: Strong transformation (significant new elements)
    - 0.60+: Heavy regen (original becomes a loose reference)

    Recommended: 0.30 for production, 0.20 for narration-heavy content.
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
                "prompt": (
                    "STRING",
                    {
                        "default": "LA noir cinematic underscore, subtle strings and ambient synth pads, warm analog feel",
                        "multiline": True,
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.30,
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
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**32 - 1,
                    },
                ),
                "api_url": (
                    "STRING",
                    {
                        "default": "http://localhost:8001",
                    },
                ),
            },
        }

    def enhance(
        self,
        audio: torch.Tensor,
        prompt: str,
        strength: float = 0.30,
        mix: float = 0.65,
        seed: int = -1,
        api_url: str = "http://localhost:8001",
    ) -> tuple[torch.Tensor]:
        """Main enhancement function called by ComfyUI."""

        log.info("DMM_MusicEnhancer: Starting audio enhancement")
        log.info("  Prompt: %s", prompt[:80])
        log.info("  Strength: %.2f | Mix: %.2f | Seed: %d", strength, mix, seed)

        # Validate input
        if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
            log.warning("Empty audio input — passing through unchanged")
            return (audio,)

        # Store original for dry/wet mix
        original = audio.clone()

        # Convert tensor to WAV bytes
        sample_rate = 48000  # DMM pipeline standard
        try:
            wav_bytes = tensor_to_wav_bytes(audio, sample_rate)
        except Exception as e:
            log.error("Failed to convert audio tensor to WAV: %s", e)
            return (audio,)

        input_duration = audio.shape[-1] / sample_rate
        log.info("  Input: %.1fs, %d channels, %d Hz",
                 input_duration, audio.shape[-2] if audio.ndim >= 2 else 1, sample_rate)

        # Connect to ACE-Step API
        client = ACEStepClient(api_url)

        if not client.health_check():
            log.warning(
                "ACE-Step API not available at %s — passing audio through unchanged. "
                "Start ACE-Step with: cd ACE-Step-1.5 && uv run acestep-api",
                api_url,
            )
            return (audio,)

        # Send to ACE-Step for enhancement
        t0 = time.time()
        try:
            enhanced_wav = client.enhance_audio(
                audio_bytes=wav_bytes,
                prompt=prompt,
                strength=strength,
                duration=input_duration,
                seed=seed,
            )
        except Exception as e:
            log.error("ACE-Step enhancement failed: %s — passing through original", e)
            return (audio,)

        elapsed = time.time() - t0
        log.info("  ACE-Step processing: %.1fs", elapsed)

        # Convert enhanced WAV back to tensor
        try:
            enhanced_tensor, enhanced_sr = wav_bytes_to_tensor(enhanced_wav)
        except Exception as e:
            log.error("Failed to decode enhanced audio: %s — passing through original", e)
            return (audio,)

        # Resample if needed (ACE-Step might output at different rate)
        if enhanced_sr != sample_rate:
            log.info("  Resampling from %d to %d Hz", enhanced_sr, sample_rate)
            enhanced_tensor = self._resample(enhanced_tensor, enhanced_sr, sample_rate)

        # Match channel count to original
        orig_channels = original.shape[-2] if original.ndim >= 2 else 1
        enh_channels = enhanced_tensor.shape[-2] if enhanced_tensor.ndim >= 2 else 1
        if enh_channels != orig_channels:
            if orig_channels == 2 and enh_channels == 1:
                enhanced_tensor = enhanced_tensor.repeat(1, 2, 1)
            elif orig_channels == 1 and enh_channels == 2:
                enhanced_tensor = enhanced_tensor[:, :1, :]

        # Dry/wet mix
        result = mix_audio(original, enhanced_tensor, mix)

        output_duration = result.shape[-1] / sample_rate
        log.info("  Output: %.1fs | Enhancement complete", output_duration)

        return (result,)

    @staticmethod
    def _resample(tensor: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
        """Simple linear resampling. For production, consider torchaudio.transforms.Resample."""
        if from_sr == to_sr:
            return tensor

        ratio = to_sr / from_sr
        new_length = int(tensor.shape[-1] * ratio)

        # Use torch interpolate for resampling
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

    def passthrough(self, audio: torch.Tensor) -> tuple[torch.Tensor]:
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
    "DMM_MusicEnhancer": "DMM Music Enhancer (ACE-Step)",
    "DMM_MusicEnhancerBypass": "DMM Music Enhancer Bypass",
}
