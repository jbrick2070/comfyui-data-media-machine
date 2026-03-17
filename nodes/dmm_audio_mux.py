"""
DMM_AudioMux — Replaces or mixes audio on a VIDEO object before SaveVideo.

Takes a VIDEO (from VideoConcat or BatchVideoGenerator) and an AUDIO (from
KokoroTTS or any ComfyUI audio source) and returns a new VIDEO with the
audio track replaced, mixed, or padded to match video duration.

v1.0  2026-03-14  Initial release.
v1.1  2026-03-14  Added background_music input — fills silence with music
                  instead of dead air.  Music plays under narration and
                  continues through procedural clip.
"""

import logging
import torch

log = logging.getLogger("DMM")


def _get_video_components(video_obj):
    """Extract (images_tensor, existing_audio, fps) from a VIDEO object."""
    # Try get_components() first (VideoFromComponents style)
    if hasattr(video_obj, "get_components"):
        comp = video_obj.get_components()
        if isinstance(comp, dict):
            return (comp.get("images"), comp.get("audio"), comp.get("fps", 25))
        if hasattr(comp, "images"):
            return (comp.images, getattr(comp, "audio", None),
                    getattr(comp, "fps", 25))

    # Try direct attributes (_ConcatVideo style)
    if hasattr(video_obj, "_images"):
        return (video_obj._images, getattr(video_obj, "_audio", None),
                getattr(video_obj, "_fps", 25))

    # Try dict-style
    if isinstance(video_obj, dict):
        return (video_obj.get("images"), video_obj.get("audio"),
                video_obj.get("fps", 25))

    return (None, None, 25)


def _build_video(images, audio, fps, original_video):
    """Reconstruct a VIDEO object with new audio, preserving the original type."""
    # Try VideoFromComponents constructor
    vid_cls = type(original_video)
    try:
        result = vid_cls(images=images, audio=audio, fps=fps)
        return result
    except Exception:
        pass

    # Try _ConcatVideo-style (from dmm_video_concat)
    try:
        from .dmm_video_concat import _build_video as _concat_build
        return _concat_build(images, audio, fps)
    except Exception:
        pass

    # Fallback: dict
    return {"images": images, "audio": audio, "fps": fps}


def _pad_or_trim_audio(audio, target_duration_sec, bg_music=None,
                       music_duck_db=-12.0):
    """Pad or trim audio to match video duration.

    If bg_music is provided, the music is layered underneath the narration
    for the full video duration (instead of padding with silence).  During
    the narration, music is ducked by *music_duck_db* dB; after narration
    ends, music plays at full volume.
    """
    if audio is None:
        return None

    waveform = audio.get("waveform") if isinstance(audio, dict) else audio
    sample_rate = int(audio.get("sample_rate", 24000)) if isinstance(audio, dict) else 24000

    if waveform is None:
        return audio

    # Ensure 3D: (batch, channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    target_samples = int(target_duration_sec * sample_rate)
    current_samples = waveform.shape[-1]
    narration_samples = current_samples  # remember original length

    if current_samples > target_samples:
        # Trim
        waveform = waveform[..., :target_samples]
        log.info("DMM_AudioMux: trimmed audio from %.1fs to %.1fs",
                 current_samples / sample_rate, target_duration_sec)
    elif current_samples < target_samples:
        pad_size = target_samples - current_samples

        if bg_music is not None:
            # ── Layer background music instead of silence ──────────────
            music_wf = bg_music.get("waveform") if isinstance(bg_music, dict) else bg_music
            music_sr = int(bg_music.get("sample_rate", 48000)) if isinstance(bg_music, dict) else 48000

            if music_wf is not None:
                # Ensure 3D
                if music_wf.dim() == 1:
                    music_wf = music_wf.unsqueeze(0).unsqueeze(0)
                elif music_wf.dim() == 2:
                    music_wf = music_wf.unsqueeze(0)

                # Resample music to match narration sample rate if needed
                if music_sr != sample_rate:
                    import torch.nn.functional as F
                    ratio = sample_rate / music_sr
                    new_len = int(music_wf.shape[-1] * ratio)
                    if music_wf.ndim == 2:
                        music_wf = F.interpolate(music_wf.unsqueeze(0).float(),
                                                  size=new_len, mode="linear",
                                                  align_corners=False).squeeze(0)
                    elif music_wf.ndim == 3:
                        music_wf = F.interpolate(music_wf.float(), size=new_len,
                                                  mode="linear",
                                                  align_corners=False)
                    log.info("DMM_AudioMux: resampled music %dHz → %dHz",
                             music_sr, sample_rate)

                # Match channel count to narration
                narr_ch = waveform.shape[1]
                mus_ch = music_wf.shape[1]
                if mus_ch < narr_ch:
                    music_wf = music_wf.repeat(1, narr_ch // mus_ch + 1, 1)[:, :narr_ch, :]
                elif mus_ch > narr_ch:
                    music_wf = music_wf[:, :narr_ch, :]

                # Pad or trim music to match target duration
                if music_wf.shape[-1] < target_samples:
                    # Loop music to fill
                    repeats = (target_samples // music_wf.shape[-1]) + 1
                    music_wf = music_wf.repeat(1, 1, repeats)[..., :target_samples]
                else:
                    music_wf = music_wf[..., :target_samples]

                # Build ducking envelope: quiet during narration, full after
                duck_linear = 10.0 ** (music_duck_db / 20.0)  # e.g. -12dB → 0.25
                envelope = torch.ones(target_samples, dtype=waveform.dtype,
                                      device=waveform.device)
                # Duck during narration
                envelope[:narration_samples] = duck_linear
                # Smooth crossfade at transition (0.5s ramp)
                ramp_samples = min(int(0.5 * sample_rate), target_samples - narration_samples)
                if ramp_samples > 0:
                    ramp = torch.linspace(duck_linear, 1.0, ramp_samples)
                    envelope[narration_samples:narration_samples + ramp_samples] = ramp

                music_wf = music_wf.float() * envelope.unsqueeze(0).unsqueeze(0)

                # Pad narration with zeros to target length, then add music
                narr_pad = torch.zeros(*waveform.shape[:-1], pad_size,
                                       dtype=waveform.dtype, device=waveform.device)
                waveform = torch.cat([waveform, narr_pad], dim=-1)

                # Mix: narration + music
                waveform = waveform.float() + music_wf.to(waveform.device).float()

                # Normalize to prevent clipping
                peak = waveform.abs().max()
                if peak > 0.95:
                    waveform = waveform * (0.95 / peak)

                log.info("DMM_AudioMux: layered background music (duck=%.0fdB during "
                         "narration, full after %.1fs, total %.1fs)",
                         music_duck_db, narration_samples / sample_rate,
                         target_duration_sec)

                return {"waveform": waveform, "sample_rate": sample_rate}

        # Fallback: pad with silence (no music provided)
        silence = torch.zeros(*waveform.shape[:-1], pad_size,
                              dtype=waveform.dtype, device=waveform.device)
        waveform = torch.cat([waveform, silence], dim=-1)
        log.info("DMM_AudioMux: padded audio from %.1fs to %.1fs (%.1fs silence)",
                 current_samples / sample_rate, target_duration_sec,
                 pad_size / sample_rate)

    return {"waveform": waveform, "sample_rate": sample_rate}


class DMMAudioMux:
    """Replaces or mixes audio on a VIDEO object."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "mux"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "mode": (["replace", "mix"],
                         {"default": "replace",
                          "tooltip": "replace: TTS only. mix: blend TTS with original audio."}),
                "mix_ratio": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When mode=mix, ratio of new audio (1.0 = all new, 0.0 = all original)"
                }),
                "pad_to_video": ("BOOLEAN", {"default": True,
                    "tooltip": "Pad or trim audio to match video duration"}),
                "background_music": ("AUDIO", {
                    "tooltip": "Optional background music (MIDI synth etc). "
                               "Plays under narration (ducked) and full volume after."
                }),
                "music_duck_db": ("FLOAT", {
                    "default": -6.0, "min": -30.0, "max": 0.0, "step": 1.0,
                    "tooltip": "How much to duck music during narration (-6dB = audible, -12dB = quiet, 0dB = no duck)"
                }),
            },
        }

    def mux(self, video, audio, mode="replace", mix_ratio=0.8,
            pad_to_video=True, background_music=None, music_duck_db=-6.0):

        images, orig_audio, fps = _get_video_components(video)

        if images is None:
            log.warning("DMM_AudioMux: could not extract frames from VIDEO")
            return (video,)

        n_frames = images.shape[0] if hasattr(images, "shape") else 0
        video_duration = n_frames / fps if fps > 0 else 0
        log.info("DMM_AudioMux: %d frames, %.1fs at %.1f fps", n_frames,
                 video_duration, fps)

        # Prepare new audio
        new_audio = audio
        if isinstance(new_audio, tuple):
            new_audio = new_audio[0]

        if pad_to_video and video_duration > 0:
            new_audio = _pad_or_trim_audio(new_audio, video_duration,
                                           bg_music=background_music,
                                           music_duck_db=music_duck_db)

        if mode == "mix" and orig_audio is not None:
            # Blend original + new
            try:
                orig_padded = _pad_or_trim_audio(orig_audio, video_duration)
                ow = orig_padded["waveform"] if isinstance(orig_padded, dict) else orig_padded
                nw = new_audio["waveform"] if isinstance(new_audio, dict) else new_audio
                sr = int(new_audio.get("sample_rate", 24000)) if isinstance(new_audio, dict) else 24000

                # Match dimensions
                min_len = min(ow.shape[-1], nw.shape[-1])
                ow = ow[..., :min_len]
                nw = nw[..., :min_len]

                mixed = mix_ratio * nw + (1.0 - mix_ratio) * ow
                mixed = mixed / (mixed.abs().max() + 1e-6)  # normalize
                new_audio = {"waveform": mixed, "sample_rate": sr}
                log.info("DMM_AudioMux: mixed audio (%.0f%% new, %.0f%% original)",
                         mix_ratio * 100, (1 - mix_ratio) * 100)
            except Exception as e:
                log.warning("DMM_AudioMux: mix failed (%s), using replace", e)

        result = _build_video(images, new_audio, fps, video)
        log.info("DMM_AudioMux: audio muxed successfully (%s mode)", mode)
        return (result,)
