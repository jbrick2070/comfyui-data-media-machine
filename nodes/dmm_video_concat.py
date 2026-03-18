"""
DMM_VideoConcat – Concatenate up to 6 VIDEO clips into one inside ComfyUI.

v3.6:
  - RTX Video Super Resolution upscaler (nvvfx, hardware-accelerated Tensor Cores)
  - Crossfade dissolve transitions between clips (configurable overlap)
  - Audio crossfade matches video overlap for seamless transitions

v3.5 (legacy):
  - SeedVR2 diffusion upscaler replaced with RTX VSR for faster, lighter upscaling

The VIDEO type is a VideoFromComponents object from comfy_api with methods:
  - get_components() → (images, audio, ...)
  - get_frame_rate() → float
  - get_frame_count() → int
  - get_dimensions() → (w, h)
  - get_duration() → float

This node extracts components from each clip, concatenates frames + audio,
optionally applies crossfade transitions, optionally upscales via NVIDIA RTX
Video Super Resolution, and rebuilds a new VideoFromComponents for SaveVideo.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger("media_machine.video_concat")

# Required properties for a valid VIDEO object passed to SaveVideo.
# 'images' = frame tensor, 'audio' = audio dict, 'fps' = frame rate.
# NOTE: The audio key is 'audio' (NOT 'audio_waveform').
# The audio dict itself contains {'waveform': tensor, 'sample_rate': int}.
REQUIRED_KEYS = {"images", "audio", "fps"}


# ---------------------------------------------------------------------------
# Extract components from a VideoFromComponents object
# ---------------------------------------------------------------------------

def _extract_components(video_obj):
    """
    Pull (images_tensor, audio_obj, fps) from a VIDEO / VideoFromComponents.
    Returns (images, audio, fps).
    """
    fps = 30.0

    # --- Primary: VideoFromComponents API ---
    if hasattr(video_obj, "get_components"):
        components = video_obj.get_components()
        logger.info("  get_components() returned type=%s", type(components).__name__)

        # get_components might return a dict, tuple, or named fields
        images = None
        audio = None

        if isinstance(components, dict):
            images = components.get("images")
            if images is None:
                images = components.get("frames")
            audio = components.get("audio")
            fps = float(components.get("fps", fps))
        elif isinstance(components, (tuple, list)):
            # Usually (images, audio) or (images, audio, fps)
            if len(components) >= 1:
                images = components[0]
            if len(components) >= 2:
                audio = components[1]
            if len(components) >= 3 and isinstance(components[2], (int, float)):
                fps = float(components[2])
        else:
            # Named tuple or dataclass (VideoComponents)
            images = getattr(components, "images", None)
            if images is None:
                images = getattr(components, "frames", None)
            audio = getattr(components, "audio", None)
            fps_val = getattr(components, "fps", None)
            if fps_val is not None:
                fps = float(fps_val)

        # Get fps from the video object itself if available
        if hasattr(video_obj, "get_frame_rate"):
            try:
                fr = video_obj.get_frame_rate()
                if fr and fr > 0:
                    fps = float(fr)
            except Exception:
                pass

        if images is not None:
            return images, audio, fps

    # --- Fallback: try get_stream_source for raw data ---
    if hasattr(video_obj, "get_stream_source"):
        logger.info("  Trying get_stream_source()")
        try:
            source = video_obj.get_stream_source()
            logger.info("  stream_source type=%s", type(source).__name__)
        except Exception as e:
            logger.warning("  get_stream_source() failed: %s", e)

    # --- Fallback: direct attribute access ---
    for img_attr in ("images", "frames", "samples", "video_frames"):
        if hasattr(video_obj, img_attr):
            val = getattr(video_obj, img_attr)
            if val is not None and isinstance(val, torch.Tensor):
                audio = getattr(video_obj, "audio", None)
                if hasattr(video_obj, "get_frame_rate"):
                    try:
                        fps = float(video_obj.get_frame_rate())
                    except Exception:
                        pass
                return val, audio, fps

    # --- Fallback: dict-like ---
    if isinstance(video_obj, dict):
        images = video_obj.get("images")
        if images is None:
            images = video_obj.get("frames")
        audio = video_obj.get("audio")
        fps = float(video_obj.get("fps", 30.0))
        if images is not None:
            return images, audio, fps

    # --- Debug dump ---
    attrs = [a for a in dir(video_obj) if not a.startswith("_")]
    logger.error(
        "DMM_VideoConcat: cannot extract from VIDEO. type=%s, attrs=%s",
        type(video_obj).__name__, attrs,
    )
    raise TypeError(
        f"DMM_VideoConcat: cannot extract frames from {type(video_obj).__name__}. "
        f"Available methods: {attrs}"
    )


class _ConcatVideo:
    """
    Lightweight VIDEO-compatible object that satisfies SaveVideo's interface.
    Mirrors VideoFromComponents: get_dimensions, get_frame_rate, get_frame_count,
    get_duration, get_components, save_to.
    """

    def __init__(self, images, audio, fps):
        self._images = images   # torch.Tensor [N, H, W, C]
        self._audio = audio     # dict with 'waveform' + 'sample_rate', or None
        self._fps = float(fps)

    def get_components(self):
        """Return a components object matching VideoComponents."""
        class _VC:
            pass
        vc = _VC()
        vc.images = self._images
        vc.audio = self._audio
        vc.fps = self._fps
        return vc

    def get_dimensions(self):
        """Return (width, height) — SaveVideo calls this at line 90."""
        return (int(self._images.shape[2]), int(self._images.shape[1]))

    def get_frame_rate(self):
        return self._fps

    def get_frame_count(self):
        return int(self._images.shape[0])

    def get_duration(self):
        return self._images.shape[0] / self._fps

    def get_stream_source(self):
        return None

    def get_container_format(self):
        return None

    def as_trimmed(self):
        return self

    def save_to(self, path, **kwargs):
        """Fallback save via PyAV — muxes video + audio if present."""
        import av
        import numpy as np

        container = None
        try:
            container = av.open(str(path), mode='w')
            w, h = self.get_dimensions()

            # Video stream
            video_stream = container.add_stream('h264', rate=int(self._fps))
            video_stream.width = w
            video_stream.height = h
            video_stream.pix_fmt = 'yuv420p'

            # Audio stream setup
            audio_stream = None
            wf_np = None
            sample_rate = 44100
            if self._audio is not None:
                try:
                    waveform = self._audio.get("waveform") if isinstance(self._audio, dict) else self._audio
                    sample_rate = int(self._audio.get("sample_rate", 44100)) if isinstance(self._audio, dict) else 44100
                    if waveform is not None:
                        wf = waveform.cpu().float()
                        if wf.dim() == 3:
                            wf = wf.squeeze(0)          # [channels, samples]
                        if wf.dim() == 1:
                            wf = wf.unsqueeze(0)        # [1, samples]
                        wf_np = wf.numpy()
                        n_channels = wf_np.shape[0]
                        layout = 'stereo' if n_channels >= 2 else 'mono'
                        if n_channels >= 2:
                            wf_np = wf_np[:2]           # keep stereo only
                        audio_stream = container.add_stream('aac', rate=sample_rate)
                        audio_stream.layout = layout
                        logger.info("_ConcatVideo: adding audio stream (%s, %dHz)", layout, sample_rate)
                except Exception as ae:
                    logger.warning("_ConcatVideo: audio stream setup failed, video-only: %s", ae)
                    audio_stream = None
                    wf_np = None

            # Encode video frames — per-frame is faster for large tensors (avoids 337MB alloc)
            for i in range(self._images.shape[0]):
                frame_data = (self._images[i].cpu().numpy() * 255).clip(0, 255).astype('uint8')
                frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

            # Encode audio frames
            if audio_stream is not None and wf_np is not None:
                try:
                    frame_size = audio_stream.codec_context.frame_size or 1024
                    total_samples = wf_np.shape[-1]
                    pts = 0
                    for start in range(0, total_samples, frame_size):
                        chunk = wf_np[:, start:start + frame_size]
                        audio_frame = av.AudioFrame.from_ndarray(chunk, format='fltp',
                                                                  layout=audio_stream.layout)
                        audio_frame.sample_rate = sample_rate
                        audio_frame.pts = pts
                        pts += chunk.shape[-1]
                        for packet in audio_stream.encode(audio_frame):
                            container.mux(packet)
                    for packet in audio_stream.encode():
                        container.mux(packet)
                    logger.info("_ConcatVideo: audio encoding complete (%d samples)", total_samples)
                except Exception as ae:
                    logger.warning("_ConcatVideo: audio encoding failed, video saved without audio: %s", ae)

            container.close()
            container = None
            logger.info("_ConcatVideo: saved %d frames to %s", self._images.shape[0], path)
        except Exception as e:
            logger.error("_ConcatVideo.save_to failed: %s", e)
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            raise


def _build_video(images, audio, fps):
    """
    Reconstruct a VIDEO-compatible object from concatenated components.

    First tries the real VideoFromComponents constructor (comfy_api),
    then falls back to our lightweight _ConcatVideo wrapper that
    implements the same interface SaveVideo expects.
    """
    # Try 1: Real VideoFromComponents
    try:
        from comfy_api.latest._input_impl.video_types import VideoFromComponents as VFC
        vid = VFC(images=images, audio=audio, fps=fps)
        # Verify it has the method SaveVideo needs
        if hasattr(vid, 'get_dimensions'):
            logger.info("_build_video: using real VideoFromComponents")
            return vid
    except Exception as e:
        logger.debug("VideoFromComponents init failed: %s", e)

    # Try 2: Our compatible wrapper
    logger.info("_build_video: using _ConcatVideo wrapper")
    return _ConcatVideo(images, audio, fps)


def _rtx_upscale(images, target_resolution=2160, quality="ULTRA"):
    """Upscale concatenated frame tensor via NVIDIA RTX Video Super Resolution.

    Uses nvvfx hardware-accelerated Tensor Core upscaling — much faster and
    lighter than diffusion-based upscalers. No large model downloads required.

    Args:
        images: (N, H, W, C) float tensor in NHWC format, values 0-1
        target_resolution: target height in pixels (2160=4K, 1080=HD)
        quality: "LOW", "MEDIUM", "HIGH", or "ULTRA"
    Returns:
        Upscaled (N, H_new, W_new, C) float tensor
    """
    import time

    try:
        import nvvfx
    except ImportError:
        raise ImportError(
            "nvvfx not installed. RTX Video Super Resolution requires the "
            "NVIDIA Video Effects SDK. Install Nvidia_RTX_Nodes_ComfyUI and "
            "its requirements."
        )

    n_frames, h, w, c = images.shape
    scale = target_resolution / h
    target_w = max(8, round((w * scale) / 8) * 8)
    target_h = max(8, round(target_resolution / 8) * 8)

    logger.info("RTX VSR upscale: %dx%d -> %dx%d (quality=%s) -- %d frames",
                w, h, target_w, target_h, quality, n_frames)
    t0 = time.time()

    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.ULTRA)

    # Max pixels per batch to avoid OOM — same limit as RTX node
    MAX_PIXELS = 1024 * 1024 * 16
    out_pixels = target_w * target_h
    batch_size = max(1, MAX_PIXELS // out_pixels)

    upscaled_batches = []

    with nvvfx.VideoSuperRes(selected_quality) as sr:
        sr.output_width = target_w
        sr.output_height = target_h
        sr.load()

        for i in range(0, n_frames, batch_size):
            batch = images[i:i + batch_size]
            batch_cuda = batch.cuda().permute(0, 3, 1, 2).contiguous()  # NCHW

            batch_outputs = []
            for j in range(batch_cuda.shape[0]):
                input_frame = batch_cuda[j]
                dlpack_out = sr.run(input_frame).image
                output = torch.from_dlpack(dlpack_out).clone()
                batch_outputs.append(output)

            batch_out_tensor = torch.stack(batch_outputs, dim=0)
            batch_out_tensor = batch_out_tensor.permute(0, 2, 3, 1).cpu()  # NHWC
            upscaled_batches.append(batch_out_tensor)

            if (i // batch_size) % 10 == 0:
                logger.info("  RTX VSR progress: %d/%d frames", min(i + batch_size, n_frames), n_frames)

    upscaled = torch.cat(upscaled_batches, dim=0)

    # Cleanup
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    fps_rate = n_frames / elapsed if elapsed > 0 else 0
    logger.info("RTX VSR upscale done in %.1fs (%.2f frames/sec, %d frames -> %dx%d)",
                elapsed, fps_rate, n_frames, target_w, target_h)
    return upscaled


def _crossfade_images(all_images, overlap_frames):
    """Apply crossfade dissolve transitions between sequential image tensors.

    For each pair of adjacent clips, the last `overlap_frames` of clip N
    are alpha-blended with the first `overlap_frames` of clip N+1.
    Total output length = sum(clip_lengths) - overlap_frames * (num_clips - 1).

    This produces smooth broadcast-quality dissolve transitions.
    """
    if overlap_frames <= 0 or len(all_images) < 2:
        return torch.cat(all_images, dim=0)

    result_parts = []
    for i, clip in enumerate(all_images):
        if i == 0:
            # First clip: keep everything except the tail overlap
            if clip.shape[0] > overlap_frames:
                result_parts.append(clip[:-overlap_frames])
            # Now blend the tail of clip[i] with the head of clip[i+1]
            if i + 1 < len(all_images):
                tail = clip[-overlap_frames:]
                head = all_images[i + 1][:overlap_frames]
                # Ensure both have enough frames
                blend_len = min(tail.shape[0], head.shape[0])
                tail = tail[:blend_len]
                head = head[:blend_len]
                # Alpha ramp: 0→1 over the blend region
                alpha = torch.linspace(0.0, 1.0, blend_len).view(-1, 1, 1, 1)
                alpha = alpha.to(tail.device)
                blended = (1.0 - alpha) * tail + alpha * head
                result_parts.append(blended)
        elif i == len(all_images) - 1:
            # Last clip: skip the head overlap (already blended above)
            if clip.shape[0] > overlap_frames:
                result_parts.append(clip[overlap_frames:])
            else:
                result_parts.append(clip)
        else:
            # Middle clip: skip head overlap (blended with prev), keep body,
            # then blend tail with next clip's head
            body_start = overlap_frames
            body_end = clip.shape[0] - overlap_frames
            if body_end > body_start:
                result_parts.append(clip[body_start:body_end])
            elif body_start < clip.shape[0]:
                # Clip is very short, just skip head overlap
                result_parts.append(clip[body_start:])

            # Blend tail with next clip's head
            if i + 1 < len(all_images):
                tail_start = max(body_end, body_start)
                tail = clip[tail_start:]
                if tail.shape[0] < overlap_frames:
                    # Pad with last frame if clip is too short
                    pad = clip[-1:].expand(overlap_frames - tail.shape[0], -1, -1, -1)
                    tail = torch.cat([tail, pad], dim=0)
                tail = tail[:overlap_frames]
                head = all_images[i + 1][:overlap_frames]
                blend_len = min(tail.shape[0], head.shape[0])
                tail = tail[:blend_len]
                head = head[:blend_len]
                alpha = torch.linspace(0.0, 1.0, blend_len).view(-1, 1, 1, 1)
                alpha = alpha.to(tail.device)
                blended = (1.0 - alpha) * tail + alpha * head
                result_parts.append(blended)

    valid_parts = [p for p in result_parts if p.shape[0] > 0]
    if not valid_parts:
        return torch.cat(all_images, dim=0)

    result = torch.cat(valid_parts, dim=0)
    orig_total = sum(c.shape[0] for c in all_images)
    logger.info("DMM_VideoConcat: crossfade applied — %d overlap frames × %d transitions, "
                "%d → %d total frames",
                overlap_frames, len(all_images) - 1, orig_total, result.shape[0])
    return result


def _crossfade_audio(audio_list, overlap_frames, fps):
    """Apply audio crossfade matching the video overlap.

    Converts overlap_frames to sample count and applies linear crossfade
    on the audio waveforms at each clip boundary.
    """
    valid = [a for a in audio_list if a is not None]
    if not valid or overlap_frames <= 0 or len(valid) < 2:
        return _concat_audio(audio_list)

    if not all(isinstance(a, dict) and "waveform" in a for a in valid):
        return _concat_audio(audio_list)

    sr = valid[0].get("sample_rate", 44100)

    # Verify all clips share the same sample rate — resample stragglers
    for i, a in enumerate(valid):
        clip_sr = a.get("sample_rate", 44100)
        if clip_sr != sr:
            logger.warning("_crossfade_audio: clip %d has sr=%d, expected %d — resampling",
                           i, clip_sr, sr)
            wf = a["waveform"]
            ratio = sr / clip_sr
            new_len = int(wf.shape[-1] * ratio)
            # Linear interpolation resample (good enough for concat boundaries)
            import torch.nn.functional as F
            # Waveform may be 2D (channels, samples) or 3D (batch, channels, samples)
            # F.interpolate linear mode needs exactly 3D
            orig_shape = wf.shape
            if wf.ndim == 2:
                wf_3d = wf.unsqueeze(0)
            elif wf.ndim == 3:
                wf_3d = wf
            else:
                wf_3d = wf.view(1, 1, -1)
            resampled = F.interpolate(wf_3d.float(),
                                       size=new_len, mode="linear")
            # Restore original dimensionality
            if len(orig_shape) == 2:
                resampled = resampled.squeeze(0)
            a["waveform"] = resampled
            a["sample_rate"] = sr

    overlap_sec = overlap_frames / fps
    overlap_samples = int(overlap_sec * sr)

    waveforms = [a["waveform"] for a in valid]
    result_parts = []

    for i, wf in enumerate(waveforms):
        n_samples = wf.shape[-1]
        if i == 0:
            # Keep everything except tail overlap
            if n_samples > overlap_samples:
                result_parts.append(wf[..., :-overlap_samples])
            # Blend tail with next head
            if i + 1 < len(waveforms):
                tail = wf[..., -overlap_samples:]
                head = waveforms[i + 1][..., :overlap_samples]
                blend_len = min(tail.shape[-1], head.shape[-1])
                alpha = torch.linspace(0.0, 1.0, blend_len).to(tail.device)
                # Broadcast alpha to match waveform dims
                while alpha.dim() < tail.dim():
                    alpha = alpha.unsqueeze(0)
                blended = (1.0 - alpha) * tail[..., :blend_len] + alpha * head[..., :blend_len]
                result_parts.append(blended)
        elif i == len(waveforms) - 1:
            # Last: skip head overlap
            if n_samples > overlap_samples:
                result_parts.append(wf[..., overlap_samples:])
            else:
                result_parts.append(wf)
        else:
            # Middle: skip head, keep body, blend tail with next
            body = wf[..., overlap_samples:-overlap_samples] if n_samples > 2 * overlap_samples else wf[..., overlap_samples:]
            result_parts.append(body)
            if i + 1 < len(waveforms):
                tail = wf[..., -overlap_samples:]
                head = waveforms[i + 1][..., :overlap_samples]
                blend_len = min(tail.shape[-1], head.shape[-1])
                alpha = torch.linspace(0.0, 1.0, blend_len).to(tail.device)
                while alpha.dim() < tail.dim():
                    alpha = alpha.unsqueeze(0)
                blended = (1.0 - alpha) * tail[..., :blend_len] + alpha * head[..., :blend_len]
                result_parts.append(blended)

    combined = torch.cat([p for p in result_parts if p.shape[-1] > 0], dim=-1)
    return {"waveform": combined, "sample_rate": sr}


def _concat_audio(audio_list):
    """
    Concatenate a list of AUDIO objects.
    AUDIO in ComfyUI is typically {"waveform": tensor, "sample_rate": int}.
    """
    valid = [a for a in audio_list if a is not None]
    if not valid:
        return None

    if all(isinstance(a, dict) and "waveform" in a for a in valid):
        sr = valid[0].get("sample_rate", 44100)
        # Resample any mismatched clips to target sr
        for i, a in enumerate(valid):
            clip_sr = a.get("sample_rate", 44100)
            if clip_sr != sr:
                logger.warning("_concat_audio: clip %d has sr=%d, expected %d — resampling",
                               i, clip_sr, sr)
                wf = a["waveform"]
                ratio = sr / clip_sr
                new_len = int(wf.shape[-1] * ratio)
                import torch.nn.functional as F
                # Waveform may be 2D (channels, samples) or 3D (batch, channels, samples)
                orig_shape = wf.shape
                if wf.ndim == 2:
                    wf_3d = wf.unsqueeze(0)
                elif wf.ndim == 3:
                    wf_3d = wf
                else:
                    wf_3d = wf.view(1, 1, -1)
                resampled = F.interpolate(wf_3d.float(),
                                           size=new_len, mode="linear")
                if len(orig_shape) == 2:
                    resampled = resampled.squeeze(0)
                a["waveform"] = resampled
        waveforms = [a["waveform"] for a in valid]
        combined = torch.cat(waveforms, dim=-1)
        return {"waveform": combined, "sample_rate": sr}

    if all(isinstance(a, torch.Tensor) for a in valid):
        return torch.cat(valid, dim=-1)

    # If mixed or unknown, just concat first valid's type
    logger.warning("DMM_VideoConcat: mixed audio types, using first clip's audio")
    return valid[0]


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------

class DMMVideoConcat:
    """
    Concatenates up to 5 VIDEO clips sequentially into a single VIDEO.
    Connect the LTX-2 subgraph outputs directly – no ffmpeg needed.
    """

    CATEGORY = "Data Media Machine"
    FUNCTION = "concatenate"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("VIDEO",),
                "upscale_4k": (["disabled", "enabled"], {"default": "disabled",
                               "tooltip": "Upscale final stitched video via NVIDIA RTX Video Super Resolution (hardware-accelerated Tensor Core upscaling)"}),
                "upscale_resolution": (["1080", "1440", "2160", "4320"], {"default": "2160",
                                       "tooltip": "Target height: 1080=HD, 1440=2K, 2160=4K, 4320=8K"}),
                "upscale_quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"], {"default": "ULTRA",
                                    "tooltip": "RTX VSR quality level. ULTRA = best quality, LOW = fastest. All levels are fast on RTX GPUs."}),
            },
            "optional": {
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "video_4": ("VIDEO",),
                "video_5": ("VIDEO",),
                "video_6": ("VIDEO",),
                "crossfade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Number of frames to crossfade between clips (0=hard cut, 10-15=smooth dissolve)"}),
            },
        }

    def concatenate(self, video_1, upscale_4k="disabled", upscale_resolution="2160", upscale_quality="ULTRA", video_2=None, video_3=None, video_4=None, video_5=None, video_6=None, crossfade_frames=0):
        clips = [v for v in [video_1, video_2, video_3, video_4, video_5, video_6] if v is not None]
        logger.info("DMM_VideoConcat: concatenating %d clips", len(clips))

        if len(clips) == 1 and upscale_4k != "enabled":
            return (clips[0],)

        all_images = []
        all_audio = []
        fps = 30.0

        for i, clip in enumerate(clips):
            images, audio, clip_fps = _extract_components(clip)
            frame_count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
            logger.info(
                "  Clip %d: %d frames, audio=%s, %.1f fps",
                i + 1, frame_count,
                "yes" if audio is not None else "no",
                clip_fps,
            )
            all_images.append(images)
            all_audio.append(audio)
            if i == 0:
                fps = clip_fps

        # Ensure all frame tensors match spatial dimensions
        if all(isinstance(img, torch.Tensor) for img in all_images):
            ref_shape = all_images[0].shape[1:]  # (H, W, C)
            for i in range(1, len(all_images)):
                if all_images[i].shape[1:] != ref_shape:
                    logger.warning(
                        "  Clip %d: resizing %s -> %s",
                        i + 1, all_images[i].shape[1:], ref_shape,
                    )
                    img = all_images[i].permute(0, 3, 1, 2)  # NCHW
                    img = torch.nn.functional.interpolate(
                        img,
                        size=(ref_shape[0], ref_shape[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    all_images[i] = img.permute(0, 2, 3, 1)  # NHWC

            # Apply crossfade or straight concat
            if crossfade_frames > 0 and len(all_images) > 1:
                combined_images = _crossfade_images(all_images, crossfade_frames)
            else:
                combined_images = torch.cat(all_images, dim=0)
        else:
            # Fallback: list concatenation
            combined_images = []
            for img_batch in all_images:
                if isinstance(img_batch, torch.Tensor):
                    combined_images.extend([img_batch[j] for j in range(img_batch.shape[0])])
                else:
                    combined_images.extend(img_batch)
            combined_images = torch.stack(combined_images, dim=0)

        # Audio: crossfade or straight concat
        if crossfade_frames > 0 and len(all_audio) > 1:
            combined_audio = _crossfade_audio(all_audio, crossfade_frames, fps)
        else:
            combined_audio = _concat_audio(all_audio)

        # RTX Video Super Resolution upscale if enabled
        if upscale_4k == "enabled":
            try:
                combined_images = _rtx_upscale(
                    combined_images,
                    target_resolution=int(upscale_resolution),
                    quality=upscale_quality,
                )
            except ImportError as e:
                logger.warning(
                    "DMM_VideoConcat: nvvfx not available -- %s. "
                    "Skipping upscale. Install Nvidia_RTX_Nodes_ComfyUI "
                    "and its requirements.",
                    e,
                )
            except Exception as e:
                logger.error("DMM_VideoConcat: RTX VSR upscale failed: %s", e)
                logger.warning("Continuing without upscale.")

        total_frames = combined_images.shape[0]
        duration = total_frames / fps
        logger.info(
            "DMM_VideoConcat: result = %d frames, %.1fs at %.0f fps",
            total_frames, duration, fps,
        )

        # Validate required keys before constructing result
        _components = {"images": combined_images, "audio": combined_audio, "fps": fps}
        missing = REQUIRED_KEYS - set(_components.keys())
        if missing:
            logger.error("DMM_VideoConcat: missing required keys: %s", missing)
        for key in REQUIRED_KEYS:
            if _components[key] is None and key != "audio":
                logger.error("DMM_VideoConcat: required key '%s' is None", key)

        # First: try cloning the native type from video_1 — avoids re-importing
        # VideoFromComponents and lets ComfyUI's SaveVideo handle audio natively.
        try:
            result = type(video_1)(images=combined_images, audio=combined_audio, fps=fps)
            logger.info("DMM_VideoConcat: rebuilt using native %s", type(video_1).__name__)
            return (result,)
        except Exception as e:
            logger.debug("Native type clone failed (%s), falling back to _build_video", e)

        # Fallback: _build_video tries VideoFromComponents import, then _ConcatVideo
        # (_ConcatVideo.save_to now muxes audio via PyAV)
        result = _build_video(combined_images, combined_audio, fps)
        return (result,)
