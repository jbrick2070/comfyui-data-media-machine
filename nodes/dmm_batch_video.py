"""
DMM_BatchVideoGenerator – generate N video clips in a single node execution.

v3.4 changes:
  - Removed dead two-pass upscaler code (Phase 3B/3C) — RTX upscale lives
    in VideoConcat now (upscales the final stitched video, not per-clip)
  - Removed upscale_model input and sigmas_stage2 parameter
  - Single-pass is the only pipeline mode

v3.2 changes:
  - RTX upscale moved to VideoConcat node

v3.1 changes:
  - Fixed: CATEGORY "DMM" → "DataMediaMachine" (consistent with all other nodes)
  - Replaced 31 print() calls with log.info() for consistent logging
  - Added per-phase timing summary at end

The model, VAE, CLIP, sampler, sigmas etc. are loaded ONCE and stay resident
in VRAM / pinned-RAM for the entire loop.  ComfyUI's memory manager never
gets a chance to tear-down between clips because everything happens inside
one FUNCTION call.

Pipeline: Stage 1 sampling → VAEDecodeTiled + AudioVAEDecode → CreateVideo
RTX upscale happens downstream in VideoConcat (post-stitch).

v2.5 optimizations:
  - VRAM resident mode (tries HIGH_VRAM state to reduce offloading)
  - torch.inference_mode() for reduced autograd overhead
  - Pre-computed noise + guiders outside sampling loops
  - CUDA synchronization barriers for accurate phase timing
  - Explicit intermediate tensor cleanup between phases
"""

from __future__ import annotations

import gc, logging, time
from contextlib import contextmanager
from typing import Any

import torch
import comfy.utils
import comfy.model_management
log = logging.getLogger("DMM.BatchVideo")


# ---------------------------------------------------------------------------
# VRAM lock helper – safe fallback if ComfyUI API doesn't exist
# ---------------------------------------------------------------------------
_lock_checked = False
_lock_works = False


@contextmanager
def _vram_lock():
    """Lock model in VRAM if push_model_lock() is available, otherwise no-op.

    Prevents DynamicVRAM from re-preparing the model on every iteration.
    Falls back gracefully if the API doesn't exist in this ComfyUI version.
    """
    global _lock_checked, _lock_works
    if not _lock_checked:
        try:
            ctx = comfy.utils.get_context_stack().push_model_lock()
            _lock_works = True
            _lock_checked = True
            log.info("DMM_BatchVideo: push_model_lock() available — VRAM locking enabled")
            with ctx:
                yield
            return
        except (AttributeError, TypeError) as e:
            _lock_works = False
            _lock_checked = True
            log.info("DMM_BatchVideo: push_model_lock() unavailable (%s) — using pre-load fallback", e)
            yield
            return

    if _lock_works:
        with comfy.utils.get_context_stack().push_model_lock():
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# VRAM state helper – boost VRAM allocation when vram_resident is enabled
# ---------------------------------------------------------------------------
@contextmanager
def _noop_boost():
    """No-op context manager when VRAM boost is disabled."""
    yield False


@contextmanager
def _vram_boost():
    """Temporarily set ComfyUI to HIGH_VRAM state to reduce offloading.

    On a 16GB card with a 21GB FP8 model, this won't keep the entire model
    resident, but it raises the threshold for when ComfyUI decides to offload,
    keeping more layers hot in VRAM and reducing async-stream overhead.
    Restores original state on exit.
    """
    original_state = None
    boosted = False
    try:
        # Try the modern API first
        original_state = getattr(comfy.model_management, 'vram_state', None)
        if hasattr(comfy.model_management, 'VRAMState'):
            high = comfy.model_management.VRAMState.HIGH_VRAM
            if hasattr(comfy.model_management, 'set_vram_state'):
                comfy.model_management.set_vram_state(high)
                boosted = True
                log.info("VRAM_BOOST: set HIGH_VRAM state")
            elif original_state is not None:
                comfy.model_management.vram_state = high
                boosted = True
                log.info("VRAM_BOOST: set HIGH_VRAM state (direct)")
        if not boosted:
            log.info("VRAM_BOOST: VRAMState API unavailable, skipping")
    except Exception as e:
        log.warning("VRAM_BOOST: failed to set HIGH_VRAM (%s), continuing normally", e)
        boosted = False

    try:
        yield boosted
    finally:
        if boosted and original_state is not None:
            try:
                if hasattr(comfy.model_management, 'set_vram_state'):
                    comfy.model_management.set_vram_state(original_state)
                else:
                    comfy.model_management.vram_state = original_state
                log.info("VRAM_BOOST: restored original VRAM state")
            except Exception as e:
                log.warning("Failed to restore VRAM state: %s", e)


# ---------------------------------------------------------------------------
# Lazy node-class resolver – these are only available after ComfyUI starts
# ---------------------------------------------------------------------------
_NODE_CACHE: dict[str, Any] = {}


def _node(name: str):
    """Fetch a ComfyUI node class by its registered name."""
    if name not in _NODE_CACHE:
        from nodes import NODE_CLASS_MAPPINGS  # noqa: late import
        cls = NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            raise RuntimeError(f"Node '{name}' not found in NODE_CLASS_MAPPINGS")
        _NODE_CACHE[name] = cls
    return _NODE_CACHE[name]


def _call(name: str, **kwargs):
    """Instantiate a node and call its FUNCTION, return raw tuple."""
    cls = _node(name)
    fn = getattr(cls, "FUNCTION", "execute")
    obj = cls()
    result = getattr(obj, fn)(**kwargs)
    # Some nodes return io.NodeOutput, unwrap it
    if hasattr(result, "args"):
        return result.args
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_sigmas(sigma_str: str) -> list[float]:
    return [float(x.strip()) for x in sigma_str.split(",") if x.strip()]


def _make_sigmas_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _cuda_sync():
    """Synchronize CUDA for accurate timing. No-op if CUDA unavailable."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class DMMBatchVideoGenerator:
    """
    Generate up to 5 LTX-2 AV video clips in one node execution.

    Models stay loaded across all clips – zero re-prep overhead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "sampler_name": (["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde"],
                                 {"default": "euler"}),
                "sigmas_stage1": ("STRING", {
                    "default": "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
                    "multiline": False,
                    "tooltip": "Comma-separated sigma schedule for Stage 1",
                }),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "length": ("INT", {"default": 61, "min": 9, "max": 257, "step": 8,
                                   "tooltip": "Number of frames (pixel-space)"}),
                "fps": ("INT", {"default": 35, "min": 1, "max": 60}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF,
                                "tooltip": "Base seed — each clip increments by 1"}),
                "batch_size": ("INT", {"default": 5, "min": 1, "max": 1024,
                                      "tooltip": "Number of clips to generate (clamped to 5 internally)"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 1024, "step": 64,
                                      "tooltip": "VAEDecodeTiled tile size"}),
                "tile_stride": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16}),
                "crf": ("INT", {"default": 30, "min": 0, "max": 51,
                                "tooltip": "CreateVideo CRF quality"}),
                "vram_resident": (["enabled", "disabled"], {"default": "enabled",
                                                             "tooltip": "HIGH_VRAM mode — reduces offloading threshold (may OOM on <16GB)"}),
            },
            "optional": {
                "prompt_1": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "Text prompt for clip 1"}),
                "prompt_2": ("STRING", {"default": "", "multiline": True}),
                "prompt_3": ("STRING", {"default": "", "multiline": True}),
                "prompt_4": ("STRING", {"default": "", "multiline": True}),
                "prompt_5": ("STRING", {"default": "", "multiline": True}),
                # v3.0 i2v conditioning: optional webcam frames + strength per clip
                "image_1": ("IMAGE", {"tooltip": "Conditioning image for clip 1 (i2v mode)"}),
                "image_2": ("IMAGE", {"tooltip": "Conditioning image for clip 2 (i2v mode)"}),
                "image_3": ("IMAGE", {"tooltip": "Conditioning image for clip 3 (i2v mode)"}),
                "image_4": ("IMAGE", {"tooltip": "Conditioning image for clip 4 (i2v mode)"}),
                "image_5": ("IMAGE", {"tooltip": "Conditioning image for clip 5 (i2v mode)"}),
                "cond_strength_1": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "i2v conditioning strength for clip 1 (0=pure t2v, 1=strong i2v)"}),
                "cond_strength_2": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cond_strength_3": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cond_strength_4": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cond_strength_5": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("VIDEO", "VIDEO", "VIDEO", "VIDEO", "VIDEO")
    RETURN_NAMES = ("video_1", "video_2", "video_3", "video_4", "video_5")
    OUTPUT_IS_LIST = (False, False, False, False, False)
    FUNCTION = "generate"
    CATEGORY = "DataMediaMachine"

    # ------------------------------------------------------------------
    def _encode_prompt(self, clip, text: str):
        """CLIPTextEncode → CONDITIONING."""
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens_scheduled(tokens)
        return output

    def _make_sampler(self, sampler_name: str):
        """KSamplerSelect → SAMPLER object."""
        return _call("KSamplerSelect", sampler_name=sampler_name)[0]

    def _make_sigmas(self, sigma_values: list[float]):
        """ManualSigmas → SIGMAS tensor."""
        return _make_sigmas_tensor(sigma_values)

    def _make_noise(self, seed: int):
        """RandomNoise → NOISE object."""
        return _call("RandomNoise", noise_seed=seed)[0]

    def _make_guider(self, model, positive, negative, cfg: float):
        """CFGGuider → GUIDER object."""
        return _call("CFGGuider", model=model, positive=positive,
                      negative=negative, cfg=cfg)[0]

    def _conditioning(self, positive, negative, frame_rate: int):
        """LTXVConditioning → (pos_out, neg_out)."""
        result = _call("LTXVConditioning",
                        positive=positive, negative=negative,
                        frame_rate=frame_rate)
        return result[0], result[1]  # positive, negative

    def _apply_i2v_conditioning(self, vae, image, latent, strength):
        """LTXVImgToVideoConditionOnly → LATENT with image conditioning.
        Encodes the image into the first frames of the video latent and
        creates a noise mask to control conditioning strength."""
        return _call("LTXVImgToVideoConditionOnly",
                      vae=vae, image=image, latent=latent,
                      strength=strength)[0]

    def _empty_video_latent(self, width, height, length, batch=1):
        """EmptyLTXVLatentVideo → LATENT."""
        return _call("EmptyLTXVLatentVideo",
                      width=width, height=height,
                      length=length, batch_size=batch)[0]

    def _empty_audio_latent(self, audio_vae, frames_number, frame_rate, batch_size=1):
        """LTXVEmptyLatentAudio → LATENT."""
        return _call("LTXVEmptyLatentAudio",
                      audio_vae=audio_vae,
                      frames_number=frames_number,
                      frame_rate=frame_rate,
                      batch_size=batch_size)[0]

    def _concat_av(self, video_latent, audio_latent):
        """LTXVConcatAVLatent → LATENT."""
        return _call("LTXVConcatAVLatent",
                      video_latent=video_latent,
                      audio_latent=audio_latent)[0]

    def _separate_av(self, av_latent):
        """LTXVSeparateAVLatent → (video_latent, audio_latent)."""
        result = _call("LTXVSeparateAVLatent", av_latent=av_latent)
        return result[0], result[1]

    def _sample(self, noise, guider, sampler, sigmas, latent_image):
        """SamplerCustomAdvanced → (output, denoised)."""
        result = _call("SamplerCustomAdvanced",
                        noise=noise, guider=guider,
                        sampler=sampler, sigmas=sigmas,
                        latent_image=latent_image)
        return result[0], result[1]

    def _vae_decode_tiled(self, samples, vae, tile_size, overlap, temporal_size, temporal_overlap):
        """VAEDecodeTiled → IMAGE."""
        return _call("VAEDecodeTiled",
                      samples=samples, vae=vae,
                      tile_size=tile_size, overlap=overlap,
                      temporal_size=temporal_size,
                      temporal_overlap=temporal_overlap)[0]

    def _audio_vae_decode(self, av_latent, audio_vae):
        """LTXVAudioVAEDecode → AUDIO."""
        return _call("LTXVAudioVAEDecode",
                      samples=av_latent, audio_vae=audio_vae)[0]

    def _create_video(self, images, audio, fps):
        """CreateVideo → VIDEO."""
        return _call("CreateVideo",
                      images=images, audio=audio, fps=fps)[0]

    def generate(
        self,
        model, clip, vae, audio_vae,
        sampler_name, sigmas_stage1,
        width, height, length, fps, cfg, seed, batch_size,
        tile_size, tile_stride, crf, vram_resident="disabled",
        prompt_1="", prompt_2="", prompt_3="", prompt_4="", prompt_5="",
        image_1=None, image_2=None, image_3=None, image_4=None, image_5=None,
        cond_strength_1=0.75, cond_strength_2=0.75, cond_strength_3=0.75,
        cond_strength_4=0.75, cond_strength_5=0.75,
    ):
        batch_size = min(int(batch_size), 5)  # clamp to 5 max clips
        prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5][:batch_size]
        prompts = [p if p else f"Cinematic aerial view of a modern city, clip {i+1}"
                   for i, p in enumerate(prompts)]
        n = len(prompts)

        # Collect i2v conditioning images + strengths (None = pure t2v for that clip)
        images = [image_1, image_2, image_3, image_4, image_5][:batch_size]
        cond_strengths = [cond_strength_1, cond_strength_2, cond_strength_3,
                          cond_strength_4, cond_strength_5][:batch_size]

        use_vram_boost = vram_resident == "enabled"

        # Pre-build shared lightweight objects ONCE
        sampler_obj = self._make_sampler(sampler_name)
        sig1 = self._make_sigmas(_parse_sigmas(sigmas_stage1))

        total_start = time.time()

        i2v_count = sum(1 for img in images if img is not None)
        log.info("Pipeline: Encode → Prepare → Sample → Decode → Mux")
        if i2v_count > 0:
            log.info("i2v conditioning: %d/%d clips have images", i2v_count, n)

        # Wrap entire pipeline in VRAM boost + inference_mode for max throughput
        with _vram_boost() if use_vram_boost else _noop_boost():
            with torch.inference_mode():
                return self._run_pipeline(
                    model, clip, vae, audio_vae,
                    sampler_obj, sig1,
                    width, height, length, fps, cfg, seed,
                    tile_size, tile_stride,
                    prompts, n, total_start,
                    images=images, cond_strengths=cond_strengths,
                )

    def _run_pipeline(
        self, model, clip, vae, audio_vae,
        sampler_obj, sig1,
        width, height, length, fps, cfg, seed,
        tile_size, tile_stride,
        prompts, n, total_start,
        images=None, cond_strengths=None,
    ):
        if images is None:
            images = [None] * n
        if cond_strengths is None:
            cond_strengths = [0.75] * n
        # =============================================================
        # ENCODE — text conditioning  (Text Encoder stays in VRAM)
        # =============================================================
        log.info("[Encode] %d prompts...", n)
        phase_start = time.time()

        log.info("  negative prompt (shared)...")
        base_negative = self._encode_prompt(clip, "")

        conditionings = []
        for i, prompt_text in enumerate(prompts):
            log.info("  prompt %d: %s...", i+1, prompt_text[:70])
            positive = self._encode_prompt(clip, prompt_text)
            pos_cond, neg_cond = self._conditioning(positive, base_negative, fps)
            conditionings.append((pos_cond, neg_cond))

        _cuda_sync()
        log.info("[Encode] done in %.1fs", time.time() - phase_start)

        # =============================================================
        # PREPARE — latent pairs + i2v conditioning
        # =============================================================
        log.info("[Prepare] %d latent pairs...", n)
        base_latents = []
        for i, _ in enumerate(prompts):
            video_latent = self._empty_video_latent(width, height, length)

            if i < len(images) and images[i] is not None:
                strength = cond_strengths[i] if i < len(cond_strengths) else 0.75
                try:
                    video_latent = self._apply_i2v_conditioning(
                        vae, images[i], video_latent, strength
                    )
                    log.info("  clip %d: i2v conditioning (strength=%.2f)", i+1, strength)
                except Exception as e:
                    log.warning("  clip %d: i2v failed (%s) — falling back to t2v", i+1, e)

            audio_latent = self._empty_audio_latent(audio_vae, length, fps)
            base_latents.append(self._concat_av(video_latent, audio_latent))

        # =============================================================
        # SAMPLE — diffusion  (Main Model locked in VRAM)
        # =============================================================
        log.info("[Sample] %d clips...", n)
        phase_start = time.time()

        # Pre-compute noise + guiders before entering the hot loop
        noise_objs = [self._make_noise(seed + i) for i in range(n)]
        guiders = [self._make_guider(model, conditionings[i][0], conditionings[i][1], cfg) for i in range(n)]

        sampled_latents = []

        try:
            comfy.model_management.load_models_gpu([model])
            log.info("  model pre-loaded to GPU")
        except Exception as e:
            log.debug("  pre-load skipped: %s", e)

        with _vram_lock():
            for i in range(n):
                log.info("  clip %d/%d seed=%d (%d steps)...", i+1, n, seed+i, len(sig1)-1)
                _, denoised = self._sample(noise_objs[i], guiders[i], sampler_obj, sig1, base_latents[i])
                sampled_latents.append(denoised)

        _cuda_sync()
        if sampled_latents:
            dev = sampled_latents[0]["samples"].device if isinstance(sampled_latents[0], dict) else "unknown"
            log.info("[Sample] done in %.1fs (device=%s)", time.time() - phase_start, dev)

        del noise_objs, guiders, base_latents, conditionings
        gc.collect()

        # =============================================================
        # DECODE — VAE to pixels + audio
        # =============================================================
        log.info("[Decode] %d clips...", n)
        phase_start = time.time()

        try:
            comfy.model_management.load_models_gpu([vae])
        except Exception as e:
            log.debug("  VAE pre-load skipped: %s", e)

        all_frames = []
        all_audio = []
        for i in range(n):
            log.info("  clip %d/%d...", i+1, n)
            vid_final, aud_final = self._separate_av(sampled_latents[i])
            frames = self._vae_decode_tiled(vid_final, vae,
                                            tile_size=tile_size,
                                            overlap=tile_stride,
                                            temporal_size=4096,
                                            temporal_overlap=8)
            audio = self._audio_vae_decode(sampled_latents[i], audio_vae)
            all_frames.append(frames)
            all_audio.append(audio)
            del vid_final, aud_final

        _cuda_sync()
        log.info("[Decode] done in %.1fs", time.time() - phase_start)

        # =============================================================
        # MUX — frames + audio → VIDEO objects
        # =============================================================
        log.info("[Mux] %d videos...", n)
        phase_start = time.time()

        videos = []
        for i in range(n):
            video = self._create_video(all_frames[i], all_audio[i], fps)
            videos.append(video)

        del all_frames, all_audio
        _cuda_sync()
        log.info("[Mux] done in %.1fs", time.time() - phase_start)

        del sampled_latents
        gc.collect()

        total = time.time() - total_start
        log.info("All %d clips done in %.1fs", n, total)

        # Pad to 5 outputs if batch_size < 5
        while len(videos) < 5:
            videos.append(videos[-1])

        return tuple(videos)
