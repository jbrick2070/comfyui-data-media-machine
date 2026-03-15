# DMM v3.5-beta Technical Briefing

**Date:** March 15, 2026
**Author:** Jeffrey A. Brick
**Branch:** `v3.5-beta`
**Status:** Regression tests passing (9/9)

---

## Summary

v3.5-beta adds a single new node — **DMM Music Enhancer (Stable Audio)** — to the DMM pipeline. It sits between the Background Music generator and AudioMux, performing music-to-music style transfer using Stable Audio Open 1.0 loaded natively via `diffusers.StableAudioPipeline`. Each run randomly selects an LA-themed style prompt from a curated pool of 12 genres (deep house, noir jazz, gospel choir, G-funk, synthwave, etc.).

No external API. No separate process. Model loads directly into VRAM as a standard PyTorch pipeline, cached after first load.

v3.4 stable on `main` is not modified. The v3.4 workflow JSON is verified unchanged by regression tests.

---

## What Changed (vs. v3.4)

### New Files (3)

| File | Purpose | Size |
|------|---------|------|
| `nodes/dmm_music_enhancer.py` | ComfyUI custom node + Stable Audio Open integration | 24,248 bytes |
| `nodes/__init__.py` | Package init, re-exports NODE_CLASS_MAPPINGS | ~130 bytes |
| `LA_DATA_REPORT_v3.5_BETA.json` | v3.5 workflow (copy of v3.4 + MusicEnhancer wired in) | 46 nodes, 80 links |

### Modified Files

None. v3.4 workflow and all existing nodes are untouched.

---

## Architecture

### Audio Signal Chain (v3.4 vs v3.5)

**v3.4:**
```
BackgroundMusic (45) --link79--> AudioMux (42) --> SaveVideo (34)
```

**v3.5-beta:**
```
BackgroundMusic (45) --link80--> MusicEnhancer (46) --link81--> AudioMux (42) --> SaveVideo (34)
```

The MusicEnhancer intercepts only the `background_music` input to AudioMux. The narration audio path (`AudioEnhance (43) --> AudioMux (42)`) is completely unchanged.

### Stable Audio Open Integration

The node uses `diffusers.StableAudioPipeline` for native PyTorch audio-to-audio inference. The pipeline encodes input audio to latent space, adds noise at the specified strength, denoises with the text prompt conditioning, and decodes back to audio — all within a single `pipe()` call.

**Model loading:** Singleton cache via `_get_or_load_model()`. Loads once on first execution, stays in VRAM for subsequent runs. No checkpoint reload overhead per frame.

**Graceful degradation:** If `diffusers` is not installed or the model fails to load, the node logs a warning and passes audio through unchanged. The pipeline never crashes.

---

## Node: DMM_MusicEnhancer

### Inputs

| Parameter | Type | Default | Range | Notes |
|-----------|------|---------|-------|-------|
| `audio` | AUDIO | (required) | — | From BackgroundMusic node |
| `strength` | FLOAT | 0.20 | 0.05–0.95 | How much Stable Audio transforms the audio |
| `mix` | FLOAT | 0.65 | 0.0–1.0 | Dry/wet blend (0=original, 1=enhanced) |
| `prompt_override` | STRING | "" | — | Leave empty for random LA style; set to override |
| `seed` | INT | -1 | -1 to 2^32-1 | Random seed (-1 = random) |
| `num_inference_steps` | INT | 20 | 5–100 | Denoising steps (20 = quality/speed balance) |

### Strength Guide

| Range | Effect |
|-------|--------|
| 0.10–0.20 | Subtle texture (ambient shimmer, warmth) |
| 0.20–0.35 | Moderate enhancement (new instrumentation blended in) |
| 0.35–0.50 | Strong transformation (significant new character) |
| 0.50+ | Heavy regeneration (original becomes loose reference) |

**Recommended:** 0.20 for production, 0.15 for narration-heavy content.

### LA Style Prompt Pool (12 genres)

When `prompt_override` is empty, one of these is randomly selected per run:

1. LA deep house (analog bassline, sunset strip, synth pads)
2. LA noir cinematic (strings, ambient pads, detective drama)
3. LA gospel choir (soulful harmonies, organ chords)
4. West coast G-funk (rolling bassline, talk box, Moog)
5. LA lowrider oldies soul (doo-wop, vinyl crackle, Whittier Blvd)
6. Laurel Canyon folk rock (fingerpicking, canyon echo, Topanga)
7. LA ambient electronic (granular textures, Blade Runner atmosphere)
8. LA jazz club (upright bass, brushed snare, Central Avenue)
9. East LA cumbia fusion (accordion, congas, Boyle Heights)
10. LA synthwave (arpeggios, drum machine, PCH midnight)
11. Venice Beach drum circle (djembe, bongos, ocean waves)
12. LA philharmonic cinematic (orchestral strings, Hollywood grandeur)

### Output

Returns `AUDIO` dict (`{"waveform": Tensor, "sample_rate": int}`) — the dry/wet mix of original and enhanced audio, soft-clipped via `tanh()` to prevent overs.

---

## Crossfade Loop (47s Ceiling)

Stable Audio Open generates up to 47 seconds per call. If the input audio exceeds this, the node generates a 47s enhanced clip and seamlessly loops it to match the original duration using raised-cosine crossfading at the join points.

**Algorithm:**
- First pass copies the full 47s clip
- Each subsequent pass overlaps by a 2-second (or 25% of clip length, whichever is smaller) crossfade region
- Fade curves use `0.5 * (1 - cos(t * pi))` — energy-preserving raised cosine
- The previous tail fades out while the new head fades in at the overlap
- Safety valve prevents infinite loops on degenerate input

For clips under 47s, the crossfade loop is never called — the output is simply trimmed to match input length.

---

## Dependencies

### Required (already in DMM)

- `torch`, `numpy` — tensor operations and audio conversion
- `wave`, `io` — WAV encoding/decoding
- `gc` — VRAM management

### Required (pip install)

- `diffusers` — Stable Audio Open pipeline
- `transformers` — tokenizer and text encoder
- `accelerate` — model loading optimization

### Model Download (automatic on first run)

- `stabilityai/stable-audio-open-1.0` — downloads from HuggingFace Hub
- ~2-3 GB disk space, ~2-3 GB VRAM when loaded

---

## VRAM Budget

| Component | VRAM | Notes |
|-----------|------|-------|
| Stable Audio Open | ~2-3 GB | Cached singleton, loaded once |
| LTX-Video | ~8-12 GB | Unloaded by the time MusicEnhancer runs |
| Total peak | ~10-15 GB | Safe on RTX 5080 (24 GB) and RTX 4070+ (12 GB+) |

VRAM is flushed with `gc.collect()` + `torch.cuda.empty_cache()` before model load to clear any lazily cached LTX-Video tensors.

---

## Regression Test Results

```
============================================================
DMM v3.5-beta REGRESSION TEST
============================================================

[TEST 1] v3.5 workflow JSON validity .............. PASS
[TEST 2] v3.4 workflow unchanged .................. PASS
[TEST 3] MusicEnhancer node present in v3.5 ....... PASS
[TEST 4] MusicEnhancer wiring (input/output) ...... PASS
[TEST 5] Link integrity (no orphans) .............. PASS
[TEST 6] Node count delta (+1) .................... PASS
[TEST 7] Widget defaults in range ................. PASS
[TEST 8] Python syntax check ...................... PASS
[TEST 9] Node registration exports ................ PASS

Errors: 0 | Warnings: 0 | ALL TESTS PASSED
```

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| `diffusers` not installed | Graceful passthrough — node logs error, pipeline continues |
| Model download fails (no internet) | Passthrough — same behavior as missing dependency |
| Audio > 47s (Stable Audio limit) | Raised-cosine crossfade loop extends to any length |
| ComfyUI AUDIO type mismatch | Dict unpacking with raw-tensor fallback for backward compat |
| Batch track drop | Per-item batch loop; failed items fall back to original track |
| Stereo/mono channel mismatch | `mix_audio()` auto-converts wet signal to match dry channel count |
| VRAM overflow on handoff | `gc.collect()` + `torch.cuda.empty_cache()` before model load |
| Orchestral prompt quality | Stable Audio trained on copyright-cleared data; electronic genres are strongest |
| v3.4 regression | Verified unchanged by automated test |

---

## Bypass Node

`DMM_MusicEnhancerBypass` is included for A/B testing. Wire it in place of DMM_MusicEnhancer to hear the original mix without enhancement.

---

## Architecture Pivot Log

v3.5-beta originally used ACE-Step 1.5 via an external REST API (`localhost:8001`). This was scrapped in favor of native Stable Audio Open for three reasons:

1. **Portability** — No separate process to install, configure, or keep running
2. **Stability** — No HTTP timeouts, connection errors, or API version mismatches
3. **VRAM control** — PyTorch manages the model lifecycle natively; ComfyUI can flush it properly

The ACE-Step client code (~3,000 bytes of HTTP/multipart logic) was replaced with a single `StableAudioPipeline.from_pretrained()` call.

---

## Resolved Design Decisions

1. **Default strength: 0.20** — Lower values prevent hallucinated lead instruments from competing with narration frequencies
2. **No bundled install script** — Users manage their own Python environments; README provides step-by-step instructions
3. **`tanh()` soft clip retained** — Adequate safety net; proper peak limiter deferred to Tier 2
4. **Random prompt pool over fixed prompt** — Each run gets a different LA-flavored style, keeping the audio fresh across repeated generations
5. **Crossfade loop over silence padding** — Raised-cosine crossfade at 47s boundaries prevents audible cuts on long clips
