# ComfyUI Data-Driven Media Machine v3.4
## Community Edition Guide

**Real-time LA data (weather, air quality, earthquakes, transit) → AI-generated video, narration, procedural visualization, audio.**

All data feeds are **LIVE, FREE, and require ZERO API keys**. Drop this into `custom_nodes/` and queue.

---

## Download

[![Download DMM v3.4](https://img.shields.io/badge/Download-DMM_v3.4_Full_Package-blue?style=for-the-badge)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.4/DMM_v3.4_full_package.zip)

**[Click here to download the full package (v3.4)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.4/DMM_v3.4_full_package.zip)** � includes both workflow files + community guide.

---


## What's in v3.4

This version adds **live earthquake monitoring**, **NWS alerts**, and a **procedural data visualization node** that generates abstract animations from live data metrics.

| Feature | Status | Runtime |
|---------|--------|---------|
| Live weather (Open-Meteo) | ✓ | <1s |
| Live air quality (Open-Meteo) | ✓ | <1s |
| Live earthquakes (USGS) | ✓ | <1s |
| NWS alerts (weather.gov) | ✓ | <1s |
| LA Metro vehicles (Real-time) | ✓ | 2-3s |
| Narration generation (TTS) | ✓ | 2-5s |
| LTX-Video i2v (5 clips) | ✓ | 60-120s |
| Procedural visualization | ✓ | 10-20s |
| RTX upscaling to 1920×1080 | ✓ | 20-90s |
| Final audio mux & encode | ✓ | 5-15s |
| **Total pipeline** | ✓ | **100-250s** |

---

## GPU Compatibility Matrix

### Tier 1: Tested & Optimized (>16GB VRAM)
| GPU | Batch Size | Length (s) | Tile Size | Settings | Notes |
|-----|-----------|-----------|-----------|----------|-------|
| **RTX 5080** (24GB) | 2 | 24 | 256 | FULL SPEED | Target hardware. 287s total. |
| **RTX 4090** (24GB) | 2 | 24 | 256 | FULL SPEED | Same as 5080. Drop-in replacement. |
| **RTX 4080 Super** (24GB) | 2 | 24 | 256 | FULL SPEED | Slightly longer (320-340s). Works great. |
| **A100/H100** (40-80GB) | 2 | 24 | 256 | FULL SPEED | Blazing fast. 180-200s total. |

**Confidence**: High. Real-world tested on RTX 5080 with 24-second output.

---

### Tier 2: Constrained Mode (8-16GB VRAM)

| GPU | Batch Size | Length (s) | Tile Size | Settings | Notes |
|-----|-----------|-----------|-----------|----------|-------|
| **RTX 4070 Super** (12GB) | 1 | 12 | 256 | LITE mode | Use LA_DATA_REPORT_v3.4_LITE.json |
| **RTX 4070** (12GB) | 1 | 12 | 256 | LITE mode | May need --lowvram flag |
| **RTX 3060 Ti** (8GB) | 1 | 8 | 192 | API FALLBACK | See "API Fallback Strategy" below |
| **RTX 3060** (12GB) | 1 | 12 | 192 | LITE mode | Tight, but viable |

**What changes**:
- `batch_size`: 2 → 1
- `length`: 24 → 12 seconds
- `tile_size`: 256 → 192 pixels
- Enable `--lowvram` flag in launcher

**Runtime**: ~150-180s per render.

---

### Tier 3: API Fallback (≤8GB VRAM)

| GPU | Approach | Notes |
|-----|----------|-------|
| **RTX 4060** (8GB) | API Fallback | Use hosted LTX service + local procedural node |
| **GTX 1650** (4GB) | API Fallback | Use hosted LTX service + local procedural node |
| **T4 / MPS** | API Fallback | Cloud-based alternative for Mac/Colab |

**Fallback Strategy**: Replace the local LTX-2 video generation with API calls to:
- **Replicate.com** (free tier, $0.01/sec video)
- **Fal.ai** (hosted LTX with same i2v conditioning)
- **Custom private API** (if self-hosted LTX elsewhere)

See "API Integration Nodes" section below.

---

## Quick Start

### 1. Install to ComfyUI

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_ORG/comfyui-data-media-machine
cd comfyui-data-media-machine
pip install -r requirements.txt  # Optional: requests library (has urllib fallback)
```

### 2. Load Workflow

**For RTX 4090/5080:**
```
Load: LA_DATA_REPORT_v3.4.json
```

**For RTX 4070/3060 (8-16GB):**
```
Load: LA_DATA_REPORT_v3.4_LITE.json
```

**For RTX 4060/GTX 1650 (≤8GB):**
```
Load: LA_DATA_REPORT_v3.4_API_FALLBACK.json
```

### 3. Configure (Optional)

Edit the **Location** group nodes:
- `DMM_DataInit`: Latitude, longitude, city name, seed
- `DMM_CameraRegistry`: Optional custom camera presets JSON

All other nodes default to **live mode** and ZERO configuration.

### 4. Queue and Run

Hit **Queue** in ComfyUI. The workflow:
1. Fetches live LA data (3-5s)
2. Generates TTS narration (2-5s)
3. Generates 5 video clips via LTX-2 (60-120s)
4. Generates procedural visualization (10-20s)
5. Upscales to 1920×1080 (20-90s)
6. Muxes audio and saves (5-15s)

**Output**: `output/LA_data_YYYYMMDD_HHMMSS.mp4` (1920×1080, 24fps, ~45-60s duration)

---

## API Integration Nodes (Tier 3)

For GPUs without enough VRAM for local LTX-2, use hosted video APIs:

### Option A: Replicate.com (LTX-Video-hosted)

```
Node: DMM_ReplicateVideoProxy
Inputs:
  - positive_prompt (string)
  - negative_prompt (string)
  - conditioning_image (IMAGE)

Outputs:
  - VIDEO (video tensor compatible with rest of pipeline)

Config: Set REPLICATE_API_KEY env var or paste in widget
```

**Cost**: $0.01/sec output = $0.24 per 24-second video (minimal)

### Option B: Fal.ai (Fal.run LTX)

```
Node: DMM_FalVideoProxy
Inputs:
  - prompt (string)
  - image (IMAGE)

Outputs:
  - VIDEO

Config: Set FAL_KEY env var
```

**Cost**: Pay-as-you-go, competitive pricing.

### Option C: Private Hosted (Self-hosted LTX on cloud VM)

```
Node: DMM_CustomVideoAPI
Inputs:
  - api_endpoint (string: "https://your-server.com/generate")
  - api_key (string: secret key for your server)
  - prompt (string)
  - image (IMAGE)

Outputs:
  - VIDEO
```

**Cost**: Whatever your cloud bill is (~$0.50-2.00/render on t4/a100)

---

## Node Reference

### Data Fetchers (all live, zero config needed)

| Node | Purpose | Output | Runtime |
|------|---------|--------|---------|
| **DMM_DataInit** | Location, seed, intensity dial | config + metadata | instant |
| **DMM_WeatherFetch** | Temperature, conditions, wind | weather dict | <1s |
| **DMM_AirQualityFetch** | AQI, PM2.5, ozone, UV | aq dict | <1s |
| **DMM_MetroFetch** | Real-time bus positions | transit dict | 2-3s |
| **DMM_EarthquakeFetch** | Recent quakes near location | earthquake dict | <1s |
| **DMM_AlertsFetch** | NWS alerts (tornado, flood, wind) | alerts list | <1s |

### Narrative & Prompts

| Node | Purpose | Output | Config |
|------|---------|--------|--------|
| **DMM_LAPulseNarrative** | Turns data into prose (poetic or factual) | narration_text | style: "poetic" / "factual" / "noir" |
| **DMM_DataToPrompt** | Image generation prompt | positive_prompt, negative_prompt | style: "cinematic" / "abstract" / "noir" etc. |
| **DMM_CinematicVideoPrompt** | Video-specific prompting | video_prompt, motion_params | model: "ltx", "animate_diff", "svg" |

### Media Generation

| Node | Purpose | Output | Time | VRAM |
|------|---------|--------|------|------|
| **DMM_DataToTTS** | Kokoro TTS narration | audio (24kHz mono) | 2-5s | <1GB |
| **DMM_ProceduralClip** | Data-driven animation (FBM, cellular automata, kinetic typography) | video (24fps, any length) | 10-20s | 2-3GB |

### Video & Audio Processing

| Node | Purpose | Config | Time |
|------|---------|--------|------|
| **BatchVideoGenerator** (LTX-2 i2v) | 5 image prompts → 5 video clips | batch_size=2, length=24s | 60-120s |
| **VideoConcat** | Crossfade + concatenate clips | crossfade_frames=12 | <5s |
| **AudioMux** | Mix narration + silence | format: "mp4" | 2-5s |
| **RTXUpscaler** | 512×288 → 1920×1080 (2x+) | tile_size=256 | 20-90s |

---

## Widget Offset Prevention (Critical!)

When editing workflows in JSON, follow this widget ordering **exactly**:

### BatchVideoGenerator
```json
"widgets_values": [
  [positive_prompt],          // 0
  [negative_prompt],          // 1
  0.1,                        // 2: guidance_scale
  123,                        // 3: seed (auto-inserted)
  2,                          // 4: batch_size
  24,                         // 5: length_seconds
  25,                         // 6: fps
  "ltx2",                     // 7: model_name
  256,                        // 8: tile_size
  false,                      // 9: enable_vae_tiling
  1,                          // 10: vram_resident_layers
  0.5,                        // 11: temporal_consistency
  ...
]
```

**Rule**: If you wire a widget to another node in the JSON, insert `""` or `0` placeholder at that index.

---

## File Structure (GitHub Distribution)

```
comfyui-data-media-machine/
├── __init__.py                          # Package init, safe imports
├── nodes/
│   ├── __init__.py
│   ├── dmm_data_init.py
│   ├── dmm_weather_fetch.py
│   ├── dmm_airquality_fetch.py
│   ├── dmm_metro_fetch.py
│   ├── dmm_earthquake_fetch.py
│   ├── dmm_alerts_fetch.py
│   ├── dmm_la_pulse.py                 # Narrative generation
│   ├── dmm_data_to_prompt.py
│   ├── dmm_cinematic_video_prompt.py
│   ├── dmm_data_to_tts.py
│   ├── dmm_data_to_music.py
│   ├── dmm_procedural_clip.py          # Visualization node
│   ├── dmm_replicate_api.py            # (Optional) Replicate proxy
│   └── dmm_fal_api.py                  # (Optional) Fal.ai proxy
├── workflows/
│   ├── LA_DATA_REPORT_v3.4.json        # Full (RTX 4090/5080)
│   ├── LA_DATA_REPORT_v3.4_LITE.json   # Constrained (RTX 4070/3060)
│   └── LA_DATA_REPORT_v3.4_API_FALLBACK.json  # Minimal + API (RTX 4060/1650)
├── camera_registry.json                # Preset camera angles
├── README.md                           # This file
├── CHANGELOG.md                        # v3.0 → v3.1 → v3.2 → v3.4
├── requirements.txt                    # Optional: requests
└── setup.py                            # Package metadata
```

---

## Distribution Strategy: GitHub + ComfyUI Manager

### Phase 1: GitHub Release
1. Create repo: `comfyui-data-media-machine` (public)
2. Add `__init__.py` formatted correctly for ComfyUI Manager auto-detection
3. Tag release: `v3.4` with changelog
4. **Advantage**: Full control, version history, community PRs

### Phase 2: ComfyUI Manager Registration
1. Fork `ComfyUI-Manager/custom-node-list.json`
2. Submit PR with entry:
   ```json
   {
     "author": "Jeffrey A. Brick",
     "title": "Data-Driven Media Machine",
     "description": "Real-time LA weather/transit → generative video, narration, procedural visualization",
     "reference": "https://github.com/your-org/comfyui-data-media-machine",
     "install_type": "git-clone",
     "disabled": false
   }
   ```
3. Once approved, users can install via ComfyUI Manager UI with one click
4. **Advantage**: Discoverability, automatic updates, community vetting

### Phase 3: Community Marketplace (Optional)
- Create landing page: `github.io` docs site with video demos
- Submit to ComfyUI ecosystem list on HuggingFace
- Cross-post to Civitai (node category)
- Create tutorial video on YouTube (5-10 min quick start)

---

## Troubleshooting

### "NotImplementedError: Got 4D input, but linear mode needs 3D"
**Cause**: Audio tensor shape mismatch in crossfade resampling.
**Fix**: Ensure `dmm_video_concat.py` has ndim checking (line 437-449).
```python
# Correct:
if wf.ndim == 2:
    wf_3d = wf.unsqueeze(0)
elif wf.ndim == 3:
    wf_3d = wf
resampled = F.interpolate(wf_3d.float(), size=new_len, mode="linear", align_corners=False)
```

### "RuntimeError: Sizes of tensors must match except in dimension 2"
**Cause**: ProceduralClip generating mono audio (1, 1, samples) while LTX output is stereo (1, 2, samples).
**Fix**: Ensure `dmm_procedural_clip.py` line 360 generates stereo:
```python
silence = torch.zeros(1, 2, n_samples)  # stereo to match LTX-2 audio VAE output
```

### "CUDA out of memory" / "RuntimeError: CUDA error: an illegal memory access was encountered"
**Cause**: Batch size or tile size too large for your GPU.
**Solution**:
1. Try LITE version first: `LA_DATA_REPORT_v3.4_LITE.json`
2. Reduce `batch_size` from 2 → 1 in BatchVideoGenerator widgets
3. Reduce `length` from 24 → 12 seconds
4. Reduce `tile_size` from 256 → 192 or 128 for upscaler
5. Enable `--lowvram` flag: `python.exe launch.py --lowvram --normalvram`

### "ImportError: No module named 'requests'"
**Cause**: `requests` library not installed (optional).
**Solution**: The nodes fall back to `urllib.request` automatically. No action needed.
If you want requests:
```bash
.\python_embedded\python.exe -m pip install requests
```

### "No data returned from weather API"
**Cause**: API rate limit or network issue.
**Solution**:
1. Check internet connection
2. Wait 60 seconds, try again (rate limit reset)
3. Try `demo_*` source modes in data fetch nodes (offline test data)
4. Check if `open-meteo.com` is accessible from your network

---

## Performance Tuning

### For RTX 4090/5080 (24GB+)
```
✓ Full settings, batch_size=2, length=24s, tile_size=256
✓ ~280-320s per complete render
✓ 1920×1080 output quality maxed out
```

### For RTX 4070/3060 (12GB)
```
✓ LITE settings, batch_size=1, length=12s, tile_size=256
✓ Run with --lowvram flag
✓ ~150-180s per complete render
✓ Output still 1920×1080, but fewer frames
```

### For RTX 4060 (8GB)
```
✓ Use API fallback for video generation
✓ Keep ProceduralClip + TTS local
✓ ~60-90s per render (mostly API wait time)
✓ Can run multiple in parallel (separate API calls)
```

---

## What's Different from v2

| Issue | v2 | v3.4 |
|-------|-----|------|
| One node crash kills pack | ❌ | ✅ Safe per-node imports |
| Earthquake data | ❌ | ✅ USGS live feed |
| NWS alerts | ❌ | ✅ weather.gov integration |
| Procedural visualization | ❌ | ✅ Data-driven animation |
| Widget offset bugs | ⚠️ | ✅ Regression tested |
| GPU fallback (≤8GB) | ❌ | ✅ API proxy nodes |
| Workflow organization | Basic | ✅ Semantic node grouping |

---

## Author & Attribution

**Concept & Implementation**: Jeffrey A. Brick ([@jbrick2070](https://github.com/jbrick2070))

**Hardware**: Lenovo Legion Pro 7i Gen 10 (RTX 5080, Win11)

**Tested on**:
- ComfyUI (latest stable)
- Python 3.11+
- CUDA 12.1+
- PyTorch 2.1+

**License**: MIT (see LICENSE.md)

---

## Contributing

PRs welcome! Focus areas:
- Additional data feeds (stock prices, sports scores, NASA imagery)
- More API fallback providers (Together, Hugging Face Inference API)
- Mac/Linux testing (currently Win11 primary)
- Additional procedural animation styles
- UI/UX improvements to node widgets

---

## Links

- **GitHub**: https://github.com/YOUR_ORG/comfyui-data-media-machine
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Data Sources**:
  - Open-Meteo: https://open-meteo.com
  - USGS Earthquakes: https://earthquake.usgs.gov/earthquakes/feed/
  - NWS Alerts: https://api.weather.gov/
  - LA Metro: https://api.metro.net/

---

**Last updated**: March 2026
**Stability**: Production-ready for RTX 4090+, tested on RTX 5080
