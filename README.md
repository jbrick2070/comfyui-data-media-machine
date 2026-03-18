# ComfyUI Data-Driven Media Machine v3.6
## Community Edition Guide

**Real-time LA data (weather, air quality, earthquakes, transit, energy grid) → AI-generated video, narration, procedural visualization, audio.**

All data feeds are **LIVE, FREE, and require ZERO API keys**. Drop this into `custom_nodes/` and queue.

---

## Download

[![Download DMM v3.6](https://img.shields.io/badge/Download-DMM_v3.6_Full_Package-blue?style=for-the-badge)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.6/DMM_v3.6_full_package.zip)

**[Click here to download the full package (v3.6)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.6/DMM_v3.6_full_package.zip)** — includes both workflow files + community guide.

---

## What's New in v3.6

### RTX Video Super Resolution
The upscaler in VideoConcat has been replaced with NVIDIA RTX Video Super Resolution (`nvvfx`). This uses hardware-accelerated Tensor Core upscaling instead of the previous SeedVR2 diffusion upscaler — significantly faster, uses far less VRAM, and requires no large model downloads. Quality levels: LOW, MEDIUM, HIGH, ULTRA.

**New dependency:** Requires [Nvidia_RTX_Nodes_ComfyUI](https://github.com/NVIDIAGameWorks/Nvidia_RTX_Nodes_ComfyUI) and its `nvvfx` package.

### AI Narration Refiner (Phi-3-mini)
New `DMM_NarrationRefiner` node sits between the NarrationDistiller and Kokoro TTS. It uses Microsoft Phi-3-mini-4k-instruct to rewrite template narration into natural broadcast-quality prose. All data facts are preserved exactly — only phrasing and cadence are improved. VRAM is automatically freed after each refinement. Five broadcast styles: late night radio, morning news, calm documentary, weather channel, noir dispatch.

**First run downloads ~4 GB** (the Phi-3-mini model). Subsequent runs load from cache.

### Breaking Changes from v3.5
- `DMM_VideoConcat` removed `upscale_precision` (fp8/fp16) and `upscale_batch_size` parameters
- `DMM_VideoConcat` added `upscale_quality` (LOW/MEDIUM/HIGH/ULTRA)
- Saved v3.4/v3.5 workflows will show red widgets on VideoConcat — use the included v3.6 workflows

---

## Node Overview (29 nodes)

### Data Collection (Live, No API Keys)
| Node | Source | Data |
|------|--------|------|
| DMM_WeatherFetch | Open-Meteo / NWS | Temperature, humidity, wind, conditions |
| DMM_AirQualityFetch | Open-Meteo | US AQI, PM2.5, UV index |
| DMM_EarthquakeFetch | USGS | Recent quakes within radius, magnitude |
| DMM_MetroFetch | LA Metro API | Real-time bus positions, speed, congestion |
| DMM_AlertsFetch | NWS + CAL FIRE | Active weather and fire alerts |
| DMM_EnergyGrid | CAISO | California grid demand and renewables |

### Webcam Pipeline
| Node | Purpose |
|------|---------|
| DMM_CameraRegistry | Loads camera database (Caltrans, YouTube, custom) |
| DMM_CameraRouter | Selects cameras per data lane (random/round-robin/fixed) with scenic swap |
| DMM_WebcamFetch | Fetches live JPEG frames with fallback chain, placeholder detection, dark-camera caching |
| DMM_FramePrep | Crops, resizes, and normalizes frames for video model input |

### Narration & Audio
| Node | Purpose |
|------|---------|
| DMM_NarrationDistiller | Compresses 5 data lanes into ~60-word broadcast narration |
| DMM_NarrationRefiner | **NEW v3.6** — Phi-3-mini rewrites narration for broadcast quality |
| DMM_AudioEnhance | Spatializes TTS audio to 48kHz stereo |
| DMM_BackgroundMusic | MIDI synth background score |
| DMM_AudioMux | Mixes narration + music with ducking |

### Video Generation & Output
| Node | Purpose |
|------|---------|
| DMM_DataToPrompt | Converts live data to text-to-video prompts |
| DMM_CinematicVideoPrompt / V2 | Camera-aware cinematic prompt generation |
| DMM_BatchVideoGenerator | Multi-clip LTX-Video generation pipeline |
| DMM_ProceduralClip | Procedural motion graphics (data overlays) |
| DMM_VideoConcat | Stitches clips + crossfade + **RTX VSR upscaling** |

---

## New to ComfyUI? Start Here

ComfyUI is a free, node-based interface for running AI image and video models locally on your GPU.

> **Already have ComfyUI installed?** Skip to Step 2 below.

### Step 1 — Install ComfyUI

Use the official desktop installer — it handles Python, Git, dependencies, and the interface automatically:

**https://www.comfy.org/download**

Advanced users can also install manually from https://github.com/comfyanonymous/ComfyUI

> The installer will prompt you to install Git if you don't have it yet — just follow the prompts.

### Step 2 — Install Required Models

> **Grab a coffee — these are big downloads.** The main model is ~9.5 GB. Plan for 15–20 min on fast internet, or 1–2 hours on slower connections.

| Model | Download | Size | Est. Time (100Mbps) | Save To |
|-------|----------|------|---------------------|---------|
| **LTX-Video v0.9.5** (required) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors) | ~9.5 GB | ~15-20 min | `ComfyUI/models/checkpoints/` |
| **LTX-Video 13B** (optional, higher quality) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled.safetensors) | ~26 GB | ~45-60 min | `ComfyUI/models/checkpoints/` |
| **Kokoro TTS voice model** | Auto via ComfyUI Manager | ~500 MB | 2-5 min | auto |
| **Phi-3-mini** (for NarrationRefiner) | Auto on first run | ~4 GB | 5-10 min | HuggingFace cache |

> Slower connections (25 Mbps or under)? Expect 1–3 hours for the main model. Start the download before bed — it'll be ready in the morning.

### Step 3 — Install Required Custom Nodes

| Custom Node | Purpose | Install |
|-------------|---------|---------|
| **ComfyUI-LTXVideo** | LTX-Video model support | ComfyUI Manager or `git clone` |
| **comfyui-kokorotts** | Kokoro TTS voice synthesis | ComfyUI Manager or `git clone` |
| **Nvidia_RTX_Nodes_ComfyUI** | RTX Video Super Resolution upscaler | `git clone https://github.com/NVIDIAGameWorks/Nvidia_RTX_Nodes_ComfyUI` into `custom_nodes/` |
| **ComfyUI-Manager** | One-click node installer | [Instructions](https://github.com/ltdrdata/ComfyUI-Manager) |

### Step 4 — Load the DMM Workflow

1. **[Download the full package](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.6/DMM_v3.6_full_package.zip)** and unzip it
2. Copy the `media_machine` folder into `ComfyUI/custom_nodes/`
3. Open ComfyUI at `http://127.0.0.1:8000`
4. Click **Load** and select `LA_DATA_REPORT_v3.6.json`
   - Use `LA_DATA_REPORT_v3.6_LITE.json` if you have limited VRAM
5. If any nodes appear red, open **Manager → Install Missing Custom Nodes → Restart**
6. Hit **Queue** — the workflow fetches live LA data and begins generating

**Which version?**

| Version | GPU | VRAM | Upscale | Sigma Steps |
|---------|-----|------|---------|-------------|
| `LA_DATA_REPORT_v3.6.json` | RTX 5080 / 4090 | 16GB+ | RTX VSR ULTRA | 9 (full) |
| `LA_DATA_REPORT_v3.6_LITE.json` | RTX 4070 / 3060 | 8-16GB | Disabled | 5 (fast) |

> **How do I check my VRAM?** On Windows: **Task Manager** (Ctrl+Shift+Esc) → **Performance** tab → **GPU** → look for "Dedicated GPU memory." On Linux: run `nvidia-smi` in a terminal.

### Troubleshooting

<details>
<summary><strong>"NotImplementedError: Got 4D input, but linear mode needs 3D"</strong></summary>

**Cause**: Audio tensor shape mismatch in crossfade resampling.
**Fix**: Ensure `dmm_video_concat.py` has ndim checking in `_crossfade_audio`.
```python
if wf.ndim == 2:
    wf_3d = wf.unsqueeze(0)
elif wf.ndim == 3:
    wf_3d = wf
resampled = F.interpolate(wf_3d.float(), size=new_len, mode="linear", align_corners=False)
```
</details>

<details>
<summary><strong>NarrationRefiner shows "UNKNOWN" in workflow</strong></summary>

**Cause**: The `media_machine` custom node folder hasn't been updated to v3.6.
**Fix**: Replace your `custom_nodes/media_machine/` folder with the v3.6 version and restart ComfyUI. You should see `[DMM] Data Media Machine v3.6: loaded 29/29 nodes` in the console.
</details>

<details>
<summary><strong>RTX upscaler fails with "nvvfx not installed"</strong></summary>

**Cause**: Missing NVIDIA Video Effects SDK dependency.
**Fix**: Install [Nvidia_RTX_Nodes_ComfyUI](https://github.com/NVIDIAGameWorks/Nvidia_RTX_Nodes_ComfyUI) into `custom_nodes/` and install its requirements. RTX upscaling requires an NVIDIA RTX GPU (20-series or newer).
</details>

---

## Setting Up Your Own Machine

If you want to set up your own machine to run this, you will need to configure OBS Studio to handle the outputs dynamically.

### Prerequisites

- [OBS Studio](https://obsproject.com/download)
- [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) (3.11.x confirmed to work)
- [Media Playlist Source Plugin](https://obsproject.com/forum/resources/media-playlist-source.1524/)
- [Directory Sorter for OBS Script](https://obsproject.com/forum/resources/directory-watch-media-sorter.1767/)

### Installation & Setup

1. **Install OBS and Python**: Ensure both are installed on your system.
2. **Install the Plugin**: Install the `media-playlist-source` plugin into your OBS directory.
3. **Load the Script**: In OBS, go to **Tools > Scripts**, select the Python settings tab, and point it to your Python 3.11 install path. Then, load the `directory_sorter_for_obs` Python script.
4. **Configure Directories**: Point the OBS directory sorter script to your designated output folder.
5. **Add the Media Source**: Add the Media Playlist Source to your OBS scene. Point it to your output folder, and you will see it automatically pulsing and updating your output as new media is generated!

### Pro-Tip for Hardware Acceleration

If you are on a system where your main GPU is busy doing heavy batch rendering for updates, set OBS to use your **Integrated GPU (iGPU)** for encoding. Select **QSV AV1** (or another integrated encoder). This ensures your OBS stream won't lag while your main GPU is maxed out!

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v3.6** | RTX VSR upscaler, Phi-3 NarrationRefiner, energy grid, removed legacy workflows |
| v3.5 | SeedVR2 upscaler, CAISO energy grid, domain throttling, dark-camera cache |
| v3.4 | Crossfade transitions, procedural clips, MIDI background music, batch auto prompts |
| v3.3 | Audio enhance (spatial 48kHz), improved audio muxing |
| v3.2 | Narration distiller, Kokoro TTS integration, audio mux |
| v3.0 | Webcam pipeline (camera registry, router, frame prep), cinematic prompt v2 |

---

*Built by Jeffrey A. Brick — Los Angeles, 2025–2026*
