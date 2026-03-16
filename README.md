# ComfyUI Data-Driven Media Machine v3.4
## Community Edition Guide

**Real-time LA data (weather, air quality, earthquakes, transit) â†’ AI-generated video, narration, procedural visualization, audio.**

All data feeds are **LIVE, FREE, and require ZERO API keys**. Drop this into `custom_nodes/` and queue.

---

## ðŸ“¦ Download

[![Download DMM v3.4](https://img.shields.io/badge/Download-DMM_v3.4_Full_Package-blue?style=for-the-badge)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.4/DMM_v3.4_full_package.zip)

**[Click here to download the full package (v3.4)](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.4/DMM_v3.4_full_package.zip)** â€” includes both workflow files + community guide.

---

## ðŸš€ New to ComfyUI? Start Here

ComfyUI is a free, node-based interface for running AI image and video models locally on your GPU.

> **Already have ComfyUI installed?** Skip to Step 2 below.

### Step 1 - Install ComfyUI

Use the official desktop installer - it handles Python, Git, dependencies, and the interface automatically:

**https://www.comfy.org/download**

Advanced users can also install manually from https://github.com/comfyanonymous/ComfyUI

> The installer will prompt you to install Git if you don't have it yet â€” just follow the prompts.

### Step 2 - Install Required Models

> **â˜• Grab a coffee â€” these are big downloads.** The main model is ~9.5 GB. Plan for 15â€“20 min on fast internet, or 1â€“2 hours on slower connections.

| Model | Download | Size | Est. Time (100Mbps) | Save To |
|-------|----------|------|---------------------|---------|
| **LTX-Video v0.9.5** (required) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors) | ~9.5 GB | ~15-20 min | `ComfyUI/models/checkpoints/` |
| **LTX-Video 13B** (optional, higher quality) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-distilled.safetensors) | ~26 GB | ~45-60 min | `ComfyUI/models/checkpoints/` |
| **Kokoro TTS voice model** | Auto via ComfyUI Manager | ~500 MB | 2-5 min | auto |

> ðŸ˜´ **Slower connections (25 Mbps or under)?** Expect 1â€“3 hours for the main model. Start the download before bed â€” it'll be ready in the morning.

### Step 3 - Install ComfyUI Manager

ComfyUI Manager lets you install missing custom nodes with one click - essential for getting DMM running.

1. Open a terminal in your ComfyUI folder
2. Run: `git clone https://github.com/ltdrdata/ComfyUI-Manager ComfyUI/custom_nodes/ComfyUI-Manager`
3. Restart ComfyUI

Or use the built-in extension manager in the ComfyUI desktop app.

### Step 4 - Load the DMM Workflow

1. **[Download the full package](https://github.com/jbrick2070/comfyui-data-media-machine/releases/download/v3.4/DMM_v3.4_full_package.zip)** and unzip it
2. Open ComfyUI at `http://127.0.0.1:8188`
3. Click **Load** and select `LA_DATA_REPORT_v3.4.json`
   - Use `LA_DATA_REPORT_v3.4_LITE.json` if you have less than 24GB VRAM
4. If any nodes appear red, open **Manager -> Install Missing Custom Nodes -> Restart**
5. Hit **Queue** - the workflow fetches live LA data and begins generating

**Which version?**

| Version | GPU | VRAM |
|---------|-----|------|
| `LA_DATA_REPORT_v3.4.json` | RTX 5080 / 4090 | 24GB |
| `LA_DATA_REPORT_v3.4_LITE.json` | RTX 4070 / 3060 | 8-16GB |

> **How do I check my VRAM?** On Windows: **Task Manager** (Ctrl+Shift+Esc) â†’ **Performance** tab â†’ **GPU** â†’ look for "Dedicated GPU memory." On Linux: run `nvidia-smi` in a terminal.

### ðŸ”§ Troubleshooting

<details>
<summary><strong>"NotImplementedError: Got 4D input, but linear mode needs 3D"</strong></summary>

**Cause**: Audio tensor shape mismatch in crossfade resampling.
**Fix**: Ensure `dmm_video_concat.py` has ndim checking (line 437-449).
```python
if wf.ndim == 2:
    wf_3d = wf.unsqueeze(0)
elif wf.ndim == 3:
    wf_3d = wf
resampled = F.interpolate(wf_3d.float(), size=new_len, mode="linear", align_corners=False)
```
</details>

---


## 🚀 Setting Up Your Own Machine

If you want to set up your own machine to run this, you will need to configure OBS Studio to handle the outputs dynamically.

### Prerequisites

- OBS Studio
- Python 3.11 (Version 3.11.9 is confirmed to work)
- Media Playlist Source Plugin
- Directory Sorter for OBS Script

### Installation & Setup

1. **Install OBS and Python**: Ensure both are installed on your system.
2. **Install the Plugin**: Install the `media-playlist-source` plugin into your OBS directory.
3. **Load the Script**: In OBS, go to **Tools > Scripts**, select the Python settings tab, and point it to your Python 3.11 install path. Then, load the `directory_sorter_for_obs` Python script.
4. **Configure Directories**: Point the OBS directory sorter script to your designated output folder.
5. **Add the Media Source**: Add the Media Playlist Source to your OBS scene. Point it to your output folder, and you will see it automatically pulsing and updating your output as new media is generated!

### 💡 Pro-Tip for Hardware Acceleration

If you are on a system where your main GPU is busy doing heavy batch rendering for updates, set OBS to use your **Integrated GPU (iGPU)** for encoding. Select **QSV AV1** (or another integrated encoder). This ensures your OBS stream won't lag while your main GPU is maxed out!

---

*Built by Jeffrey A. Brick — Los Angeles, 2025-2026*