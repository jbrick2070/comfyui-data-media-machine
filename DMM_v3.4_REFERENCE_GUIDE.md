# Data Media Machine (DMM) v3.4 — Complete Reference Guide

**Version:** 3.4
**Status:** Stable
**Node Count:** 27
**Framework:** ComfyUI (Custom Nodes)
**Created:** March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Complete Node Reference](#complete-node-reference)
5. [Workflow Walkthrough](#workflow-walkthrough)
6. [Audio Pipeline & Ducking](#audio-pipeline--ducking)
7. [MIDI Synthesis Engine](#midi-synthesis-engine)
8. [Data Structures](#data-structures)
9. [Troubleshooting](#troubleshooting)
10. [v3.5 Roadmap](#v35-roadmap)

---

## Overview

Data Media Machine (DMM) is a ComfyUI-based workflow for generating cinematic video narratives from structured data. It combines:

- **Data input** → structured datasets, geographic coordinates, temporal information
- **Narration synthesis** → text-to-speech with custom voice, tone, and timing
- **Procedural video generation** → LLM-guided synthetic visuals with camera movement
- **Background music synthesis** → MIDI-to-audio rendering with intelligent ducking
- **Final compositing** → multi-track audio/video mixing with dynamic volume envelopes

**Use case:** Automated documentary-style video generation for sports events, geographic data journalism, and cultural narratives.

**Example:** LA28 Olympic Anthem Project — 135 BPM LA-inspired march (trumpet + piano) layered under narration about Olympic host cities.

---

## System Architecture

### High-Level Signal Flow

```
┌─────────────────┐
│  DATA SOURCE    │ (CSV, JSON, coordinates, timeline)
└────────┬────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
┌───▼──────────┐           ┌──────────▼────┐
│ NARRATION    │           │ PROCEDURAL     │
│ ENGINE       │           │ VIDEO ENGINE   │
└───┬──────────┘           └──────────┬────┘
    │                                 │
    │ (AUDIO: TTS)                   │ (VIDEO: MP4)
    │                                 │
    └────────────┬────────────────────┘
                 │
         ┌───────▼────────┐
         │ BACKGROUND     │
         │ MUSIC SYNTH    │
         └───────┬────────┘
                 │ (AUDIO: MIDI→Tensor)
                 │
         ┌───────▼────────────────┐
         │ AUDIO MUX              │
         │ (Mix + Duck + Pad)      │
         └───────┬────────────────┘
                 │
         ┌───────▼─────────────────┐
         │ FINAL COMPOSITING       │
         │ (Video + Audio Sync)    │
         └───────┬─────────────────┘
                 │
         ┌───────▼────────────────┐
         │ OUTPUT VIDEO (MP4)     │
         └────────────────────────┘
```

### Node Categories

**Data Input Layer** (5 nodes)
- CSVReader, DataCleaner, GeocoderNode, TimelineBuilder, DataValidator

**Narration Layer** (4 nodes)
- TextProcessor, TTSVoiceSelector, NarrationDistiller, SpeechRateAdapter

**Video Layer** (7 nodes)
- ProceduralClip, CameraPath, LLMDirector, VideoCodec, VideoNormalizer, FrameBlender, TimingSync

**Audio Layer** (3 nodes)
- BackgroundMusic (new in v3.4), AudioMux, AudioNormalizer

**Compositing Layer** (2 nodes)
- VideoConcat, FinalRenderer

**Utility Nodes** (6 nodes)
- PromptBuilder, JSONBuilder, Logger, Profiler, ErrorHandler, SchemaValidator

---

## Installation & Setup

### Prerequisites

- **ComfyUI** (latest)
- **Python 3.9+** (in ComfyUI's venv)
- **CUDA 11.8+** or **AMD GPU support** (for video synthesis)

### Step 1: Install Media Machine Custom Nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/[repo]/media_machine.git
cd media_machine
pip install -r requirements.txt
```

### Step 2: Install Audio Dependencies

The `dmm_background_music.py` node requires MIDI parsing. Install both:

```bash
# Primary (recommended)
pip install pretty_midi

# Fallback (if pretty_midi unavailable)
pip install mido
```

**Windows users:** Run in Windows Command Prompt (Admin):
```cmd
C:\Users\[Username]\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install mido
```

### Step 3: Load Workflow

1. Launch ComfyUI
2. Click **Load** → select `LA_DATA_REPORT_v3.4.json`
3. Verify console output: `[DMM] Data Media Machine v3.4: loaded 27/27 nodes`

### Step 4: Verify Audio Rendering

Check ComfyUI console during first run:
```
[BackgroundMusic] mido fallback: 1524 notes rendered
[BackgroundMusic] output: 213.3s, 48000Hz, stereo, vol=30%
```

If you see `ImportError: Need either 'pretty_midi' or 'mido'`, re-run the pip install above.

---

## Complete Node Reference

### 1. DMM_CSVReader

**Purpose:** Ingest tabular data from CSV files.

**Inputs:**
- `file_path` (STRING): Path to CSV file relative to ComfyUI/input

**Outputs:**
- `data_table` (DATA): Parsed rows + headers
- `row_count` (INT): Number of rows

**Example:**
```
Input: "cities_data.csv"
Output:
  - 15 rows (Los Angeles, San Francisco, Phoenix, etc.)
  - Columns: [name, lat, lon, population, year]
```

---

### 2. DMM_DataCleaner

**Purpose:** Normalize and validate tabular data; handle missing values.

**Inputs:**
- `data_table` (DATA): Raw table
- `mode` (ENUM): ["remove_nulls", "interpolate", "forward_fill"]
- `tolerance` (FLOAT): 0.0–1.0, threshold for accepting rows

**Outputs:**
- `cleaned_data` (DATA): Deduplicated, validated rows
- `report` (STRING): Summary of changes (rows removed, interpolated, etc.)

**Example:**
```
Input: 15 rows, 2 with NULL populations
Mode: "forward_fill"
Output: 15 rows (NULL values filled with previous row value)
Report: "Cleaned 2 rows, 0 removed"
```

---

### 3. DMM_Geocoder

**Purpose:** Convert city names or coordinates to high-resolution map overlays.

**Inputs:**
- `data_table` (DATA): Rows with `lat`, `lon` columns
- `zoom_level` (INT): 8–16 (8=continent view, 16=street level)
- `style` (ENUM): ["satellite", "terrain", "vector"]

**Outputs:**
- `map_frames` (VIDEO): Sequence of map frames with location pins
- `coordinates` (DATA): Validated lat/lon pairs

---

### 4. DMM_TimelineBuilder

**Purpose:** Create temporal markers (key event timestamps) from data rows.

**Inputs:**
- `data_table` (DATA): Must include `date` or `year` column
- `duration_sec` (FLOAT): Total video length in seconds
- `snap_to_beat` (BOOL): Align events to musical beat (uses BPM)
- `bpm` (INT): Default 135 (used if snap_to_beat=True)

**Outputs:**
- `timeline_marks` (DATA): Sorted list of {timestamp_sec, event_name, row_index}
- `event_count` (INT): Number of events mapped

**Example:**
```
Input: 15 cities, video duration 41 seconds, BPM 135
Output:
  - 15 marks distributed across 41 seconds
  - Each aligned to nearest beat (135 BPM = beat every ~0.44 sec)
```

---

### 5. DMM_DataValidator

**Purpose:** Schema validation and type checking before pipeline execution.

**Inputs:**
- `data_table` (DATA): Table to validate
- `schema` (STRING): JSON schema definition
- `strict` (BOOL): If true, fail on ANY deviation; if false, warn and continue

**Outputs:**
- `valid` (BOOL): True if passes validation
- `errors` (STRING): List of schema violations (if any)

---

### 6. DMM_PromptBuilder

**Purpose:** Convert data rows into LLM prompts for narration or video direction.

**Inputs:**
- `data_table` (DATA): Rows to narrate
- `template` (STRING): Prompt template with {placeholders}
- `style` (ENUM): ["documentary", "poetic", "technical", "casual"]
- `voice_tone` (ENUM): ["neutral", "excited", "solemn", "humorous"]

**Outputs:**
- `prompts` (LIST[STRING]): One prompt per data row
- `token_count` (INT): Approximate total tokens for all prompts

**Example Template:**
```
"Tell me about {name} (population {population}). It hosted the Olympics in {year}.
 Make it {voice_tone} and {style}."
```

---

### 7. DMM_TextProcessor

**Purpose:** Clean, tokenize, and format text for TTS ingestion.

**Inputs:**
- `prompts` (LIST[STRING]): Raw prompt strings
- `max_chars_per_chunk` (INT): Split long prompts into chunks (0=no split)
- `remove_special_chars` (BOOL): Strip non-ASCII, URLs, etc.

**Outputs:**
- `processed_text` (LIST[STRING]): Cleaned prompts, ready for TTS
- `char_count` (INT): Total characters

---

### 8. DMM_TTSVoiceSelector

**Purpose:** Choose TTS engine, voice, and parameters (pitch, speed, accent).

**Inputs:**
- `engine` (ENUM): ["google", "azure", "elevenlabs", "bark"]
- `voice_id` (STRING): Voice identifier (e.g., "en-US-Neural2-C" for Google)
- `pitch` (FLOAT): -20 to +20 semitones
- `rate` (FLOAT): 0.5 (slow) to 2.0 (fast); default 1.0
- `accent` (ENUM): ["neutral", "british", "southern_us", "australian"]

**Outputs:**
- `config` (DICT): TTS configuration (serialized for later use)
- `voice_name` (STRING): Human-readable name

**Example:**
```
Input: engine=google, voice_id=en-US-Neural2-C, pitch=+3, rate=1.1
Output: config ready for NarrationDistiller
```

---

### 9. DMM_NarrationDistiller

**Purpose:** Generate TTS audio from prompts using selected voice + config.

**Inputs:**
- `prompts` (LIST[STRING]): Text to synthesize
- `tts_config` (DICT): From TTSVoiceSelector
- `output_format` (ENUM): ["mp3", "wav", "float32_tensor"]
- `sample_rate` (INT): 22050, 44100, or 48000 Hz

**Outputs:**
- `audio` (AUDIO): Stereo tensor {waveform: (1, 2, samples), sample_rate: int}
- `durations` (LIST[FLOAT]): Duration in seconds of each prompt's audio
- `transcript` (LIST[STRING]): Echo of input prompts (for logging)

**Example Output:**
```
audio.waveform: shape (1, 2, 1058400)  # 22.05 seconds at 48kHz
durations: [3.1, 2.8, 4.2, 5.6, ... ]  # 15 values
sample_rate: 48000
```

---

### 10. DMM_SpeechRateAdapter

**Purpose:** Speed up or slow down TTS audio without pitch shift.

**Inputs:**
- `audio` (AUDIO): TTS output from NarrationDistiller
- `target_duration_sec` (FLOAT): Desired total narration length
- `method` (ENUM): ["time_stretch", "resample_linear", "phase_vocoder"]

**Outputs:**
- `audio_stretched` (AUDIO): Same content, modified duration
- `stretch_ratio` (FLOAT): Factor applied (1.2 = 20% faster)

**Example:**
```
Input: 22.3 seconds of narration, target 18 seconds
Method: time_stretch
Output: 18 seconds (20% speed increase, pitch preserved)
```

---

### 11. DMM_ProceduralClip

**Purpose:** Generate synthetic video frames using LLM-guided prompts.

**Inputs:**
- `narrative` (STRING): Scene description from PromptBuilder
- `style` (ENUM): ["photorealistic", "cinematic", "animated", "documentary"]
- `duration_sec` (FLOAT): How long to render this clip
- `fps` (INT): 24 or 30
- `resolution` (ENUM): ["720p", "1080p", "2k"]

**Outputs:**
- `video` (VIDEO): Sequence of frames (compressed MP4)
- `frame_count` (INT): Number of frames generated

**Example:**
```
Input: "Los Angeles, 1932. Sweeping aerial view of downtown.
        Camera pans from Griffith Observatory to the Coliseum."
Duration: 5 seconds, 30 fps
Output: 150 frames of synthetic video
```

---

### 12. DMM_CameraPath

**Purpose:** Define smooth camera motion (pan, zoom, dolly) for procedural clips.

**Inputs:**
- `start_pos` (VEC3): (x, y, z) starting position
- `end_pos` (VEC3): (x, y, z) ending position
- `duration_sec` (FLOAT): Time to complete motion
- `easing` (ENUM): ["linear", "ease_in", "ease_out", "ease_in_out"]

**Outputs:**
- `camera_keyframes` (DATA): Frame-by-frame camera position + rotation
- `motion_vector` (VEC3): Net displacement

---

### 13. DMM_LLMDirector

**Purpose:** Generate detailed visual direction prompts for procedural video using an LLM.

**Inputs:**
- `scene_description` (STRING): High-level scene from PromptBuilder
- `style_reference` (STRING): Reference for visual tone ("BBC documentary", "Terrence Malick", etc.)
- `duration_sec` (FLOAT): Scene length (influences detail level)
- `model` (ENUM): ["gpt-4", "claude-3-opus", "gemini-pro"]

**Outputs:**
- `detailed_prompt` (STRING): Full visual specification for ProceduralClip
- `tokens_used` (INT): LLM token count

---

### 14. DMM_VideoCodec

**Purpose:** Encode procedural video frames into compressed MP4.

**Inputs:**
- `frames` (VIDEO): Raw frame sequence
- `codec` (ENUM): ["h264", "h265", "vp9"]
- `bitrate_mbps` (INT): 5–50 Mbps
- `fps` (INT): 24 or 30

**Outputs:**
- `video_mp4` (VIDEO): Compressed MP4 file
- `file_size_mb` (FLOAT): Compressed size

---

### 15. DMM_VideoNormalizer

**Purpose:** Adjust video color, brightness, contrast for consistency.

**Inputs:**
- `video` (VIDEO): Input video
- `brightness` (FLOAT): -1.0 (dark) to +1.0 (bright)
- `contrast` (FLOAT): 0.5 (low) to 2.0 (high)
- `saturation` (FLOAT): 0.0 (grayscale) to 2.0 (oversaturated)

**Outputs:**
- `normalized_video` (VIDEO): Color-corrected output

---

### 16. DMM_FrameBlender

**Purpose:** Smooth transitions between procedural video clips using cross-fade or dissolve.

**Inputs:**
- `video_a` (VIDEO): First clip
- `video_b` (VIDEO): Second clip
- `transition_frames` (INT): Duration of blend in frames (12 frames @ 30fps = 0.4 sec)
- `blend_mode` (ENUM): ["cross_fade", "dissolve", "wipe", "slide"]

**Outputs:**
- `blended_video` (VIDEO): Seamless transition between clips

---

### 17. DMM_TimingSync

**Purpose:** Align video frame timing with audio narration (frame-by-frame sync).

**Inputs:**
- `video` (VIDEO): Procedural video clip
- `audio_durations` (LIST[FLOAT]): Narration segment durations from NarrationDistiller
- `sync_mode` (ENUM): ["strict_snap", "flexible_stretch", "fit_to_audio"]

**Outputs:**
- `synced_video` (VIDEO): Video duration matched to audio
- `frames_added_or_removed` (INT): Net frame change

---

### 18. DMM_VideoConcat

**Purpose:** Concatenate multiple procedural video clips into one continuous sequence.

**Inputs:**
- `videos` (LIST[VIDEO]): List of clips to join
- `transitions` (ENUM): ["cut", "fade", "dissolve"]
- `fade_frames` (INT): If transitions="fade", frames for fade (0 = cut)

**Outputs:**
- `concatenated_video` (VIDEO): Single video file
- `total_duration_sec` (FLOAT): Total length

---

### 19. DMM_BackgroundMusic (NEW in v3.4)

**Purpose:** Synthesize MIDI file to audio tensor using additive synthesis (no external synth).

**Inputs:**
- `midi_file` (STRING): Path to .mid file (relative to ComfyUI/input or custom_nodes/media_machine/)
- `duration_sec` (FLOAT): Clip length in seconds; if 0.0, uses MIDI duration
- `volume` (FLOAT): 0.0–1.0, output amplitude (default 0.3 = 30%)
- `loop` (BOOL): If True, repeat MIDI to fill duration; if False, trim or pad with silence
- `fade_in_sec` (FLOAT): Fade-in envelope duration (default 0.5 sec)
- `fade_out_sec` (FLOAT): Fade-out envelope duration (default 2.0 sec)
- `sample_rate` (INT): Audio sample rate (default 48000 Hz)

**Outputs:**
- `audio` (AUDIO): Stereo tensor {waveform: (1, 2, samples), sample_rate: int}

**Synthesis Details:**
- Additive synthesis: each note rendered as sine wave with ADSR envelope
- GM program → instrument function mapping:
  - **0–31:** Brass (trumpet, trombone, horn, tuba) → bright sine harmonics
  - **32–47:** Piano → complex spectrum (fundamental + harmonics)
  - **48–63:** Strings → warm, slow attack sine wave
  - **64+:** Generic → sine wave, no harmonics
- Pitch-to-frequency: `f = 440 * 2^((midi_note - 69) / 12)`
- Stereo width: 0.4ms Haas delay between L and R channels
- Rendering backend: `pretty_midi` (primary) or `mido` (fallback)

**Example:**
```
Input: "seventysixcities.mid" (trumpet + piano, 213.3 sec, 135 BPM)
       volume=0.3, loop=True, duration_sec=41.0
Output: 41 seconds of looped music, 30% volume, 48kHz stereo
Console: "[BackgroundMusic] mido fallback: 1524 notes rendered"
```

**MIDI File Resolution:**
- Searches in order: `pkg_dir` → `ComfyUI/input` → `ComfyUI/custom_nodes/media_machine/` → current working directory
- Accepts: `.mid`, `.midi`
- Returns error if file not found

---

### 20. DMM_AudioMux

**Purpose:** Mix multiple audio streams (TTS narration + background music) with intelligent ducking and padding.

**Inputs:**
- `video` (VIDEO): Procedural video (used only for duration reference)
- `audio` (AUDIO): TTS narration from NarrationDistiller
- `mode` (ENUM): ["replace", "mix", "duck"] (default "replace")
- `mix_ratio` (FLOAT): Balance for "mix" mode (0.0 = 100% narration, 1.0 = 100% music)
- `pad_to_video` (BOOL): Extend audio to match video duration (default True)
- `background_music` (AUDIO, optional): Music from BackgroundMusic node
- `music_duck_db` (FLOAT): Ducking depth (default -6.0 dB)

**Outputs:**
- `audio_mixed` (AUDIO): Final composited audio
- `duration_sec` (FLOAT): Output audio length

**Ducking Envelope Logic:**
When `background_music` is provided:
1. During narration (TTS active): apply `music_duck_db` attenuation (e.g., -6dB = 50% volume)
2. After narration ends: ramp music from ducked level to full volume over next 2 seconds
3. If `pad_to_video=True`: extend audio with background music to match video duration

**Example:**
```
Input:
  - video: 41 seconds @ 30fps
  - audio (TTS): 22.3 seconds (narration)
  - background_music: looped trumpet+piano
  - music_duck_db: -6.0

Processing:
  - Mix narration + music (music ducked -6dB during narration)
  - After narration ends (22.3s), ramp music to full volume
  - Pad with music from 22.3s to 41.0s (18.7s silence gap → music)

Output: 41 seconds of mixed audio (narration + music blend)
```

**Ducking Envelope Detail:**
```
Time 0:           | TTS begins, music starts at -6dB
Time 22.3s:       | TTS ends, music begins ramp to 0dB
Time 24.3s:       | Ramp complete, music at full volume
Time 41.0s:       | Video ends
```

---

### 21. DMM_AudioNormalizer

**Purpose:** Normalize audio loudness to a target level (prevent clipping, ensure consistency).

**Inputs:**
- `audio` (AUDIO): From AudioMux
- `target_loudness_lufs` (FLOAT): Target loudness in LUFS (-23 is broadcast standard)
- `max_headroom_db` (FLOAT): Prevent peaks above this level (default -3dB)

**Outputs:**
- `normalized_audio` (AUDIO): Loudness-matched output
- `gain_applied_db` (FLOAT): Gain applied

---

### 22. DMM_FinalRenderer

**Purpose:** Composite synchronized video + audio into final MP4 output.

**Inputs:**
- `video` (VIDEO): From VideoConcat
- `audio` (AUDIO): From AudioNormalizer
- `output_format` (ENUM): ["mp4", "mov", "webm"]
- `video_codec` (ENUM): ["h264", "h265"]
- `audio_codec` (ENUM): ["aac", "flac", "opus"]

**Outputs:**
- `output_file` (STRING): Path to final video file
- `file_size_mb` (FLOAT): Output file size

---

### 23. DMM_Logger

**Purpose:** Log execution events, timing, and metrics to console and file.

**Inputs:**
- `message` (STRING): Event description
- `level` (ENUM): ["info", "warning", "error", "debug"]
- `log_file` (STRING): Output file path

**Outputs:**
- `logged` (BOOL): True if successfully logged

---

### 24. DMM_Profiler

**Purpose:** Measure execution time and memory usage of any node or subgraph.

**Inputs:**
- `node_name` (STRING): Node to profile
- `iterations` (INT): Number of runs to average

**Outputs:**
- `avg_time_sec` (FLOAT): Average execution time
- `peak_memory_mb` (FLOAT): Peak memory usage
- `report` (STRING): Human-readable summary

---

### 25. DMM_ErrorHandler

**Purpose:** Graceful error recovery and fallback logic.

**Inputs:**
- `on_error` (ENUM): ["skip_node", "use_default", "halt_pipeline"]
- `default_value` (any): Fallback value if error occurs

**Outputs:**
- `error_occurred` (BOOL): True if exception was caught
- `error_message` (STRING): Exception details

---

### 26. DMM_JSONBuilder

**Purpose:** Export workflow state and parameters as JSON for archiving or re-runs.

**Inputs:**
- `all_data_tables` (LIST[DATA]): Data at each pipeline stage
- `all_configs` (LIST[DICT]): All node configurations
- `metadata` (DICT): Custom metadata (author, date, version, etc.)

**Outputs:**
- `output_json` (STRING): Serialized workflow state
- `file_saved` (BOOL): True if written to disk

---

### 27. DMM_SchemaValidator

**Purpose:** Validate data schema at pipeline entry point before execution.

**Inputs:**
- `data_table` (DATA): Input table
- `schema` (STRING): JSON schema URL or inline definition

**Outputs:**
- `valid` (BOOL): Passes validation
- `errors` (LIST[STRING]): Schema violations (if any)

---

## Workflow Walkthrough

### LA28 Olympic Anthem — Step-by-Step Execution

**Dataset:** 15 Olympic host cities (LA, Tokyo, Paris, Beijing, etc.) with years and coordinates

**Workflow Steps:**

1. **CSVReader** → Load cities_data.csv (15 rows)

2. **DataCleaner** → Remove any null populations, forward-fill missing coords

3. **DataValidator** → Verify schema (name, lat, lon, year are required)

4. **PromptBuilder** → For each city, create narration prompt:
   - Template: `"[city_name], hosted the Olympics in [year]. Population [pop]. ..."`
   - Style: poetic
   - Tone: excited

5. **TextProcessor** → Clean prompts, tokenize

6. **TTSVoiceSelector** → Select Google Neural2-C, pitch +3 semitones, rate 1.1x (slightly fast)

7. **NarrationDistiller** → Generate TTS audio
   - Output: 22.3 seconds of narration, 48kHz stereo

8. **TimelineBuilder** → Map 15 cities across 41-second video
   - Snap to 135 BPM beats (~0.44 sec per beat)
   - Cities aligned at beats 0, 3, 6, 9, ... for visual sync

9. **LLMDirector** (for each city) → Generate visual direction:
   - City name, year, landscape, camera motion
   - Example: `"Aerial view of downtown Los Angeles, 1932. Camera pans from Griffith Observatory to the Coliseum. Sunset light. Vintage newsreel aesthetic."`

10. **ProceduralClip** (× 15) → Render each city as synthetic video
    - Style: cinematic
    - Duration: ~2.7 sec per city (41 sec / 15 cities)
    - Resolution: 1080p, 30 fps

11. **FrameBlender** (× 14) → Cross-fade between cities (0.4 sec transitions)

12. **VideoConcat** → Join 15 clips + 14 transitions → 41-second video

13. **TimingSync** → Stretch/trim procedural video to exactly match narration (22.3s) + music pad (18.7s)

14. **BackgroundMusic** → Synthesize MIDI (seventysixcities.mid)
    - Input: trumpet + piano march, 213.3 seconds, 135 BPM
    - Loop to 41 seconds
    - Volume: 30%
    - Output: 48kHz stereo

15. **AudioMux** → Composite narration + music
    - Input: 22.3s TTS + 41s looped music
    - Music ducked -6dB during narration (music audible but secondary)
    - Music ramps to full volume after narration ends
    - Pad to video duration (41s)

16. **AudioNormalizer** → Ensure -23 LUFS broadcast loudness

17. **FinalRenderer** → Sync audio + video, output MP4 (h264, aac)

**Total output:** `olympic_anthem_v3.4.mp4` (41 sec, 1080p, ~80 MB)

---

## Audio Pipeline & Ducking

### Why Ducking?

Humans can't perceive two simultaneous speech streams. When music and narration play together, one must be perceptually dominant. **Ducking** reduces the volume of secondary content (music) when primary content (speech) is present, allowing both to be heard without interference.

### Ducking Envelope (v3.4)

```
Volume
(dB)
 0  ├─────────────────────────────────────────────────
    │                                                 ╱
-6  │ NARRATION ACTIVE (Music at -6dB, ~50% vol)   ╱
    │ ╱─────────────────────────────┐              ╱
    │╱                              │             ╱
-∞  ├──────────────────────────────┼────────────╱─────
    │ (No music before narration)  │ RAMP UP   │
    │                              │ (2.0 sec) │ FULL VOL
    0s                           22.3s      24.3s      41s
    ├─────────────────────────────┤──────────┤─────────┤
    │    NARRATION DURATION        │ MUSIC-ONLY (background)
    └─────────────────────────────┴──────────┴─────────┘
```

### Parameters

- **`music_duck_db`** (default -6.0): Attenuation during narration
  - `-12.0` = 25% volume (very quiet)
  - `-6.0` = 50% volume (audible, secondary)
  - `-3.0` = 70% volume (prominent)
  - `0.0` = 100% volume (no ducking)

- **Ramp duration**: Hard-coded to 2.0 seconds (smooth transition post-narration)

### Audio Mixing Formula

At time *t*:
- If *t* < narration_end_time:
  - `output = narration + (music × 10^(duck_db / 20))`
- If narration_end_time ≤ *t* < narration_end_time + 2.0:
  - Ramp factor *r* = (*t* - narration_end_time) / 2.0
  - `output = narration × (1 - r) + music × (10^(duck_db / 20) + r × (1 - 10^(duck_db / 20)))`
- If *t* ≥ narration_end_time + 2.0:
  - `output = music`

---

## MIDI Synthesis Engine

### Overview

The **BackgroundMusic node** synthesizes MIDI to audio using **additive synthesis** (sine waves + harmonics) with no external synth engine (no FluidSynth, no soundfont). This keeps the pipeline lightweight and GPU-agnostic.

### Note Synthesis

Each note is rendered as:

```
y(t) = Σ A_h × sin(2π × f_h × t + φ) × envelope(t)
       h=1..N
```

Where:
- *f* = fundamental frequency (Hz), derived from MIDI note number
- *f_h* = harmonic frequency (*h* × *f*)
- *A_h* = harmonic amplitude (decays with harmonic number)
- *envelope(t)* = ADSR time-varying amplitude
- *N* = number of harmonics (depends on instrument)

### Instrument-Specific Synthesis

**Brass (Programs 56–62: Trumpet, Trombone, Horn, Tuba)**
```
Harmonics: 1, 2, 3, 4, 5
Amplitudes: 1.0, 0.6, 0.3, 0.15, 0.08
ADSR: A=0.05s, D=0.1s, S=0.7, R=0.15s (bright, quick attack)
Pitch bend: ±200 cents per 0.1s (for expressiveness)
```

**Piano (Program 0–7: Acoustic, Electric, Grand Piano)**
```
Harmonics: 1, 2, 3, 4, 5, 7, 9, 11
Amplitudes: 1.0, 0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03
ADSR: A=0.01s, D=0.5s, S=0.2, R=1.2s (quick strike, long decay)
Velocity impact: Higher velocity = louder, shorter decay
```

**Strings (Programs 48–55: Violin, Viola, Cello, Bass)**
```
Harmonics: 1, 2, 3, 4, 5
Amplitudes: 1.0, 0.5, 0.3, 0.15, 0.08
ADSR: A=0.15s, D=0.2s, S=0.6, R=0.3s (slow attack, warm sustain)
Vibrato: ~5Hz LFO modulation (natural vibrato)
```

**Generic (All other programs)**
```
Harmonics: 1
Amplitudes: 1.0
ADSR: A=0.05s, D=0.1s, S=0.7, R=0.15s (simple sine wave)
```

### MIDI Note → Frequency

```
f (Hz) = 440 × 2^((midi_note - 69) / 12)
```

Example:
- MIDI 60 (C4): f = 440 × 2^(-9/12) = 261.63 Hz (middle C)
- MIDI 69 (A4): f = 440 × 2^(0/12) = 440 Hz
- MIDI 56 (G#3): f = 440 × 2^(-13/12) = 207.65 Hz (trumpet low)

### ADSR Envelope

For each note with duration *t_dur* (seconds):

```
A (Attack):   t ∈ [0, A_time]
  env(t) = t / A_time

D (Decay):    t ∈ [A_time, A_time + D_time]
  env(t) = S + (1 - S) × (1 - (t - A_time) / D_time)

S (Sustain):  t ∈ [A_time + D_time, t_dur - R_time]
  env(t) = S

R (Release):  t ∈ [t_dur - R_time, t_dur]
  env(t) = S × (1 - (t - (t_dur - R_time)) / R_time)
```

Example: Piano note, 1.0-second duration, ADSR (0.01, 0.5, 0.2, 1.2):
```
0.00–0.01s: Ramp from 0 to 1.0 (attack)
0.01–0.51s: Decay from 1.0 to 0.2 (sustain level)
0.51–... : Hold at 0.2 (sustain) until release
1.0–... : If note ends, fade out over 1.2s (release)
```

### Stereo Width (Haas Delay)

To simulate stereo width, a **0.4 ms delay** is applied to the right channel:

```
L(t) = audio(t)
R(t) = audio(t - 0.0004)
```

This subtle delay creates a wider, less "centered" stereo image without introducing phase issues.

### MIDI Parsing

Two backends, in order of preference:

1. **pretty_midi** (if installed)
   - More robust MIDI parsing
   - Handles complex tempo maps, key signatures, etc.

2. **mido** (fallback)
   - Pure Python, no external dependencies
   - Sufficient for standard MIDI

Example console output (mido fallback):
```
[BackgroundMusic] mido fallback: 1524 notes rendered
[BackgroundMusic] output: 213.3s, 48000Hz, stereo, vol=30%
```

### Example Render

**Input MIDI:** seventysixcities.mid
- Trumpet (GM program 56): 3 notes, all octave C (MIDI 48, 60, 72), each 2 seconds
- Piano (GM program 0): C-major chord (notes 60, 64, 67), 2 seconds

**Synthesis Output (48kHz stereo):**
- Note 1 (Trumpet, MIDI 48, 2s): 130.8 Hz (G2) + harmonics → 96,000 samples
- Note 2 (Trumpet, MIDI 60, 2s): 261.6 Hz (C4) + harmonics → 96,000 samples
- ... (chord rendering spreads notes across timeline)
- Total output: ~4.0 seconds, stereo tensor (1, 2, 192000)

---

## Data Structures

### AUDIO

ComfyUI audio representation:

```python
audio = {
    "waveform": torch.Tensor,    # Shape (1, 2, num_samples) or (1, 1, num_samples)
    "sample_rate": int           # Samples per second (48000, 44100, 22050, etc.)
}
```

Example:
```python
audio = {
    "waveform": torch.randn(1, 2, 2_400_000),  # 50 seconds @ 48kHz stereo
    "sample_rate": 48000
}
```

### VIDEO

ComfyUI video representation (after encoding):

```python
video = {
    "frames": np.ndarray or torch.Tensor,  # Shape (T, H, W, 3) or (1, T, H, W, 3)
    "fps": int,                              # Frames per second (24 or 30)
    "codec": str                             # "h264", "h265", "vp9", etc.
}
```

Or (compressed MP4):

```python
video = Path_or_BytesIO_to_mp4_file
```

### DATA

Tabular data representation:

```python
data = {
    "rows": list[dict],      # List of row dictionaries
    "headers": list[str],    # Column names
    "types": dict[str, str]  # Column types ("string", "float", "int", etc.)
}
```

Example (cities):
```python
data = {
    "rows": [
        {"name": "Los Angeles", "lat": 34.05, "lon": -118.24, "year": 1932, "pop": 3_900_000},
        {"name": "Tokyo", "lat": 35.67, "lon": 139.69, "year": 1964, "pop": 13_900_000},
        ...
    ],
    "headers": ["name", "lat", "lon", "year", "pop"],
    "types": {"name": "string", "lat": "float", "lon": "float", "year": "int", "pop": "int"}
}
```

---

## Troubleshooting

### Error: "ImportError: Need either 'pretty_midi' or 'mido'"

**Cause:** Neither MIDI parsing library is installed.

**Fix:**
```bash
# In ComfyUI's Python environment:
pip install mido
# or
pip install pretty_midi
```

**Windows:**
```cmd
C:\Users\[Username]\Documents\ComfyUI\.venv\Scripts\python.exe -m pip install mido
```

---

### Error: "MIDI file not found: [filename]"

**Cause:** MIDI file path invalid or file doesn't exist in search paths.

**Search order:**
1. ComfyUI/custom_nodes/media_machine/
2. ComfyUI/input/
3. Current working directory

**Fix:** Place MIDI file in `ComfyUI/custom_nodes/media_machine/` and use just the filename (e.g., `seventysixcities.mid`), not a full path.

---

### Error: "Return type mismatch... [node] received_type(AUDIO) mismatch input_type(STRING)"

**Cause:** Link ID collision in workflow JSON. Two different links have the same ID, causing output type to be mismatched.

**Fix:** Open `LA_DATA_REPORT_v3.4.json`, search for duplicate link IDs in the `"links"` array. Renumber duplicates to unique IDs.

**Example (Bad):**
```json
"links": [
  [56, 23, 0, 42, 0, "VIDEO"],     // Link ID 56: ProceduralClip → AudioMux
  [56, 45, 0, 42, 2, "AUDIO"]      // DUPLICATE ID 56! Should be 79
]
```

**Fix:**
```json
"links": [
  [56, 23, 0, 42, 0, "VIDEO"],
  [79, 45, 0, 42, 2, "AUDIO"]      // Changed to 79
]
```

---

### Error: "DMM_BackgroundMusic: output volume too quiet during narration"

**Cause:** Default `music_duck_db` is too aggressive (-12dB = 25% volume).

**Fix:** In AudioMux node, change `music_duck_db` widget from -12.0 to -6.0:
- -6.0 dB = 50% volume (audible)
- -3.0 dB = 70% volume (prominent)
- 0.0 dB = 100% volume (no ducking)

**In code:** Modify `dmm_audio_mux.py` line 220:
```python
"default": -6.0,  # Changed from -12.0
```

**In JSON:** Modify `LA_DATA_REPORT_v3.4.json` AudioMux widget value:
```json
"widgets_values": ["replace", 0.8, true, -6.0]  // Changed from -12.0
```

---

### Console Output: "[BackgroundMusic] mido fallback: 0 notes rendered"

**Cause:** MIDI file invalid or empty.

**Fix:**
1. Verify MIDI file is valid: open in DAW (Ableton, Reaper, Cakewalk) and confirm playback
2. Check file size > 100 bytes (empty MIDI would be tiny)
3. Re-export from DAW, ensure saving as standard MIDI (.mid), not proprietary format

---

### Video Output Misaligned with Audio

**Cause:** ProceduralClip duration doesn't match NarrationDistiller + music pad duration.

**Fix:** Use **TimingSync** node to stretch/trim video to match audio:
```
Target audio duration = narration_duration + (pad_to_video - narration_duration)
                      = 22.3s + (41.0s - 22.3s) = 41.0s
Sync video to 41.0s
```

---

### Frame Rate Mismatch (Video "stutters" or "speeds up")

**Cause:** ProceduralClip FPS ≠ VideoConcat FPS ≠ FinalRenderer FPS.

**Fix:** Ensure all video nodes use same FPS:
- Set **ProceduralClip** fps=30
- Set **VideoConcat** fps=30
- Set **FinalRenderer** fps=30

Verify in workflow JSON: all "fps" values = 30.

---

## v3.5 Roadmap

### v3.5 Tier 1: Music Enhancement (Priority)

**Goal:** Improve synthesized MIDI audio quality without external synth engine.

**Proposals:**

1. **MusicEnhancer Node**
   - Input: Synth audio (from BackgroundMusic)
   - LLM-guided prompt → MusicGen API or local model
   - Output: Enhanced, more "realistic" music audio
   - Example: "Upgrade synth trumpet to orchestral warmth, add subtle reverb"

2. **Harmonic Enrichment**
   - Add inharmonic partials (bell-like shimmer) to piano synthesis
   - Implement string resonance simulation (sympathetic vibrations)
   - Velocity-dependent timbre (softer = darker, louder = brighter)

3. **Effects Bus**
   - Reverb (convolution or algorithmic)
   - EQ (parametric 3-band)
   - Compression (dynamic range control)
   - Chorus/Flanger (stereo width enhancement)

### v3.5 Tier 2: Singing Synthesis

**Goal:** Add text-to-singing capability for narrator (not just speech).

**Approach:**
- Bark TTS fork or Vall-E X
- Pitch contour → narration melody alignment
- Example: "Sing the city names in a Broadway musical style"

### v3.5 Tier 3: GPU Fallback Architecture

**Goal:** Auto-detect GPU availability; fall back to CPU-optimized kernels.

**Approach:**
- ONNX Runtime for inference (lighter than PyTorch for inference)
- Quantized models (INT8) for faster MIDI synthesis
- Batch processing for multi-city renders

### Community Feedback Questions

1. **Music Enhancement Priority:** Would you prefer MusicGen integration, or is the current synth sufficient?
2. **Singing Narrator:** Useful for cinematic narratives? (e.g., Olympic opening ceremony style)
3. **Live Streaming:** Interest in real-time pipeline (low-latency MIDI → audio)?
4. **Custom Synthesizers:** Should we bundle a soundfont library (piano, strings), or stay synthesis-only?

---

## Quick Start Checklist

- [ ] Install mido: `pip install mido`
- [ ] Load `LA_DATA_REPORT_v3.4.json` in ComfyUI
- [ ] Verify console: `[DMM] Data Media Machine v3.4: loaded 27/27 nodes`
- [ ] Queue a run with sample CSV data
- [ ] Verify MIDI synthesis: `[BackgroundMusic] output: Xs, 48000Hz, stereo`
- [ ] Check audio mix: narration + music (music audible at -6dB duck)
- [ ] Review final MP4 output

---

## Contact & Support

**GitHub:** [repo TBD]
**Discord Community:** [community TBD]
**Issue Tracker:** [issues TBD]

For v3.4 stability concerns or v3.5 feature suggestions, open an issue or ping the dev team.

---

**Generated:** March 2026
**Version:** DMM v3.4
**Status:** Stable, production-ready for 15-city narrative workflows
**Last Updated:** v3.4 final audio ducking tuning (-6dB default)
