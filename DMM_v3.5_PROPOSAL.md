# DMM v3.5 Proposal
## High-Level Roadmap for Community Feedback

---

## v3.4 Status: STABLE & PRODUCTION READY ✅

**Current Release (March 14, 2026)**

**What's Working:**
- 27 custom nodes covering full data→video pipeline
- Live LA data (weather, earthquakes, air quality, transit, alerts)
- TTS narration with audio enhancement (24kHz mono → 48kHz stereo)
- LTX-2 video generation (5 data-driven clips)
- **NEW: Background music (MIDI synth)** fills the 18.7s silence gap
- Procedural data visualization (24-second animation)
- RTX upscaling to 1920×1080
- Complete audio mux with music ducking

**Tested:**
- RTX 5080: 287 seconds per full render, clean output
- No tensor shape errors, no widget offset issues
- Music properly layers under narration (-12dB), swells after

**Ready for:**
- GitHub release + ComfyUI Manager registration
- Community distribution

---

## v3.5 Vision: Audio Enhancement & Voice

**Problem Statement:**
The current audio is functional but static:
- Narration + background music is straightforward layering
- No adaptive richness or cinematic evolution

**Proposed Solutions (3 tiers):**

### **Tier 1: Music Enhancement (Highest Priority)**
**Node: `DMM_MusicEnhancer`**

Takes the final mixed audio (narration + background music blend) and evolves it generatively.

**Input:**
- Audio tensor (narration + music, 48kHz stereo)
- Text prompt: "LA noir cinematic underscore, subtle strings and synth pads"

**Process:**
- Run through MusicGen or ACE-Step with harmonic conditioning
- Model learns from the existing trumpet/piano arrangement
- Adds texture without replacing or overwhelming the original

**Output:**
- Enhanced audio at same duration, same timing
- Richer orchestration, more cinematic feel
- Still intelligible narration underneath

**Tech:**
- MusicGen (Meta) — lightweight, <2GB VRAM
- ACE-Step 1.5 — <4GB VRAM, supports voice cloning
- Sits right after AudioMux, before SaveVideo

**Effort:** Medium (model integration + prompt templating)

---

### **Tier 2: Singing Synthesis (Optional)**
**Node: `DMM_SingingSynth`**

Turn narration into sung lyrics over the MIDI melody.

**Input:**
- Narration text
- MIDI file (melodic reference)
- Voice profile (optional LoRA)

**Process:**
- Extract phonemes from narration
- Map to MIDI note sequence
- Generate sung vocals (SoulX-Singer or DiffSinger)

**Output:**
- Sung narration at same duration
- Replaces spoken TTS with singing

**Tech:**
- SoulX-Singer — <8GB VRAM, zero-shot singing
- DiffSinger (OpenVPI) — more mature, stronger community

**Effort:** High (phoneme alignment, MIDI-audio sync)

**Note:** This is the "big swing" — makes the entire video feel like a music video rather than a data report. Risky but very cool if it works.

---

### **Tier 3: GPU Fallback (Nice-to-Have)**
**Nodes: `DMM_ReplicateVideoProxy`, `DMM_FalVideoProxy`**

For users with ≤8GB VRAM who can't run local LTX-2.

**Input:**
- Prompts, images, seed (same as local LTX)
- API key (Replicate or Fal.ai)

**Output:**
- Video from hosted service
- Same format as local, slots right into pipeline

**Tech:**
- Replicate.com: $0.01/sec output = ~$0.24 per 24s video
- Fal.ai: Similar pricing, slightly different API

**Effort:** Low (API wrappers, error handling)

---

## Recommendation: Start with Tier 1

**Why Music Enhancement first:**
1. **Low risk** — doesn't change narration, just enhances music
2. **Fast iteration** — MusicGen is well-documented
3. **Immediate impact** — makes audio feel more professional
4. **Proves concept** — validates generative audio in the pipeline before tackling harder stuff (singing)
5. **Community ready** — less controversial than singing (singing alignment is hard)

**If it lands well:**
→ Tier 2 (singing) becomes more appealing to pursue
→ Tier 3 (API fallback) becomes easier to justify

---

## Questions for Community / Team

1. **Music Enhancement:** Would you use generative audio enhancement if it added 10-15s to the pipeline runtime? Or prefer to keep it fast?

2. **Singing Synthesis:** Is turning narration into singing a cool feature or overengineering? Worth the complexity?

3. **GPU Fallback:** Should we build for RTX 4060 users, or is RTX 4070+ the target audience?

4. **Performance Budget:** What's acceptable pipeline runtime? Current: 287s. Would 320s for enhanced music be ok?

5. **Data Sync:** Should music/narration timing be more strict (frame-locked) or is current crossfade/duck approach sufficient?

---

## Implementation Order (if approved)

```
v3.4.1 (2 weeks):
  - Music Enhancement proof-of-concept
  - Test MusicGen vs ACE-Step
  - Benchmark VRAM/time

v3.5 (1 month):
  - Production Music Enhancement node
  - GPU Fallback API nodes (optional)
  - Singing Synthesis (if we go that route)

v3.6 (future):
  - Batch processing dashboard
  - Streaming mode
  - Multi-location support
```

---

## Current Assets Ready for Reuse

- `dmm_background_music.py` — MIDI synthesis (can be adapted for music enhancement)
- `dmm_audio_mux.py` — Audio ducking logic (reusable for dynamic mixing)
- Workflow structure — all nodes tested, no architectural changes needed

---

**Status:** Ready for feedback. v3.4 is stable and shippable independently.
