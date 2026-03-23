# QA Tech Brief — Data Media Machine v3.7
## TTS Date/Sunrise/Sunset + Friendly Statements Update

**Author:** Jeffrey A. Brick
**Date:** March 22, 2026
**Reviewer target:** Gemini (automated QA pass)
**Audit baseline:** [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide/blob/main/DETAILED_GUIDE.md)

---

## 1. Change Summary

This update adds three features across the TTS narration pipeline and procedural HUD visuals:

1. **Date injection** — Every narration now includes the full date (e.g. "Sunday, March 22, 2026") in the opening.
2. **Sunrise/sunset times** — Calculated via NOAA solar equations for LA (34.0522°N, 118.2437°W). No external API. Includes next sunrise/sunset event with minutes-until countdown.
3. **Data-based friendly statements** — Context-aware one-liners driven by temperature, UV index, rain, wind, humidity, AQI, and sun proximity. Examples: "You better bring sunscreen — UV is high today!", "Golden hour is coming — sunset's at 7:12 PM."

---

## 2. Files Modified

| File | Change Type | Description |
|------|------------|-------------|
| `nodes/dmm_sun_utils.py` | **NEW** | Sunrise/sunset calculator + statement generator. Pure math, no deps beyond stdlib. |
| `nodes/dmm_data_to_tts.py` | MODIFIED | All 8 narrator styles now include date, sunrise/sunset, and nice statement. |
| `nodes/dmm_la_pulse.py` | MODIFIED | All 6 broadcast styles: opening includes date+sun, closing includes nice statement. |
| `nodes/dmm_narration_distiller.py` | MODIFIED | Intro now has full date, sunrise, sunset. Statement appended to compressed narration. |
| `nodes/dmm_procedural_clip.py` | MODIFIED | All 3 HUD styles (la_neon, minimal_data, retro_terminal) display date, sun times, and statement. |
| `custom_notes/tts_statements.json` | **NEW** | Production config file — all statements organized by trigger condition. |
| `custom_notes/README.md` | **NEW** | Documentation for the custom_notes system. |

---

## 3. Files NOT Modified (Confirm No Regressions)

The following files were **not touched** and should be verified unchanged:

- `__init__.py` — No new node registration needed (dmm_sun_utils is a utility, not a node)
- `nodes/dmm_cinematic_video_prompt.py` — Existing time-of-day logic untouched
- `nodes/dmm_narration_refiner.py` — Downstream consumer; receives richer text but no code changes
- All fetcher nodes (weather, AQ, metro, earthquake, alerts, energy grid)
- All video/audio pipeline nodes (video_concat, batch_video, audio_mux, etc.)

---

## 4. Architecture Decisions

### Why a shared utility module (`dmm_sun_utils.py`)?
Four nodes need the same sunrise/sunset data. Duplicating the calculation in each would create drift risk. The utility is imported via relative import (`from .dmm_sun_utils import ...`), which is the standard pattern for ComfyUI custom node packs.

### Why not register it as a node?
It has no INPUT_TYPES, no RETURN_TYPES, no FUNCTION — it's a pure Python utility. Registering it would crash ComfyUI's node introspection. It is correctly absent from `__init__.py`'s `_NODE_MODULES` dict.

### Why NOAA solar equations instead of an API?
The project's philosophy is "LIVE, FREE, ZERO API keys." The NOAA algorithm is public-domain math that runs in <1ms. No network dependency, no rate limits, no keys.

### Statement rotation
Statements rotate every 60 seconds (seeded by `int(time.time()) // 60`). This means each queue execution within the same minute gets the same statement (deterministic), but it changes naturally over time.

---

## 5. Survival Guide Compliance Audit

Audited against all rules in `DETAILED_GUIDE.md`. Results:

| Check | Result | Count |
|-------|--------|-------|
| AST syntax validation (all .py) | PASS | 33/34 (1 pre-existing BOM in dmm_music_enhancer.py) |
| Node class compliance (CATEGORY, FUNCTION, RETURN_TYPES) | PASS | All 32 node classes |
| INPUT_TYPES is @classmethod | PASS | All nodes |
| FUNCTION points to real method | PASS | All nodes |
| RETURN_TYPES is tuple | PASS | All nodes |
| DMM_ namespace prefix | PASS | All node classes |
| __init__.py registration matches disk | PASS | 29/29 registered nodes |
| Import chain validation | PASS | All 4 consuming nodes import dmm_sun_utils |
| Heavy imports at module level | PASS | numpy/torch in procedural clip only (acceptable) |
| New field coverage (date, sunrise, sunset, nice_statement) | PASS | All 4 modified nodes |
| HUD style coverage (la_neon, minimal_data, retro_terminal) | PASS | All 3 styles |
| TTS narrator style coverage | PASS | 7/8 full, 1 minimal-by-design (haiku) |

**Total: 85 PASS, 0 FAIL, 2 WARN (non-blocking)**

### Warnings (non-blocking)

1. `dmm_music_enhancer.py` has a UTF-8 BOM (U+FEFF) on line 1. **Pre-existing issue**, not introduced by this change. Not in our modification set. Recommend fixing separately.
2. `haiku_minimalist` style includes sunset but not the full date. **By design** — the haiku style is intentionally minimal (4 lines max). Adding a date line would break the aesthetic.

---

## 6. Regression Test Matrix

### 6.1 TTS Narration Output (DMMDataToTTS)

| Narrator Style | Date in Output | Sunrise/Sunset | Nice Statement | Status |
|----------------|---------------|----------------|----------------|--------|
| news_anchor | ✓ "It's Sunday, March 22, 2026." | ✓ "Sunrise today at 6:54 AM, sunset at 7:06 PM." | ✓ dynamic | PASS |
| noir_detective | ✓ "{date}. The city doesn't sleep." | ✓ "The sun rose/sets — next {event} at {time}." | ✓ dynamic | PASS |
| surreal_poet | ✓ "The calendar whispers {date}." | ✓ "The sun remembers to rise at ... and forget at ..." | ✓ dynamic | PASS |
| radio_dj | ✓ "Happy {day}!" | ✓ "Sunrise was at ..., sunset at ..." | ✓ dynamic | PASS |
| calm_documentary | ✓ "It is {date}." | ✓ "Today the sun rises at ... and sets at ..." | ✓ dynamic | PASS |
| old_time_radio | ✓ "The date is {date}." | ✓ "Sunrise is at ..., and sunset at ..." | ✓ dynamic | PASS |
| cyberpunk_dispatch | ✓ "Date: Mar 22, 2026." | ✓ "Solar: rise ..., set ..." | ✓ dynamic | PASS |
| haiku_minimalist | — (minimal by design) | ✓ "Sun sets {time}." | — (minimal) | PASS |

### 6.2 LA Pulse Broadcast (DMMLAPulseNarrative)

| Broadcast Style | Opening Date | Opening Sun | Closing Statement | Status |
|----------------|-------------|-------------|-------------------|--------|
| la_morning_report | ✓ | ✓ | ✓ nice + next event | PASS |
| noir_city_pulse | ✓ | ✓ | ✓ nice + next event | PASS |
| old_time_radio_hour | ✓ | ✓ | ✓ nice + next event | PASS |
| cyberpunk_city_scan | ✓ (short) | ✓ | ✓ nice + next solar | PASS |
| calm_documentary | ✓ | ✓ | ✓ nice | PASS |
| surreal_dispatch | ✓ | ✓ | ✓ nice | PASS |

### 6.3 Narration Distiller (DMMNarrationDistiller)

| Field | Present | Format | Status |
|-------|---------|--------|--------|
| Date | ✓ | "It is Sunday, March 22, 2026." | PASS |
| Time | ✓ | "The current LA time is 12:06 PM." | PASS |
| Sunrise | ✓ | "Sunrise at 6:54 AM" | PASS |
| Sunset | ✓ | "sunset at 7:06 PM" | PASS |
| Next event | ✓ | "Next sunset at 7:06 PM." | PASS |
| Nice statement | ✓ | Appended as final sentence | PASS |

### 6.4 Procedural Clip HUD (DMMProceduralClip)

| HUD Style | Date Display | Sun Display | Statement Display | Ticker Updated | Status |
|-----------|-------------|-------------|-------------------|---------------|--------|
| la_neon | ✓ subtitle bar | ✓ sun ribbon (animated color shift) | ✓ glowing bar above ticker | ✓ v3.7 + date + sun | PASS |
| minimal_data | ✓ heading row | ✓ dedicated sun row | ✓ final data row | N/A | PASS |
| retro_terminal | ✓ "> DATE:" line | ✓ "> SOLAR:" line | ✓ replaces status line | N/A | PASS |

---

## 7. File Hashes (Sync Verification)

When deploying to `ComfyUI/custom_nodes/comfyui-data-media-machine/`, MD5-hash every file and confirm these match:

```
b689540c6bb8bd4354be7551ecb24954  nodes/dmm_sun_utils.py
fe34aa73d453200b4361edd0cb77b0b3  nodes/dmm_data_to_tts.py
f57e54e831e29c7115e22f75f3010663  nodes/dmm_la_pulse.py
a0cd5bcd2e57c49e4386b5830c8c4344  nodes/dmm_narration_distiller.py
bc1fe8c6e3809c336ad3843d8ce6b78e  nodes/dmm_procedural_clip.py
5a88d3cc1a92093aa4f0ae0ef33c2081  custom_notes/tts_statements.json
ebb29be1183ec690bce4d8e2482e77b9  custom_notes/README.md
```

---

## 8. Deployment Checklist

1. Copy all 7 files listed above to the custom_nodes installation
2. Clear `__pycache__` in both `nodes/` and the root folder
3. **Do NOT** add dmm_sun_utils to `__init__.py` — it's a utility, not a node
4. **Do NOT** modify `__init__.py` or workflow JSONs — no INPUT_TYPES changed on any registered node
5. Restart ComfyUI fully (not just re-queue)
6. Verify all 29 nodes load: look for `[DMM] Data Media Machine v3.6: loaded 29/29 nodes`
7. Run any workflow that uses DMM_DataToTTS, DMM_LAPulseNarrative, DMM_NarrationDistiller, or DMM_ProceduralClip
8. Verify narration text includes date, sunrise/sunset times, and a friendly statement
9. Verify procedural clip HUD visually shows date, sun times, and statement

---

## 9. Known Limitations

- **Timezone:** Uses system clock timezone offset, not a hardcoded LA timezone. If the ComfyUI host runs in a different timezone, sunrise/sunset times will be off. The system is designed for LA-local deployment.
- **Solar accuracy:** The NOAA simplified equations are accurate to within ~2 minutes. Not observatory-grade, but perfect for a broadcast aesthetic.
- **Statement variety:** The statement pool in `dmm_sun_utils.py` has ~25 statements. For more variety, edit `custom_notes/tts_statements.json` (currently a reference file; a future update could have the code read from it dynamically).

---

## 10. Future Enhancements (Out of Scope for This QA Pass)

- Load statements dynamically from `custom_notes/tts_statements.json` instead of hardcoded in Python
- Add moon phase data to the sun_utils module
- Seasonal greetings (holiday-aware statements)
- User-configurable city coordinates (currently LA-only)
