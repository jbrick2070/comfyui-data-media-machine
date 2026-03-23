"""
DMM Music Enhancer -- MusicGen-melody audio-to-audio music enhancement.

Takes the background music track from BackgroundMusic and runs it through
Meta's MusicGen-melody for music-to-music style transfer via chroma/melody
conditioning. Each run selects one of 40 diverse LA-community style prompts using
data-driven scoring (time of day + weather conditions) and generates
music that follows the harmonic contour of the original.

Architecture:
  BackgroundMusic (48kHz stereo) -> MusicEnhancer -> AudioMux -> SaveVideo

Melody conditioning: MusicGen reads the chroma (pitch/harmonic) features
of the input audio and generates new music that tracks those shapes while
adopting the prompted style. The result is blended back with the original
at the specified mix ratio.

Strength widget maps to MusicGen guidance_scale (how strongly the text
prompt drives the output vs the input melody):
  Low  (0.05-0.20): Melody leads -- output closely follows input harmony
  Mid  (0.30-0.50): Balanced -- style prompt and melody co-guide output
  High (0.60-0.95): Prompt dominates -- melody is a loose reference

num_inference_steps widget = seconds of audio to generate (5-30s).
Output is looped (with raised-cosine crossfade) or trimmed to match input.

No external API required. Model loads directly into VRAM (~1.5GB).
Cached after first load.

Requirements:
  - transformers >= 4.35.0
  - Install: pip install transformers

VRAM Budget:
  - MusicGen-melody: ~1.5GB VRAM
  - Much lighter than Stable Audio Open (2-3GB)
  - Safe on RTX 5080 (24GB), RTX 4070+ (12GB+), RTX 3060 (12GB)

Author: Jeffrey A. Brick
Version: 3.7 (data-driven prompt selection, 40 LA styles, no external API)
"""

import gc
import io
import logging
import random
import time
import wave

import numpy as np
import torch

log = logging.getLogger("DMM.MusicEnhancer")

# MusicGen-melody native sample rate
MUSICGEN_SR = 32000


# ---------------------------------------------------------------------------
# LA-themed style prompt pool
# Each run randomly picks one to guide the music-to-music style transfer.
# References real synthesizer models and instruments native to LA 80s
# culture and LA indigenous culture.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tagged LA style prompts -- each has mood + energy for data-driven selection
#   mood:   calm, warm, moody, intense, spiritual, bright
#   energy: low (0.0-0.3), mid (0.3-0.6), high (0.6-1.0)
# ---------------------------------------------------------------------------
LA_STYLE_PROMPTS = [
    # (prompt_text, mood, energy)

    # ── BLACK LA ─────────────────────────────────────────────────────
    ("West coast G-funk, Roland TR-808 bass thump, Moog Minimoog sliding bassline, "
     "talk box melody over pentatonic chords, Parliament-Funkadelic groove, "
     "Compton summer heat", "warm", 0.6),

    ("Los Angeles Central Avenue jazz 1948, upright bass walking line, wire brushed snare, "
     "Dexter Gordon tenor saxophone blue note improvisation, Steinway grand chord stabs, "
     "after-hours glow, Dunbar Hotel midnight session", "moody", 0.3),

    ("Los Angeles Black church gospel, Hammond B3 organ full drawbar swell, "
     "soulful soprano lead over four-part choir, tambourine on the two and four, "
     "congregation clap-back, sanctified reverb, West Adams Sunday morning", "bright", 0.7),

    ("Los Angeles 1986 R&B slow jam, Yamaha DX7 electric piano, Linn LM-1 snare crack, "
     "Fender Rhodes chord stabs, velvet reverb tail, late-night studio session", "warm", 0.3),

    ("Leimert Park spoken word jazz, upright bass ostinato, djembe pulse, "
     "alto saxophone melody floating over brushed snare, "
     "World Stage open mic, Crenshaw evening breeze", "calm", 0.3),

    ("Watts street funk, Hohner Clavinet D6 wah stabs, slap bass groove, "
     "tight horn section staccato hits, Roland TR-707 snare, "
     "Simon Rodia tower silhouette at golden hour", "bright", 0.7),

    ("Los Angeles underground hip-hop, E-mu SP-1200 crunchy drum chops, "
     "Akai MPC 3000 swing quantize, dusty jazz vinyl sample, "
     "Project Blowed freestyle cipher, Leimert Park after dark", "moody", 0.5),

    # ── CHICANO / LATIN LA ───────────────────────────────────────────
    ("East Los Angeles Chicano punk rock, Fender Telecaster overdriven power chords, "
     "fast snare blast beat, shouted bilingual vocals, "
     "Thee Midniters garage energy, backyard show in Boyle Heights", "intense", 0.9),

    ("East LA lowrider soul, Fender Stratocaster clean-tone chord strum, "
     "marimba counter-melody, smooth doo-wop harmonies, light conga groove, "
     "cruising Whittier Boulevard on a warm Sunday evening", "warm", 0.3),

    ("South Central Chicano rap, Roland TR-808 deep kick, "
     "nylon string acoustic guitar picked melody, slow tempo cruiser beat, "
     "Eazy-E era swagger, Lincoln Heights street corner twilight", "moody", 0.4),

    ("Los Angeles cumbia sonidera, Korg bass synth wobble, guira scrape rhythm, "
     "button accordion melodic lead, crowd echo chant, "
     "MacArthur Park Sunday dance, Pico-Union block party energy", "bright", 0.7),

    ("Banda sinaloense brass, tuba oom-pah bassline, trumpet melodic fanfare, "
     "clarinet counter-melody, tarola snare roll, "
     "mariachi brass warmth, Olvera Street evening celebration", "bright", 0.6),

    ("East Los Angeles Aztec ceremony fusion, huehuetl heartbeat drum, teponaztli log drum "
     "call-and-response, conchero shell rattle cascade, copal smoke and street murals, "
     "Boyle Heights dusk ritual", "spiritual", 0.4),

    # ── ASIAN LA ─────────────────────────────────────────────────────
    ("Thai Town luk thung pop, phin electric guitar twangy melody, "
     "khaen mouth organ sustained chord, ching finger cymbals on the beat, "
     "ramwong dance rhythm, Hollywood and Western neon glow, warm night air", "warm", 0.5),

    ("Little Saigon Vietnamese bolero, dan tranh zither tremolo melody, "
     "nylon string guitar arpeggio, soft brushed snare, "
     "Saigon cafe nostalgia, Westminster evening, jasmine tea steam", "calm", 0.2),

    ("Koreatown fusion, gayageum plucked melody over electronic beat, "
     "janggu hourglass drum rhythmic pattern, synth pad shimmer, "
     "pansori vocal intensity meets neon pop, Olympic Boulevard midnight", "intense", 0.6),

    ("Little Tokyo taiko fusion, odaiko thundering bass drum, shime-daiko sharp crack, "
     "shamisen plucked melody, electronic sub-bass undertow, "
     "Nisei Week festival energy, First Street lantern glow", "intense", 0.8),

    ("Historic Filipinotown kulintang ensemble, tuned gong cascade melody, "
     "rondalla string pluck harmony, bamboo percussion, "
     "bandurria tremolo, Temple Street evening warmth, ocean memory", "calm", 0.3),

    ("Chinatown erhu and pipa duet, erhu bowed melody legato, "
     "pipa rapid tremolo pluck, woodblock steady pulse, "
     "Cantonese opera fragment, Broadway and Hill Street, red lantern glow", "moody", 0.3),

    ("Little India Artesia fusion, tabla rhythmic cycle teental, "
     "sitar melodic raga phrase, tanpura drone, electronic bass pulse, "
     "Pioneer Boulevard spice market evening, Bollywood meets LA bass music", "warm", 0.5),

    # ── INDIGENOUS LA ────────────────────────────────────────────────
    ("Tongva Gabrielino indigenous soundscape, elderberry flute breathy melody, "
     "deer-hoof rattle shimmer, clapstick rhythm over basket drum pulse, "
     "coastal sage and salt wind, Ballona Creek ceremony", "spiritual", 0.2),

    ("Chumash coastal California ceremonial, bone whistle ascending melody, "
     "gourd rattle wash, split-stick clapper steady pulse, ocean drum resonance, "
     "Santa Monica Mountains morning mist and wild sage", "spiritual", 0.2),

    # ── LA PUNK / HARDCORE ───────────────────────────────────────────
    ("Hollywood 1977 punk rock, buzzing Gibson Les Paul power chords, "
     "snare on every beat, shouted vocal aggression, "
     "The Masque basement reverb, Cherokee Avenue grit and sweat", "intense", 0.9),

    ("Hermosa Beach hardcore punk, Marshall amp full distortion, "
     "blast beat double-time drums, palm-muted breakdowns, "
     "Church on York energy, all-ages show in a garage, Redondo heat", "intense", 1.0),

    ("Los Angeles ska-punk, upstroke guitar rhythm, walking bass line, "
     "trumpet and trombone unison riff, fast hi-hat, "
     "Sublime Long Beach vibes, backyard party under string lights", "bright", 0.8),

    # ── PERSIAN / ARMENIAN / MIDDLE EASTERN LA ───────────────────────
    ("Tehrangeles Persian pop, tar plucked melodic ornament, tombak goblet drum rhythm, "
     "Yamaha DX7 string pad, santour hammered dulcimer shimmer, "
     "Westwood evening, pomegranate sweetness, Farsi love ballad warmth", "warm", 0.4),

    ("Glendale Armenian folk fusion, duduk reedy sustained melody, "
     "dhol hand drum driving rhythm, oud ornamental run, "
     "zurna bright call, Brand Boulevard evening, apricot blossom memory", "moody", 0.4),

    # ── CARIBBEAN / CENTRAL AMERICAN LA ──────────────────────────────
    ("South LA Belizean punta rock, turtle shell percussion scrape, "
     "segunda bass guitar groove, primera lead guitar melody, "
     "paranda acoustic rhythm, Garifuna celebration, palm tree sway", "bright", 0.7),

    ("Inglewood dancehall riddim, deep sub-bass wobble, "
     "digital snare crack, hi-hat triplet roll, "
     "ragga vocal energy, Crenshaw cruise, palm-lined boulevard bass bounce", "intense", 0.7),

    ("Pico-Union Salvadoran xuc dance, marimba wooden key melody, "
     "acoustic guitar rasgueado strum, light percussion shaker, "
     "pupusa stand Saturday night, Central American warmth in LA", "warm", 0.5),

    # ── ELECTRONIC / MODERN LA ───────────────────────────────────────
    ("Los Angeles 1984 synthpop, Oberheim OB-Xa polyphonic pad wash, "
     "Sequential Prophet-5 lead arpeggio, Roland TR-808 drum machine, "
     "Sunset Strip neon, cold-wave pulse", "moody", 0.5),

    ("Los Angeles synthwave, Roland Jupiter-8 sweeping pads, Roland Juno-106 chorus shimmer, "
     "analog sequencer pulse, Pacific Coast Highway midnight drive, neon rain reflection",
     "moody", 0.4),

    ("Los Angeles late-night ambient, Roland SH-101 sub-bass drone, "
     "Sequential Circuits Prophet-VS granular shimmer pad, freeway overpass texture, "
     "Blade Runner 2019 skyline, slow evolving resonance", "calm", 0.1),

    ("Low End Theory beat scene, Roland SP-404 warped sample chop, "
     "modular synthesizer generative bleeps, broken beat rhythm, "
     "Lincoln Heights warehouse, projector glow, head-nod bass weight", "moody", 0.5),

    # ── HOLLYWOOD / CINEMATIC ────────────────────────────────────────
    ("Hollywood cinematic score, sweeping string orchestra swell, French horn heroic motif, "
     "Steinway grand piano cascading runs, deep timpani roll, "
     "Bernard Herrmann tension and release, wide-screen grandeur", "intense", 0.6),

    ("Laurel Canyon 1969 folk rock, Rickenbacker 12-string jangle, "
     "three-part vocal harmony, upright bass warm pulse, "
     "tambourine on the offbeat, canyon creek and eucalyptus breeze", "calm", 0.3),

    ("Venice Beach sunset drum circle, djembe lead rhythm, "
     "conga call and response, shekere shaker wash, "
     "ocean wave texture underneath, boardwalk chatter fade, golden hour glow", "warm", 0.5),

    ("Boyle Heights mariachi, vihuela rhythmic strum, guitarron bass pulse, "
     "dual trumpet melodic fanfare, violin lyrical counter-melody, "
     "Mariachi Plaza at dusk, son jalisciense dignity and pride", "bright", 0.6),
]


# ---------------------------------------------------------------------------
# Data-driven prompt selector -- picks style based on weather + time of day
# ---------------------------------------------------------------------------
# Time-of-day -> preferred moods (weighted)
_TOD_MOOD_MAP = {
    "early_morning": ["calm", "spiritual", "warm"],        # 5-8 AM
    "morning":       ["bright", "warm", "calm"],            # 8-11 AM
    "midday":        ["bright", "intense", "warm"],         # 11 AM-2 PM
    "afternoon":     ["warm", "bright", "moody"],           # 2-5 PM
    "evening":       ["moody", "warm", "intense"],          # 5-8 PM
    "night":         ["moody", "calm", "intense"],          # 8-11 PM
    "late_night":    ["calm", "moody", "spiritual"],        # 11 PM-5 AM
}

# Weather condition -> mood + energy modifiers
_WEATHER_MOOD = {
    "clear":        ("bright", 0.1),
    "sunny":        ("bright", 0.1),
    "clouds":       ("moody", -0.05),
    "overcast":     ("moody", -0.1),
    "rain":         ("moody", -0.15),
    "drizzle":      ("calm", -0.1),
    "thunderstorm": ("intense", 0.2),
    "snow":         ("calm", -0.2),
    "fog":          ("calm", -0.15),
    "mist":         ("calm", -0.1),
    "haze":         ("moody", -0.05),
    "wind":         ("intense", 0.1),
}

# Data-driven tempo/feel modifiers appended to the selected prompt
_TEMPO_MODIFIERS = {
    "slow":   "slow tempo, relaxed pace, breathing room between notes",
    "mid":    "moderate tempo, steady groove, natural rhythm",
    "fast":   "uptempo, driving rhythm, energetic pace",
}

_FEEL_MODIFIERS = {
    "rain":         "rain texture in the background, wet reverb, muted highs",
    "thunderstorm": "thunder rumble undertone, storm energy, electric tension",
    "wind":         "wind noise wash, gusting dynamics, restless motion",
    "fog":          "foggy atmosphere, distant sounds, muted clarity",
    "clear":        "crisp clarity, open air, bright presence",
    "hot":          "hazy heat shimmer, slow heavy bass, sun-baked groove",
    "cold":         "crystalline high frequencies, brittle textures, frosty air",
}


def _get_time_of_day_detailed():
    """Return detailed time-of-day bucket."""
    from datetime import datetime
    hour = datetime.now().hour
    if hour < 5:
        return "late_night"
    elif hour < 8:
        return "early_morning"
    elif hour < 11:
        return "morning"
    elif hour < 14:
        return "midday"
    elif hour < 17:
        return "afternoon"
    elif hour < 20:
        return "evening"
    elif hour < 23:
        return "night"
    else:
        return "late_night"


def _parse_weather_for_music(weather_str):
    """Extract condition, temp, wind from weather summary string.

    Returns dict with keys: condition, temp_f, wind_mph
    """
    import re
    result = {"condition": "", "temp_f": 72, "wind_mph": 5}
    if not weather_str:
        return result

    # Condition from first segment after colon
    desc = re.search(r":\s*([^|]+)", weather_str)
    if desc:
        result["condition"] = desc.group(1).strip().lower()

    # Temperature
    temp = re.search(r"(\d+)\s*[°*]?\s*F", weather_str, re.IGNORECASE)
    if temp:
        result["temp_f"] = int(temp.group(1))

    # Wind
    wind = re.search(r"[Ww]ind\s*:?\s*(\d+)\s*mph", weather_str, re.IGNORECASE)
    if wind:
        result["wind_mph"] = int(wind.group(1))

    return result


def select_data_driven_prompt(rng, weather_summary=""):
    """Pick an LA style prompt based on time of day and weather conditions.

    Returns (final_prompt_string, selection_info_string).
    The prompt is the base LA style + data-driven tempo/feel modifiers.
    """
    tod = _get_time_of_day_detailed()
    weather = _parse_weather_for_music(weather_summary)
    condition = weather["condition"]
    temp_f = weather["temp_f"]
    wind_mph = weather["wind_mph"]

    # ── 1. Score each prompt by mood + energy fit ────────────────────
    preferred_moods = _TOD_MOOD_MAP.get(tod, ["warm", "moody", "calm"])

    # Weather mood influence
    weather_mood = None
    energy_shift = 0.0
    for key, (mood, shift) in _WEATHER_MOOD.items():
        if key in condition:
            weather_mood = mood
            energy_shift = shift
            break

    # Target energy from wind + time
    base_energy = 0.4
    if tod in ("late_night", "early_morning"):
        base_energy = 0.2
    elif tod in ("midday", "afternoon"):
        base_energy = 0.5
    elif tod == "evening":
        base_energy = 0.45

    # Wind pushes energy up
    if wind_mph > 15:
        base_energy += 0.15
    elif wind_mph > 25:
        base_energy += 0.25

    # Temperature extremes push energy
    if temp_f > 90:
        base_energy -= 0.1  # heat slows things down
    elif temp_f < 50:
        base_energy -= 0.05

    target_energy = max(0.0, min(1.0, base_energy + energy_shift))

    # Score each prompt
    scored = []
    for prompt_text, mood, energy in LA_STYLE_PROMPTS:
        score = 0.0

        # Mood match: primary preferred mood = 3pts, secondary = 2pts, tertiary = 1pt
        if mood in preferred_moods:
            idx = preferred_moods.index(mood)
            score += (3 - idx)  # 3, 2, or 1

        # Weather mood bonus — nudges toward weather mood without locking it in
        if weather_mood and mood == weather_mood:
            score += 1.0  # was 2.0: reduced so jitter can still break through

        # Energy proximity (closer = better, max 3pts)
        energy_dist = abs(energy - target_energy)
        score += max(0, 3.0 - energy_dist * 5.0)

        # Jitter — large enough to let non-dominant moods occasionally win
        score += rng.random() * 3.0  # was 1.5: wider window for variety

        scored.append((score, prompt_text, mood, energy))

    # Sort by score descending, pick from top 8 to keep variety across runs
    scored.sort(key=lambda x: -x[0])
    top_n = scored[:8]  # was 5: wider pool gives non-moody styles a path in
    pick = rng.choice(top_n)
    base_prompt = pick[1]
    picked_mood = pick[2]
    picked_energy = pick[3]

    # ── 2. Append data-driven modifiers ──────────────────────────────
    modifiers = []

    # Tempo feel based on target energy
    if target_energy < 0.3:
        modifiers.append(_TEMPO_MODIFIERS["slow"])
    elif target_energy > 0.65:
        modifiers.append(_TEMPO_MODIFIERS["fast"])
    else:
        modifiers.append(_TEMPO_MODIFIERS["mid"])

    # Weather feel
    for key, feel in _FEEL_MODIFIERS.items():
        if key in condition:
            modifiers.append(feel)
            break
    else:
        # Temperature-based feel
        if temp_f > 90:
            modifiers.append(_FEEL_MODIFIERS["hot"])
        elif temp_f < 50:
            modifiers.append(_FEEL_MODIFIERS["cold"])

    final_prompt = base_prompt + ", " + ", ".join(modifiers)

    info = (f"tod={tod} | condition={condition or 'n/a'} | temp={temp_f}F | "
            f"wind={wind_mph}mph | target_energy={target_energy:.2f} | "
            f"picked_mood={picked_mood} picked_energy={picked_energy:.1f}")

    return final_prompt, info


# ---------------------------------------------------------------------------
# Model cache (singleton -- loads once, reuses across executions)
# ---------------------------------------------------------------------------
_model_cache = {
    "processor": None,
    "model": None,
    "device": None,
}


def _get_or_load_model(device: torch.device = None):
    """Load MusicGen-melody processor and model, cached after first call.

    Invalidates and reloads if called with a different device than the
    cached model (e.g. user switches --cpu or GPU IDs between runs). (R7 fix)

    Returns (processor, model) tuple, or (None, None) if not available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Invalidate cache on device change (R7)
    if (
        _model_cache["model"] is not None
        and _model_cache["device"] is not None
        and str(_model_cache["device"]) != str(device)
    ):
        log.info(
            "  Device changed (%s -> %s) -- clearing model cache",
            _model_cache["device"], device,
        )
        _model_cache["processor"] = None
        _model_cache["model"] = None
        _model_cache["device"] = None

    if _model_cache["model"] is not None:
        return _model_cache["processor"], _model_cache["model"]

    try:
        # Suppress HF hub / httpx / transformers verbosity before any network calls.
        # Eliminates: HTTP HEAD spam, unauthenticated-request warnings,
        # weight-loading tqdm bar, and LOAD REPORT unexpected-key table.
        import logging as _stdlib_logging
        for _noisy in (
            "httpx",
            "httpcore",
            "huggingface_hub",
            "huggingface_hub.utils._headers",
            "filelock",
        ):
            _stdlib_logging.getLogger(_noisy).setLevel(_stdlib_logging.WARNING)

        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
        import transformers as _transformers
        _transformers.logging.set_verbosity_error()

        log.info("  Loading MusicGen-melody (first run only)...")
        t0 = time.time()

        processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
        model = MusicgenMelodyForConditionalGeneration.from_pretrained(
            "facebook/musicgen-melody",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = model.to(device)
        model.eval()

        elapsed = time.time() - t0
        log.info("  MusicGen-melody loaded in %.1fs on %s", elapsed, device)

        _model_cache["processor"] = processor
        _model_cache["model"] = model
        _model_cache["device"] = device
        return processor, model

    except ImportError:
        log.warning(
            "transformers not installed. "
            "Run: pip install transformers"
        )
        return None, None
    except Exception as e:
        log.error("Failed to load MusicGen-melody: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Audio resampling utility
# ---------------------------------------------------------------------------
def _resample_tensor(tensor: torch.Tensor, from_sr: int, to_sr: int) -> torch.Tensor:
    """Resample audio tensor using torchaudio polyphase resampler. (R2 fix)

    Polyphase resampling avoids the high-frequency aliasing that F.interpolate
    introduces on the 48kHz->32kHz->48kHz round trip.
    Falls back to linear interpolation if torchaudio is not available.

    Handles both (C, T) and (B, C, T) shapes. Returns same shape as input.
    """
    if from_sr == to_sr:
        return tensor

    squeezed = tensor.ndim == 2
    if squeezed:
        tensor = tensor.unsqueeze(0)

    try:
        import torchaudio
        resampler = torchaudio.transforms.Resample(
            orig_freq=from_sr, new_freq=to_sr,
        ).to(tensor.device)
        resampled = resampler(tensor.float())
    except ImportError:
        log.warning("torchaudio unavailable -- falling back to linear resampling")
        new_length = int(tensor.shape[-1] * to_sr / from_sr)
        resampled = torch.nn.functional.interpolate(
            tensor.float(),
            size=new_length,
            mode="linear",
            align_corners=False,
        )

    if squeezed:
        resampled = resampled.squeeze(0)

    return resampled


# ---------------------------------------------------------------------------
# Crossfade loop -- for generated clips shorter than the input timeline
# ---------------------------------------------------------------------------
def crossfade_loop(
    audio: torch.Tensor,
    target_samples: int,
    crossfade_samples: int = None,
    sample_rate: int = 48000,
) -> torch.Tensor:
    """Seamlessly loop audio to reach target_samples using raised-cosine crossfade.

    If MusicGen's generated clip is shorter than the video timeline,
    this loops it with smooth crossfades at boundaries.

    Args:
        audio: Tensor (..., samples)
        target_samples: Desired total length in samples.
        crossfade_samples: Overlap region. Defaults to 2s or 25% of clip.
        sample_rate: Used to calculate default crossfade duration.

    Returns:
        Tensor (..., target_samples) with seamless loops.
    """
    src_len = audio.shape[-1]

    if src_len >= target_samples:
        return audio[..., :target_samples]

    if crossfade_samples is None:
        crossfade_samples = min(sample_rate * 2, src_len // 4)
    # Clamp to src_len // 3 BEFORE computing loop_body_len. (R5 fix)
    # Without this, crossfade_samples >= src_len makes loop_body_len <= 0
    # and the while loop never advances pos, spinning to the safety ceiling.
    crossfade_samples = max(min(crossfade_samples, src_len // 3), 1)

    # Raised-cosine fade curves
    fade = torch.linspace(0.0, 1.0, crossfade_samples, device=audio.device)
    fade = 0.5 * (1.0 - torch.cos(fade * torch.pi))
    fade_in = fade
    fade_out = 1.0 - fade

    loop_body_len = src_len - crossfade_samples
    result = torch.zeros(*audio.shape[:-1], target_samples, device=audio.device)

    pos = 0
    iteration = 0
    while pos < target_samples:
        remaining = target_samples - pos

        if iteration == 0:
            copy_len = min(src_len, remaining)
            result[..., pos:pos + copy_len] = audio[..., :copy_len]
            pos += loop_body_len
        else:
            if pos - crossfade_samples >= 0 and remaining > 0:
                xf_start = pos - crossfade_samples
                xf_end = pos
                result[..., xf_start:xf_end] *= fade_out
                head = audio[..., :crossfade_samples] * fade_in
                result[..., xf_start:xf_end] += head

            body_remaining = min(loop_body_len, remaining)
            copy_src = audio[..., crossfade_samples:crossfade_samples + body_remaining]
            copy_len = copy_src.shape[-1]

            if pos + copy_len > target_samples:
                copy_len = target_samples - pos

            if copy_len > 0:
                result[..., pos:pos + copy_len] = (
                    audio[..., crossfade_samples:crossfade_samples + copy_len]
                )

            pos += loop_body_len

        iteration += 1
        if iteration > (target_samples // max(loop_body_len, 1)) + 2:
            break

    log.info(
        "  Crossfade loop: %d iterations, %.1fs -> %.1fs",
        iteration,
        src_len / sample_rate,
        target_samples / sample_rate,
    )
    return result[..., :target_samples]


# ---------------------------------------------------------------------------
# Dry/wet mix utility (channel + length matching)
# ---------------------------------------------------------------------------
def mix_audio(dry: torch.Tensor, wet: torch.Tensor, mix_val: float) -> torch.Tensor:
    """Blend original (dry) and generated (wet) audio.

    Handles channel count and length mismatches automatically.
    Applies soft clip (tanh) to prevent overs.
    """
    # Ensure both tensors are on the same device
    if dry.device != wet.device:
        wet = wet.to(dry.device)
    # Also ensure matching dtype
    if dry.dtype != wet.dtype:
        wet = wet.to(dry.dtype)

    dry_ch = dry.shape[-2] if dry.ndim >= 2 else 1
    wet_ch = wet.shape[-2] if wet.ndim >= 2 else 1

    if dry_ch != wet_ch:
        log.info("  Channel mismatch: dry=%dch wet=%dch -- adjusting wet", dry_ch, wet_ch)
        if dry_ch == 2 and wet_ch == 1:
            wet = wet.repeat(*(1,) * (wet.ndim - 2), 2, 1)
        elif dry_ch == 1 and wet_ch == 2:
            wet = wet[:, :1, :] if wet.ndim == 3 else wet[:1, :]
        else:
            log.warning("  Unusual channel mismatch (%d vs %d) -- truncating wet", dry_ch, wet_ch)
            wet = wet[..., :dry_ch, :]

    dry_len = dry.shape[-1]
    wet_len = wet.shape[-1]
    if dry_len > wet_len:
        pad = torch.zeros(*wet.shape[:-1], dry_len - wet_len, device=wet.device)
        wet = torch.cat([wet, pad], dim=-1)
    elif wet_len > dry_len:
        wet = wet[..., :dry_len]

    mixed = (1.0 - mix_val) * dry + mix_val * wet
    return torch.tanh(mixed)


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancer
# ---------------------------------------------------------------------------
class DMM_MusicEnhancer:
    """Music-to-music enhancement using MusicGen-melody.

    Sits between BackgroundMusic and AudioMux in the DMM pipeline.
    Extracts the harmonic/chroma contour of the input music and generates
    new music conditioned on both the contour and a randomly selected
    LA-themed style prompt. Result is dry/wet blended with the original.

    Widget mapping:
      strength          -> guidance_scale (0.05->1.5, 0.20->2.8, 0.50->5.5)
                          Low = melody-driven | High = style-driven
      mix               -> dry/wet blend (0=original, 1=MusicGen only)
      prompt_override   -> leave blank for random LA prompt
      seed              -> -1 for random each run
      num_inference_steps -> seconds of audio to generate (5-30s)
                            looped or trimmed to match input length
    """

    CATEGORY = "DataMediaMachine"
    FUNCTION = "enhance"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.05,
                        "max": 0.95,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Prompt guidance strength. "
                            "Low=melody leads. High=style prompt leads."
                        ),
                    },
                ),
                "mix": (
                    "FLOAT",
                    {
                        "default": 0.65,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Dry/wet blend. 0=original only, 1=MusicGen only.",
                    },
                ),
            },
            "optional": {
                "prompt_override": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Leave empty for random LA style prompt each run.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**32 - 1,
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 5,
                        "max": 30,
                        "step": 1,
                        "tooltip": (
                            "Seconds of music to generate per pass (5-30s). "
                            "Output is looped/trimmed to match input clip length."
                        ),
                    },
                ),
                "weather_summary": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Weather summary string from WeatherFetch. "
                            "Drives data-aware style/mood/tempo selection."
                        ),
                    },
                ),
            },
        }

    def enhance(
        self,
        audio,
        strength: float = 0.20,
        mix: float = 0.65,
        prompt_override: str = "",
        seed: int = -1,
        num_inference_steps: int = 20,
        weather_summary: str = "",
    ) -> tuple:
        """Main enhancement function called by ComfyUI."""

        # -------------------------------------------------------------------
        # 1. Seed + prompt selection
        # -------------------------------------------------------------------
        if seed == -1:
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        rng = random.Random(actual_seed)

        if prompt_override and prompt_override.strip():
            prompt = prompt_override.strip()
            log.info("DMM_MusicEnhancer: Using custom prompt")
        else:
            prompt, selection_info = select_data_driven_prompt(rng, weather_summary)
            log.info("DMM_MusicEnhancer: Data-driven LA style | %s", selection_info)

        # Map strength (0.05-0.95) -> guidance_scale (1.5-9.55)
        # MusicGen default is 3.0; strength=0.20 -> ~3.3 (close to default)
        guidance_scale = 1.5 + strength * 8.5

        log.info("  Prompt: %s", prompt[:80])
        log.info(
            "  Guidance: %.1f (strength=%.2f) | Mix: %.2f | Seed: %d | Duration: %ds",
            guidance_scale, strength, mix, actual_seed, num_inference_steps,
        )

        # -------------------------------------------------------------------
        # 2. Unpack ComfyUI AUDIO dict
        # -------------------------------------------------------------------
        if isinstance(audio, dict):
            if "waveform" not in audio:
                log.warning("Audio dict missing 'waveform' key -- passing through unchanged")
                return (audio,)
            audio_tensor = audio["waveform"]
            sample_rate = audio.get("sample_rate", 48000)
        elif isinstance(audio, torch.Tensor):
            audio_tensor = audio
            sample_rate = 48000
        else:
            log.warning(
                "Unexpected audio type %s -- passing through unchanged",
                type(audio).__name__,
            )
            return (audio,)

        if audio_tensor is None or audio_tensor.numel() == 0:
            log.warning("Empty audio input -- passing through unchanged")
            return (audio,)

        original_tensor = audio_tensor.clone()

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, C, T)

        input_duration = audio_tensor.shape[-1] / sample_rate
        log.info(
            "  Input: %.1fs, %dch, %dHz, batch=%d",
            input_duration, audio_tensor.shape[-2], sample_rate, audio_tensor.shape[0],
        )

        # -------------------------------------------------------------------
        # 3. VRAM flush
        # -------------------------------------------------------------------
        log.info("  Flushing VRAM...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            log.info("  VRAM free after flush: %.0f MB", free_mb)

        # -------------------------------------------------------------------
        # 4. Load MusicGen-melody (cached after first call)
        # -------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor, model = _get_or_load_model(device)

        if model is None:
            log.warning(
                "MusicGen-melody not available -- passing through unchanged\n"
                "  Install: pip install transformers"
            )
            return (
                audio
                if isinstance(audio, dict)
                else {"waveform": original_tensor, "sample_rate": sample_rate},
            )

        # -------------------------------------------------------------------
        # 5. Generate per batch item
        # -------------------------------------------------------------------
        # num_inference_steps is repurposed as generation duration (seconds).
        # MusicGen outputs 50 tokens/sec at its 32kHz / 320-sample frame rate.
        duration_s = max(5, min(30, num_inference_steps))
        max_new_tokens = int(duration_s * 50)

        batch_size = audio_tensor.shape[0]
        enhanced_batches = []
        t0 = time.time()

        for b in range(batch_size):
            clip = audio_tensor[b]  # (C, T)
            orig_samples = clip.shape[-1]

            if batch_size > 1:
                log.info("  Processing batch item %d/%d", b + 1, batch_size)

            try:
                # Resample input to MusicGen's 32kHz for melody conditioning
                clip_32k = _resample_tensor(clip, sample_rate, MUSICGEN_SR)

                # Mono numpy for melody conditioning (processor expects 1-D)
                mono_32k = clip_32k.mean(dim=0).cpu().float().numpy()  # (T,)

                # Build processor inputs
                processor_out = processor(
                    text=[prompt],
                    audio=[mono_32k],
                    sampling_rate=MUSICGEN_SR,
                    padding=True,
                    return_tensors="pt",
                )
                # Move to device AND cast float tensors to model dtype
                # (processor outputs float32 but model may be float16 on CUDA)
                model_dtype = next(model.parameters()).dtype
                inputs = {}
                for k, v in processor_out.items():
                    if hasattr(v, "to"):
                        v = v.to(device)
                        if v.is_floating_point():
                            v = v.to(model_dtype)
                    inputs[k] = v

                # Generate
                with torch.no_grad():
                    audio_values = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                        do_sample=True,
                        temperature=0.95,  # R6: slight reduction prevents tail noise
                    )

                # audio_values: (1, 1, T) at 32kHz
                generated = audio_values[0, 0].float()  # (T,)

                # Resample back to original sample rate
                generated = generated.unsqueeze(0)  # (1, T)
                generated = _resample_tensor(generated, MUSICGEN_SR, sample_rate)
                # generated: (1, T)

                # Expand to match original channel count
                orig_channels = clip.shape[0]
                if orig_channels == 2:
                    generated = generated.repeat(2, 1)  # (2, T)
                # else stays (1, T)

                # Trim or crossfade-loop to match original length
                gen_samples = generated.shape[-1]
                if gen_samples >= orig_samples:
                    generated = generated[..., :orig_samples]
                else:
                    generated = crossfade_loop(
                        generated, orig_samples, sample_rate=sample_rate
                    )

                enhanced_batches.append(generated.unsqueeze(0))  # (1, C, T)

            except Exception as e:
                log.error(
                    "MusicGen generation failed on batch %d: %s -- using original", b, e
                )
                enhanced_batches.append(clip.unsqueeze(0))

        elapsed = time.time() - t0
        log.info("  MusicGen processing: %.1fs (%d batch items)", elapsed, batch_size)

        # -------------------------------------------------------------------
        # 6. Stack + dry/wet mix
        # -------------------------------------------------------------------
        enhanced_full = torch.cat(enhanced_batches, dim=0)  # (B, C, T)
        result_tensor = mix_audio(original_tensor, enhanced_full, mix)

        log.info(
            "  Output: %.1fs, batch=%d | Enhancement complete",
            result_tensor.shape[-1] / sample_rate,
            result_tensor.shape[0],
        )

        # -------------------------------------------------------------------
        # 7. Repack ComfyUI AUDIO dict
        # -------------------------------------------------------------------
        return ({"waveform": result_tensor, "sample_rate": sample_rate},)


# ---------------------------------------------------------------------------
# ComfyUI Node: DMM_MusicEnhancerBypass
# ---------------------------------------------------------------------------
class DMM_MusicEnhancerBypass:
    """Simple bypass/passthrough for A/B testing.

    Wire in place of DMM_MusicEnhancer to hear the original mix
    without enhancement.
    """

    CATEGORY = "DataMediaMachine"
    FUNCTION = "passthrough"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    def passthrough(self, audio) -> tuple:
        log.info("DMM_MusicEnhancerBypass: Passing audio through unchanged")
        return (audio,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "DMM_MusicEnhancer": DMM_MusicEnhancer,
    "DMM_MusicEnhancerBypass": DMM_MusicEnhancerBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DMM_MusicEnhancer": "DMM Music Enhancer (MusicGen)",
    "DMM_MusicEnhancerBypass": "DMM Music Enhancer Bypass",
}
