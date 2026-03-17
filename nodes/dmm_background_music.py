"""
DMM_BackgroundMusic — Synthesizes a MIDI file into an AUDIO tensor for
use as background music in the DMM pipeline.

Uses additive synthesis (no FluidSynth required) with instrument-specific
harmonic profiles for trumpet, piano, strings, etc.

v1.0  2026-03-14  Initial release — fills the 18.7s silence gap.
"""

import logging
import os
import math
import torch
import numpy as np

log = logging.getLogger("DMM")


# ─── instrument synth profiles ───────────────────────────────────────────────

def _midi_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _adsr(n_samples, sr, attack=0.02, decay=0.05, sustain=0.7, release=0.05):
    """ADSR envelope — safe for very short notes."""
    a = min(int(attack * sr), n_samples)
    d = min(int(decay * sr), max(0, n_samples - a))
    r = min(int(release * sr), max(0, n_samples - a - d))
    s = max(0, n_samples - a - d - r)

    parts = []
    if a > 0:
        parts.append(np.linspace(0, 1, a))
    if d > 0:
        parts.append(np.linspace(1, sustain, d))
    if s > 0:
        parts.append(np.full(s, sustain))
    if r > 0:
        parts.append(np.linspace(sustain, 0, r))

    if not parts:
        return np.zeros(n_samples)
    env = np.concatenate(parts)
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]


def _synth_brass(freq, duration, sr, velocity=0.8):
    """Brass-like: strong odd harmonics, punchy attack."""
    n = int(duration * sr)
    if n < 1:
        return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * freq * t) * 1.0
           + np.sin(2 * np.pi * freq * 2 * t) * 0.6
           + np.sin(2 * np.pi * freq * 3 * t) * 0.4
           + np.sin(2 * np.pi * freq * 4 * t) * 0.2
           + np.sin(2 * np.pi * freq * 5 * t) * 0.15
           + np.sin(2 * np.pi * freq * 6 * t) * 0.08)
    env = _adsr(n, sr, attack=0.03, decay=0.08, sustain=0.75, release=0.08)
    return sig * env * velocity


def _synth_piano(freq, duration, sr, velocity=0.8):
    """Piano-like: all harmonics, fast initial decay."""
    n = int(duration * sr)
    if n < 1:
        return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * freq * t) * 1.0
           + np.sin(2 * np.pi * freq * 2 * t) * 0.5
           + np.sin(2 * np.pi * freq * 3 * t) * 0.25
           + np.sin(2 * np.pi * freq * 4 * t) * 0.12
           + np.sin(2 * np.pi * freq * 5 * t) * 0.06)
    decay_time = min(duration * 0.8, 2.0)
    env = _adsr(n, sr, attack=0.005, decay=decay_time, sustain=0.3, release=0.1)
    return sig * env * velocity


def _synth_strings(freq, duration, sr, velocity=0.8):
    """Strings-like: warm, slow attack, sustained."""
    n = int(duration * sr)
    if n < 1:
        return np.zeros(1)
    t = np.arange(n) / sr
    # Slight vibrato
    vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 5.5 * t)
    sig = (np.sin(2 * np.pi * freq * vibrato * t) * 1.0
           + np.sin(2 * np.pi * freq * 2 * vibrato * t) * 0.4
           + np.sin(2 * np.pi * freq * 3 * vibrato * t) * 0.2
           + np.sin(2 * np.pi * freq * 4 * vibrato * t) * 0.1)
    env = _adsr(n, sr, attack=0.12, decay=0.1, sustain=0.85, release=0.15)
    return sig * env * velocity


def _synth_generic(freq, duration, sr, velocity=0.8):
    """Simple sine + harmonics for unknown instruments."""
    n = int(duration * sr)
    if n < 1:
        return np.zeros(1)
    t = np.arange(n) / sr
    sig = (np.sin(2 * np.pi * freq * t) * 1.0
           + np.sin(2 * np.pi * freq * 2 * t) * 0.3
           + np.sin(2 * np.pi * freq * 3 * t) * 0.1)
    env = _adsr(n, sr, attack=0.01, decay=0.1, sustain=0.6, release=0.05)
    return sig * env * velocity


# GM program → synth function mapping (rough categories)
_PROGRAM_MAP = {}
for p in range(0, 8):       # Piano family
    _PROGRAM_MAP[p] = _synth_piano
for p in range(24, 32):     # Guitar family
    _PROGRAM_MAP[p] = _synth_piano
for p in range(40, 48):     # Strings
    _PROGRAM_MAP[p] = _synth_strings
for p in range(48, 56):     # Ensemble
    _PROGRAM_MAP[p] = _synth_strings
for p in range(56, 64):     # Brass
    _PROGRAM_MAP[p] = _synth_brass
for p in range(64, 72):     # Reed
    _PROGRAM_MAP[p] = _synth_brass
for p in range(72, 80):     # Pipe
    _PROGRAM_MAP[p] = _synth_brass


def _get_synth_fn(program, name=""):
    """Pick synth function from GM program number or instrument name."""
    name_lower = name.lower()
    if "trumpet" in name_lower or "brass" in name_lower or "horn" in name_lower:
        return _synth_brass
    if "piano" in name_lower or "key" in name_lower:
        return _synth_piano
    if "string" in name_lower or "violin" in name_lower or "cello" in name_lower:
        return _synth_strings
    return _PROGRAM_MAP.get(program, _synth_generic)


# ─── MIDI → audio rendering ─────────────────────────────────────────────────

def render_midi_to_audio(midi_path, sr=48000, target_duration=None,
                         loop=True, fade_in=0.5, fade_out=1.0):
    """
    Render a MIDI file to a numpy audio array using additive synthesis.

    Returns: (audio_np, actual_duration)
        audio_np: shape (samples,) float64, normalized to [-0.85, 0.85]
    """
    try:
        import pretty_midi
    except ImportError:
        try:
            import mido
            return _render_with_mido(midi_path, sr, target_duration, loop,
                                     fade_in, fade_out)
        except ImportError:
            raise ImportError("Need either 'pretty_midi' or 'mido' for MIDI. "
                              "pip install pretty_midi")

    mid = pretty_midi.PrettyMIDI(midi_path)
    midi_duration = mid.get_end_time()

    if target_duration is None:
        target_duration = midi_duration

    total_samples = int(target_duration * sr)
    audio = np.zeros(total_samples, dtype=np.float64)

    note_count = 0
    for inst in mid.instruments:
        if inst.is_drum:
            continue
        synth_fn = _get_synth_fn(inst.program, inst.name)
        for note in inst.notes:
            # Handle looping
            note_start = note.start
            note_end = note.end

            if loop and midi_duration > 0:
                # Render all loop iterations that fit
                loops = int(target_duration / midi_duration) + 1
                for loop_i in range(loops):
                    offset = loop_i * midi_duration
                    s = note_start + offset
                    e = note_end + offset
                    if s >= target_duration:
                        break
                    dur = min(e - s, target_duration - s)
                    if dur < 0.005:
                        continue
                    freq = _midi_to_freq(note.pitch)
                    vel = note.velocity / 127.0
                    rendered = synth_fn(freq, dur, sr, vel)
                    start_idx = int(s * sr)
                    end_idx = start_idx + len(rendered)
                    if end_idx > total_samples:
                        rendered = rendered[:total_samples - start_idx]
                        end_idx = total_samples
                    audio[start_idx:end_idx] += rendered
                    note_count += 1
            else:
                if note_start >= target_duration:
                    continue
                dur = min(note_end - note_start, target_duration - note_start)
                if dur < 0.005:
                    continue
                freq = _midi_to_freq(note.pitch)
                vel = note.velocity / 127.0
                rendered = synth_fn(freq, dur, sr, vel)
                start_idx = int(note_start * sr)
                end_idx = start_idx + len(rendered)
                if end_idx > total_samples:
                    rendered = rendered[:total_samples - start_idx]
                    end_idx = total_samples
                audio[start_idx:end_idx] += rendered
                note_count += 1

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    # Apply fade-in / fade-out
    if fade_in > 0:
        fi_samples = min(int(fade_in * sr), total_samples)
        audio[:fi_samples] *= np.linspace(0, 1, fi_samples)
    if fade_out > 0:
        fo_samples = min(int(fade_out * sr), total_samples)
        audio[-fo_samples:] *= np.linspace(1, 0, fo_samples)

    log.info("[BackgroundMusic] rendered %d notes, %.1fs, peak=%.3f",
             note_count, target_duration, peak)
    return audio, target_duration


def _render_with_mido(midi_path, sr, target_duration, loop, fade_in, fade_out):
    """Fallback renderer using mido (no pretty_midi)."""
    import mido
    mid = mido.MidiFile(midi_path)
    midi_duration = mid.length

    if target_duration is None:
        target_duration = midi_duration

    total_samples = int(target_duration * sr)
    audio = np.zeros(total_samples, dtype=np.float64)

    note_count = 0
    for track in mid.tracks:
        abs_time = 0
        tempo = 500000  # default 120 BPM
        active_notes = {}

        # Detect instrument from program_change
        synth_fn = _synth_generic
        for msg in track:
            if msg.type == 'program_change':
                synth_fn = _PROGRAM_MAP.get(msg.program, _synth_generic)
                break

        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (abs_time, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start, vel = active_notes.pop(msg.note)
                    freq = _midi_to_freq(msg.note)
                    dur = abs_time - start
                    if dur < 0.005 or start >= target_duration:
                        continue
                    dur = min(dur, target_duration - start)
                    rendered = synth_fn(freq, dur, sr, vel / 127.0)
                    si = int(start * sr)
                    ei = si + len(rendered)
                    if ei > total_samples:
                        rendered = rendered[:total_samples - si]
                        ei = total_samples
                    audio[si:ei] += rendered
                    note_count += 1

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85
    if fade_in > 0:
        fi = min(int(fade_in * sr), total_samples)
        audio[:fi] *= np.linspace(0, 1, fi)
    if fade_out > 0:
        fo = min(int(fade_out * sr), total_samples)
        audio[-fo:] *= np.linspace(1, 0, fo)

    log.info("[BackgroundMusic] mido fallback: %d notes rendered", note_count)
    return audio, target_duration


# ─── ComfyUI Node ────────────────────────────────────────────────────────────

class DMMBackgroundMusic:
    """Synthesizes a MIDI file into an AUDIO tensor for background music."""

    CATEGORY = "DataMediaMachine"
    FUNCTION = "render"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("music",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_file": ("STRING", {
                    "default": "seventy_six_cities.mid",
                    "tooltip": "Path to MIDI file (relative to ComfyUI/input or absolute)"
                }),
            },
            "optional": {
                "duration_sec": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 1.0,
                    "tooltip": "Target duration in seconds (0 = use full MIDI length)"
                }),
                "volume": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Music volume (0.3 = subtle background, 1.0 = full volume)"
                }),
                "loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Loop MIDI to fill target duration"
                }),
                "fade_in_sec": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Fade-in duration in seconds"
                }),
                "fade_out_sec": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Fade-out duration in seconds"
                }),
                "sample_rate": ("INT", {
                    "default": 48000, "min": 8000, "max": 96000, "step": 1000,
                    "tooltip": "Output sample rate (48000 to match LTX-2 pipeline)"
                }),
            },
        }

    def render(self, midi_file, duration_sec=0.0, volume=0.3, loop=True,
               fade_in_sec=0.5, fade_out_sec=2.0, sample_rate=48000):

        # Resolve MIDI path
        midi_path = midi_file
        if not os.path.isabs(midi_path):
            # Search order: package dir → ComfyUI input → CWD
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            comfy_root = os.path.dirname(pkg_dir)
            for search_dir in [
                pkg_dir,  # media_machine/ folder (ship MIDI with the pack)
                os.path.join(comfy_root, "input"),
                os.path.join(comfy_root, "..", "input"),
                os.getcwd(),
            ]:
                candidate = os.path.join(search_dir, midi_path)
                if os.path.isfile(candidate):
                    midi_path = candidate
                    break

        if not os.path.isfile(midi_path):
            log.error("[BackgroundMusic] MIDI file not found: %s", midi_path)
            # Return silence
            n = int((duration_sec or 10.0) * sample_rate)
            silence = torch.zeros(1, 2, n)
            return ({"waveform": silence, "sample_rate": sample_rate},)

        target = duration_sec if duration_sec > 0 else None
        audio_np, actual_dur = render_midi_to_audio(
            midi_path, sr=sample_rate, target_duration=target,
            loop=loop, fade_in=fade_in_sec, fade_out=fade_out_sec
        )

        # Apply volume
        audio_np = audio_np * volume

        # Convert to torch tensor: (1, 2, samples) stereo
        audio_t = torch.from_numpy(audio_np).float()
        # Stack stereo with slight Haas delay for width
        delay_samples = int(0.0004 * sample_rate)  # 0.4ms
        left = audio_t
        right = torch.cat([torch.zeros(delay_samples), audio_t[:-delay_samples]])
        stereo = torch.stack([left, right]).unsqueeze(0)  # (1, 2, samples)

        log.info("[BackgroundMusic] output: %.1fs, %dHz, stereo, vol=%.0f%%",
                 actual_dur, sample_rate, volume * 100)

        return ({"waveform": stereo, "sample_rate": sample_rate},)
