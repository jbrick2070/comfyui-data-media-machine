# Custom Notes — Production TTS Configuration

This folder contains production configuration for the Data Media Machine
TTS narration pipeline. These are the "nice statements" and data-driven
commentary that make the broadcast feel human and alive.

## What's here

- `tts_statements.json` — The master list of data-based friendly statements
  organized by trigger condition (temperature, UV, rain, wind, humidity, AQI,
  sunrise/sunset proximity). Used by `dmm_sun_utils.get_nice_statement()`.
- `README.md` — This file.

## How it works

The TTS pipeline now includes:
1. **Date** — Full date spoken in every narration opening
2. **Sunrise / Sunset** — Calculated via NOAA solar equations for LA coords
3. **Nice statements** — Friendly, data-driven one-liners based on current
   weather, AQI, UV, and sun position. Changes every minute.

## Examples of nice statements

- "You better bring sunscreen — UV is high today!"
- "Grab a jacket, it's cooler than you think out there."
- "Beautiful evening ahead — sunset's at 7:12 PM."
- "Don't forget your umbrella today!"
- "The kind of day that makes you glad you're in LA."
- "Stay hydrated out there — it's scorching today."
- "Layer up before you head out — it's brisk."

## Where these show up

- **TTS narration** (all 8 narrator styles in DMMDataToTTS)
- **LA Pulse broadcast** (all 6 broadcast styles, opening + closing)
- **Narration Distiller** (appended to compressed broadcast)
- **Procedural Clip HUD** (all 3 visual styles: la_neon, minimal_data, retro_terminal)
  - la_neon: glowing statement bar + sun info ribbon + updated ticker
  - minimal_data: date/sun row + statement row in data display
  - retro_terminal: DATE/SOLAR lines + statement as status line
