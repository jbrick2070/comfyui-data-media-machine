# DMM v3.5-beta QA Brief — For External Review

**Date:** 2026-03-17
**Author:** Jeffrey A. Brick
**Reviewer:** Please audit for correctness, edge cases, and integration risks.

---

## What Changed in v3.5-beta (This Session)

### New Node
- **DMMEnergyGrid** (`dmm_energy_grid.py`) — California ISO real-time grid demand
  - Endpoint: `https://oasis.caiso.com/oasisapi/SingleZip` (no key, returns ZIP/CSV)
  - Fetches Day-Ahead Market system load forecast for LADWP + SCE-TAC
  - Computes `grid_stress` (0.0-1.0) from current hour vs daily peak/trough
  - Returns `DMM_ENERGY` type: `current_mw`, `peak_mw`, `trough_mw`, `grid_mood`, `hourly_mw`
  - Demo modes: `demo_peak`, `demo_overnight`, `demo_blackout`
  - Registered in `__init__.py` as `DMM_EnergyGrid`

### Upgraded Nodes

**DMMAlertsFetch** (`dmm_alerts_fetch.py`)
- Added CAL FIRE wildfire incident feed: `https://incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList?inactive=false`
- No API key. GeoJSON format. Filters by 200-mile haversine radius from config lat/lon.
- Fire severity mapped from acres burned + containment %
- `nws_live` source now auto-merges NWS weather alerts + CAL FIRE incidents
- New source: `calfire_only` for fire-specific queries
- New output fields: `fire_count`, `nearest_fire` (name, distance, acres, containment)
- Added `IS_CHANGED` returning `float("nan")` (was missing — would have caused stale cache)

**DMMEarthquakeFetch** (`dmm_earthquake_fetch.py`)
- Added derived seismic fields from existing USGS data (zero new API calls):
  - `seismic_intensity` (0.0-1.0 composite: 30% count + 50% max_mag + 20% felt)
  - `felt_quake_count`, `total_felt_reports`
  - `avg_depth_km`, `max_significance`, `tsunami_flags`, `strongest_place`
- Added `IS_CHANGED` returning `float("nan")` (was missing)

**DMMAirQualityFetch** (`dmm_airquality_fetch.py`)
- Added AirNow EPA sensor data as Tier 2 source (free API key stored in `config.json`)
- Endpoint: `https://www.airnowapi.org/aq/observation/latLong/current/`
- Key auto-loaded from `custom_nodes/media_machine/config.json`
- Falls back to Open-Meteo if key missing or AirNow fails
- New output fields when using AirNow: `data_type: "sensor"`, `reporting_area`
- Added `IS_CHANGED` returning `float("nan")` (was missing)
- **Sticky widget fix:** `airnow_sensor` appended at END of combo list to avoid positional desync

**DMMWebcamFetch** (`dmm_webcam_fetch.py`)
- Luminance gate: skip SSIM when mean brightness < 0.12 (nighttime false-positive fix)
- MIME type validation: reject non-image `Content-Type` responses
- Filesize floor: reject responses < 2KB
- Per-domain rate limiter: 2s min between requests to same netloc, thread-safe
- Dark-camera cache: hard cap at 500 entries with LRU eviction
- Domain throttle: prune entries > 24h old when dict exceeds 50 entries

**DMMCameraRouter** (`dmm_camera_router.py`)
- YouTube cache-buster: `?t={unix_timestamp}` on `img.youtube.com` URLs
- `import time` moved to module scope (was re-importing per call)
- Disabled camera filter: `cameras = [c for c in cameras if not c.get("disabled", False)]`

**camera_registry.json**
- Added `_meta` block: version, generated_at, center_lat/lon, source_counts
- New cameras: Mt. Wilson HPWREN (1), Venice Beach YouTube (1), LAX YouTube (1)
- Total: 291 category slots, ~208 unique URLs
- Sources: Caltrans D7 (275), Weingart/ipcamlive (3), ABC7 KABC (4), YouTube (5), HPWREN (1)

### Infrastructure
- `config.json` created for API keys (gitignored, never committed)
- `.gitignore` updated to exclude `config.json`

---

## Data Sources Summary (All Active)

| Data Type | Source | Endpoint | Key? | Live? |
|-----------|--------|----------|------|-------|
| Weather | Open-Meteo | `api.open-meteo.com/v1/forecast` | No | Yes |
| Weather | NWS (fallback) | `api.weather.gov/points` → `/observations/latest` | No | Yes |
| Air Quality | Open-Meteo (Tier 1) | `air-quality-api.open-meteo.com/v1/air-quality` | No | Yes |
| Air Quality | AirNow (Tier 2) | `airnowapi.org/aq/observation/latLong/current/` | Free key | Yes |
| Earthquakes | USGS | `earthquake.usgs.gov/fdsnws/event/1/query` | No | Yes |
| Alerts | NWS | `api.weather.gov/alerts/active` | No | Yes |
| Alerts | CAL FIRE | `incidents.fire.ca.gov/umbraco/api/IncidentApi/GeoJsonList` | No | Yes |
| Transit | Synthetic model | N/A (20 hardcoded LA Metro routes, time-aware) | N/A | Synthetic |
| Energy Grid | CAISO | `oasis.caiso.com/oasisapi/SingleZip` | No | Yes |
| Webcams | Multiple | Caltrans, ipcamlive, ABC7, YouTube, HPWREN | No | Yes |
| Satellite | NASA GIBS | `wvs.earthdata.nasa.gov/api/v1/snapshot` | No | Yes (1-day delay) |

---

## Known Limitations

1. **Transit is synthetic.** LA Metro GTFS-RT (`api.gtfsrt.metro.net`) is DNS-dead. Swiftly API requires key. No free keyless path to real-time transit.
2. **AirNow requires manual key signup.** User must register at `docs.airnowapi.org` and paste key into `config.json`.
3. **CAL FIRE returns empty when no active fires.** This is correct behavior, not a bug.
4. **CAISO returns Day-Ahead Market forecasts**, not real-time telemetry. Data is hourly, updated daily.
5. **YouTube thumbnail cache:** Google CDNs may still serve stale frames despite `?t=` cache-buster (30s-2min lag).

---

## QA Checklist (Please Verify)

1. Does `DMMEnergyGrid._fetch_caiso()` correctly handle ZIP extraction edge cases (empty ZIP, corrupt CSV, missing LADWP/SCE rows)?
2. Does `DMMAlertsFetch._fetch_calfire()` haversine distance calculation look correct? (We filter to 200mi radius.)
3. Is the `seismic_intensity` formula reasonable? (`count/20 * 0.3 + max_mag/7 * 0.5 + felt/1000 * 0.2`)
4. For AirNow: is taking `max(O3_AQI, PM25_AQI, PM10_AQI)` as the overall AQI correct per EPA methodology?
5. Are there any race conditions in `_domain_throttle()` given the `time.sleep()` inside a `threading.Lock`?
6. Does the `_dark_camera_cache` hard cap (500) with sorted eviction have performance concerns for the sort step?
7. Sticky widget: is `airnow_sensor` at end of combo list safe for all existing workflow JSONs?
8. Are there edge cases in the CAISO UTC→PST hour conversion (`(now.hour - 7) % 24`)?
