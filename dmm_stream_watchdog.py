"""
DMM Stream Watchdog — Bridges ComfyUI renders to OBS Studio via dynamic M3U8 playlist.

Architecture:
  ComfyUI (Auto Queue) → /renders/*.mp4 → Watchdog detects → ffprobe validates →
  playlist.m3u8 updated → OBS VLC Source reads → YouTube Live stream.

Key design decisions:
  - ffprobe validation prevents half-written or corrupt clips from entering playlist
  - File stability check (size unchanged for 2s) ensures ComfyUI finished writing
  - Playlist capped at N most recent clips to prevent unbounded growth
  - Atomic playlist writes (write tmp → rename) prevent OBS from reading partial files
  - NVENC encoding in OBS uses dedicated hardware, leaving CUDA free for LTX-2

Usage:
  python dmm_stream_watchdog.py --render-dir C:\DMM_Stream\renders --playlist C:\DMM_Stream\live_playlist.m3u8

Author: Jeffrey A. Brick
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DMM.StreamWatchdog")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mkv"}
FILE_STABLE_WAIT = 2.0        # seconds to wait for file size to stabilize
FILE_STABLE_INTERVAL = 0.5    # poll interval during stability check
POLL_INTERVAL = 3.0           # seconds between directory scans
MIN_CLIP_DURATION = 1.0       # reject clips shorter than this (seconds)
MAX_CLIP_DURATION = 300.0     # reject clips longer than this (seconds)


# ---------------------------------------------------------------------------
# ffprobe validation
# ---------------------------------------------------------------------------
def ffprobe_validate(filepath: str) -> dict | None:
    """Validate a video file using ffprobe. Returns metadata dict or None if invalid.

    Checks:
      - ffprobe can read the file without error
      - Duration is within acceptable range
      - At least one video stream exists
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            log.warning("ffprobe failed for %s (rc=%d)", filepath, result.returncode)
            return None

        data = json.loads(result.stdout)

        # Check for video stream
        streams = data.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        if not has_video:
            log.warning("No video stream in %s", filepath)
            return None

        # Check duration
        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0))
        if duration < MIN_CLIP_DURATION:
            log.warning("Clip too short (%.1fs): %s", duration, filepath)
            return None
        if duration > MAX_CLIP_DURATION:
            log.warning("Clip too long (%.1fs): %s", duration, filepath)
            return None

        return {
            "path": filepath,
            "duration": duration,
            "size_mb": int(fmt.get("size", 0)) / (1024 * 1024),
            "streams": len(streams),
        }

    except FileNotFoundError:
        log.error("ffprobe not found! Install FFmpeg and ensure it's on PATH.")
        return None
    except subprocess.TimeoutExpired:
        log.warning("ffprobe timed out for %s", filepath)
        return None
    except Exception as e:
        log.warning("ffprobe error for %s: %s", filepath, e)
        return None


def wait_for_stable(filepath: str, timeout: float = FILE_STABLE_WAIT) -> bool:
    """Wait until file size stops changing (ComfyUI finished writing)."""
    prev_size = -1
    elapsed = 0.0
    while elapsed < timeout + FILE_STABLE_WAIT:
        try:
            current_size = os.path.getsize(filepath)
        except OSError:
            return False

        if current_size == prev_size and current_size > 0:
            return True

        prev_size = current_size
        time.sleep(FILE_STABLE_INTERVAL)
        elapsed += FILE_STABLE_INTERVAL

    # If we got here, size was still changing — give it one more check
    return prev_size == os.path.getsize(filepath)


# ---------------------------------------------------------------------------
# Playlist manager
# ---------------------------------------------------------------------------
class PlaylistManager:
    """Manages the M3U8 playlist that OBS reads.

    Keeps a rolling window of the most recent N validated clips.
    Writes are atomic (tmp file → rename) to prevent OBS from reading
    a half-written playlist.
    """

    def __init__(self, playlist_path: str, max_clips: int = 50):
        self.playlist_path = Path(playlist_path)
        self.max_clips = max_clips
        self.clips: deque[str] = deque(maxlen=max_clips)
        self._load_existing()

    def _load_existing(self):
        """Load existing playlist if present (survive restarts)."""
        if self.playlist_path.exists():
            try:
                lines = self.playlist_path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if os.path.exists(line):
                            self.clips.append(line)
                log.info("Loaded %d existing clips from playlist", len(self.clips))
            except Exception as e:
                log.warning("Could not load existing playlist: %s", e)

    def add_clip(self, filepath: str) -> bool:
        """Add a validated clip to the playlist. Returns True if added."""
        abs_path = str(Path(filepath).resolve())

        # Deduplicate
        if abs_path in self.clips:
            log.debug("Clip already in playlist: %s", abs_path)
            return False

        self.clips.append(abs_path)
        self._write_playlist()
        log.info("Added to playlist (%d/%d): %s",
                 len(self.clips), self.max_clips, Path(abs_path).name)
        return True

    def _write_playlist(self):
        """Atomic write: tmp file → rename to prevent OBS reading partial data."""
        playlist_dir = self.playlist_path.parent
        playlist_dir.mkdir(parents=True, exist_ok=True)

        # Build M3U8 content
        lines = ["#EXTM3U"]
        for clip in self.clips:
            lines.append(clip)
        content = "\n".join(lines) + "\n"

        # Write to temp, then atomic rename
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(playlist_dir), suffix=".m3u8.tmp", prefix="dmm_"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            # On Windows, os.rename fails if target exists — use replace
            shutil.move(tmp_path, str(self.playlist_path))
        except Exception as e:
            log.error("Failed to write playlist: %s", e)
            # Clean up temp file if it still exists
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @property
    def clip_count(self) -> int:
        return len(self.clips)


# ---------------------------------------------------------------------------
# Directory watcher (polling-based for cross-platform reliability)
# ---------------------------------------------------------------------------
class RenderWatcher:
    """Polls a directory for new video files, validates them, and feeds
    them to the PlaylistManager.

    Uses polling instead of filesystem events (watchdog library) because:
      - Cross-platform (Windows/Linux/macOS) without extra deps
      - More reliable with network drives and some Windows configurations
      - ComfyUI writes can trigger multiple events; polling is simpler
    """

    def __init__(self, render_dir: str, playlist: PlaylistManager):
        self.render_dir = Path(render_dir)
        self.playlist = playlist
        self.known_files: set[str] = set()
        self._running = True

        # Seed known_files with what's already there (don't re-process on restart)
        self._seed_known_files()

    def _seed_known_files(self):
        """Mark existing files as known so we don't re-validate the entire
        back catalog on every restart."""
        if self.render_dir.exists():
            for f in self.render_dir.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.known_files.add(str(f.resolve()))
            log.info("Seeded %d known files from render directory", len(self.known_files))

    def scan_once(self) -> int:
        """Scan for new files, validate, and add to playlist. Returns count of new clips added."""
        if not self.render_dir.exists():
            return 0

        added = 0
        for f in sorted(self.render_dir.iterdir(), key=lambda p: p.stat().st_mtime):
            if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            abs_path = str(f.resolve())
            if abs_path in self.known_files:
                continue

            # New file detected
            log.info("New render detected: %s", f.name)

            # Wait for file to stabilize (ComfyUI still writing)
            if not wait_for_stable(abs_path):
                log.warning("File never stabilized, skipping: %s", f.name)
                continue

            # Validate with ffprobe
            meta = ffprobe_validate(abs_path)
            if meta is None:
                log.warning("Validation failed, skipping: %s", f.name)
                self.known_files.add(abs_path)  # Don't retry bad files
                continue

            # Add to playlist
            if self.playlist.add_clip(abs_path):
                added += 1

            self.known_files.add(abs_path)
            log.info("Validated: %s (%.1fs, %.1f MB)",
                     f.name, meta["duration"], meta["size_mb"])

        return added

    def run(self, poll_interval: float = POLL_INTERVAL):
        """Main polling loop. Runs until interrupted."""
        log.info("Watching: %s", self.render_dir)
        log.info("Playlist: %s", self.playlist.playlist_path)
        log.info("Poll interval: %.1fs", poll_interval)
        log.info("Press Ctrl+C to stop.\n")

        while self._running:
            try:
                self.scan_once()
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                log.info("Shutdown requested.")
                self._running = False
            except Exception as e:
                log.error("Scan error: %s", e)
                time.sleep(poll_interval)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------
def print_status(playlist: PlaylistManager, render_dir: Path):
    """Print current buffer status — useful for pre-roll check."""
    count = playlist.clip_count
    status = "READY" if count >= 20 else "BUFFERING"
    bar = "#" * min(count, 50) + "." * max(0, 50 - count)
    log.info("[%s] Buffer: [%s] %d clips (recommend 20+ before going live)",
             status, bar, count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DMM Stream Watchdog — Bridges ComfyUI renders to OBS via M3U8 playlist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dmm_stream_watchdog.py --render-dir C:\\DMM_Stream\\renders --playlist C:\\DMM_Stream\\live_playlist.m3u8
  python dmm_stream_watchdog.py --render-dir ./renders --max-clips 100 --poll 5
        """,
    )

    parser.add_argument(
        "--render-dir", "-r",
        required=True,
        help="Directory where ComfyUI saves rendered video clips",
    )
    parser.add_argument(
        "--playlist", "-p",
        default=None,
        help="Path to output M3U8 playlist file (default: <render-dir>/live_playlist.m3u8)",
    )
    parser.add_argument(
        "--max-clips", "-m",
        type=int,
        default=50,
        help="Maximum clips in playlist rotation (default: 50)",
    )
    parser.add_argument(
        "--poll", "-t",
        type=float,
        default=POLL_INTERVAL,
        help="Seconds between directory scans (default: %.1f)" % POLL_INTERVAL,
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print buffer status and exit (don't start watching)",
    )

    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    if not render_dir.exists():
        render_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created render directory: %s", render_dir)

    playlist_path = args.playlist or str(render_dir / "live_playlist.m3u8")

    playlist = PlaylistManager(playlist_path, max_clips=args.max_clips)

    if args.status:
        print_status(playlist, render_dir)
        return

    # Print startup banner
    log.info("=" * 60)
    log.info("  DMM Stream Watchdog")
    log.info("  Render dir : %s", render_dir.resolve())
    log.info("  Playlist   : %s", Path(playlist_path).resolve())
    log.info("  Max clips  : %d", args.max_clips)
    log.info("  Poll rate  : %.1fs", args.poll)
    log.info("  Buffer     : %d clips loaded", playlist.clip_count)
    log.info("=" * 60)

    if playlist.clip_count < 20:
        log.warning("Buffer has %d clips — recommend 20+ before starting OBS stream!",
                    playlist.clip_count)

    watcher = RenderWatcher(str(render_dir), playlist)
    watcher.run(poll_interval=args.poll)

    log.info("Watchdog stopped. Final playlist has %d clips.", playlist.clip_count)


if __name__ == "__main__":
    main()
