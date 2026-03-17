import os, time, subprocess, argparse, tempfile, shutil, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("DMM.Watchdog")

def get_duration(filepath):
    try:
        cmd = ['ffprobe.exe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
        out = subprocess.check_output(cmd, timeout=5, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return float(out) if out and out != 'N/A' else -1.0
    except: return -1.0

def run_watchdog(render_dir, playlist_path, max_clips, poll):
    log.info("="*60 + "\n   DMM Stream Watchdog (Liberal v2)\n   Monitoring: " + render_dir + "\n" + "="*60)
    while True:
        # RECURSIVE SEARCH: This finds clips in subfolders!
        files = []
        for root, _, filenames in os.walk(render_dir):
            for f in filenames:
                if f.endswith(('.mp4', '.webm')): files.append(os.path.join(root, f))
        
        valid_files = []
        for f in files:
            # Wait 2 seconds to ensure ComfyUI is done writing
            if (time.time() - os.path.getmtime(f)) > 2.0:
                duration = get_duration(f)
                # Accepts anything from 0.1s to 300s
                if 0.1 <= duration <= 300.0:
                    valid_files.append((os.path.getmtime(f), f, duration))

        valid_files.sort(key=lambda x: x[0], reverse=True)
        selected = valid_files[:max_clips]
        selected.reverse()

        if selected:
            temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(playlist_path), text=True)
            with os.fdopen(temp_fd, 'w') as f:
                f.write("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-PLAYLIST-TYPE:EVENT\n")
                for _, clip_path, duration in selected:
                    f.write(f"#EXTINF:{duration:.3f},\n{clip_path}\n")
            shutil.move(temp_path, playlist_path)
            log.info(f"Playlist updated: {len(selected)} clips loaded.")
        else:
            log.warning("Buffer remains at 0 — Check subfolders for valid .mp4 files!")
        time.sleep(poll)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--playlist", required=True)
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument("--poll", type=float, default=3.0)
    args = parser.parse_args()
    run_watchdog(args.render_dir, args.playlist, args.max_clips, args.poll)