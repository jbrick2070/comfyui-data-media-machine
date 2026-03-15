"""
DMM Camera Health Check — Validates Caltrans CCTV URLs before ComfyUI boots.

Pings every URL in camera_registry.json, checks for:
  - HTTP 200 response
  - Content-Type is image/*
  - Response body is a valid JPEG (starts with FFD8)
  - Response time under threshold (dead cameras often hang)

Outputs:
  - Console report with status per camera
  - Optional: writes a filtered camera_registry_live.json with only working URLs
  - Optional: can be scheduled via cron / Task Scheduler to run daily

Usage:
  python dmm_camera_healthcheck.py
  python dmm_camera_healthcheck.py --registry camera_registry.json --output camera_registry_live.json
  python dmm_camera_healthcheck.py --timeout 8 --verbose

Author: Jeffrey A. Brick
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DMM.CameraHealth")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 6          # seconds per request
MAX_WORKERS = 8              # parallel checks
JPEG_MAGIC = b"\xff\xd8"    # first 2 bytes of a valid JPEG
MIN_JPEG_SIZE = 2000         # bytes — anything smaller is probably an error page


# ---------------------------------------------------------------------------
# Single URL check
# ---------------------------------------------------------------------------
def check_url(url: str, label: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Check a single camera URL. Returns a result dict."""
    import requests

    result = {
        "url": url,
        "label": label,
        "alive": False,
        "status_code": None,
        "content_type": None,
        "size_bytes": 0,
        "valid_jpeg": False,
        "response_ms": 0,
        "error": None,
    }

    start = time.time()
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "DMM-CameraHealthCheck/1.0"
        })
        elapsed_ms = int((time.time() - start) * 1000)
        result["response_ms"] = elapsed_ms
        result["status_code"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "")
        result["size_bytes"] = len(resp.content)

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        # Check content type
        ct = result["content_type"].lower()
        if "image" not in ct and "jpeg" not in ct:
            result["error"] = f"Bad content-type: {ct}"
            return result

        # Check JPEG magic bytes
        if len(resp.content) < 2 or resp.content[:2] != JPEG_MAGIC:
            result["error"] = "Not a valid JPEG (bad magic bytes)"
            return result

        # Check minimum size (tiny responses are usually error pages)
        if result["size_bytes"] < MIN_JPEG_SIZE:
            result["error"] = f"Suspiciously small ({result['size_bytes']} bytes)"
            return result

        # All checks passed
        result["alive"] = True
        result["valid_jpeg"] = True

    except requests.exceptions.Timeout:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["error"] = f"Timeout ({timeout}s)"
    except requests.exceptions.ConnectionError as e:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["error"] = f"Connection error: {e}"
    except Exception as e:
        result["response_ms"] = int((time.time() - start) * 1000)
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Full registry check
# ---------------------------------------------------------------------------
def check_registry(registry: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Check all URLs in the registry in parallel. Returns results by focus type."""
    all_results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for focus, cameras in registry.items():
            if focus.startswith("_"):
                continue  # skip metadata fields
            all_results[focus] = []
            for cam in cameras:
                url = cam.get("url", "")
                label = cam.get("label", "unknown")
                if not url:
                    continue
                future = pool.submit(check_url, url, label, timeout)
                futures[future] = (focus, cam)

        for future in as_completed(futures):
            focus, cam = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "url": cam.get("url", ""),
                    "label": cam.get("label", ""),
                    "alive": False,
                    "error": str(e),
                }
            all_results[focus].append(result)

    # Sort each focus group by original priority
    for focus in all_results:
        all_results[focus].sort(key=lambda r: r["url"])

    return all_results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(results: dict):
    """Print a human-readable health report."""
    total = 0
    alive = 0

    print()
    print("=" * 72)
    print("  DMM Camera Health Report")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    for focus, checks in sorted(results.items()):
        print(f"\n  [{focus.upper()}]")
        for r in checks:
            total += 1
            status = "LIVE" if r["alive"] else "DEAD"
            icon = "+" if r["alive"] else "X"
            if r["alive"]:
                alive += 1
                print(f"    [{icon}] {status}  {r['label']:40s}  {r['response_ms']:4d}ms  "
                      f"{r['size_bytes']//1024:3d}KB  {r['url'][:60]}")
            else:
                print(f"    [{icon}] {status}  {r['label']:40s}  err={r.get('error','?')[:40]}")

    print()
    print("-" * 72)
    dead = total - alive
    pct = (alive / total * 100) if total > 0 else 0
    print(f"  TOTAL: {alive}/{total} cameras alive ({pct:.0f}%)")
    if dead > 0:
        print(f"  WARNING: {dead} dead camera(s) — pipeline will fallback to t2v for those segments")
    else:
        print(f"  All cameras operational — full i2v pipeline available")
    print("=" * 72)
    print()

    return alive, total


# ---------------------------------------------------------------------------
# Filter and write live-only registry
# ---------------------------------------------------------------------------
def write_live_registry(original: dict, results: dict, output_path: str):
    """Write a filtered registry containing only live cameras."""
    live_urls = set()
    for checks in results.values():
        for r in checks:
            if r["alive"]:
                live_urls.add(r["url"])

    filtered = {}
    for focus, cameras in original.items():
        if focus.startswith("_"):
            filtered[focus] = cameras
            continue
        live_cams = [c for c in cameras if c.get("url", "") in live_urls]
        if live_cams:
            filtered[focus] = live_cams
        else:
            # Keep at least the first camera even if dead (so the node doesn't crash)
            filtered[focus] = cameras[:1] if cameras else []
            log.warning("Focus '%s': ALL cameras dead — keeping first as fallback", focus)

    filtered["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    filtered["_comment"] = "Auto-generated by dmm_camera_healthcheck.py — live cameras only"

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=4)

    log.info("Wrote live registry to %s", output_path)
    return filtered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DMM Camera Health Check — Validate Caltrans CCTV URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry", "-r",
        default="camera_registry.json",
        help="Path to camera_registry.json (default: camera_registry.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Write filtered live-only registry to this path (optional)",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-URL timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for scripting)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load registry
    reg_path = Path(args.registry)
    if not reg_path.exists():
        # Try relative to script location
        alt = Path(__file__).parent / args.registry
        if alt.exists():
            reg_path = alt
        else:
            log.error("Registry not found: %s", args.registry)
            sys.exit(1)

    log.info("Loading registry: %s", reg_path)
    with open(reg_path) as f:
        registry = json.load(f)

    # Count cameras
    cam_count = sum(
        len(cams) for key, cams in registry.items()
        if not key.startswith("_") and isinstance(cams, list)
    )
    log.info("Checking %d cameras across %d focus types (timeout=%ds)...",
             cam_count,
             sum(1 for k in registry if not k.startswith("_") and isinstance(registry[k], list)),
             args.timeout)

    # Run checks
    results = check_registry(registry, timeout=args.timeout)

    if args.json:
        # JSON output mode
        flat = []
        for focus, checks in results.items():
            for r in checks:
                r["focus"] = focus
                flat.append(r)
        print(json.dumps(flat, indent=2))
    else:
        alive, total = print_report(results)

    # Write filtered registry if requested
    if args.output:
        write_live_registry(registry, results, args.output)

    # Exit code: 0 if all alive, 1 if any dead
    all_alive = all(r["alive"] for checks in results.values() for r in checks)
    sys.exit(0 if all_alive else 1)


if __name__ == "__main__":
    main()
