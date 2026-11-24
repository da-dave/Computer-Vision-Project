"""Measure what the live camera stream actually delivers: frames per second, frame size and bandwidth.

  ./run.sh stream-fps                                        (the camera server on this computer)
  ./run.sh stream-fps --url https://<laptop-ip>:8443/camera/stream --insecure      (as the Quest receives it)

The result is saved to results/stream.json, and the evaluation report includes it.
"""
import argparse
import json
import ssl
import time
import urllib.request
from pathlib import Path

import numpy as np

from common import DEFAULT_RESULTS


def frame_arrivals(stream, seconds):
    """Read an MJPEG stream for a while and return [(arrival time, JPEG size in bytes)]."""
    arrivals, buf, start = [], b"", time.monotonic()
    while time.monotonic() - start < seconds:
        chunk = stream.read1(65536)
        if not chunk:
            break
        buf += chunk
        while True:
            begin = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", begin + 2) if begin != -1 else -1
            if end == -1:
                buf = buf[begin:] if begin != -1 else buf[-1:]
                break
            arrivals.append((time.monotonic() - start, end + 2 - begin))
            buf = buf[end + 2:]
    return arrivals


def summarize(arrivals):
    if len(arrivals) < 3:
        return None
    arrivals = arrivals[1:]  # the first frame arrives on connection, not at the stream's rate
    times = np.array([t for t, _ in arrivals])
    sizes = np.array([s for _, s in arrivals], dtype=np.float64)
    gaps = np.diff(times)
    span = times[-1] - times[0]
    return {
        "frames": len(arrivals),
        "seconds": round(float(span), 1),
        "fps": round(float((len(arrivals) - 1) / span), 1),
        "gap_ms_median": round(float(np.median(gaps) * 1000), 1),
        "gap_ms_p95": round(float(np.percentile(gaps, 95) * 1000), 1),
        "frame_kb_median": round(float(np.median(sizes) / 1024), 1),
        "mbps": round(float(sizes[1:].sum() * 8 / span / 1e6), 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Measure the live stream's real frame rate and bandwidth.")
    ap.add_argument("--url", default="http://127.0.0.1:8000/stream")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--insecure", action="store_true", help="accept Vein Assist's self-signed https certificate")
    ap.add_argument("--out", default=str(DEFAULT_RESULTS / "stream.json"))
    args = ap.parse_args()

    context = ssl._create_unverified_context() if args.insecure else None
    print(f"Measuring {args.url} for {args.seconds:.0f} s...", flush=True)
    try:
        with urllib.request.urlopen(args.url, timeout=15, context=context) as stream:
            arrivals = frame_arrivals(stream, args.seconds)
    except OSError as exc:
        raise SystemExit(f"Couldn't read the stream ({exc}). Is the camera server running?")
    result = summarize(arrivals)
    if result is None:
        raise SystemExit("The stream connected but sent too few frames to measure. Is the camera connected?")
    result["url"] = args.url
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{result['fps']} FPS, {result['mbps']} Mbps, {result['frame_kb_median']} KB per frame, "
          f"gap between frames {result['gap_ms_median']} ms typical, {result['gap_ms_p95']} ms at worst 5%")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
