"""Save raw infrared frames from the vein camera into the project's dataset.

  ./run.sh capture --person P01 --site forearm --light dim

Press A in the preview window to start. Then move the arm to a new position and hold it still: a frame is saved
automatically. Each saved frame has to differ from the others in this session, so the dataset isn't full of
near-identical copies. Press A again to pause while the arm is out of view, so empty frames aren't saved.
Frames are saved exactly as the camera sends them, before any enhancement, so the evaluation can compare
against the raw image.

The Tapo allows one viewer per stream, so stop the camera server first (or pass --source with stream2).
"""
import argparse
import os
import re
import sys
import time
from collections import deque

import cv2
import numpy as np

from common import DEFAULT_DATA, LIGHTS, SITES, append_manifest, camera, frames_dir, read_manifest

CHECK_EVERY = 0.1  # seconds between checks of the live image
SMALL_WIDTH = 160
PROGRESS_MARK = 300


def tiny(frame):
    """A small grey copy: enough to tell whether the picture changed, and ignores sensor grain."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (SMALL_WIDTH, max(1, SMALL_WIDTH * h // w)), interpolation=cv2.INTER_AREA).astype(np.float32)


def difference(a, b):
    return float(np.mean(np.abs(a - b)))


class Gate:
    """Decides when to save: the arm has been still for a moment and looks different from every saved frame."""

    def __init__(self, interval=1.5, min_change=2.0, settle=0.5):
        self.interval, self.min_change, self.settle = interval, min_change, settle
        self.still_limit = min_change / 2
        self.saved_images, self.last_saved = [], float("-inf")
        self.recent = deque()

    def check(self, small, now):
        """Returns "ok", or the reason not to save yet: "wait", "moving" or "similar"."""
        self.recent.append((now, small))
        while len(self.recent) > 1 and now - self.recent[1][0] >= self.settle:
            self.recent.popleft()
        then, earlier = self.recent[0]
        if now - self.last_saved < self.interval or now - then < self.settle:
            return "wait"
        if difference(small, earlier) > self.still_limit:
            return "moving"
        if self.saved_images and min(difference(small, s) for s in self.saved_images) < self.min_change:
            return "similar"
        return "ok"

    def saved(self, small, now):
        self.saved_images.append(small)
        self.last_saved = now


def looks_like_color(frame):
    """In infrared mode the Tapo's picture has no color, so color means Night Vision isn't set to Infrared."""
    sample = frame[::16, ::16].astype(np.int16)
    return float(np.mean(np.abs(sample[..., 0] - sample[..., 2]))) > 5


def save_frame(data, frame, args, source_kind, index):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{args.person}_{args.site}_{args.light}_{stamp}_{index:04d}.png"
    frames_dir(data).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(frames_dir(data) / name), frame):
        sys.exit(f"Couldn't write {name}. Is the disk full?")
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    append_manifest(data, dict(file=name, person=args.person, site=args.site, light=args.light,
                               captured_at=time.strftime("%Y-%m-%dT%H:%M:%S"), width=w, height=h,
                               sharpness=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1), source=source_kind))
    return name


MESSAGES = {
    "ok": "Saving",
    "wait": "Move the arm to a new position, then hold still",
    "moving": "Hold still",
    "similar": "Too similar to a saved frame: move the arm or change the angle",
}


def draw_preview(frame, auto, state, saved, count, color_warning):
    h, w = frame.shape[:2]
    view = cv2.resize(frame, (960, 960 * h // w), interpolation=cv2.INTER_AREA)
    lines = [f"{saved}/{count} saved   {'AUTO' if auto else 'PAUSED'}   A auto  SPACE save  Q quit", MESSAGES.get(state, "")]
    if color_warning:
        lines.append("Color picture: set Night Vision to Infrared in the Tapo app")
    for i, text in enumerate(lines):
        y = 30 + 30 * i
        cv2.putText(view, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(view, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser(description="Save raw infrared frames for the dataset.")
    ap.add_argument("--person", required=True, help="a code such as P01, never a name")
    ap.add_argument("--site", required=True, choices=SITES)
    ap.add_argument("--light", required=True, choices=LIGHTS, help="room lighting for this session")
    ap.add_argument("--count", type=int, default=30, help="frames to save this session (default 30)")
    ap.add_argument("--interval", type=float, default=1.5, help="minimum seconds between saves")
    ap.add_argument("--min-change", type=float, default=2.0,
                    help="how different a frame must be from saved ones, in grey levels (lower it if nothing saves)")
    ap.add_argument("--source", help="camera address, clip or webcam (default: CAMERA_SOURCE in vein-camera-stream/.env)")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="dataset folder")
    ap.add_argument("--no-preview", action="store_true", help="no window; start saving straight away")
    ap.add_argument("--timeout", type=float, default=30, help="give up if the camera sends nothing for this long (seconds)")
    args = ap.parse_args()

    if not re.fullmatch(r"P\d{2,3}", args.person):
        sys.exit("--person must be a code like P01, not a name, so the dataset stays anonymous.")
    camera.load_env(os.path.join(camera.HERE, ".env"))
    src = args.source or os.environ.get("CAMERA_SOURCE")
    if not src:
        sys.exit("No camera set: put CAMERA_SOURCE in vein-camera-stream/.env or pass --source.")

    before = len(read_manifest(args.data))
    print(f"Dataset: {before} frames so far. Connecting to {camera.safe(src)}...", flush=True)
    print("If it can't connect, stop the camera server first: the Tapo allows one viewer per stream.", flush=True)
    source = camera.Source(src)
    gate = Gate(args.interval, args.min_change)
    auto, preview = args.no_preview, not args.no_preview
    saved, last_frame, state, started, color_warning = 0, None, "wait", time.time(), False

    try:
        while saved < args.count:
            frame = source.frame
            if frame is None:
                if time.time() - started > args.timeout:
                    sys.exit(f"No picture after {args.timeout:.0f} s. Check the camera is on and on this network, "
                             "and that nothing else is viewing it.")
                time.sleep(0.05)
                continue
            now = time.monotonic()
            small = tiny(frame)
            state = gate.check(small, now)
            manual = False
            if preview:
                color_warning = color_warning or looks_like_color(frame)
                try:
                    cv2.imshow("VeinViz capture", draw_preview(frame, auto, state, saved, args.count, color_warning))
                    key = cv2.waitKey(int(CHECK_EVERY * 1000)) & 0xFF
                except cv2.error:
                    print("No preview window available here, so saving automatically.", flush=True)
                    preview, auto = False, True
                    key = 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    auto = not auto
                manual = key == ord(" ")
            else:
                time.sleep(CHECK_EVERY)
                if frame is last_frame:
                    continue
            last_frame = frame
            if manual or (auto and state == "ok"):
                name = save_frame(args.data, frame, args, "camera", before + saved + 1)
                gate.saved(small, now)
                saved += 1
                print(f"[{saved}/{args.count}] {name}", flush=True)
    except KeyboardInterrupt:
        pass
    if preview:
        cv2.destroyAllWindows()
    total = before + saved
    remaining = max(0, PROGRESS_MARK + 1 - total)
    print(f"\nSaved {saved} this session. Dataset total: {total}." +
          (f" {remaining} more for the Own Your Data achievement (more than {PROGRESS_MARK})." if remaining else ""))


if __name__ == "__main__":
    main()
