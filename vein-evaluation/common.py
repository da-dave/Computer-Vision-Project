"""Paths and dataset bookkeeping shared by the capture, labeling and evaluation tools."""
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAMERA_DIR = HERE.parent / "vein-camera-stream"
DEFAULT_DATA = HERE / "dataset"
DEFAULT_RESULTS = HERE / "results"
sys.path[:0] = [str(CAMERA_DIR), str(CAMERA_DIR / "archive")]  # the live pipeline and the archived detectors

import server as camera  # noqa: E402  (vein-camera-stream/server.py: the code that runs live)

# The live server's default settings (its --contrast, --darken and --gamma options).
LIVE_SETTINGS = dict(contrast=3.0, darken=1.5, gamma=1.4)
FIELDS = ["file", "person", "site", "light", "captured_at", "width", "height", "sharpness", "source"]
SITES = ("hand", "forearm", "elbow")
LIGHTS = ("dim", "room", "bright")
OWN_DATA_TARGET = 300  # the "Own Your Data" achievement needs more than 300 samples


def live_enhancer(width):
    return camera.Enhancer(width, **LIVE_SETTINGS)


def frames_dir(data):
    return Path(data) / "frames"


def manifest_path(data):
    return Path(data) / "manifest.csv"


def read_manifest(data):
    """Every captured frame still on disk, as dicts with a `stem` added."""
    path = manifest_path(data)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r, stem=Path(r["file"]).stem) for r in csv.DictReader(fh)]
    return [r for r in rows if (frames_dir(data) / r["file"]).exists()]


def append_manifest(data, row):
    path = manifest_path(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def is_synthetic(rows):
    return any(r.get("source", "").startswith("synthetic") for r in rows)


# ---------------------------------------------------------------- labels
def label_dir(data, labeler):
    return Path(data) / "labels" / labeler


def labelers(data):
    root = Path(data) / "labels"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def read_label(data, labeler, stem):
    """The label's details, or None if there isn't one yet."""
    path = label_dir(data, labeler) / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def is_complete(meta):
    """A finished label has an arm outline, and either veins or a note that none are visible."""
    return bool(meta) and len(meta.get("arm", [])) >= 3 and bool(meta.get("has_veins") or meta.get("no_veins"))


def complete_stems(data, labeler):
    folder = label_dir(data, labeler)
    if not folder.exists():
        return set()
    return {p.stem for p in folder.glob("*.json") if is_complete(json.loads(p.read_text(encoding="utf-8")))}


def read_vein_mask(data, labeler, stem):
    return cv2.imread(str(label_dir(data, labeler) / f"{stem}.veins.png"), cv2.IMREAD_GRAYSCALE)


def arm_mask(points, native_w, size):
    """Rasterize an arm outline drawn on the full-size image at another image size (W, H)."""
    W, H = size
    poly = np.round(np.array(points, np.float32) * (W / native_w)).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


# ---------------------------------------------------------------- which frames to label
def label_queue(rows, already_labeled, target):
    """Frames to hand-label: anything already labeled, then a spread across people, sites and lighting.

    People take turns, so every person is represented however many frames they contributed. Within a
    person, sites and lighting take turns too. Always the same order for the same frames."""
    by_person = {}
    for r in sorted(rows, key=lambda r: r["file"]):
        by_person.setdefault(r["person"], {}).setdefault((r["site"], r["light"]), []).append(r["stem"])
    orders = []
    for person in sorted(by_person):
        groups = [list(v) for _, v in sorted(by_person[person].items())]
        rng = random.Random(person)  # scattered through each session rather than consecutive frames
        for group in groups:
            rng.shuffle(group)
        order = []
        while any(groups):
            for group in groups:
                if group:
                    order.append(group.pop())
        orders.append(order)
    known = {r["stem"] for r in rows}
    queue = [s for s in sorted(already_labeled) if s in known]
    chosen = set(queue)
    while len(queue) < target and any(orders):
        for order in orders:
            while order:
                stem = order.pop(0)
                if stem not in chosen:
                    queue.append(stem)
                    chosen.add(stem)
                    break
            if len(queue) >= target:
                break
    return queue
