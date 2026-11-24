"""Fake infrared arms with known veins, to test the tools without a camera. Never use these as project results:
their manifest rows are marked `synthetic`, and the evaluation stamps its report and figures when it sees that."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import append_manifest, frames_dir, label_dir  # noqa: E402

W, H = 1280, 720


def fake_frame(rng, depth=26.0, grain=5.0):
    """A bright arm across a dark towel, with uneven lighting, sensor grain and darker vein lines.
    Returns (BGR frame, vein mask, arm outline)."""
    tilt = rng.uniform(-0.12, 0.12)
    top, bottom = rng.uniform(150, 210), rng.uniform(510, 570)
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)[:, None]
    upper = top + tilt * x
    lower = bottom + tilt * x
    arm = ((y > upper) & (y < lower)).astype(np.float32)
    outline = [[0, float(top)], [W - 1, float(top + tilt * (W - 1))], [W - 1, float(bottom + tilt * (W - 1))], [0, float(bottom)]]

    lighting = 150 + 50 * np.sin(x / W * rng.uniform(2, 4) + rng.uniform(0, 3)) * np.cos(y / H * 3)
    picture = 24 + arm * lighting
    vein = np.zeros((H, W), np.uint8)
    for _ in range(rng.integers(3, 6)):
        start = rng.uniform(top + 70, bottom - 70)
        wobble, phase = rng.uniform(15, 40), rng.uniform(0, 6)
        points = [(int(px), int(start + tilt * px + wobble * np.sin(px / 170 + phase))) for px in np.arange(40, W - 40, 20)]
        cv2.polylines(vein, [np.array(points, np.int32)], False, 255, int(rng.integers(8, 15)), cv2.LINE_AA)
    vein = np.where((vein > 127) & (arm > 0), 255, 0).astype(np.uint8)
    soft = cv2.GaussianBlur(vein.astype(np.float32) / 255, (0, 0), 2.5)
    picture = picture - depth * soft * arm
    picture = picture + rng.normal(0, grain, (H, W))
    gray = np.clip(picture, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), vein, outline


def write_label(data, labeler, stem, vein, outline):
    folder = label_dir(data, labeler)
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{stem}.veins.png"), vein)
    meta = dict(arm=outline, has_veins=bool(vein.any()), no_veins=False, labeler=labeler, width=W, height=H)
    (folder / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def make_dataset(data, people=4, per_person=4, second_labeler=True, seed=1):
    """A complete labeled dataset. The second labeler's labels are shifted slightly, as a real person's would be."""
    rng = np.random.default_rng(seed)
    frames_dir(data).mkdir(parents=True, exist_ok=True)
    sites, lights = ("hand", "forearm", "elbow"), ("dim", "room")
    for p in range(1, people + 1):
        person = f"P{p:02d}"
        for i in range(per_person):
            site, light = sites[i % 3], lights[i % 2]
            frame, vein, outline = fake_frame(rng)
            stem = f"{person}_{site}_{light}_synthetic_{i:04d}"
            cv2.imwrite(str(frames_dir(data) / f"{stem}.png"), frame)
            append_manifest(data, dict(file=f"{stem}.png", person=person, site=site, light=light, captured_at="synthetic",
                                       width=W, height=H, sharpness=0, source="synthetic"))
            write_label(data, "A", stem, vein, outline)
            if second_labeler:
                shifted = cv2.warpAffine(vein, np.float32([[1, 0, 2], [0, 1, 3]]), (W, H))
                write_label(data, "B", stem, shifted, outline)
