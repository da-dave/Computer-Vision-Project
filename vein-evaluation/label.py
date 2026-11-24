"""Hand-label veins in the browser: outline the arm, paint the veins. These labels are the evaluation's ground truth.

  ./run.sh label --labeler A          then open http://localhost:8090

Each person labels under their own --labeler letter. If two people label the same frames, the evaluation
reports how closely they agree, which is the best score any detector could be expected to reach.
Runs on this computer only (127.0.0.1), so the images aren't shared on the network.
"""
import argparse
import base64
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory, Response

from common import DEFAULT_DATA, complete_stems, frames_dir, is_complete, label_dir, label_queue, labelers, \
    live_enhancer, read_label, read_manifest

VEIN_COLOR = (255, 229, 0)  # BGR cyan, drawn over the grey picture


def create_app(data, labeler, target):
    data = Path(data)
    app = Flask(__name__, static_folder=None)
    rows = {r["stem"]: r for r in read_manifest(data)}
    labeled_by_anyone = set().union(*[complete_stems(data, name) for name in labelers(data)], set())
    queue = label_queue(list(rows.values()), labeled_by_anyone, target)
    folder = label_dir(data, labeler)

    def row_for(stem):
        if stem not in queue:  # only frames in the queue, so a request can't reach other files
            abort(404)
        return rows[stem]

    @lru_cache(maxsize=8)
    def enhanced_jpeg(stem):
        frame = cv2.imread(str(frames_dir(data) / rows[stem]["file"]))
        ok, jpg = cv2.imencode(".jpg", live_enhancer(frame.shape[1])(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jpg.tobytes()

    @app.route("/")
    def index():
        return send_from_directory(Path(__file__).parent, "label.html")

    @app.route("/api/queue")
    def api_queue():
        done = complete_stems(data, labeler)
        return jsonify(labeler=labeler, items=[
            dict(stem=s, person=rows[s]["person"], site=rows[s]["site"], light=rows[s]["light"],
                 width=int(rows[s]["width"]), height=int(rows[s]["height"]), done=s in done) for s in queue])

    @app.route("/image/<stem>/raw")
    def image_raw(stem):
        return send_file(frames_dir(data) / row_for(stem)["file"])

    @app.route("/image/<stem>/enhanced")
    def image_enhanced(stem):
        row_for(stem)
        return Response(enhanced_jpeg(stem), mimetype="image/jpeg")

    @app.route("/api/label/<stem>", methods=["GET"])
    def get_label(stem):
        row_for(stem)
        return jsonify(read_label(data, labeler, stem) or {})

    @app.route("/api/label/<stem>/veins.png")
    def get_veins(stem):
        """The painted veins as a transparent overlay, so the page can load them back for editing."""
        row_for(stem)
        path = folder / f"{stem}.veins.png"
        if not path.exists():
            abort(404)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        overlay = np.zeros((*mask.shape, 4), np.uint8)
        overlay[mask > 0] = (*VEIN_COLOR, 255)
        return Response(cv2.imencode(".png", overlay)[1].tobytes(), mimetype="image/png")

    @app.route("/api/label/<stem>", methods=["POST"])
    def save_label(stem):
        row = row_for(stem)
        body = request.get_json()
        arm = [[float(x), float(y)] for x, y in body.get("arm", [])]
        png = base64.b64decode(str(body["veins_png"]).split(",", 1)[1])
        painted = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if painted is None or painted.ndim != 3 or painted.shape[2] != 4:
            abort(400, "veins_png must be a transparent PNG")
        if painted.shape[:2] != (int(row["height"]), int(row["width"])):
            abort(400, "veins_png must match the image size")
        mask = np.where(painted[..., 3] > 127, 255, 0).astype(np.uint8)
        meta = dict(arm=arm, has_veins=bool(mask.any()), no_veins=bool(body.get("no_veins")), labeler=labeler,
                    width=int(row["width"]), height=int(row["height"]), saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{stem}.veins.png"), mask)
        tmp = folder / f"{stem}.json.tmp"  # the details go last, and appear whole, so a crash can't leave half a label
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        tmp.replace(folder / f"{stem}.json")
        return jsonify(done=is_complete(meta))

    return app


def main():
    ap = argparse.ArgumentParser(description="Hand-label veins in the browser.")
    ap.add_argument("--labeler", required=True, help="your letter, e.g. A (one folder of labels per person)")
    ap.add_argument("--target", type=int, default=40, help="how many frames to label (default 40)")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="dataset folder")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]{1,8}", args.labeler):
        sys.exit("--labeler must be a short code such as A.")
    if not read_manifest(args.data):
        sys.exit(f"No frames in {args.data} yet. Capture some first: ./run.sh capture --person P01 --site forearm --light dim")
    print(f"Open http://localhost:{args.port}", flush=True)
    create_app(args.data, args.labeler, args.target).run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
