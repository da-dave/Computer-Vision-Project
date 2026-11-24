"""Score the vein pipeline against the hand labels, and write slide-ready tables and figures to results/.

  ./run.sh evaluate

What it measures:
  Enhancement: does each step of the live pipeline make veins easier to tell apart from nearby skin?
  Detection:   how closely do vein detectors match the hand labels?
  Speed:       how fast does the code run, and (from stream-fps) what does the stream really deliver?

Frames from one person are alike, so people, not frames, are the unit behind the confidence intervals, and
detector settings are always chosen on other people's frames (leave-one-person-out).
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_DATA, DEFAULT_RESULTS, OWN_DATA_TARGET, arm_mask, complete_stems, frames_dir, is_synthetic, \
    labelers, live_enhancer, read_label, read_manifest, read_vein_mask

from heatmap import flatten, vesselness  # noqa: E402  (vein-camera-stream/archive)
from veins import VeinDetector  # noqa: E402

OUT_W = 1280   # the live stream's width: enhancement is scored on what viewers see
PROC_W = 640   # the width the archived detectors were designed for
STAGES = [("raw", "Raw camera image"), ("flattened", "+ Lighting flattened"), ("clahe", "+ CLAHE contrast"),
          ("darkened", "+ Dark lines deepened"), ("final", "+ Gamma (live output)")]
BASELINE = "Local threshold on raw image"
ENHANCED_INPUT = "Local threshold on live enhancement"  # its speed includes running the enhancement first
SPEED_FRAMES = 30
TOP_PERCENTS = (3, 5, 7, 10, 15, 20, 26, 35)  # settings to choose from: % of the arm marked as vein
MIN_BLOB = 60          # smallest vein blob kept, in pixels at PROC_W (as in the archived detector)
EDGE = 9               # pixels ignored along the arm outline (as in the archived detector)
RELAX_PX = 3           # hand labels aren't pixel perfect: a match within this distance at PROC_W counts
SYNTHETIC_NOTE = "SYNTHETIC TEST DATA: checks the tools work. These are not project results."
INK, MUTED, GRID, BLUE, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


# ---------------------------------------------------------------- metrics
def ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def skin_ring(vein, arm):
    """Skin right next to the veins: a band around them, leaving a small gap for imprecise label edges."""
    W = vein.shape[1]
    near = cv2.dilate(vein, ellipse(max(1, round(W / 320))))
    wide = cv2.dilate(vein, ellipse(max(2, round(W / 64))))
    return (wide > 0) & (near == 0) & (arm > 0)


def contrast_to_noise(skin, vein):
    """(skin brightness - vein brightness) / combined spread. Positive means veins are darker, as they should be."""
    skin, vein = skin.astype(np.float64), vein.astype(np.float64)
    spread = np.sqrt(skin.var() + vein.var())
    return float((skin.mean() - vein.mean()) / spread) if spread > 0 else 0.0


def separability(skin, vein):
    """AUC: the chance a random skin pixel is brighter than a random vein pixel (0.5 = no better than a coin flip).

    Only the order of brightness values counts, so a brightness curve such as gamma can't inflate it."""
    skin_hist = np.bincount(skin.ravel(), minlength=256).astype(np.float64)
    vein_hist = np.bincount(vein.ravel(), minlength=256).astype(np.float64)
    darker = np.concatenate(([0.0], np.cumsum(vein_hist)[:-1]))
    return float((skin_hist * (darker + 0.5 * vein_hist)).sum() / (skin_hist.sum() * vein_hist.sum()))


def enhancement_scores(image, vein, arm):
    inside = (vein > 0) & (arm > 0)
    ring = skin_ring(vein, arm)
    if inside.sum() < 50 or ring.sum() < 50:
        return None
    return dict(cnr=contrast_to_noise(image[ring], image[inside]), auc=separability(image[ring], image[inside]))


def overlap_scores(pred, truth, relax=RELAX_PX):
    """Exact overlap (Dice, IoU), plus tolerant precision, recall and Dice that forgive small label offsets."""
    p, t = pred > 0, truth > 0
    both, np_, nt = (p & t).sum(), p.sum(), t.sum()
    near_truth = cv2.dilate(truth, ellipse(relax)) > 0
    near_pred = cv2.dilate(pred, ellipse(relax)) > 0
    precision = float((p & near_truth).sum() / np_) if np_ else 0.0
    recall = float((t & near_pred).sum() / nt) if nt else 0.0
    return dict(dice=float(2 * both / (np_ + nt)) if np_ + nt else 0.0,
                iou=float(both / (p | t).sum()) if (p | t).any() else 0.0,
                precision=precision, recall=recall,
                relaxed_dice=2 * precision * recall / (precision + recall) if precision + recall else 0.0)


# ---------------------------------------------------------------- statistics
def person_summary(values_by_person, rounds=2000, seed=0):
    """Mean over people (each person counts equally) with a 95% bootstrap interval over people.

    The interval needs at least 3 people; with fewer, `low` and `high` are None."""
    means = np.array([np.mean(v) for v in values_by_person.values() if len(v)], dtype=np.float64)
    if not len(means):
        return dict(mean=None, low=None, high=None, people=0)
    result = dict(mean=float(means.mean()), low=None, high=None, people=int(len(means)))
    if len(means) >= 3:
        rng = np.random.default_rng(seed)
        draws = means[rng.integers(0, len(means), (rounds, len(means)))].mean(axis=1)
        result["low"], result["high"] = (float(x) for x in np.percentile(draws, (2.5, 97.5)))
    return result


def leave_one_person_out(scores):
    """scores[person][setting] = per-frame scores. Pick each person's setting using only the other people.

    Returns {person: setting}. With just one person there's nobody else to learn from, so it uses their own best."""
    people = sorted(scores)
    chosen = {}
    for person in people:
        others = [p for p in people if p != person] or [person]
        settings = sorted(scores[person])
        chosen[person] = max(settings, key=lambda s: (np.mean([np.mean(scores[p][s]) for p in others]), -s))
    return chosen


# ---------------------------------------------------------------- detectors
@dataclass
class Frame:
    """One labeled frame prepared for the detectors, all at PROC_W."""
    gray: np.ndarray
    enhanced: np.ndarray
    arm: np.ndarray
    inner: np.ndarray


def prepare(frame, arm_points, enhancer):
    h, w = frame.shape[:2]
    size = (PROC_W, round(h * PROC_W / w))
    gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    arm = arm_mask(arm_points, w, size)
    inner = cv2.erode(arm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (EDGE, EDGE)),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return Frame(gray, enhancer(frame), arm, inner)


def local_darkness(image):
    """How much darker each pixel is than its surroundings."""
    g = image.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), g.shape[1] / 90) - g


def keep_top_percent(score, inner, percent):
    values = score[inner > 0]
    if values.size < 50:
        return np.zeros_like(inner)
    mask = ((score > max(float(np.percentile(values, 100 - percent)), 0)) & (inner > 0)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_BLOB
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def blackhat_hysteresis(f, percent):
    settings = dict(contrast=3.0, width=21, smooth=0.6, min_size=MIN_BLOB, sens=percent)  # archived detector's defaults
    return VeinDetector()(f.gray, f.arm, settings, moving=False)


DETECTORS = {
    BASELINE: lambda f, s: keep_top_percent(local_darkness(f.gray), f.inner, s),
    ENHANCED_INPUT: lambda f, s: keep_top_percent(local_darkness(f.enhanced), f.inner, s),
    "Hessian vesselness": lambda f, s: keep_top_percent(
        vesselness(cv2.GaussianBlur(flatten(f.gray.astype(np.float32)), (0, 0), 1.0)), f.inner, s),
    "Black-hat + hysteresis": blackhat_hysteresis,
}


# ---------------------------------------------------------------- the evaluation
def evaluate(data, labeler, second_labeler):
    rows = read_manifest(data)
    done = complete_stems(data, labeler) if labeler else set()
    labeled = [r for r in rows if r["stem"] in done]
    stage_scores, per_frame, no_veins, too_small = defaultdict(lambda: defaultdict(list)), [], 0, 0
    detection = {name: defaultdict(lambda: defaultdict(list)) for name in DETECTORS}
    speed = {name: [] for name in DETECTORS}
    detail = {}  # (detector, person, setting) -> [(stem, scores)]
    live_times, enhance_times, enhance_out, enhance_proc = [], [], live_enhancer(OUT_W), live_enhancer(PROC_W)

    for row in rows[:SPEED_FRAMES]:  # the same work the live server does for each frame
        frame = cv2.imread(str(frames_dir(data) / row["file"]))
        started = time.perf_counter()
        cv2.imencode(".jpg", enhance_out(frame), [cv2.IMWRITE_JPEG_QUALITY, 85])
        live_times.append(time.perf_counter() - started)

    for i, row in enumerate(labeled, 1):
        print(f"[{i}/{len(labeled)}] {row['stem']}", flush=True)
        frame = cv2.imread(str(frames_dir(data) / row["file"]))
        meta = read_label(data, labeler, row["stem"])
        if not meta.get("has_veins"):
            no_veins += 1
            continue
        h, w = frame.shape[:2]
        mask = read_vein_mask(data, labeler, row["stem"])

        size = (OUT_W, round(h * OUT_W / w))
        vein = np.where(cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 64, 255, 0).astype(np.uint8)
        arm = arm_mask(meta["arm"], w, size)
        images = enhance_out.stages(frame)
        scored = {key: enhancement_scores(images[key], vein, arm) for key, _ in STAGES}
        if any(v is None for v in scored.values()):
            too_small += 1
            continue
        for key, scores in scored.items():
            stage_scores[key][row["person"]].append(scores)
            per_frame.append(dict(stem=row["stem"], person=row["person"], site=row["site"], light=row["light"],
                                  kind="enhancement", name=key, **scores))

        started = time.perf_counter()
        enhance_proc(frame)
        enhance_times.append(time.perf_counter() - started)
        prepared = prepare(frame, meta["arm"], enhance_proc)
        truth = np.where(cv2.resize(mask, prepared.gray.shape[::-1], interpolation=cv2.INTER_AREA) > 64, 255, 0).astype(np.uint8)
        truth[prepared.arm == 0] = 0
        for name, detect in DETECTORS.items():
            for percent in TOP_PERCENTS:
                started = time.perf_counter()
                pred = detect(prepared, percent)
                speed[name].append(time.perf_counter() - started)
                scores = overlap_scores(pred, truth)
                detection[name][row["person"]][percent].append(scores["relaxed_dice"])
                detail.setdefault((name, row["person"], percent), []).append((row, scores))

    results = dict(dataset=dataset_summary(rows), labeler=labeler, labeled_frames=len(labeled), no_visible_veins=no_veins,
                   excluded_small_labels=too_small, synthetic=is_synthetic(rows), enhancement={}, detection={}, speed={})
    results["speed"]["live_pipeline_fps"] = round(1 / float(np.median(live_times)), 1) if live_times else None
    for key, label in STAGES:
        people = stage_scores[key]
        if not people:
            continue
        entry = dict(label=label)
        for metric in ("cnr", "auc"):
            entry[metric] = person_summary({p: [s[metric] for s in v] for p, v in people.items()})
        if key != "raw":
            for metric in ("cnr", "auc"):
                entry[f"{metric}_gain"] = person_summary(
                    {p: [a[metric] - b[metric] for a, b in zip(people[p], stage_scores["raw"][p])] for p in people})
                frames = [(a[metric], b[metric]) for p in people for a, b in zip(people[p], stage_scores["raw"][p])]
                entry[f"{metric}_better_frames"] = [sum(a > b for a, b in frames), len(frames)]
        results["enhancement"][key] = entry
    results["groups"] = group_summary(per_frame)

    chosen_settings = {}
    for name in DETECTORS:
        if not detection[name]:
            continue
        chosen = leave_one_person_out(detection[name])
        chosen_settings[name] = chosen
        held_out = defaultdict(list)
        for person, setting in chosen.items():
            for row, scores in detail[(name, person, setting)]:
                held_out[person].append(scores)
                per_frame.append(dict(stem=row["stem"], person=person, site=row["site"], light=row["light"],
                                      kind="detection", name=name, setting=setting, **scores))
        results["detection"][name] = dict(
            settings=dict(Counter(chosen.values())), single_person=len(chosen) == 1,
            **{m: person_summary({p: [s[m] for s in v] for p, v in held_out.items()})
               for m in ("dice", "iou", "precision", "recall", "relaxed_dice")},
            fps=round(1 / (float(np.median(speed[name])) + (float(np.median(enhance_times)) if name == ENHANCED_INPUT else 0)), 1))
    results["agreement"] = labeler_agreement(data, rows, labeler, second_labeler)
    return results, per_frame, chosen_settings


def dataset_summary(rows):
    return dict(frames=len(rows), people=len({r["person"] for r in rows}), target=OWN_DATA_TARGET,
                meets_own_data_target=len(rows) > OWN_DATA_TARGET,
                by_person=dict(sorted(Counter(r["person"] for r in rows).items())),
                by_site=dict(sorted(Counter(r["site"] for r in rows).items())),
                by_light=dict(sorted(Counter(r["light"] for r in rows).items())))


def group_summary(per_frame):
    """Live output vs raw, by site, lighting and person: where does the enhancement help, and where not?"""
    final = {r["stem"]: r for r in per_frame if r["kind"] == "enhancement" and r["name"] == "final"}
    raw = {r["stem"]: r for r in per_frame if r["kind"] == "enhancement" and r["name"] == "raw"}
    groups = {}
    for field in ("site", "light", "person"):
        table = defaultdict(list)
        for stem, row in final.items():
            table[row[field]].append((raw[stem]["auc"], row["auc"]))
        groups[field] = {k: dict(frames=len(v), raw_auc=float(np.mean([a for a, _ in v])), final_auc=float(np.mean([b for _, b in v])))
                         for k, v in sorted(table.items())}
    return groups


def labeler_agreement(data, rows, first, second):
    """How closely two people's labels match on the same frames: the realistic ceiling for any detector."""
    if not first or not second:
        return None
    shared = complete_stems(data, first) & complete_stems(data, second)
    by_person = defaultdict(list)
    for row in rows:
        if row["stem"] not in shared:
            continue
        a, b = (read_vein_mask(data, who, row["stem"]) for who in (first, second))
        if a is None or b is None or not a.any() or not b.any():
            continue
        h, w = a.shape
        size = (PROC_W, round(h * PROC_W / w))
        a, b = (np.where(cv2.resize(m, size, interpolation=cv2.INTER_AREA) > 64, 255, 0).astype(np.uint8) for m in (a, b))
        by_person[row["person"]].append(overlap_scores(a, b))
    if not by_person:
        return None
    return dict(labelers=[first, second], frames=sum(len(v) for v in by_person.values()),
                relaxed_dice=person_summary({p: [s["relaxed_dice"] for s in v] for p, v in by_person.items()}),
                dice=person_summary({p: [s["dice"] for s in v] for p, v in by_person.items()}))


# ---------------------------------------------------------------- report
def fmt(summary, digits=2):
    if not summary or summary["mean"] is None:
        return "n/a"
    text = f"{summary['mean']:.{digits}f}"
    if summary["low"] is not None:
        text += f" [{summary['low']:.{digits}f}, {summary['high']:.{digits}f}]"
    return text


def counts(table):
    return ", ".join(f"{key} {value}" for key, value in table.items())


def write_summary(results, path, stream):
    d = results["dataset"]
    out = ["# VeinViz evaluation results", ""]
    if results["synthetic"]:
        out += [f"> **{SYNTHETIC_NOTE}**", ""]
    labels = (f"- Hand-labeled by `{results['labeler']}`: {results['labeled_frames']} frames "
              f"({results['no_visible_veins']} marked no visible veins, {results['excluded_small_labels']} too small to score)."
              if results["labeler"] else "- No hand labels yet. Run `./run.sh label --labeler A`.")
    out += ["## Dataset", "",
            f"- **{d['frames']} raw infrared frames from {d['people']} {'person' if d['people'] == 1 else 'people'}.** "
            + ("Meets" if d["meets_own_data_target"] else "Doesn't yet meet")
            + f" the Own Your Data target (more than {d['target']}).",
            f"- By site: {counts(d['by_site'])}. By lighting: {counts(d['by_light'])}. By person: {counts(d['by_person'])}.",
            labels, ""]

    if results["enhancement"]:
        people = next(iter(results["enhancement"].values()))["cnr"]["people"]
        note = "95% intervals over people in brackets." if people >= 3 else "Too few people for intervals (needs 3 or more)."
        out += ["## Enhancement: can veins be told apart from nearby skin?", "",
                f"Mean over {people} people. {note} Separability (AUC) can't be inflated by a brightness curve; "
                "0.5 is a coin flip, 1.0 is perfect.", "",
                "| Pipeline step | Contrast-to-noise | Separability (AUC) | Change in AUC vs raw | Frames better than raw |",
                "|---|---:|---:|---:|---:|"]
        for key, e in results["enhancement"].items():
            gain = fmt(e.get("auc_gain"), 3) if "auc_gain" in e else "baseline"
            better = "{}/{}".format(*e["auc_better_frames"]) if "auc_better_frames" in e else ""
            out.append(f"| {e['label']} | {fmt(e['cnr'])} | {fmt(e['auc'], 3)} | {gain} | {better} |")
        out += ["", "### Where it helps (live output vs raw, AUC)", "", "| Group | Frames | Raw | Live output |", "|---|---:|---:|---:|"]
        for field, table in results["groups"].items():
            for key, g in table.items():
                out.append(f"| {field}: {key} | {g['frames']} | {g['raw_auc']:.3f} | {g['final_auc']:.3f} |")
        out.append("")

    if results["detection"]:
        out += ["## Detection: how closely do detectors match the hand labels?", "",
                "Settings are chosen on other people's frames, then scored on the person left out. "
                f"Relaxed scores count a match within {RELAX_PX} px at {PROC_W} px wide. Detectors get the hand-drawn arm "
                "outline, so finding the arm automatically isn't being tested.", "",
                "| Detector | Relaxed Dice | Precision | Recall | Exact Dice | Exact IoU | FPS |", "|---|---:|---:|---:|---:|---:|---:|"]
        ranked = sorted(results["detection"].items(), key=lambda kv: -kv[1]["relaxed_dice"]["mean"])
        for name, e in ranked:
            out.append(f"| {name} | {fmt(e['relaxed_dice'])} | {fmt(e['precision'])} | {fmt(e['recall'])} | "
                       f"{fmt(e['dice'])} | {fmt(e['iou'])} | {e['fps']:.0f} |")
        if next(iter(results["detection"].values()))["single_person"]:
            out += ["", "Only one person is labeled, so settings were chosen on the same frames they're scored on. "
                        "These scores are optimistic; label a second person."]
        out.append("")
    if results["agreement"]:
        a = results["agreement"]
        out += ["## Agreement between labelers", "",
                f"`{a['labelers'][0]}` vs `{a['labelers'][1]}` on {a['frames']} shared frames: relaxed Dice {fmt(a['relaxed_dice'])}, "
                f"exact Dice {fmt(a['dice'])}. Detectors can't be expected to beat how well people agree with each other.", ""]

    out += ["## Speed", ""]
    fps = results["speed"]["live_pipeline_fps"]
    out.append(f"- Live pipeline code (enhance and JPEG, {OUT_W} px wide): **{fps} FPS** on this computer.")
    if stream:
        verdict = "meets" if stream["fps"] >= 30 else "doesn't meet"
        out.append(f"- Stream as delivered ({stream['url']}): **{stream['fps']} FPS**, {stream['mbps']} Mbps, "
                   f"{stream['frame_kb_median']} KB per frame. This {verdict} the Blazingly Fast 30+ FPS bar.")
    else:
        out.append("- Stream as delivered: not measured yet. With the camera running: `./run.sh stream-fps`. "
                   "This is the number to quote for Blazingly Fast, because the camera limits it, not the code.")
    out += ["", "## Limits to state with these numbers", "",
            "- Ground truth is people's reading of the infrared image, not a clinical measurement such as ultrasound.",
            "- Labelers can see the enhanced view, which may favor the enhancement. Label from raw first, and use a second labeler to check.",
            "- The result shows veins are easier to see. It doesn't show better needle placement or clinical outcomes.", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_per_frame(per_frame, path):
    fields = ["stem", "person", "site", "light", "kind", "name", "setting", "cnr", "auc", "dice", "iou", "precision", "recall", "relaxed_dice"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, length=0, labelsize=10)
    ax.tick_params(axis="y", colors=INK, length=0, labelsize=11)


def dot_panel(ax, labels, summaries, title, label_rows, ceiling=None, title_pad=14):
    """One dot per row with its 95% interval: magnitudes with uncertainty, without implying a zero baseline."""
    y = np.arange(len(labels))[::-1]
    for yi, s, i in zip(y, summaries, range(len(labels))):
        if s["low"] is not None:
            ax.plot([s["low"], s["high"]], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round", alpha=0.45, zorder=2)
        ax.plot(s["mean"], yi, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        if i in label_rows:
            ax.annotate(f"{s['mean']:.2f}", (s["mean"], yi), xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=11)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(labels) - 0.3)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=title_pad)
    lows = [s["low"] if s["low"] is not None else s["mean"] for s in summaries]
    highs = [s["high"] if s["high"] is not None else s["mean"] for s in summaries]
    span = max(max(highs) - min(lows), 0.02)
    upper = max(highs) + 0.12 * span
    ax.set_xlim(min(lows) - 0.12 * span, min(upper, ceiling + 0.05 * span) if ceiling else upper)
    style_axes(ax)


def stamp(fig, results):
    if results["synthetic"]:
        fig.text(0.5, 0.005, SYNTHETIC_NOTE, ha="center", va="bottom", color="#c0392b", fontsize=11, fontweight="bold")


def write_charts(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results["enhancement"]:
        entries = list(results["enhancement"].values())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
        last = len(entries) - 1
        dot_panel(axes[0], [e["label"] for e in entries], [e["cnr"] for e in entries], "Contrast-to-noise", {0, last})
        dot_panel(axes[1], [e["label"] for e in entries], [e["auc"] for e in entries], "Separability (AUC)", {0, last}, ceiling=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        stamp(fig, results)
        fig.savefig(out / "fig_enhancement.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)

    if results["detection"]:
        ranked = sorted(results["detection"].items(), key=lambda kv: -kv[1]["relaxed_dice"]["mean"])
        fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=SURFACE)
        names = [n for n, _ in ranked]
        best_and_baseline = {0, names.index(BASELINE)} if BASELINE in names else {0}
        dot_panel(ax, names, [e["relaxed_dice"] for _, e in ranked], "Match to hand labels (relaxed Dice)", best_and_baseline,
                  ceiling=1.0, title_pad=26 if results["agreement"] else 14)
        if results["agreement"]:
            level = results["agreement"]["relaxed_dice"]["mean"]
            ax.axvline(level, color=MUTED, linestyle="--", linewidth=1.5, zorder=1)
            low, high = ax.get_xlim()
            ax.set_xlim(min(low, level - 0.02), max(high, min(level + 0.02, 1.005)))
            ax.annotate("Labelers' agreement", (level, 1), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center", color=MUTED, fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        stamp(fig, results)
        fig.savefig(out / "fig_detection.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)


def write_dataset_grid(rows, results, out):
    """A grid of frames for the Data slide: as many different people, sites and lighting setups as fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks, seen = [], set()
    for r in sorted(rows, key=lambda r: r["file"]):
        key = (r["person"], r["site"], r["light"])
        if key not in seen:
            seen.add(key)
            picks.append(r)
    for r in sorted(rows, key=lambda r: r["file"]):
        if len(picks) >= 12:
            break
        if r not in picks:
            picks.append(r)
    picks = picks[:12]
    if not picks:
        return
    cols = min(4, len(picks))
    grid_rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(3 * cols, 2.1 * grid_rows + 0.3), squeeze=False, facecolor=SURFACE)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), picks):
        image = cv2.cvtColor(cv2.imread(str(frames_dir(results["_data"]) / r["file"])), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(f"{r['person']} · {r['site']} · {r['light']}", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig, results)
    fig.savefig(out / "fig_dataset.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def overlay(gray, mask, color):
    picture = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    picture[mask > 0] = (0.55 * picture[mask > 0] + 0.45 * np.array(color)).astype(np.uint8)
    picture[edges > 0] = color
    return picture


def write_examples(data, labeler, per_frame, chosen_settings, results, out):
    """Typical and hardest frames for the Results slide: raw, enhanced, hand label, best detector."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results["detection"]:
        return
    best = max(results["detection"], key=lambda n: results["detection"][n]["relaxed_dice"]["mean"])
    scored = sorted((r for r in per_frame if r["kind"] == "detection" and r["name"] == best), key=lambda r: r["relaxed_dice"])
    picks = [("Typical frame", scored[len(scored) // 2]), ("Hardest frame", scored[0])]
    files = {r["stem"]: r["file"] for r in read_manifest(data)}
    enhancer = live_enhancer(PROC_W)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.2), facecolor=SURFACE)
    for row_axes, (title, r) in zip(axes, picks):
        frame = cv2.imread(str(frames_dir(data) / files[r["stem"]]))
        meta = read_label(data, labeler, r["stem"])
        f = prepare(frame, meta["arm"], enhancer)
        truth = np.where(cv2.resize(read_vein_mask(data, labeler, r["stem"]), f.gray.shape[::-1], interpolation=cv2.INTER_AREA) > 64, 255, 0).astype(np.uint8)
        pred = DETECTORS[best](f, chosen_settings[best][r["person"]])
        panels = [(cv2.cvtColor(f.gray, cv2.COLOR_GRAY2RGB), f"{title} ({r['person']}): raw"),
                  (cv2.cvtColor(f.enhanced, cv2.COLOR_GRAY2RGB), "Live enhancement"),
                  (overlay(f.gray, truth, (0, 229, 255)), "Hand label"),
                  (overlay(f.gray, pred, (255, 197, 61)), f"{best}\nrelaxed Dice {r['relaxed_dice']:.2f}")]
        for ax, (picture, caption) in zip(row_axes, panels):
            ax.imshow(picture)
            ax.set_title(caption, fontsize=10, color=INK, loc="left")
            ax.axis("off")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stamp(fig, results)
    fig.savefig(out / "fig_examples.png", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def choose_labelers(data, first, second):
    available = sorted(labelers(data), key=lambda name: -len(complete_stems(data, name)))
    first = first or next((n for n in available if complete_stems(data, n)), None)
    second = second or next((n for n in available if n != first and complete_stems(data, n)), None)
    return first, second


def main():
    ap = argparse.ArgumentParser(description="Score the vein pipeline against the hand labels.")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="dataset folder")
    ap.add_argument("--out", default=str(DEFAULT_RESULTS), help="where to write tables and figures")
    ap.add_argument("--labeler", help="whose labels are the ground truth (default: whoever labeled the most)")
    ap.add_argument("--second-labeler", help="compare against these labels (default: the next labeler, if any)")
    args = ap.parse_args()

    rows = read_manifest(args.data)
    if not rows:
        raise SystemExit(f"No frames in {args.data} yet. Capture some first: ./run.sh capture --person P01 --site forearm --light dim")
    first, second = choose_labelers(args.data, args.labeler, args.second_labeler)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, per_frame, chosen = evaluate(args.data, first, second)
    results["_data"] = args.data
    stream_path = out / "stream.json"
    stream = json.loads(stream_path.read_text(encoding="utf-8")) if stream_path.exists() else None
    write_dataset_grid(rows, results, out)
    write_charts(results, out)
    write_examples(args.data, first, per_frame, chosen, results, out)
    write_per_frame(per_frame, out / "per_frame.csv")
    write_summary(results, out / "summary.md", stream)
    results.pop("_data")
    results["stream"] = stream
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if results["synthetic"]:
        print(f"\n{SYNTHETIC_NOTE}")
    if not first:
        print("\nNo labels yet, so only the dataset summary was written. Label some frames: ./run.sh label --labeler A")
    print(f"\nWrote {out / 'summary.md'} and the figures next to it.")


if __name__ == "__main__":
    main()
