# Vein Evaluation

The project's own dataset and the numbers for the final presentation. Capture raw infrared frames, hand-label the
veins in some of them, then run one command to get slide-ready tables and figures.

```
capture.py ──> dataset/frames + manifest.csv ──> label.py ──> dataset/labels ──> evaluate.py ──> results/
```

You need Python 3.10 or newer. The first run creates a `.venv` folder and installs the packages. On Windows, use
`run.bat` instead of `./run.sh`. The Windows script hasn't been tested yet.

The dataset and results are images of people's arms, so they're kept out of git. Back them up somewhere private.

## 1. Capture (about 25 minutes for 360 frames)

Stop the camera server first: the Tapo allows only one viewer per stream. Set up the rig as in
[`../vein-camera-stream/README.md`](../vein-camera-stream/README.md), then run one session per person, site and lighting:

```
./run.sh capture --person P01 --site forearm --light dim
```

In the preview window, press **A** to start. Move the arm to a new position and hold it still, and a frame saves by
itself. Each frame must differ from the others in that session, so the dataset isn't full of near-copies. Press **A** to
pause while the arm is out of view, **Space** to save one frame by hand, and **Q** to quit.

| Option | Values |
|---|---|
| `--person` | A code such as `P01`. Never a name. Keep any list of who's who out of this folder. |
| `--site` | `hand`, `forearm`, `elbow` |
| `--light` | `dim`, `room`, `bright` |
| `--count` | Frames this session (default 30) |
| `--min-change` | If it keeps saying "too similar", lower this from the default 2.0 |

**A plan that clears Own Your Data (more than 300):** 4 or more people × 3 sites × 30 frames = 360. More people matters
more than more frames, because the results are averaged per person. Try to include different skin tones and some `room`
lighting. Ask each person's permission, and tell them what the images are for.

## 2. Label (about 2 minutes per frame)

```
./run.sh label --labeler A
```

Open http://localhost:8090. It picks 40 frames spread across people, sites and lighting (`--target` changes this).

1. **Outline the arm:** click 6 to 12 points around its edge.
2. **Paint the veins** (**V**) you're confident about. Shift-drag or right-drag erases, **Z** undoes, **[** and **]** change
   the brush, **+** and **-** zoom. Press **N** if no veins are visible.
3. **Enter** saves and moves on. Work is also saved when you move between frames or leave the page.

Label from the **raw** view first and use **E** to compare with the enhanced view, so the labels don't simply favor the
enhancement. If you can, check against the real arm while labeling.

A second person should label too, with `--labeler B`. They get the same frames first, and the report shows how
closely the two of you agree. Aim for at least 10 frames in common.

## 3. Measure the live stream

With the camera server running and the real camera connected:

```
./run.sh stream-fps
```

This is the frame rate to quote for **Blazingly Fast**, because the camera limits it, not the code. To measure what the
Quest receives, run it from another computer on the network with `--url https://<laptop-ip>:8443/camera/stream --insecure`.

**Delay from arm to screen** has no script: film the arm and the screen together in slow motion on a phone, snap your
fingers in view, and count frames until the screen shows it. **Voice answer time** is in Vein Assist's terminal, on the
`[timing]` lines.

## 4. Evaluate

```
./run.sh evaluate
```

Everything goes to `results/`:

| File | Use it for |
|---|---|
| `summary.md` | All the tables, ready to read |
| `fig_dataset.png` | Slide 2: Data |
| `fig_enhancement.png`, `fig_detection.png`, `fig_examples.png` | Slide 4: Results |
| `per_frame.csv`, `metrics.json` | Every score, if you want your own charts |

## What the numbers mean

**Enhancement.** Do the veins stand out more from the skin right next to them after each pipeline step?
- **Separability (AUC):** the chance a random nearby skin pixel is brighter than a random vein pixel. 0.5 is a coin flip
  and 1.0 is perfect. Only the order of brightness values counts, so a brightness curve like gamma can't inflate it.
- **Contrast-to-noise:** the skin-to-vein brightness gap divided by the spread of brightness. Higher is better.

**Detection.** Four detectors are compared against the hand labels: a local-threshold baseline on the raw image, the same on
the live enhancement, Hessian vesselness, and black-hat with hysteresis (the last two from `../vein-camera-stream/archive`).
- **Relaxed Dice:** overlap that forgives a miss of 3 px at 640 px wide, with precision and recall alongside it. Exact Dice
  and IoU are also reported.
- **Labelers' agreement:** a detector can't be expected to match the labels better than two people match each other.

**Keeping the numbers honest.**
- Each person counts equally, however many frames they contributed, and the 95% intervals are over people (they need at
  least 3). Frames from one person are too alike to count as independent evidence.
- Detector settings are chosen on other people's frames and scored on the person left out.
- Detectors are given the hand-drawn arm outline, so finding the arm automatically isn't tested. This also keeps deep
  learning out of the comparison.

## Limits to state on Slide 5

- The ground truth is people's reading of infrared images, not ultrasound or another clinical measure.
- It shows veins are easier to see. It doesn't show better needle placement or clinical outcomes.
- The sample is a small group of students, and may not represent other skin tones, ages or body types.

## Testing the tools

`./run.sh test` runs the checks on fake arms from `tests/synthetic.py`. Anything made from those is stamped
**SYNTHETIC TEST DATA** in the report and figures. Never present it as results.
