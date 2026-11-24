"""Checks that the metrics mean what they claim, and that the tools work together. Run with: ./run.sh test"""
import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
from capture import Gate, looks_like_color  # noqa: E402
from common import complete_stems, label_queue, live_enhancer, read_manifest  # noqa: E402
from label import create_app  # noqa: E402
from measure_stream import summarize  # noqa: E402
from synthetic import fake_frame, make_dataset  # noqa: E402


class Metrics(unittest.TestCase):
    def test_separability_extremes(self):
        skin, vein = np.full(100, 200, np.uint8), np.full(100, 50, np.uint8)
        self.assertEqual(ev.separability(skin, vein), 1.0)
        self.assertEqual(ev.separability(vein, skin), 0.0)
        self.assertEqual(ev.separability(skin, skin), 0.5)

    def test_gamma_cannot_inflate_separability(self):
        rng = np.random.default_rng(0)
        skin = rng.normal(150, 25, 20000).clip(0, 255).astype(np.uint8)
        vein = rng.normal(125, 25, 20000).clip(0, 255).astype(np.uint8)
        lut = (255 * (np.arange(256) / 255) ** 1.4).astype(np.uint8)
        self.assertAlmostEqual(ev.separability(skin, vein), ev.separability(lut[skin], lut[vein]), places=2)

    def test_contrast_to_noise_sign(self):
        skin, vein = np.array([150, 160], np.uint8), np.array([90, 100], np.uint8)
        self.assertGreater(ev.contrast_to_noise(skin, vein), 0)
        self.assertLess(ev.contrast_to_noise(vein, skin), 0)

    def test_overlap_identical_shifted_and_disjoint(self):
        truth = np.zeros((200, 200), np.uint8)
        cv2.line(truth, (20, 100), (180, 100), 255, 6)
        self.assertAlmostEqual(ev.overlap_scores(truth, truth)["dice"], 1.0)
        shifted = np.roll(truth, 2, axis=0)
        scores = ev.overlap_scores(shifted, truth)
        self.assertLess(scores["dice"], 1.0)
        self.assertAlmostEqual(scores["relaxed_dice"], 1.0)  # within tolerance
        far = np.roll(truth, 60, axis=0)
        self.assertEqual(ev.overlap_scores(far, truth)["relaxed_dice"], 0.0)
        self.assertEqual(ev.overlap_scores(np.zeros_like(truth), truth)["relaxed_dice"], 0.0)

    def test_wrong_labels_score_badly(self):
        """A control: real detectors must do much worse against labels that aren't on the veins."""
        frame, vein, outline = fake_frame(np.random.default_rng(3))
        f = ev.prepare(frame, outline, live_enhancer(ev.PROC_W))
        truth = np.where(cv2.resize(vein, f.gray.shape[::-1], interpolation=cv2.INTER_AREA) > 64, 255, 0).astype(np.uint8)
        wrong = np.roll(truth, 40, axis=0)
        wrong[f.arm == 0] = 0
        pred = ev.DETECTORS["Hessian vesselness"](f, 15)
        self.assertGreater(ev.overlap_scores(pred, truth)["relaxed_dice"], ev.overlap_scores(pred, wrong)["relaxed_dice"] + 0.2)


class Statistics(unittest.TestCase):
    def test_each_person_counts_equally(self):
        summary = ev.person_summary({"P01": [1.0] * 100, "P02": [0.0]})
        self.assertAlmostEqual(summary["mean"], 0.5)

    def test_interval_needs_three_people(self):
        self.assertIsNone(ev.person_summary({"P01": [1.0], "P02": [0.5]})["low"])
        s = ev.person_summary({"P01": [0.2], "P02": [0.5], "P03": [0.8]})
        self.assertLessEqual(s["low"], s["mean"])
        self.assertGreaterEqual(s["high"], s["mean"])

    def test_setting_is_chosen_without_the_person_left_out(self):
        # P01 alone is best at setting 5 and P02 and P03 at 10, so P01 must get 10 despite its own score.
        # P02's choice comes from P01 and P03, who average 0.6 at setting 5 and 0.4 at setting 10.
        scores = {"P01": {5: [1.0], 10: [0.0]}, "P02": {5: [0.2], 10: [0.8]}, "P03": {5: [0.2], 10: [0.8]}}
        chosen = ev.leave_one_person_out(scores)
        self.assertEqual(chosen["P01"], 10)
        self.assertEqual(chosen["P02"], 5)


class CaptureGate(unittest.TestCase):
    def setUp(self):
        self.a = np.full((90, 160), 100.0, np.float32)
        self.b = np.full((90, 160), 120.0, np.float32)

    def settled(self, gate, image, start):
        for t in np.arange(start, start + 0.7, 0.1):
            state = gate.check(image, float(t))
        return state

    def test_saves_once_still_and_refuses_duplicates(self):
        gate = Gate(interval=1.0, min_change=2.0, settle=0.5)
        self.assertEqual(self.settled(gate, self.a, 0.0), "ok")
        gate.saved(self.a, 0.6)
        self.assertEqual(self.settled(gate, self.a, 2.0), "similar")
        self.assertEqual(self.settled(gate, self.b, 4.0), "ok")

    def test_waits_while_moving(self):
        gate = Gate(interval=1.0, min_change=2.0, settle=0.5)
        state = None
        for i, t in enumerate(np.arange(0, 1.5, 0.1)):
            state = gate.check(self.a if i % 2 else self.b, float(t))
        self.assertEqual(state, "moving")

    def test_color_warning(self):
        grey = np.full((64, 64, 3), 120, np.uint8)
        color = grey.copy()
        color[..., 2] = 200
        self.assertFalse(looks_like_color(grey))
        self.assertTrue(looks_like_color(color))


class LabelQueue(unittest.TestCase):
    def rows(self):
        return [dict(file=f"{p}_{s}_{i}.png", stem=f"{p}_{s}_{i}", person=p, site=s, light="dim")
                for p, n in (("P01", 200), ("P02", 10)) for s in ("hand", "forearm") for i in range(n)]

    def test_people_are_balanced_and_order_is_stable(self):
        queue = label_queue(self.rows(), set(), 20)
        self.assertEqual(len(queue), 20)
        self.assertEqual(sum(s.startswith("P01") for s in queue), 10)
        self.assertEqual(sum("hand" in s for s in queue), 10)
        self.assertEqual(queue, label_queue(self.rows(), set(), 20))

    def test_labeled_frames_stay_in_the_queue(self):
        queue = label_queue(self.rows(), {"P02_hand_3"}, 5)
        self.assertIn("P02_hand_3", queue)
        self.assertEqual(len(queue), 5)
        self.assertEqual(len(set(queue)), 5)


class Stream(unittest.TestCase):
    def test_rate_ignores_the_first_frame(self):
        arrivals = [(0.0, 1000)] + [(1.0 + i / 10, 1000) for i in range(21)]
        self.assertAlmostEqual(summarize(arrivals)["fps"], 10.0)
        self.assertIsNone(summarize(arrivals[:2]))


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "dataset"
        cls.out = Path(cls.tmp.name) / "results"
        make_dataset(cls.data, people=3, per_person=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_evaluation(self):
        argv = sys.argv
        sys.argv = ["evaluate.py", "--data", str(self.data), "--out", str(self.out)]
        try:
            ev.main()
        finally:
            sys.argv = argv

    def test_report_is_written_and_stamped_synthetic(self):
        self.run_evaluation()
        summary = (self.out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("SYNTHETIC", summary)
        for name in ("fig_enhancement.png", "fig_detection.png", "fig_examples.png", "fig_dataset.png", "per_frame.csv", "metrics.json"):
            self.assertTrue((self.out / name).exists(), name)
        metrics = json.loads((self.out / "metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["synthetic"])
        self.assertEqual(metrics["agreement"]["frames"], 9)
        for name, entry in metrics["detection"].items():
            self.assertTrue(0 <= entry["relaxed_dice"]["mean"] <= 1, name)

    def test_label_server_round_trip(self):
        client = create_app(self.data, "C", 4).test_client()
        items = client.get("/api/queue").get_json()["items"]
        self.assertEqual(len(items), 9)  # everything already labeled stays in the queue
        stem, w, h = items[0]["stem"], items[0]["width"], items[0]["height"]
        painted = np.zeros((h, w, 4), np.uint8)
        painted[300:320, 100:900] = (255, 229, 0, 255)
        url = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", painted)[1].tobytes()).decode()
        reply = client.post(f"/api/label/{stem}", json=dict(arm=[[0, 0], [100, 0], [100, 100]], veins_png=url))
        self.assertTrue(reply.get_json()["done"])
        self.assertIn(stem, complete_stems(self.data, "C"))
        saved = cv2.imread(str(self.data / "labels" / "C" / f"{stem}.veins.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(int(saved[310, 500]), 255)
        self.assertEqual(int(saved[0, 0]), 0)
        self.assertEqual(client.get(f"/api/label/{stem}/veins.png").status_code, 200)
        self.assertEqual(client.get("/image/raw/raw").status_code, 404)  # not in the queue
        self.assertEqual(client.get(f"/image/{stem}/enhanced").status_code, 200)
        wrong_size = base64.b64encode(cv2.imencode(".png", painted[:10, :10])[1].tobytes()).decode()
        bad = client.post(f"/api/label/{stem}", json=dict(arm=[], veins_png="data:image/png;base64," + wrong_size))
        self.assertEqual(bad.status_code, 400)
        self.assertTrue(read_manifest(self.data))


if __name__ == "__main__":
    unittest.main()
