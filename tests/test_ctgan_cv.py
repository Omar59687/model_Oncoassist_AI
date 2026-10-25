"""Small safety checks with a fake generator; no CTGAN training in unit tests."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import ctgan_cv
from baseline_cv import FILES


class FakeEncoder:
    def __init__(self, dim):
        self.dim = dim

    def predict(self, X, verbose=0):
        out = np.zeros((len(X), self.dim), dtype=float)
        out[:, 0] = X[:, 0]
        return out


def fake_encoder(X_fit, X_stop, latent_dim, seed):
    return FakeEncoder(latent_dim)


def fixture(root: Path):
    ids = [f"sample-{i:03d}" for i in range(40)]
    labels = ["1"] * 28 + ["2"] * 12
    for k, filename in enumerate(FILES.values()):
        pd.DataFrame({"SAMPLE_ID": ids, "feature": np.arange(40, dtype=float) + k,
                      "CLASS": labels}).to_csv(root / filename, index=False)


def fake_tf():
    return SimpleNamespace(
        keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=lambda seed: None)),
        config=SimpleNamespace(experimental=SimpleNamespace(enable_op_determinism=lambda: None)),
    )


class CTGANChecks(unittest.TestCase):
    def test_pac_and_even_batch_constraints(self):
        self.assertEqual(ctgan_cv.ctgan_batch_size(86), 80)
        self.assertEqual(ctgan_cv.ctgan_batch_size(69), 60)
        self.assertEqual(ctgan_cv.ctgan_batch_size(68), 60)
        with self.assertRaises(ValueError):
            ctgan_cv.ctgan_batch_size(9)

    def test_three_conditions_and_no_silent_fallback(self):
        X = np.arange(30, dtype=float).reshape(10, 3)
        y = np.array(["1"] * 7 + ["2"] * 3)
        seen = []

        def sampler(minority, needed, seed):
            seen.append((minority.copy(), needed, seed))
            return np.full((needed, X.shape[1]), 0.1), {"actual_method": "ctgan"}

        results = {}
        for arm in ctgan_cv.METHODS:
            xa, ya, info = ctgan_cv.augment(X, y, arm, 43, sampler)
            results[arm] = (xa, ya, info)
        self.assertEqual([results[a][2]["actual_method"] for a in ctgan_cv.METHODS],
                         list(ctgan_cv.METHODS))
        self.assertEqual(results["none"][2]["after"], {"1": 7, "2": 3})
        for arm in ("bootstrap", "ctgan"):
            self.assertEqual(results[arm][2]["after"], {"1": 7, "2": 7})
            self.assertEqual(len(results[arm][1]), 14)
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0][0], X[y == "2"])
        self.assertEqual(seen[0][1], 4)

        with self.assertRaisesRegex(RuntimeError, "genuine"):
            ctgan_cv.augment(X, y, "ctgan", 43,
                            lambda real, n, s: (np.zeros((n, 3)), {"actual_method": "bootstrap"}))
        with self.assertRaisesRegex(AssertionError, "fit failed"):
            ctgan_cv.augment(X, y, "ctgan", 43,
                            lambda real, n, s: (_ for _ in ()).throw(AssertionError("fit failed")))

    def test_fold_comparison_output_and_incomplete_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            calls = []

            def sampler(minority, needed, seed):
                calls.append((len(minority), needed, seed))
                rng = np.random.default_rng(seed)
                return np.repeat(minority[:1], needed, axis=0) + rng.normal(
                    0, .01, (needed, minority.shape[1])
                ), {"actual_method": "ctgan"}

            with patch.dict(sys.modules, {"tensorflow": fake_tf()}):
                output = ctgan_cv.run(root, root / "output", "full", fake_encoder, sampler)
                saved = json.loads(output.read_text())
                self.assertEqual(saved["status"], "completed")
                self.assertEqual(saved["heldout"]["excluded_count"], 8)
                self.assertFalse(saved["heldout"]["used_for_cv"])
                self.assertEqual(len(saved["completed_folds"]), 5)
                self.assertEqual(set(saved["aggregate"]), set(ctgan_cv.METHODS))
                self.assertEqual(len(calls), 5)
                for fold in saved["completed_folds"]:
                    self.assertEqual(set(fold["conditions"]), set(ctgan_cv.METHODS))
                    self.assertEqual(sum(fold["conditions"]["ctgan"]["class_support"].values()),
                                     fold["validation_count"])
                    self.assertEqual(fold["conditions"]["none"]["augmentation"]["added"], 0)
                self.assertEqual(saved["aggregate"]["ctgan"]["actual_method"], "ctgan")

                def failed(real, needed, seed):
                    raise AssertionError("deliberate training failure")

                with self.assertRaisesRegex(AssertionError, "deliberate"):
                    ctgan_cv.run(root, root / "output", "full", fake_encoder, failed)
            reports = list((root / "output").glob("*/ctgan_comparison.json"))
            self.assertEqual(len(reports), 2)
            incomplete = next(json.loads(p.read_text()) for p in reports
                              if json.loads(p.read_text())["status"] == "incomplete")
            self.assertIsNone(incomplete["aggregate"])
            self.assertEqual(incomplete["error"]["type"], "AssertionError")
            self.assertIn("in_progress_fold", incomplete)


if __name__ == "__main__":
    unittest.main()
