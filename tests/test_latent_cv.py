"""Fast fold-isolation tests; no TensorFlow or model training required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import latent_cv
from baseline_cv import CLASS_CODES, SEED


class FakeEncoder:
    def __init__(self, latent_dim, marker):
        self.latent_dim = latent_dim
        self.marker = marker

    def predict(self, X, verbose=0):
        output = np.zeros((len(X), self.latent_dim), dtype=float)
        output[:, 0] = X[:, 0] + self.marker
        return output


class EncoderSpy:
    def __init__(self):
        self.calls = []
        self.encoders = []

    def __call__(self, X_fit, X_stop, latent_dim, seed):
        self.calls.append((X_fit.copy(), X_stop.copy(), latent_dim, seed))
        encoder = FakeEncoder(latent_dim, len(self.calls))
        self.encoders.append(encoder)
        return encoder


def fixture(n=30):
    ids = [f"sample-{i:03d}" for i in range(n)]
    X = pd.DataFrame({
        "mGE::gene": np.arange(n, dtype=float),
        "mDM::site": np.arange(n, dtype=float) * 2,
        "mCNA::region": np.arange(n, dtype=float) % 5,
    }, index=ids)
    y = pd.Series(["1", "2"] * (n // 2), index=ids)
    return X, y


class LatentCVTests(unittest.TestCase):
    def test_modality_alignment_and_shape(self):
        X, y = fixture()
        spy = EncoderSpy()
        train_z, valid_z = latent_cv.fold_latents(
            X.iloc[:24], y.iloc[:24], X.iloc[24:], 1, spy
        )
        self.assertEqual(train_z.shape, (24, 3))
        self.assertEqual(valid_z.shape, (6, 3))
        self.assertEqual([c[2] for c in spy.calls], [1, 1, 1])
        self.assertEqual([c[3] for c in spy.calls], [142, 143, 144])
        # First validation value in each modality is transformed by its own scaler.
        for i, modality in enumerate(("mGE::gene", "mDM::site", "mCNA::region")):
            col = X.loc[X.index[24], modality]
            training = X.loc[X.index[:24], modality]
            expected = (col - training.mean()) / training.std(ddof=0) + i + 1
            self.assertAlmostEqual(valid_z[0, i], expected, places=5)

    def test_target_and_overlap_rejected(self):
        X, y = fixture()
        spy = EncoderSpy()
        with self.assertRaisesRegex(ValueError, "overlap"):
            latent_cv.fold_latents(X.iloc[:24], y.iloc[:24], X.iloc[23:], 1, spy)
        X["mGE::CLASS"] = y.to_numpy()
        with self.assertRaisesRegex(ValueError, "Target"):
            latent_cv.fold_latents(X.iloc[:24], y.iloc[:24], X.iloc[24:], 1, spy)
        self.assertEqual(len(spy.calls), 0)

    def test_validation_outlier_not_seen_by_encoders(self):
        X, y = fixture()
        X.loc[X.index[0], "mGE::gene"] = 1e9
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
        fold_number, (train_idx, valid_idx) = next(
            (i, pair) for i, pair in enumerate(folds, start=1) if 0 in pair[1]
        )
        spy = EncoderSpy()
        latent_cv.fold_latents(
            X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], fold_number, spy
        )
        self.assertEqual(len(spy.calls), 3)
        self.assertLess(max(np.max(np.abs(c[0])) for c in spy.calls), 10)
        self.assertLess(max(np.max(np.abs(c[1])) for c in spy.calls), 10)

    def test_fresh_encoders_per_fold_and_metric_schema(self):
        X, y = fixture()
        spy = EncoderSpy()
        result = latent_cv.evaluate_latent_cv(X, y, folds=5, fit_encoder=spy)
        self.assertEqual(len(spy.encoders), 15)
        self.assertEqual(len({id(model) for model in spy.encoders}), 15)
        self.assertEqual([f["validation_count"] for f in result["folds"]], [6] * 5)
        combined = np.sum([f["confusion_matrix"] for f in result["folds"]], axis=0)
        self.assertEqual(combined.tolist(), result["out_of_fold"]["confusion_matrix"])
        self.assertEqual(result["out_of_fold"]["class_support"], {"1": 15, "2": 15})
        self.assertEqual(result["out_of_fold"]["positive_class_code"], CLASS_CODES[1])
        self.assertEqual(result["roc_auc_mean_across_folds"],
                         float(np.mean([f["roc_auc"] for f in result["folds"]])))
        self.assertTrue(0 <= result["out_of_fold"]["accuracy"] <= 1)
        tn, fp = combined[0]
        fn, tp = combined[1]
        self.assertAlmostEqual(result["out_of_fold"]["positive_class"]["recall"], tp / (tp + fn))
        self.assertAlmostEqual(result["out_of_fold"]["positive_class"]["precision"], tp / (tp + fp))

    def test_saved_output_schema_and_unique_run_directory(self):
        X, y = fixture()
        fake_tf = SimpleNamespace(
            keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=lambda seed: None)),
            config=SimpleNamespace(experimental=SimpleNamespace(enable_op_determinism=lambda: None)),
        )
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, {"tensorflow": fake_tf}), \
                 patch.object(latent_cv, "load_aligned_data", return_value=(X, y)), \
                 patch.object(latent_cv, "evaluate_latent_cv", return_value={"folds": []}):
                first = latent_cv.run(Path(temp), Path(temp) / "output")
                second = latent_cv.run(Path(temp), Path(temp) / "output")
            self.assertNotEqual(first, second)
            saved = json.loads(first.read_text())
            self.assertEqual(saved["training_pool_count"], 24)
            self.assertEqual(saved["heldout"]["excluded_count"], 6)
            self.assertFalse(saved["heldout"]["used_for_cv"])
            self.assertEqual(saved["cross_validation"]["folds"], 5)
            self.assertEqual(saved["architecture"]["latent_capacities"], {"mGE": 16, "mDM": 16, "mCNA": 8})
            self.assertIsNone(saved["architecture"]["augmentation"])
            self.assertEqual(saved["label_semantics"], "unverified; positive means CLASS code 2 only")


if __name__ == "__main__":
    unittest.main()
