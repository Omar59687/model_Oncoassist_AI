"""Focused tests using only Python unittest and small in-memory CSV fixtures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import baseline_cv


class RecordingImputer(SimpleImputer):
    seen_ids = []

    def fit(self, X, y=None):
        self.seen_ids.append(set(X.index))
        return super().fit(X, y)


class RecordingScaler(StandardScaler):
    fit_maxima = []

    def fit(self, X, y=None, sample_weight=None):
        self.fit_maxima.append(float(np.max(X[:, 0])))
        return super().fit(X, y, sample_weight=sample_weight)


def small_dataset(n=30):
    ids = [f"sample-{i:03d}" for i in range(n)]
    X = pd.DataFrame({"mGE::f": np.arange(n, dtype=float),
                      "mDM::f": np.arange(n, dtype=float) % 3}, index=ids)
    y = pd.Series(["1", "2"] * (n // 2), index=ids)
    return X, y


def one_model():
    return {"logistic": Pipeline([
        ("imputer", RecordingImputer(strategy="median")),
        ("scaler", RecordingScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=42)),
    ])}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        RecordingImputer.seen_ids = []
        RecordingScaler.fit_maxima = []

    def test_alignment_and_target_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            X, y = small_dataset()
            ids = X.index.tolist()
            for modality, filename in baseline_cv.FILES.items():
                rows = pd.DataFrame({"SAMPLE_ID": ids,
                                     "gene": np.arange(len(ids), dtype=float),
                                     "CLASS": y.to_numpy()})
                if modality == "mDM":
                    rows = rows.iloc[::-1]
                rows.to_csv(root / filename, index=False)
            loaded_X, loaded_y = baseline_cv.load_aligned_data(root)
            self.assertTrue(loaded_X.index.equals(loaded_y.index))
            self.assertEqual(loaded_X.shape, (30, 3))
            self.assertFalse(any("CLASS" in c or "SAMPLE_ID" in c for c in loaded_X))
            self.assertEqual(list(loaded_y), list(y))
            self.assertEqual(float(loaded_X.loc[ids[0], "mDM::gene"]), 0.0)

            bad = pd.read_csv(root / baseline_cv.FILES["mDM"])
            bad.loc[0, "CLASS"] = 1 if bad.loc[0, "CLASS"] == 2 else 2
            bad.to_csv(root / baseline_cv.FILES["mDM"], index=False)
            with self.assertRaisesRegex(ValueError, "CLASS differs"):
                baseline_cv.load_aligned_data(root)

    def test_original_test_membership_excluded(self):
        X, y = small_dataset()
        train_X, train_y, info = baseline_cv.original_training_pool(X, y)
        expected, excluded = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42,
            stratify=(y.to_numpy() == "2").astype(int)
        )
        self.assertEqual(list(train_X.index), list(X.index[expected]))
        self.assertEqual(list(train_y.index), list(train_X.index))
        self.assertTrue(set(train_X.index).isdisjoint(X.index[excluded]))
        self.assertEqual(info["excluded_count"], len(excluded))
        self.assertFalse(info["used_for_cv"])

    def test_fold_isolation_and_preprocessing(self):
        X, y = small_dataset()
        X.iloc[0, 0] = 1e9  # this sample must not enter the validation fold's fit
        with patch.object(baseline_cv, "make_models", side_effect=one_model):
            output = baseline_cv.evaluate(X, y, folds=3)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        expected_train = [set(X.index[tr]) for tr, _ in cv.split(X, y)]
        self.assertEqual(RecordingImputer.seen_ids, expected_train)
        for (train_idx, valid_idx), seen, maximum in zip(
            cv.split(X, y), RecordingImputer.seen_ids, RecordingScaler.fit_maxima
        ):
            self.assertFalse(seen & set(X.index[valid_idx]))
            if 0 in valid_idx:
                self.assertLess(maximum, 30.0)
            else:
                self.assertGreater(maximum, 1e8)
        self.assertEqual(output["logistic"]["out_of_fold"]["class_support"], {"1": 15, "2": 15})

    def test_repeatable_metrics_and_no_overwrite(self):
        X, y = small_dataset()
        with patch.object(baseline_cv, "make_models", side_effect=one_model):
            first = baseline_cv.evaluate(X, y, folds=3)
            second = baseline_cv.evaluate(X, y, folds=3)
        self.assertEqual(first, second)
        self.assertIn("roc_auc_mean_across_folds", first["logistic"])
        self.assertEqual(sum(map(sum, first["logistic"]["out_of_fold"]["confusion_matrix"])), 30)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for modality, filename in baseline_cv.FILES.items():
                pd.DataFrame({"SAMPLE_ID": X.index, "feature": X.iloc[:, 0],
                              "CLASS": y.to_numpy()}).to_csv(root / filename, index=False)
            with patch.object(baseline_cv, "make_models", side_effect=one_model):
                a = baseline_cv.run(root, root / "outputs", folds=3)
                b = baseline_cv.run(root, root / "outputs", folds=3)
            self.assertNotEqual(a, b)
            self.assertEqual(json.loads(a.read_text()), json.loads(b.read_text()))
            self.assertEqual(json.loads(a.read_text())["scope"],
                             "conditional on released preselected features; not end-to-end leakage-free")


if __name__ == "__main__":
    unittest.main()
