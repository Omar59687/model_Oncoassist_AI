"""Checkpoint 2A: fixed-parameter baselines on the original training pool only.

The released feature columns were selected upstream by an undocumented process.
These CV estimates are conditional on those columns, not end-to-end leakage-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
CLASS_CODES = ("1", "2")
FILES = {"mGE": "mGE (1).csv", "mDM": "mDM (1).csv", "mCNA": "mCNA (1).csv"}


def load_aligned_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    tables: dict[str, pd.DataFrame] = {}
    for modality, filename in FILES.items():
        table = pd.read_csv(
            data_dir / filename, dtype={"SAMPLE_ID": str, "CLASS": str}
        )
        if "SAMPLE_ID" not in table or "CLASS" not in table:
            raise ValueError(f"{filename}: SAMPLE_ID and CLASS are required")
        if table["SAMPLE_ID"].isna().any() or table["SAMPLE_ID"].duplicated().any():
            raise ValueError(f"{filename}: missing or duplicate SAMPLE_ID")
        tables[modality] = table.set_index("SAMPLE_ID")

    common_ids = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    if not common_ids:
        raise ValueError("No shared SAMPLE_ID values")
    labels = tables["mGE"].loc[common_ids, "CLASS"]
    if labels.isna().any() or set(labels.unique()) != set(CLASS_CODES):
        raise ValueError("Expected both neutral CLASS codes 1 and 2")

    features = []
    for modality, table in tables.items():
        aligned = table.loc[common_ids]
        if not aligned["CLASS"].equals(labels):
            raise ValueError(f"{modality}: CLASS differs for aligned samples")
        values = aligned.drop(columns="CLASS")
        if values.empty:
            raise ValueError(f"{modality}: no feature columns")
        values = values.apply(pd.to_numeric, errors="raise")
        values.columns = [f"{modality}::{name}" for name in values.columns]
        features.append(values)
    X = pd.concat(features, axis=1)
    if X.columns.duplicated().any() or not np.isfinite(X.to_numpy(dtype=float)[~X.isna().to_numpy()]).all():
        raise ValueError("Duplicate columns or non-finite input values")
    return X, labels


def original_training_pool(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Recover main.py's sorted-ID, stratified 80/20 split; never evaluate on test."""
    if not X.index.equals(y.index) or not X.index.is_unique or not X.index.is_monotonic_increasing:
        raise ValueError("Sorted unique SAMPLE_ID index and aligned labels required")
    if set(y.unique()) != set(CLASS_CODES):
        raise ValueError("Cannot reproduce original split for different class codes")
    # The original script maps sorted CLASS codes to 0/1 before this split.
    binary_for_split = (y.to_numpy() == CLASS_CODES[1]).astype(int)
    train_idx, heldout_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=binary_for_split
    )
    train_ids = set(X.index[train_idx])
    heldout_ids = set(X.index[heldout_idx])
    if train_ids & heldout_ids or len(train_ids | heldout_ids) != len(y):
        raise AssertionError("Original split is not disjoint and complete")
    heldout_digest = hashlib.sha256("\n".join(sorted(heldout_ids)).encode()).hexdigest()
    return X.iloc[train_idx], y.iloc[train_idx], {
        "recovery_method": "main.py train_test_split on sorted shared IDs; seed=42, stratify=ordered binary CLASS, test_size=0.2",
        "excluded_count": len(heldout_ids),
        "excluded_id_sha256": heldout_digest,
        "used_for_cv": False,
    }


def make_models() -> dict[str, Pipeline]:
    scaled = lambda model: Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("scaler", StandardScaler()), ("model", model)]
    )
    return {
        "logistic_regression": scaled(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )),
        "svm_rbf": scaled(SVC(kernel="rbf", class_weight="balanced")),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1
            )),
        ]),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    pos_p, pos_r, pos_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[CLASS_CODES[1]], zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_CODES), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_class_code": CLASS_CODES[1],
        "positive_class": {"precision": float(pos_p[0]), "recall": float(pos_r[0]), "f1": float(pos_f[0])},
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASS_CODES)).tolist(),
        "class_support": {code: int(np.sum(y_true == code)) for code in CLASS_CODES},
    }


def evaluate(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("X and y must share unique, ordered sample IDs")
    if set(y.unique()) != set(CLASS_CODES):
        raise ValueError("Expected neutral CLASS codes 1 and 2")
    if folds < 2 or min(y.value_counts()) < folds:
        raise ValueError("Each class must have at least one member per fold")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y))
    validation_ids = []
    for train_idx, valid_idx in splits:
        if set(X.index[train_idx]) & set(X.index[valid_idx]):
            raise AssertionError("SAMPLE_ID crosses a fold boundary")
        validation_ids.extend(X.index[valid_idx])
    if len(validation_ids) != len(X) or len(set(validation_ids)) != len(X):
        raise AssertionError("Each training-pool sample must be validated once")

    outputs = {}
    for name, pipeline in make_models().items():
        oof = np.empty(len(y), dtype=object)
        aucs = []
        fold_results = []
        for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
            model = clone(pipeline)  # fresh, unfitted preprocessing and model per fold
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            score = model.decision_function(X.iloc[valid_idx]) if hasattr(model, "decision_function") else model.predict_proba(X.iloc[valid_idx])[:, list(model.classes_).index(CLASS_CODES[1])]
            if np.asarray(score).ndim != 1:
                raise ValueError("Expected one score per validation sample")
            auc = roc_auc_score((y.iloc[valid_idx].to_numpy() == CLASS_CODES[1]).astype(int), score)
            oof[valid_idx] = pred
            aucs.append(float(auc))
            fold_results.append({"fold": fold_number, "train_count": len(train_idx), "validation_count": len(valid_idx), "roc_auc": float(auc), **classification_metrics(y.iloc[valid_idx].to_numpy(), pred)})
        outputs[name] = {
            "out_of_fold": classification_metrics(y.to_numpy(), oof),
            "roc_auc_mean_across_folds": float(np.mean(aucs)),
            "roc_auc_std_across_folds": float(np.std(aucs)),
            "folds": fold_results,
        }
    return outputs


def run(data_dir: Path, output_root: Path, folds: int = 5) -> Path:
    X, y = load_aligned_data(data_dir)
    train_X, train_y, heldout = original_training_pool(X, y)
    results = {
        "checkpoint": "2A",
        "scope": "conditional on released preselected features; not end-to-end leakage-free",
        "label_semantics": "unverified; CLASS codes 1 and 2 only",
        "heldout": heldout,
        "training_pool_count": len(train_y),
        "training_pool_support": {code: int((train_y == code).sum()) for code in CLASS_CODES},
        "cross_validation": {"type": "StratifiedKFold", "folds": folds, "seed": SEED,
                             "fixed_hyperparameters_no_test_selection": True,
                             "preprocessing_fitted_within_each_fold": True},
        "models": evaluate(train_X, train_y, folds),
    }
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "baseline_cv.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
        stream.write("\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "checkpoint_2a_outputs")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(run(args.data_dir, args.output_root, args.folds))
