"""Nested CV experiment for threshold/model selection on fold-isolated latents.

The 81-sample original holdout is excluded by ``baseline_cv.original_training_pool``.
This script selects both model settings and the CLASS-code-2 probability threshold
using inner CV inside each outer training fold, then evaluates once on that outer
fold's unseen validation samples. The upstream 79-feature selection remains an
unverified limitation; results are conditional on those released features.

Run with Python 3.12 and the repository's TensorFlow/scikit-learn dependencies:
    python latent_threshold_cv.py
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from baseline_cv import CLASS_CODES, SEED, load_aligned_data, original_training_pool
from latent_cv import fold_latents

OUTER_FOLDS = 5
INNER_FOLDS = 3
CS = (0.3, 1.0, 3.0)
CLASS2_WEIGHT_MULTIPLIERS = (1.0, 1.5, 2.0)


def candidate_specs() -> list[dict]:
    """Small, fixed grid declared before evaluation; balanced baseline is included."""
    specs = []
    for model_name in ("logistic_regression", "svm_rbf"):
        for c_value in CS:
            specs.append({
                "model": model_name,
                "C": c_value,
                "class_weight": "balanced",
                "weight_label": "balanced",
            })
            for multiplier in CLASS2_WEIGHT_MULTIPLIERS:
                specs.append({
                    "model": model_name,
                    "C": c_value,
                    "class_weight": {"1": 1.0, "2": multiplier},
                    "weight_label": f"code2_x{multiplier:g}",
                })
    return specs


def make_estimator(spec: dict):
    weights = spec["class_weight"]
    if isinstance(weights, dict):
        weights = {str(k): float(v) for k, v in weights.items()}
    if spec["model"] == "logistic_regression":
        return LogisticRegression(
            C=float(spec["C"]), class_weight=weights, max_iter=2000,
            solver="liblinear", random_state=SEED,
        )
    return SVC(
        C=float(spec["C"]), kernel="rbf", class_weight=weights,
        probability=True, random_state=SEED,
    )


def code2_probability(model, features: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if CLASS_CODES[1] not in classes:
        raise ValueError("Fitted classifier did not retain CLASS code 2")
    return model.predict_proba(features)[:, classes.index(CLASS_CODES[1])]


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = np.where(y_score >= threshold, CLASS_CODES[1], CLASS_CODES[0])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_CODES), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class_1": float(precision[0]),
        "precision_class_2": float(precision[1]),
        "recall_class_1": float(recall[0]),
        "recall_class_2": float(recall[1]),
        "f1_class_1": float(f1[0]),
        "f1_class_2": float(f1[1]),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_CODES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score((y_true == CLASS_CODES[1]).astype(int), y_score)),
        "confusion_matrix_rows_true_columns_pred_class_1_then_2": confusion_matrix(
            y_true, y_pred, labels=list(CLASS_CODES)
        ).tolist(),
        "support_class_1": int(support[0]),
        "support_class_2": int(support[1]),
        "threshold_probability_class_2": float(threshold),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Maximize inner-OOF class-2 F1; deterministic ties prefer precision then .5."""
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique)))
    ranked = []
    for threshold in candidates:
        pred = np.where(scores >= threshold, CLASS_CODES[1], CLASS_CODES[0])
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, labels=[CLASS_CODES[1]], zero_division=0,
        )
        ranked.append((float(f[0]), float(p[0]), -abs(float(threshold) - 0.5), float(threshold)))
    best = max(ranked)
    return best[3], best[0]


def nested_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    specs = candidate_specs()
    oof_scores = np.full(len(y), np.nan, dtype=float)
    oof_thresholds = np.full(len(y), np.nan, dtype=float)
    fold_results = []
    seen_validation = set()

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(X, y), start=1):
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        X_valid, y_valid = X.iloc[outer_valid_idx], y.iloc[outer_valid_idx]
        valid_ids = set(X_valid.index)
        if valid_ids & (set(X_train.index) | seen_validation):
            raise AssertionError("Outer validation SAMPLE_ID overlaps another partition")
        seen_validation.update(valid_ids)

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + outer_fold,
        )
        candidate_oof = {i: np.full(len(y_train), np.nan, dtype=float) for i in range(len(specs))}
        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner.split(X_train, y_train), start=1):
            X_inner_train = X_train.iloc[inner_train_idx]
            y_inner_train = y_train.iloc[inner_train_idx]
            X_inner_valid = X_train.iloc[inner_valid_idx]
            if set(X_inner_train.index) & set(X_inner_valid.index):
                raise AssertionError("Inner train/validation SAMPLE_ID overlap")
            train_z, valid_z = fold_latents(
                X_inner_train, y_inner_train, X_inner_valid,
                fold=outer_fold * 100 + inner_fold,
            )
            for candidate_index, spec in enumerate(specs):
                model = make_estimator(spec)
                model.fit(train_z, y_inner_train)
                candidate_oof[candidate_index][inner_valid_idx] = code2_probability(model, valid_z)

        choices = []
        y_train_values = y_train.to_numpy()
        for candidate_index, spec in enumerate(specs):
            scores = candidate_oof[candidate_index]
            if not np.isfinite(scores).all():
                raise AssertionError("Inner OOF scores are incomplete")
            threshold, inner_f1 = choose_threshold(y_train_values, scores)
            inner_metrics = metrics(y_train_values, scores, threshold)
            # Primary objective is inner OOF code-2 F1; macro F1 and precision
            # are deterministic tie-breakers, followed by proximity to baseline.
            ranking = (
                inner_f1, inner_metrics["macro_f1"],
                inner_metrics["precision_class_2"],
                -abs(threshold - 0.5), -spec["C"],
            )
            choices.append((ranking, candidate_index, threshold, inner_metrics))
        _, chosen_index, chosen_threshold, selected_inner_metrics = max(choices, key=lambda item: item[0])
        selected_spec = specs[chosen_index]

        train_z, valid_z = fold_latents(
            X_train, y_train, X_valid, fold=outer_fold,
        )
        final_model = make_estimator(selected_spec)
        final_model.fit(train_z, y_train)
        valid_scores = code2_probability(final_model, valid_z)
        oof_scores[outer_valid_idx] = valid_scores
        oof_thresholds[outer_valid_idx] = chosen_threshold
        fold_metrics = metrics(y_valid.to_numpy(), valid_scores, chosen_threshold)
        fold_results.append({
            "fold": outer_fold,
            "train_count": len(outer_train_idx),
            "validation_count": len(outer_valid_idx),
            "selected_model": selected_spec,
            "selected_threshold": float(chosen_threshold),
            "inner_training_oof_metrics_at_selected_threshold": selected_inner_metrics,
            "outer_validation_metrics": fold_metrics,
        })

    if len(seen_validation) != len(X) or not np.isfinite(oof_scores).all():
        raise AssertionError("Each training-pool sample must be outer-validated exactly once")
    pooled = metrics(y.to_numpy(), oof_scores, 0.5)
    # Each outer fold has its own training-only threshold; apply those fold-wise
    # for the pooled nested estimate rather than fitting one threshold globally.
    pooled = metrics(y.to_numpy(), oof_scores, 0.5)
    foldwise_predictions = np.where(oof_scores >= oof_thresholds, CLASS_CODES[1], CLASS_CODES[0])
    p, r, f, support = precision_recall_fscore_support(
        y.to_numpy(), foldwise_predictions, labels=list(CLASS_CODES), zero_division=0,
    )
    pooled_thresholded = {
        "accuracy": float(accuracy_score(y, foldwise_predictions)),
        "precision_class_1": float(p[0]), "precision_class_2": float(p[1]),
        "recall_class_1": float(r[0]), "recall_class_2": float(r[1]),
        "f1_class_1": float(f[0]), "f1_class_2": float(f[1]),
        "macro_f1": float(f1_score(y, foldwise_predictions, labels=list(CLASS_CODES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, foldwise_predictions)),
        "roc_auc_pooled_oof": float(roc_auc_score((y.to_numpy() == CLASS_CODES[1]).astype(int), oof_scores)),
        "confusion_matrix_rows_true_columns_pred_class_1_then_2": confusion_matrix(
            y.to_numpy(), foldwise_predictions, labels=list(CLASS_CODES)
        ).tolist(),
        "support_class_1": int(support[0]), "support_class_2": int(support[1]),
        "threshold_note": "Per-sample threshold comes from the corresponding outer fold's inner CV; no outer label is used to choose it.",
    }
    numeric_keys = [
        "accuracy", "precision_class_1", "precision_class_2", "recall_class_1",
        "recall_class_2", "f1_class_1", "f1_class_2", "macro_f1",
        "balanced_accuracy", "roc_auc", "threshold_probability_class_2",
    ]
    variability = {}
    for key in numeric_keys:
        vals = [fold["outer_validation_metrics"][key] for fold in fold_results]
        variability[key] = {"mean": float(np.mean(vals)), "std_population": float(np.std(vals)),
                            "min": float(np.min(vals)), "max": float(np.max(vals))}
    return {
        "outer_cv_metrics_using_fold_specific_selected_thresholds": pooled_thresholded,
        "outer_fold_metrics_at_threshold_0_5_for_diagnostic_only": pooled,
        "fold_variability_mean_std_min_max": variability,
        "folds": fold_results,
    }


def package_versions() -> dict:
    found = {"python": platform.python_version()}
    for label, package in (("numpy", "numpy"), ("pandas", "pandas"),
                           ("scikit_learn", "scikit-learn"), ("tensorflow", "tensorflow")):
        try:
            found[label] = version(package)
        except PackageNotFoundError:
            found[label] = None
    return found


def render_summary(result: dict) -> str:
    status = result["status"]
    lines = [
        "# Nested latent threshold and classifier experiment",
        "",
        f"**Status:** {status}",
        "",
        "## Protocol and safeguards",
        "",
        "- Reconstructs the original stratified 80/20 split (seed 42); 81 holdout samples are excluded from all model fitting, threshold selection, and evaluation.",
        "- Uses five stratified outer folds and three inner folds. Each encoder, imputer, scaler, classifier, hyperparameter choice, and threshold is learned inside the relevant training partition.",
        "- Threshold and candidate settings are chosen by inner out-of-fold Class-2 F1; final metrics come only from the untouched outer validation fold.",
        "- All labels are reported as neutral CLASS codes. Clinical meaning is unverified. Results remain conditional on the 79 released features because upstream feature-selection timing is unknown.",
        "- Repository audit also documents a 404-versus-204 cohort discrepancy against the paper, uncertain supervised-feature-selection provenance, and no external validation. These prevent clinical interpretation or a claim of established generalization.",
        "",
    ]
    if status != "completed":
        lines += ["## Execution outcome", "", result.get("execution_note", "No experiment metrics were produced."), "",
                  "No new performance estimate is claimed. Install the documented dependencies and rerun the PowerShell command in the report.", ""]
    else:
        metric = result["experiment"]["outer_cv_metrics_using_fold_specific_selected_thresholds"]
        lines += [
            "## Nested outer-CV result",
            "",
            f"- Accuracy: {metric['accuracy']:.3f}; Class-2 precision: {metric['precision_class_2']:.3f}; Class-2 recall: {metric['recall_class_2']:.3f}; Class-2 F1: {metric['f1_class_2']:.3f}.",
            f"- Class-1 precision/recall/F1: {metric['precision_class_1']:.3f}/{metric['recall_class_1']:.3f}/{metric['f1_class_1']:.3f}; macro F1: {metric['macro_f1']:.3f}; balanced accuracy: {metric['balanced_accuracy']:.3f}; pooled OOF ROC-AUC: {metric['roc_auc_pooled_oof']:.3f}.",
            f"- Confusion matrix, rows=true and columns=predicted in order [1, 2]: `{metric['confusion_matrix_rows_true_columns_pred_class_1_then_2']}`.",
            "- Fold-specific thresholds and complete fold metrics are in `latent_threshold_cv.json`.",
            "",
            "## Interpretation",
            "",
            "Treat changes versus prior CV as uncertain exploratory evidence. Fold-specific selection, small minority support, and uncertain feature provenance preclude a claim of validated improvement. The holdout remains unopened for this experiment.",
            "",
        ]
    lines += [
        "## Existing reported benchmarks (provided results)",
        "",
        "| Condition | Accuracy | Class-2 precision | Class-2 recall | Class-2 F1 | Mean fold ROC-AUC | Protocol comparability |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| No augmentation, latent logistic (user-confirmed) | 73.68% | 50.45% | 65.12% | 56.85% | 0.7846 | Same 323-sample/5-fold family; new nested selection protocol differs |",
        "| CTGAN, latent logistic (user-confirmed) | 78.02% | 61.19% | 47.67% | 53.59% | 0.8042 | Prior fixed-parameter evaluation; not directly comparable to nested tuned result |",
        "| CTGAN, archived checkpoint 3A | 76.78% | 57.75% | 47.67% | 52.23% | 0.7903 | Saved same-fold checkpoint; differs from current user-confirmed CTGAN result |",
        "",
        "The current CTGAN confusion matrix supplied by the user is `[[211, 26], [45, 41]]`. The branch's archived `checkpoint_3a_outputs/20261025T120811006346Z/ctgan_comparison.json` instead reports `[[207, 30], [45, 41]]`, 76.78% accuracy, and mean fold ROC-AUC 0.790326 (SD 0.020227). Treat these as different result records; the current result's run provenance/fold metrics are not present in the repository. Neither result uses verified clinical label names. These comparisons must not be read as external validation.",
        "",
        "### Full metrics recoverable from the supplied OOF results",
        "",
        "| Condition | Class-1 P/R/F1 | Class-2 P/R/F1 | Macro F1 | Balanced accuracy | Confusion matrix [1,2] | Fold variability available here |",
        "|---|---|---|---:|---:|---|---|",
        "| No augmentation (user-confirmed OOF) | 0.8585 / 0.7679 / 0.8107 | 0.5045 / 0.6512 / 0.5685 | 0.6896 | 0.7095 | `[[182,55],[30,56]]` | Saved checkpoint has per-fold metrics; AUC there is 0.784880 ± 0.025710; see `checkpoint_2b_outputs/20261025T113837678917Z/latent_cv.json` |",
        "| CTGAN (user-confirmed record) | 0.8242 / 0.8903 / 0.8570 | 0.6119 / 0.4767 / 0.5359 | 0.6965 | 0.6835 | `[[211,26],[45,41]]` | Not supplied for this record; do not infer fold SD from it |",
        "| CTGAN (archived checkpoint 3A) | 0.8214 / 0.8734 / 0.8466 | 0.5775 / 0.4767 / 0.5223 | 0.6845 | 0.6751 | `[[207,30],[45,41]]` | AUC 0.790326 ± 0.020227; per-fold records are in the checkpoint JSON |",
        "",
        "No-augmentation class-1 metrics and confusion matrix are recovered from the saved checkpoint-2B OOF output; the user-confirmed AUC 0.7846 differs slightly from that artifact's 0.784880. User-confirmed CTGAN classwise metrics, macro F1, and balanced accuracy are derived directly from the supplied confusion matrix; its reported ROC-AUC is the user-supplied mean across folds, with no matching fold-level record found in the repository.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "py -3.12 -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install --upgrade pip",
        "python -m pip install numpy==2.3.5 pandas==2.2.3 scikit-learn==1.8.0 tensorflow==2.21.0 pytest",
        "python -m pytest -q",
        "$env:TF_ENABLE_ONEDNN_OPTS = '0'",
        "$env:TF_NUM_INTRAOP_THREADS = '2'",
        "$env:TF_NUM_INTEROP_THREADS = '2'",
        "python .\\latent_threshold_cv.py --data-dir . --output-root .\\latent_threshold_outputs",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(data_dir: Path, output_root: Path) -> Path:
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    base = {
        "experiment": "nested latent classifier and threshold selection",
        "label_semantics": "unverified; CLASS codes 1 and 2 only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": package_versions(),
        "protocol": {"outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
                     "seed": SEED, "selection_metric": "inner OOF Class-2 F1",
                     "threshold_selection": "max inner OOF Class-2 F1; ties by precision, nearest 0.5",
                     "feature_selection": "not repeated; released 79-feature provenance is uncertain",
                     "holdout_use": "excluded from model fitting, selection and evaluation"},
    }
    X, y = load_aligned_data(data_dir)
    train_X, train_y, heldout = original_training_pool(X, y)
    if len(train_y) != 323 or train_y.value_counts().to_dict() != {"1": 237, "2": 86}:
        raise ValueError("Recovered training pool differs from confirmed counts; stopping before model fitting")
    base.update({
        "aligned_sample_count": len(X),
        "feature_count": X.shape[1],
        "heldout_exclusion": {"excluded_count": heldout["excluded_count"],
                              "excluded_id_sha256": heldout["excluded_id_sha256"],
                              "used_for_model_selection_or_evaluation": False},
        "training_pool_count": len(train_y),
        "training_pool_support": {c: int((train_y == c).sum()) for c in CLASS_CODES},
    })
    try:
        import tensorflow as tf
    except ImportError:
        base.update({"status": "blocked_missing_dependency",
                     "execution_note": "TensorFlow is not installed in the execution environment; the fold-isolated autoencoder experiment could not be run. No new performance metrics were generated."})
    else:
        tf.keras.utils.set_random_seed(SEED)
        try:
            tf.config.experimental.enable_op_determinism()
            base["tensorflow_determinism"] = "op determinism requested"
        except (AttributeError, RuntimeError) as exc:
            base["tensorflow_determinism"] = f"unavailable: {type(exc).__name__}"
        base.update({
            "status": "completed",
            "experiment": nested_evaluate(train_X, train_y),
        })
    (output_dir / "latent_threshold_cv.json").write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    (output_dir / "RESEARCH_SUMMARY.md").write_text(render_summary(base), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "latent_threshold_outputs")
    args = parser.parse_args()
    print(run(args.data_dir, args.output_root))
