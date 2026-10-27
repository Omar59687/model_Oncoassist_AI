"""Nested-CV comparison of AE + tuned RBF SVM with and without CTGAN.

Each outer fold has an inner three-fold search over a small fixed RBF-SVM grid,
Class-2 thresholds, and the augmentation arm. Autoencoders, CTGAN, and classifiers
are fitted only on the applicable inner/outer training partition. Outer validation
folds are used once for evaluation. The original 81-sample holdout is excluded.

CTGAN runs in the existing isolated CPU worker. The default 20 epochs are a
runtime-conscious exploratory setting, not a claim of optimal CTGAN convergence.

Example (Python 3.12):
    python ctgan_svm_nested_cv.py --data-dir . --output-root ctgan_svm_nested_outputs
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from baseline_cv import CLASS_CODES, SEED, load_aligned_data, original_training_pool
from ctgan_cv import PAC, augment, ctgan_batch_size, generate_ctgan
from latent_cv import fold_latents
from latent_threshold_cv import choose_threshold, metrics as classification_metrics

OUTER_FOLDS = 5
INNER_FOLDS = 3
CTGAN_EPOCHS = 20
# Eight predeclared candidates per augmentation arm: C x gamma x class weighting.
SVM_GRID = tuple(
    {"C": c_value, "gamma": gamma, "class_weight": weights, "weight_label": weight_label}
    for c_value in (0.5, 2.0)
    for gamma in ("scale", 0.01)
    for weights, weight_label in (
        ("balanced", "balanced"),
        ({"1": 1.0, "2": 1.5}, "code2_x1.5"),
    )
)
ARMS = ("none", "ctgan")


def package_versions() -> dict:
    result = {"python": platform.python_version()}
    for label, distribution in (
        ("numpy", "numpy"), ("pandas", "pandas"),
        ("scikit_learn", "scikit-learn"), ("tensorflow", "tensorflow"),
        ("sdv", "sdv"), ("ctgan", "ctgan"), ("torch", "torch"),
    ):
        try:
            result[label] = version(distribution)
        except PackageNotFoundError:
            result[label] = None
    return result


def make_svm(spec: dict, seed: int) -> SVC:
    return SVC(
        C=float(spec["C"]), kernel="rbf", gamma=spec["gamma"],
        class_weight=spec["class_weight"], probability=True, random_state=seed,
    )


def code2_scores(model: SVC, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if CLASS_CODES[1] not in classes:
        raise ValueError("RBF SVM did not retain CLASS code 2")
    scores = model.predict_proba(X)[:, classes.index(CLASS_CODES[1])]
    if not np.isfinite(scores).all():
        raise ValueError("SVM returned non-finite Class-2 probabilities")
    return scores


def choose_candidate(y_true: np.ndarray, candidate_oof: dict[int, np.ndarray], specs: tuple[dict, ...]) -> dict:
    """Select an arm-specific SVM and threshold from inner OOF predictions only."""
    ranked = []
    for candidate_id, spec in enumerate(specs):
        scores = candidate_oof[candidate_id]
        if not np.isfinite(scores).all():
            raise AssertionError("Inner OOF probability vector is incomplete")
        threshold, inner_f1 = choose_threshold(y_true, scores)
        inner_metrics = classification_metrics(y_true, scores, threshold)
        ranking = (
            inner_f1,
            inner_metrics["macro_f1"],
            inner_metrics["precision_class_2"],
            -abs(threshold - 0.5),
            -float(spec["C"]),
            int(spec["gamma"] == "scale"),
        )
        ranked.append((ranking, candidate_id, threshold, inner_metrics))
    _, candidate_id, threshold, inner_metrics = max(ranked, key=lambda row: row[0])
    return {
        "candidate_id": candidate_id,
        "parameters": specs[candidate_id],
        "threshold": float(threshold),
        "inner_oof_metrics_at_selected_threshold": inner_metrics,
    }


def _fold_variability(fold_metrics: list[dict]) -> dict:
    keys = (
        "accuracy", "precision_class_1", "precision_class_2", "recall_class_1",
        "recall_class_2", "f1_class_1", "f1_class_2", "macro_f1",
        "balanced_accuracy", "roc_auc", "threshold_probability_class_2",
    )
    summary = {}
    for key in keys:
        values = np.asarray([item[key] for item in fold_metrics], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std_population": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def _pooled_metrics(y: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict:
    if not (len(y) == len(scores) == len(thresholds)) or not np.isfinite(scores).all() or not np.isfinite(thresholds).all():
        raise AssertionError("Incomplete or misaligned outer-fold predictions")
    predictions = np.where(scores >= thresholds, CLASS_CODES[1], CLASS_CODES[0])
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predictions, labels=list(CLASS_CODES), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_class_1": float(precision[0]), "precision_class_2": float(precision[1]),
        "recall_class_1": float(recall[0]), "recall_class_2": float(recall[1]),
        "f1_class_1": float(f1[0]), "f1_class_2": float(f1[1]),
        "macro_f1": float(f1_score(y, predictions, labels=list(CLASS_CODES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "roc_auc_pooled_oof": float(roc_auc_score((y == CLASS_CODES[1]).astype(int), scores)),
        "confusion_matrix_rows_true_columns_pred_class_1_then_2": confusion_matrix(
            y, predictions, labels=list(CLASS_CODES),
        ).tolist(),
        "support_class_1": int(support[0]), "support_class_2": int(support[1]),
        "threshold_note": "Each sample uses the threshold selected inside its corresponding outer training fold.",
    }


def nested_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    fit_latents: Callable = fold_latents,
    ctgan_sampler: Callable = generate_ctgan,
    ctgan_epochs: int = CTGAN_EPOCHS,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    specs: tuple[dict, ...] = SVM_GRID,
) -> dict:
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Feature and label SAMPLE_ID indices must be aligned and unique")
    if set(y.unique()) != set(CLASS_CODES):
        raise ValueError("Expected neutral CLASS codes 1 and 2")
    if min(y.value_counts()) < outer_folds:
        raise ValueError("Each class needs at least one sample in every outer fold")

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=SEED)
    oof_scores = {arm: np.full(len(y), np.nan, dtype=float) for arm in ARMS}
    oof_thresholds = {arm: np.full(len(y), np.nan, dtype=float) for arm in ARMS}
    adaptive_scores = np.full(len(y), np.nan, dtype=float)
    adaptive_thresholds = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    seen_valid_ids: set[str] = set()

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[outer_train_idx], y.iloc[outer_train_idx]
        X_valid, y_valid = X.iloc[outer_valid_idx], y.iloc[outer_valid_idx]
        valid_ids = set(X_valid.index)
        if valid_ids & (set(X_train.index) | seen_valid_ids):
            raise AssertionError("Outer validation IDs overlap training or a prior outer validation fold")
        seen_valid_ids.update(valid_ids)

        inner_cv = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=SEED + outer_fold,
        )
        inner_oof = {
            arm: {candidate_id: np.full(len(y_train), np.nan, dtype=float)
                  for candidate_id in range(len(specs))}
            for arm in ARMS
        }
        inner_generation_records = []

        for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
            X_inner_train, y_inner_train = X_train.iloc[inner_train_idx], y_train.iloc[inner_train_idx]
            X_inner_valid = X_train.iloc[inner_valid_idx]
            if set(X_inner_train.index) & set(X_inner_valid.index):
                raise AssertionError("Inner training and validation IDs overlap")

            z_inner_train, z_inner_valid = fit_latents(
                X_inner_train, y_inner_train, X_inner_valid,
                fold=outer_fold * 100 + inner_fold,
            )
            split_seed = SEED + outer_fold * 100 + inner_fold
            # CTGAN sees only latent vectors from this inner training split.
            augmented_z, augmented_y, augmentation_info = augment(
                z_inner_train, y_inner_train.to_numpy(), "ctgan", split_seed,
                ctgan_sampler=lambda minority, needed, seed: ctgan_sampler(
                    minority, needed, seed, epochs=ctgan_epochs,
                ),
            )
            if augmentation_info.get("actual_method") != "ctgan":
                raise RuntimeError("A genuine CTGAN sample set was not generated; refusing fallback")
            if augmentation_info.get("after", {}).get(CLASS_CODES[1], 0) != augmentation_info.get("after", {}).get(CLASS_CODES[0], 0):
                raise AssertionError("CTGAN training set was not balanced to the majority count")
            inner_generation_records.append({
                "inner_fold": inner_fold,
                "training_count_before": len(y_inner_train),
                "validation_count_excluded_from_generation": len(inner_valid_idx),
                "augmentation": augmentation_info,
            })

            for candidate_id, spec in enumerate(specs):
                for arm, features, labels in (
                    ("none", z_inner_train, y_inner_train.to_numpy()),
                    ("ctgan", augmented_z, augmented_y),
                ):
                    model = make_svm(spec, seed=split_seed + candidate_id)
                    model.fit(features, labels)
                    inner_oof[arm][candidate_id][inner_valid_idx] = code2_scores(model, z_inner_valid)

        y_train_values = y_train.to_numpy()
        selected = {
            arm: choose_candidate(y_train_values, inner_oof[arm], specs)
            for arm in ARMS
        }
        # The adaptive-policy arm is selected using inner OOF F1 only. Exact ties
        # favor no augmentation to avoid added complexity without evidence of gain.
        arm_rank = {
            arm: (
                selected[arm]["inner_oof_metrics_at_selected_threshold"]["f1_class_2"],
                selected[arm]["inner_oof_metrics_at_selected_threshold"]["macro_f1"],
                selected[arm]["inner_oof_metrics_at_selected_threshold"]["precision_class_2"],
                int(arm == "none"),
            ) for arm in ARMS
        }
        selected_arm = max(ARMS, key=lambda arm: arm_rank[arm])

        # Refit the encoders on the complete outer training partition. Outer
        # validation covariates are transformed only by these fitted encoders.
        z_outer_train, z_outer_valid = fit_latents(
            X_train, y_train, X_valid, fold=outer_fold,
        )
        outer_augmented_z, outer_augmented_y, outer_augmentation_info = augment(
            z_outer_train, y_train.to_numpy(), "ctgan", SEED + outer_fold,
            ctgan_sampler=lambda minority, needed, seed: ctgan_sampler(
                minority, needed, seed, epochs=ctgan_epochs,
            ),
        )
        if outer_augmentation_info.get("actual_method") != "ctgan":
            raise RuntimeError("Outer-fold CTGAN training failed genuine-generation check")

        fold_metrics = {}
        fold_scores = {}
        for arm in ARMS:
            spec = selected[arm]["parameters"]
            features, labels = (
                (z_outer_train, y_train.to_numpy()) if arm == "none"
                else (outer_augmented_z, outer_augmented_y)
            )
            final_model = make_svm(spec, seed=SEED + outer_fold)
            final_model.fit(features, labels)
            scores = code2_scores(final_model, z_outer_valid)
            threshold = selected[arm]["threshold"]
            oof_scores[arm][outer_valid_idx] = scores
            oof_thresholds[arm][outer_valid_idx] = threshold
            fold_scores[arm] = scores
            fold_metrics[arm] = classification_metrics(y_valid.to_numpy(), scores, threshold)

        adaptive_scores[outer_valid_idx] = fold_scores[selected_arm]
        adaptive_thresholds[outer_valid_idx] = selected[selected_arm]["threshold"]
        fold_rows.append({
            "outer_fold": outer_fold,
            "outer_train_count": len(outer_train_idx),
            "outer_validation_count": len(outer_valid_idx),
            "outer_validation_class_support": {code: int(np.sum(y_valid.to_numpy() == code)) for code in CLASS_CODES},
            "inner_fold_ctgan_records": inner_generation_records,
            "selected_within_inner_cv": selected,
            "inner_selected_augmentation_arm_for_adaptive_policy": selected_arm,
            "outer_ctgan_training": outer_augmentation_info,
            "outer_validation_metrics": fold_metrics,
        })

    if len(seen_valid_ids) != len(X):
        raise AssertionError("Each training-pool SAMPLE_ID must be outer validation exactly once")

    results = {}
    for arm in ARMS:
        fold_metrics = [row["outer_validation_metrics"][arm] for row in fold_rows]
        fold_aucs = [entry["roc_auc"] for entry in fold_metrics]
        pooled = _pooled_metrics(y.to_numpy(), oof_scores[arm], oof_thresholds[arm])
        results[arm] = {
            "pooled_outer_oof_metrics": pooled,
            "mean_outer_fold_roc_auc": float(np.mean(fold_aucs)),
            "std_outer_fold_roc_auc_population": float(np.std(fold_aucs)),
            "fold_variability_mean_std_min_max": _fold_variability(fold_metrics),
            "folds": [
                {"outer_fold": row["outer_fold"],
                 "selected_svm": row["selected_within_inner_cv"][arm]["parameters"],
                 "selected_threshold": row["selected_within_inner_cv"][arm]["threshold"],
                 "inner_oof_metrics": row["selected_within_inner_cv"][arm]["inner_oof_metrics_at_selected_threshold"],
                 "outer_validation_metrics": row["outer_validation_metrics"][arm]}
                for row in fold_rows
            ],
        }

    adaptive_fold_metrics = []
    adaptive_fold_rows = []
    for row in fold_rows:
        arm = row["inner_selected_augmentation_arm_for_adaptive_policy"]
        adaptive_fold_metrics.append(row["outer_validation_metrics"][arm])
        adaptive_fold_rows.append({"outer_fold": row["outer_fold"], "selected_arm_inside_inner_cv": arm,
                                   "outer_validation_metrics": row["outer_validation_metrics"][arm]})
    adaptive_auc = [item["roc_auc"] for item in adaptive_fold_metrics]

    # Paired differences are computed at the outer-fold level; no arm is selected
    # based on these outer results.
    paired = {}
    for metric_key in ("accuracy", "precision_class_2", "recall_class_2", "f1_class_2",
                       "macro_f1", "balanced_accuracy", "roc_auc"):
        diffs = np.asarray([
            row["outer_validation_metrics"]["ctgan"][metric_key]
            - row["outer_validation_metrics"]["none"][metric_key]
            for row in fold_rows
        ])
        paired[metric_key + "_ctgan_minus_none"] = {
            "fold_values": diffs.tolist(), "mean": float(diffs.mean()),
            "std_population": float(diffs.std()),
        }

    return {
        "arm_comparison": results,
        "inner_selected_arm_policy": {
            "pooled_outer_oof_metrics": _pooled_metrics(y.to_numpy(), adaptive_scores, adaptive_thresholds),
            "mean_outer_fold_roc_auc": float(np.mean(adaptive_auc)),
            "std_outer_fold_roc_auc_population": float(np.std(adaptive_auc)),
            "fold_variability_mean_std_min_max": _fold_variability(adaptive_fold_metrics),
            "folds": adaptive_fold_rows,
            "selection_rule": "Within each outer training partition, select the arm with higher inner-OOF Class-2 F1; ties prefer no augmentation.",
        },
        "paired_outer_fold_differences": paired,
        "outer_fold_audit": [
            {"outer_fold": row["outer_fold"], "outer_train_count": row["outer_train_count"],
             "outer_validation_count": row["outer_validation_count"],
             "inner_selected_arm": row["inner_selected_augmentation_arm_for_adaptive_policy"],
             "ctgan_epochs": ctgan_epochs}
            for row in fold_rows
        ],
        "fold_details": fold_rows,
    }


def render_summary(result: dict) -> str:
    lines = [
        "# Nested autoencoder, CTGAN, and RBF-SVM comparison",
        "",
        f"**Run status:** {result['status']}",
        "",
        "## Protocol and safeguards",
        "",
        "- Same original 323-sample training pool and five stratified outer folds (seed 42); original 81-sample holdout excluded from fitting, selection, and evaluation.",
        "- Three inner folds per outer training partition select the SVM settings and Class-2 threshold separately for no-augmentation and CTGAN arms. The adaptive policy selects its augmentation arm using inner-OOF Class-2 F1 only.",
        "- A fresh set of three modality autoencoders is fitted per inner split and per outer fold. Imputation, scaling, autoencoder early stopping, and SVM fitting use only the respective training partition.",
        "- CTGAN is fitted only to the Class-2 latent vectors from that training partition, using the existing isolated CPU worker. Synthetic rows are training data only and never validation observations.",
        "- The 20-epoch CTGAN setting and small fixed SVM grid were chosen for laptop runtime. These are exploratory, not claimed optimal settings.",
        "- Labels remain neutral CLASS codes. The released 79-feature set has uncertain upstream supervised-selection provenance; this is not an end-to-end leakage-controlled estimate.",
        "",
    ]
    if result["status"] != "completed":
        lines += ["## Execution outcome", "", result.get("execution_note", "No complete experiment result was generated."), "",
                  "No new performance score is claimed. See the JSON run record for the dependency or execution error.", ""]
    else:
        lines += [
            "## Outer-CV arm comparison",
            "",
            "| Arm | Accuracy | Class-1 P/R/F1 | Class-2 P/R/F1 | Macro F1 | Balanced accuracy | Pooled OOF ROC-AUC | Mean fold ROC-AUC ± SD | Confusion matrix [1,2] |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
        for arm in ARMS:
            m = result["experiment"]["arm_comparison"][arm]["pooled_outer_oof_metrics"]
            auc = result["experiment"]["arm_comparison"][arm]
            lines.append(
                f"| {arm} | {m['accuracy']:.4f} | {m['precision_class_1']:.4f}/{m['recall_class_1']:.4f}/{m['f1_class_1']:.4f} | "
                f"{m['precision_class_2']:.4f}/{m['recall_class_2']:.4f}/{m['f1_class_2']:.4f} | {m['macro_f1']:.4f} | "
                f"{m['balanced_accuracy']:.4f} | {m['roc_auc_pooled_oof']:.4f} | {auc['mean_outer_fold_roc_auc']:.4f} ± {auc['std_outer_fold_roc_auc_population']:.4f} | "
                f"`{m['confusion_matrix_rows_true_columns_pred_class_1_then_2']}` |"
            )
        lines += [
            "",
            "The two arms receive the same outer folds and inner split definitions, but each arm selects its own SVM settings and threshold using inner CV. The table reports both arms without selecting a winner from outer-fold results. Per-fold metrics, chosen parameters, thresholds, paired fold differences, and the adaptive inner-selected arm policy are in `ctgan_svm_nested_cv.json`.",
            "",
            "## Fold variability",
            "",
            "The JSON includes mean, population SD, minimum, and maximum across the five untouched outer folds for accuracy, classwise precision/recall/F1, macro F1, balanced accuracy, ROC-AUC, and selected threshold, for both arms and the adaptive policy.",
            "",
            "## Interpretation",
            "",
            "This is exploratory internal validation on a small cohort. CTGAN gains or losses must be judged across the paired outer folds and all classwise metrics. Unknown class semantics, possible upstream feature-selection leakage, the unresolved cohort discrepancy, small Class-2 support, synthetic-data fidelity, and no external validation prevent clinical interpretation.",
            "",
        ]
    lines += [
        "## Current reference result supplied for the no-augmentation tuned RBF-SVM",
        "",
        "Accuracy 73.99%; Class-2 precision 50.85%; recall 69.77%; F1 58.82%; pooled ROC-AUC 0.8087. These values were supplied in the task prompt and serve as context; do not treat them as the result of this script unless its run record says so.",
        "",
        "## PowerShell reproduction (Python 3.12, TensorFlow 2.20.0)",
        "",
        "```powershell",
        "py -3.12 -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install --upgrade pip",
        "python -m pip install numpy pandas scikit-learn tensorflow==2.20.0 sdv==1.38.5 ctgan==0.12.1 torch pytest",
        "$env:TF_ENABLE_ONEDNN_OPTS = '0'",
        "$env:TF_NUM_INTRAOP_THREADS = '2'",
        "$env:TF_NUM_INTEROP_THREADS = '2'",
        "python -m pytest -q",
        "python .\\ctgan_svm_nested_cv.py --data-dir . --output-root .\\ctgan_svm_nested_outputs --ctgan-epochs 20",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(data_dir: Path, output_root: Path, ctgan_epochs: int = CTGAN_EPOCHS) -> Path:
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "nested AE + CTGAN + tuned RBF SVM vs nested AE + tuned RBF SVM",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": package_versions(),
        "label_semantics": "unverified; CLASS codes 1 and 2 only",
        "provided_reference_no_augmentation_tuned_rbf_svm": {
            "source": "user-supplied context; not regenerated in this run",
            "accuracy": 0.7399,
            "precision_class_2": 0.5085,
            "recall_class_2": 0.6977,
            "f1_class_2": 0.5882,
            "pooled_roc_auc": 0.8087,
        },
        "protocol": {
            "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS, "seed": SEED,
            "selection_metric": "inner OOF Class-2 F1",
            "arms": list(ARMS), "svm_grid": list(SVM_GRID),
            "threshold_selection": "maximize inner OOF Class-2 F1; tie-break by precision and closeness to 0.5",
            "arm_tie_break": "no augmentation",
            "ctgan_epochs": ctgan_epochs, "ctgan_pac": PAC,
            "ctgan_batch_rule": f"largest multiple of lcm(2,{PAC}) no greater than min(80, minority training rows)",
            "ctgan_execution": "existing ctgan_worker.py; CPU-only, separate SDV/PyTorch process",
            "feature_selection": "released 79 features; upstream selection timing unknown; not reselected",
            "holdout_usage": "excluded from all fitting, inner selection, and outer evaluation",
        },
    }
    json_path = output_dir / "ctgan_svm_nested_cv.json"
    md_path = output_dir / "RESEARCH_SUMMARY.md"
    try:
        X, y = load_aligned_data(data_dir)
        train_X, train_y, heldout = original_training_pool(X, y)
        if len(train_y) != 323 or train_y.value_counts().to_dict() != {"1": 237, "2": 86}:
            raise ValueError("Recovered training pool differs from the confirmed 323 / (237, 86) split")
        result.update({
            "aligned_sample_count": len(X), "feature_count": X.shape[1],
            "training_pool_count": len(train_y),
            "training_pool_support": {code: int((train_y == code).sum()) for code in CLASS_CODES},
            "heldout_exclusion": {
                "excluded_count": heldout["excluded_count"],
                "excluded_id_sha256": heldout["excluded_id_sha256"],
                "used_for_model_selection_or_evaluation": False,
            },
        })
        try:
            import tensorflow as tf
        except ImportError as exc:
            result["status"] = "blocked_missing_tensorflow"
            result["execution_note"] = "TensorFlow is unavailable; no autoencoders, CTGAN, or model comparisons were run."
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            tf.keras.utils.set_random_seed(SEED)
            try:
                tf.config.experimental.enable_op_determinism()
                result["tensorflow_determinism"] = "requested"
            except (AttributeError, RuntimeError) as exc:
                result["tensorflow_determinism"] = f"unavailable: {type(exc).__name__}"
            try:
                result["experiment"] = nested_evaluate(train_X, train_y, ctgan_epochs=ctgan_epochs)
                result["status"] = "completed"
            finally:
                tf.keras.backend.clear_session()
    except BaseException as exc:
        result["status"] = "incomplete"
        result["execution_note"] = "The requested experiment did not complete; partial or failed execution is not an experimental result."
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=20)}
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_summary(result), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "ctgan_svm_nested_outputs")
    parser.add_argument("--ctgan-epochs", type=int, default=CTGAN_EPOCHS)
    args = parser.parse_args()
    print(run(args.data_dir, args.output_root, args.ctgan_epochs))
