"""Checkpoint 3A: fixed-fold latent Logistic Regression augmentation comparison.

Run `python ctgan_cv.py --smoke` before `python ctgan_cv.py --full`.
All three arms share fold-specific encoders and identical validation samples.
The original 81-sample holdout is reconstructed only for exclusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from baseline_cv import CLASS_CODES, SEED, classification_metrics, load_aligned_data, original_training_pool
from latent_cv import fold_latents, keras_fit_encoder, library_versions

CTGAN_EPOCHS = 100
PAC = 10
MAX_BATCH = 80
BOOTSTRAP_NOISE_STD = 0.01
METHODS = ("none", "bootstrap", "ctgan")


def package_versions() -> dict:
    values = library_versions()
    for package in ("sdv", "ctgan", "torch"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def ctgan_batch_size(n_minority: int, pac: int = PAC) -> int:
    if pac <= 0:
        raise ValueError("pac must be positive")
    unit = math.lcm(2, pac)
    batch = min(MAX_BATCH, n_minority) // unit * unit
    if batch < unit:
        raise ValueError("Too few minority rows for a valid CTGAN batch")
    return batch


def quality_diagnostics(real: np.ndarray, generated: np.ndarray) -> dict:
    """Descriptive latent-space checks, not biological fidelity/privacy proof."""
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    if real.ndim != 2 or generated.shape[1:] != real.shape[1:] or len(generated) == 0:
        raise ValueError("Invalid synthetic latent shape")
    if not np.isfinite(generated).all():
        raise ValueError("Non-finite synthetic values")
    scale = np.maximum(np.std(real, axis=0), 1e-8)
    standardized_mean_gap = np.mean(np.abs((generated.mean(axis=0) - real.mean(axis=0)) / scale))
    dist = np.linalg.norm(generated[:, None, :] - real[None, :, :], axis=2)
    nearest = dist.min(axis=1)
    unique_rows = np.unique(np.round(generated, decimals=6), axis=0)
    return {
        "mean_absolute_standardized_mean_gap": float(standardized_mean_gap),
        "synthetic_to_nearest_real_distance_min": float(nearest.min()),
        "synthetic_to_nearest_real_distance_median": float(np.median(nearest)),
        "rounded_6dp_unique_fraction": float(len(unique_rows) / len(generated)),
        "interpretation": "Descriptive checks only; no fidelity, privacy, or clinical validity claim",
    }


def generate_ctgan(
    minority_latent: np.ndarray, needed: int, seed: int,
    epochs: int = CTGAN_EPOCHS,
) -> tuple[np.ndarray, dict]:
    if needed <= 0:
        raise ValueError("CTGAN requires a positive augmentation count")
    batch_size = ctgan_batch_size(len(minority_latent))
    # The installed TF/PyTorch combination crashes when CTGAN fits in the TF
    # process. Transfer only training-fold minority latents to a clean worker.
    with tempfile.TemporaryDirectory(prefix="oncoassist_ctgan_") as temp:
        input_path = Path(temp) / "minority.npz"
        output_path = Path(temp) / "generated.npy"
        np.savez_compressed(input_path, minority=minority_latent, needed=needed)
        command = [sys.executable, str(Path(__file__).with_name("ctgan_worker.py")),
                   "--input", str(input_path), "--output", str(output_path),
                   "--epochs", str(epochs), "--batch-size", str(batch_size),
                   "--pac", str(PAC), "--seed", str(seed)]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=900,
                              check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"CTGAN worker failed with exit {proc.returncode}: {proc.stderr[-3000:]}"
            )
        generated = np.load(output_path, allow_pickle=False)
    if generated.shape != (needed, minority_latent.shape[1]) or not np.isfinite(generated).all():
        raise ValueError("CTGAN returned wrong shape or non-finite values")
    return generated, {
        "actual_method": "ctgan", "epochs": epochs, "batch_size": batch_size,
        "pac": PAC, "seed": seed, "enable_gpu": False,
        "process_isolation": "separate SDV/PyTorch worker; only outer-training minority latents transferred",
        "quality": quality_diagnostics(minority_latent, generated),
    }


def augment(
    X_train: np.ndarray, y_train: np.ndarray, method: str, seed: int,
    ctgan_sampler: Callable = generate_ctgan,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if set(y_train) != set(CLASS_CODES) or len(X_train) != len(y_train):
        raise ValueError("Expected aligned latent features with neutral CLASS codes")
    counts = {code: int(np.sum(y_train == code)) for code in CLASS_CODES}
    minority = min(CLASS_CODES, key=lambda c: counts[c])
    needed = max(counts.values()) - counts[minority]
    info = {"requested_method": method, "actual_method": method,
            "minority_code": minority, "before": counts, "added": needed if method != "none" else 0,
            "seed": seed}
    if method == "none":
        info["after"] = counts.copy()
        return X_train.copy(), y_train.copy(), info
    if needed <= 0:
        raise ValueError("No minority class exists to oversample")
    minority_data = X_train[y_train == minority]
    if method == "bootstrap":
        rng = np.random.default_rng(seed)
        generated = minority_data[rng.choice(len(minority_data), size=needed, replace=True)]
        generated = generated + rng.normal(0, BOOTSTRAP_NOISE_STD, generated.shape)
        info["noise_std"] = BOOTSTRAP_NOISE_STD
        info["quality"] = quality_diagnostics(minority_data, generated)
    else:
        generated, ctgan_info = ctgan_sampler(minority_data.copy(), needed, seed)
        if ctgan_info.get("actual_method") != "ctgan":
            raise RuntimeError("CTGAN did not produce genuine samples")
        if generated.shape != (needed, X_train.shape[1]) or not np.isfinite(generated).all():
            raise ValueError("Invalid CTGAN samples")
        info.update(ctgan_info)
    X_aug = np.vstack([X_train, generated])
    y_aug = np.concatenate([y_train, np.repeat(minority, needed)])
    # Same seed convention for all methods; validation never enters this step.
    permutation = np.random.default_rng(seed + 1).permutation(len(y_aug))
    info["after"] = {code: int(np.sum(y_aug == code)) for code in CLASS_CODES}
    return X_aug[permutation], y_aug[permutation], info


def reproduce_prior_assertion() -> dict:
    """Diagnose the old batch=86 configuration on non-patient dummy data."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ctgan_worker.py")), "--diagnose"],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"CTGAN diagnostic worker failed: {proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def _write(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temp_path.replace(path)


def _base_result(heldout: dict, n_train: int, support: dict, mode: str) -> dict:
    return {
        "checkpoint": "3A", "status": "running", "mode": mode,
        "scope": "conditional on released preselected features; no verified clinical label semantics",
        "heldout": heldout, "training_pool_count": n_train, "training_pool_support": support,
        "cv": {"type": "StratifiedKFold", "folds": 5, "seed": SEED,
               "same_partitions_as_2A_2B": True},
        "architecture": "2B omics autoencoders + fixed LogisticRegression(max_iter=2000, class_weight=balanced); no calibration",
        "augmentation": {"arms": list(METHODS), "ctgan_epochs_full": CTGAN_EPOCHS,
                         "pac": PAC, "max_batch": MAX_BATCH,
                         "bootstrap_noise_std": BOOTSTRAP_NOISE_STD},
        "versions": package_versions(),
        "environment": {k: os.getenv(k) for k in ("TF_ENABLE_ONEDNN_OPTS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "completed_folds": [], "aggregate": None,
    }


def run(
    data_dir: Path, output_root: Path, mode: str,
    fit_encoder: Callable = keras_fit_encoder,
    ctgan_sampler: Callable = generate_ctgan,
    deadline_seconds: int = 1200,
) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
        determinism = "requested"
    except (AttributeError, RuntimeError) as exc:
        determinism = f"unavailable: {type(exc).__name__}"
    start = time.monotonic()
    X, y = load_aligned_data(data_dir)
    train_X, train_y, heldout = original_training_pool(X, y)
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / ("ctgan_smoke.json" if mode == "smoke" else "ctgan_comparison.json")
    result = _base_result(heldout, len(train_y),
                          {c: int(np.sum(train_y == c)) for c in CLASS_CODES}, mode)
    result["tensorflow_determinism"] = determinism
    _write(path, result)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {arm: np.empty(len(train_y), dtype=object) for arm in METHODS}
    try:
        if mode == "smoke":
            result["prior_assertion_reproduction"] = reproduce_prior_assertion()
            _write(path, result)
        for fold, (tr, va) in enumerate(cv.split(train_X, train_y), start=1):
            if time.monotonic() - start > deadline_seconds:
                raise TimeoutError("Checkpoint 3A time budget exhausted before fold")
            if set(train_X.index[tr]) & set(train_X.index[va]):
                raise AssertionError("Train/validation ID overlap")
            z_train, z_valid = fold_latents(
                train_X.iloc[tr], train_y.iloc[tr], train_X.iloc[va], fold, fit_encoder
            )
            fold_data = {"fold": fold, "train_count": len(tr), "validation_count": len(va),
                         "conditions": {}}
            arms = ("ctgan",) if mode == "smoke" else METHODS
            for arm in arms:
                seed = SEED + fold
                if mode == "smoke":
                    # Identical generator and fold data; 2 epochs solely verify viability.
                    sampler = lambda real, needed, s: generate_ctgan(real, needed, s, epochs=2)
                else:
                    sampler = ctgan_sampler
                X_aug, y_aug, aug = augment(z_train, train_y.iloc[tr].to_numpy(),
                                            arm, seed, sampler)
                if mode == "smoke":
                    fold_data["conditions"][arm] = {"augmentation": aug,
                                                      "smoke_only_no_model_metrics": True}
                else:
                    model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=SEED)
                    model.fit(X_aug, y_aug)
                    pred = model.predict(z_valid)
                    scores = model.decision_function(z_valid)
                    true = train_y.iloc[va].to_numpy()
                    auc = float(roc_auc_score((true == CLASS_CODES[1]).astype(int), scores))
                    oof[arm][va] = pred
                    fold_data["conditions"][arm] = {"augmentation": aug,
                        "roc_auc": auc, **classification_metrics(true, pred)}
                result["in_progress_fold"] = fold_data
                _write(path, result)
            result["completed_folds"].append(fold_data)
            result.pop("in_progress_fold", None)
            _write(path, result)
            if fit_encoder is keras_fit_encoder:
                tf.keras.backend.clear_session()
            if mode == "smoke":
                result["status"] = "smoke_succeeded"
                break
        if mode == "full":
            if len(result["completed_folds"]) != 5:
                raise AssertionError("Five complete folds required")
            result["aggregate"] = {}
            for arm in METHODS:
                aucs = [f["conditions"][arm]["roc_auc"] for f in result["completed_folds"]]
                result["aggregate"][arm] = {
                    "actual_method": arm,
                    "mean_fold_roc_auc": float(np.mean(aucs)),
                    "std_fold_roc_auc": float(np.std(aucs)),
                    "out_of_fold": classification_metrics(train_y.to_numpy(), oof[arm]),
                }
            result["status"] = "completed"
        result["elapsed_seconds"] = float(time.monotonic() - start)
    except BaseException as exc:
        result["status"] = "incomplete"
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=18)}
        result["elapsed_seconds"] = float(time.monotonic() - start)
        _write(path, result)
        raise
    _write(path, result)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoint_3a_outputs")
    args = parser.parse_args()
    print(run(args.data_dir, args.output_root, "smoke" if args.smoke else "full"))
