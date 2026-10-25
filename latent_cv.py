"""Checkpoint 2B: fold-isolated multi-omics autoencoders and logistic regression.

Usage: python latent_cv.py
The original 81-sample holdout is reconstructed and excluded by baseline_cv.
Scores remain conditional on the released, preselected feature columns.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline_cv import (
    CLASS_CODES, FILES, SEED, classification_metrics, load_aligned_data,
    original_training_pool,
)

LATENT_DIMS = {"mGE": 16, "mDM": 16, "mCNA": 8}
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
INNER_VALIDATION_FRACTION = 0.2


def keras_fit_encoder(
    X_fit: np.ndarray, X_early_stop: np.ndarray, latent_dim: int, seed: int
):
    """Same Dense/activation/loss/early-stopping layout as main.py, per fold."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint 2B requires TensorFlow; no autoencoder results were generated"
        ) from exc

    tf.keras.utils.set_random_seed(seed)
    input_dim = X_fit.shape[1]
    inputs = tf.keras.Input(shape=(input_dim,))
    hidden = tf.keras.layers.Dense(max(32, input_dim), activation="relu")(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation="linear", name="latent")(hidden)
    hidden = tf.keras.layers.Dense(max(32, input_dim), activation="relu")(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(hidden)
    autoencoder = tf.keras.Model(inputs, outputs)
    encoder = tf.keras.Model(inputs, latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_fit, X_fit,
        validation_data=(X_early_stop, X_early_stop),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )],
    )
    return encoder


def fold_latents(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    fold: int,
    fit_encoder: Callable = keras_fit_encoder,
) -> tuple[np.ndarray, np.ndarray]:
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must be aligned")
    if not X_train.index.is_unique or not X_valid.index.is_unique:
        raise ValueError("SAMPLE_ID must be unique")
    if set(X_train.index) & set(X_valid.index):
        raise ValueError("Train and validation SAMPLE_ID overlap")
    if list(X_train.columns) != list(X_valid.columns):
        raise ValueError("Train and validation feature schemas differ")
    if any(c.split("::")[-1] in {"CLASS", "SAMPLE_ID"} for c in X_train):
        raise ValueError("Target or sample identifier present in model features")

    inner_fit, inner_stop = train_test_split(
        np.arange(len(y_train)), test_size=INNER_VALIDATION_FRACTION,
        random_state=SEED + fold, stratify=y_train,
    )
    train_parts, valid_parts = [], []
    for modality_index, modality in enumerate(FILES):
        columns = [c for c in X_train if c.startswith(f"{modality}::")]
        if not columns:
            raise ValueError(f"Missing or invalid {modality} features")
        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        tr = preprocessor.fit_transform(X_train.loc[:, columns])
        va = preprocessor.transform(X_valid.loc[:, columns])
        latent_dim = min(LATENT_DIMS[modality], len(columns))
        # Early-stopping data also comes exclusively from the outer training fold.
        encoder = fit_encoder(
            tr[inner_fit].astype("float32"), tr[inner_stop].astype("float32"),
            latent_dim, SEED + 100 * fold + modality_index,
        )
        train_z = encoder.predict(tr.astype("float32"), verbose=0)
        valid_z = encoder.predict(va.astype("float32"), verbose=0)
        if train_z.shape != (len(tr), latent_dim) or valid_z.shape != (len(va), latent_dim):
            raise ValueError(f"Invalid latent dimensions for {modality}")
        train_parts.append(train_z)
        valid_parts.append(valid_z)
    return np.concatenate(train_parts, axis=1), np.concatenate(valid_parts, axis=1)


def evaluate_latent_cv(
    X: pd.DataFrame, y: pd.Series, folds: int = 5,
    fit_encoder: Callable = keras_fit_encoder,
) -> dict:
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Features and labels must have unique, aligned SAMPLE_ID")
    if set(y.unique()) != set(CLASS_CODES) or folds < 2 or min(y.value_counts()) < folds:
        raise ValueError("Both neutral CLASS codes need at least one sample per fold")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.empty(len(y), dtype=object)
    per_fold, aucs, seen = [], [], set()
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        valid_ids = set(X.index[valid_idx])
        if valid_ids & (seen | set(X.index[train_idx])):
            raise AssertionError("SAMPLE_ID reused or crossed a fold boundary")
        seen.update(valid_ids)
        train_z, valid_z = fold_latents(
            X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], fold, fit_encoder
        )
        classifier = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )
        classifier.fit(train_z, y.iloc[train_idx])
        pred = classifier.predict(valid_z)
        scores = classifier.decision_function(valid_z)
        true = y.iloc[valid_idx].to_numpy()
        auc = float(roc_auc_score((true == CLASS_CODES[1]).astype(int), scores))
        aucs.append(auc)
        oof[valid_idx] = pred
        per_fold.append({"fold": fold, "train_count": len(train_idx),
                         "validation_count": len(valid_idx), "roc_auc": auc,
                         **classification_metrics(true, pred)})
        if fit_encoder is keras_fit_encoder:
            import tensorflow as tf
            tf.keras.backend.clear_session()
    if len(seen) != len(X):
        raise AssertionError("Each training-pool sample must be validated exactly once")
    return {
        "out_of_fold": classification_metrics(y.to_numpy(), oof),
        "roc_auc_mean_across_folds": float(np.mean(aucs)),
        "roc_auc_std_across_folds": float(np.std(aucs)),
        "folds": per_fold,
    }


def library_versions() -> dict:
    packages = {"numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn",
                "tensorflow": "tensorflow"}
    found = {"python": platform.python_version()}
    for label, package in packages.items():
        try:
            found[label] = version(package)
        except PackageNotFoundError:
            if label == "tensorflow":
                try:
                    found[label] = version("tensorflow-cpu")
                except PackageNotFoundError:
                    found[label] = None
            else:
                found[label] = None
    return found


def run(data_dir: Path, output_root: Path, folds: int = 5) -> Path:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required; no output was generated") from exc
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
        determinism = "TensorFlow op determinism requested"
    except (AttributeError, RuntimeError) as exc:
        determinism = f"TensorFlow op determinism unavailable: {type(exc).__name__}"

    X, y = load_aligned_data(data_dir)
    train_X, train_y, heldout = original_training_pool(X, y)
    result = {
        "checkpoint": "2B",
        "scope": "conditional on released preselected features; not end-to-end leakage-free",
        "label_semantics": "unverified; positive means CLASS code 2 only",
        "heldout": heldout,
        "training_pool_count": len(train_y),
        "training_pool_support": {c: int((train_y == c).sum()) for c in CLASS_CODES},
        "architecture": {
            "modalities": list(FILES), "encoder_per_modality_per_fold": True,
            "hidden_units": "max(32, input_dim), latent_dim, max(32, input_dim)",
            "activations": ["relu", "linear", "relu", "linear"],
            "latent_capacities": LATENT_DIMS,
            "reconstruction_loss": "mse", "optimizer": "adam",
            "epochs_max": EPOCHS, "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "early_stop_fraction_of_outer_training": INNER_VALIDATION_FRACTION,
            "classifier": "LogisticRegression(max_iter=2000, class_weight=balanced)",
            "augmentation": None, "calibration": None,
            "deviation_from_main": "Encoders and preprocessors fitted independently inside every outer CV fold; early-stop split inside that fold",
        },
        "cross_validation": {"type": "StratifiedKFold", "folds": folds,
                             "shuffle": True, "random_state": SEED,
                             "same_splits_as_checkpoint_2a": True},
        "reproducibility": {"base_seed": SEED,
                            "encoder_seed_rule": "42 + 100*fold_number + modality_index",
                            "tf_op_determinism": determinism,
                            "environment": {key: os.getenv(key) for key in (
                                "TF_ENABLE_ONEDNN_OPTS", "TF_NUM_INTRAOP_THREADS",
                                "TF_NUM_INTEROP_THREADS")},
                            "remaining_limit": "Numerical results can vary across TensorFlow, hardware, and thread implementations"},
        "library_versions": library_versions(),
        "models": {"latent_logistic_regression": evaluate_latent_cv(train_X, train_y, folds)},
    }
    output_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "latent_cv.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoint_2b_outputs")
    args = parser.parse_args()
    print(run(args.data_dir, args.output_root))
