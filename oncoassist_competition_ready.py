import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input


np.random.seed(42)
tf.random.set_seed(42)


# =========================================
# SECTION 1: Imports and configuration
# =========================================
# This script refactors the doctor's notebook into a clean, non-leaky,
# competition-ready training and inference workflow.


# ==================================================
# ADDED BLOCK -- DATA + LABEL REVIEWED TRAINING PIPELINE
# Purpose: clean and reproducible baseline replacing fragmented notebook cells
# Where to place: standalone script for this model-only project
# ==================================================


"""
Purpose of this function

This function:

loads the 3 omics files
checks required columns
finds common samples across all files
aligns them in the same order
separates features and labels
converts labels into binary format

This is one of the most important functions in the script.
"""


def load_and_align_multiomics(
    rna_path: Path,
    dna_path: Path,
    cna_path: Path,
) -> Dict[str, Any]:
    rna_df = pd.read_csv(rna_path)
    dna_df = pd.read_csv(dna_path)
    cna_df = pd.read_csv(cna_path)

    required_rna_cols = {"SAMPLE_ID", "CLASS"}
    required_other_cols = {"SAMPLE_ID"}
    if not required_rna_cols.issubset(rna_df.columns):
        raise ValueError("RNA file must contain SAMPLE_ID and CLASS columns.")
    if not required_other_cols.issubset(dna_df.columns):
        raise ValueError("DNA methylation file must contain SAMPLE_ID column.")
    if not required_other_cols.issubset(cna_df.columns):
        raise ValueError("CNA file must contain SAMPLE_ID column.")

    common_ids = sorted(
        set(rna_df["SAMPLE_ID"]).intersection(dna_df["SAMPLE_ID"], cna_df["SAMPLE_ID"])
    )
    if len(common_ids) == 0:
        raise ValueError("No common SAMPLE_ID values across omics tables.")

    rna_df = (
        rna_df[rna_df["SAMPLE_ID"].isin(common_ids)]
        .set_index("SAMPLE_ID")
        .loc[common_ids]
    )
    dna_df = (
        dna_df[dna_df["SAMPLE_ID"].isin(common_ids)]
        .set_index("SAMPLE_ID")
        .loc[common_ids]
    )
    cna_df = (
        cna_df[cna_df["SAMPLE_ID"].isin(common_ids)]
        .set_index("SAMPLE_ID")
        .loc[common_ids]
    )

    rna_feature_cols = [c for c in rna_df.columns if c != "CLASS"]
    dna_feature_cols = list(dna_df.columns)
    cna_feature_cols = list(cna_df.columns)

    X_rna = rna_df[rna_feature_cols]
    X_dna = dna_df[dna_feature_cols]
    X_cna = cna_df[cna_feature_cols]
    y_raw = rna_df["CLASS"].values

    unique_labels = sorted(np.unique(y_raw).tolist())
    if len(unique_labels) != 2:
        raise ValueError(
            f"This training script expects binary CLASS labels. Found: {unique_labels}"
        )

    # ==================================================
    # MODIFIED BLOCK -- LABEL MAPPING FIX
    # Reason: notebook mixed class labels with synthetic/original labels, causing target corruption
    # ==================================================
    # We map original CLASS values to a stable binary target used by the classifier.
    # Assumption used here: lower CLASS value -> Low-TMB, higher CLASS value -> High-TMB.
    low_label = unique_labels[0]
    high_label = unique_labels[-1]
    y_binary = (y_raw == high_label).astype(int)

    label_mapping = {
        "raw_to_binary": {str(low_label): 0, str(high_label): 1},
        "binary_to_tmb": {0: "Low-TMB", 1: "High-TMB"},
        "raw_semantics": {str(low_label): "Low-TMB", str(high_label): "High-TMB"},
    }

    return {
        "X_rna": X_rna,
        "X_dna": X_dna,
        "X_cna": X_cna,
        "y_binary": y_binary,
        "sample_ids": common_ids,
        "feature_columns": {
            "rna": rna_feature_cols,
            "dna": dna_feature_cols,
            "cna": cna_feature_cols,
        },
        "label_mapping": label_mapping,
    }


# =========================================
# SECTION 2: Safe preprocessing utilities
# =========================================
# What: per-modality impute + scale pipelines
# Why: avoid leakage and ensure stable latent learning
# Input: training DataFrames per modality
# Output: fitted sklearn pipelines

"""
This function creates a preprocessing pipeline that will be used for each omics dataset:

RNA
DNA methylation
CNA

So instead of preprocessing each dataset manually, the script uses one reusable function.
"""


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# =========================================
# SECTION 3: Encoder / latent feature extraction
# =========================================
# What: train per-modality autoencoders and extract latent features
# Why: preserve original latent-space modeling intent from notebook
# Input: preprocessed modality matrices
# Output: trained encoder models and concatenated latent representation

"""
This section is the feature learning part of the pipeline.

Instead of feeding the raw RNA, DNA, and CNA features directly into the final classifier, the script first compresses each modality into a smaller learned representation called a latent vector.

That is why this section is called latent feature extraction.

Big idea of Section 3

For each omics modality:

RNA gets its own autoencoder
DNA gets its own autoencoder
CNA gets its own autoencoder

Each autoencoder learns how to compress the input into a smaller hidden representation.

Then the script keeps only the encoder part and uses it to produce latent features.

Finally, it concatenates the three latent vectors into one combined feature vector.

So the flow is:

preprocessed RNA / DNA / CNA
→ encode each one separately
→ get smaller latent vectors
→ join them together
→ send to logistic regression
"""


def build_autoencoder(input_dim: int, latent_dim: int) -> Tuple[Model, Model]:
    inputs = Input(shape=(input_dim,))
    x = Dense(max(32, input_dim), activation="relu")(inputs)
    latent = Dense(latent_dim, activation="linear", name="latent")(x)
    x = Dense(max(32, input_dim), activation="relu")(latent)
    outputs = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inputs=inputs, outputs=outputs)
    encoder = Model(inputs=inputs, outputs=latent)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


"""
This function trains an autoencoder and returns only the encoder.
"""


def fit_encoder(X_train: np.ndarray, X_val: np.ndarray, latent_dim: int) -> Model:
    autoencoder, encoder = build_autoencoder(
        input_dim=X_train.shape[1], latent_dim=latent_dim
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )
    return encoder


"""
This function takes the trained encoders and uses them to transform full datasets into latent features.
"""


def transform_latent(
    encoder_rna: Model,
    encoder_dna: Model,
    encoder_cna: Model,
    X_rna: np.ndarray,
    X_dna: np.ndarray,
    X_cna: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    z_rna = encoder_rna.predict(X_rna, verbose=0)
    z_dna = encoder_dna.predict(X_dna, verbose=0)
    z_cna = encoder_cna.predict(X_cna, verbose=0)
    latent = np.concatenate([z_rna, z_dna, z_cna], axis=1)

    start = 0
    modality_slices = {}
    for name, z in [("mGE", z_rna), ("mDM", z_dna), ("CNA", z_cna)]:
        modality_slices[name] = (start, start + z.shape[1])
        start += z.shape[1]
    return latent, modality_slices


# =========================================
# SECTION 4: Augmentation (train-only, non-leaky)
# =========================================
# What: balance only the training fold in latent space
# Why: prevent leakage from synthetic data into held-out evaluation
# Input: latent train matrix + train labels
# Output: augmented latent train matrix and labels

"""
This section fixes class imbalance.

That means if one class has fewer samples than the other, the script creates extra training samples for the minority class so the classifier does not become biased toward the majority class.

Very important: it does this only on the training set, not on the test set.

That is why the comment says train-only, non-leaky.
"""


def augment_minority_class(
    X_latent_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    classes, counts = np.unique(y_train, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    minority_count = counts[np.argmin(counts)]
    majority_count = counts[np.argmax(counts)]
    needed = max(0, majority_count - minority_count)

    if needed == 0:
        return (
            X_latent_train,
            y_train,
            {
                "augmentation_source": "none",
                "minority_class": int(minority_class),
                "added_samples": 0,
            },
        )

    minority_data = X_latent_train[y_train == minority_class]

    synthetic_samples: np.ndarray
    augmentation_source = "bootstrap_fallback"

    try:
        import importlib

        sdv_metadata_module = importlib.import_module("sdv.metadata")
        sdv_single_table_module = importlib.import_module("sdv.single_table")
        SingleTableMetadata = getattr(sdv_metadata_module, "SingleTableMetadata")
        CTGANSynthesizer = getattr(sdv_single_table_module, "CTGANSynthesizer")

        latent_df = pd.DataFrame(minority_data)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=latent_df)
        ctgan = CTGANSynthesizer(
            metadata=metadata, epochs=100, batch_size=min(500, len(latent_df))
        )
        ctgan.fit(latent_df)
        synthetic_df = ctgan.sample(needed)
        synthetic_samples = np.asarray(synthetic_df, dtype=np.float32)
        augmentation_source = "ctgan"
    except Exception:
        # If SDV/CTGAN is unavailable, we keep augmentation deterministic and simple.
        synthetic_samples = np.asarray(
            resample(
                minority_data,
                replace=True,
                n_samples=needed,
                random_state=random_state,
            ),
            dtype=np.float32,
        )
        noise = np.random.normal(0, 0.01, size=synthetic_samples.shape).astype(
            np.float32
        )
        synthetic_samples = synthetic_samples + noise

    X_aug = np.vstack([X_latent_train, synthetic_samples])
    y_aug = np.concatenate(
        [y_train, np.full(shape=(needed,), fill_value=minority_class)]
    )

    perm = np.random.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    return (
        X_aug,
        y_aug,
        {
            "augmentation_source": augmentation_source,
            "minority_class": int(minority_class),
            "majority_class": int(majority_class),
            "added_samples": int(needed),
        },
    )


# =========================================
# SECTION 5: Prediction model and helper metrics
# =========================================
# What: logistic classifier + optional disagreement ensemble
# Why: interpretable latent contributions and robust reliability checks
# Input: augmented latent train data
# Output: fitted classifier and ensemble members

"""
🔹 Big idea of Section 5

This section introduces:

1. Main model

👉 Logistic Regression (your classifier)

2. Ensemble models

👉 Multiple versions of the model to check disagreement

3. Helper functions

👉 For measuring:

distance (OOD detection)
data completeness
"""


def train_disagreement_ensemble(
    X_train: np.ndarray, y_train: np.ndarray, n_models: int = 5
) -> List[LogisticRegression]:
    ensemble = []
    n = len(X_train)
    for i in range(n_models):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42 + i
        )
        clf.fit(X_train[idx], y_train[idx])
        ensemble.append(clf)
    return ensemble


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    diff = x - mean
    return float(np.sqrt(np.maximum(0.0, diff @ inv_cov @ diff.T)))


def compute_data_completeness(
    row_rna: pd.DataFrame,
    row_dna: pd.DataFrame,
    row_cna: pd.DataFrame,
) -> float:
    total = row_rna.shape[1] + row_dna.shape[1] + row_cna.shape[1]
    non_missing = (
        row_rna.notna().sum(axis=1).iloc[0]
        + row_dna.notna().sum(axis=1).iloc[0]
        + row_cna.notna().sum(axis=1).iloc[0]
    )
    return float(non_missing / total)


# =========================================
# SECTION 6A: Gene-level explainer (Option 1)
# =========================================
# What: train a secondary interpretable model in preprocessed feature space
# Why: provide top contributing genes/features while keeping latent model untouched


def build_gene_feature_names(feature_columns: Dict[str, List[str]]) -> List[str]:
    names = []
    names.extend([f"mGE::{c}" for c in feature_columns["rna"]])
    names.extend([f"mDM::{c}" for c in feature_columns["dna"]])
    names.extend([f"CNA::{c}" for c in feature_columns["cna"]])
    return names


def stack_preprocessed_modalities(
    X_rna: np.ndarray,
    X_dna: np.ndarray,
    X_cna: np.ndarray,
) -> np.ndarray:
    return np.concatenate([X_rna, X_dna, X_cna], axis=1)


def train_gene_explainer(
    X_gene_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    explainer = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.5,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )
    explainer.fit(X_gene_train, y_train)
    return explainer


def explain_top_genes_from_explainer(
    gene_vector: np.ndarray,
    prob_high_tmb: float,
    gene_explainer: LogisticRegression,
    gene_feature_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    coef = gene_explainer.coef_.reshape(-1)
    contributions = gene_vector.reshape(-1) * coef

    top_indices = np.argsort(np.abs(contributions))[-top_k:][::-1]
    top_genes = []
    for idx in top_indices:
        contribution = float(contributions[idx])
        direction = "supports_high_tmb" if contribution >= 0 else "supports_low_tmb"
        top_genes.append(
            {
                "feature_name": gene_feature_names[int(idx)],
                "contribution": round(contribution, 6),
                "direction": direction,
            }
        )

    predicted_class = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "top_contributing_genes": top_genes,
        "gene_explanation_summary": (
            "Top contributors are produced by a secondary L1-logistic explainer trained "
            "in preprocessed feature space. Primary prediction still comes from the latent-space model."
        ),
    }


# =========================================
# SECTION 6: Explainability layer
# =========================================
# What: honest latent-space explanation and modality contribution ranking
# Why: avoid misleading raw-gene explanations when model consumes latent features
# Input: latent vector, classifier, modality slices
# Output: structured explanation dictionary


"""
This section explains why the model made a prediction.

But it does it in a careful way.

It does not pretend to explain raw genes directly, because the final classifier is not using raw genes.
It is using latent features from the autoencoders.

That is why this section is actually very good scientifically: it avoids fake or misleading explanations.

So instead of saying:

❌ “Gene X caused this prediction” (misleading)

It correctly says:

✅ “Latent components and modalities contributed”

"""


def explain_latent_prediction(
    latent_vector: np.ndarray,
    prob_high_tmb: float,
    classifier: LogisticRegression,
    modality_slices: Dict[str, Tuple[int, int]],
    top_k: int = 5,
) -> Dict[str, Any]:
    # ==================================================
    # ADDED BLOCK -- EXPLAINABILITY MODULE
    # Purpose: produce honest explanation outputs for latent-space prediction
    # Where to place: after classifier training and before final inference report assembly
    # ==================================================
    # Important scientific note:
    # The classifier consumes latent components, not raw genes directly.
    # Therefore, reporting "Top 5 Genes" here would be misleading unless a validated
    # latent-to-gene attribution pipeline is implemented. We expose top latent signals
    # and modality-level contributions instead.

    coef = classifier.coef_.reshape(-1)
    contributions = latent_vector.reshape(-1) * coef

    top_indices = np.argsort(np.abs(contributions))[-top_k:][::-1]
    top_signals = []
    for idx in top_indices:
        contribution = float(contributions[idx])
        direction = "supports_high_tmb" if contribution >= 0 else "supports_low_tmb"
        top_signals.append(
            {
                "latent_component": f"z_{int(idx)}",
                "contribution": round(contribution, 6),
                "direction": direction,
            }
        )

    modality_scores = {}
    for modality, (start, end) in modality_slices.items():
        modality_scores[modality] = float(np.sum(np.abs(contributions[start:end])))

    dominant_omics = [
        k
        for k, _ in sorted(modality_scores.items(), key=lambda kv: kv[1], reverse=True)
    ]

    predicted_class = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "dominant_omics": dominant_omics,
        "top_contributing_signals": top_signals,
        "modality_contribution_scores": modality_scores,
        "explanation_summary": (
            "Prediction rationale is provided in latent/component space, "
            "with modality-level contribution ranking. Raw gene ranking is intentionally "
            "disabled to avoid misleading attribution in this latent-space pipeline."
        ),
    }


# =========================================
# SECTION 7: Reliability / uncertainty / data quality layer
# =========================================
# What: confidence, inconclusive zone, data completeness, OOD risk, disagreement
# Why: model must identify weak-evidence cases and recommend confirmation
# Input: probability + latent vector + training distribution references
# Output: structured reliability dictionary


"""
This section is one of the most important parts of the whole script.

Why?
Because a model should not only say:

“Here is my prediction”

It should also say:

“How trustworthy is this prediction?”
“Is this sample unusual?”
“Is the input incomplete?”
“Are models disagreeing?”
“Should this be treated with caution?”

That is exactly what Section 7 does.
"""


def reliability_assessment(
    prob_high_tmb: float,
    data_completeness: float,
    latent_vector: np.ndarray,
    train_mean: np.ndarray,
    train_inv_cov: np.ndarray,
    ood_threshold: float,
    disagreement_models: List[LogisticRegression],
    inconclusive_prob_low: float = 0.40,
    inconclusive_prob_high: float = 0.60,
    disagreement_std_threshold: float = 0.10,
) -> Dict[str, Any]:
    # ==================================================
    # ADDED BLOCK -- RELIABILITY / UNCERTAINTY MODULE
    # Purpose: add safety framing and detect weak-confidence / weak-quality cases
    # Where to place: after explanation generation and before clinical output assembly
    # ==================================================

    # CHANGED:
    # Inconclusive zone is probability-based around 0.4-0.6 as requested.
    # This directly captures borderline model belief near the decision boundary.
    confidence = float(max(prob_high_tmb, 1.0 - prob_high_tmb))
    is_inconclusive = inconclusive_prob_low <= prob_high_tmb <= inconclusive_prob_high

    if data_completeness < 0.80:
        data_quality_warning = "high_missingness_detected"
    elif data_completeness < 0.95:
        data_quality_warning = "moderate_missingness_detected"
    else:
        data_quality_warning = "none"

    sample_dist = mahalanobis_distance(
        latent_vector.reshape(-1), train_mean, train_inv_cov
    )
    if sample_dist >= ood_threshold:
        ood_risk = "high"
    elif sample_dist >= 0.85 * ood_threshold:
        ood_risk = "moderate"
    else:
        ood_risk = "low"

    # Optional disagreement logic: practical here using bootstrap logistic ensemble.
    ensemble_probs = [
        m.predict_proba(latent_vector.reshape(1, -1))[0, 1] for m in disagreement_models
    ]
    disagreement_std = float(np.std(ensemble_probs))
    model_disagreement_flag = disagreement_std >= disagreement_std_threshold

    caution_reasons = []
    if is_inconclusive:
        caution_reasons.append("borderline_confidence")
    if data_quality_warning != "none":
        caution_reasons.append(data_quality_warning)
    if ood_risk in {"moderate", "high"}:
        caution_reasons.append(f"ood_risk_{ood_risk}")
    if model_disagreement_flag:
        caution_reasons.append("model_disagreement")

    if len(caution_reasons) == 0:
        reliability_status = "Reliable"
        reliability_summary = "Prediction confidence and sample quality appear acceptable for decision-support use."
        recommendation = "Use as decision-support only; correlate with full clinical context and standard testing."
    else:
        reliability_status = "Caution"
        reliability_summary = (
            "Prediction carries caution flags: " + ", ".join(caution_reasons) + "."
        )
        recommendation = (
            "Treat as inconclusive/at-risk decision-support output; perform confirmatory review "
            "with orthogonal clinical or molecular evidence."
        )

    return {
        "confidence": round(confidence, 6),
        "is_inconclusive": bool(is_inconclusive),
        "data_completeness": round(data_completeness, 6),
        "data_quality_warning": data_quality_warning,
        "out_of_distribution_risk": ood_risk,
        "mahalanobis_distance": round(sample_dist, 6),
        "ood_threshold": round(float(ood_threshold), 6),
        "model_disagreement_std": round(disagreement_std, 6),
        "model_disagreement_flag": bool(model_disagreement_flag),
        "reliability_status": reliability_status,
        "warnings": caution_reasons,
        "reliability_summary": reliability_summary,
        "recommendation_for_confirmation": recommendation,
    }


def build_clinical_output(
    prob_high_tmb: float,
    explanation: Dict[str, Any],
    reliability: Dict[str, Any],
    gene_explanation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # =========================================
    # SECTION 8: Clinical decision-support report
    # =========================================
    # What: convert model outputs to medically cautious report format
    # Why: frontend and clinical users need structured, careful interpretation
    # Input: probability + explainability output + reliability output
    # Output: JSON-serializable report dictionary

    # ==================================================
    # ADDED BLOCK -- CLINICAL DECISION-SUPPORT OUTPUT
    # Purpose: build a structured inference-ready report object for later project integration
    # Where to place: final stage of inference after explainability + reliability are computed
    # ==================================================

    """
        This section takes the outputs from the earlier sections and turns them into a final report object that is ready for:

    frontend display
    JSON export
    API response
    safer clinical-style interpretation

    So before this section, the script has separate pieces:

    prediction probability
    explainability info
    reliability info

    Section 8 combines them into one final structured report.
    """

    prediction = "High-TMB" if prob_high_tmb >= 0.5 else "Low-TMB"
    predicted_tmb_category = (
        "Inconclusive" if reliability["is_inconclusive"] else prediction
    )

    if prediction == "High-TMB":
        clinical_interpretation = (
            "This profile may indicate higher tumor mutational burden signal in latent multi-omics space. "
            "This output is decision-support only and should be clinically validated."
        )
    else:
        clinical_interpretation = (
            "This profile may indicate lower tumor mutational burden signal in latent multi-omics space. "
            "This output is decision-support only and should be clinically validated."
        )

    if gene_explanation is None:
        top_genes = []
        gene_explanation_summary = (
            "No gene-level explainer output available for this run."
        )
    else:
        top_genes = gene_explanation["top_contributing_genes"]
        gene_explanation_summary = gene_explanation["gene_explanation_summary"]

    return {
        "prediction": prediction,
        "predicted_tmb_category": predicted_tmb_category,
        "probability_high_tmb": round(float(prob_high_tmb), 6),
        "confidence": reliability["confidence"],
        "inconclusive": reliability["is_inconclusive"],
        "data_completeness": reliability["data_completeness"],
        "data_quality_warning": reliability["data_quality_warning"],
        "out_of_distribution_risk": reliability["out_of_distribution_risk"],
        "main_contributing_omics": explanation["dominant_omics"],
        "top_model_signals": explanation["top_contributing_signals"],
        "top_contributing_genes": top_genes,
        "gene_explanation_summary": gene_explanation_summary,
        "dominant_omics": explanation["dominant_omics"],
        "top_contributing_signals": explanation["top_contributing_signals"],
        "explanation_summary": explanation["explanation_summary"],
        "reliability_status": reliability["reliability_status"],
        "warnings": reliability["warnings"],
        "clinical_interpretation": clinical_interpretation,
        "reliability_summary": reliability["reliability_summary"],
        "recommendation": reliability["recommendation_for_confirmation"],
        "disclaimer": "Decision-support only. Requires clinical confirmation.",
    }


# =========================================
# SECTION 9: Evaluation + inference helper
# =========================================
# What: reusable single-sample inference function from loaded artifacts
# Why: clean integration path for your main project runtime
# Input: loaded bundle + one-row DataFrames for each omics modality
# Output: one structured clinical report dictionary

"""
This section is very important for deployment/integration.

Earlier sections were mostly about:

training
explanation
reliability
report building

Section 9 answers a different question:

After I save the trained artifacts, how do I use them later on one new sample?

That is exactly what this section does.
"""


def build_report_from_loaded_bundle(
    loaded_bundle: Dict[str, Any],
    raw_rna_row: pd.DataFrame,
    raw_dna_row: pd.DataFrame,
    raw_cna_row: pd.DataFrame,
) -> Dict[str, Any]:
    pre_rna = loaded_bundle["preprocessors"]["rna"]
    pre_dna = loaded_bundle["preprocessors"]["dna"]
    pre_cna = loaded_bundle["preprocessors"]["cna"]

    feature_cols = loaded_bundle["feature_columns"]
    raw_rna_row = pd.DataFrame(raw_rna_row.loc[:, feature_cols["rna"]])
    raw_dna_row = pd.DataFrame(raw_dna_row.loc[:, feature_cols["dna"]])
    raw_cna_row = pd.DataFrame(raw_cna_row.loc[:, feature_cols["cna"]])

    x_rna = pre_rna.transform(raw_rna_row)
    x_dna = pre_dna.transform(raw_dna_row)
    x_cna = pre_cna.transform(raw_cna_row)
    x_gene = stack_preprocessed_modalities(x_rna, x_dna, x_cna)

    encoder_rna = loaded_bundle["encoders"]["rna"]
    encoder_dna = loaded_bundle["encoders"]["dna"]
    encoder_cna = loaded_bundle["encoders"]["cna"]

    latent, _ = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, x_rna, x_dna, x_cna
    )
    latent_vector = latent[0]

    classifier = loaded_bundle["classifier"]
    prob_high = float(classifier.predict_proba(latent)[0, 1])

    explanation = explain_latent_prediction(
        latent_vector=latent_vector,
        prob_high_tmb=prob_high,
        classifier=classifier,
        modality_slices=loaded_bundle["modality_slices"],
        top_k=loaded_bundle["explainability_config"]["top_k_signals"],
    )

    ood_stats = loaded_bundle["ood_stats"]
    rel_cfg = loaded_bundle["reliability_config"]
    reliability = reliability_assessment(
        prob_high_tmb=prob_high,
        data_completeness=compute_data_completeness(
            raw_rna_row, raw_dna_row, raw_cna_row
        ),
        latent_vector=latent_vector,
        train_mean=ood_stats["train_mean"],
        train_inv_cov=ood_stats["train_inv_cov"],
        ood_threshold=ood_stats["ood_threshold"],
        disagreement_models=loaded_bundle["disagreement_models"],
        inconclusive_prob_low=rel_cfg["inconclusive_prob_low"],
        inconclusive_prob_high=rel_cfg["inconclusive_prob_high"],
        disagreement_std_threshold=rel_cfg["disagreement_std_threshold"],
    )

    gene_explanation = None
    if (
        "gene_explainer_model" in loaded_bundle
        and "gene_feature_names" in loaded_bundle
    ):
        gene_explanation = explain_top_genes_from_explainer(
            gene_vector=x_gene[0],
            prob_high_tmb=prob_high,
            gene_explainer=loaded_bundle["gene_explainer_model"],
            gene_feature_names=loaded_bundle["gene_feature_names"],
            top_k=loaded_bundle.get("explainability_config", {}).get(
                "top_k_signals", 5
            ),
        )

    return build_clinical_output(
        prob_high_tmb=prob_high,
        explanation=explanation,
        reliability=reliability,
        gene_explanation=gene_explanation,
    )


# =========================================
# SECTION 10: SAVING ARTIFACTS
# =========================================
# What: robust export strategy for mixed sklearn + Keras objects
# Why: raw pickling Keras models is fragile across environments


def save_export_bundle(
    export_dir: Path,
    encoder_rna: Model,
    encoder_dna: Model,
    encoder_cna: Model,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    # ==================================================
    # ADDED BLOCK -- EXPORT / PKL PREPARATION
    # Purpose: prepare bundled inference artifact for use in another project
    # Where to place: after model training/validation and before integration handoff
    # ==================================================
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras encoders separately for robust TF loading in other projects.
    # Keeping them outside the PKL avoids brittle TensorFlow pickling behavior.
    encoder_rna_path = export_dir / "encoder_rna.keras"
    encoder_dna_path = export_dir / "encoder_dna.keras"
    encoder_cna_path = export_dir / "encoder_cna.keras"
    encoder_rna.save(encoder_rna_path)
    encoder_dna.save(encoder_dna_path)
    encoder_cna.save(encoder_cna_path)

    bundle_path = export_dir / "oncoassist_inference_bundle.pkl"
    bundle_payload = {
        **payload,
        "encoder_paths": {
            "rna": encoder_rna_path.name,
            "dna": encoder_dna_path.name,
            "cna": encoder_cna_path.name,
        },
    }
    # What is saved in PKL:
    # - fitted preprocessors
    # - fitted classifier + disagreement ensemble
    # - label mapping
    # - latent modality layout and OOD statistics
    # - reliability / explainability configuration
    # - metadata required for future inference assembly
    joblib.dump(bundle_payload, bundle_path)

    return {
        "bundle_pkl": str(bundle_path),
        "encoder_rna": str(encoder_rna_path),
        "encoder_dna": str(encoder_dna_path),
        "encoder_cna": str(encoder_cna_path),
    }


# =========================================
# SECTION 11: LOADING FOR INFERENCE
# =========================================
# What: load PKL metadata plus Keras encoder files
# Why: reconstruct full inference pipeline in deployment


def load_export_bundle(export_dir: Path) -> Dict[str, Any]:
    bundle_path = export_dir / "oncoassist_inference_bundle.pkl"
    payload = joblib.load(bundle_path)
    payload["encoders"] = {
        "rna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["rna"]),
        "dna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["dna"]),
        "cna": tf.keras.models.load_model(export_dir / payload["encoder_paths"]["cna"]),
    }
    return payload


# =========================================
# SECTION 12: End-to-end training and run
# =========================================
# What: train, evaluate, explain, assess reliability, report, save, reload
# Why: one-command reproducible model development flow


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rna_path = base_dir / "mGE (1).csv"
    dna_path = base_dir / "mDM (1).csv"
    cna_path = base_dir / "mCNA (1).csv"

    data = load_and_align_multiomics(
        rna_path=rna_path, dna_path=dna_path, cna_path=cna_path
    )

    X_rna = data["X_rna"]
    X_dna = data["X_dna"]
    X_cna = data["X_cna"]
    y = data["y_binary"]

    split_idx_train, split_idx_test = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_rna_train_df, X_rna_test_df = (
        X_rna.iloc[split_idx_train],
        X_rna.iloc[split_idx_test],
    )
    X_dna_train_df, X_dna_test_df = (
        X_dna.iloc[split_idx_train],
        X_dna.iloc[split_idx_test],
    )
    X_cna_train_df, X_cna_test_df = (
        X_cna.iloc[split_idx_train],
        X_cna.iloc[split_idx_test],
    )
    y_train, y_test = y[split_idx_train], y[split_idx_test]

    pre_rna = build_preprocessor()
    pre_dna = build_preprocessor()
    pre_cna = build_preprocessor()

    X_rna_train = pre_rna.fit_transform(X_rna_train_df)
    X_dna_train = pre_dna.fit_transform(X_dna_train_df)
    X_cna_train = pre_cna.fit_transform(X_cna_train_df)

    X_rna_test = pre_rna.transform(X_rna_test_df)
    X_dna_test = pre_dna.transform(X_dna_test_df)
    X_cna_test = pre_cna.transform(X_cna_test_df)

    X_gene_train = stack_preprocessed_modalities(X_rna_train, X_dna_train, X_cna_train)
    X_gene_test = stack_preprocessed_modalities(X_rna_test, X_dna_test, X_cna_test)
    gene_feature_names = build_gene_feature_names(data["feature_columns"])
    gene_explainer_model = train_gene_explainer(X_gene_train, y_train)

    latent_rna_dim = min(16, X_rna_train.shape[1])
    latent_dna_dim = min(16, X_dna_train.shape[1])
    latent_cna_dim = min(8, X_cna_train.shape[1])

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2, random_state=42, stratify=y_train
    )

    encoder_rna = fit_encoder(
        X_rna_train[train_idx], X_rna_train[val_idx], latent_rna_dim
    )
    encoder_dna = fit_encoder(
        X_dna_train[train_idx], X_dna_train[val_idx], latent_dna_dim
    )
    encoder_cna = fit_encoder(
        X_cna_train[train_idx], X_cna_train[val_idx], latent_cna_dim
    )

    X_latent_train, modality_slices = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, X_rna_train, X_dna_train, X_cna_train
    )
    X_latent_test, _ = transform_latent(
        encoder_rna, encoder_dna, encoder_cna, X_rna_test, X_dna_test, X_cna_test
    )

    X_train_aug, y_train_aug, aug_info = augment_minority_class(X_latent_train, y_train)

    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )
    classifier.fit(X_train_aug, y_train_aug)

    disagreement_models = train_disagreement_ensemble(
        X_train_aug, y_train_aug, n_models=5
    )

    y_test_prob = classifier.predict_proba(X_latent_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_test_prob)),
        "classification_report": classification_report(
            y_test, y_test_pred, output_dict=True
        ),
    }

    # OOD statistics from training latent distribution
    cov = np.cov(X_latent_train.T)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    train_mean = X_latent_train.mean(axis=0)
    train_inv_cov = np.linalg.pinv(cov)
    train_distances = np.array(
        [mahalanobis_distance(x, train_mean, train_inv_cov) for x in X_latent_train]
    )
    ood_threshold = float(np.quantile(train_distances, 0.95))

    # Use first held-out sample as demonstration for block-by-block outputs
    demo_idx = 0
    demo_prob = float(y_test_prob[demo_idx])
    demo_latent = X_latent_test[demo_idx]
    demo_gene_vector = X_gene_test[demo_idx]

    demo_completeness = compute_data_completeness(
        X_rna_test_df.iloc[[demo_idx]],
        X_dna_test_df.iloc[[demo_idx]],
        X_cna_test_df.iloc[[demo_idx]],
    )

    explanation = explain_latent_prediction(
        latent_vector=demo_latent,
        prob_high_tmb=demo_prob,
        classifier=classifier,
        modality_slices=modality_slices,
        top_k=5,
    )

    gene_explanation = explain_top_genes_from_explainer(
        gene_vector=demo_gene_vector,
        prob_high_tmb=demo_prob,
        gene_explainer=gene_explainer_model,
        gene_feature_names=gene_feature_names,
        top_k=5,
    )

    reliability = reliability_assessment(
        prob_high_tmb=demo_prob,
        data_completeness=demo_completeness,
        latent_vector=demo_latent,
        train_mean=train_mean,
        train_inv_cov=train_inv_cov,
        ood_threshold=ood_threshold,
        disagreement_models=disagreement_models,
    )

    clinical_output = build_clinical_output(
        prob_high_tmb=demo_prob,
        explanation=explanation,
        reliability=reliability,
        gene_explanation=gene_explanation,
    )

    export_dir = base_dir / "export_artifacts"
    export_paths = save_export_bundle(
        export_dir=export_dir,
        encoder_rna=encoder_rna,
        encoder_dna=encoder_dna,
        encoder_cna=encoder_cna,
        payload={
            "preprocessors": {"rna": pre_rna, "dna": pre_dna, "cna": pre_cna},
            "classifier": classifier,
            "gene_explainer_model": gene_explainer_model,
            "gene_feature_names": gene_feature_names,
            "disagreement_models": disagreement_models,
            "feature_columns": data["feature_columns"],
            "label_mapping": data["label_mapping"],
            "modality_slices": modality_slices,
            "reliability_config": {
                "inconclusive_prob_low": 0.40,
                "inconclusive_prob_high": 0.60,
                "disagreement_std_threshold": 0.10,
                "ood_quantile": 0.95,
            },
            "explainability_config": {
                "top_k_signals": 5,
                "note": "Latent-space and modality-level explanation only; no raw top-gene attribution.",
                "gene_explainer_note": "Top contributing genes/features are from secondary L1-logistic explainer in preprocessed feature space.",
            },
            "ood_stats": {
                "train_mean": train_mean,
                "train_inv_cov": train_inv_cov,
                "ood_threshold": ood_threshold,
            },
            "augmentation_info": aug_info,
            "training_metrics": metrics,
            "demo_output": clinical_output,
        },
    )

    # Verify artifact loading
    loaded_bundle = load_export_bundle(export_dir)
    loaded_keys = sorted([k for k in loaded_bundle.keys() if k != "encoders"])

    inference_report = build_report_from_loaded_bundle(
        loaded_bundle=loaded_bundle,
        raw_rna_row=X_rna_test_df.iloc[[demo_idx]],
        raw_dna_row=X_dna_test_df.iloc[[demo_idx]],
        raw_cna_row=X_cna_test_df.iloc[[demo_idx]],
    )

    print("=== TRAINING / EVAL SUMMARY ===")
    print(json.dumps({"metrics": metrics, "augmentation_info": aug_info}, indent=2))

    print("\n=== AFTER EXPLAINABILITY BLOCK ===")
    print(json.dumps(explanation, indent=2))

    print("\n=== AFTER TOP-5 GENE EXPLAINER BLOCK ===")
    print(json.dumps(gene_explanation, indent=2))

    print("\n=== AFTER RELIABILITY BLOCK ===")
    print(json.dumps(reliability, indent=2))

    print("\n=== AFTER CLINICAL OUTPUT BLOCK ===")
    print(json.dumps(clinical_output, indent=2))

    print("\n=== AFTER EXPORT BLOCK ===")
    print(
        json.dumps(
            {
                "export_paths": export_paths,
                "loaded_bundle_keys": loaded_keys,
                "inference_report_from_loaded_bundle": inference_report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
