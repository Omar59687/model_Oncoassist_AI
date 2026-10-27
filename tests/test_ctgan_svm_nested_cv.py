"""Fast mechanics tests; the sampler is mocked and produces no research result."""

import numpy as np
import pandas as pd

from ctgan_svm_nested_cv import SVM_GRID, nested_evaluate


def test_svm_grid_is_small_and_covers_rbf_regularization_gamma_and_weighting():
    assert len(SVM_GRID) == 8
    assert {item["C"] for item in SVM_GRID} == {0.5, 2.0}
    assert {item["gamma"] for item in SVM_GRID} == {"scale", 0.01}
    assert {item["weight_label"] for item in SVM_GRID} == {"balanced", "code2_x1.5"}


def test_nested_cv_keeps_synthetic_training_rows_out_of_inner_validation():
    rng = np.random.default_rng(73)
    n = 60
    labels = np.array(["1"] * 42 + ["2"] * 18)
    order = rng.permutation(n)
    labels = labels[order]
    raw = rng.normal(size=(n, 3))
    raw[:, 0] = order  # immutable marker used only by this mechanics test
    X = pd.DataFrame(raw, columns=["marker", "x1", "x2"],
                     index=[f"sample-{i:03d}" for i in range(n)])
    y = pd.Series(labels, index=X.index)
    validation_markers = set()

    def fake_latents(X_train, y_train, X_valid, fold):
        del y_train, fold
        validation_markers.clear()
        validation_markers.update(X_valid["marker"].astype(int).tolist())
        return X_train.to_numpy(dtype=float), X_valid.to_numpy(dtype=float)

    def fake_ctgan_sampler(minority, needed, seed, *, epochs):
        assert epochs == 1
        assert not (set(minority[:, 0].astype(int)) & validation_markers)
        local_rng = np.random.default_rng(seed)
        generated = minority[local_rng.integers(0, len(minority), size=needed)].copy()
        generated += local_rng.normal(0, 0.001, size=generated.shape)
        return generated, {"actual_method": "ctgan", "epochs": epochs, "batch_size": 10,
                           "pac": 10, "seed": seed,
                           "quality": {"test_only_mock": True}}

    result = nested_evaluate(
        X, y, fit_latents=fake_latents, ctgan_sampler=fake_ctgan_sampler,
        ctgan_epochs=1, outer_folds=3, inner_folds=2,
        specs=({"C": 1.0, "gamma": "scale", "class_weight": "balanced",
                "weight_label": "test"},),
    )
    assert len(result["arm_comparison"]["none"]["folds"]) == 3
    assert len(result["arm_comparison"]["ctgan"]["folds"]) == 3
    assert result["arm_comparison"]["none"]["pooled_outer_oof_metrics"]["support_class_2"] == 18
    assert result["arm_comparison"]["ctgan"]["pooled_outer_oof_metrics"]["support_class_2"] == 18
    assert len(result["paired_outer_fold_differences"]["f1_class_2_ctgan_minus_none"]["fold_values"]) == 3
