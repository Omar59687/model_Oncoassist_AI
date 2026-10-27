# Nested latent threshold and classifier experiment

**Status:** blocked_missing_dependency

## Protocol and safeguards

- Reconstructs the original stratified 80/20 split (seed 42); 81 holdout samples are excluded from all model fitting, threshold selection, and evaluation.
- Uses five stratified outer folds and three inner folds. Each encoder, imputer, scaler, classifier, hyperparameter choice, and threshold is learned inside the relevant training partition.
- Threshold and candidate settings are chosen by inner out-of-fold Class-2 F1; final metrics come only from the untouched outer validation fold.
- All labels are reported as neutral CLASS codes. Clinical meaning is unverified. Results remain conditional on the 79 released features because upstream feature-selection timing is unknown.
- Repository audit also documents a 404-versus-204 cohort discrepancy against the paper, uncertain supervised-feature-selection provenance, and no external validation. These prevent clinical interpretation or a claim of established generalization.

## Execution outcome

TensorFlow is not installed in the execution environment; the fold-isolated autoencoder experiment could not be run. No new performance metrics were generated.

No new performance estimate is claimed. Install the documented dependencies and rerun the PowerShell command in the report.

## Existing reported benchmarks (provided results)

| Condition | Accuracy | Class-2 precision | Class-2 recall | Class-2 F1 | Mean fold ROC-AUC | Protocol comparability |
|---|---:|---:|---:|---:|---:|---|
| No augmentation, latent logistic (user-confirmed) | 73.68% | 50.45% | 65.12% | 56.85% | 0.7846 | Same 323-sample/5-fold family; new nested selection protocol differs |
| CTGAN, latent logistic (user-confirmed) | 78.02% | 61.19% | 47.67% | 53.59% | 0.8042 | Prior fixed-parameter evaluation; not directly comparable to nested tuned result |
| CTGAN, archived checkpoint 3A | 76.78% | 57.75% | 47.67% | 52.23% | 0.7903 | Saved same-fold checkpoint; differs from current user-confirmed CTGAN result |

The current CTGAN confusion matrix supplied by the user is `[[211, 26], [45, 41]]`. The branch's archived `checkpoint_3a_outputs/20261025T120811006346Z/ctgan_comparison.json` instead reports `[[207, 30], [45, 41]]`, 76.78% accuracy, and mean fold ROC-AUC 0.790326 (SD 0.020227). Treat these as different result records; the current result's run provenance/fold metrics are not present in the repository. Neither result uses verified clinical label names. These comparisons must not be read as external validation.

### Full metrics recoverable from the supplied OOF results

| Condition | Class-1 P/R/F1 | Class-2 P/R/F1 | Macro F1 | Balanced accuracy | Confusion matrix [1,2] | Fold variability available here |
|---|---|---|---:|---:|---|---|
| No augmentation (user-confirmed OOF) | 0.8585 / 0.7679 / 0.8107 | 0.5045 / 0.6512 / 0.5685 | 0.6896 | 0.7095 | `[[182,55],[30,56]]` | Saved checkpoint has per-fold metrics; AUC there is 0.784880 ± 0.025710; see `checkpoint_2b_outputs/20261025T113837678917Z/latent_cv.json` |
| CTGAN (user-confirmed record) | 0.8242 / 0.8903 / 0.8570 | 0.6119 / 0.4767 / 0.5359 | 0.6965 | 0.6835 | `[[211,26],[45,41]]` | Not supplied for this record; do not infer fold SD from it |
| CTGAN (archived checkpoint 3A) | 0.8214 / 0.8734 / 0.8466 | 0.5775 / 0.4767 / 0.5223 | 0.6845 | 0.6751 | `[[207,30],[45,41]]` | AUC 0.790326 ± 0.020227; per-fold records are in the checkpoint JSON |

No-augmentation class-1 metrics and confusion matrix are recovered from the saved checkpoint-2B OOF output; the user-confirmed AUC 0.7846 differs slightly from that artifact's 0.784880. User-confirmed CTGAN classwise metrics, macro F1, and balanced accuracy are derived directly from the supplied confusion matrix; its reported ROC-AUC is the user-supplied mean across folds, with no matching fold-level record found in the repository.

## Reproduction

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy==2.3.5 pandas==2.2.3 scikit-learn==1.8.0 tensorflow==2.21.0 pytest
python -m pytest -q
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
python .\latent_threshold_cv.py --data-dir . --output-root .\latent_threshold_outputs
```
