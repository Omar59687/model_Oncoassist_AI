# Nested autoencoder, CTGAN, and RBF-SVM comparison

**Run status:** blocked_missing_tensorflow

## Protocol and safeguards

- Same original 323-sample training pool and five stratified outer folds (seed 42); original 81-sample holdout excluded from fitting, selection, and evaluation.
- Three inner folds per outer training partition select the SVM settings and Class-2 threshold separately for no-augmentation and CTGAN arms. The adaptive policy selects its augmentation arm using inner-OOF Class-2 F1 only.
- A fresh set of three modality autoencoders is fitted per inner split and per outer fold. Imputation, scaling, autoencoder early stopping, and SVM fitting use only the respective training partition.
- CTGAN is fitted only to the Class-2 latent vectors from that training partition, using the existing isolated CPU worker. Synthetic rows are training data only and never validation observations.
- The 20-epoch CTGAN setting and small fixed SVM grid were chosen for laptop runtime. These are exploratory, not claimed optimal settings.
- Labels remain neutral CLASS codes. The released 79-feature set has uncertain upstream supervised-selection provenance; this is not an end-to-end leakage-controlled estimate.

## Execution outcome

TensorFlow is unavailable; no autoencoders, CTGAN, or model comparisons were run.

No new performance score is claimed. See the JSON run record for the dependency or execution error.

## Current reference result supplied for the no-augmentation tuned RBF-SVM

Accuracy 73.99%; Class-2 precision 50.85%; recall 69.77%; F1 58.82%; pooled ROC-AUC 0.8087. These values were supplied in the task prompt and serve as context; do not treat them as the result of this script unless its run record says so.

## PowerShell reproduction (Python 3.12, TensorFlow 2.20.0)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn tensorflow==2.20.0 sdv==1.38.5 ctgan==0.12.1 torch pytest
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
python -m pytest -q
python .\ctgan_svm_nested_cv.py --data-dir . --output-root .\ctgan_svm_nested_outputs --ctgan-epochs 20
```
