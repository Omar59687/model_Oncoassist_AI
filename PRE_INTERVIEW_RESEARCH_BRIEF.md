# OncoAssist AI — pre-interview research brief

**Scope:** BLCA exploratory research extension; checkpoints 1, 2A, 2B, and 3A on `research/pre-interview-validation`, reviewed 2026-10-25. All new cross-validation (CV) estimates are **conditional on the released, preselected features**. Numeric `CLASS` codes have no verified clinical interpretation. This work is not ready for clinical use.

## Research idea and contribution boundaries

The [published AE-CTGAN study](https://doi.org/10.3390/a19020095), by Ibrahim Al-Hurani, Sara H. ElFar, Abedalrhman Alkhateeb, and Salama Ikki, proposed supervised feature selection, separate autoencoder representations for multi-omics modalities, latent-space GAN/CTGAN minority augmentation, and downstream neural-network prediction. It studied BLCA tumor mutational burden (TMB) and BRCA menopausal status. Its reported outcomes belong to the **paper's authors** and cannot be compared directly with the experiments here: cohort, feature provenance, evaluation, and classifier are not established as equivalent.

This repository is an **ongoing team research extension** of that work, with collaboration from one original researcher reported by the team. The existing [main implementation](oncoassist_competition_ready.py) and saved artifacts predate these checkpoints. The new work adds [provenance and evaluation warnings](RESEARCH_VALIDATION.md), [direct-feature CV baselines](baseline_cv.py), [fold-isolated autoencoder CV](latent_cv.py), and a [controlled augmentation comparison](ctgan_cv.py). These are evaluation and reliability contributions by the extension effort; they are **not** claims that the team invented AE-CTGAN or reproduced the paper. The repository is hosted under Omar's GitHub account, but the available record does not establish which individual wrote each component, who ran each experiment, or Omar's individual contribution. Confirm individual and team credit with the contributors before presenting it.

## What is verified, and what is not

- [Checkpoint 1](RESEARCH_VALIDATION.md) compared the three local CSVs with the [paper authors' released files at revision `0b55bfa`](https://github.com/Ibrahimalhurani/Dataset_and_Code/tree/0b55bfa24b608660562069802e12340370899b2b) after line-ending normalization. They have 404 shared unique sample IDs, consistent labels across modalities, 297 code-1 and 107 code-2 rows, and 23/42/14 selected omics features. The [paper's Table 2](https://doi.org/10.3390/a19020095) lists BLCA 97 Low-TMB and 107 High-TMB cases (204 total). The 200-row gap is unresolved; numerical count agreement does not establish code semantics or an exclusion rule.
- The [authors' released notebook](https://github.com/Ibrahimalhurani/Dataset_and_Code/blob/0b55bfa24b608660562069802e12340370899b2b/AE_CTGAN_BLCA.ipynb) includes `CLASS` in autoencoder inputs via `iloc[:, 1:]`, a confirmed target leak in that notebook. We cannot establish that this exact execution generated the paper's results. The paper describes supervised t-test, fold-change, and FDR feature selection, but released raw-selection code and timing are absent. Selection using all labels before splitting would leak; whether that happened is **unverified**.
- [Checkpoint 2A](baseline_cv.py) aligns by `SAMPLE_ID`, checks cross-modality labels and duplicate IDs, excludes identifiers and target from features, and reconstructs the original sorted-ID stratified split. Its 81 held-out samples are excluded; 323 training-pool samples (237 code 1, 86 code 2) enter fixed-seed five-fold stratified CV. Imputation and applicable scaling fit only fold training data. [Checkpoint 2B](latent_cv.py) additionally trains fresh, separate omics autoencoders within each fold, with early stopping on an inner training-fold split. Focused tests cover isolation, alignment, exclusions, and output metrics. These checks validate implemented boundaries, **not** upstream feature selection, biomarker meaning, causal biology, or external generalization.
- [Checkpoint 3A](ctgan_cv.py) compares no augmentation, bootstrap with small noise, and actual CTGAN on the same folds and fold-specific latent representations, with a fixed logistic classifier. Augmentation sees only each training fold. Reports record seeds, counts, versions, method actually used, per-fold metrics, and descriptive synthetic-data checks. The original saved [research report](export_artifacts/research_outputs/research_report.json) instead records `bootstrap_fallback`; it must not be described as a successful CTGAN run.

## Internal CV results

Values are **mean of five fold ROC-AUCs**, then **pooled out-of-fold (OOF)** accuracy, code-2 precision/recall/F1, and macro F1. The positive code is **2**, without a Low-/High-TMB assignment. Each row uses the same 323-sample training pool; no result below is from the separate 81-sample holdout. Sources: [2A output](checkpoint_2a_outputs/20261025T102245073104Z/baseline_cv.json), [2B output](checkpoint_2b_outputs/20261025T113837678917Z/latent_cv.json), [3A output](checkpoint_3a_outputs/20261025T120811006346Z/ctgan_comparison.json).

| Representation / classifier / augmentation | Mean fold ROC-AUC | OOF accuracy | Code-2 precision | Code-2 recall | Code-2 F1 | OOF macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct features / logistic / none | 0.721 | 0.693 | 0.441 | 0.570 | 0.497 | 0.638 |
| Direct features / SVM-RBF / none | 0.789 | 0.724 | 0.487 | 0.651 | 0.557 | 0.679 |
| Direct features / random forest / none | 0.769 | 0.746 | 0.611 | 0.128 | 0.212 | 0.530 |
| Fold-isolated latent / logistic / none | 0.785 | 0.737 | 0.505 | 0.651 | 0.569 | 0.690 |
| Fold-isolated latent / logistic / bootstrap | 0.783 | 0.731 | 0.496 | 0.651 | 0.563 | 0.684 |
| Fold-isolated latent / logistic / CTGAN | 0.790 | 0.768 | 0.577 | 0.477 | 0.522 | 0.684 |

These are descriptive internal CV comparisons, with no uncertainty analysis, external cohort, or justified clinical threshold. The latent/no-augmentation line is the same condition in 2B and 3A, not an independent replication.

## CTGAN diagnosis and trade-off

The old saved run reports `AssertionError` and bootstrap fallback. In the **current** SDV 1.38.5 / CTGAN 0.12.1 environment, [the smoke diagnostic](checkpoint_3a_outputs/20261025T120720007578Z/ctgan_smoke.json) reproduced an assertion in the discriminator: batch size 86 is not divisible by `pac=10`. The old run's exact dependency versions are not recorded, so its precise historical failure remains uncertain. The repair selects a valid even batch size divisible by `pac` (60 on these folds), and a separate SDV/PyTorch worker avoids an observed TensorFlow/CTGAN process crash. Five 100-epoch fold-specific CTGAN runs genuinely completed; failure now stops explicitly instead of switching augmentation methods.

Against latent/no augmentation, CTGAN changed mean fold AUC 0.785→0.790 and OOF accuracy 0.737→0.768, while code-2 recall fell 0.651→0.477 and code-2 F1 0.569→0.522; macro F1 fell 0.690→0.684. Bootstrap produced no clear gain. The CTGAN synthetic latents also showed large descriptive distribution gaps; those checks do not establish fidelity, privacy, or clinical plausibility. There is no demonstrated overall benefit under these metrics.

## Open questions and interview answers

1. **How do you know code 2 means High-TMB?** We do not. Class frequency is insufficient. Obtain the source data dictionary or per-sample TMB values, cutoff, and code mapping from the data producer.
2. **Why do released files contain 404 cases when Table 2 has 204?** All three released modalities align at 404, so cross-omics intersection is not the cause. We need the exact paper cohort ID list and inclusion/exclusion script to explain the 200 cases.
3. **Is this leakage-free?** Imputation, scaling, autoencoder fitting, and augmentation are isolated by CV fold, and the 81 held-out samples are unused here. The preselected features may have used labels across splits; raw features and selection code are needed for an end-to-end leakage-controlled estimate. The authors' notebook also exposes a separate direct target leak.
4. **Does CTGAN help the minority class?** Not consistently in this experiment. AUC and accuracy increased slightly, but code-2 recall and F1 declined. The fixed threshold, small folds, and uncertain synthetic fidelity limit interpretation; we should not choose a winner from these CV numbers and call it final test performance.
5. **What would make the finding persuasive?** Resolve labels and the cohort manifest; perform supervised feature selection inside every training fold from raw features, predefine primary metrics and decision rules, assess variability and synthetic-data quality, then evaluate a locked pipeline once on an untouched cohort and, ideally, independent external data. None of that validation is complete.

## 90-second meeting explanation

“Our starting point is the published AE-CTGAN study, which combines multi-omics autoencoders with synthetic minority-sample generation for cancer-outcome prediction. Our team is extending that research by checking data provenance and making evaluation more reliable, rather than claiming a new clinical model. We found that the released bladder-cancer files contain 404 aligned samples, while the paper reports 204 for its bladder cohort; we still need the authors' cohort manifest to explain the difference. We also cannot yet verify what numeric class codes 1 and 2 mean clinically.

“We constructed reproducible five-fold comparisons on a 323-sample training pool, leaving the original 81-sample holdout unused. Each fold fits imputation, scaling, and three autoencoders on training data only. We compared direct-feature baselines with an autoencoder-plus-logistic model, then compared that latent model without augmentation, with bootstrap oversampling, and with genuine CTGAN augmentation. We fixed CTGAN's batch and packing incompatibility and confirmed five real CTGAN runs. Its mean fold AUC rose slightly from 0.785 to 0.790, but code-2 recall fell from 0.651 to 0.477 and code-2 F1 from 0.569 to 0.522. That is a trade-off, not a demonstrated improvement.

“Our next priority is to verify labels, recover cohort exclusions and raw feature-selection provenance, then lock an end-to-end evaluation before any external validation or clinical interpretation.” 
