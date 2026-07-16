# OncoAssist Research Evaluation

## Proposed Model
- Raw AUC: 0.6579365079365079
- Calibrated AUC: 0.6674603174603174
- Calibrated Brier: 0.23000319750591028
- Calibrated ECE: 0.19893989406257634

## Reliability Separation
- Error rate (Reliable): 0.2391304347826087
- Error rate (Caution): 0.4857142857142857
- Coverage (Reliable): 0.5679012345679012
- Coverage (Inconclusive): 0.16049382716049382

## Notes
- CTGAN usage is tracked in selected_augmentation. Do not claim CTGAN unless source is 'ctgan'.
- Expression modality inference is heuristic only; verify with data dictionary.
- Auxiliary feature explainer does not claim biomarker validation.
- Label semantics verified by user: False
- External validation executed: False