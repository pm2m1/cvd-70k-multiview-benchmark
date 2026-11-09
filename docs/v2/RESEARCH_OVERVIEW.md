# CVD-70K Research V2

Two separate questions: improve internal ranking, and test seven-module selective adaptation
as labeled target support increases. All comparative records are real trained-model outputs.
V1 commit88c8020072fa3bf48d627bd3c2216d0b8e683181 remains intact. Branch
`research/v2-optimization-transfer`; main is not rewritten or automatically merged.

1054 successful V2 predictor fits, including operational
smokes; 495 registered inner fits. Successful summed fit time
2254.0s. Original690 V1 fits are separate. Counts do not turn repeated
seeds/folds/target draws into independent validation. Calibration mappings are separate fits.

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| logistic/selected | 0.77272 ± 0.00476 | 0.79076 ± 0.00219 | 0.18734 ± 0.00090 | 0.72653 ± 0.00062 | 0.67123 ± 0.00326 |
| random_forest/selected | 0.77847 ± 0.00181 | 0.79616 ± 0.00175 | 0.18310 ± 0.00046 | 0.72958 ± 0.00177 | 0.67266 ± 0.01348 |
| xgboost/selected | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17988 ± 0.00030 | 0.73519 ± 0.00182 | 0.69551 ± 0.00812 |
| mlp/selected | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18141 ± 0.00060 | 0.73354 ± 0.00164 | 0.70635 ± 0.01482 |
| multiview/selected | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18082 ± 0.00084 | 0.73559 ± 0.00198 | 0.69613 ± 0.03855 |
| ft_transformer/selected | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18202 ± 0.00077 | 0.73105 ± 0.00217 | 0.68142 ± 0.02100 |
| tabnet/selected | 0.77422 ± 0.00259 | 0.79233 ± 0.00134 | 0.18507 ± 0.00104 | 0.72573 ± 0.00253 | 0.67689 ± 0.01618 |

## Evidence map

- [Historical training audit](V1_TRAINING_DIAGNOSTIC.md)
- [Optimization protocol](OPTIMIZATION_PROTOCOL.md) and [actual results](OPTIMIZATION_RESULTS.md)
- [Modern tabular models](TABULAR_MODEL_COMPARISON.md)
- [Feature ablations](FEATURE_ENGINEERING_ABLATIONS.md)
- [Calibration/ensembles](CALIBRATION_AND_ENSEMBLES.md)
- [Imbalance](CLASS_IMBALANCE_RESULTS.md)
- [Transfer feasibility](TRANSFER_FEASIBILITY.md), [protocol](INTERNAL_TRANSFER_PROTOCOL.md),
  [actual selective results](SELECTIVE_TRANSFER_RESULTS.md)
- [External status](EXTERNAL_VALIDATION_STATUS.md)
- [Scientific conclusions](SCIENTIFIC_CONCLUSIONS.md)
- [Reproduction](REPRODUCIBILITY.md)

Data provenance/licence remains limited: official Kaggle metadata again saysUnknown. No raw
or per-person data/predictions, fitted models, learned preprocessing or credentials are public.
Only original code, invented fixtures, aggregate research outputs and figures are released.
V1 validation/test rows are never used for V2 model development or evaluation.
