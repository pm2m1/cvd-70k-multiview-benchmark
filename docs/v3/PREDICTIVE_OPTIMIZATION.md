# Predictive optimization

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| v1_reference | 0.77994 | 0.79638 | 0.18307 | 0.72832 | 0.68609 |
| v2_reference | 0.78230 | 0.79963 | 0.18186 | 0.73256 | 0.69255 |
| gated | 0.78172 | 0.79946 | 0.18156 | 0.73503 | 0.69806 |
| residual | 0.78110 | 0.79880 | 0.18233 | 0.73052 | 0.64419 |
| attention | 0.78305 | 0.80017 | 0.18162 | 0.73368 | 0.68626 |
| pairwise | 0.78274 | 0.79974 | 0.18170 | 0.73238 | 0.65955 |
| flat_attention_matched | 0.78305 | 0.80007 | 0.18140 | 0.73298 | 0.67723 |
| ft_reference | 0.78529 | 0.80202 | 0.18027 | 0.73540 | 0.69152 |
| selected_xgboost | 0.78629 | 0.80338 | 0.17954 | 0.73770 | 0.69099 |
| selected_lightgbm | 0.78630 | 0.80282 | 0.17989 | 0.73643 | 0.68687 |
| selected_catboost | 0.78569 | 0.80302 | 0.17967 | 0.73591 | 0.69438 |
| selected_neural_seed_average | 0.78383 | 0.80108 | 0.18091 | 0.73453 | 0.67515 |
| selected_tree_seed_average | 0.78609 | 0.80325 | 0.17971 | 0.73631 | 0.68956 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Best individual AP point estimate (post-hoc descriptive): {'model': 'lightgbm_1', 'ap': 0.7870275852769693, 'auroc': 0.8022134228366253, 'brier': 0.18048974159120731, 'ece_10': 0.021225128456970607, 'balanced_accuracy': 0.7370657588098518, 'recall': 0.6849088851144186, 'specificity': 0.7892226325052851, 'precision': 0.7644828949916603}

Primary procedure comparisons:
```json
{
  "selected_neural_seed_average minus v2_reference": {
    "ap_difference": 0.0015291687256172952,
    "auroc_difference": 0.0014571015426699436,
    "ap_ci95": [
      -0.0007131556909468845,
      0.003947842201326114
    ],
    "auroc_ci95": [
      0.0006249772223577366,
      0.0023670112805537877
    ],
    "replicates": 1000,
    "unit": "modeled-feature group, stratified by fixed outer fold",
    "scope": "conditional on fitted models; exploratory; not external validation or independent-fold inference"
  },
  "selected_tree_seed_average minus v2_reference": {
    "ap_difference": 0.00378876275346407,
    "auroc_difference": 0.0036269982997837826,
    "ap_ci95": [
      0.0013471838387664554,
      0.006442764047918025
    ],
    "auroc_ci95": [
      0.0026307417317949964,
      0.004611186372213502
    ],
    "replicates": 1000,
    "unit": "modeled-feature group, stratified by fixed outer fold",
    "scope": "conditional on fitted models; exploratory; not external validation or independent-fold inference"
  }
}
```

V1/V2 references are configurations refitted on V3 roles, not historical final-test scores. The V2 reference uses its missing-indicator configuration; V2 also selected robust combined features in one historical fold. Search family budgets differ; no universal architecture superiority is inferred.
