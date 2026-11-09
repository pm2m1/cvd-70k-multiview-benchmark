# V2 predictive optimization — nested internal evaluation

Only V1 training rows. Three outer folds, three inner folds, separate calibration roles.
Metrics below are mean ± SD across nine dependent fold×seed evaluations for selected models.
They are **not** the V1 opened test estimates. Historical V1 XGBoost AP .7696/AUROC .7887
cannot be subtracted from these CV numbers as a paired improvement. The matching refitted
V1-equivalent five-view gives AP 0.77872, AUROC 0.79540; selected
V2 five-view gives 0.78387/0.80136.

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| logistic/selected | 0.77272 ± 0.00476 | 0.79076 ± 0.00219 | 0.18734 ± 0.00090 | 0.72653 ± 0.00062 | 0.67123 ± 0.00326 |
| random_forest/selected | 0.77847 ± 0.00181 | 0.79616 ± 0.00175 | 0.18310 ± 0.00046 | 0.72958 ± 0.00177 | 0.67266 ± 0.01348 |
| xgboost/selected | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17988 ± 0.00030 | 0.73519 ± 0.00182 | 0.69551 ± 0.00812 |
| mlp/selected | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18141 ± 0.00060 | 0.73354 ± 0.00164 | 0.70635 ± 0.01482 |
| multiview/selected | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18082 ± 0.00084 | 0.73559 ± 0.00198 | 0.69613 ± 0.03855 |
| ft_transformer/selected | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18202 ± 0.00077 | 0.73105 ± 0.00217 | 0.68142 ± 0.02100 |
| tabnet/selected | 0.77422 ± 0.00259 | 0.79233 ± 0.00134 | 0.18507 ± 0.00104 | 0.72573 ± 0.00253 | 0.67689 ± 0.01618 |

| A − B | ΔAP | 95% group-bootstrap CI | ΔAUROC | 95% CI |
| --- | --- | --- | --- | --- |
| multiview/selected − multiview/v1_equivalent | +0.00515 | [0.00318, 0.00706] | +0.00596 | [0.00493, 0.007] |
| multiview/selected − mlp/selected | +0.00054 | [-0.00155, 0.00259] | +0.00169 | [0.00087, 0.00246] |
| multiview/selected − xgboost/selected | -0.00044 | [-0.00275, 0.00182] | -0.00086 | [-0.00168, -0.00011] |
| ft_transformer/selected − xgboost/selected | -0.00267 | [-0.00526, -0.00019] | -0.00400 | [-0.00499, -0.003] |
| tabnet/selected − xgboost/selected | -0.01009 | [-0.01276, -0.00766] | -0.00989 | [-0.01108, -0.00872] |
| selection/best_ensemble − selection/best_single | -0.00017 | [-0.00145, 0.00106] | +0.00028 | [-0.00014, 0.00067] |

Intervals resample feature groups within fixed outer folds and average paired seeds and
fold metrics. They condition on fitted models and fixed folds, do not quantify full nested
search uncertainty, and are not multiple-comparison adjusted. Neither folds nor seeds are
independent external populations. No outer result revises inner model selection.

## Measured resources

| Selected pipeline | Mean trainable parameters | Mean refit seconds | Mean outer CPU prediction seconds | Mean refit epochs/rounds |
| --- | --- | --- | --- | --- |
| ft_transformer/selected | 18089 | 9.598 | 0.0818 | 19.7 |
| logistic/selected | 17 | 0.010 | 0.7167 | 1.0 |
| mlp/selected | 9921 | 0.859 | 0.0058 | 7.7 |
| multiview/selected | 8268 | 1.001 | 0.0076 | 4.3 |
| random_forest/selected | N/A (tree) | 0.865 | 0.8802 | 1.0 |
| tabnet/selected | 23739 | 13.074 | 0.0980 | 23.7 |
| xgboost/selected | N/A (tree) | 0.099 | 0.7533 | 114.7 |

Times are descriptive on a shared machine with some concurrent jobs; inference includes
checkpoint loading and preprocessing. Tree scalar-count fields are N/A rather than invented
neural-equivalent parameter counts.

## Prespecified training/capacity effects, seed101 in all three outer folds

- 120 versus60 maximum epochs: ΔAP +0.00000; ΔAUROC +0.00000.
- 200 versus60 maximum epochs: ΔAP +0.00000; ΔAUROC +0.00000.
- AP versus BCE stopping (same120/patience16): ΔAP +0.00231; ΔAUROC +0.00109.
- AUROC versus BCE stopping: ΔAP +0.00011; ΔAUROC -0.00020.
- Batch256 versus1024 with AP stopping: ΔAP -0.00024; ΔAUROC +0.00046.
- Embedding32 versus16: ΔAP -0.00007; ΔAUROC -0.00086.
- Embedding64 versus16: ΔAP +0.00016; ΔAUROC -0.00113.

These are prespecified paired internal ablations, not independent clinical causal effects.
Regularized optimizer/schedule combinations are bundled candidates, so their individual
components are not causally identified. Flat-network capacity matching is approximate,
not exact at every width; actual parameter counts accompany every refit. Missingness gains
may exploit data-quality/collection artifacts and require independent sensitivity validation. Gradient clipping was not added to the V1-equivalent
control. Scratch epochzero is diagnostic only; refit epochs derive from inner folds.

| Outer | Chosen five-view candidate | Configuration | Median inner best epoch |
| --- | --- | --- | --- |
| 0 | missing_indicators | {'family': 'multiview', 'name': 'missing_indicators', 'embedding': 16, 'fusion': [64, 32], 'hidden': [96, 64], 'dropout': 0.0, 'lr': 0.001, 'batch': 256, 'epochs': 120, 'patience': 16, 'stop': 'ap', 'optimizer': 'adamw', 'weight_decay': 0.0001, 'schedule': 'constant', 'features': 'missing', 'scale': 'standard'} | 4 |
| 1 | missing_indicators | {'family': 'multiview', 'name': 'missing_indicators', 'embedding': 16, 'fusion': [64, 32], 'hidden': [96, 64], 'dropout': 0.0, 'lr': 0.001, 'batch': 256, 'epochs': 120, 'patience': 16, 'stop': 'ap', 'optimizer': 'adamw', 'weight_decay': 0.0001, 'schedule': 'constant', 'features': 'missing', 'scale': 'standard'} | 4 |
| 2 | robust_combined | {'family': 'multiview', 'name': 'robust_combined', 'embedding': 16, 'fusion': [64, 32], 'hidden': [96, 64], 'dropout': 0.0, 'lr': 0.001, 'batch': 256, 'epochs': 120, 'patience': 16, 'stop': 'ap', 'optimizer': 'adamw', 'weight_decay': 0.0001, 'schedule': 'constant', 'features': 'both', 'scale': 'robust'} | 5 |

| model | ap | auroc | brier | best_epoch | fit_seconds |
| --- | --- | --- | --- | --- | --- |
| multiview/v1_equivalent | 0.77941 ± 0.00056 | 0.79594 ± 0.00202 | 0.18308 ± 0.00073 | 13.33333 ± 4.04145 | 0.84586 ± 0.15264 |
| multiview/epochs120 | 0.77941 ± 0.00056 | 0.79594 ± 0.00202 | 0.18308 ± 0.00073 | 13.33333 ± 4.04145 | 0.79192 ± 0.22861 |
| multiview/epochs200 | 0.77941 ± 0.00056 | 0.79594 ± 0.00202 | 0.18308 ± 0.00073 | 13.33333 ± 4.04145 | 0.78703 ± 0.21347 |
| multiview/patience16 | 0.77657 ± 0.00191 | 0.79460 ± 0.00285 | 0.18421 ± 0.00165 | 19.00000 ± 7.21110 | 1.09750 ± 0.40535 |
| multiview/patience24 | 0.77657 ± 0.00191 | 0.79460 ± 0.00285 | 0.18421 ± 0.00165 | 19.00000 ± 7.21110 | 1.11506 ± 0.42798 |
| multiview/ap_stop | 0.77888 ± 0.00113 | 0.79569 ± 0.00132 | 0.18329 ± 0.00043 | 15.00000 ± 5.29150 | 0.94114 ± 0.36402 |
| multiview/auroc_stop | 0.77668 ± 0.00152 | 0.79440 ± 0.00196 | 0.18389 ± 0.00077 | 20.66667 ± 4.04145 | 1.27239 ± 0.26572 |
| multiview/batch256 | 0.77864 ± 0.00169 | 0.79615 ± 0.00218 | 0.18380 ± 0.00178 | 4.33333 ± 1.15470 | 1.00394 ± 0.25287 |
| multiview/width32 | 0.77857 ± 0.00104 | 0.79529 ± 0.00122 | 0.18425 ± 0.00153 | 3.00000 ± 1.73205 | 0.70001 ± 0.38447 |
| multiview/width64 | 0.77880 ± 0.00305 | 0.79502 ± 0.00182 | 0.18420 ± 0.00136 | 2.33333 ± 1.52753 | 0.58361 ± 0.36433 |
| multiview/regularized32 | 0.78181 ± 0.00235 | 0.79730 ± 0.00082 | 0.18249 ± 0.00033 | 6.66667 ± 3.05505 | 1.51674 ± 0.66649 |
| multiview/adam_plateau | 0.77782 ± 0.00358 | 0.79685 ± 0.00161 | 0.18270 ± 0.00087 | 15.33333 ± 10.96966 | 3.35942 ± 2.35456 |
| multiview/slow_regularized | 0.77530 ± 0.00221 | 0.79443 ± 0.00227 | 0.18436 ± 0.00104 | 59.33333 ± 7.76745 | 7.11461 ± 1.17273 |
| multiview/fast_small_batch | 0.77915 ± 0.00269 | 0.79650 ± 0.00178 | 0.18319 ± 0.00081 | 3.66667 ± 1.52753 | 1.91080 ± 0.84271 |
| multiview/derived_features | 0.78075 ± 0.00304 | 0.79725 ± 0.00155 | 0.18300 ± 0.00019 | 3.33333 ± 1.15470 | 0.87141 ± 0.38634 |
| multiview/missing_indicators | 0.78312 ± 0.00162 | 0.80112 ± 0.00210 | 0.18092 ± 0.00128 | 5.66667 ± 2.88675 | 1.35396 ± 0.78427 |
| multiview/combined_features | 0.78310 ± 0.00306 | 0.80054 ± 0.00183 | 0.18114 ± 0.00070 | 4.66667 ± 1.15470 | 1.05254 ± 0.24268 |
| multiview/robust_combined | 0.78348 ± 0.00215 | 0.80097 ± 0.00247 | 0.18123 ± 0.00141 | 4.33333 ± 0.57735 | 0.97416 ± 0.15517 |
| imbalance/0.5/standard | 0.78323 ± 0.00081 | 0.79941 ± 0.00298 | 0.18166 ± 0.00170 | nan ± nan | nan ± nan |
| imbalance/0.5/weighted | 0.78339 ± 0.00101 | 0.79953 ± 0.00293 | 0.18161 ± 0.00168 | nan ± nan | nan ± nan |
| imbalance/0.5/resampled | 0.78152 ± 0.00151 | 0.79937 ± 0.00348 | 0.18235 ± 0.00187 | nan ± nan | nan ± nan |
| imbalance/0.0742/standard | 0.77587 ± 0.00144 | 0.79413 ± 0.00071 | 0.32846 ± 0.00870 | nan ± nan | nan ± nan |
| imbalance/0.0742/weighted | 0.77287 ± 0.00460 | 0.79116 ± 0.00254 | 0.18751 ± 0.00298 | nan ± nan | nan ± nan |
| imbalance/0.0742/resampled | 0.76319 ± 0.00352 | 0.77493 ± 0.00227 | 0.19840 ± 0.00245 | nan ± nan | nan ± nan |

Exact best epochs, costs, generalization differences between inner and outer AP, parameters
and all curves are in trial records and `selected_hyperparameters.json`. `parameter_count`
is N/A for trees. `inference_seconds` measures an actual end-to-end CPU outer prediction,
including checkpoint load/reconstruction and preprocessing; it is not a deployed latency
benchmark. Three separate development-only diagnostic refits record genuine train AP/AUROC
and validation AP/AUROC curves without rewriting historical logs or changing selection.
Their weight differences from matching original inner checkpoints are recorded explicitly. Larger search budgets
also increase selection opportunity; no extrapolation to an artificial .90 target is made.
