# Calibration, thresholds and ensembles

Calibration mappings fit a dedicated outer-training calibration role, never reported outer
rows. Within-fold strictly increasing Platt/temperature preserve ranks; isotonic can tie
scores. Calibration intercept/slope are outer diagnostic regressions only, never refitted
predictions on those rows. Pooled cross-fold rank changes can also reflect different mappings.

| model | calibration | ap | auroc | brier | log_loss | ece_10 |
| --- | --- | --- | --- | --- | --- | --- |
| logistic/selected | none | 0.77272 ± 0.00476 | 0.79076 ± 0.00219 | 0.18734 ± 0.00090 | 0.56134 ± 0.00305 | 0.03382 ± 0.00171 |
| logistic/selected | platt | 0.77272 ± 0.00476 | 0.79076 ± 0.00219 | 0.18729 ± 0.00087 | 0.56142 ± 0.00313 | 0.03320 ± 0.00243 |
| logistic/selected | isotonic | 0.76530 ± 0.00447 | 0.78948 ± 0.00225 | 0.18613 ± 0.00085 | 0.55789 ± 0.00364 | 0.01406 ± 0.00314 |
| random_forest/selected | none | 0.77847 ± 0.00181 | 0.79616 ± 0.00175 | 0.18310 ± 0.00046 | 0.54770 ± 0.00113 | 0.01762 ± 0.00461 |
| random_forest/selected | platt | 0.77847 ± 0.00181 | 0.79616 ± 0.00175 | 0.18287 ± 0.00066 | 0.54690 ± 0.00178 | 0.01389 ± 0.00188 |
| random_forest/selected | isotonic | 0.77071 ± 0.00179 | 0.79525 ± 0.00192 | 0.18329 ± 0.00081 | 0.55117 ± 0.00386 | 0.01346 ± 0.00207 |
| xgboost/selected | none | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17988 ± 0.00030 | 0.53985 ± 0.00077 | 0.00935 ± 0.00157 |
| xgboost/selected | platt | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17986 ± 0.00032 | 0.53979 ± 0.00080 | 0.00896 ± 0.00251 |
| xgboost/selected | isotonic | 0.77827 ± 0.00186 | 0.80157 ± 0.00093 | 0.18019 ± 0.00025 | 0.54170 ± 0.00061 | 0.01145 ± 0.00314 |
| mlp/selected | none | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18141 ± 0.00060 | 0.54365 ± 0.00143 | 0.01285 ± 0.00306 |
| mlp/selected | platt | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18132 ± 0.00058 | 0.54342 ± 0.00142 | 0.01035 ± 0.00204 |
| mlp/selected | isotonic | 0.77638 ± 0.00252 | 0.79871 ± 0.00155 | 0.18177 ± 0.00062 | 0.54687 ± 0.00209 | 0.01288 ± 0.00248 |
| mlp/selected | temperature | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18141 ± 0.00061 | 0.54361 ± 0.00144 | 0.01250 ± 0.00315 |
| multiview/selected | none | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18082 ± 0.00084 | 0.54217 ± 0.00193 | 0.01931 ± 0.01096 |
| multiview/selected | platt | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18033 ± 0.00077 | 0.54108 ± 0.00182 | 0.01102 ± 0.00344 |
| multiview/selected | isotonic | 0.77766 ± 0.00263 | 0.80047 ± 0.00204 | 0.18068 ± 0.00082 | 0.54441 ± 0.00182 | 0.01265 ± 0.00258 |
| multiview/selected | temperature | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18083 ± 0.00085 | 0.54221 ± 0.00200 | 0.01998 ± 0.01103 |
| ft_transformer/selected | none | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18202 ± 0.00077 | 0.54467 ± 0.00187 | 0.01372 ± 0.00360 |
| ft_transformer/selected | platt | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18193 ± 0.00081 | 0.54445 ± 0.00202 | 0.01094 ± 0.00314 |
| ft_transformer/selected | isotonic | 0.77465 ± 0.00132 | 0.79732 ± 0.00147 | 0.18235 ± 0.00075 | 0.54820 ± 0.00247 | 0.01326 ± 0.00313 |
| ft_transformer/selected | temperature | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18200 ± 0.00079 | 0.54461 ± 0.00192 | 0.01309 ± 0.00294 |
| tabnet/selected | none | 0.77422 ± 0.00259 | 0.79233 ± 0.00134 | 0.18507 ± 0.00104 | 0.55217 ± 0.00248 | 0.01864 ± 0.00539 |
| tabnet/selected | platt | 0.77422 ± 0.00259 | 0.79233 ± 0.00134 | 0.18477 ± 0.00089 | 0.55150 ± 0.00227 | 0.01235 ± 0.00226 |
| tabnet/selected | isotonic | 0.76850 ± 0.00254 | 0.79173 ± 0.00135 | 0.18494 ± 0.00096 | 0.55479 ± 0.00408 | 0.01274 ± 0.00281 |

Thresholds use calibration labels, not outer outcomes. .50, balanced accuracy, F1,
precision≥.75/max-recall and recall≥.80/max-precision were registered. They do not change AP or
AUROC, and none is recommended for deployment.

| operating | recall | precision | specificity | balanced_accuracy | f1 |
| --- | --- | --- | --- | --- | --- |
| fixed_050 | 0.69613 ± 0.03855 | 0.75726 ± 0.02243 | 0.77505 ± 0.04031 | 0.73559 ± 0.00198 | 0.72425 ± 0.01041 |
| balanced_accuracy | 0.70145 ± 0.02386 | 0.75268 ± 0.01550 | 0.76888 ± 0.02630 | 0.73517 ± 0.00204 | 0.72569 ± 0.00615 |
| f1 | 0.80626 ± 0.02359 | 0.68777 ± 0.01587 | 0.63318 ± 0.03755 | 0.71972 ± 0.00742 | 0.74187 ± 0.00243 |
| precision_075 | 0.70217 ± 0.00922 | 0.75252 ± 0.00344 | 0.76930 ± 0.00673 | 0.73574 ± 0.00233 | 0.72643 ± 0.00402 |
| recall_080 | 0.80150 ± 0.00813 | 0.69005 ± 0.00300 | 0.64037 ± 0.00826 | 0.72093 ± 0.00159 | 0.74158 ± 0.00237 |

Inner-OOF predictions fit weights for registered combinations; equal/convex candidate choice
also occurs within inner OOF. Best single and best ensemble are per-outer **selection
procedures**, not retrospectively chosen outer winners. Mean AP 0.78431 versus
0.78414; paired difference/interval is in optimization comparisons.

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| selection/best_single | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17988 ± 0.00030 | 0.73519 ± 0.00182 | 0.69551 ± 0.00812 |
| ensemble/xgboost+multiview/equal | 0.78623 ± 0.00156 | 0.80324 ± 0.00119 | 0.17959 ± 0.00047 | 0.73694 ± 0.00170 | 0.69562 ± 0.02124 |
| ensemble/xgboost+multiview/convex | 0.78586 ± 0.00148 | 0.80321 ± 0.00098 | 0.17951 ± 0.00038 | 0.73702 ± 0.00173 | 0.69517 ± 0.01589 |
| ensemble/xgboost+ft_transformer/equal | 0.78478 ± 0.00198 | 0.80239 ± 0.00129 | 0.18004 ± 0.00052 | 0.73560 ± 0.00219 | 0.69279 ± 0.01392 |
| ensemble/xgboost+ft_transformer/convex | 0.78472 ± 0.00194 | 0.80252 ± 0.00128 | 0.17991 ± 0.00055 | 0.73563 ± 0.00220 | 0.69364 ± 0.01330 |
| ensemble/xgboost+tabnet/equal | 0.78222 ± 0.00145 | 0.80096 ± 0.00082 | 0.18094 ± 0.00030 | 0.73436 ± 0.00224 | 0.69181 ± 0.00664 |
| ensemble/xgboost+tabnet/convex | 0.78393 ± 0.00150 | 0.80220 ± 0.00083 | 0.18009 ± 0.00028 | 0.73576 ± 0.00219 | 0.69557 ± 0.00521 |
| ensemble/mlp+multiview/equal | 0.78602 ± 0.00133 | 0.80209 ± 0.00145 | 0.18023 ± 0.00056 | 0.73564 ± 0.00184 | 0.70171 ± 0.02275 |
| ensemble/mlp+multiview/convex | 0.78583 ± 0.00141 | 0.80200 ± 0.00161 | 0.18028 ± 0.00065 | 0.73575 ± 0.00193 | 0.70020 ± 0.02342 |
| ensemble/xgboost+multiview+ft_transformer/equal | 0.78632 ± 0.00187 | 0.80330 ± 0.00128 | 0.17958 ± 0.00049 | 0.73720 ± 0.00155 | 0.69671 ± 0.01720 |
| ensemble/xgboost+multiview+ft_transformer/convex | 0.78581 ± 0.00176 | 0.80317 ± 0.00099 | 0.17955 ± 0.00041 | 0.73715 ± 0.00163 | 0.69377 ± 0.01548 |
| selection/best_ensemble | 0.78414 ± 0.00151 | 0.80250 ± 0.00080 | 0.17994 ± 0.00034 | 0.73655 ± 0.00110 | 0.69208 ± 0.00825 |

A positive point difference alone is not proof of an ensemble benefit. Learned mappings and
weights remain private; all aggregate outer metrics and reproducible fitting code are public.
