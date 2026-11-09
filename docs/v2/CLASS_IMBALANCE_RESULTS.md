# V2 class-imbalance follow-up

Same selected multiview configuration per outer fold, fixed feasible total training size,
50% versus7.42% available positives, three paired samples, standard/weighted/oversampled
training. Calibration and evaluation retain natural approximately50% prevalence. Epoch
counts come from natural-prevalence inner development; no rare-condition tuning used outer
labels. AP baselines are each outer fold's recorded prevalence, not7.42%.

| model | calibration | operating | ap | auroc | brier | recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| imbalance/0.5/standard | none | fixed_050 | 0.78176 ± 0.00301 | 0.79907 ± 0.00210 | 0.18184 ± 0.00102 | 0.69008 ± 0.03063 | 0.73330 ± 0.00158 |
| imbalance/0.5/standard | none | balanced_accuracy | 0.78176 ± 0.00301 | 0.79907 ± 0.00210 | 0.18184 ± 0.00102 | 0.69870 ± 0.02114 | 0.73325 ± 0.00191 |
| imbalance/0.5/standard | platt | fixed_050 | 0.78176 ± 0.00301 | 0.79907 ± 0.00210 | 0.18136 ± 0.00095 | 0.68646 ± 0.01480 | 0.73378 ± 0.00176 |
| imbalance/0.5/standard | platt | balanced_accuracy | 0.78176 ± 0.00301 | 0.79907 ± 0.00210 | 0.18136 ± 0.00095 | 0.69500 ± 0.01822 | 0.73344 ± 0.00199 |
| imbalance/0.5/weighted | none | fixed_050 | 0.78189 ± 0.00291 | 0.79910 ± 0.00210 | 0.18182 ± 0.00101 | 0.68931 ± 0.03162 | 0.73314 ± 0.00167 |
| imbalance/0.5/weighted | none | balanced_accuracy | 0.78189 ± 0.00291 | 0.79910 ± 0.00210 | 0.18182 ± 0.00101 | 0.69984 ± 0.02102 | 0.73308 ± 0.00191 |
| imbalance/0.5/weighted | platt | fixed_050 | 0.78189 ± 0.00291 | 0.79910 ± 0.00210 | 0.18134 ± 0.00095 | 0.68585 ± 0.01523 | 0.73393 ± 0.00174 |
| imbalance/0.5/weighted | platt | balanced_accuracy | 0.78189 ± 0.00291 | 0.79910 ± 0.00210 | 0.18134 ± 0.00095 | 0.69516 ± 0.02031 | 0.73324 ± 0.00213 |
| imbalance/0.5/resampled | none | fixed_050 | 0.78084 ± 0.00273 | 0.79926 ± 0.00231 | 0.18208 ± 0.00123 | 0.69787 ± 0.03504 | 0.73297 ± 0.00177 |
| imbalance/0.5/resampled | none | balanced_accuracy | 0.78084 ± 0.00273 | 0.79926 ± 0.00231 | 0.18208 ± 0.00123 | 0.70223 ± 0.02176 | 0.73375 ± 0.00329 |
| imbalance/0.5/resampled | platt | fixed_050 | 0.78084 ± 0.00273 | 0.79926 ± 0.00231 | 0.18129 ± 0.00101 | 0.69072 ± 0.01297 | 0.73473 ± 0.00230 |
| imbalance/0.5/resampled | platt | balanced_accuracy | 0.78084 ± 0.00273 | 0.79926 ± 0.00231 | 0.18129 ± 0.00101 | 0.69855 ± 0.02068 | 0.73396 ± 0.00333 |
| imbalance/0.0742/standard | none | fixed_050 | 0.77429 ± 0.00678 | 0.79250 ± 0.00328 | 0.33632 ± 0.01423 | 0.01261 ± 0.01749 | 0.50519 ± 0.00709 |
| imbalance/0.0742/standard | none | balanced_accuracy | 0.77429 ± 0.00678 | 0.79250 ± 0.00328 | 0.33632 ± 0.01423 | 0.68236 ± 0.02295 | 0.72869 ± 0.00345 |
| imbalance/0.0742/standard | platt | fixed_050 | 0.77429 ± 0.00678 | 0.79250 ± 0.00328 | 0.18501 ± 0.00154 | 0.68284 ± 0.01421 | 0.72915 ± 0.00312 |
| imbalance/0.0742/standard | platt | balanced_accuracy | 0.77429 ± 0.00678 | 0.79250 ± 0.00328 | 0.18501 ± 0.00154 | 0.69095 ± 0.03083 | 0.72829 ± 0.00323 |
| imbalance/0.0742/weighted | none | fixed_050 | 0.77207 ± 0.00601 | 0.79120 ± 0.00393 | 0.18661 ± 0.00238 | 0.71381 ± 0.02927 | 0.72775 ± 0.00289 |
| imbalance/0.0742/weighted | none | balanced_accuracy | 0.77207 ± 0.00601 | 0.79120 ± 0.00393 | 0.18661 ± 0.00238 | 0.68618 ± 0.02688 | 0.72789 ± 0.00284 |
| imbalance/0.0742/weighted | platt | fixed_050 | 0.77207 ± 0.00601 | 0.79120 ± 0.00393 | 0.18532 ± 0.00191 | 0.69658 ± 0.01702 | 0.72889 ± 0.00295 |
| imbalance/0.0742/weighted | platt | balanced_accuracy | 0.77207 ± 0.00601 | 0.79120 ± 0.00393 | 0.18532 ± 0.00191 | 0.68991 ± 0.01316 | 0.72852 ± 0.00320 |
| imbalance/0.0742/resampled | none | fixed_050 | 0.75950 ± 0.00419 | 0.77088 ± 0.00420 | 0.20008 ± 0.00223 | 0.65587 ± 0.02091 | 0.71229 ± 0.00416 |
| imbalance/0.0742/resampled | none | balanced_accuracy | 0.75950 ± 0.00419 | 0.77088 ± 0.00420 | 0.20008 ± 0.00223 | 0.63708 ± 0.03014 | 0.71170 ± 0.00455 |
| imbalance/0.0742/resampled | platt | fixed_050 | 0.75950 ± 0.00419 | 0.77088 ± 0.00420 | 0.19447 ± 0.00188 | 0.69765 ± 0.00781 | 0.70991 ± 0.00596 |
| imbalance/0.0742/resampled | platt | balanced_accuracy | 0.75950 ± 0.00419 | 0.77088 ± 0.00420 | 0.19447 ± 0.00188 | 0.64257 ± 0.03229 | 0.71208 ± 0.00394 |

Ranking deterioration, prior/calibration mismatch and fixed-threshold failure are distinct.
Thresholding cannot alter AP/AUROC/Brier. Calibration cannot be assumed to improve ranking.
No artificially rare evaluation sample is presented as a real low-prevalence cohort, and
external rare-prevalence validation remains pending.
