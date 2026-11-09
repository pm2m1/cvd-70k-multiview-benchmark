# Prespecified ablation results

| model | ap | auroc | brier |
| --- | --- | --- | --- |
| age_spline | 0.78418 | 0.80080 | 0.18112 |
| nonlinear | 0.78245 | 0.79955 | 0.18165 |
| quantile | 0.78240 | 0.80114 | 0.18104 |
| shuffled_quality | 0.77804 | 0.79446 | 0.18443 |
| shuffled_quality_strict_common | 0.77867 | 0.79797 | 0.18245 |
| strict_reference_strict_common | 0.78083 | 0.79984 | 0.18190 |
| v1_reference_strict_common | 0.78086 | 0.80036 | 0.18104 |
| v2_reference | 0.78230 | 0.79963 | 0.18186 |
| v2_reference_strict_common | 0.78037 | 0.79926 | 0.18195 |


| positive_budget | model | ap | auroc |
| --- | --- | --- | --- |
| 25 | fusion1_only | 0.78637 | 0.72574 |
| 25 | fusion2_only | 0.79159 | 0.73171 |
| 25 | fusion_both | 0.78568 | 0.72418 |
| 25 | reg_shuffled | 0.78538 | 0.72388 |
| 25 | soft_random | 0.79053 | 0.73045 |
| 25 | soft_shuffled | 0.79174 | 0.73200 |
| 25 | view_anthro | 0.79228 | 0.73265 |
| 25 | view_background | 0.79210 | 0.73255 |
| 25 | view_behaviour | 0.79204 | 0.73243 |
| 25 | view_hemo | 0.79207 | 0.73242 |
| 25 | view_metabolic | 0.79229 | 0.73267 |
| 250 | fusion1_only | 0.78930 | 0.72897 |
| 250 | fusion2_only | 0.79069 | 0.73033 |
| 250 | fusion_both | 0.78893 | 0.72899 |
| 250 | reg_shuffled | 0.79194 | 0.73251 |
| 250 | soft_random | 0.79105 | 0.73159 |
| 250 | soft_shuffled | 0.79175 | 0.73256 |
| 250 | view_anthro | 0.79224 | 0.73262 |
| 250 | view_background | 0.79223 | 0.73256 |
| 250 | view_behaviour | 0.79223 | 0.73265 |
| 250 | view_hemo | 0.79230 | 0.73268 |
| 250 | view_metabolic | 0.79239 | 0.73283 |
| 1000 | fusion1_only | 0.79174 | 0.73241 |
| 1000 | fusion2_only | 0.79192 | 0.73304 |
| 1000 | fusion_both | 0.79218 | 0.73259 |
| 1000 | reg_shuffled | 0.79222 | 0.73335 |
| 1000 | soft_random | 0.79201 | 0.73245 |
| 1000 | soft_shuffled | 0.79204 | 0.73302 |
| 1000 | view_anthro | 0.79175 | 0.73233 |
| 1000 | view_background | 0.79216 | 0.73340 |
| 1000 | view_behaviour | 0.79175 | 0.73258 |
| 1000 | view_hemo | 0.79219 | 0.73291 |
| 1000 | view_metabolic | 0.79233 | 0.73293 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Quality indicators arise from analyst masking of implausible measurements. Their association with outcomes may reflect cleaning/acquisition artifacts. Shuffling does not remove all artifacts because imputed values may themselves reveal invalidity. The strict-common subset comparisons share evaluation rows with each other but must not be ranked against full-population AP. Splines and nonlinear interactions add no independent clinical measurement; all learned transformations fit on inner training records only.
