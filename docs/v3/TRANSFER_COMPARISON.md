# Internal age-subgroup transfer comparison

| positive_budget | model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | full | 0.78582 | 0.72436 | 0.21803 | 0.67616 | 0.60891 |
| 25 | full_cost | 0.78974 | 0.72883 | 0.21028 | 0.68256 | 0.61741 |
| 25 | full_sp_high | 0.78735 | 0.72685 | 0.21519 | 0.67956 | 0.61305 |
| 25 | full_sp_low | 0.78609 | 0.72466 | 0.21770 | 0.67626 | 0.60981 |
| 25 | head | 0.79206 | 0.73255 | 0.20576 | 0.68259 | 0.64464 |
| 25 | logical | 0.79232 | 0.73268 | 0.21147 | 0.68212 | 0.61049 |
| 25 | logical_sp_low | 0.79231 | 0.73268 | 0.21146 | 0.68212 | 0.61049 |
| 25 | random_matched | 0.79210 | 0.73243 | 0.20592 | 0.68177 | 0.64080 |
| 25 | reg_combined | 0.78464 | 0.72279 | 0.21885 | 0.67592 | 0.60530 |
| 25 | selected_new_policy | 0.79065 | 0.73075 | 0.21429 | 0.68047 | 0.60395 |
| 25 | soft_evidence | 0.79170 | 0.73178 | 0.20830 | 0.68335 | 0.62711 |
| 25 | soft_learned | 0.78609 | 0.72466 | 0.21770 | 0.67626 | 0.60981 |
| 25 | soft_uniform | 0.78538 | 0.72361 | 0.21281 | 0.67768 | 0.63576 |
| 25 | source_only | 0.79221 | 0.73261 | 0.20589 | 0.68288 | 0.64441 |
| 25 | target_lr_cost | 0.76786 | 0.70405 | 0.22381 | 0.65791 | 0.59598 |
| 25 | target_xgb_cost | 0.77073 | 0.70374 | 0.22094 | 0.66036 | 0.62515 |
| 250 | full | 0.79126 | 0.73122 | 0.21012 | 0.68268 | 0.63704 |
| 250 | full_cost | 0.79034 | 0.72989 | 0.21009 | 0.68279 | 0.62252 |
| 250 | full_sp_high | 0.79201 | 0.73256 | 0.20868 | 0.68407 | 0.63042 |
| 250 | full_sp_low | 0.79116 | 0.73107 | 0.21006 | 0.68276 | 0.63824 |
| 250 | head | 0.79214 | 0.73257 | 0.20846 | 0.68445 | 0.62433 |
| 250 | logical | 0.79093 | 0.73153 | 0.20912 | 0.68322 | 0.62095 |
| 250 | logical_sp_low | 0.79093 | 0.73148 | 0.20936 | 0.68363 | 0.61944 |
| 250 | random_matched | 0.79131 | 0.73180 | 0.20885 | 0.68338 | 0.62102 |
| 250 | reg_combined | 0.79172 | 0.73229 | 0.20945 | 0.68365 | 0.62877 |
| 250 | selected_new_policy | 0.79155 | 0.73214 | 0.20928 | 0.68451 | 0.62410 |
| 250 | soft_evidence | 0.79184 | 0.73271 | 0.20936 | 0.68398 | 0.62456 |
| 250 | soft_learned | 0.79116 | 0.73107 | 0.21006 | 0.68276 | 0.63824 |
| 250 | soft_uniform | 0.79096 | 0.73137 | 0.20970 | 0.68455 | 0.63335 |
| 250 | source_only | 0.79221 | 0.73261 | 0.20589 | 0.68288 | 0.64441 |
| 250 | target_lr_cost | 0.78932 | 0.73348 | 0.21114 | 0.68651 | 0.62809 |
| 250 | target_xgb_cost | 0.78937 | 0.72953 | 0.21017 | 0.68115 | 0.64072 |
| 1000 | full | 0.79227 | 0.73263 | 0.20845 | 0.68418 | 0.62809 |
| 1000 | full_cost | 0.79158 | 0.73369 | 0.20926 | 0.68534 | 0.61253 |
| 1000 | full_sp_high | 0.79197 | 0.73312 | 0.20820 | 0.68409 | 0.61839 |
| 1000 | full_sp_low | 0.79245 | 0.73250 | 0.20828 | 0.68393 | 0.63501 |
| 1000 | head | 0.79208 | 0.73253 | 0.20795 | 0.68365 | 0.61854 |
| 1000 | logical | 0.79258 | 0.73343 | 0.20874 | 0.68456 | 0.61967 |
| 1000 | logical_sp_low | 0.79238 | 0.73305 | 0.20861 | 0.68462 | 0.61666 |
| 1000 | random_matched | 0.79259 | 0.73333 | 0.20880 | 0.68456 | 0.61945 |
| 1000 | reg_combined | 0.79212 | 0.73339 | 0.20825 | 0.68443 | 0.61501 |
| 1000 | selected_new_policy | 0.79183 | 0.73320 | 0.20840 | 0.68401 | 0.61230 |
| 1000 | soft_evidence | 0.79167 | 0.73310 | 0.20787 | 0.68496 | 0.62779 |
| 1000 | soft_learned | 0.79245 | 0.73250 | 0.20828 | 0.68393 | 0.63501 |
| 1000 | soft_uniform | 0.79228 | 0.73254 | 0.20767 | 0.68377 | 0.63644 |
| 1000 | source_only | 0.79221 | 0.73261 | 0.20589 | 0.68288 | 0.64441 |
| 1000 | target_lr_cost | 0.79148 | 0.73674 | 0.20948 | 0.68685 | 0.63005 |
| 1000 | target_xgb_cost | 0.79550 | 0.73668 | 0.20748 | 0.68800 | 0.64396 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Primary P1000 conditional paired result:
```json
{
  "ap_difference": 0.00025659389438004165,
  "auroc_difference": -0.00048821004865366263,
  "ap_ci95": [
    -0.0013491807279656764,
    0.0019790141413158387
  ],
  "auroc_ci95": [
    -0.0015815087928392358,
    0.0007022669028068389
  ],
  "replicates": 1000,
  "unit": "modeled-feature group, stratified by fixed outer fold",
  "scope": "conditional on fitted models; exploratory; not external validation or independent-fold inference"
}
```

Policies selected only on inner target model-selection labels:
```json
{
  "o0_p25": "full_sp_low",
  "o0_p250": "reg_mmd",
  "o0_p1000": "soft_evidence",
  "o1_p25": "full_sp_low",
  "o1_p250": "soft_evidence",
  "o1_p1000": "reg_combined",
  "o2_p25": "soft_evidence",
  "o2_p250": "reg_combined",
  "o2_p1000": "reg_combined"
}
```

Natural target evaluation prevalence differs from balanced training/selection roles; AP baseline is recorded for every fold. No threshold was selected on outer data; .50 is descriptive. Repeated source-only entries reuse the same checkpoint, not independent fits.
