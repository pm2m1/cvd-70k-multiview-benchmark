# Locked empirical results

Generated from 680 successfully fitted/evaluated models plus 10 validation
search fits. Each reported value is computed from actual dataset rows and saved models.
Five seeds share one frozen held-out test: these are not five independent validations.

The validation-selected winner is **XGBoost**, mean validation AP
0.785829. Its mean final test AP is 0.769569, AUROC
0.788665, Brier 0.186963. The primary test has 10,501
records, 5,247 positives, prevalence 0.499667. The
AP no-skill baseline is this approximately 50% prevalence. Test results do not select or
revise the winner.

| Model | AP mean ± seed SD | AP group-bootstrap 95% CI | AUROC | Brier | Recall @.50 | Balanced accuracy @.50 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic | 0.7573 ± 0.0000 | [0.7443, 0.7718] | 0.7788 ± 0.0000 | 0.1925 ± 0.0000 | 0.6600 ± 0.0000 | 0.7143 ± 0.0000 |
| Random forest | 0.7671 ± 0.0009 | [0.7538, 0.7818] | 0.7853 ± 0.0004 | 0.1882 ± 0.0001 | 0.6794 ± 0.0012 | 0.7215 ± 0.0010 |
| XGBoost | 0.7696 ± 0.0008 | [0.7567, 0.7842] | 0.7887 ± 0.0002 | 0.1870 ± 0.0000 | 0.6773 ± 0.0013 | 0.7216 ± 0.0004 |
| Flat MLP | 0.7651 ± 0.0012 | [0.7525, 0.7785] | 0.7849 ± 0.0005 | 0.1886 ± 0.0003 | 0.6891 ± 0.0055 | 0.7200 ± 0.0012 |
| Five-view MLP | 0.7656 ± 0.0011 | [0.7533, 0.7789] | 0.7856 ± 0.0014 | 0.1883 ± 0.0008 | 0.6935 ± 0.0160 | 0.7192 ± 0.0008 |
| Five-view MLP (weighted) | 0.7655 ± 0.0015 | [0.7532, 0.7791] | 0.7855 ± 0.0010 | 0.1883 ± 0.0007 | 0.6896 ± 0.0169 | 0.7203 ± 0.0007 |
| Five-view MLP (resampled) | 0.7655 ± 0.0024 | [0.7532, 0.7790] | 0.7859 ± 0.0011 | 0.1880 ± 0.0004 | 0.6842 ± 0.0073 | 0.7208 ± 0.0016 |

Intervals use 1,000 paired feature-group bootstrap samples and average seed AP inside each
sample. Seed SD measures fit variation on the same records; intervals condition on trained
models. Groups approximate participants because actual person linkage is unavailable.
For standard five-view minus flat MLP, mean paired bootstrap AP difference is
+0.000494, 95% interval [-0.002634, +0.003813].
Interpret this descriptive interval without a claim of external/clinical superiority.
Full machine-readable metrics include log loss, ECE, F1, precision, specificity, accuracy,
confusion counts and duration for **every fit** at both operating points.

## Validation-selected thresholds for the selected model

Maximize validation balanced accuracy on .01… .99, ties closest to .50 then lower.
Thresholds were frozen before final testing. Values below describe test performance at
those validation-derived points. Different seeds have separate fitted thresholds.

| seed | threshold | recall | specificity | balanced_accuracy | precision | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 101.0000 | 0.4400 | 0.7351 | 0.6997 | 0.7174 | 0.7097 | 0.7221 |
| 202.0000 | 0.4800 | 0.6979 | 0.7427 | 0.7203 | 0.7304 | 0.7138 |
| 303.0000 | 0.4600 | 0.7172 | 0.7195 | 0.7183 | 0.7185 | 0.7179 |
| 404.0000 | 0.4500 | 0.7280 | 0.7094 | 0.7187 | 0.7144 | 0.7212 |
| 505.0000 | 0.4500 | 0.7244 | 0.7113 | 0.7178 | 0.7147 | 0.7195 |

## Imbalance experiments

All 600 planned fits completed: two sampling designs × four fractions × five model
families × three interventions × five paired seeds. Original validation and test stay
approximately balanced. AP values are comparable here because the test rows and prevalence
are identical; they must not be compared directly with Phase 1 MHAS AP as if the tasks or
AP baselines matched. Class weighting and oversampling alter the effective training prior;
calibration and recall can change even without ranking improvement.

- fixed_total, standard: mean AP 0.7645 at 50% → 0.7570 at 7.42%; rare-condition recall 0.0079 at .50 and Brier 0.3431.
- fixed_total, weighted: mean AP 0.7645 at 50% → 0.7616 at 7.42%; rare-condition recall 0.6702 at .50 and Brier 0.1899.
- fixed_total, resampled: mean AP 0.7642 at 50% → 0.7573 at 7.42%; rare-condition recall 0.6772 at .50 and Brier 0.1913.
- fixed_positive, standard: mean AP 0.7551 at 50% → 0.7559 at 7.42%; rare-condition recall 0.0111 at .50 and Brier 0.3391.
- fixed_positive, weighted: mean AP 0.7551 at 50% → 0.7610 at 7.42%; rare-condition recall 0.6890 at .50 and Brier 0.1898.
- fixed_positive, resampled: mean AP 0.7556 at 50% → 0.7578 at 7.42%; rare-condition recall 0.6808 at .50 and Brier 0.1907.

[Full multiview table](../reports/imbalance_multiview.md) gives both threshold
operating points. [Condition summary](../aggregate/condition_summary.csv) includes
all five families; paired multiview/flat intervals for all 24 conditions are in
`bootstrap_imbalance.json`, and rare-condition intervention intervals in
`bootstrap_rare_interventions.json`. These many comparisons are descriptive.

Validation threshold selection alone substantially restores rare-condition standard
multiview performance: mean balanced accuracy is 0.7118
(A) and 0.7126 (B), versus
0.5034 and 0.5047 at .50.
Mean selected thresholds are 0.078 and
0.086. A threshold
change leaves AP and Brier unchanged. Weighting improves Brier under this *balanced*
evaluation prior; this does not establish calibration at a rare-prevalence deployment site.

The standard flat MLP restored epoch 1 in 10 of
10 7.42% fits under the prespecified
unweighted natural-validation BCE stopping rule and shared full-training hyperparameters.
Its poor rare-condition AP is therefore specific to this training/stopping protocol.
This observed checkpoint pattern does not prove an inherent multiview advantage; alternative
stopping objectives or condition-specific tuning would require a new validation-only study.

A holds 26,478 rows fixed; positives decline from 13,239 to 1,965, so positive support and
negative support change together. B holds 1,964 positives fixed, increasing total from
3,928 to 26,469. Both designs are constrained by the available 24,514 training negatives.
Neither is a pure causal prevalence manipulation independent of sample support.

## Cleaning and anthropometric sensitivity

| Analysis | Cleaning | Anthropometry | Model | Test n | Test prevalence | AP | AUROC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict | all | logistic | 10276 | 0.4942 | 0.7584 ± 0.0000 | 0.7831 ± 0.0000 | 0.1905 ± 0.0000 |
| full | strict | all | random_forest | 10276 | 0.4942 | 0.7690 ± 0.0006 | 0.7911 ± 0.0002 | 0.1855 ± 0.0001 |
| full | strict | all | xgboost | 10276 | 0.4942 | 0.7710 ± 0.0006 | 0.7939 ± 0.0002 | 0.1844 ± 0.0000 |
| full | strict | all | mlp | 10276 | 0.4942 | 0.7663 ± 0.0019 | 0.7902 ± 0.0005 | 0.1860 ± 0.0003 |
| full | strict | all | multiview | 10276 | 0.4942 | 0.7667 ± 0.0026 | 0.7908 ± 0.0013 | 0.1857 ± 0.0004 |
| anthropometry | minimal | bmi_only | mlp | 10501 | 0.4997 | 0.7647 ± 0.0009 | 0.7854 ± 0.0003 | 0.1883 ± 0.0001 |
| anthropometry | minimal | bmi_only | multiview | 10501 | 0.4997 | 0.7644 ± 0.0013 | 0.7857 ± 0.0004 | 0.1879 ± 0.0003 |
| anthropometry | minimal | parents_only | mlp | 10501 | 0.4997 | 0.7657 ± 0.0020 | 0.7857 ± 0.0006 | 0.1882 ± 0.0003 |
| anthropometry | minimal | parents_only | multiview | 10501 | 0.4997 | 0.7656 ± 0.0013 | 0.7850 ± 0.0006 | 0.1884 ± 0.0004 |

Strict results use fewer held-out records and a different test prevalence. Their raw AP
cannot be read as a paired benefit from cleaning. BMI sensitivities use identical primary
test rows and compare dropping derived BMI versus dropping its parents. Cleaning and
feature choices were prespecified; no sensitivity is selected after test inspection.

## Runtime and artifacts

Summed fit time is 543.6 seconds, including search, across
690 completed fits. This excludes extraction, reporting, process
startup outside recorded tabular fit time, and the failed native attempt. See
`runtime.json`, `hardware.json`, `development_lock.json` and `test_access.json`.
Figures 01–14 (plus quality and design variants) have PNG/PDF versions in
`results/figures/`; all 136 prespecified model conditions have additional PR, ROC,
calibration, confusion and learning diagnostics, showing all five actual seed traces.
No transfer/adaptation experiment was run because no defensible recorded domains exist.
