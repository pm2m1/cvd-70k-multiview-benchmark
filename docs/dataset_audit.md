# Dataset audit

Archive SHA-256: `b78ff607480810c3e27de3e11d52c4962a3a81dbb3d69365e1cf3394a67c2b23`. CSV SHA-256: `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c`.
Verified ZIP has one semicolon-separated `cardio_train.csv`, 2,941,524
uncompressed bytes. Integrity/CRC, path safety, symlinks, encryption, total size and
compression ratio checked before extraction. No other dataset was downloaded.

Shape: [70000, 13]. Target 0=35,021,
1=34,979. All source columns are numeric with no missing values;
weight is floating point and remaining columns integer. Identifier is unique and excluded.
No missing target or out-of-schema categorical code. Source provenance/licensing caveats
are in [release boundary](provenance_and_release.md).

Including ID: 0 duplicate rows. Excluding ID but retaining
label: 24 redundant rows. There are
41 raw feature-identical groups, including
17 with conflicting labels. After deterministic masking,
69,959 modeled feature groups remain; all are split-isolated.
We retain observations and label ambiguity rather than silently deduplicating them.

Minimal retains 70,000; strict retains 68,603,
excluding 1,397 rows. The union of all quality flags is
1,397; counts below overlap. Flags are analyst-specified research
quality heuristics, not proof that every flagged value is physiologically impossible.

| Flag (overlapping) | Records |
| --- | --- |
| age_outside_18_100 | 0 |
| height_outside_120_220_cm | 53 |
| weight_outside_30_200_kg | 7 |
| systolic_outside_80_250 | 247 |
| diastolic_outside_40_150 | 1034 |
| systolic_not_above_diastolic | 1236 |
| bmi_outside_10_70 | 3 |

Invalid anthropometry is masked before BMI derivation; invalid BP order masks both BPs.
Minimal imputes flagged values with training medians. No original raw values are overwritten.
There is no target-based outlier cleaning, resampling before splitting or test rebalancing.

| Cleaning | Partition | Records | Positives | Negatives | Prevalence |
| --- | --- | --- | --- | --- | --- |
| minimal | train | 48999 | 24485 | 24514 | 0.499704 |
| minimal | validation | 10500 | 5247 | 5253 | 0.499714 |
| minimal | test | 10501 | 5247 | 5254 | 0.499667 |
| strict | train | 48039 | 23764 | 24275 | 0.494681 |
| strict | validation | 10288 | 5096 | 5192 | 0.495334 |
| strict | test | 10276 | 5078 | 5198 | 0.494161 |

Split seed 20261109; local assignment hash `af2a78195ab5e972a616ea1aea4a40811050d79fb1779e6e38b2ab50c6ebe899`.
Actual counts reflect indivisible groups rather than exact 70/15/15 percentages.

| Column | Observed minimum | Observed maximum |
| --- | --- | --- |
| id | 0.0 | 99999.0 |
| age | 10798.0 | 23713.0 |
| gender | 1.0 | 2.0 |
| height | 55.0 | 250.0 |
| weight | 10.0 | 200.0 |
| ap_hi | -150.0 | 16020.0 |
| ap_lo | -70.0 | 11000.0 |
| cholesterol | 1.0 | 3.0 |
| gluc | 1.0 | 3.0 |
| smoke | 0.0 | 1.0 |
| alco | 0.0 | 1.0 |
| active | 0.0 | 1.0 |
| cardio | 0.0 | 1.0 |

Detailed categorical counts, missingness before/after masking, dtypes and archive entries
are in `results/aggregate/dataset_audit.json`. Ranges are aggregate summaries, not released
participant rows. Age conversion is /365.25 without integer-year rounding.
