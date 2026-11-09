# Frozen protocol, version 1.0

Written before any model fit or final test prediction. This is an internal cross-sectional
label classification benchmark, separate from the HAALSI→MHAS study. No independent cohort,
acquisition-site labels, incident outcomes or transfer effects are claimed.

## Partition and quality decisions

Seed 20261109, twenty approximately stratified feature-group folds: first fourteen train,
next three validation, final three test. Membership is generated on all raw observations
before cleaning and saved only locally with a public SHA-256. Group on the complete modeled
feature vector after deterministic invalid-value masking, excluding target and identifier.
This also isolates raw feature duplicates and contradictory labels. Do not remove exact
observations: retain multiplicities and use group bootstrap. A unique row ID does not verify
unique people. Repeated-person linkage beyond exact feature identity is unavailable.

Research quality bounds, inclusive: age 18–100 years; height 120–220 cm; weight 30–200 kg;
systolic 80–250 mmHg; diastolic 40–150 mmHg; systolic strictly above diastolic; BMI 10–70
kg/m². These intentionally broad analyst-specified plausibility heuristics are not clinical
diagnostic rules or authoritative universal impossibility limits. They flag possible unit,
transcription and measurement errors. The minimal primary dataset retains all rows, masks
flagged measurements and uses training-only median imputation. Invalid BP ordering masks
both pressures. Derive BMI after checking its parents. Strict sensitivity excludes any
flagged or missing feature record. No bound is selected by test performance.

Categorical levels are schema-specified: binary gender code (1→0, 2→1, without assuming
sex mapping), one-hot cholesterol/glucose levels 1,2,3, binary behaviour. Age /365.25.
All 16 encoded inputs are available to every main model; standardize with train-only mean
and variance after train-only imputation. BMI shares information with height and weight;
BMI-only and parent-only sensitivities use both neural families and five seeds.

## Primary inference and selection

AP is primary. AUROC, Brier, log loss, ten-bin equal-width ECE, accuracy, balanced accuracy,
recall, specificity, precision, F1, confusion counts, wall-clock fit time and parameter
count are secondary/descriptive. ECE bins [0,.1),…,[.9,1] are weighted by observation count.
Test prevalence remains natural in every experiment and supplies the AP no-skill baseline.
Accuracy cannot select a winner. Select hyperparameters by validation AP (seed 101), two
candidates per family, ties first. Shared settings are reused in both cleaning sensitivities
and all imbalance/intervention experiments, with no condition-specific extra search.
Compare seven full-training conditions: five standard families plus weighted and resampled
multiview. The winner is the highest mean validation AP across five seeds in these seven
conditions; final test AP describes generalization and does not revise this selection.

Neural: five encoders (16 each), fusion 64,32, ReLU then LayerNorm, one logit. Flat MLP
96,64 with ReLU/LayerNorm. AdamW, weight decay .0001, lr {.001,.0003}, batch 1024, ≤60
epochs, patience 8, min_delta 1e-5. Stop on *unweighted natural-validation BCE* for all
interventions. Epoch zero is logged but ineligible for scratch training; pretrained models
would require epoch-zero eligibility and are not implemented in this benchmark. Restore
best checkpoint. XGBoost uses up to 600 trees, depth {3,5}, learning rate .05, histogram
method, subsample/column fraction .9, L2 1; validation log loss early stopping patience 30.
RF: 200 trees, leaf minimum 10, depth {8,unlimited}. LR L2 C {.1,1}, max_iter 2000.
Five paired seeds 101,202,303,404,505; LR is effectively deterministic at a fixed sample.
MPS is used after gradient parity and finite-optimization testing when reliable, else CPU.
Record hardware/software and deterministic-algorithm configuration. Cross-device bitwise
reproduction is not promised.

Select an operating threshold separately for every fitted model using validation balanced
accuracy over .01,.02,…,.99. Ties: closest to .5, then lower. Report that threshold and .5.
A fixed descriptive threshold grid 0,.05,…,1 is not a tuning tool. No clinical utility
objective or deployment threshold is inferred.

## Controlled prevalence analyses

Training-only class sampling without replacement, nested class-specific permutations for
paired comparisons. Four requested prevalences .50,.25,.15,.0742. Keep original validation
and final test fixed. A: choose the largest common total N feasible for every fraction,
N=floor(min_q(min(P/q,Nneg/(1-q)))); round N*q positives. B: choose the largest common
positive count P*=floor(min(P,min_q(Nneg*q/(1-q)))); round P*(1-q)/q negatives. Thus both
analyses use the same available training pool; A cannot use the full pool at rare prevalence,
and B necessarily has low positive support at every fraction. No single design identifies
a pure causal prevalence effect: A trades positive and negative support; B trades negative
support and total size. Report these limitations explicitly.

Run all five model families under standard, class weighting and random oversampling for
both analyses, all four prevalences and all five paired seeds (600 fits). Weight positive
loss/class by training negatives/positives. Oversampling retains every sampled row once
and adds minority repeats to achieve 50/50, using only the training sample. No SMOTE.
Sampling seed equals model seed, so imbalance seed variability combines sample and fit
variation. Full-training seed variability measures stochastic fitting on the same people.
No resampling of validation or test; no automatic prior correction or calibration fitting.
The 7.42% condition resembles MHAS prevalence only, not its cohort or domain shift.

## Lock, evaluation and uncertainty

Complete development, save checkpoints and validation-selected thresholds, write a hashed
model/development manifest and an explicit development lock before final evaluation.
The evaluation command refuses an absent/stale lock and records its first test opening.
No post-test tuning. Tests and reporting fixes may be made without changing fitted models.
AP 95% intervals use 1,000 paired modeled-feature-group bootstrap draws, average AP across
seeds within each draw. This conservatively handles exact duplicates; actual participant
identity is unknown. Report paired AP difference versus flat MLP and multiview versus flat
for matched prevalence/intervention conditions. Seed SD and participant/group bootstrap
answer different questions; neither supplies external validation. No multiple-comparison
correction: secondary and imbalance analyses are descriptive, not confirmatory superiority
claims. No transfer experiment: no defensible recorded domains have been established.
