# Methodology

This standalone benchmark classifies the recorded cross-sectional `cardio` label. It does
not predict incident heart attack/stroke or future 5-/10-year risk. It is not Phase 1 external
validation and provides no clinical deployment evidence.

The architecture takes conceptual inspiration from the prior project's named view encoders,
concatenation, two fusion blocks and prediction head. The new implementation is independent;
no private cohort code, historical results or report assets were copied. No licence was
present for the previous repository, so conceptual reference is credited without relicensing
its code. Reference: https://github.com/pm2m1/cvd-selective-transfer-research . The prior study
was inconclusive regarding superiority over full fine-tuning; that result is not a baseline
on this new dataset. No selection/transfer modules are needed for a random-split benchmark.

The five views are background (age, numeric gender code); haemodynamic (SBP, DBP);
anthropometric (height, weight, BMI); metabolic (cholesterol and glucose one-hot categories);
and behaviour (smoke, alcohol, activity). Dimensions are 2,2,3,6,3. All clinical encoders use
Linear→ReLU→LayerNorm with width 16. Concatenated dimension 80 passes through widths 64
and 32 with the same activation/normalization, then one binary logit. The model has 7,985
parameters. The direct flat MLP takes the identical 16 inputs, widths 96,64, LayerNorm,
and one logit: 8,225 parameters (3.0% more). Both use BCE-with-logits and AdamW.

BMI = kg/(cm/100)², conditional on valid height and weight. This intentional derived-variable
redundancy is checked through BMI-only and height/weight-only neural sensitivities. Gender
code semantics are not independently verified; no male/female labeling is inferred from codes.

Every preprocessing fit uses only that run's sampled training rows. All preprocessing
parameters, actual fitted models, assignments and predictions remain local. Feature-group
membership uses target-independent deterministic quality masking. Duplicates are retained,
not assumed to be distinct participants. Contradictory-label duplicate groups remain together
in one partition. Group bootstrap is the available conservative approximation to participant
bootstrap; unknown repeat-person linkage cannot be recovered from the row ID.

See [frozen protocol](experiment_protocol.md) for exact bounds, budgets, sampling formulas,
selection, thresholds and uncertainty definitions. The test membership was frozen before
training. Model selection uses validation only. The final test evaluation is a separate
command gated by checksums of protocol, configuration, splits, inference/training code,
development manifest and every fitted checkpoint.

Calibration error is ECE = Σ_b (n_b/N) × |mean predicted probability_b − observed positive
fraction_b|, with ten equal-width probability bins [0,.1),…,[.9,1]. Empty bins contribute
zero. Brier and log loss use original unweighted test labels; lower is better. AP is
scikit-learn's non-interpolated average precision, not trapezoidal PR area. Undefined
precision with zero predicted positives is reported as zero; confusion denominators are
explicit in aggregate outputs. Tree models have no directly comparable neural trainable
scalar count: that field is blank/N/A. RF node counts and XGBoost best boosting rounds
are recorded as separate capacity descriptors in development metadata.
