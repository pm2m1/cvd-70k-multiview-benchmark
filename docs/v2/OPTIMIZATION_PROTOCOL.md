# V2 optimization protocol — fixed before new training

Exploratory internal evidence informed by V1. Eligible population: only 48,999 original V1
training records. Never use the 10,500 old validation or 10,501 old test records for V2.
V1 reported test values are historical context, not a paired V2 comparator. Primary paired
comparator is V1-equivalent five-view refitted inside these same V2 folds.

Three outer StratifiedGroupKFold partitions, seed 711. For each outer training set, reserve
one of five grouped folds (seed 800+outer) solely for probability calibration and thresholds;
remaining four folds form the model-fitting pool. Three inner grouped folds (seed 900+outer)
select candidates and epochs. Feature-identical groups use V1's target-independent masking
and grouping. All transforms fitted independently on inner training only. Outer labels are
unavailable to model, checkpoint, hyperparameter, calibration, threshold or ensemble selection.
Inner stopping folds also score their candidates, as in ordinary nested CV; that optimism is
contained inside the outer evaluation. No outcome-based cleaning; retain V1 minimal masking.

Candidate table in `configs/v2/study.json` is frozen before fitting. 18 explicitly paired
configurations per flat/five-view family cover caps 60/120/200, patience 8/16/24, BCE/AP/AUROC
stopping, batch 1024 versus 256, embeddings 16/32/64, matched flat capacities, dropout
0/.1/.2/.3, AdamW/Adam, several learning rates/decays/schedules, and deterministic feature
blocks. No blind Cartesian grid. Every candidate evaluated on all three inner folds in each
outer fold. FT-Transformer 3 candidates; TabNet 2; XGBoost 10; logistic/RF 2 each. Unequal
budgets are explicit, not an equal-compute superiority claim. XGBoost proposals use a seeded,
nonadaptive Optuna sampler to keep the same finite candidate set across outer folds.
No additional candidates are introduced after outer evaluation.

Scratch epoch zero logged but ineligible. Min improvement 1e-5 for BCE, 1e-4 for AP/AUROC.
Selected candidate maximizes mean inner AP, ties lower candidate index. Outer refits use the
median selected inner checkpoint epoch and all fitting-pool data, with no outer early stopping.
Three paired seeds for every family's selected candidate. All 18 flat/five-view candidates
also receive one seed-101 outer refit for prespecified ablation effects. These outer ablation
results do not revise candidate selection. More parameters and mechanism are never outcomes.
No gated/attention fusion added to the five-view model: capacity/regularization and documented
FT-Transformer supply interpretable controls without unablated architectural novelty.

Feature blocks: derived pulse pressure/MAP, age×SBP, age×BMI, BMI²; invalid/missing indicators;
combined; robust scaling control. These are deterministic redundancies, not new measurements.
No undocumented clinical categorical cutoffs. Encoders retain the five clinical views.
Official FT-Transformer receives continuous tokens and reconstructed integer categorical
embeddings (gender, cholesterol, glucose, smoking, alcohol, activity). TabNet receives the
same available encoded feature information. Report exact software implementation and counts.

Selected-family inner out-of-fold predictions fit convex probability weights for prespecified
pairs (XGB+multiview, XGB+FT, XGB+TabNet, flat+multiview) and XGB+multiview+FT. Equal weights
and grid-convex weights (step .1) compared. Select best single and ensemble *within* inner OOF
only; report their outer selection procedures. No stacking beyond this finite scope.
Dedicated calibration rows fit monotone Platt, isotonic, and neural temperature mappings.
Evaluate these exclusively on outer folds. Fitted calibration/ensemble artifacts stay private.
Threshold rules .50, calibration-balanced accuracy, calibration-F1, precision≥.75/max-recall,
recall≥.80/max-precision use calibration rows only; infeasible constraints are recorded.
Thresholds do not change ranking metrics. Isotonic may introduce ties; strictly increasing
positive-slope Platt/temperature transformations preserve ranks.

Primary metric non-interpolated sklearn AP. Secondary AUROC, Brier, log loss, 10-bin equal-width
ECE, accuracy/BA/recall/specificity/precision/F1, confusion counts, duration and parameter count.
Intervals: 1,000 paired feature-group bootstrap draws on concatenated outer predictions,
shared folds and paired seeds. Treat fitted folds as fixed; no claim of independence among
folds or external confirmation. Report pooled OOF and fold-wise mean; fold score-scale
variation can affect pooled AP. Multiple exploratory comparisons are not multiplicity-adjusted.

Imbalance follow-up: each outer fold's selected multiview config, original fold-fitting pool,
**three** paired seeds, largest common fixed total at 50% and 7.42%, standard,
class-weighted and training-only oversampling. Fixed natural outer evaluation, dedicated
natural calibration. Evaluate raw and Platt, thresholds .50/BA/F1/constraints. Epoch budget
comes from that candidate's natural-prevalence inner checkpoints; this tests that selected
training procedure under imbalance without additional condition-specific tuning. No sampled
rare-prevalence evaluation is represented as a clinical population.

TabPFN is gated before any pretrained download by actual installed version, official model
licence, authentication/terms, data contamination statements and hardware limits. No token or
licence workaround. If blocked, record NOT EXECUTED; no surrogate result. No external private
Phase 1 data/checkpoints are accessed. Exact trial completion/failures and hashes are recorded.
