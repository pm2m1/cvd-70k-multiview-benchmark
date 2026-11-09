# Internal age-subgroup adaptation — locally prespecified V2 protocol

Only original V1 training rows. Source age <55 years; target age ≥55. Boundary chosen from a
round recorded age and class-count feasibility before comparative performance, never row IDs.
This is INTERNAL SUBGROUP ADAPTATION, not external or cross-country transfer. Age remains a
predictor but its truncated ranges and associated case mix constrain inference.

Disjoint grouped roles (ten stratified folds, seeds 1201/1202): source 8 training, 1 stopping,
1 selector-reference; target 4 fitting pool, 3 selector-calibration pool, 1 stopping, 2 evaluation.
Exact counts in `transfer_feasibility_counts.json`. Selector labels are separate from fitting,
stopping and evaluation, an explicit stricter change from Phase 1's reuse of adaptation rows.
Count all these target labels, including selector and stopping labels, not only adapted cases.
No unknown repeated-person linkage beyond feature groups is claimed.

Three source seeds and paired target draws 101,202,303. Source original 16→64→32 five-view,
AdamW lr .001, batch 512, ≤120 epochs, patience16 AP. Training-only preprocessing fitted on
source training and held fixed for adaptation. Target-only references fit their own target
training preprocessing; their greater preprocessing flexibility is disclosed.

Budgets 25/100/250/500/1000 positives with equal negatives in *each* of target fit and separate
selector samples. Class-specific nested permutations pair budgets. Fixed stopping/evaluation
populations. Complementary prevalence diagnostic: 250 positives/750 negatives (25%) versus
250/250 (50%), separately sampled in each fitting/selector role. All are feasible by counts.
Increasing support at fixed prevalence also increases total labels; no pure causal claim.

Methods: source only, target only, head only, full, fixed fusion1+fusion2, closest-parameter
random mask, legacy normalized-MMD/discriminability top-half selector, calibrated threshold,
calibrated logical, plus target-only logistic/XGBoost. Head always trainable, seven internal
modules selectable. MMD .5/1/2/4 Gaussian kernels, biased MMD², pooled label-blind RMS scaling,
199 domain permutations, p=(1+exceedances)/200; cap128/domain. Full available selector geometry
(primary) with stratified199 bootstraps; valid source D≥.1, minimum5/class; source bootstrap
contrast validity≥.95. Signed R=cos(direction)×Dtarget/Dsource, U=95th-percentile retention.
Logical: ADAPT if U<.75 OR (p≤.05 AND R<.8 AND U<1), only with valid geometry. Invalid→SHARE.
Threshold policy uses R<.5 and U<1, or the corroborated domain/retention clause. Legacy=.65
normalized MMD + .35 inverse normalized target D, top ceil(7/2)=4, stable name ties.
These reproduce the published Phase 1 mathematical rules, with independently written code.
MMD is evidence for masks, never an optimization loss. Verify every frozen tensor unchanged.

Closest-parameter random mask sampled uniformly over all 128 masks minimizing absolute count
error; include the logical mask. Report candidate count/error and equality. A collapsed random
control is disclosed rather than called independent diversity. Fixed mask is not budget-matched.

Main adaptations use AdamW .001, batch128, max80, patience12 AP, with **epoch zero eligible**.
Source initialization identical across paired methods. Checkpoint ablations on head/full/logical:
BCE with zero eligible and AP with zero *ineligible*, versus main AP/zero eligible. These
explicitly distinguish stopping objective from source-model protection. Record whether best
checkpoint is zero and whether evaluation performance falls despite validation selection.
Never choose checkpoint using target evaluation. Target-only scratch has no epoch-zero eligibility.

Sample-cap ablation at budgets25 and1000, every seed: (a) cap geometry at128 stratified rows,
keep MMD128; (b) full geometry, MMD512. Primary already uses all available geometry, as did
historical calibrated geometry; do not pretend increasing MMD cap supplies extra geometry
labels. Refit logical masks when changed, report decisions/predictions/cost.

Comparison primary logical−full, logical−head, logical−matched-random at each budget.
AP/AUROC and group-aware paired bootstrap on the fixed target evaluation, averaging paired
seeds inside each draw; shared source fits and overlapping draws imply dependence. Report
geometry invalidity, all-share/head-only frequency, each module's ADAPT frequency and mask
agreement, uncertainty R/U and permutation evidence. More ADAPT flags alone is not success.
No target evaluation before selections/checkpoints/calibration records are locked.
