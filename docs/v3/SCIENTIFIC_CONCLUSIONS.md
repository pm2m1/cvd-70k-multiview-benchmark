# V3 scientific conclusions

Primary selected-policy minus matched-full AP: +0.000257, 95% conditional CI [-0.0013491807279656764, 0.0019790141413158387]; AUROC -0.000488, CI [-0.0015815087928392358, 0.0007022669028068389].

V3 does not establish a positive AP advantage over label-cost-matched full fine-tuning. A tiny favorable point estimate or isolated secondary win is insufficient.

Development-selected policies: {"soft_evidence": 3, "reg_combined": 3, "full_sp_low": 2, "reg_mmd": 1}. Their outer results were not used to select the policy. Each selected procedure consumes the entire registered candidate-selection supervision budget.

Classification, feature, architecture and loss effects are in PREDICTIVE_OPTIMIZATION and ADVANCED_MULTIVIEW_ARCHITECTURES with matched V1/V2 configuration refits. Stronger trees, attention, ranking losses and source regularization are hypotheses tested, not presumed improvements. Negative-transfer frequencies and actual module drift distinguish representation movement from useful prediction. Quality flags may encode acquisition/cleaning artifacts; conditional internal gains cannot establish clinical utility.

EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED. A next study requires a separately permitted cohort with a documented cross-sectional endpoint crosswalk, person linkage/overlap checks, sufficient positives, independent calibration and locked evaluation. No future-event or clinical deployment claim is supported. Learned meta-selection remains infeasible without independent transfer tasks. More seeds/folds and truly label-limited roles are needed to estimate algorithm-selection stability.

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.


## Measured component findings

Highest individual full-population AP point estimate was LightGBM configuration1 (0.78703); this is a descriptive post-evaluation ranking. The independently inner-selected tree seed-average procedure achieved AP/AUROC 0.78609/0.80325. Its matched delta AP versus V2 reference was +0.00379, conditional95% interval [0.0013471838387664554, 0.006442764047918025].

Among the five-view candidates, age splines had AP 0.78418 when age_spline was the highest descriptive point estimate. Age-spline delta AP versus V2 reference was +0.00188, CI [-0.0006642509569300893, 0.004426911857991886]; an AP advantage is not established. Gated fusion underperformed the reference point estimate, residual fusion underperformed its depth control, and attention essentially tied its parameter-matched flat control. Focal reduced AP; ranking losses did not establish an AP gain (the .01 variant had a small positive point difference). These family comparisons have unequal search multiplicity and limited seeds.

At P1000, the development-selected new procedure fell below source-only AP in all three outer episodes (mean loss -0.000377). Low-strength full L2-SP changed AP by +0.000171 versus the same-fitting-label full baseline, too small to establish a practical gain. Raw selected-policy Brier 0.208401 was worse than source-only 0.205887; balanced target fitting can alter scores under natural target prevalence. No post-evaluation calibration or threshold optimization was performed.

More positives increased mean logical ADAPT count from1.67 to3.00, while invalid geometry stayed14.3% and mask Jaccard did not improve monotonically (.444/.361/.400). This does not reproduce a claim that more positive support necessarily stabilizes selection. No all-SHARE masks occurred. Evidence-benefit correlations on the small selection sets are descriptive; shift evidence is not a validated predictor of useful module updates. Uniform/evidence/random/shuffled soft controls and penalty controls do not establish that evidence-guided adaptation improves ranking.
