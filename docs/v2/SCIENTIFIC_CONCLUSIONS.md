# Scientific conclusions — distinguish the two research questions

RQ-A predictive optimization: matched V2 selected five-view versus refitted V1-equivalent
mean outer AP difference +0.00515,95% conditional group-bootstrap interval
[0.0031798941113366595, 0.0070604282226044935]. Tuned XGBoost mean outer AP/AUROC 0.78431/0.80222;
selected five-view 0.78387/0.80136. These are nested internal estimates;
original V1 test AP .7656 versus .7696 is historical context, not an independent V2 comparison.
The component effects and their intervals are in optimization reports. No required .90
outcome, no unsupported substantial-improvement claim, and no post-outer retuning.

RQ-B full selective adaptation is independently implemented and verified with valid/invalid
synthetic geometry, permutations, random budgets, frozen integrity and epochzero tests.
It was executed on real recorded age subgroups with disjoint label roles. Main evidence:

| Fit positives | Logical−full AP | 95% CI | Logical−full AUROC |
| --- | --- | --- | --- |
| 25 | +0.00055 | [0.00027, 0.00085] | +0.00063 |
| 100 | -0.00103 | [-0.0029, 0.00092] | +0.00040 |
| 250 | -0.00285 | [-0.00516, -0.0005] | -0.00079 |
| 500 | -0.00074 | [-0.00254, 0.00093] | -0.00181 |
| 1000 | -0.00058 | [-0.00214, 0.00102] | +0.00030 |

The gain is a modest internal pipeline effect; the matched width/batch ablations do not
establish an intrinsic five-view advantage over independently tuned families. Cap120−60 AP:
+0.000000 (95% CI [0.0, 0.0]); cap200−60: +0.000000 (95% CI [0.0, 0.0]).
AP−BCE stopping: +0.002312 (95% CI [0.0006638387658321727, 0.0038188161071730973]). Missing-indicator−base AP:
+0.004484 (95% CI [0.0017604592696324852, 0.007244356739233428]). This feature may encode collection/quality artifacts.

No budget established a conditional AP advantage over all three prespecified controls.
Budgets with a positive conditional logical−full AP interval: [25]; budgets with
positive intervals against **all** full/head/matched-random controls: [].
These are multiplicity-unadjusted exploratory comparisons, not confirmatory significance.
From25→1000 positives, mean ADAPT count changed
2.33→4.33, Jaccard
agreement 0.417→0.867, invalid geometry
14.3%→14.3%, and all-SHARE/head-only
mask frequency 0.0%→0.0%. Selecting modules
for ADAPT and retaining an updated checkpoint are different; epochzero may retain source weights.

At the largest budget, logical AP/AUROC differences versus full are
-0.000577/+0.000300, versus head
-0.003149/-0.001128, versus matched
random +0.000059/+0.000615.
The full-cost-matched comparison has AP difference -0.003282,
95% interval [-0.006467121130575885, -0.00010350354106567309]; all AUROC intervals are reported separately.

Inner-selected ensemble−single AP is -0.000167,95% interval
[-0.001445818855748214, 0.0010559142648515105]; positive point estimates alone cannot establish benefit. Five-view
Brier raw/Platt 0.180822/0.180334; ECE
0.019306/0.011023. Isotonic AP
0.777660 versus raw 0.783875; ties can change ranking.
At7.42% training prevalence, standard recall@.50 is0.012606, while validation
threshold BA is0.728688. Calibration and thresholds address
score/prior mismatch, not an independent rare-prevalence clinical population.


No genuine cross-cohort experiment executed: **EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED**.
TabPFN also NOT EXECUTED because licensed access is unverified. FT-Transformer/TabNet are
actual completed fits, not substitute algorithms. No private Phase1 participants/models used.

Both studies share Kaggle development participants and were motivated by V1. Fold/draw
reuse, imperfect person linkage, uncertain diagnosis/provenance/licence, label-prior shifts,
unequal finite tuning budgets and many comparisons limit inference. The source-age range is
restricted, and target stopping labels are numerous. None predicts future heart attack/stroke
or establishes clinical utility/deployment validity. Next scientific step: independent,
properly permitted data with an explicit endpoint and a locked confirmatory evaluation.
