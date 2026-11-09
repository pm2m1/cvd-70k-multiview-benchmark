# Selector evidence study

| positive_budget | module | adapt_frequency | valid_frequency | U_mean | R_mean |
| --- | --- | --- | --- | --- | --- |
| 25 | fusion1 | 0.00000 | 1.00000 | 1.22448 | 0.79480 |
| 25 | fusion2 | 0.33333 | 1.00000 | 1.15186 | 0.74754 |
| 25 | view:anthropometric | 0.00000 | 1.00000 | 1.03744 | -0.05098 |
| 25 | view:background | 1.00000 | 1.00000 | 0.37350 | 0.04687 |
| 25 | view:behaviour | 0.00000 | 0.00000 | nan | nan |
| 25 | view:haemodynamic | 0.33333 | 1.00000 | 1.14263 | 0.71029 |
| 25 | view:metabolic | 0.00000 | 1.00000 | 2.83517 | 1.72055 |
| 250 | fusion1 | 0.33333 | 1.00000 | 0.95808 | 0.82977 |
| 250 | fusion2 | 0.33333 | 1.00000 | 0.94844 | 0.82078 |
| 250 | view:anthropometric | 0.33333 | 1.00000 | 1.21701 | 0.79579 |
| 250 | view:background | 1.00000 | 1.00000 | 0.22037 | 0.09662 |
| 250 | view:behaviour | 0.00000 | 0.00000 | nan | nan |
| 250 | view:haemodynamic | 0.00000 | 1.00000 | 0.98480 | 0.84119 |
| 250 | view:metabolic | 0.00000 | 1.00000 | 1.28596 | 0.89797 |
| 1000 | fusion1 | 0.66667 | 1.00000 | 0.86385 | 0.77771 |
| 1000 | fusion2 | 0.66667 | 1.00000 | 0.86042 | 0.77243 |
| 1000 | view:anthropometric | 0.33333 | 1.00000 | 0.83934 | 0.64214 |
| 1000 | view:background | 1.00000 | 1.00000 | 0.13069 | 0.07111 |
| 1000 | view:behaviour | 0.00000 | 0.00000 | nan | nan |
| 1000 | view:haemodynamic | 0.33333 | 1.00000 | 0.84127 | 0.75759 |
| 1000 | view:metabolic | 0.00000 | 1.00000 | 1.10041 | 0.85786 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Development-only single-module benefit correlations:
```json
{
  "U": {
    "spearman": 0.11957764741476934,
    "n": 54,
    "scope": "Dependent module/fold/budget observations, selection role only; descriptive, no causal or confirmatory p value"
  },
  "R": {
    "spearman": 0.23641076482830709,
    "n": 54,
    "scope": "Dependent module/fold/budget observations, selection role only; descriptive, no causal or confirmatory p value"
  },
  "p_mmd": {
    "spearman": -0.3414724863169359,
    "n": 63,
    "scope": "Dependent module/fold/budget observations, selection role only; descriptive, no causal or confirmatory p value"
  }
}
```

These dependent observations use only 100 selection labels per episode. Correlation cannot establish mechanism or causal adaptation value. No outer outcome was used to learn a policy. Learned meta-selection: NOT EXECUTED because three overlapping age tasks do not provide independent meta-training/evaluation populations. The evidence-guided policies are fixed formulas, not learned meta-selectors. Larger support reaches geometry but MMD remains capped128.
