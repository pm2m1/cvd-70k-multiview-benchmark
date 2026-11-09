# Statistical analysis

Primary P1000 selected-policy minus label-cost-matched full:
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

All comparisons are in classification_comparisons.json and transfer_comparisons.json. Paired bootstrap resamples modeled-feature groups within each fixed outer fold, preserving paired predictions. Intervals condition on fitted checkpoints, split assignments and development selections; they do not resample the full search/source training or establish independent population replication. Multiplicity-unadjusted secondary comparisons are exploratory. Three folds/draws share source participants and have only one seed per transfer fold; effects of seed, target draw and fold cannot be separated. Practical reference was +.005 AP; neither favorable point estimates nor increased ADAPT decisions alone establish superiority.

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Negative transfer:
| positive_budget | model | negative_transfer_frequency | mean_ap_difference | mean_loss_when_harm | mean_gain_when_help | n_outer_episodes |
| --- | --- | --- | --- | --- | --- | --- |
| 1000 | full | 0.33333 | 0.00006 | -0.00085 | 0.00104 | 3 |
| 1000 | full_cost | 0.66667 | -0.00063 | -0.00107 | 0.00023 | 3 |
| 1000 | full_sp_low | 0.33333 | 0.00023 | -0.00085 | 0.00156 | 3 |
| 1000 | head | 0.66667 | -0.00013 | -0.00037 | 0.00036 | 3 |
| 1000 | logical | 0.33333 | 0.00037 | -0.00099 | 0.00105 | 3 |
| 1000 | selected_new_policy | 1.00000 | -0.00038 | -0.00038 | 0.00000 | 3 |
| 1000 | soft_evidence | 0.66667 | -0.00054 | -0.00089 | 0.00017 | 3 |


Checkpoint behavior:
| method | epoch_zero_fraction | median_best_epoch |
| --- | --- | --- |
| full | 0.11111 | 6.00000 |
| full_cost | 0.11111 | 2.00000 |
| full_sp_high | 0.00000 | 23.00000 |
| full_sp_low | 0.11111 | 8.00000 |
| fusion1_only | 0.11111 | 6.00000 |
| fusion2_only | 0.00000 | 15.00000 |
| fusion_both | 0.11111 | 6.00000 |
| head | 0.11111 | 10.00000 |
| logical | 0.00000 | 8.00000 |
| logical_sp_high | 0.00000 | 8.00000 |
| logical_sp_low | 0.00000 | 8.00000 |
| random_matched | 0.11111 | 4.00000 |
| reg_combined | 0.00000 | 8.00000 |
| reg_geometry | 0.00000 | 17.00000 |
| reg_mmd | 0.11111 | 4.00000 |
| reg_shuffled | 0.00000 | 7.00000 |
| smallstop_full_cost | 0.33333 | 3.00000 |
| smallstop_full_sp_low | 0.00000 | 6.50000 |
| smallstop_head | 0.66667 | 0.00000 |
| smallstop_soft_evidence | 0.00000 | 8.00000 |
| soft_evidence | 0.00000 | 15.00000 |
| soft_learned | 0.11111 | 8.00000 |
| soft_random | 0.11111 | 10.00000 |
| soft_shuffled | 0.00000 | 16.00000 |
| soft_uniform | 0.11111 | 9.00000 |
| view_anthro | 0.00000 | 14.00000 |
| view_background | 0.11111 | 18.00000 |
| view_behaviour | 0.11111 | 13.00000 |
| view_hemo | 0.11111 | 15.00000 |
| view_metabolic | 0.11111 | 22.00000 |
