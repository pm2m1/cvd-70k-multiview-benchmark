# Smaller stopping-set sensitivity

| comparison_method | ap | auroc | brier | recall |
| --- | --- | --- | --- | --- |
| full_cost | 0.79060 | 0.73150 | 0.20968 | 0.62087 |
| head | 0.79215 | 0.73260 | 0.20581 | 0.64501 |
| selected_new_policy | 0.79146 | 0.73166 | 0.21233 | 0.60898 |


P25 fitting budget, 20 stopping labels (10/class), same source checkpoint and paired draws. The new policy was already chosen using the main100-label stopping design and its disjoint100-label selection role; it was not reselected after this sensitivity. The other80 stopping labels are withheld. This tests checkpoint allocation for a fixed chosen policy, not a fully retuned20-label pipeline. Selected and full-cost procedures access220 target labels in total. Head fitting/stopping uses70; the sensitivity manifest conservatively also charges100 policy-selection labels, recording170 accessible labels for this fixed control. No predictor fit was added after outer evaluation.
