# Source preservation with L2-SP

| positive_budget | model | ap | auroc | brier |
| --- | --- | --- | --- | --- |
| 25 | full | 0.78582 | 0.72436 | 0.21803 |
| 25 | full_sp_high | 0.78735 | 0.72685 | 0.21519 |
| 25 | full_sp_low | 0.78609 | 0.72466 | 0.21770 |
| 25 | logical | 0.79232 | 0.73268 | 0.21147 |
| 25 | logical_sp_high | 0.79237 | 0.73273 | 0.20846 |
| 25 | logical_sp_low | 0.79231 | 0.73268 | 0.21146 |
| 25 | reg_combined | 0.78464 | 0.72279 | 0.21885 |
| 25 | reg_geometry | 0.78722 | 0.72665 | 0.21543 |
| 25 | reg_mmd | 0.78573 | 0.72422 | 0.21778 |
| 25 | reg_shuffled | 0.78538 | 0.72388 | 0.21820 |
| 250 | full | 0.79126 | 0.73122 | 0.21012 |
| 250 | full_sp_high | 0.79201 | 0.73256 | 0.20868 |
| 250 | full_sp_low | 0.79116 | 0.73107 | 0.21006 |
| 250 | logical | 0.79093 | 0.73153 | 0.20912 |
| 250 | logical_sp_high | 0.79149 | 0.73195 | 0.20871 |
| 250 | logical_sp_low | 0.79093 | 0.73148 | 0.20936 |
| 250 | reg_combined | 0.79172 | 0.73229 | 0.20945 |
| 250 | reg_geometry | 0.79178 | 0.73237 | 0.20952 |
| 250 | reg_mmd | 0.79116 | 0.73132 | 0.20963 |
| 250 | reg_shuffled | 0.79194 | 0.73251 | 0.20922 |
| 1000 | full | 0.79227 | 0.73263 | 0.20845 |
| 1000 | full_sp_high | 0.79197 | 0.73312 | 0.20820 |
| 1000 | full_sp_low | 0.79245 | 0.73250 | 0.20828 |
| 1000 | logical | 0.79258 | 0.73343 | 0.20874 |
| 1000 | logical_sp_high | 0.79205 | 0.73282 | 0.20786 |
| 1000 | logical_sp_low | 0.79238 | 0.73305 | 0.20861 |
| 1000 | reg_combined | 0.79212 | 0.73339 | 0.20825 |
| 1000 | reg_geometry | 0.79268 | 0.73346 | 0.20808 |
| 1000 | reg_mmd | 0.79226 | 0.73269 | 0.20843 |
| 1000 | reg_shuffled | 0.79222 | 0.73335 | 0.20696 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

The penalty is a sum of squared distances to detached source backbone weights, excluding the head. This is distinct from ordinary weight decay. Reference: [Li et al., ICML 2018](https://proceedings.mlr.press/v80/li18a.html). Strengths .01/1 were registered; geometry/MMD preservation penalties are deterministic and contrasted with shuffled assignments. Drift data are actual retained-checkpoint differences, including epoch-zero zeros.
