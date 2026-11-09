# Deterministic feature ablations

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| multiview/batch256 | 0.77864 ± 0.00169 | 0.79615 ± 0.00218 | 0.18380 ± 0.00178 | 0.72756 ± 0.00528 | 0.71697 ± 0.03681 |
| multiview/derived_features | 0.78075 ± 0.00304 | 0.79725 ± 0.00155 | 0.18300 ± 0.00019 | 0.72714 ± 0.00036 | 0.66608 ± 0.04059 |
| multiview/missing_indicators | 0.78312 ± 0.00162 | 0.80112 ± 0.00210 | 0.18092 ± 0.00128 | 0.73547 ± 0.00136 | 0.70341 ± 0.04000 |
| multiview/combined_features | 0.78310 ± 0.00306 | 0.80054 ± 0.00183 | 0.18114 ± 0.00070 | 0.73366 ± 0.00059 | 0.68450 ± 0.03147 |
| multiview/robust_combined | 0.78348 ± 0.00215 | 0.80097 ± 0.00247 | 0.18123 ± 0.00141 | 0.73525 ± 0.00313 | 0.68079 ± 0.04359 |

Derived block adds pulse pressure, approximate MAP, age×SBP, age×BMI and BMI². Missingness
block records invalid/missing inputs. The robust control changes fold-fitted scaling only.
All are functions of existing measurements, not independent clinical information. No
external clinical cutoff is invented. Same seed101, base architecture, AP stopping and
batch256 across these comparisons. Added input dimensions also change parameter count.

Derived versus base: ΔAP +0.00211; ΔAUROC +0.00110.
Missing indicators versus base: ΔAP +0.00448; ΔAUROC +0.00497.
Combined versus base: ΔAP +0.00445; ΔAUROC +0.00439.
Robust versus standard combined: ΔAP +0.00039; ΔAUROC +0.00043.

Outer-fold consistency must be assessed from `optimization_metrics.csv`, not just pooled
point estimates. These ablations were not used to revise any outer fold's candidate choice.
