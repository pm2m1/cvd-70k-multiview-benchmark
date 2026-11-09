# Figure gallery

All values come from the supplied dataset audit or locked fitted-model results. Architecture is explicitly conceptual. Main figures have both PNG and vector PDF versions.

- [01 class distribution](01_class_distribution.png) · [PDF](01_class_distribution.pdf)
- [02 feature audit](02_feature_audit.png) · [PDF](02_feature_audit.pdf)
- [02b quality flags](02b_quality_flags.png) · [PDF](02b_quality_flags.pdf)
- [03 preprocessing flow](03_preprocessing_flow.png) · [PDF](03_preprocessing_flow.pdf)
- [04 architecture](04_architecture.png) · [PDF](04_architecture.pdf)
- [05 learning curves](05_learning_curves.png) · [PDF](05_learning_curves.pdf)
- [06 baseline comparison](06_baseline_comparison.png) · [PDF](06_baseline_comparison.pdf)
- [07 precision recall](07_precision_recall.png) · [PDF](07_precision_recall.pdf)
- [08 roc](08_roc.png) · [PDF](08_roc.pdf)
- [09 confusion matrices](09_confusion_matrices.png) · [PDF](09_confusion_matrices.pdf)
- [10 calibration](10_calibration.png) · [PDF](10_calibration.pdf)
- [11 imbalance fixed positive](11_imbalance_fixed_positive.png) · [PDF](11_imbalance_fixed_positive.pdf)
- [11 imbalance fixed total](11_imbalance_fixed_total.png) · [PDF](11_imbalance_fixed_total.pdf)
- [12 threshold sensitivity](12_threshold_sensitivity.png) · [PDF](12_threshold_sensitivity.pdf)
- [13 accuracy balanced accuracy](13_accuracy_balanced_accuracy.png) · [PDF](13_accuracy_balanced_accuracy.pdf)
- [14 parameters performance](14_parameters_performance.png) · [PDF](14_parameters_performance.pdf)

`diagnostics/` contains 136 condition plots, each showing all five actual seed fits: PR, ROC, calibration, both threshold confusion matrices and available learning curves. Exact per-seed counts and operating metrics are in `../aggregate/test_metrics.csv`.

Thin traces show separate fitted seeds; none is an unregistered ensemble. Confusion matrices show explicitly labeled mean seed counts. Shaded imbalance regions are seed SD, not independent-population confidence intervals.
