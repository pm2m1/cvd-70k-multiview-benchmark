| Analysis | Cleaning | Anthropometry | Model | Test n | Test prevalence | AP | AUROC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | strict | all | logistic | 10276 | 0.4942 | 0.7584 ± 0.0000 | 0.7831 ± 0.0000 | 0.1905 ± 0.0000 |
| full | strict | all | random_forest | 10276 | 0.4942 | 0.7690 ± 0.0006 | 0.7911 ± 0.0002 | 0.1855 ± 0.0001 |
| full | strict | all | xgboost | 10276 | 0.4942 | 0.7710 ± 0.0006 | 0.7939 ± 0.0002 | 0.1844 ± 0.0000 |
| full | strict | all | mlp | 10276 | 0.4942 | 0.7663 ± 0.0019 | 0.7902 ± 0.0005 | 0.1860 ± 0.0003 |
| full | strict | all | multiview | 10276 | 0.4942 | 0.7667 ± 0.0026 | 0.7908 ± 0.0013 | 0.1857 ± 0.0004 |
| anthropometry | minimal | bmi_only | mlp | 10501 | 0.4997 | 0.7647 ± 0.0009 | 0.7854 ± 0.0003 | 0.1883 ± 0.0001 |
| anthropometry | minimal | bmi_only | multiview | 10501 | 0.4997 | 0.7644 ± 0.0013 | 0.7857 ± 0.0004 | 0.1879 ± 0.0003 |
| anthropometry | minimal | parents_only | mlp | 10501 | 0.4997 | 0.7657 ± 0.0020 | 0.7857 ± 0.0006 | 0.1882 ± 0.0003 |
| anthropometry | minimal | parents_only | multiview | 10501 | 0.4997 | 0.7656 ± 0.0013 | 0.7850 ± 0.0006 | 0.1884 ± 0.0004 |
