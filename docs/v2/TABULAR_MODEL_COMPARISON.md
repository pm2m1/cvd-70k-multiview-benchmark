# Modern tabular model comparison

| model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- |
| logistic/selected | 0.77272 ± 0.00476 | 0.79076 ± 0.00219 | 0.18734 ± 0.00090 | 0.72653 ± 0.00062 | 0.67123 ± 0.00326 |
| random_forest/selected | 0.77847 ± 0.00181 | 0.79616 ± 0.00175 | 0.18310 ± 0.00046 | 0.72958 ± 0.00177 | 0.67266 ± 0.01348 |
| xgboost/selected | 0.78431 ± 0.00170 | 0.80222 ± 0.00086 | 0.17988 ± 0.00030 | 0.73519 ± 0.00182 | 0.69551 ± 0.00812 |
| mlp/selected | 0.78333 ± 0.00171 | 0.79967 ± 0.00146 | 0.18141 ± 0.00060 | 0.73354 ± 0.00164 | 0.70635 ± 0.01482 |
| multiview/selected | 0.78387 ± 0.00203 | 0.80136 ± 0.00191 | 0.18082 ± 0.00084 | 0.73559 ± 0.00198 | 0.69613 ± 0.03855 |
| ft_transformer/selected | 0.78164 ± 0.00221 | 0.79822 ± 0.00164 | 0.18202 ± 0.00077 | 0.73105 ± 0.00217 | 0.68142 ± 0.02100 |
| tabnet/selected | 0.77422 ± 0.00259 | 0.79233 ± 0.00134 | 0.18507 ± 0.00104 | 0.72573 ± 0.00253 | 0.67689 ± 0.01618 |

Tuned XGBoost mean outer AP 0.78431, AUROC 0.80222.
FT-Transformer uses official `rtdl-revisiting-models`0.0.2, native numeric/categorical tokens,
three registered candidates and validation-AP stopping. TabNet uses established
`pytorch-tabnet`4.1.0, two candidates and sklearn AP. Both ran on CPU after the recorded MPS
smoke failures; five-view/flat ran on reliable MPS. XGBoost is isolated from Torch/OpenMP.
FT/TabNet candidates use the base clinical encoding; optimized five-view/flat/XGBoost
pipelines can select registered engineered blocks. Thus comparisons reflect these explicitly
different pipeline/search budgets, not an intrinsic architecture ranking.
Its AP-stopping candidates use negative sklearn non-interpolated AP, not `aucpr`; other
candidates stop on log loss. Inner candidate selection always uses sklearn AP.

Budgets are explicitly unequal: flat/five-view18 each, FT3, TabNet2, XGB10, LR/RF2 each,
all on three inner folds per outer fold. This is a finite benchmark, not equal-compute proof
that one family intrinsically dominates. TabPFN9.2.0/default3.5 is NOT EXECUTED: no verified
licensed authenticated session, no terms bypass, no weights downloaded. Official model card
states synthetic pretraining; a checkpoint-level derivative/benchmark-selection contamination
audit remains unavailable. Older2.5 limits are not assumed to describe3.5. See gate record.

Official references: [FT](https://github.com/yandex-research/rtdl-revisiting-models),
[TabNet](https://github.com/dreamquark-ai/tabnet),
[TabPFN](https://github.com/PriorLabs/TabPFN),
[3.5 model/licence](https://huggingface.co/Prior-Labs/tabpfn_3_5).

| outer | model | ap | auroc | brier | balanced_accuracy | recall |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | logistic/selected | 0.77900 ± 0.00000 | 0.79225 ± 0.00000 | 0.18657 ± 0.00000 | 0.72630 ± 0.00000 | 0.67104 ± 0.00000 |
| 0 | random_forest/selected | 0.77677 ± 0.00079 | 0.79487 ± 0.00020 | 0.18345 ± 0.00009 | 0.72815 ± 0.00094 | 0.68986 ± 0.00211 |
| 0 | xgboost/selected | 0.78341 ± 0.00017 | 0.80147 ± 0.00019 | 0.18012 ± 0.00005 | 0.73288 ± 0.00083 | 0.69321 ± 0.00181 |
| 0 | mlp/selected | 0.78217 ± 0.00180 | 0.79876 ± 0.00089 | 0.18166 ± 0.00054 | 0.73218 ± 0.00152 | 0.70526 ± 0.01808 |
| 0 | multiview/selected | 0.78285 ± 0.00276 | 0.80093 ± 0.00121 | 0.18062 ± 0.00020 | 0.73558 ± 0.00055 | 0.66205 ± 0.01362 |
| 0 | ft_transformer/selected | 0.78016 ± 0.00011 | 0.79758 ± 0.00024 | 0.18227 ± 0.00020 | 0.72855 ± 0.00067 | 0.67773 ± 0.01729 |
| 0 | tabnet/selected | 0.77593 ± 0.00067 | 0.79311 ± 0.00040 | 0.18440 ± 0.00023 | 0.72655 ± 0.00245 | 0.67532 ± 0.01967 |
| 1 | logistic/selected | 0.76881 ± 0.00000 | 0.78783 ± 0.00000 | 0.18852 ± 0.00000 | 0.72596 ± 0.00000 | 0.66757 ± 0.00000 |
| 1 | random_forest/selected | 0.77819 ± 0.00117 | 0.79513 ± 0.00026 | 0.18336 ± 0.00007 | 0.72879 ± 0.00030 | 0.65997 ± 0.00044 |
| 1 | xgboost/selected | 0.78648 ± 0.00083 | 0.80190 ± 0.00026 | 0.18002 ± 0.00010 | 0.73615 ± 0.00055 | 0.68758 ± 0.00046 |
| 1 | mlp/selected | 0.78355 ± 0.00036 | 0.79880 ± 0.00052 | 0.18176 ± 0.00039 | 0.73381 ± 0.00129 | 0.69673 ± 0.00604 |
| 1 | multiview/selected | 0.78339 ± 0.00130 | 0.79955 ± 0.00063 | 0.18150 ± 0.00069 | 0.73447 ± 0.00106 | 0.70653 ± 0.02844 |
| 1 | ft_transformer/selected | 0.78278 ± 0.00227 | 0.79676 ± 0.00031 | 0.18274 ± 0.00016 | 0.73122 ± 0.00047 | 0.66418 ± 0.00935 |
| 1 | tabnet/selected | 0.77445 ± 0.00251 | 0.79217 ± 0.00080 | 0.18503 ± 0.00042 | 0.72703 ± 0.00224 | 0.68660 ± 0.01779 |
| 2 | logistic/selected | 0.77034 ± 0.00000 | 0.79219 ± 0.00000 | 0.18694 ± 0.00000 | 0.72733 ± 0.00000 | 0.67508 ± 0.00000 |
| 2 | random_forest/selected | 0.78043 ± 0.00092 | 0.79848 ± 0.00017 | 0.18250 ± 0.00001 | 0.73180 ± 0.00037 | 0.66814 ± 0.00252 |
| 2 | xgboost/selected | 0.78304 ± 0.00048 | 0.80328 ± 0.00037 | 0.17949 ± 0.00012 | 0.73653 ± 0.00040 | 0.70575 ± 0.00074 |
| 2 | mlp/selected | 0.78429 ± 0.00220 | 0.80145 ± 0.00054 | 0.18082 ± 0.00043 | 0.73464 ± 0.00144 | 0.71706 ± 0.01421 |
| 2 | multiview/selected | 0.78538 ± 0.00136 | 0.80360 ± 0.00012 | 0.18034 ± 0.00111 | 0.73672 ± 0.00322 | 0.71980 ± 0.04696 |
| 2 | ft_transformer/selected | 0.78199 ± 0.00299 | 0.80031 ± 0.00051 | 0.18105 ± 0.00012 | 0.73339 ± 0.00073 | 0.70236 ± 0.01590 |
| 2 | tabnet/selected | 0.77229 ± 0.00316 | 0.79170 ± 0.00221 | 0.18578 ± 0.00162 | 0.72362 ± 0.00210 | 0.66875 ± 0.00996 |

Across9 selected TabNet fits, aggregate training attention-mask importance had mean pairwise Spearman 0.3577 and minimum -0.0324. These are global mask summaries, not causal clinical explanations. Exact step masks can differ when selected decision-step counts differ; no patient-level mask is released.
