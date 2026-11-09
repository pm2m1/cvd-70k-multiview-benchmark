# Advanced multiview architectures

| model | ap | auroc | brier |
| --- | --- | --- | --- |
| attention | 0.78305 | 0.80017 | 0.18162 |
| attention_no_identity | 0.78333 | 0.80072 | 0.18134 |
| depth_control | 0.78124 | 0.79952 | 0.18280 |
| flat_attention_matched | 0.78305 | 0.80007 | 0.18140 |
| flat_pairwise_matched | 0.78124 | 0.79854 | 0.18219 |
| focal | 0.78073 | 0.79858 | 0.20306 |
| gated | 0.78172 | 0.79946 | 0.18156 |
| gated_regularized | 0.78266 | 0.79960 | 0.18155 |
| pairwise | 0.78274 | 0.79974 | 0.18170 |
| random_gate | 0.78304 | 0.79969 | 0.18187 |
| rank_001 | 0.78291 | 0.79977 | 0.18133 |
| rank_01 | 0.78175 | 0.79972 | 0.18230 |
| residual | 0.78110 | 0.79880 | 0.18233 |
| uniform_gate | 0.78095 | 0.79839 | 0.18247 |
| v2_reference | 0.78230 | 0.79963 | 0.18186 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

Global sigmoid view weights are bounded and regularized toward .5; uniform/random controls retain five gate parameters with gradients disabled. Residual and depth controls share all parameters. Attention uses five identity vectors; the no-identity ablation fixes these vectors to zero. Pairwise dot products add deterministic interactions. Flat widths minimize parameter discrepancy over a prespecified grid. Focal and ranking objectives are compared under matched V2 features/architecture. See trial records for exact counts, epochs and runtime. No interpretable view weight is a medical explanation.

CPU inference profiling (1000 rows, three repetitions) is in inference_profile.json. Timings include preprocessing and model reconstruction; tree timings include process startup and must not be read as pure kernel latency. Stored and trainable parameter counts differ for radial adaptation, which retains a frozen source anchor. CatBoost search varies depth, learning rate and L2; its generic child field is not used by CatBoost. LightGBM also varies leaf count/min-child samples and XGBoost child weight. Native categorical handling is used for CatBoost; other trees share one-hot availability.
