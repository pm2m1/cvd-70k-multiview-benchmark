# Constrained radial soft adaptation

| positive_budget | model | ap | auroc | brier |
| --- | --- | --- | --- | --- |
| 25 | soft_evidence | 0.79170 | 0.73178 | 0.20830 |
| 25 | soft_learned | 0.78609 | 0.72466 | 0.21770 |
| 25 | soft_random | 0.79053 | 0.73045 | 0.21408 |
| 25 | soft_shuffled | 0.79174 | 0.73200 | 0.20895 |
| 25 | soft_uniform | 0.78538 | 0.72361 | 0.21281 |
| 250 | soft_evidence | 0.79184 | 0.73271 | 0.20936 |
| 250 | soft_learned | 0.79116 | 0.73107 | 0.21006 |
| 250 | soft_random | 0.79105 | 0.73159 | 0.20976 |
| 250 | soft_shuffled | 0.79175 | 0.73256 | 0.20931 |
| 250 | soft_uniform | 0.79096 | 0.73137 | 0.20970 |
| 1000 | soft_evidence | 0.79167 | 0.73310 | 0.20787 |
| 1000 | soft_learned | 0.79245 | 0.73250 | 0.20828 |
| 1000 | soft_random | 0.79201 | 0.73245 | 0.20766 |
| 1000 | soft_shuffled | 0.79204 | 0.73302 | 0.20863 |
| 1000 | soft_uniform | 0.79228 | 0.73254 | 0.20767 |

All estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.

No independently trainable gate/delta scaling pair is present. Delta has a fixed norm cap; effective radial gate is cap times norm(delta)/radius. Thus reported gate values are identified by effective drift, not free attribution parameters. The learned-radial cap1 differs from fixed uniform cap.5: this is a constraint ablation, not a claim that gate learning alone caused differences. Evidence/random/shuffled/uniform controls share the parameterization. Frozen zero-cap modules and exact source initialization are verified.

The learned-radial retained gates reached at most0.46868, below their cap1. Thus the cap1 projection was not active at retained checkpoints; this variant is close to a reparameterized L2-SP fit, not evidence of an independently learned clinical gate. Fixed caps also scale effective update steps. Evidence caps and learned radial norms must not be conflated. No unconstrained joint sigmoid-gate/delta model was executed.
