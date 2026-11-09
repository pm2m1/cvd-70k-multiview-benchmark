# V1 training diagnostic — historical development evidence only

290 neural fits inspected from the saved V1 development manifest; no test predictions
or V1 checkpoints were evaluated. All allowed at most 60 epochs. 0
selected checkpoints lie within five epochs of that limit. 0
fits completed 60 epochs. Median best BCE epoch is 10.0; median
completed epochs is 18.0. AP peaked after BCE's selected epoch
in 170 fits, earlier in
66; these are diagnostics, not permission to retrospectively
replace historical checkpoints. Median possible validation AP reselection gain is
0.000582 (optimistically measured on the same validation).

All 10 standard flat-MLP rare-prevalence fits selected BCE epoch 1 and stopped at
[9]. This directly verifies early checkpoint
selection under training/evaluation-prior mismatch. It does not prove batch size 1024 caused
underfitting. Weighted/resampled models changed the training objective and effective prior;
comparative curve/epoch records show their actual behavior. No saved gradient norms,
per-epoch training AP, or per-epoch AUROC/probabilities exist, so those historical trajectories
cannot be reconstructed honestly without new development-only fitting. No optimizer failures
are recorded in successful neural runs; the separately documented native XGBoost crash is
not a neural convergence failure.

Verified finding: the selected-checkpoint limit was usually below 60, so a larger epoch cap
alone cannot rescue runs already stopped by patience. Hypotheses to test: AP/AUROC stopping
may preserve useful ranking despite BCE prior mismatch; smaller batches or changed capacity
may alter optimization. Falling training loss with worsening validation BCE/AP is compatible
with overfitting, but stochastic validation variation and prior mismatch also matter. No
single curve proves underfitting. The V2 nested protocol tests these possibilities separately.

Full-training selected-epoch summaries:

                        min  median  max
family    intervention
mlp       standard       14    20.0   26
multiview resampled       6    12.0   14
          standard        6    12.0   15
          weighted        6    13.0   20

The CSV and plotted curves are aggregate per-fit/per-epoch records, not participant data.
