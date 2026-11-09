"""Historical diagnostics, using development artifacts only; no test predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cvd70k.utils.io import read_json

root = Path.cwd()
dev = read_json(root / "results/aggregate/development.json")
rows = []
curves = []
for record in dev["records"]:
    spec = record["specification"]
    if spec["family"] not in ["mlp", "multiview"]:
        continue
    curve = pd.DataFrame(record["training"]["curve"])
    eligible = curve[curve.epoch > 0]
    best = int(record["training"]["best_epoch"])
    ap_best = int(eligible.loc[eligible.validation_ap.idxmax(), "epoch"])
    rows.append(
        {
            **{
                k: spec[k]
                for k in [
                    "run_id",
                    "family",
                    "seed",
                    "analysis",
                    "cleaning",
                    "intervention",
                    "requested_prevalence",
                ]
            },
            "max_epochs": 60,
            "actual_epochs": record["training"]["epochs_run"],
            "best_bce_epoch": best,
            "best_ap_epoch": ap_best,
            "ap_minus_bce_epoch": ap_best - best,
            "best_near_max": best >= 55,
            "selected_validation_ap": record["validation_default"]["ap"],
            "peak_validation_ap": eligible.validation_ap.max(),
            "ap_gain_by_reselection_only": eligible.validation_ap.max()
            - record["validation_default"]["ap"],
            "best_train_loss": float(curve.loc[curve.epoch == best, "train_objective"].iloc[0]),
            "best_validation_bce": float(curve.loc[curve.epoch == best, "validation_loss"].iloc[0]),
        }
    )
    curve["run_id"] = spec["run_id"]
    curves.append(curve)
frame = pd.DataFrame(rows)
frame.to_csv(root / "results/v2/audit/v1_training_diagnostics.csv", index=False)
pd.concat(curves).to_csv(root / "results/v2/audit/v1_learning_curves.csv", index=False)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for family, color in [("mlp", "tab:blue"), ("multiview", "tab:orange")]:
    subset = frame[
        (frame.family == family)
        & (frame.analysis == "full")
        & (frame.cleaning == "minimal")
        & (frame.intervention == "standard")
    ]
    for j, run in enumerate(subset.run_id):
        curve = next(c for c in curves if c.run_id.iloc[0] == run)
        for ax, metric in zip(
            axes, ["train_objective", "validation_loss", "validation_ap"], strict=True
        ):
            ax.plot(
                curve.epoch,
                curve[metric],
                color=color,
                alpha=0.65,
                label=family if j == 0 else None,
            )
            ax.set(xlabel="Epoch", ylabel=metric.replace("_", " "))
for ax in axes:
    ax.legend()
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        root / f"results/v2/figures/v1_learning_diagnostics.{ext}", dpi=200, bbox_inches="tight"
    )
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, family in enumerate(["mlp", "multiview"]):
    s = frame[frame.family == family]
    axes[0].scatter([i] * len(s), s.best_bce_epoch, alpha=0.15, s=12)
    axes[1].hist(s.ap_minus_bce_epoch, bins=25, alpha=0.5, label=family)
axes[0].set(
    xticks=[0, 1],
    xticklabels=["Flat MLP", "Five-view"],
    ylabel="Selected BCE epoch",
    title="All historical neural conditions",
)
axes[1].set(xlabel="AP-optimal epoch minus BCE-optimal epoch", ylabel="Fits")
axes[1].legend()
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(
        root / f"results/v2/figures/v1_checkpoint_diagnostics.{ext}", dpi=200, bbox_inches="tight"
    )
plt.close(fig)
rare = frame[
    (frame.family == "mlp")
    & (frame.requested_prevalence == 0.0742)
    & (frame.intervention == "standard")
]
main = frame[(frame.analysis == "full") & (frame.cleaning == "minimal")]
text = f"""# V1 training diagnostic — historical development evidence only

{len(frame)} neural fits inspected from the saved V1 development manifest; no test predictions
or V1 checkpoints were evaluated. All allowed at most 60 epochs. {int(frame.best_near_max.sum())}
selected checkpoints lie within five epochs of that limit. {int((frame.actual_epochs == 60).sum())}
fits completed 60 epochs. Median best BCE epoch is {frame.best_bce_epoch.median():.1f}; median
completed epochs is {frame.actual_epochs.median():.1f}. AP peaked after BCE's selected epoch
in {int((frame.ap_minus_bce_epoch > 0).sum())} fits, earlier in
{int((frame.ap_minus_bce_epoch < 0).sum())}; these are diagnostics, not permission to retrospectively
replace historical checkpoints. Median possible validation AP reselection gain is
{frame.ap_gain_by_reselection_only.median():.6f} (optimistically measured on the same validation).

All {len(rare)} standard flat-MLP rare-prevalence fits selected BCE epoch 1 and stopped at
{sorted(rare.actual_epochs.unique().tolist())}. This directly verifies early checkpoint
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

Full-training selected-epoch summaries:\n\n{main.groupby(["family", "intervention"]).best_bce_epoch.agg(["min", "median", "max"]).to_markdown() if False else main.groupby(["family", "intervention"]).best_bce_epoch.agg(["min", "median", "max"]).to_string()}

The CSV and plotted curves are aggregate per-fit/per-epoch records, not participant data.
"""
(root / "docs/v2/V1_TRAINING_DIAGNOSTIC.md").write_text(text)
print(frame.groupby(["family", "analysis"]).best_bce_epoch.agg(["count", "min", "median", "max"]))
