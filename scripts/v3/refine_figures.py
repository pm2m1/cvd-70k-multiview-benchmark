"""Presentation-only refinement of locked aggregate figures; no fitting or selection."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = Path("results/v3/figures")
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=220, metadata={"Software": "cvd70k V3"})
    fig.savefig(
        FIG / f"{name}.pdf",
        metadata={"Creator": "cvd70k V3", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main():
    summary = pd.read_csv("results/v3/aggregate/classification_summary.csv").set_index("model")
    metrics = pd.read_csv("results/v3/aggregate/classification_metrics.csv")
    names = [
        "v1_reference",
        "v2_reference",
        "gated",
        "residual",
        "attention",
        "pairwise",
        "flat_attention_matched",
        "ft_reference",
        "selected_xgboost",
        "selected_lightgbm",
        "selected_catboost",
        "selected_neural_seed_average",
        "selected_tree_seed_average",
    ]
    colors = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        row = summary.loc[name]
        ax.scatter(
            row.auroc, row.ap, label=name, color=colors(i), marker=["o", "s", "^", "D"][i % 4], s=42
        )
    ax.set(
        xlabel="Mean outer AUROC",
        ylabel="Mean outer AP",
        title="Matched V3 internal classification",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    save(fig, "03_ap_auroc")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        subset = metrics[metrics.model == name]
        count = subset.parameter_count.mean()
        if pd.notna(count):
            ax.scatter(
                count,
                subset.ap.mean(),
                label=name,
                color=colors(i),
                marker=["o", "s", "^", "D"][i % 4],
                s=42,
            )
    ax.set(
        xlabel="Stored neural parameters",
        ylabel="Mean outer AP",
        title="Actual architecture capacity and performance",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    save(fig, "04_capacity")
    transfer = pd.read_csv("results/v3/aggregate/transfer_summary.csv")
    methods = [
        "source_only",
        "head",
        "full",
        "full_cost",
        "logical",
        "random_matched",
        "full_sp_low",
        "full_sp_high",
        "soft_uniform",
        "soft_evidence",
        "soft_learned",
        "reg_combined",
        "selected_new_policy",
    ]
    for number, column in [(7, "ap"), (8, "auroc")]:
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, method in enumerate(methods):
            subset = transfer[transfer.model == method].sort_values("positive_budget")
            ax.plot(
                subset.positive_budget,
                subset[column],
                marker=["o", "s", "^", "D"][i % 4],
                linestyle=["-", "--", ":"][i % 3],
                color=colors(i),
                label=method,
                linewidth=1.5,
            )
        ax.set_xscale("log")
        ax.set_xticks([25, 250, 1000], ["25", "250", "1000"])
        ax.set(
            xlabel="Fitting positives (equal fitting negatives; total label access differs)",
            ylabel="Mean outer " + column.upper(),
            title="Internal age-subgroup adaptation; fixed natural evaluation population",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
        save(fig, f"{number:02d}_label_budget_{column}")


if __name__ == "__main__":
    main()
