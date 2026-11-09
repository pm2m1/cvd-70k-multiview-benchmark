"""All plotted data come from saved audit or locked fitted-model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from cvd70k.data.pipeline import clean, load_csv
from cvd70k.evaluation.metrics import calibration_bins
from cvd70k.utils.io import read_json

COLORS = plt.get_cmap("tab10").colors
LABELS = {
    "logistic": "Logistic",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "mlp": "Flat MLP",
    "multiview": "Five-view MLP",
}
METRICS = [
    "ap",
    "auroc",
    "brier",
    "balanced_accuracy",
    "recall",
    "precision",
    "f1",
    "specificity",
    "accuracy",
    "ece_10",
    "log_loss",
]


def setup():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label(family, intervention="standard"):
    return LABELS[family] + ("" if intervention == "standard" else f" ({intervention})")


def save(fig, folder, name, pdf=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotated_confusion(ax, values, title):
    values = np.asarray(values).reshape(2, 2)
    ax.imshow(values, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{values[i, j]:,.1f}",
                ha="center",
                va="center",
                color="white" if values[i, j] > values.max() * 0.6 else "black",
            )
    ax.set(
        xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="Recorded label", title=title
    )
    ax.grid(False)


def dataset_figures(root):
    root = Path(root)
    folder = root / "results/figures"
    audit = read_json(root / "results/aggregate/dataset_audit.json")
    df, _ = load_csv(root / "data/raw/cardio_train.csv")
    strict, _ = clean(df, "strict")
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [audit["class_counts"][str(i)] for i in [0, 1]]
    ax.bar(["Absent (0)", "Present (1)"], vals, color=[COLORS[0], COLORS[1]])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}\n{v / sum(vals):.2%}", ha="center", va="bottom")
    ax.set(
        ylabel="Records", title="Recorded cardiovascular disease label", ylim=(0, max(vals) * 1.17)
    )
    fig.tight_layout()
    save(fig, folder, "01_class_distribution")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (col, title, limits) in zip(
        axes.flat,
        [
            ("age", "Age (years)", (20, 70)),
            ("height", "Height (cm)", (40, 240)),
            ("weight", "Weight (kg)", (0, 220)),
            ("ap_hi", "Systolic BP (mmHg)", (0, 300)),
            ("ap_lo", "Diastolic BP (mmHg)", (0, 200)),
            ("cholesterol", "Cholesterol category", (0.5, 3.5)),
        ],
        strict=True,
    ):
        a, b = df[col], strict[col]
        if col == "age":
            a, b = a / 365.25, b / 365.25
        bins = np.linspace(*limits, 45) if col != "cholesterol" else np.arange(0.5, 4, 1)
        ax.hist(a, bins=bins, alpha=0.5, label="Raw records", color=COLORS[0])
        ax.hist(b, bins=bins, histtype="step", lw=1.5, label="Strict retained", color=COLORS[1])
        outside = int(((a < limits[0]) | (a > limits[1])).sum())
        ax.set(xlabel=title, ylabel="Records", title=f"{outside:,} outside display limits")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Observed feature distributions; bounds shown are display limits", y=1.01)
    fig.tight_layout()
    save(fig, folder, "02_feature_audit")

    fig, ax = plt.subplots(figsize=(10, 5))
    flags = audit["flags_overlapping"]
    ax.barh([k.replace("_", " ") for k in flags], list(flags.values()), color=COLORS[3])
    ax.set(xlabel="Flagged records (overlapping reasons)", title="Prespecified plausibility checks")
    for i, v in enumerate(flags.values()):
        ax.text(v + 8, i, str(v), va="center")
    ax.set_xlim(0, max(flags.values()) * 1.15)
    fig.tight_layout()
    save(fig, folder, "02b_quality_flags")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    def box(x, y, text, color=COLORS[0], w=0.28, h=0.19):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=.015",
            fc=color,
            alpha=0.12,
            ec=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes)

    box(0.36, 0.76, f"Authorized ZIP / CSV\n{len(df):,} records; {df.shape[1]} columns")
    box(0.36, 0.45, "Fixed quality rules + feature groups\nID excluded; duplicates isolated")
    box(
        0.02,
        0.08,
        f"Minimal: {len(df):,} retained\nInvalid measurements masked\nTrain-only imputation",
    )
    box(
        0.36,
        0.08,
        f"Strict: {len(strict):,} retained\n{len(df) - len(strict):,} excluded\nSensitivity analysis",
        COLORS[1],
    )
    box(
        0.70,
        0.08,
        "Frozen group folds\n70% / 15% / 15% (approx.)\nNo evaluation rebalancing",
        COLORS[2],
    )
    for a, b in [
        ((0.5, 0.76), (0.5, 0.64)),
        ((0.5, 0.45), (0.16, 0.27)),
        ((0.5, 0.45), (0.50, 0.27)),
        ((0.5, 0.45), (0.84, 0.27)),
    ]:
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#444"},
        )
    ax.set_title("Observed preprocessing flow and prespecified partition design")
    save(fig, folder, "03_preprocessing_flow")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    names = [
        "Background\nAge + gender (2)",
        "Haemodynamic\nSBP + DBP (2)",
        "Anthropometric\nHeight, weight, BMI (3)",
        "Metabolic\nCholesterol + glucose (6)",
        "Behaviour\nSmoke, alcohol, active (3)",
    ]
    for i, name in enumerate(names):
        yy = 0.86 - i * 0.18
        ax.text(
            0.01,
            yy,
            name,
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "#edf2f8", "ec": COLORS[i]},
        )
        ax.text(
            0.43,
            yy,
            "Encoder 16\nLinear → ReLU → LayerNorm",
            va="center",
            ha="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.annotate(
            "", (0.30, yy), (0.25, yy), xycoords="axes fraction", arrowprops={"arrowstyle": "->"}
        )
        ax.annotate(
            "", (0.61, 0.5), (0.56, yy), xycoords="axes fraction", arrowprops={"arrowstyle": "->"}
        )
    ax.text(
        0.69,
        0.5,
        "Concatenate 80\n↓\nFusion 64\n↓\nFusion 32\n↓\nOne binary logit",
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "fc": "#eff7ec", "ec": COLORS[2]},
    )
    ax.text(
        0.90,
        0.5,
        "BCE with logits\n\n7,985 parameters",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title("Conceptual architecture; configured dimensions and computed parameter count")
    save(fig, folder, "04_architecture")


def prediction(root, run_id):
    return np.load(Path(root) / "private/test_predictions" / f"{run_id}.npz")


def baseline_figures(root):
    root = Path(root)
    folder = root / "results/figures"
    df = pd.read_csv(root / "results/aggregate/test_metrics.csv")
    dev = read_json(root / "results/aggregate/development.json")
    main = df[(df.analysis == "full") & (df.cleaning == "minimal") & (df.operating == "default")]
    groups = list(main.groupby(["family", "intervention"], sort=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["ap", "auroc"], strict=True):
        means = [g[metric].mean() for _, g in groups]
        sd = [g[metric].std() for _, g in groups]
        for i, (mean, deviation) in enumerate(zip(means, sd, strict=True)):
            ax.errorbar(mean, i, xerr=deviation, fmt="o", color=COLORS[i], capsize=3)
            ax.text(mean + 0.0025, i, f"{mean:.4f}", va="center", fontsize=9)
        ax.set_yticks(range(len(groups)), [label(*k) for k, _ in groups])
        baseline = main.prevalence.iloc[0] if metric == "ap" else 0.5
        ax.set(
            xlabel=f"{metric.upper()}\nNo-skill baseline: {baseline:.4f} (outside axis)",
            xlim=(0.74, 0.81),
            title="Mean ± seed SD (same test records)",
        )
    fig.tight_layout()
    save(fig, folder, "06_baseline_comparison")

    for kind in ["pr", "roc", "calibration"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (key, group) in enumerate(groups):
            # Each faint trace is an actual seed fit, not a fictitious ensemble.
            for j, run in enumerate(group.run_id):
                data = prediction(root, run)
                y, p = data["label"], data["probability"]
                legend = label(*key) if j == 0 else None
                if kind == "pr":
                    precision, recall, _ = precision_recall_curve(y, p)
                    ax.plot(recall, precision, color=COLORS[i], alpha=0.6, lw=1, label=legend)
                elif kind == "roc":
                    fpr, tpr, _ = roc_curve(y, p)
                    ax.plot(fpr, tpr, color=COLORS[i], alpha=0.6, lw=1, label=legend)
                else:
                    bins = [b for b in calibration_bins(y, p) if b["n"]]
                    ax.plot(
                        [b["mean_probability"] for b in bins],
                        [b["positive_fraction"] for b in bins],
                        color=COLORS[i],
                        alpha=0.6,
                        lw=1,
                        marker=".",
                        label=legend,
                    )
        if kind == "pr":
            ax.axhline(
                main.prevalence.iloc[0],
                ls="--",
                color="gray",
                label="Test prevalence / AP baseline",
            )
            ax.set(
                xlabel="Recall", ylabel="Precision", title="Precision–recall: all five seed fits"
            )
            name = "07_precision_recall"
        elif kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="gray")
            ax.set(
                xlabel="False positive rate", ylabel="Sensitivity", title="ROC: all five seed fits"
            )
            name = "08_roc"
        else:
            ax.plot([0, 1], [0, 1], ls="--", color="gray")
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed positive fraction",
                title="Calibration: 10 equal-width probability bins",
            )
            name = "10_calibration"
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        save(fig, folder, name)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax, (key, group) in zip(axes.flat, groups):
        annotated_confusion(ax, group[["tn", "fp", "fn", "tp"]].mean().to_numpy(), label(*key))
    axes.flat[-1].axis("off")
    fig.suptitle("Default threshold 0.50; mean counts over five fitted seeds")
    fig.tight_layout()
    save(fig, folder, "09_confusion_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (key, group) in enumerate(groups):
        for ax, metric in zip(axes, ["accuracy", "balanced_accuracy"], strict=True):
            ax.bar(i, group[metric].mean(), yerr=group[metric].std(), color=COLORS[i])
            ax.set_xticks(
                range(len(groups)),
                [label(*k) for k, _ in groups],
                rotation=35,
                ha="right",
                fontsize=8,
            )
            ax.set(ylabel=metric.replace("_", " ").title(), ylim=(0.5, 0.85))
    fig.tight_layout()
    save(fig, folder, "13_accuracy_balanced_accuracy")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (key, group) in enumerate(groups):
        if group.parameter_count.notna().any():
            ax.errorbar(
                group.parameter_count.mean(),
                group.ap.mean(),
                yerr=group.ap.std(),
                marker="o",
                capsize=3,
                label=label(*key),
                color=COLORS[i],
            )
    ax.set(
        xlabel="Trainable scalar coefficients (log scale)",
        ylabel="Test AP",
        xscale="log",
        title="Capacity versus performance; tree parameter counts are not comparable",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, folder, "14_parameters_performance")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family_index, family in enumerate(["mlp", "multiview"]):
        records = [
            r
            for r in dev["records"]
            if r["specification"]["analysis"] == "full"
            and r["specification"]["cleaning"] == "minimal"
            and r["specification"]["family"] == family
            and r["specification"]["intervention"] == "standard"
        ]
        for j, r in enumerate(records):
            curve = pd.DataFrame(r["training"]["curve"])
            axes[0].plot(
                curve.epoch,
                curve.validation_loss,
                alpha=0.65,
                label=LABELS[family] if j == 0 else None,
                color=COLORS[family_index],
            )
            axes[1].plot(
                curve.epoch,
                curve.validation_ap,
                alpha=0.65,
                label=LABELS[family] if j == 0 else None,
                color=COLORS[family_index],
            )
    axes[0].set(xlabel="Epoch (zero = random initialization)", ylabel="Validation BCE")
    axes[1].set(xlabel="Epoch", ylabel="Validation AP")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save(fig, folder, "05_learning_curves")

    grid = pd.read_csv(root / "results/aggregate/threshold_grid.csv")
    full = grid[(grid.analysis == "full") & (grid.cleaning == "minimal")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, ((family, intervention), g) in enumerate(
        full.groupby(["family", "intervention"], sort=False)
    ):
        means = g.groupby("threshold")[["precision", "recall"]].mean()
        for ax, metric in zip(axes, ["precision", "recall"], strict=True):
            ax.plot(means.index, means[metric], label=label(family, intervention), color=COLORS[i])
            ax.set(
                xlabel="Descriptive classification threshold", ylabel=metric.title(), ylim=(0, 1)
            )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, folder, "12_threshold_sensitivity")


def imbalance_figures(root):
    root = Path(root)
    df = pd.read_csv(root / "results/aggregate/test_metrics.csv")
    im = df[df.analysis.isin(["fixed_total", "fixed_positive"]) & (df.operating == "default")]
    for analysis in ["fixed_total", "fixed_positive"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        for ax, metric in zip(
            axes.flat,
            ["ap", "auroc", "brier", "recall", "balanced_accuracy", "accuracy"],
            strict=True,
        ):
            for i, family in enumerate(LABELS):
                for intervention, style in [
                    ("standard", "-"),
                    ("weighted", "--"),
                    ("resampled", ":"),
                ]:
                    sub = im[
                        (im.analysis == analysis)
                        & (im.family == family)
                        & (im.intervention == intervention)
                    ]
                    stat = (
                        sub.groupby("requested_prevalence")[metric]
                        .agg(["mean", "std"])
                        .sort_index()
                    )
                    xx = stat.index.to_numpy() * 100
                    ax.plot(
                        xx,
                        stat["mean"],
                        style,
                        color=COLORS[i],
                        label=label(family, intervention),
                        lw=1.5,
                    )
                    ax.fill_between(
                        xx,
                        stat["mean"] - stat["std"],
                        stat["mean"] + stat["std"],
                        color=COLORS[i],
                        alpha=0.05,
                    )
            ax.set(
                xlabel="Training positives (%)",
                ylabel=metric.replace("_", " ").upper(),
                xticks=[7.42, 15, 25, 50],
            )
            if metric == "ap":
                ax.axhline(im.prevalence.iloc[0], color="gray", ls="--", lw=0.8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.05)
        )
        fig.suptitle(
            f"{analysis.replace('_', ' ').title()}: means ± seed SD; fixed natural test prevalence"
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        save(fig, root / "results/figures", f"11_imbalance_{analysis}")


def diagnostic_figures(root):
    """Every fitted seed appears in its condition's PR/ROC/calibration traces."""
    root = Path(root)
    frame = pd.read_csv(root / "results/aggregate/test_metrics.csv")
    records = read_json(root / "results/aggregate/development.json")["records"]
    by_id = {r["specification"]["run_id"]: r for r in records}
    default = frame[frame.operating == "default"]
    keys = [
        "analysis",
        "cleaning",
        "anthropometry",
        "requested_prevalence",
        "family",
        "intervention",
    ]
    for key, group in default.groupby(keys, dropna=False, sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for i, run in enumerate(group.run_id):
            data = prediction(root, run)
            y, p = data["label"], data["probability"]
            precision, recall, _ = precision_recall_curve(y, p)
            fpr, tpr, _ = roc_curve(y, p)
            bins = [b for b in calibration_bins(y, p) if b["n"]]
            seed = by_id[run]["specification"]["seed"]
            axes[0, 0].plot(
                recall, precision, lw=1, alpha=0.8, label=f"Seed {seed}", color=COLORS[i]
            )
            axes[0, 1].plot(fpr, tpr, lw=1, alpha=0.8, color=COLORS[i])
            axes[0, 2].plot(
                [b["mean_probability"] for b in bins],
                [b["positive_fraction"] for b in bins],
                lw=1,
                alpha=0.8,
                marker=".",
                color=COLORS[i],
            )
            curve = pd.DataFrame(by_id[run]["training"]["curve"])
            if not curve.empty:
                axes[1, 2].plot(
                    curve.epoch,
                    curve.validation_loss,
                    alpha=0.8,
                    color=COLORS[i],
                    label=f"Val {seed}",
                )
                if "train_objective" in curve:
                    axes[1, 2].plot(
                        curve.epoch, curve.train_objective, ls=":", alpha=0.6, color=COLORS[i]
                    )
        axes[0, 0].axhline(y.mean(), ls="--", color="gray")
        axes[0, 0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
        axes[0, 1].plot([0, 1], [0, 1], ls="--", color="gray")
        axes[0, 1].set(xlabel="False positive rate", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
        axes[0, 2].plot([0, 1], [0, 1], ls="--", color="gray")
        axes[0, 2].set(
            xlabel="Mean probability", ylabel="Positive fraction", xlim=(0, 1), ylim=(0, 1)
        )
        axes[0, 0].legend(fontsize=8)
        annotated_confusion(
            axes[1, 0],
            group[["tn", "fp", "fn", "tp"]].mean().to_numpy(),
            "Threshold .50; mean seed counts",
        )
        selected = frame[
            (frame.run_id.isin(group.run_id)) & (frame.operating == "validation_selected")
        ]
        annotated_confusion(
            axes[1, 1],
            selected[["tn", "fp", "fn", "tp"]].mean().to_numpy(),
            "Validation threshold; mean seed counts",
        )
        axes[1, 2].set(
            xlabel="Epoch / boosting round",
            ylabel="Loss",
            title="Validation BCE; dotted = train objective",
        )
        if curve.empty:
            axes[1, 2].text(
                0.5,
                0.5,
                "No iterative validation curve\nfor this estimator",
                ha="center",
                va="center",
                transform=axes[1, 2].transAxes,
            )
        title = " | ".join(str(x) for x in key if str(x) != "nan")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        name = "_".join(str(x) for x in key).replace(".", "p").replace("/", "_")
        save(fig, root / "results/figures/diagnostics", name, pdf=False)


def render_all(root):
    setup()
    dataset_figures(root)
    baseline_figures(root)
    imbalance_figures(root)
    diagnostic_figures(root)
