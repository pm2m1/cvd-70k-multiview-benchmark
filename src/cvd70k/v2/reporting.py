"""Generate V2 evidence tables, paired uncertainty, reports and publication plots."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve

from cvd70k.evaluation.metrics import calibration_bins
from cvd70k.evaluation.report import markdown_table
from cvd70k.utils.io import read_json, write_json
from cvd70k.v2.data import development_data, partition_transfer
from cvd70k.v2.engine import PRIVATE, TRIALS
from cvd70k.v2.models import MODULES
from cvd70k.v2.statistics import paired_interval

ROOT = Path.cwd()
AGG = ROOT / "results/v2/aggregate"
DOC = ROOT / "docs/v2"
FIG = ROOT / "results/v2/figures"
CORE = [
    "ap",
    "auroc",
    "brier",
    "log_loss",
    "ece_10",
    "balanced_accuracy",
    "recall",
    "specificity",
    "precision",
    "f1",
    "accuracy",
]


def save(fig, name):
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def table(frame, group="model", metrics=("ap", "auroc", "brier", "balanced_accuracy", "recall")):
    rows = []
    for key, g in frame.groupby(group, sort=False):
        keys = list(key) if isinstance(key, tuple) else [key]
        rows.append(
            keys
            + [
                f"{g[c].mean():.5f} ± {g[c].std():.5f}" if len(g) > 1 else f"{g[c].iloc[0]:.5f}"
                for c in metrics
            ]
        )
    headers = [group] if isinstance(group, str) else group
    return markdown_table(headers + list(metrics), rows)


def document(name, text):
    (DOC / name).write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def outer_arrays(model, seed, calibration="none"):
    pieces = []
    rows = []
    folds = []
    for outer in range(3):
        path = (
            PRIVATE
            / "outer_predictions"
            / f"{model.replace(chr(47), chr(95) * 2)}_o{outer}_s{seed}_{calibration}.npz"
        )
        data = np.load(path)
        pieces.append(data["probability"])
        rows.append(data["row"])
        folds.append(np.full(len(data["row"]), outer))
    row = np.concatenate(rows)
    order = np.argsort(row)
    return row[order], np.concatenate(pieces)[order], np.concatenate(folds)[order]


def optimization_statistics(frame, cfg):
    df = development_data()
    comparisons = []
    pairs = [
        ("multiview/selected", "multiview/v1_equivalent"),
        ("multiview/selected", "mlp/selected"),
        ("multiview/selected", "xgboost/selected"),
        ("ft_transformer/selected", "xgboost/selected"),
        ("tabnet/selected", "xgboost/selected"),
        ("selection/best_ensemble", "selection/best_single"),
    ]
    for a, b in pairs:
        seeds = cfg["seeds"]
        aa = []
        bb = []
        for seed in seeds:
            row, pa, fold = outer_arrays(a, seed)
            row_b, pb, fold_b = outer_arrays(b, seed)
            assert np.array_equal(row, row_b) and np.array_equal(fold, fold_b)
            aa.append(pa)
            bb.append(pb)
        result = paired_interval(
            df.loc[row, "cardio"].to_numpy(),
            aa,
            bb,
            df.loc[row, "group"].to_numpy(),
            fold,
            cfg["bootstrap_replicates"],
        )
        comparisons.append({"a": a, "b": b, **result})
        print("Optimization paired comparison", a, b, flush=True)
    # Prespecified one-seed architecture/training ablations, three matched outer folds.
    pairs = [
        ("epochs120", "v1_equivalent"),
        ("epochs200", "v1_equivalent"),
        ("patience16", "epochs120"),
        ("ap_stop", "patience16"),
        ("auroc_stop", "patience16"),
        ("batch256", "ap_stop"),
        ("width32", "batch256"),
        ("width64", "batch256"),
        ("derived_features", "batch256"),
        ("missing_indicators", "batch256"),
        ("combined_features", "batch256"),
        ("robust_combined", "combined_features"),
    ]
    for a, b in pairs:
        ra, pa, fold = outer_arrays("multiview/" + a, 101)
        rb, pb, _ = outer_arrays("multiview/" + b, 101)
        assert np.array_equal(ra, rb)
        result = paired_interval(
            df.loc[ra, "cardio"].to_numpy(),
            [pa],
            [pb],
            df.loc[ra, "group"].to_numpy(),
            fold,
            cfg["bootstrap_replicates"],
        )
        comparisons.append({"a": "multiview/" + a, "b": "multiview/" + b, "seeds": [101], **result})
    write_json(AGG / "optimization_comparisons.json", comparisons)
    return comparisons


def transfer_statistics(frame, cfg):
    parts = partition_transfer(development_data())
    test = parts["target_evaluation"]
    comparisons = []
    main = frame[frame.design == "support"]
    for positive in cfg["transfer"]["positive_budgets"]:
        for reference in ["full", "head_only", "random_matched", "full_cost_matched"]:
            a = []
            b = []
            for seed in cfg["seeds"]:
                left = main[
                    (main.positive == positive)
                    & (main.seed == seed)
                    & (main.method == "calibrated_logical")
                ].iloc[0]
                right = main[
                    (main.positive == positive) & (main.seed == seed) & (main.method == reference)
                ].iloc[0]
                for collection, record in [(a, left), (b, right)]:
                    data = np.load(PRIVATE / "transfer_predictions" / f"{record.run}.npz")
                    assert np.array_equal(data["row"], test.index)
                    collection.append(data["probability"])
            result = paired_interval(
                test.cardio.to_numpy(),
                a,
                b,
                test.group.to_numpy(),
                replicates=cfg["bootstrap_replicates"],
            )
            comparisons.append(
                {"positive_budget": positive, "a": "calibrated_logical", "b": reference, **result}
            )
    write_json(AGG / "transfer_comparisons.json", comparisons)
    evidence = read_json(AGG / "transfer_evidence.json")
    rows = []
    stability = []
    for record in evidence:
        for module, ev in record["modules"].items():
            rows.append(
                {
                    "positive_budget": record["positive"],
                    "negative_budget": record["negative"],
                    "seed": record["seed"],
                    "condition": record["condition"],
                    "module": module,
                    "adapt": record["logical_mask"][module],
                    **{
                        k: ev.get(k)
                        for k in [
                            "geometry_status",
                            "R",
                            "U",
                            "domain_p",
                            "source_discriminability",
                            "target_discriminability",
                            "bootstrap_valid_fraction",
                        ]
                    },
                }
            )
    e = pd.DataFrame(rows)
    e.to_csv(AGG / "selector_module_evidence.csv", index=False)
    for positive in cfg["transfer"]["positive_budgets"]:
        subset = [
            r
            for r in evidence
            if r["condition"] == "primary"
            and r["positive"] == positive
            and r["design"] == "support"
        ]
        masks = [r["logical_mask"] for r in subset]
        agreement = []
        jaccard = []
        for a, b in itertools.combinations(masks, 2):
            agreement.append(np.mean([a[n] == b[n] for n in MODULES]))
            sa, sb = {n for n in MODULES if a[n]}, {n for n in MODULES if b[n]}
            jaccard.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        stability.append(
            {
                "positive_budget": positive,
                "head_only_fraction": float(np.mean([not any(m.values()) for m in masks])),
                "module_agreement": float(np.mean(agreement)),
                "adapt_set_jaccard": float(np.mean(jaccard)),
                "invalid_geometry_fraction": float(
                    np.mean(
                        [
                            v["geometry_status"] != "valid"
                            for r in subset
                            for v in r["modules"].values()
                        ]
                    )
                ),
                "adapt_frequency": {n: float(np.mean([m[n] for m in masks])) for n in MODULES},
                "random_equals_logical_fraction": float(
                    np.mean([r["random_control"]["equals_logical"] for r in subset])
                ),
            }
        )
    write_json(AGG / "selector_stability.json", stability)
    return comparisons, e, stability


def plots(optimization, transfer, evidence, cfg):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    raw = optimization[
        (optimization.calibration == "none") & (optimization.operating == "fixed_050")
    ]
    selected = raw[raw.model.str.endswith("/selected")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(selected.model.unique())
    for ax, metric in zip(axes, ["ap", "auroc"], strict=True):
        for i, name in enumerate(names):
            g = selected[selected.model == name]
            ax.errorbar(g[metric].mean(), i, xerr=g[metric].std(), fmt="o", capsize=3)
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel=metric.upper(), title="V2 outer-fold mean ± fold/seed SD")
    save(fig, "v2_outer_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for family in ["mlp", "multiview"]:
        g = raw[(raw.family == family) & (raw.seed == 101) & ~raw.model.str.endswith("/selected")]
        means = g.groupby("model")[["ap", "auroc"]].mean()
        for ax, metric in zip(axes, ["ap", "auroc"], strict=True):
            ax.plot(means.index.str.split("/").str[-1], means[metric], marker="o", label=family)
            ax.tick_params(axis="x", rotation=90)
            ax.set(ylabel=metric.upper())
            ax.legend()
    save(fig, "v2_training_architecture_ablations")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family in cfg["families"]:
        g = optimization[
            (optimization.model == family + "/selected") & (optimization.operating == "fixed_050")
        ]
        means = g.groupby("calibration")[["brier", "ece_10"]].mean()
        for ax, metric in zip(axes, ["brier", "ece_10"], strict=True):
            ax.plot(means.index, means[metric], marker="o", label=family)
            ax.set(ylabel=metric)
    axes[0].legend(fontsize=8)
    save(fig, "v2_calibration_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["selection/best_single", "selection/best_ensemble"]:
        g = raw[raw.model == name]
        for ax, metric in zip(axes, ["ap", "auroc"], strict=True):
            stat = g.groupby("outer")[metric].agg(["mean", "std"])
            ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], marker="o", label=name)
            ax.set(xlabel="Outer fold", ylabel=metric.upper(), xticks=[0, 1, 2])
            ax.legend(fontsize=8)
    save(fig, "v2_ensembles")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for intervention, style in [("standard", "-"), ("weighted", "--"), ("resampled", ":")]:
        g = optimization[
            (optimization.model.str.startswith("imbalance/"))
            & (optimization.model.str.endswith("/" + intervention))
            & (optimization.calibration == "none")
            & (optimization.operating == "fixed_050")
        ].copy()
        g["train_prevalence"] = g.model.str.split("/").str[1].astype(float) * 100
        for ax, metric in zip(axes, ["ap", "brier", "recall"], strict=True):
            stat = g.groupby("train_prevalence")[metric].agg(["mean", "std"])
            ax.errorbar(
                stat.index, stat["mean"], yerr=stat["std"], marker="o", ls=style, label=intervention
            )
            ax.set(xlabel="Training positives (%)", ylabel=metric, xticks=[7.42, 50])
            ax.legend(fontsize=8)
    save(fig, "v2_imbalance")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = [
        "source_only",
        "target_only",
        "head_only",
        "full",
        "fixed",
        "random_matched",
        "legacy_topk",
        "calibrated_threshold",
        "calibrated_logical",
        "full_cost_matched",
        "target_xgboost",
    ]
    main = transfer[transfer.design == "support"]
    for method in methods:
        g = main[main.method == method]
        for ax, metric in zip(axes, ["ap", "auroc"], strict=True):
            stat = g.groupby("positive")[metric].agg(["mean", "std"])
            ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], marker=".", label=method, lw=1)
            ax.set(
                xlabel="Target fitting positive budget (+ equal negatives)",
                ylabel=metric.upper(),
                xscale="log",
                xticks=[25, 100, 250, 500, 1000],
            )
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "v2_transfer_label_budgets")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    primary = evidence[
        (evidence.condition == "primary") & (evidence.negative_budget == evidence.positive_budget)
    ]
    frequency = primary.pivot_table(
        index="module", columns="positive_budget", values="adapt", aggfunc="mean"
    ).reindex(MODULES)
    image = axes[0].imshow(frequency, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[0].set(
        xticks=range(5),
        xticklabels=frequency.columns,
        yticks=range(7),
        yticklabels=MODULES,
        xlabel="Positive fitting budget",
        title="Logical ADAPT frequency over paired seeds",
    )
    fig.colorbar(image, ax=axes[0], label="Fraction ADAPT")
    for i, module in enumerate(MODULES):
        g = primary[primary.module == module].groupby("positive_budget").U.mean()
        axes[1].plot(g.index, g, marker=".", label=module)
    axes[1].axhline(0.75, color="black", ls="--", lw=1)
    axes[1].set(
        xscale="log", xlabel="Positive fitting budget", ylabel="Mean upper retention bound U"
    )
    axes[1].legend(fontsize=7)
    save(fig, "v2_selector_masks_uncertainty")
    # Actual saved inner learning curves for chosen multiview candidates.
    selections = read_json(AGG / "optimization_selections.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for outer in range(3):
        c = selections[str(outer)]["selected"]["multiview"]
        for j in range(3):
            curve = pd.DataFrame(
                read_json(TRIALS / f"inner_o{outer}_j{j}_multiview_c{c}_s101.json")["training"][
                    "curve"
                ]
            )
            for ax, metric in zip(
                axes, ["train_loss", "validation_bce", "validation_ap"], strict=True
            ):
                ax.plot(
                    curve.epoch,
                    curve[metric],
                    alpha=0.6,
                    label=f"Outer {outer}" if j == 0 else None,
                )
                ax.set(xlabel="Epoch", ylabel=metric)
    axes[0].legend()
    save(fig, "v2_selected_learning_curves")


def additional_plots(optimization, transfer):
    raw = optimization[
        (optimization.calibration == "none") & (optimization.operating == "fixed_050")
    ]
    df = development_data()
    for kind in ["pr", "roc"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for family in ["xgboost", "mlp", "multiview", "ft_transformer", "tabnet"]:
            row, p, _ = outer_arrays(family + "/selected", 101)
            y = df.loc[row, "cardio"].to_numpy()
            if kind == "pr":
                precision, recall, _ = precision_recall_curve(y, p)
                ax.plot(recall, precision, label=family, lw=1)
            else:
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, label=family, lw=1)
        if kind == "pr":
            ax.axhline(y.mean(), color="gray", ls="--")
            ax.set(xlabel="Recall", ylabel="Precision")
        else:
            ax.plot([0, 1], [0, 1], ls="--", color="gray")
            ax.set(xlabel="False positive rate", ylabel="Sensitivity")
        ax.set(
            title="Pooled outer predictions, seed101; primary metrics are fold-wise",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
        save(fig, "v2_" + kind + "_curves")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, family in zip(
        axes.flat, ["xgboost", "multiview", "ft_transformer", "tabnet"], strict=True
    ):
        for kind in ["none", "platt", "isotonic"]:
            row, p, _ = outer_arrays(family + "/selected", 101, kind)
            bins = [b for b in calibration_bins(df.loc[row, "cardio"].to_numpy(), p) if b["n"]]
            ax.plot(
                [b["mean_probability"] for b in bins],
                [b["positive_fraction"] for b in bins],
                marker=".",
                label=kind,
            )
        ax.plot([0, 1], [0, 1], color="gray", ls="--")
        ax.set(
            title=family,
            xlabel="Mean probability",
            ylabel="Observed positive fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    fig.suptitle("OOF reliability, seed101;10 equal-width bins")
    save(fig, "v2_reliability")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, g in raw[raw.model.str.endswith("/selected")].groupby("model"):
        if g.parameter_count.notna().any():
            axes[0].scatter(g.parameter_count.mean(), g.ap.mean(), label=name)
        if "inference_seconds" in g:
            axes[1].scatter((g.inference_seconds / g.n * 1000).mean(), g.ap.mean(), label=name)
    axes[0].set(xscale="log", xlabel="Trainable scalar parameters (trees N/A)", ylabel="Outer AP")
    axes[1].set(
        xscale="log", xlabel="CPU prediction seconds /1000 rows (load included)", ylabel="Outer AP"
    )
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    save(fig, "v2_capacity_prediction_cost")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    main = transfer[
        (transfer.design == "support")
        & transfer.method.isin(
            [
                "head_only",
                "full",
                "calibrated_logical",
                "head_only_bce_zero",
                "full_bce_zero",
                "calibrated_logical_bce_zero",
                "head_only_ap_nozero",
                "full_ap_nozero",
                "calibrated_logical_ap_nozero",
            ]
        )
    ]
    for name, g in main.groupby("method"):
        stat = g.groupby("positive")[["ap", "best_epoch"]].mean()
        for ax, metric in zip(axes, ["ap", "best_epoch"], strict=True):
            ax.plot(stat.index, stat[metric], marker=".", label=name)
            ax.set(xscale="log", xlabel="Positive fitting budget", ylabel=metric)
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "v2_adaptation_checkpoint_rules")


def diagnostic_ap_plot():
    records = read_json(AGG / "learning_diagnostics.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, record in zip(axes, records, strict=True):
        curve = pd.DataFrame(record["training"]["curve"])
        ax.plot(curve.epoch, curve.train_ap, label="Train AP (eval mode)")
        ax.plot(curve.epoch, curve.validation_ap, label="Inner validation AP")
        ax.set(xlabel="Epoch", ylabel="AP", title=f"Diagnostic refit, outer{record['outer']}")
        ax.legend(fontsize=8)
    save(fig, "v2_training_validation_ap")


def tabnet_stability():
    selections = read_json(AGG / "optimization_selections.json")
    rows = []
    for outer in range(3):
        c = selections[str(outer)]["selected"]["tabnet"]
        for seed in [101, 202, 303]:
            r = read_json(TRIALS / f"refit_o{outer}_tabnet_c{c}_s{seed}.json")
            rows.append(
                {
                    "outer": outer,
                    "seed": seed,
                    "aggregate_mask_importance": r["training"]["aggregate_feature_importance"],
                }
            )
    correlations = []
    for a, b in itertools.combinations(rows, 2):
        value = float(
            spearmanr(a["aggregate_mask_importance"], b["aggregate_mask_importance"]).statistic
        )
        if np.isfinite(value):
            correlations.append(value)
    result = {
        "scope": "training-aggregate attention-mask importance; not participant explanations or causal medical evidence",
        "model_count": len(rows),
        "mean_pairwise_spearman": float(np.mean(correlations)) if correlations else None,
        "minimum_pairwise_spearman": float(np.min(correlations)) if correlations else None,
        "aggregate_masks": rows,
    }
    write_json(AGG / "tabnet_mask_stability.json", result)
    return result


def generate(skip_bootstrap=False):
    cfg = read_json(ROOT / "configs/v2/study.json")
    optimization = pd.read_csv(AGG / "optimization_metrics.csv")
    transfer = pd.read_csv(AGG / "transfer_metrics.csv")
    oc = (
        read_json(AGG / "optimization_comparisons.json")
        if skip_bootstrap
        else optimization_statistics(optimization, cfg)
    )
    if skip_bootstrap:
        tc = read_json(AGG / "transfer_comparisons.json")
        evidence = pd.read_csv(AGG / "selector_module_evidence.csv")
        stability = read_json(AGG / "selector_stability.json")
    else:
        tc, evidence, stability = transfer_statistics(transfer, cfg)
    raw = optimization[
        (optimization.calibration == "none") & (optimization.operating == "fixed_050")
    ]
    selected = raw[raw.model.str.endswith("/selected")]
    selected_table = table(selected)
    folds_table = table(selected, ["outer", "model"])
    optimization.groupby(["model", "calibration", "operating"], dropna=False)[CORE].agg(
        ["mean", "std"]
    ).to_csv(AGG / "optimization_summary.csv")
    transfer.groupby(["positive", "negative", "method"])[
        CORE + ["trainable_parameters", "total_target_labels"]
    ].agg(["mean", "std"]).to_csv(AGG / "transfer_summary.csv")
    selections = read_json(AGG / "optimization_selections.json")
    hp = []
    for outer, spec in selections.items():
        for family, c in spec["selected"].items():
            candidate = cfg["candidates"][family][c]
            score = next(
                r for r in spec["candidate_scores"] if r["family"] == family and r["candidate"] == c
            )
            hp.append(
                {
                    "outer": int(outer),
                    "family": family,
                    "candidate": c,
                    "configuration": candidate,
                    "refit_epochs": score["refit_epochs"],
                    "mean_inner_ap": score["inner_mean_ap"],
                    "mean_inner_auroc": score["inner_mean_auroc"],
                    "mean_outer_ap": float(
                        selected[
                            (selected.outer == int(outer)) & (selected.family == family)
                        ].ap.mean()
                    ),
                    "outer_minus_inner_ap": float(
                        selected[
                            (selected.outer == int(outer)) & (selected.family == family)
                        ].ap.mean()
                        - score["inner_mean_ap"]
                    ),
                }
            )
    write_json(AGG / "selected_hyperparameters.json", hp)
    comparison_table = markdown_table(
        ["A − B", "ΔAP", "95% group-bootstrap CI", "ΔAUROC", "95% CI"],
        [
            [
                r["a"] + " − " + r["b"],
                f"{r['ap_difference']:+.5f}",
                str([round(v, 5) for v in r["ap_ci95"]]),
                f"{r['auroc_difference']:+.5f}",
                str([round(v, 5) for v in r["auroc_ci95"]]),
            ]
            for r in oc[:6]
        ],
    )
    ablations = raw[
        (raw.family == "multiview") & (raw.seed == 101) & ~raw.model.str.endswith("/selected")
    ]
    ablation_table = table(ablations, metrics=("ap", "auroc", "brier", "best_epoch", "fit_seconds"))
    primary = transfer[transfer.design == "support"]
    transfer_table = table(primary, ["positive", "method"])
    transfer_comparison = markdown_table(
        ["Fit positives", "Logical − comparator", "ΔAP", "AP95% CI", "ΔAUROC", "AUROC95% CI"],
        [
            [
                r["positive_budget"],
                r["b"],
                f"{r['ap_difference']:+.5f}",
                str([round(v, 5) for v in r["ap_ci95"]]),
                f"{r['auroc_difference']:+.5f}",
                str([round(v, 5) for v in r["auroc_ci95"]]),
            ]
            for r in tc
        ],
    )
    single = raw[raw.model == "selection/best_single"]
    ensemble = raw[raw.model == "selection/best_ensemble"]
    base = raw[raw.model == "multiview/v1_equivalent"]
    mv = raw[raw.model == "multiview/selected"]
    xgb = raw[raw.model == "xgboost/selected"]

    def effect(a, b):
        aa = ablations[ablations.model == "multiview/" + a]
        bb = ablations[ablations.model == "multiview/" + b]
        return f"ΔAP {aa.ap.mean() - bb.ap.mean():+.5f}; ΔAUROC {aa.auroc.mean() - bb.auroc.mean():+.5f}"

    hp_table = markdown_table(
        ["Outer", "Chosen five-view candidate", "Configuration", "Median inner best epoch"],
        [
            [r["outer"], r["configuration"]["name"], str(r["configuration"]), r["refit_epochs"]]
            for r in hp
            if r["family"] == "multiview"
        ],
    )
    document(
        "OPTIMIZATION_RESULTS.md",
        f"""# V2 predictive optimization — nested internal evaluation

Only V1 training rows. Three outer folds, three inner folds, separate calibration roles.
Metrics below are mean ± SD across nine dependent fold×seed evaluations for selected models.
They are **not** the V1 opened test estimates. Historical V1 XGBoost AP .7696/AUROC .7887
cannot be subtracted from these CV numbers as a paired improvement. The matching refitted
V1-equivalent five-view gives AP {base.ap.mean():.5f}, AUROC {base.auroc.mean():.5f}; selected
V2 five-view gives {mv.ap.mean():.5f}/{mv.auroc.mean():.5f}.

{selected_table}

{comparison_table}

Intervals resample feature groups within fixed outer folds and average paired seeds and
fold metrics. They condition on fitted models and fixed folds, do not quantify full nested
search uncertainty, and are not multiple-comparison adjusted. Neither folds nor seeds are
independent external populations. No outer result revises inner model selection.

## Prespecified training/capacity effects, seed101 in all three outer folds

- 120 versus60 maximum epochs: {effect("epochs120", "v1_equivalent")}.
- 200 versus60 maximum epochs: {effect("epochs200", "v1_equivalent")}.
- AP versus BCE stopping (same120/patience16): {effect("ap_stop", "patience16")}.
- AUROC versus BCE stopping: {effect("auroc_stop", "patience16")}.
- Batch256 versus1024 with AP stopping: {effect("batch256", "ap_stop")}.
- Embedding32 versus16: {effect("width32", "batch256")}.
- Embedding64 versus16: {effect("width64", "batch256")}.

These are prespecified paired internal ablations, not independent clinical causal effects.
Regularized optimizer/schedule combinations are bundled candidates, so their individual
components are not causally identified. Gradient clipping was not added to the V1-equivalent
control. Scratch epochzero is diagnostic only; refit epochs derive from inner folds.

{hp_table}

{ablation_table}

Exact best epochs, costs, generalization differences between inner and outer AP, parameters
and all curves are in trial records and `selected_hyperparameters.json`. `parameter_count`
is N/A for trees. `inference_seconds` measures an actual end-to-end CPU outer prediction,
including checkpoint load/reconstruction and preprocessing; it is not a deployed latency
benchmark. Three separate development-only diagnostic refits record genuine train AP/AUROC
and validation AP/AUROC curves without rewriting historical logs or changing selection.
Their weight differences from matching original inner checkpoints are recorded explicitly. Larger search budgets
also increase selection opportunity; no extrapolation to an artificial .90 target is made.
""",
    )
    document(
        "TABULAR_MODEL_COMPARISON.md",
        f"""# Modern tabular model comparison

{selected_table}

Tuned XGBoost mean outer AP {xgb.ap.mean():.5f}, AUROC {xgb.auroc.mean():.5f}.
FT-Transformer uses official `rtdl-revisiting-models`0.0.2, native numeric/categorical tokens,
three registered candidates and validation-AP stopping. TabNet uses established
`pytorch-tabnet`4.1.0, two candidates and sklearn AP. Both ran on CPU after the recorded MPS
smoke failures; five-view/flat ran on reliable MPS. XGBoost is isolated from Torch/OpenMP.
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

{folds_table}
""",
    )
    feature = ablations[
        ablations.model.str.contains(
            "batch256|derived_features|missing_indicators|combined_features|robust_combined"
        )
    ]
    document(
        "FEATURE_ENGINEERING_ABLATIONS.md",
        f"""# Deterministic feature ablations

{table(feature)}

Derived block adds pulse pressure, approximate MAP, age×SBP, age×BMI and BMI². Missingness
block records invalid/missing inputs. The robust control changes fold-fitted scaling only.
All are functions of existing measurements, not independent clinical information. No
external clinical cutoff is invented. Same seed101, base architecture, AP stopping and
batch256 across these comparisons. Added input dimensions also change parameter count.

Derived versus base: {effect("derived_features", "batch256")}.
Missing indicators versus base: {effect("missing_indicators", "batch256")}.
Combined versus base: {effect("combined_features", "batch256")}.
Robust versus standard combined: {effect("robust_combined", "combined_features")}.

Outer-fold consistency must be assessed from `optimization_metrics.csv`, not just pooled
point estimates. These ablations were not used to revise any outer fold's candidate choice.
""",
    )
    calibration = optimization[
        optimization.model.str.endswith("/selected") & (optimization.operating == "fixed_050")
    ]
    ensemble_rows = raw[raw.model.str.startswith(("ensemble/", "selection/"))]
    threshold = optimization[
        (optimization.model == "multiview/selected") & (optimization.calibration == "none")
    ]
    document(
        "CALIBRATION_AND_ENSEMBLES.md",
        f"""# Calibration, thresholds and ensembles

Calibration mappings fit a dedicated outer-training calibration role, never reported outer
rows. Within-fold strictly increasing Platt/temperature preserve ranks; isotonic can tie
scores. Calibration intercept/slope are outer diagnostic regressions only, never refitted
predictions on those rows. Pooled cross-fold rank changes can also reflect different mappings.

{table(calibration, ["model", "calibration"], ("ap", "auroc", "brier", "log_loss", "ece_10"))}

Thresholds use calibration labels, not outer outcomes. .50, balanced accuracy, F1,
precision≥.75/max-recall and recall≥.80/max-precision were registered. They do not change AP or
AUROC, and none is recommended for deployment.

{table(threshold, "operating", ("recall", "precision", "specificity", "balanced_accuracy", "f1"))}

Inner-OOF predictions fit weights for registered combinations; equal/convex candidate choice
also occurs within inner OOF. Best single and best ensemble are per-outer **selection
procedures**, not retrospectively chosen outer winners. Mean AP {single.ap.mean():.5f} versus
{ensemble.ap.mean():.5f}; paired difference/interval is in optimization comparisons.

{table(ensemble_rows)}

A positive point difference alone is not proof of an ensemble benefit. Learned mappings and
weights remain private; all aggregate outer metrics and reproducible fitting code are public.
""",
    )
    mask_table = markdown_table(
        [
            "Fit positives",
            "All SHARE/head-only",
            "Invalid geometry",
            "Module agreement",
            "Random=logical",
        ],
        [
            [
                r["positive_budget"],
                f"{r['head_only_fraction']:.3f}",
                f"{r['invalid_geometry_fraction']:.3f}",
                f"{r['module_agreement']:.3f}",
                f"{r['random_equals_logical_fraction']:.3f}",
            ]
            for r in stability
        ],
    )
    document(
        "SELECTIVE_TRANSFER_RESULTS.md",
        f"""# Internal age-subgroup selective adaptation — actual locked results

Source age<55→target age≥55, only V1 training records. Target evaluation4,383 rows,
2,660 positives (AP baseline {2660 / 4383:.5f}); this is internal subgroup adaptation, not an
independent cohort. All seven module decisions, MMD permutation evidence, discriminability,
R/U bounds and frozen checks were recorded before target evaluation. No MMD training loss.

{mask_table}

{transfer_comparison}

{transfer_table}

At budgetP, primary selective fitting usesP positives/P negatives; separate selector calibration
uses anotherP/P; stopping uses2,191 labels including1,330 positives. Thus the small adaptation
budget does **not** imply only25 positive labels were available to the overall workflow.
Head/full do not consume selector labels; the secondary full-cost-matched control combines
fitting+selector rows and uses the same total target fitting/selection/stopping label budget.
Evaluation labels are additional and fixed; no method fits on them.

Actual module frequencies are in `selector_stability.json` and `selector_module_evidence.csv`.
Invalid geometry is abstention, not evidence for SHARE. Larger support may narrow bounds
without fixing weak source geometry. ADAPT frequency is not a predictive outcome. Random
masks include all closest-budget possibilities and may equal/collapse to logical; equality
is explicitly reported rather than advertised as independent random diversity.

Main adaptation eligible checkpointzero protects the untouched source model. Separate
BCE/zero and AP/nozero ablations isolate stopping behavior. Source initialization and target
samples are paired. `best_epoch=0` records no retained parameter update, even though training
was attempted. The extra25%-prevalence condition uses250 positives/750 negatives in each
fit/selector role. Sample-cap controls compare geometry128 and MMD512 at25/1000 positives.
Full geometry is already primary; a larger MMD cap does not add geometry labels.

Bootstrap differences condition on shared trained source models, fixed target people/groups
and three overlapping paired draws. They are not independent population replications or
multiplicity-adjusted significance tests. Read all comparisons, including head/full and
label-cost-matched full, before attributing any effect to selective adaptation.
""",
    )
    im = optimization[
        optimization.model.str.startswith("imbalance/")
        & (optimization.operating.isin(["fixed_050", "balanced_accuracy"]))
    ]
    document(
        "CLASS_IMBALANCE_RESULTS.md",
        f"""# V2 class-imbalance follow-up

Same selected multiview configuration per outer fold, fixed feasible total training size,
50% versus7.42% available positives, three paired samples, standard/weighted/oversampled
training. Calibration and evaluation retain natural approximately50% prevalence. Epoch
counts come from natural-prevalence inner development; no rare-condition tuning used outer
labels. AP baselines are each outer fold's recorded prevalence, not7.42%.

{table(im, ["model", "calibration", "operating"], ("ap", "auroc", "brier", "recall", "balanced_accuracy"))}

Ranking deterioration, prior/calibration mismatch and fixed-threshold failure are distinct.
Thresholding cannot alter AP/AUROC/Brier. Calibration cannot be assumed to improve ranking.
No artificially rare evaluation sample is presented as a real low-prevalence cohort, and
external rare-prevalence validation remains pending.
""",
    )
    neural = oc[0]
    primary_transfer = [r for r in tc if r["b"] == "full"]
    conclusion = markdown_table(
        ["Fit positives", "Logical−full AP", "95% CI", "Logical−full AUROC"],
        [
            [
                r["positive_budget"],
                f"{r['ap_difference']:+.5f}",
                str([round(v, 5) for v in r["ap_ci95"]]),
                f"{r['auroc_difference']:+.5f}",
            ]
            for r in primary_transfer
        ],
    )
    document(
        "SCIENTIFIC_CONCLUSIONS.md",
        f"""# Scientific conclusions — distinguish the two research questions

RQ-A predictive optimization: matched V2 selected five-view versus refitted V1-equivalent
mean outer AP difference {neural["ap_difference"]:+.5f},95% conditional group-bootstrap interval
{neural["ap_ci95"]}. Tuned XGBoost mean outer AP/AUROC {xgb.ap.mean():.5f}/{xgb.auroc.mean():.5f};
selected five-view {mv.ap.mean():.5f}/{mv.auroc.mean():.5f}. These are nested internal estimates;
original V1 test AP .7656 versus .7696 is historical context, not an independent V2 comparison.
The component effects and their intervals are in optimization reports. No required .90
outcome, no unsupported substantial-improvement claim, and no post-outer retuning.

RQ-B full selective adaptation is independently implemented and verified with valid/invalid
synthetic geometry, permutations, random budgets, frozen integrity and epochzero tests.
It was executed on real recorded age subgroups with disjoint label roles. Main evidence:

{conclusion}

More positive support is not itself proof that a selector becomes useful. Evaluate whether
paired intervals support advantage over full, head-only and matched random, whether weak
source geometry still forces abstention, and whether full-cost-matched training is competitive.
A positive conditional point difference in one budget is exploratory evidence only.

No genuine cross-cohort experiment executed: **EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED**.
TabPFN also NOT EXECUTED because licensed access is unverified. FT-Transformer/TabNet are
actual completed fits, not substitute algorithms. No private Phase1 participants/models used.

Both studies share Kaggle development participants and were motivated by V1. Fold/draw
reuse, imperfect person linkage, uncertain diagnosis/provenance/licence, label-prior shifts,
unequal finite tuning budgets and many comparisons limit inference. The source-age range is
restricted, and target stopping labels are numerous. None predicts future heart attack/stroke
or establishes clinical utility/deployment validity. Next scientific step: independent,
properly permitted data with an explicit endpoint and a locked confirmatory evaluation.
""",
    )
    trials = [read_json(p) for p in TRIALS.glob("*.json")]
    adapt = [read_json(p) for p in (AGG / "transfer_trials").glob("*.json")]
    diagnostic_refits = read_json(AGG / "learning_diagnostics.json")
    runtime = {
        "successful_predictor_fits": len(trials) + len(adapt) + len(diagnostic_refits),
        "diagnostic_refits": len(diagnostic_refits),
        "standard_trial_fits": len(trials),
        "adaptation_fits": len(adapt),
        "fit_seconds": sum(
            r["training"]["fit_seconds"] for r in trials + adapt + diagnostic_refits
        ),
        "cv_inner_fits": sum(r["specification"]["run_id"].startswith("inner_") for r in trials),
        "v1_fits_unchanged": 690,
        "calibration_fits_reported_separately": True,
        "hardware": "Apple M5 Pro48GiB; MPS flat/five-view/adaptation; CPU FT/TabNet/native trees",
        "failures": read_json(AGG / "device_probe.json")["families"],
        "external_status": "EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED",
        "tabpfn_status": "NOT_EXECUTED",
    }
    write_json(AGG / "runtime.json", runtime)
    document(
        "RESEARCH_OVERVIEW.md",
        f"""# CVD-70K Research V2

Two separate questions: improve internal ranking, and test seven-module selective adaptation
as labeled target support increases. All comparative records are real trained-model outputs.
V1 commit88c8020072fa3bf48d627bd3c2216d0b8e683181 remains intact. Branch
`research/v2-optimization-transfer`; main is not rewritten or automatically merged.

{runtime["successful_predictor_fits"]} successful V2 predictor fits, including operational
smokes; {runtime["cv_inner_fits"]} registered inner fits. Successful summed fit time
{runtime["fit_seconds"]:.1f}s. Original690 V1 fits are separate. Counts do not turn repeated
seeds/folds/target draws into independent validation. Calibration mappings are separate fits.

{selected_table}

## Evidence map

- [Historical training audit](V1_TRAINING_DIAGNOSTIC.md)
- [Optimization protocol](OPTIMIZATION_PROTOCOL.md) and [actual results](OPTIMIZATION_RESULTS.md)
- [Modern tabular models](TABULAR_MODEL_COMPARISON.md)
- [Feature ablations](FEATURE_ENGINEERING_ABLATIONS.md)
- [Calibration/ensembles](CALIBRATION_AND_ENSEMBLES.md)
- [Imbalance](CLASS_IMBALANCE_RESULTS.md)
- [Transfer feasibility](TRANSFER_FEASIBILITY.md), [protocol](INTERNAL_TRANSFER_PROTOCOL.md),
  [actual selective results](SELECTIVE_TRANSFER_RESULTS.md)
- [External status](EXTERNAL_VALIDATION_STATUS.md)
- [Scientific conclusions](SCIENTIFIC_CONCLUSIONS.md)
- [Reproduction](REPRODUCIBILITY.md)

Data provenance/licence remains limited: official Kaggle metadata again saysUnknown. No raw
or per-person data/predictions, fitted models, learned preprocessing or credentials are public.
Only original code, invented fixtures, aggregate research outputs and figures are released.
V1 validation/test rows are never used for V2 model development or evaluation.
""",
    )
    plots(optimization, transfer, evidence, cfg)
    additional_plots(optimization, transfer)
    tabnet_stability()
    diagnostic_ap_plot()
    print("V2 reports and figures generated", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    generate(args.skip_bootstrap)
