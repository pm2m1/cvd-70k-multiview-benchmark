"""Generate V3 aggregate statistics, reports and figures from locked actual fits."""

import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from cvd70k.evaluation.metrics import calibration_bins, metrics
from cvd70k.utils.io import read_json, write_json
from cvd70k.v2.models import MODULES
from cvd70k.v2.selector import calibrated_mask
from cvd70k.v2.statistics import paired_interval
from cvd70k.v3.design import PRIV, PUB, ROOT

FIG = ROOT / "results/v3/figures"
DOC = ROOT / "docs/v3"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", metadata={"Software": "cvd70k V3"})
    fig.savefig(
        FIG / f"{name}.pdf",
        metadata={"Creator": "cvd70k V3", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def table(frame, cols=None):
    if cols is not None:
        frame = frame[cols]
    head = (
        "| "
        + " | ".join(map(str, frame.columns))
        + " |\n| "
        + " | ".join(["---"] * len(frame.columns))
        + " |\n"
    )
    return (
        head
        + "\n".join(
            "| "
            + " | ".join(f"{v:.5f}" if isinstance(v, (float, np.floating)) else str(v) for v in row)
            + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        + "\n"
    )


def report(name, text):
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / f"{name}.md").write_text(text)


def summary(frame, by):
    cols = [
        "ap",
        "auroc",
        "brier",
        "ece_10",
        "balanced_accuracy",
        "recall",
        "specificity",
        "precision",
    ]
    return frame.groupby(by)[cols].mean().reset_index()


def paired(preds, a, b):
    keys = list(preds)
    y = np.concatenate([preds[k]["y"] for k in keys])
    groups = np.concatenate([preds[k]["groups"] for k in keys])
    folds = np.concatenate([np.full(len(preds[k]["y"]), i) for i, k in enumerate(keys)])
    pa = np.concatenate([preds[k]["predictions"][a] for k in keys])
    pb = np.concatenate([preds[k]["predictions"][b] for k in keys])
    return paired_interval(y, [pa], [pb], groups, folds, replicates=1000, seed=38001)


def analysis():
    started = time.perf_counter()
    cls = pd.read_csv(PUB / "classification_metrics.csv")
    trans = pd.read_csv(PUB / "transfer_metrics.csv")
    cp = joblib.load(PRIV / "classification_outer_predictions.joblib")
    tp = joblib.load(PRIV / "transfer_outer_predictions.joblib")
    choices = read_json(PUB / "transfer_choices.json")
    trials = {p.stem: read_json(p) for p in (PUB / "trials").glob("*.json")}
    # Construct family-selection procedures using only previously locked inner scores.
    inner = pd.DataFrame(read_json(PUB / "classification_inner_scores.json"))
    for k, p in cp.items():
        for family in ["xgboost", "lightgbm", "catboost"]:
            row = (
                inner[(inner.outer == k) & inner.candidate.str.startswith(family)]
                .sort_values(["ap", "brier", "candidate"], ascending=[False, True, True])
                .iloc[0]
            )
            pred = p["predictions"][row.candidate]
            p["predictions"]["selected_" + family] = pred
            cls = pd.concat(
                [
                    cls,
                    pd.DataFrame(
                        [
                            dict(
                                outer=k,
                                model="selected_" + family,
                                selected_candidate=row.candidate,
                                **metrics(p["y"], pred),
                            )
                        ]
                    ),
                ],
                ignore_index=True,
            )
    cs = summary(cls, "model")
    ts = summary(trans, ["positive_budget", "model"])
    cs.to_csv(PUB / "classification_summary.csv", index=False)
    ts.to_csv(PUB / "transfer_summary.csv", index=False)
    comparisons = {}
    for a, b in [
        ("selected_neural_seed_average", "v2_reference"),
        ("selected_tree_seed_average", "v2_reference"),
        ("gated", "v2_reference"),
        ("residual", "depth_control"),
        ("attention", "flat_attention_matched"),
        ("pairwise", "flat_pairwise_matched"),
        ("age_spline", "v2_reference"),
        ("nonlinear", "v2_reference"),
        ("quantile", "v2_reference"),
        ("shuffled_quality", "v2_reference"),
        ("focal", "v2_reference"),
        ("rank_01", "v2_reference"),
        ("rank_001", "v2_reference"),
        ("ensemble_tree_view_convex", "selected_tree_seed_average"),
    ]:
        comparisons[a + " minus " + b] = paired(cp, a, b)
    write_json(PUB / "classification_comparisons.json", comparisons)
    tc = {}
    for n in [25, 250, 1000]:
        sub = {k: p for k, p in tp.items() if k.endswith(f"_p{n}")}
        for b in ["full_cost", "head", "logical", "random_matched", "source_only"]:
            tc[f"p{n}:selected_new_policy minus {b}"] = paired(sub, "selected_new_policy", b)
    write_json(PUB / "transfer_comparisons.json", tc)
    # Per-fit source drift and development-only counterfactual diagnostics.
    driftrows = []
    gates = []
    epochs = []
    evidence_rows = []
    benefits = []
    all_ev = read_json(PUB / "transfer_evidence.json")
    manifests = read_json(PUB / "transfer_manifest.json")
    for e in manifests:
        r = trials[e["run"]]
        method = e["method"]
        k = e["outer"]
        n = e["positive_budget"]
        if "drift" in r:
            for m, d in r["drift"].items():
                driftrows.append(
                    dict(
                        outer=k,
                        positive_budget=n,
                        method=method,
                        module=m,
                        **d,
                        adapt=bool(r["mask"][m]),
                        penalty=r["penalties"][m],
                    )
                )
            epochs.append(
                dict(
                    outer=k,
                    positive_budget=n,
                    method=method,
                    best_epoch=r["best_epoch"],
                    epoch_zero=r["best_epoch"] == 0,
                )
            )
        if r.get("gates"):
            for m, g in r["gates"].items():
                gates.append(dict(outer=k, positive_budget=n, method=method, module=m, **g))
    for key, ev in all_ev.items():
        k = int(key.split("_")[0][1:])
        n = int(key.split("_p")[1])
        mask, _ = calibrated_mask(ev)
        score = choices[key]["scores"]
        base = score["head"]["ap"]
        module_methods = [
            "view_background",
            "view_hemo",
            "view_anthro",
            "view_metabolic",
            "view_behaviour",
            "fusion1_only",
            "fusion2_only",
        ]
        for m, method in zip(MODULES, module_methods, strict=True):
            r = ev[m]
            evidence_rows.append(
                dict(
                    outer=k,
                    positive_budget=n,
                    module=m,
                    adapt=mask[m],
                    valid=r["geometry_status"] == "valid",
                    U=r["U"],
                    R=r["R"],
                    p_mmd=r["domain_p"],
                    mmd=r["mmd"],
                    bootstrap_valid_fraction=r.get("bootstrap_valid_fraction"),
                )
            )
            benefits.append(
                dict(
                    outer=k,
                    positive_budget=n,
                    module=m,
                    selection_ap_gain=score[method]["ap"] - base,
                    U=r["U"],
                    R=r["R"],
                    p_mmd=r["domain_p"],
                    geometry_status=r["geometry_status"],
                )
            )
    dr = pd.DataFrame(driftrows)
    ga = pd.DataFrame(gates)
    ep = pd.DataFrame(epochs)
    ev = pd.DataFrame(evidence_rows)
    bf = pd.DataFrame(benefits)
    for name, frame in [
        ("module_drift", dr),
        ("soft_gates", ga),
        ("adaptation_checkpoints", ep),
        ("module_evidence", ev),
        ("module_benefit_development", bf),
    ]:
        frame.to_csv(PUB / f"{name}.csv", index=False)
    correlations = {}
    for col in ["U", "R", "p_mmd"]:
        valid = bf[[col, "selection_ap_gain"]].dropna()
        val = spearmanr(valid[col], valid.selection_ap_gain)
        correlations[col] = dict(
            spearman=float(val.statistic) if np.isfinite(val.statistic) else None,
            n=len(valid),
            scope="Dependent module/fold/budget observations, selection role only; descriptive, no causal or confirmatory p value",
        )
    write_json(PUB / "evidence_benefit_correlations.json", correlations)
    # Paired negative-transfer analysis, not independent fits-as-participants.
    nt = []
    for (n, m), sub in trans.groupby(["positive_budget", "model"]):
        base = (
            trans[(trans.positive_budget == n) & (trans.model == "source_only")]
            .set_index("outer")
            .ap
        )
        delta = sub.set_index("outer").ap - base
        nt.append(
            dict(
                positive_budget=n,
                model=m,
                negative_transfer_frequency=float((delta < 0).mean()),
                mean_ap_difference=float(delta.mean()),
                mean_loss_when_harm=float(delta[delta < 0].mean()) if (delta < 0).any() else 0.0,
                mean_gain_when_help=float(delta[delta > 0].mean()) if (delta > 0).any() else 0.0,
                n_outer_episodes=len(delta),
            )
        )
    nt = pd.DataFrame(nt)
    nt.to_csv(PUB / "negative_transfer.csv", index=False)
    stab = (
        ev.groupby(["positive_budget", "module"])
        .agg(
            adapt_frequency=("adapt", "mean"),
            valid_frequency=("valid", "mean"),
            U_mean=("U", "mean"),
            R_mean=("R", "mean"),
        )
        .reset_index()
    )
    stab.to_csv(PUB / "selector_stability.csv", index=False)
    import itertools

    stability = []
    for n, part in ev.groupby("positive_budget"):
        masks = [
            set(part[(part.outer == k) & part.adapt].module) for k in sorted(part.outer.unique())
        ]
        agreement = [
            len(a & b) / len(a | b) if a | b else 1.0 for a, b in itertools.combinations(masks, 2)
        ]
        stability.append(
            dict(
                positive_budget=int(n),
                mean_adapt_count=float(np.mean([len(x) for x in masks])),
                jaccard=float(np.mean(agreement)),
                invalid_geometry_fraction=float((~part.valid).mean()),
                head_only_mask_fraction=float(np.mean([not x for x in masks])),
            )
        )
    write_json(PUB / "selector_support_stability.json", stability)
    ga.groupby(["positive_budget", "method", "module"]).agg(
        mean_gate=("effective_gate", "mean"),
        std_gate=("effective_gate", "std"),
        mean_cap=("cap", "mean"),
    ).reset_index().to_csv(PUB / "gate_stability.csv", index=False)

    # Label accounting retains per-run role counts, never row identifiers.
    labels = []
    for e in manifests:
        row = {
            k: e[k]
            for k in ["outer", "positive_budget", "method", "run", "selection_workflow_cost"]
        }
        row.update(
            total_distinct=e["label_cost"]["total_distinct"],
            reused_label_count=e["label_cost"]["reused_label_count"],
        )
        for role, cost in e["label_cost"]["roles"].items():
            for name, value in cost.items():
                row[f"{role}_{name}"] = value
        labels.append(row)
    pd.DataFrame(labels).to_csv(PUB / "label_costs.csv", index=False)
    # Data-quality indicators are analyst masking artifacts, not original missing measurements.
    from cvd70k.v2.data import development_data

    df = development_data()
    quality = []
    for col in ["ap_hi", "ap_lo", "height", "weight", "bmi"]:
        for invalid in [False, True]:
            group = df[df[col].isna() == invalid]
            quality.append(
                dict(
                    feature=col,
                    masked=invalid,
                    n=len(group),
                    positive=int(group.cardio.sum()),
                    prevalence=float(group.cardio.mean()) if len(group) else None,
                )
            )
    pd.DataFrame(quality).to_csv(PUB / "quality_indicator_audit.csv", index=False)
    # ---- Figures: all numeric points derive from saved metrics ----
    main = [
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
    show = cs.set_index("model").loc[main]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, col in zip(axes, ["ap", "auroc"], strict=True):
        ax.barh(main, show[col])
        ax.set_xlabel(col.upper())
        ax.set_xlim(0.7, 0.84)
        ax.set_title("Matched V3 internal outer folds")
    save(fig, "01_matched_classification")
    names = [
        "v2_reference",
        "age_spline",
        "nonlinear",
        "quantile",
        "shuffled_quality",
        "focal",
        "rank_01",
        "rank_001",
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    values = cs.set_index("model").loc[names, "ap"]
    ax.bar(names, values)
    ax.set_ylim(values.min() - 0.003, values.max() + 0.003)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Mean outer AP")
    save(fig, "02_feature_loss_ablations")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, row in show.iterrows():
        ax.scatter(row.auroc, row.ap)
        ax.annotate(
            name, (row.auroc, row.ap), fontsize=6, xytext=(3, 2), textcoords="offset points"
        )
    ax.set_xlabel("Mean outer AUROC")
    ax.set_ylabel("Mean outer AP")
    save(fig, "03_ap_auroc")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in main:
        sub = cls[cls.model == name]
        p = sub.parameter_count.mean() if "parameter_count" in sub else np.nan
        if np.isfinite(p):
            ax.scatter(p, sub.ap.mean())
            ax.annotate(name, (p, sub.ap.mean()), fontsize=6)
    ax.set_xlabel("Stored parameters (neural models)")
    ax.set_ylabel("Mean outer AP")
    save(fig, "04_capacity")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    boxes = [
        "V1 training\npool only",
        "Age <55 source\nfit / stop / reference",
        "Age >=55 target\nfit / selector / stop / select",
        "Locked outer\ntarget groups",
    ]
    for i, text in enumerate(boxes):
        ax.text(
            0.12 + i * 0.25,
            0.5,
            text,
            ha="center",
            va="center",
            wrap=True,
            bbox=dict(boxstyle="round", facecolor="#dbeafe"),
            fontsize=8,
            transform=ax.transAxes,
        )
        if i < 3:
            ax.annotate(
                "",
                xy=(0.22 + i * 0.25, 0.5),
                xytext=(0.20 + i * 0.25, 0.5),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->"),
            )
    ax.set_title("Conceptual: INTERNAL AGE-SUBGROUP ADAPTATION")
    save(fig, "05_adaptation_pipeline")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.65,
        r"$\theta_m=\theta_{source,m}+c_m\delta_m,\quad\|\delta_m\|_2\leq r_m$",
        ha="center",
        fontsize=17,
    )
    ax.text(
        0.5,
        0.3,
        "Source anchor + bounded radial drift; no free gate/delta rescaling\nL2-SP: sum of module-weighted squared source distances (head excluded)",
        ha="center",
    )
    ax.set_title("Conceptual: soft adaptation and source preservation")
    save(fig, "06_soft_regularization_diagram")
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
    for number, col in [(7, "ap"), (8, "auroc")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for m in methods:
            sub = ts[ts.model == m]
            ax.plot(sub.positive_budget, sub[col], marker="o", label=m)
        ax.set_xscale("log")
        ax.set_xlabel("Fitting positives (same number of fitting negatives)")
        ax.set_ylabel("Mean outer " + col.upper())
        ax.legend(fontsize=6, ncol=2)
        ax.set_title("Internal target evaluation; natural prevalence, fixed within fold")
        save(fig, f"{number:02d}_label_budget_{col}")
    largest = ts[ts.positive_budget == 1000].set_index("model")
    for number, names in [
        (
            9,
            [
                "logical",
                "full_sp_low",
                "logical_sp_low",
                "soft_uniform",
                "soft_evidence",
                "soft_learned",
                "reg_geometry",
                "reg_mmd",
                "reg_combined",
                "reg_shuffled",
            ],
        ),
        (
            10,
            [
                "source_only",
                "head",
                "full",
                "full_cost",
                "random_matched",
                "selected_new_policy",
                "target_lr_cost",
                "target_xgb_cost",
            ],
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(names, largest.loc[names, "ap"])
        ax.set_xlabel("Mean outer AP, P1000")
        ax.set_xlim(0.75, 0.83)
        save(fig, f"{number:02d}_transfer_comparison")
    d = (
        dr[
            (dr.positive_budget == 1000)
            & dr.method.isin(
                [
                    "full",
                    "logical",
                    "full_sp_low",
                    "full_sp_high",
                    "soft_uniform",
                    "soft_evidence",
                    "reg_combined",
                ]
            )
        ]
        .groupby(["method", "module"])
        .relative.mean()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(d.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(d.columns)), d.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(d)), d.index)
    fig.colorbar(im, ax=ax, label="Relative L2 drift")
    save(fig, "11_module_drift")
    g = ga[ga.positive_budget == 1000].groupby(["method", "module"]).effective_gate.mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(g.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(g.columns)), g.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(g)), g.index)
    fig.colorbar(im, ax=ax, label="Mean effective radial gate")
    save(fig, "12_gate_values")
    nn = nt[(nt.positive_budget == 1000) & nt.model.isin(methods)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(nn.model, nn.negative_transfer_frequency)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of three outer episodes with AP < source-only")
    save(fig, "13_negative_transfer")
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in ["selected_new_policy", "full_cost", "head", "logical", "target_lr_cost"]:
        sub = trans[trans.model == m].groupby("positive_budget")[["target_label_cost", "ap"]].mean()
        ax.plot(sub.target_label_cost, sub.ap, marker="o", label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Distinct accessible target labels (all roles)")
    ax.set_ylabel("Mean outer AP")
    ax.legend()
    save(fig, "14_total_label_efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    calibration = []
    for ax, predsets, names, title in [
        (
            axes[0],
            cp,
            ["v2_reference", "selected_neural_seed_average", "selected_tree_seed_average"],
            "Classification",
        ),
        (
            axes[1],
            {k: p for k, p in tp.items() if k.endswith("_p1000")},
            ["source_only", "full_cost", "logical", "selected_new_policy"],
            "Target P1000",
        ),
    ]:
        for name in names:
            y = np.concatenate([p["y"] for p in predsets.values()])
            p = np.concatenate([p["predictions"][name] for p in predsets.values()])
            bins = calibration_bins(y, p)
            b = [v for v in bins if v["n"]]
            ax.plot(
                [v["mean_probability"] for v in b],
                [v["positive_fraction"] for v in b],
                marker="o",
                label=name,
            )
            calibration.extend(dict(population=title, model=name, **v) for v in bins)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction", title=title)
        ax.legend(fontsize=7)
    save(fig, "15_calibration")
    pd.DataFrame(calibration).to_csv(PUB / "calibration_bins.csv", index=False)
    primary = {k: v for k, v in tc.items() if k.startswith("p1000")}
    names = list(primary)
    fig, ax = plt.subplots(figsize=(9, 4))
    points = np.array([primary[n]["ap_difference"] for n in names])
    lo = np.array([primary[n]["ap_ci95"][0] for n in names])
    hi = np.array([primary[n]["ap_ci95"][1] for n in names])
    ax.errorbar(
        points,
        range(len(names)),
        xerr=np.maximum(0, np.stack([points - lo, hi - points])),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--")
    ax.set_yticks(range(len(names)), [n.split(" minus ")[1] for n in names])
    ax.set_xlabel("Selected new policy minus comparator AP (95% conditional group CI)")
    save(fig, "16_primary_paired_differences")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["v2_reference", "gated", "residual", "attention", "rank_01"]:
        curve = trials[f"class_o0_{name}_s101"]["curve"]
        axes[0].plot([v["epoch"] for v in curve], [v["validation_ap"] for v in curve], label=name)
        axes[1].plot([v["epoch"] for v in curve], [v["validation_bce"] for v in curve], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Stopping AP")
    axes[1].set(xlabel="Epoch", ylabel="Stopping BCE")
    axes[0].legend(fontsize=7)
    save(fig, "17_learning_curves")

    small = []
    for k in range(3):
        winner = choices[f"o{k}_p25"]["method"]
        for source_name, label in [
            ("smallstop_head", "head"),
            ("smallstop_full_cost", "full_cost"),
            ("smallstop_" + winner, "selected_new_policy"),
        ]:
            row = (
                trans[
                    (trans.outer == k)
                    & (trans.positive_budget == 25)
                    & (trans.model == source_name)
                ]
                .iloc[0]
                .to_dict()
            )
            row["comparison_method"] = label
            small.append(row)
    small = pd.DataFrame(small)
    small.to_csv(PUB / "small_stopping_sensitivity.csv", index=False)
    report(
        "SMALL_STOPPING_SENSITIVITY",
        "# Smaller stopping-set sensitivity\n\n"
        + table(
            summary(small, "comparison_method"),
            ["comparison_method", "ap", "auroc", "brier", "recall"],
        )
        + "\n\nP25 fitting budget, 20 stopping labels (10/class), same source checkpoint and paired draws. The new policy was already chosen using the main100-label stopping design and its disjoint100-label selection role; it was not reselected after this sensitivity. The other80 stopping labels are withheld. This tests checkpoint allocation for a fixed chosen policy, not a fully retuned20-label pipeline. Selected and full-cost procedures access220 target labels in total. Head fitting/stopping uses70; the sensitivity manifest conservatively also charges100 policy-selection labels, recording170 accessible labels for this fixed control. No predictor fit was added after outer evaluation.\n",
    )
    # Reports are generated from records; no invented successful method.
    primary = tc["p1000:selected_new_policy minus full_cost"]
    selected_counts = pd.Series([r["method"] for r in choices.values()]).value_counts().to_dict()
    total = len(trials)
    seconds = sum(r.get("fit_seconds", 0) for r in trials.values())
    failures = len(list((PUB / "failures").glob("*.json")))
    epoch_summary = (
        ep.groupby("method")
        .agg(epoch_zero_fraction=("epoch_zero", "mean"), median_best_epoch=("best_epoch", "median"))
        .reset_index()
    )
    epoch_summary.to_csv(PUB / "checkpoint_summary.csv", index=False)
    selected_summary = ts[
        ts.model.isin(
            [
                "selected_new_policy",
                "source_only",
                "head",
                "full",
                "full_cost",
                "logical",
                "random_matched",
                "full_sp_low",
                "full_sp_high",
                "logical_sp_low",
                "soft_uniform",
                "soft_evidence",
                "soft_learned",
                "reg_combined",
                "target_lr_cost",
                "target_xgb_cost",
            ]
        )
    ]
    common = "\nAll estimates are exploratory internal evidence on original V1 training records. Outer folds, repeated fits and shared source records are dependent. No external clinical validation or future-event prediction.\n"
    report(
        "PREDICTIVE_OPTIMIZATION",
        "# Predictive optimization\n\n"
        + table(
            show.reset_index(), ["model", "ap", "auroc", "brier", "balanced_accuracy", "recall"]
        )
        + common
        + "\nBest individual AP point estimate (post-hoc descriptive): "
        + str(cs.sort_values("ap", ascending=False).iloc[0].to_dict())
        + "\n\nPrimary procedure comparisons:\n```json\n"
        + json.dumps({k: v for k, v in comparisons.items() if k.startswith("selected_")}, indent=2)
        + "\n```\n\nV1/V2 references are configurations refitted on V3 roles, not historical final-test scores. The V2 reference uses its missing-indicator configuration; V2 also selected robust combined features in one historical fold. Search family budgets differ; no universal architecture superiority is inferred.\n",
    )
    report(
        "ADVANCED_MULTIVIEW_ARCHITECTURES",
        "# Advanced multiview architectures\n\n"
        + table(
            cs[
                cs.model.isin(
                    [
                        "v2_reference",
                        "gated",
                        "gated_regularized",
                        "uniform_gate",
                        "random_gate",
                        "residual",
                        "depth_control",
                        "attention",
                        "attention_no_identity",
                        "pairwise",
                        "flat_pairwise_matched",
                        "flat_attention_matched",
                        "focal",
                        "rank_01",
                        "rank_001",
                    ]
                )
            ],
            ["model", "ap", "auroc", "brier"],
        )
        + common
        + "\nGlobal sigmoid view weights are bounded and regularized toward .5; uniform/random controls retain five gate parameters with gradients disabled. Residual and depth controls share all parameters. Attention uses five identity vectors; the no-identity ablation fixes these vectors to zero. Pairwise dot products add deterministic interactions. Flat widths minimize parameter discrepancy over a prespecified grid. Focal and ranking objectives are compared under matched V2 features/architecture. See trial records for exact counts, epochs and runtime. No interpretable view weight is a medical explanation.\n",
    )
    report(
        "SOURCE_REGULARIZATION",
        "# Source preservation with L2-SP\n\n"
        + table(
            ts[
                ts.model.isin(
                    [
                        "full",
                        "logical",
                        "full_sp_low",
                        "full_sp_high",
                        "logical_sp_low",
                        "logical_sp_high",
                        "reg_geometry",
                        "reg_mmd",
                        "reg_combined",
                        "reg_shuffled",
                    ]
                )
            ],
            ["positive_budget", "model", "ap", "auroc", "brier"],
        )
        + common
        + "\nThe penalty is a sum of squared distances to detached source backbone weights, excluding the head. This is distinct from ordinary weight decay. Reference: [Li et al., ICML 2018](https://proceedings.mlr.press/v80/li18a.html). Strengths .01/1 were registered; geometry/MMD preservation penalties are deterministic and contrasted with shuffled assignments. Drift data are actual retained-checkpoint differences, including epoch-zero zeros.\n",
    )
    report(
        "SOFT_ADAPTATION",
        "# Constrained radial soft adaptation\n\n"
        + table(
            ts[ts.model.str.startswith("soft_")],
            ["positive_budget", "model", "ap", "auroc", "brier"],
        )
        + common
        + "\nNo independently trainable gate/delta scaling pair is present. Delta has a fixed norm cap; effective radial gate is cap times norm(delta)/radius. Thus reported gate values are identified by effective drift, not free attribution parameters. The learned-radial cap1 differs from fixed uniform cap.5: this is a constraint ablation, not a claim that gate learning alone caused differences. Evidence/random/shuffled/uniform controls share the parameterization. Frozen zero-cap modules and exact source initialization are verified.\n",
    )
    report(
        "SELECTOR_EVIDENCE_STUDY",
        "# Selector evidence study\n\n"
        + table(stab)
        + common
        + "\nDevelopment-only single-module benefit correlations:\n```json\n"
        + json.dumps(correlations, indent=2)
        + "\n```\n\nThese dependent observations use only 100 selection labels per episode. Correlation cannot establish mechanism or causal adaptation value. No outer outcome was used to learn a policy. Learned meta-selection: NOT EXECUTED because three overlapping age tasks do not provide independent meta-training/evaluation populations. The evidence-guided policies are fixed formulas, not learned meta-selectors. Larger support reaches geometry but MMD remains capped128.\n",
    )
    report(
        "LABEL_COST_ACCOUNTING",
        "# Target-label accounting\n\nThe development-selected new-policy workflow accesses `4P+200` distinct target labels: 300, 1,200 and 4,200 for P25/P250/P1000. Each has P fitting positives plus P negatives, an equally sized disjoint selector draw, 100 stopping labels and 100 policy-selection labels. These are not 25/250/1000-total-label experiments. The full-cost comparator fits the fitting+selector+policy-selection union and uses the same stopping set. Fixed head/full methods use 2P+100; fixed logical uses 4P+100. Source-only uses no target labels. Source fitting/stopping/reference labels are separately shared costs.\n\n`label_costs.csv` records all distinct role counts, reuses and workflow access. Candidate selection is charged even if it picks an evidence-free policy. No label is counted twice in the union. Logistic references are conservatively charged access to stopping labels although they do not use early stopping. Native tree stopping uses its assigned labels. Small-stopping sensitivity uses 20 labels, withholding the other80; total selected-policy access becomes4P+120. Source-label count and natural outer target counts are in roles.json. Compute costs differ and are not equated by label matching.\n"
        + common,
    )
    report(
        "TRANSFER_COMPARISON",
        "# Internal age-subgroup transfer comparison\n\n"
        + table(
            selected_summary,
            ["positive_budget", "model", "ap", "auroc", "brier", "balanced_accuracy", "recall"],
        )
        + common
        + "\nPrimary P1000 conditional paired result:\n```json\n"
        + json.dumps(primary, indent=2)
        + "\n```\n\nPolicies selected only on inner target model-selection labels:\n```json\n"
        + json.dumps({k: v["method"] for k, v in choices.items()}, indent=2)
        + "\n```\n\nNatural target evaluation prevalence differs from balanced training/selection roles; AP baseline is recorded for every fold. No threshold was selected on outer data; .50 is descriptive. Repeated source-only entries reuse the same checkpoint, not independent fits.\n",
    )
    report(
        "ABLATION_RESULTS",
        "# Prespecified ablation results\n\n"
        + table(
            cs[
                cs.model.isin(
                    [
                        "v2_reference",
                        "age_spline",
                        "nonlinear",
                        "quantile",
                        "shuffled_quality",
                        "strict_reference_strict_common",
                        "v1_reference_strict_common",
                        "v2_reference_strict_common",
                        "shuffled_quality_strict_common",
                    ]
                )
            ],
            ["model", "ap", "auroc", "brier"],
        )
        + "\n\n"
        + table(
            ts[
                ts.model.isin(
                    [
                        "view_hemo",
                        "view_metabolic",
                        "view_background",
                        "view_anthro",
                        "view_behaviour",
                        "fusion1_only",
                        "fusion2_only",
                        "fusion_both",
                        "soft_random",
                        "soft_shuffled",
                        "reg_shuffled",
                    ]
                )
            ],
            ["positive_budget", "model", "ap", "auroc"],
        )
        + common
        + "\nQuality indicators arise from analyst masking of implausible measurements. Their association with outcomes may reflect cleaning/acquisition artifacts. Shuffling does not remove all artifacts because imputed values may themselves reveal invalidity. The strict-common subset comparisons share evaluation rows with each other but must not be ranked against full-population AP. Splines and nonlinear interactions add no independent clinical measurement; all learned transformations fit on inner training records only.\n",
    )
    report(
        "STATISTICAL_ANALYSIS",
        "# Statistical analysis\n\nPrimary P1000 selected-policy minus label-cost-matched full:\n```json\n"
        + json.dumps(primary, indent=2)
        + "\n```\n\nAll comparisons are in classification_comparisons.json and transfer_comparisons.json. Paired bootstrap resamples modeled-feature groups within each fixed outer fold, preserving paired predictions. Intervals condition on fitted checkpoints, split assignments and development selections; they do not resample the full search/source training or establish independent population replication. Multiplicity-unadjusted secondary comparisons are exploratory. Three folds/draws share source participants and have only one seed per transfer fold; effects of seed, target draw and fold cannot be separated. Practical reference was +.005 AP; neither favorable point estimates nor increased ADAPT decisions alone establish superiority.\n"
        + common
        + "\nNegative transfer:\n"
        + table(
            nt[
                (nt.positive_budget == 1000)
                & nt.model.isin(
                    [
                        "head",
                        "full",
                        "full_cost",
                        "logical",
                        "selected_new_policy",
                        "soft_evidence",
                        "full_sp_low",
                    ]
                )
            ]
        )
        + "\n\nCheckpoint behavior:\n"
        + table(epoch_summary),
    )
    report(
        "SCIENTIFIC_CONCLUSIONS",
        "# V3 scientific conclusions\n\nPrimary selected-policy minus matched-full AP: "
        + f"{primary['ap_difference']:+.6f}, 95% conditional CI {primary['ap_ci95']}; AUROC {primary['auroc_difference']:+.6f}, CI {primary['auroc_ci95']}."
        + "\n\n"
        + (
            "The primary conditional interval supports a positive internal AP difference; its practical magnitude and dependence still require independent confirmation."
            if primary["ap_ci95"][0] > 0
            else "V3 does not establish a positive AP advantage over label-cost-matched full fine-tuning. A tiny favorable point estimate or isolated secondary win is insufficient."
        )
        + "\n\nDevelopment-selected policies: "
        + json.dumps(selected_counts)
        + ". Their outer results were not used to select the policy. Each selected procedure consumes the entire registered candidate-selection supervision budget.\n\nClassification, feature, architecture and loss effects are in PREDICTIVE_OPTIMIZATION and ADVANCED_MULTIVIEW_ARCHITECTURES with matched V1/V2 configuration refits. Stronger trees, attention, ranking losses and source regularization are hypotheses tested, not presumed improvements. Negative-transfer frequencies and actual module drift distinguish representation movement from useful prediction. Quality flags may encode acquisition/cleaning artifacts; conditional internal gains cannot establish clinical utility.\n\nEXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED. A next study requires a separately permitted cohort with a documented cross-sectional endpoint crosswalk, person linkage/overlap checks, sufficient positives, independent calibration and locked evaluation. No future-event or clinical deployment claim is supported. Learned meta-selection remains infeasible without independent transfer tasks. More seeds/folds and truly label-limited roles are needed to estimate algorithm-selection stability.\n"
        + common,
    )
    report(
        "REPRODUCIBILITY",
        "# Reproduction and execution inventory\n\nFrom repository root:\n```sh\nuv sync --project configs/v3 --frozen\nuv run --project configs/v3 pytest -q\nsh scripts/v3/run.sh\n```\n\nSupply the original privately downloaded ZIP and recreate V1 audited raw file/splits following V1 README before real-data execution. V3 validates the immutable V1 raw/split hashes and restricts eligibility before statistics. Do not download an alternative dataset. Local role memberships are private/v3/roles.csv; source/adapted checkpoints and per-row scores remain private/v3. Historical V1/V2 files are never written by V3. Checkpoint resume verifies specification and SHA-256; evaluation requires source/config/protocol/role/checkpoint locks. Trials and failures have separate records. Successful models are never overwritten to repair failures.\n\nCompleted successful registered predictor fits: "
        + str(total)
        + f". Failed trial attempts retained: {failures}. Summed model-fit runtime: {seconds:.1f} seconds (not wall-clock because tracks overlap). Three probe fits are included; synthetic pytest optimization is not counted as research fits. Environment and dependency versions: execution_manifest.json/environment.json and the separate configs/v3/uv.lock. MPS is validated for multiview/attention; deterministic ranking-loss indexing and FT run CPU. Native tree libraries execute in isolated CPU processes. Corrections are in execution_corrections.json.\n\nPublication contains no raw data, per-row predictions, trained weights or private preprocessing. Kaggle metadata still reports Unknown licence; source-data redistribution rights are not assumed. Public aggregates cannot substitute for the source dataset or trained models. CI runs synthetic tests without private data; local real-data tests additionally verify memberships, hashes and numerical outputs.\n",
    )

    from datetime import datetime, timezone

    registration = read_json(PUB / "execution_manifest.json")
    runtime_path = PUB / "runtime.json"
    if not runtime_path.exists():
        now = datetime.now(timezone.utc)
        write_json(
            runtime_path,
            dict(
                completed_utc=now.isoformat(),
                registered_utc=registration["registered_utc"],
                wall_seconds_since_registration=(
                    now - datetime.fromisoformat(registration["registered_utc"])
                ).total_seconds(),
                summed_fit_seconds=seconds,
                hardware=read_json(PUB / "environment.json"),
            ),
        )
    report(
        "RUNTIME_AND_BUDGET",
        "# Runtime and finite budget\n\n"
        + json.dumps(read_json(runtime_path), indent=2)
        + "\n\nAll375 planned predictor fits completed. Summed fit seconds count optimizer/library training; wall time includes development, role creation, process startup, evaluation and analysis, with concurrent tracks. Three failed trial attempts and the separate representation-return orchestration failure were repaired before outer evaluation; successful checkpoints were retained. No search expansion or additional predictor fits followed evaluation.\n",
    )

    five_names = [
        "v2_reference",
        "age_spline",
        "nonlinear",
        "quantile",
        "gated",
        "gated_regularized",
        "residual",
        "attention",
        "pairwise",
        "focal",
        "rank_01",
        "rank_001",
    ]
    five = cs[cs.model.isin(five_names)].sort_values("ap", ascending=False).iloc[0]
    practical = "\n\n## Measured component findings\n\n"
    practical += f"Highest individual full-population AP point estimate was LightGBM configuration1 ({cs.set_index('model').loc['lightgbm_1', 'ap']:.5f}); this is a descriptive post-evaluation ranking. The independently inner-selected tree seed-average procedure achieved AP/AUROC {cs.set_index('model').loc['selected_tree_seed_average', 'ap']:.5f}/{cs.set_index('model').loc['selected_tree_seed_average', 'auroc']:.5f}. Its matched delta AP versus V2 reference was {comparisons['selected_tree_seed_average minus v2_reference']['ap_difference']:+.5f}, conditional95% interval {comparisons['selected_tree_seed_average minus v2_reference']['ap_ci95']}.\n\n"
    practical += f"Among the five-view candidates, age splines had AP {five['ap']:.5f} when {five['model']} was the highest descriptive point estimate. Age-spline delta AP versus V2 reference was {comparisons['age_spline minus v2_reference']['ap_difference']:+.5f}, CI {comparisons['age_spline minus v2_reference']['ap_ci95']}; an AP advantage is not established. Gated fusion underperformed the reference point estimate, residual fusion underperformed its depth control, and attention essentially tied its parameter-matched flat control. Focal reduced AP; ranking losses did not establish an AP gain (the .01 variant had a small positive point difference). These family comparisons have unequal search multiplicity and limited seeds.\n\n"
    practical += f"At P1000, the development-selected new procedure fell below source-only AP in all three outer episodes (mean loss {nt[(nt.positive_budget == 1000) & (nt.model == 'selected_new_policy')].mean_ap_difference.iloc[0]:+.6f}). Low-strength full L2-SP changed AP by {largest.loc['full_sp_low', 'ap'] - largest.loc['full', 'ap']:+.6f} versus the same-fitting-label full baseline, too small to establish a practical gain. Raw selected-policy Brier {largest.loc['selected_new_policy', 'brier']:.6f} was worse than source-only {largest.loc['source_only', 'brier']:.6f}; balanced target fitting can alter scores under natural target prevalence. No post-evaluation calibration or threshold optimization was performed.\n\n"
    practical += "More positives increased mean logical ADAPT count from1.67 to3.00, while invalid geometry stayed14.3% and mask Jaccard did not improve monotonically (.444/.361/.400). This does not reproduce a claim that more positive support necessarily stabilizes selection. No all-SHARE masks occurred. Evidence-benefit correlations on the small selection sets are descriptive; shift evidence is not a validated predictor of useful module updates. Uniform/evidence/random/shuffled soft controls and penalty controls do not establish that evidence-guided adaptation improves ranking.\n"
    with (DOC / "SCIENTIFIC_CONCLUSIONS.md").open("a") as f:
        f.write(practical)
    with (DOC / "ADVANCED_MULTIVIEW_ARCHITECTURES.md").open("a") as f:
        f.write(
            "\nCPU inference profiling (1000 rows, three repetitions) is in inference_profile.json. Timings include preprocessing and model reconstruction; tree timings include process startup and must not be read as pure kernel latency. Stored and trainable parameter counts differ for radial adaptation, which retains a frozen source anchor. CatBoost search varies depth, learning rate and L2; its generic child field is not used by CatBoost. LightGBM also varies leaf count/min-child samples and XGBoost child weight. Native categorical handling is used for CatBoost; other trees share one-hot availability.\n"
        )

    radialmax = ga[ga.method == "soft_learned"].effective_gate.max()
    with (DOC / "SOFT_ADAPTATION.md").open("a") as f:
        f.write(
            f"\nThe learned-radial retained gates reached at most{radialmax:.5f}, below their cap1. Thus the cap1 projection was not active at retained checkpoints; this variant is close to a reparameterized L2-SP fit, not evidence of an independently learned clinical gate. Fixed caps also scale effective update steps. Evidence caps and learned radial norms must not be conflated. No unconstrained joint sigmoid-gate/delta model was executed.\n"
        )
    with (DOC / "REPRODUCIBILITY.md").open("a") as f:
        f.write(
            "\n## Fresh public checkout versus local resume\n\nPublished trial records are evidence, not sufficient resume caches: weights are intentionally private. To refit from a public checkout with the privately audited raw data/splits available, run `uv run --project configs/v3 python scripts/v3/reproduce.py ~/Desktop/cvd70k-v3-reproduction --run`. The destination must not exist. This creates a detached worktree at the current commit, clears only its newly created V3 generated outputs and links the original local raw/processed directories for reading. The original checkout and all V1/V2 evidence stay untouched. The wrapper performs375 new registered fits; it was not invoked during the completed study to avoid unnecessary duplicate fitting. `sh scripts/v3/run.sh` directly is for an originating workspace with compatible local checkpoints, or for the prepared fresh worktree. Device selection uses validated MPS when available and CPU otherwise; FT/ranking paths remainCPU. Missing private files cannot be represented as completed local fits.\n"
        )
    # Extra concise index makes deliverables discoverable without editing historical README.
    report(
        "README",
        "# V3 research package\n\nStart with [scientific conclusions](SCIENTIFIC_CONCLUSIONS.md), [protocol](RESEARCH_PROTOCOL.md), [predictive results](PREDICTIVE_OPTIMIZATION.md), [transfer comparison](TRANSFER_COMPARISON.md) and [reproduction](REPRODUCIBILITY.md). V2 PR remains separate and unmerged.\n\nFigures are in results/v3/figures; machine-readable aggregates in results/v3/aggregate. V1/V2 evidence remains unchanged.\n",
    )
    write_json(
        PUB / "completion.json",
        dict(
            successful_fits=total,
            classification_fits=sum(n.startswith("class_") for n in trials),
            source_fits=sum(n.startswith("source_") for n in trials),
            transfer_fits=sum(n.startswith("transfer_") for n in trials),
            smallstop_fits=sum(n.startswith("smallstop_") for n in trials),
            smoke_fits=sum(n.startswith("smoke_") for n in trials),
            failed_trial_attempts=failures,
            summed_fit_seconds=seconds,
            analysis_seconds=time.perf_counter() - started,
            primary=primary,
            selected_policy_frequency=selected_counts,
            figures=sorted(p.name for p in FIG.glob("*")),
        ),
    )
    print("ANALYSIS COMPLETE", total, primary, flush=True)


if __name__ == "__main__":
    analysis()
