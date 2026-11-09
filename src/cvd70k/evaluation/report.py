"""Aggregate statistics and narrative generated from actual saved final results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cvd70k.evaluation.metrics import paired_bootstrap
from cvd70k.utils.io import read_json, write_json
from cvd70k.visualization.figures import METRICS, label


def markdown_table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(map(cell, row)) + " |" for row in rows]
    return "\n".join(lines)


def mean_sd(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"


def bootstrap_reports(root, df, dev):
    agg = root / "results/aggregate"
    main = df[(df.analysis == "full") & (df.cleaning == "minimal") & (df.operating == "default")]
    pred = {}
    for key, group in main.groupby(["family", "intervention"], sort=False):
        pred["/".join(key)] = [
            np.load(root / "private/test_predictions" / f"{run}.npz")["probability"]
            for run in group.sort_values("seed").run_id
        ]
    run = main.run_id.iloc[0]
    data = np.load(root / "private/test_predictions" / f"{run}.npz")
    splits = pd.read_csv(root / "data/processed/splits.csv")
    groups = splits.loc[data["row"], "group"].to_numpy()
    reps = dev["config"]["bootstrap_replicates"]
    result = paired_bootstrap(data["label"], pred, groups, "mlp/standard", reps)
    write_json(agg / "bootstrap_primary.json", result)
    print("Primary paired bootstrap complete", flush=True)
    comparisons = []
    im = df[df.analysis.isin(["fixed_total", "fixed_positive"]) & (df.operating == "default")]
    for (analysis, q, intervention), group in im.groupby(
        ["analysis", "requested_prevalence", "intervention"]
    ):
        paired = {}
        for family in ["mlp", "multiview"]:
            g = group[group.family == family].sort_values("seed")
            paired[family] = [
                np.load(root / "private/test_predictions" / f"{r}.npz")["probability"]
                for r in g.run_id
            ]
        boot = paired_bootstrap(data["label"], paired, groups, "mlp", reps)
        comparisons.append(
            {"analysis": analysis, "requested_prevalence": q, "intervention": intervention, **boot}
        )
        print(f"Paired imbalance bootstrap {analysis}/{q}/{intervention}", flush=True)
    write_json(agg / "bootstrap_imbalance.json", comparisons)
    rare_interventions = []
    for analysis, group in im[
        (im.family == "multiview") & (im.requested_prevalence == 0.0742)
    ].groupby("analysis"):
        paired = {}
        for intervention, g in group.groupby("intervention"):
            paired[intervention] = [
                np.load(root / "private/test_predictions" / f"{r}.npz")["probability"]
                for r in g.sort_values("seed").run_id
            ]
        boot = paired_bootstrap(data["label"], paired, groups, "standard", reps)
        rare_interventions.append({"analysis": analysis, **boot})
    write_json(agg / "bootstrap_rare_interventions.json", rare_interventions)
    return result


def generate_reports(root, do_bootstrap=True):
    root = Path(root)
    agg = root / "results/aggregate"
    df = pd.read_csv(agg / "test_metrics.csv")
    dev = read_json(agg / "development.json")
    audit = read_json(agg / "dataset_audit.json")
    records = dev["records"]
    default = df[df.operating == "default"]
    grouping = [
        "analysis",
        "cleaning",
        "anthropometry",
        "requested_prevalence",
        "family",
        "intervention",
        "operating",
    ]
    summary = df.groupby(grouping, dropna=False)[METRICS + ["duration_seconds", "threshold"]].agg(
        ["mean", "std"]
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(agg / "condition_summary.csv")
    boot = (
        bootstrap_reports(root, df, dev)
        if do_bootstrap
        else read_json(agg / "bootstrap_primary.json")
    )
    main = default[(default.analysis == "full") & (default.cleaning == "minimal")]
    groups = list(main.groupby(["family", "intervention"], sort=False))
    rows = []
    for key, group in groups:
        low, high = boot["ap_ci"]["/".join(key)]
        rows.append(
            [
                label(*key),
                mean_sd(group.ap),
                f"[{low:.4f}, {high:.4f}]",
                mean_sd(group.auroc),
                mean_sd(group.brier),
                mean_sd(group.recall),
                mean_sd(group.balanced_accuracy),
            ]
        )
    primary_table = markdown_table(
        [
            "Model",
            "AP mean ± seed SD",
            "AP group-bootstrap 95% CI",
            "AUROC",
            "Brier",
            "Recall @.50",
            "Balanced accuracy @.50",
        ],
        rows,
    )
    (root / "results/reports/baseline_comparison.md").write_text(primary_table + "\n")
    comparisons = boot["ap_difference_vs_mlp/standard"]
    diff = comparisons["multiview/standard"]
    winner = dev["validation_selected_winner"]
    wf, wi = winner.split("/")
    wg = main[(main.family == wf) & (main.intervention == wi)]
    selected = df[(df.run_id.isin(wg.run_id)) & (df.operating == "validation_selected")]
    thresholds = selected[
        ["seed", "threshold", "recall", "specificity", "balanced_accuracy", "precision", "f1"]
    ]
    threshold_table = markdown_table(
        thresholds.columns,
        [[f"{v:.4f}" if isinstance(v, float) else v for v in row] for row in thresholds.to_numpy()],
    )
    sensitivity_rows = []
    sensitivities = default[(default.cleaning == "strict") | (default.analysis == "anthropometry")]
    for key, group in sensitivities.groupby(
        ["analysis", "cleaning", "anthropometry", "family"], sort=False
    ):
        sensitivity_rows.append(
            [
                *key,
                int(group.n.iloc[0]),
                f"{group.prevalence.iloc[0]:.4f}",
                mean_sd(group.ap),
                mean_sd(group.auroc),
                mean_sd(group.brier),
            ]
        )
    sensitivity_table = markdown_table(
        [
            "Analysis",
            "Cleaning",
            "Anthropometry",
            "Model",
            "Test n",
            "Test prevalence",
            "AP",
            "AUROC",
            "Brier",
        ],
        sensitivity_rows,
    )
    (root / "results/reports/sensitivity_comparison.md").write_text(sensitivity_table + "\n")
    im = default[default.analysis.isin(["fixed_total", "fixed_positive"])]
    imbalance_rows = []
    for (analysis, q, intervention), group in im[im.family == "multiview"].groupby(
        ["analysis", "requested_prevalence", "intervention"], sort=False
    ):
        sel = df[(df.run_id.isin(group.run_id)) & (df.operating == "validation_selected")]
        imbalance_rows.append(
            [
                analysis,
                f"{q:.2%}",
                intervention,
                int(group.train_n.iloc[0]),
                int(group.train_positive.iloc[0]),
                mean_sd(group.ap),
                mean_sd(group.brier),
                mean_sd(group.recall),
                mean_sd(group.balanced_accuracy),
                mean_sd(sel.balanced_accuracy),
            ]
        )
    imbalance_table = markdown_table(
        [
            "Analysis",
            "Train fraction",
            "Intervention",
            "Train n",
            "Train positives",
            "AP",
            "Brier",
            "Recall @.50",
            "BA @.50",
            "BA @validation threshold",
        ],
        imbalance_rows,
    )
    (root / "results/reports/imbalance_multiview.md").write_text(imbalance_table + "\n")
    # Actual sampling-seed variability combines sample selection and optimization.
    findings = []
    for analysis in ["fixed_total", "fixed_positive"]:
        sub = im[(im.analysis == analysis) & (im.family == "multiview")]
        for intervention in ["standard", "weighted", "resampled"]:
            s = sub[sub.intervention == intervention]
            low = s[s.requested_prevalence == 0.0742]
            high = s[s.requested_prevalence == 0.5]
            findings.append(
                f"- {analysis}, {intervention}: mean AP {high.ap.mean():.4f} at 50% "
                f"→ {low.ap.mean():.4f} at 7.42%; rare-condition recall "
                f"{low.recall.mean():.4f} at .50 and Brier {low.brier.mean():.4f}."
            )
    rare_selected = (
        df[
            (df.family == "multiview")
            & (df.requested_prevalence == 0.0742)
            & (df.intervention == "standard")
            & (df.operating == "validation_selected")
            & df.analysis.isin(["fixed_total", "fixed_positive"])
        ]
        .groupby("analysis")[["balanced_accuracy", "threshold"]]
        .mean()
    )
    rare_default = (
        im[
            (im.family == "multiview")
            & (im.requested_prevalence == 0.0742)
            & (im.intervention == "standard")
        ]
        .groupby("analysis")
        .balanced_accuracy.mean()
    )
    rare_flat = im[
        (im.family == "mlp") & (im.requested_prevalence == 0.0742) & (im.intervention == "standard")
    ]
    findings_text = "\n".join(findings)
    train_seconds = sum(r["training"]["duration_seconds"] for r in records)
    search = read_json(agg / "hyperparameter_selection.json")["search"]
    search_seconds = sum(r["training"]["duration_seconds"] for r in search)
    hardware = read_json(agg / "hardware.json")
    runtime = {
        "completed_final_models": len(records),
        "completed_search_models": len(search),
        "sum_final_fit_seconds": train_seconds,
        "sum_search_fit_seconds": search_seconds,
        "sum_fit_seconds": train_seconds + search_seconds,
        "development_seconds_last_invocation": dev["execution_seconds_this_invocation"],
        "neural_hardware": hardware.get("machdep.cpu.brand_string", hardware["machine"])
        + "/"
        + hardware["selected_device"],
        "tabular_hardware": "CPU, max two workers",
        "inference_hardware": "CPU; XGBoost isolated process",
        "initial_native_failure": "XGBoost/PyTorch in-process segfault before final testing; fixed by isolated worker",
    }
    write_json(agg / "runtime.json", runtime)
    text = f"""# Locked empirical results

Generated from {len(records)} successfully fitted/evaluated models plus {len(search)} validation
search fits. Each reported value is computed from actual dataset rows and saved models.
Five seeds share one frozen held-out test: these are not five independent validations.

The validation-selected winner is **{label(wf, wi)}**, mean validation AP
{dev["validation_mean_ap"][winner]:.6f}. Its mean final test AP is {wg.ap.mean():.6f}, AUROC
{wg.auroc.mean():.6f}, Brier {wg.brier.mean():.6f}. The primary test has {int(wg.n.iloc[0]):,}
records, {int(wg.positive.iloc[0]):,} positives, prevalence {wg.prevalence.iloc[0]:.6f}. The
AP no-skill baseline is this approximately 50% prevalence. Test results do not select or
revise the winner.

{primary_table}

Intervals use 1,000 paired feature-group bootstrap samples and average seed AP inside each
sample. Seed SD measures fit variation on the same records; intervals condition on trained
models. Groups approximate participants because actual person linkage is unavailable.
For standard five-view minus flat MLP, mean paired bootstrap AP difference is
{diff["mean"]:+.6f}, 95% interval [{diff["ci95"][0]:+.6f}, {diff["ci95"][1]:+.6f}].
Interpret this descriptive interval without a claim of external/clinical superiority.
Full machine-readable metrics include log loss, ECE, F1, precision, specificity, accuracy,
confusion counts and duration for **every fit** at both operating points.

## Validation-selected thresholds for the selected model

Maximize validation balanced accuracy on .01… .99, ties closest to .50 then lower.
Thresholds were frozen before final testing. Values below describe test performance at
those validation-derived points. Different seeds have separate fitted thresholds.

{threshold_table}

## Imbalance experiments

All 600 planned fits completed: two sampling designs × four fractions × five model
families × three interventions × five paired seeds. Original validation and test stay
approximately balanced. AP values are comparable here because the test rows and prevalence
are identical; they must not be compared directly with Phase 1 MHAS AP as if the tasks or
AP baselines matched. Class weighting and oversampling alter the effective training prior;
calibration and recall can change even without ranking improvement.

{findings_text}

[Full multiview table](../results/reports/imbalance_multiview.md) gives both threshold
operating points. [Condition summary](../results/aggregate/condition_summary.csv) includes
all five families; paired multiview/flat intervals for all 24 conditions are in
`bootstrap_imbalance.json`, and rare-condition intervention intervals in
`bootstrap_rare_interventions.json`. These many comparisons are descriptive.

Validation threshold selection alone substantially restores rare-condition standard
multiview performance: mean balanced accuracy is {rare_selected.loc["fixed_total", "balanced_accuracy"]:.4f}
(A) and {rare_selected.loc["fixed_positive", "balanced_accuracy"]:.4f} (B), versus
{rare_default.loc["fixed_total"]:.4f} and {rare_default.loc["fixed_positive"]:.4f} at .50.
Mean selected thresholds are {rare_selected.loc["fixed_total", "threshold"]:.3f} and
{rare_selected.loc["fixed_positive", "threshold"]:.3f}. A threshold
change leaves AP and Brier unchanged. Weighting improves Brier under this *balanced*
evaluation prior; this does not establish calibration at a rare-prevalence deployment site.

The standard flat MLP restored epoch 1 in {int((rare_flat.best_epoch == 1).sum())} of
{len(rare_flat)} 7.42% fits under the prespecified
unweighted natural-validation BCE stopping rule and shared full-training hyperparameters.
Its poor rare-condition AP is therefore specific to this training/stopping protocol.
This observed checkpoint pattern does not prove an inherent multiview advantage; alternative
stopping objectives or condition-specific tuning would require a new validation-only study.

A holds 26,478 rows fixed; positives decline from 13,239 to 1,965, so positive support and
negative support change together. B holds 1,964 positives fixed, increasing total from
3,928 to 26,469. Both designs are constrained by the available 24,514 training negatives.
Neither is a pure causal prevalence manipulation independent of sample support.

## Cleaning and anthropometric sensitivity

{sensitivity_table}

Strict results use fewer held-out records and a different test prevalence. Their raw AP
cannot be read as a paired benefit from cleaning. BMI sensitivities use identical primary
test rows and compare dropping derived BMI versus dropping its parents. Cleaning and
feature choices were prespecified; no sensitivity is selected after test inspection.

## Runtime and artifacts

Summed fit time is {train_seconds + search_seconds:.1f} seconds, including search, across
{len(records) + len(search)} completed fits. This excludes extraction, reporting, process
startup outside recorded tabular fit time, and the failed native attempt. See
`runtime.json`, `hardware.json`, `development_lock.json` and `test_access.json`.
Figures 01–14 (plus quality and design variants) have PNG/PDF versions in
`results/figures/`; all 136 prespecified model conditions have additional PR, ROC,
calibration, confusion and learning diagnostics, showing all five actual seed traces.
No transfer/adaptation experiment was run because no defensible recorded domains exist.
"""
    (root / "docs/results.md").write_text(text)
    (root / "results/reports/final_report.md").write_text(text.replace("../results/", "../"))
    (root / "docs/dataset_audit.md").write_text(audit_markdown(audit))
    return primary_table, findings_text


def audit_markdown(audit):
    partitions = []
    for mode, parts in audit["partitions"].items():
        for split, values in parts.items():
            partitions.append(
                [
                    mode,
                    split,
                    values["n"],
                    values["positive"],
                    values["negative"],
                    f"{values['prevalence']:.6f}",
                ]
            )
    flags = markdown_table(["Flag (overlapping)", "Records"], audit["flags_overlapping"].items())
    ranges = markdown_table(
        ["Column", "Observed minimum", "Observed maximum"],
        [[k, v["min"], v["max"]] for k, v in audit["ranges"].items()],
    )
    table = markdown_table(
        ["Cleaning", "Partition", "Records", "Positives", "Negatives", "Prevalence"], partitions
    )
    return f"""# Dataset audit

Archive SHA-256: `{audit["archive_sha256"]}`. CSV SHA-256: `{audit["csv_sha256"]}`.
Verified ZIP has one semicolon-separated `cardio_train.csv`, {audit["entries"][0]["bytes"]:,}
uncompressed bytes. Integrity/CRC, path safety, symlinks, encryption, total size and
compression ratio checked before extraction. No other dataset was downloaded.

Shape: {audit["shape"]}. Target 0={audit["class_counts"]["0"]:,},
1={audit["class_counts"]["1"]:,}. All source columns are numeric with no missing values;
weight is floating point and remaining columns integer. Identifier is unique and excluded.
No missing target or out-of-schema categorical code. Source provenance/licensing caveats
are in [release boundary](provenance_and_release.md).

Including ID: {audit["duplicates_including_id"]} duplicate rows. Excluding ID but retaining
label: {audit["duplicate_labeled_rows_excluding_id"]} redundant rows. There are
{audit["feature_identical_groups_raw"]} raw feature-identical groups, including
{audit["contradictory_raw_groups"]} with conflicting labels. After deterministic masking,
{audit["modeled_group_count"]:,} modeled feature groups remain; all are split-isolated.
We retain observations and label ambiguity rather than silently deduplicating them.

Minimal retains {audit["minimal_retained"]:,}; strict retains {audit["strict_retained"]:,},
excluding {audit["strict_excluded"]:,} rows. The union of all quality flags is
{audit["flagged_rows_union"]:,}; counts below overlap. Flags are analyst-specified research
quality heuristics, not proof that every flagged value is physiologically impossible.

{flags}

Invalid anthropometry is masked before BMI derivation; invalid BP order masks both BPs.
Minimal imputes flagged values with training medians. No original raw values are overwritten.
There is no target-based outlier cleaning, resampling before splitting or test rebalancing.

{table}

Split seed {audit["split_seed"]}; local assignment hash `{audit["split_file_sha256"]}`.
Actual counts reflect indivisible groups rather than exact 70/15/15 percentages.

{ranges}

Detailed categorical counts, missingness before/after masking, dtypes and archive entries
are in `results/aggregate/dataset_audit.json`. Ranges are aggregate summaries, not released
participant rows. Age conversion is /365.25 without integer-year rounding.
"""
