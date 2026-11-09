"""Synthetic release gates and optional real-data/locked-result verification."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd70k.utils.io import read_json, sha256
from cvd70k.v2.statistics import paired_interval


def audit_module():
    spec = importlib.util.spec_from_file_location("v3audit", "scripts/v3/audit_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_artifact_rejections():
    audit = audit_module()
    for path, body in [
        ("results/v3/a.pt", b"x"),
        ("results/v3/a.csv", b"row,prediction\n1,.7"),
        ("results/v2/a.json", b"{}"),
        ("results/v3/a.csv", b"age,cardio,ap_hi\n55,1,120"),
    ]:
        with pytest.raises(ValueError):
            audit.check_artifact(path, body)
    assert audit.check_artifact(
        "results/v3/aggregate/fixture.csv", b"model,ap,auroc\nsynthetic,.5,.5\n"
    )


def test_registered_design_and_budget():
    d = read_json("configs/v3/study.json")
    assert len(d["candidates"]) == 28 and len(d["methods"]) == 30
    assert d["planned_fits"] == 90 + 282 + 3 == 375
    assert d["planned_fits"] <= d["fit_limit"] == 400
    assert set(d["new_policies"]) <= set(d["methods"])
    assert not {"full_cost", "head", "random_matched"} & set(d["new_policies"])


def test_real_membership_isolation():
    if not Path("private/v3/roles.csv").exists():
        pytest.skip("Local membership file excluded from public repository")
    from cvd70k.v2.data import development_data

    df = development_data()
    roles = pd.read_csv("private/v3/roles.csv")
    assert set(roles.row) == set(df.index)
    for _, part in roles.groupby(["track", "outer"]):
        assert part.row.is_unique
        mapping = part.merge(df[["group"]], left_on="row", right_index=True)
        assert mapping.groupby("group").role.nunique().max() == 1
    assert sha256("private/v3/roles.csv") == read_json("results/v3/aggregate/roles.json")["sha256"]


def test_bootstrap_paired_identity_and_metrics():
    y = np.tile([0, 1], 30)
    p = np.linspace(0.01, 0.99, 60)
    groups = np.repeat(np.arange(30), 2)
    r = paired_interval(y, [p], [p], groups, replicates=30)
    assert r["ap_difference"] == 0 and r["ap_ci95"] == [0, 0] and r["auroc_ci95"] == [0, 0]


def test_completed_metrics_and_locks():
    if not Path("results/v3/aggregate/completion.json").exists():
        pytest.skip("Research evaluation not yet completed")
    done = read_json("results/v3/aggregate/completion.json")
    assert done["successful_fits"] == 375
    for track in ["classification", "transfer"]:
        frame = pd.read_csv(f"results/v3/aggregate/{track}_metrics.csv")
        assert frame[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(frame.n).all()
        assert (frame.tp + frame.fn).eq(frame.positive).all()
        assert ((frame.tp + frame.fn) / frame.n).eq(frame.prevalence).all() or np.allclose(
            (frame.tp + frame.fn) / frame.n, frame.prevalence
        )
        lock = read_json(f"results/v3/aggregate/{track}_lock.json")
        for name, h in lock["code"].items():
            assert sha256(name) == h
        if Path("private/v3/roles.csv").exists():
            assert sha256("private/v3/roles.csv") == lock["roles_sha256"]
    labels = pd.read_csv("results/v3/aggregate/label_costs.csv")
    assert labels.reused_label_count.ge(0).all()
    for n in [25, 250, 1000]:
        x = labels[(labels.positive_budget == n) & (labels.method == "full_cost")]
        assert x.total_distinct.eq(4 * n + 200).all()


def test_reports_and_figures_present():
    if not Path("results/v3/aggregate/completion.json").exists():
        pytest.skip("Reports not generated yet")
    for name in [
        "RESEARCH_PROTOCOL",
        "V2_FAILURE_ANALYSIS",
        "PREDICTIVE_OPTIMIZATION",
        "ADVANCED_MULTIVIEW_ARCHITECTURES",
        "SOURCE_REGULARIZATION",
        "SOFT_ADAPTATION",
        "SELECTOR_EVIDENCE_STUDY",
        "LABEL_COST_ACCOUNTING",
        "TRANSFER_COMPARISON",
        "ABLATION_RESULTS",
        "STATISTICAL_ANALYSIS",
        "SCIENTIFIC_CONCLUSIONS",
        "REPRODUCIBILITY",
    ]:
        assert (Path("docs/v3") / (name + ".md")).stat().st_size > 300
    figs = Path("results/v3/figures")
    assert len(list(figs.glob("*.png"))) >= 16 and len(list(figs.glob("*.pdf"))) >= 16
    for p in figs.glob("*"):
        assert p.stat().st_size > 1000


def test_inner_selection_and_oof_ensemble_lock():
    p = Path("results/v3/aggregate/classification_lock.json")
    if not p.exists():
        pytest.skip("Classification not locked yet")
    lock = read_json(p)
    scores = pd.DataFrame(read_json("results/v3/aggregate/classification_inner_scores.json"))
    for outer, choice in lock["selected"].items():
        subset = scores[scores.outer == int(outer)]
        trees = subset[subset.candidate.str.startswith(("xgboost", "catboost", "lightgbm"))]
        expected = (
            trees.sort_values(["ap", "brier", "candidate"], ascending=[False, True, True])
            .iloc[0]
            .candidate
        )
        assert choice["tree"] == expected
        for ensemble in choice["ensembles"].values():
            assert np.isclose(sum(ensemble["weights"]), 1)
            assert all(0 <= w <= 1 for w in ensemble["weights"])
            assert len(ensemble["names"]) == len(ensemble["weights"])


def test_policy_selection_and_budget():
    p = Path("results/v3/aggregate/transfer_choices.json")
    if not p.exists():
        pytest.skip("Transfer not locked yet")
    choices = read_json(p)
    eligible = read_json("configs/v3/study.json")["new_policies"]
    for key, choice in choices.items():
        expected = min(
            eligible, key=lambda m: (-choice["scores"][m]["ap"], choice["scores"][m]["brier"], m)
        )
        assert choice["method"] == expected
        n = int(key.split("_p")[1])
        assert choice["selection_workflow_cost"]["total_distinct"] == 4 * n + 200


def test_plot_table_reproduction(tmp_path):
    spec = importlib.util.spec_from_file_location("v3analysis", "scripts/v3/analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({"model": ["synthetic"], "ap": [0.7]})
    assert module.table(frame) == module.table(frame)
    module.FIG = tmp_path

    def render():
        fig, ax = module.plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("Synthetic fixture")
        module.save(fig, "fixture")
        return (tmp_path / "fixture.png").read_bytes()

    assert render() == render()


def test_fresh_reproduction_protects_existing_paths(tmp_path):
    spec = importlib.util.spec_from_file_location("fresh_repro", "scripts/v3/reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="must not exist"):
        module.prepare_fresh(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="audited V1"):
        module.prepare_fresh(tmp_path, tmp_path / "new")


def test_fresh_worktree_preserves_original_data_and_results(tmp_path):
    import subprocess

    spec = importlib.util.spec_from_file_location("fresh_repro_complete", "scripts/v3/reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    for name, body in {
        "data/raw/cardio_train.csv": "synthetic fixture only",
        "data/processed/splits.csv": "synthetic split only",
        "results/v3/aggregate/old.json": '{"synthetic":true}',
    }.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Synthetic reproduction fixture",
        ],
        cwd=source,
        check=True,
    )
    # Real participant data are ignored in the actual repository; model that here.
    subprocess.run(
        ["git", "rm", "--cached", "-q", "data/raw/cardio_train.csv", "data/processed/splits.csv"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Keep synthetic data untracked",
        ],
        cwd=source,
        check=True,
    )
    fresh = module.prepare_fresh(source, tmp_path / "fresh")
    assert (source / "results/v3/aggregate/old.json").read_text() == '{"synthetic":true}'
    assert list((fresh / "results/v3/aggregate").iterdir()) == []
    assert (fresh / "data/raw").is_symlink()
    assert (fresh / "data/raw/cardio_train.csv").read_text() == "synthetic fixture only"
