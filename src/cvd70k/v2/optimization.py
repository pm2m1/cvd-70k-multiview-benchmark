"""Resumable 3×3 nested study; outer predictions only after inner choices lock."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cvd70k.data.sampling import feasible_plans, sample_indices
from cvd70k.evaluation.metrics import metrics
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.data import prepare_folds
from cvd70k.v2.engine import PRIVATE, TRIALS, fit_trial, predict_trial
from cvd70k.v2.postprocess import (
    Calibrator,
    calibration_diagnostic,
    convex_weights,
    operating_points,
)

ROOT = Path.cwd()
AGG = ROOT / "results/v2/aggregate"


def training_device(family):
    return "mps" if family in ["mlp", "multiview"] and torch.backends.mps.is_available() else "cpu"


def inner_id(o, j, f, c):
    return f"inner_o{o}_j{j}_{f}_c{c}_s101"


def refit_id(o, f, c, seed):
    return f"refit_o{o}_{f}_c{c}_s{seed}"


def develop():
    if (AGG / "optimization_outer_access.json").exists():
        raise RuntimeError("Outer evaluation already opened; development frozen")
    torch.set_num_threads(2)
    cfg = read_json(ROOT / "configs/v2/study.json")
    df, partitions = prepare_folds()
    selections = {}
    all_scores = []
    for outer, parts in partitions.items():
        train = df.loc[parts["fit"]]
        chosen = {}
        oof = {}
        candidates_cfg = cfg["candidates"]
        for family in cfg["families"]:
            best_score = -1.0
            best = None
            for c, settings in enumerate(candidates_cfg[family]):
                records = []
                p_oof = np.full(len(train), np.nan)
                for j in range(3):
                    tr = train.iloc[parts["inner"] != j]
                    val = train.iloc[parts["inner"] == j]
                    run = inner_id(outer, j, family, c)
                    record = fit_trial(run, tr, val, settings, 101, training_device(family))
                    p_oof[parts["inner"] == j] = np.load(
                        PRIVATE / "inner_predictions" / f"{run}.npz"
                    )["probability"]
                    records.append(record)
                score = float(np.mean([r["validation"]["ap"] for r in records]))
                epochs = max(1, int(np.median([r["training"]["best_epoch"] for r in records])))
                all_scores.append(
                    {
                        "outer": outer,
                        "family": family,
                        "candidate": c,
                        "name": settings["name"],
                        "inner_mean_ap": score,
                        "inner_mean_auroc": float(
                            np.mean([r["validation"]["auroc"] for r in records])
                        ),
                        "refit_epochs": epochs,
                        "inner_best_epochs": [r["training"]["best_epoch"] for r in records],
                    }
                )
                if score > best_score:
                    best_score = score
                    best = c
                    oof[family] = p_oof
            chosen[family] = best
        # All candidate selection depends only on that outer fold's inner predictions.
        selections[str(outer)] = {
            "selected": chosen,
            "candidate_scores": [s for s in all_scores if s["outer"] == outer],
        }
        best_single = max(
            cfg["families"], key=lambda f: metrics(train.cardio.to_numpy(), oof[f])["ap"]
        )
        choices = [
            ["xgboost", "multiview"],
            ["xgboost", "ft_transformer"],
            ["xgboost", "tabnet"],
            ["mlp", "multiview"],
            ["xgboost", "multiview", "ft_transformer"],
        ]
        ensembles = {}
        best_ensemble = None
        best_ap = -1.0
        for families in choices:
            name = "+".join(families)
            for strategy in ["equal", "convex"]:
                weights = np.ones(len(families)) / len(families)
                if strategy == "convex":
                    _, weights = convex_weights(
                        train.cardio.to_numpy(), {f: oof[f] for f in families}
                    )
                score = metrics(
                    train.cardio.to_numpy(),
                    sum(w * oof[f] for w, f in zip(weights, families, strict=True)),
                )["ap"]
                ensembles[name + "/" + strategy] = {
                    "families": families,
                    "weights": weights,
                    "inner_ap": score,
                }
                if score > best_ap:
                    best_ap = score
                    best_ensemble = name + "/" + strategy
        selections[str(outer)]["best_single"] = best_single
        selections[str(outer)]["best_ensemble"] = best_ensemble
        dest = PRIVATE / f"outer{outer}_ensembles.joblib"
        joblib.dump(ensembles, dest)
        selections[str(outer)]["ensemble_artifact_sha256"] = sha256(dest)
        write_json(AGG / "optimization_selections.json", selections)
    # Refit all prespecified neural ablations, plus three seeds of selected families.
    for outer, parts in partitions.items():
        train = df.loc[parts["fit"]]
        selection = selections[str(outer)]
        for family in cfg["families"]:
            all_c = (
                list(range(len(cfg["candidates"][family])))
                if family in ["mlp", "multiview"]
                else [selection["selected"][family]]
            )
            for c in all_c:
                settings = cfg["candidates"][family][c]
                row = next(
                    s
                    for s in all_scores
                    if s["outer"] == outer and s["family"] == family and s["candidate"] == c
                )
                seeds = cfg["seeds"] if c in [0, selection["selected"][family]] else [101]
                for seed in seeds:
                    fit_trial(
                        refit_id(outer, family, c, seed),
                        train,
                        None,
                        settings,
                        seed,
                        training_device(family),
                        fixed_epochs=row["refit_epochs"],
                    )
        # Registered prevalence follow-up, same natural-calibration/evaluation roles.
        c = selection["selected"]["multiview"]
        settings = cfg["candidates"]["multiview"][c]
        epochs = next(
            s["refit_epochs"]
            for s in all_scores
            if s["outer"] == outer and s["family"] == "multiview" and s["candidate"] == c
        )
        plans = [
            p for p in feasible_plans(train.cardio, [0.5, 0.0742]) if p["analysis"] == "fixed_total"
        ]
        for plan in plans:
            for seed in cfg["seeds"]:
                index = sample_indices(train.cardio, plan["positive"], plan["negative"], seed)
                for intervention in ["standard", "weighted", "resampled"]:
                    run = (
                        f"imbalance_o{outer}_q{plan['requested_prevalence']}_{intervention}_s{seed}"
                    )
                    fit_trial(
                        run,
                        train.iloc[index],
                        None,
                        settings,
                        seed,
                        training_device("multiview"),
                        fixed_epochs=epochs,
                        intervention=intervention,
                    )
    write_json(AGG / "optimization_inner_scores.json", all_scores)
    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "study_sha256": sha256(ROOT / "configs/v2/study.json"),
        "selection_sha256": sha256(AGG / "optimization_selections.json"),
        "protocol_sha256": sha256(ROOT / "docs/v2/OPTIMIZATION_PROTOCOL.md"),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in (ROOT / "src/cvd70k/v2").glob("*.py")
            if p.name
            in [
                "data.py",
                "engine.py",
                "models.py",
                "optimization.py",
                "postprocess.py",
                "tree_worker.py",
                "tabnet_support.py",
            ]
        },
        "checkpoints": {
            p.stem: read_json(p)["checkpoint_sha256"]
            for p in TRIALS.glob("*.json")
            if p.stem.startswith(("inner_", "refit_", "imbalance_"))
        },
    }
    write_json(AGG / "optimization_lock.json", lock)
    print("OPTIMIZATION DEVELOPMENT LOCKED", len(lock["checkpoints"]), flush=True)


def evaluate():
    torch.set_num_threads(2)
    lock = read_json(AGG / "optimization_lock.json")
    for p, h in [
        ("configs/v2/study.json", lock["study_sha256"]),
        ("results/v2/aggregate/optimization_selections.json", lock["selection_sha256"]),
        ("docs/v2/OPTIMIZATION_PROTOCOL.md", lock["protocol_sha256"]),
    ] + list(lock["source_sha256"].items()):
        if sha256(ROOT / p) != h:
            raise ValueError("Stale optimization lock " + p)
    for run, h in lock["checkpoints"].items():
        if sha256(PRIVATE / "models" / f"{run}.joblib") != h:
            raise ValueError("Corrupt model " + run)
    if not (AGG / "optimization_outer_access.json").exists():
        write_json(
            AGG / "optimization_outer_access.json",
            {
                "opened_utc": datetime.now(timezone.utc).isoformat(),
                "lock_sha256": sha256(AGG / "optimization_lock.json"),
            },
        )
    cfg = read_json(ROOT / "configs/v2/study.json")
    selection = read_json(AGG / "optimization_selections.json")
    df, parts = prepare_folds()
    rows = []
    infeasible = []
    pred_dir = PRIVATE / "outer_predictions"
    pred_dir.mkdir(exist_ok=True)

    def emit(name, outer, seed, p, cal_p, cal_y, test, metadata, calibrations=("none",)):
        for kind in calibrations:
            calibrator = Calibrator(kind).fit(cal_y, cal_p)
            cp = calibrator.transform(cal_p)
            pp = calibrator.transform(p)
            thresholds = operating_points(cal_y, cp)
            diag = calibration_diagnostic(test.cardio.to_numpy(), pp)
            out = pred_dir / f"{name.replace(chr(47), chr(95) * 2)}_o{outer}_s{seed}_{kind}.npz"
            np.savez_compressed(out, probability=pp, row=test.index.to_numpy())
            for operating, t in thresholds.items():
                if t is None:
                    infeasible.append(
                        {
                            "model": name,
                            "outer": outer,
                            "seed": seed,
                            "calibration": kind,
                            "operating": operating,
                            "status": "infeasible_on_calibration",
                        }
                    )
                    continue
                rows.append(
                    {
                        "model": name,
                        "outer": outer,
                        "seed": seed,
                        "calibration": kind,
                        "operating": operating,
                        **metadata,
                        **metrics(test.cardio.to_numpy(), pp, t),
                        **diag,
                    }
                )

    for outer, part in parts.items():
        test = df.loc[part["outer_test"]]
        cal = df.loc[part["calibration"]]
        sel = selection[str(outer)]
        predictions = {}
        calibration_predictions = {}
        for family in cfg["families"]:
            selected = sel["selected"][family]
            candidates = list(range(18)) if family in ["mlp", "multiview"] else [selected]
            for c in candidates:
                settings = cfg["candidates"][family][c]
                for seed in cfg["seeds"] if c in [0, selected] else [101]:
                    run = refit_id(outer, family, c, seed)
                    inference_start = time.perf_counter()
                    p = predict_trial(run, test)
                    inference_seconds = time.perf_counter() - inference_start
                    pc = predict_trial(run, cal)
                    info = read_json(TRIALS / f"{run}.json")["training"]
                    name = family + "/" + settings["name"]
                    metadata = {
                        "family": family,
                        "candidate": c,
                        "selected": c == selected,
                        "parameter_count": info["parameter_count"],
                        "fit_seconds": info["fit_seconds"],
                        "best_epoch": info["best_epoch"],
                        "inference_seconds": inference_seconds,
                        "inference_rows_per_second": len(test) / inference_seconds,
                    }
                    emit(name, outer, seed, p, pc, cal.cardio.to_numpy(), test, metadata)
                    if c == selected:
                        predictions[(family, seed)] = p
                        calibration_predictions[(family, seed)] = pc
                        calibrations = ["none", "platt", "isotonic"] + (
                            ["temperature"]
                            if family in ["mlp", "multiview", "ft_transformer"]
                            else []
                        )
                        emit(
                            family + "/selected",
                            outer,
                            seed,
                            p,
                            pc,
                            cal.cardio.to_numpy(),
                            test,
                            metadata,
                            calibrations,
                        )
        for seed in cfg["seeds"]:
            f = sel["best_single"]
            emit(
                "selection/best_single",
                outer,
                seed,
                predictions[(f, seed)],
                calibration_predictions[(f, seed)],
                cal.cardio.to_numpy(),
                test,
                {"family": f, "selected": True},
            )
            ensembles = joblib.load(PRIVATE / f"outer{outer}_ensembles.joblib")
            for key, spec in ensembles.items():
                p = sum(
                    w * predictions[(f, seed)]
                    for w, f in zip(spec["weights"], spec["families"], strict=True)
                )
                pc = sum(
                    w * calibration_predictions[(f, seed)]
                    for w, f in zip(spec["weights"], spec["families"], strict=True)
                )
                emit(
                    "ensemble/" + key,
                    outer,
                    seed,
                    p,
                    pc,
                    cal.cardio.to_numpy(),
                    test,
                    {"family": "ensemble", "selected": key == sel["best_ensemble"]},
                )
                if key == sel["best_ensemble"]:
                    emit(
                        "selection/best_ensemble",
                        outer,
                        seed,
                        p,
                        pc,
                        cal.cardio.to_numpy(),
                        test,
                        {"family": "ensemble", "selected": True},
                    )
        # Imbalance result naming fixed before evaluation.
        for q in [0.5, 0.0742]:
            for seed in cfg["seeds"]:
                for intervention in ["standard", "weighted", "resampled"]:
                    run = f"imbalance_o{outer}_q{q}_{intervention}_s{seed}"
                    p = predict_trial(run, test)
                    pc = predict_trial(run, cal)
                    emit(
                        f"imbalance/{q}/{intervention}",
                        outer,
                        seed,
                        p,
                        pc,
                        cal.cardio.to_numpy(),
                        test,
                        {"family": "multiview", "selected": False},
                        ("none", "platt"),
                    )
    pd.DataFrame(rows).to_csv(AGG / "optimization_metrics.csv", index=False)
    write_json(AGG / "optimization_threshold_infeasible.json", infeasible)
    write_json(
        AGG / "optimization_complete.json",
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "metric_rows": len(rows),
            "outer_evaluation": "V1 training population only; exploratory nested internal CV",
        },
    )
    print("OPTIMIZATION OUTER EVALUATION COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "evaluate"])
    args = parser.parse_args()
    develop() if args.stage == "develop" else evaluate()
