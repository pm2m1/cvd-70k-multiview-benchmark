"""Development-only training followed by a separate locked final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from cvd70k.data.pipeline import Preprocessor, clean, load_csv
from cvd70k.data.sampling import feasible_plans, oversample_indices, sample_indices
from cvd70k.evaluation.metrics import metrics, select_threshold, threshold_grid
from cvd70k.training.engine import build_model, predict_nn, train_nn
from cvd70k.utils.io import read_json, sha256, write_json

ROOT = Path.cwd()
PRIVATE = ROOT / "private"
AGG = ROOT / "results/aggregate"
NEURAL = ["mlp", "multiview"]


def xgb_process(request):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xgb_", dir=PRIVATE) as folder:
        folder = Path(folder)
        request["response_path"] = str(
            folder / ("response.npy" if request["operation"] == "predict" else "response.joblib")
        )
        joblib.dump(request, folder / "request.joblib")
        subprocess.run(
            [sys.executable, "-m", "cvd70k.training.xgb_worker", str(folder / "request.joblib")],
            check=True,
            timeout=180,
        )
        return (
            np.load(request["response_path"])
            if request["operation"] == "predict"
            else joblib.load(request["response_path"])
        )


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_partitions(mode):
    df, _ = load_csv(ROOT / "data/raw/cardio_train.csv")
    df, _ = clean(df, mode)
    assignments = pd.read_csv(ROOT / "data/processed/splits.csv")
    return df, assignments


def get_partition(df, assignments, part):
    idx = assignments.index[assignments.split == part].intersection(df.index)
    return df.loc[idx]


def fit_run(
    run_id,
    family,
    intervention,
    train,
    validation,
    settings,
    cfg,
    seed,
    device,
    mode="minimal",
    anthropometry="all",
    analysis="full",
    requested=None,
):
    meta_path = PRIVATE / "runs" / f"{run_id}.json"
    specification = {
        "run_id": run_id,
        "family": family,
        "intervention": intervention,
        "settings": settings,
        "seed": seed,
        "cleaning": mode,
        "anthropometry": anthropometry,
        "analysis": analysis,
        "requested_prevalence": requested,
        "config_sha": digest(cfg),
    }
    if meta_path.exists():
        saved = read_json(meta_path)
        if saved["specification"] != specification:
            raise ValueError(f"Incompatible resumed run {run_id}")
        if sha256(PRIVATE / "models" / f"{run_id}.joblib") != saved["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint integrity mismatch {run_id}")
        return saved
    prep = Preprocessor(anthropometry).fit(train)
    x, xv = prep.transform(train), prep.transform(validation)
    y, yv = train.cardio.to_numpy(), validation.cardio.to_numpy()
    dims = [len(v) for v in prep.views.values()]
    start = time.perf_counter()
    if family in NEURAL:
        nc = {**cfg["neural"], **settings}
        model, info = train_nn(x, y, xv, yv, family, dims, nc, seed, intervention, device)
        pv = predict_nn(model, xv, device)
        payload = {
            "state": {k: v.cpu() for k, v in model.state_dict().items()},
            "neural_config": nc,
            "dims": dims,
        }
    else:
        indices = oversample_indices(y, seed) if intervention == "resampled" else np.arange(len(y))
        weight = float((y == 0).sum() / (y == 1).sum()) if intervention == "weighted" else 1.0
        if family == "logistic":
            model = LogisticRegression(
                **settings, max_iter=2000, random_state=seed, class_weight={0: 1.0, 1: weight}
            )
        elif family == "random_forest":
            model = RandomForestClassifier(
                **cfg["random_forest"],
                **settings,
                random_state=seed,
                class_weight={0: 1.0, 1: weight},
            )
        elif family != "xgboost":
            raise ValueError(family)
        if family == "xgboost":
            native_path = PRIVATE / "xgboost" / f"{run_id}.joblib"
            native_path.parent.mkdir(parents=True, exist_ok=True)
            params = {
                **cfg["xgboost"],
                **settings,
                "random_state": seed,
                "scale_pos_weight": weight,
            }
            response = xgb_process(
                {
                    "operation": "fit",
                    "x": x[indices],
                    "y": y[indices],
                    "xv": xv,
                    "yv": yv,
                    "params": params,
                    "model_path": str(native_path),
                }
            )
            pv, curve = response["probability"], response["curve"]
        else:
            model.fit(x[indices], y[indices])
            curve = []
            pv = model.predict_proba(xv)[:, 1]
        info = {
            "duration_seconds": time.perf_counter() - start,
            "curve": curve,
            "parameter_count": int(model.coef_.size + model.intercept_.size)
            if family == "logistic"
            else None,
            "tree_nodes": sum(t.tree_.node_count for t in model.estimators_)
            if family == "random_forest"
            else None,
            "best_epoch": response["best_epoch"] if family == "xgboost" else None,
            "fit_examples_per_epoch": len(indices),
            "pos_weight": weight,
        }
        payload = (
            {
                "xgboost_model_path": str(native_path.relative_to(ROOT)),
                "xgboost_sha256": sha256(native_path),
            }
            if family == "xgboost"
            else {"estimator": model}
        )
    threshold = select_threshold(yv, pv)
    payload.update({"preprocessor": prep, "family": family})
    path = PRIVATE / "models" / f"{run_id}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path, compress=3)
    pred_path = PRIVATE / "validation" / f"{run_id}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, probability=pv)
    record = {
        "specification": specification,
        "train_n": len(y),
        "train_positive": int(y.sum()),
        "train_negative": int((y == 0).sum()),
        "train_prevalence": float(y.mean()),
        "validation_default": metrics(yv, pv),
        "validation_selected": metrics(yv, pv, threshold),
        "threshold": threshold,
        "training": info,
        "checkpoint_sha256": sha256(path),
        "finished_utc": stamp(),
    }
    write_json(meta_path, record)
    print(
        f"{run_id}: val AP={record['validation_default']['ap']:.4f} "
        f"n={len(y)} fit={info['duration_seconds']:.2f}s",
        flush=True,
    )
    return record


def development(cfg):
    if (AGG / "test_access.json").exists():
        raise RuntimeError("Final test already opened; no further development in this workspace")
    start = time.perf_counter()
    hardware = read_json(AGG / "hardware.json")
    device = hardware["selected_device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Recorded MPS selected but unavailable in current execution context")
    torch.set_num_threads(2)
    minimal, assignments = load_partitions("minimal")
    train = get_partition(minimal, assignments, "train")
    val = get_partition(minimal, assignments, "validation")
    chosen, search_records = {}, []
    for family in cfg["families"]:
        records = []
        for k, settings in enumerate(cfg["search"][family]):
            record = fit_run(
                f"search_{family}_{k}",
                family,
                "standard",
                train,
                val,
                settings,
                cfg,
                cfg["seeds"][0],
                device,
                analysis="search",
            )
            records.append(record)
            search_records.append(record)
        best = max(range(len(records)), key=lambda k: records[k]["validation_default"]["ap"])
        chosen[family] = cfg["search"][family][best]
    write_json(
        AGG / "hyperparameter_selection.json", {"selected": chosen, "search": search_records}
    )
    records = []
    for mode in cfg["cleaning_modes"]:
        df, assignments = load_partitions(mode)
        tr, va = [get_partition(df, assignments, part) for part in ["train", "validation"]]
        for family in cfg["families"]:
            interventions = (
                cfg["interventions"]
                if mode == "minimal" and family == "multiview"
                else ["standard"]
            )
            for intervention in interventions:
                for seed in cfg["seeds"]:
                    records.append(
                        fit_run(
                            f"full_{mode}_{family}_{intervention}_{seed}",
                            family,
                            intervention,
                            tr,
                            va,
                            chosen[family],
                            cfg,
                            seed,
                            device,
                            mode=mode,
                        )
                    )
    for anthropometry in cfg["anthropometry_sensitivity"]:
        for family in NEURAL:
            for seed in cfg["seeds"]:
                records.append(
                    fit_run(
                        f"anthro_{anthropometry}_{family}_{seed}",
                        family,
                        "standard",
                        train,
                        val,
                        chosen[family],
                        cfg,
                        seed,
                        device,
                        anthropometry=anthropometry,
                        analysis="anthropometry",
                    )
                )
    plans = feasible_plans(train.cardio, cfg["prevalences"])
    write_json(
        AGG / "sampling_plan.json",
        {
            "train_positive_pool": int(train.cardio.sum()),
            "train_negative_pool": int((train.cardio == 0).sum()),
            "plans": plans,
        },
    )
    for plan in plans:
        for seed in cfg["seeds"]:
            indices = sample_indices(train.cardio, plan["positive"], plan["negative"], seed)
            subset = train.iloc[indices]
            for family in cfg["families"]:
                for intervention in cfg["interventions"]:
                    run_id = f"{plan['analysis']}_{plan['requested_prevalence']}_{family}_{intervention}_{seed}"
                    records.append(
                        fit_run(
                            run_id,
                            family,
                            intervention,
                            subset,
                            val,
                            chosen[family],
                            cfg,
                            seed,
                            device,
                            analysis=plan["analysis"],
                            requested=plan["requested_prevalence"],
                        )
                    )
    # Lock the winner by *mean validation AP*, never by test metrics.
    scores = {}
    for r in records:
        s = r["specification"]
        if s["analysis"] == "full" and s["cleaning"] == "minimal":
            key = s["family"] + "/" + s["intervention"]
            scores.setdefault(key, []).append(r["validation_default"]["ap"])
    mean_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    development = {
        "config": cfg,
        "selected_hyperparameters": chosen,
        "validation_mean_ap": mean_scores,
        "validation_selected_winner": max(mean_scores, key=mean_scores.get),
        "records": records,
        "execution_seconds_this_invocation": time.perf_counter() - start,
        "platform": platform.platform(),
        "finished_utc": stamp(),
    }
    write_json(AGG / "development.json", development)
    inference_files = [
        "src/cvd70k/data/pipeline.py",
        "src/cvd70k/models/networks.py",
        "src/cvd70k/training/engine.py",
        "src/cvd70k/training/xgb_worker.py",
        "src/cvd70k/experiments/runner.py",
        "src/cvd70k/evaluation/metrics.py",
        "src/cvd70k/data/sampling.py",
    ]
    lock = {
        "locked_utc": stamp(),
        "development_sha256": sha256(AGG / "development.json"),
        "config_sha256": sha256(ROOT / "configs/benchmark.json"),
        "protocol_sha256": sha256(ROOT / "docs/experiment_protocol.md"),
        "splits_sha256": sha256(ROOT / "data/processed/splits.csv"),
        "source_sha256": {p: sha256(ROOT / p) for p in inference_files},
        "validation_selected_winner": development["validation_selected_winner"],
        "evaluated_run_count": len(records),
        "search_run_count": len(search_records),
    }
    write_json(AGG / "development_lock.json", lock)
    print("DEVELOPMENT LOCKED", lock, flush=True)


def final_evaluation():
    lock = read_json(AGG / "development_lock.json")
    for path, expected in [
        ("results/aggregate/development.json", lock["development_sha256"]),
        ("configs/benchmark.json", lock["config_sha256"]),
        ("docs/experiment_protocol.md", lock["protocol_sha256"]),
        ("data/processed/splits.csv", lock["splits_sha256"]),
    ] + list(lock["source_sha256"].items()):
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Stale development lock: {path}")
    dev = read_json(AGG / "development.json")
    # Verify all checkpoints before opening test inference.
    for r in dev["records"]:
        p = PRIVATE / "models" / (r["specification"]["run_id"] + ".joblib")
        if sha256(p) != r["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch {p.name}")
    access_path = AGG / "test_access.json"
    if not access_path.exists():
        write_json(
            access_path,
            {
                "first_opened_utc": stamp(),
                "lock_sha256": sha256(AGG / "development_lock.json"),
                "purpose": "One locked evaluation of all prespecified fitted models",
            },
        )
    torch.set_num_threads(2)
    rows, grids = [], []
    partitions = {}
    for mode in dev["config"]["cleaning_modes"]:
        df, assignments = load_partitions(mode)
        test = get_partition(df, assignments, "test")
        partitions[mode] = test
    out = PRIVATE / "test_predictions"
    out.mkdir(parents=True, exist_ok=True)
    for r in dev["records"]:
        spec = r["specification"]
        run_id = spec["run_id"]
        test = partitions[spec["cleaning"]]
        y = test.cardio.to_numpy()
        path = out / f"{run_id}.npz"
        if path.exists():
            p = np.load(path)["probability"]
        else:
            payload = joblib.load(PRIVATE / "models" / f"{run_id}.joblib")
            x = payload["preprocessor"].transform(test)
            if spec["family"] in NEURAL:
                model = build_model(spec["family"], payload["dims"], payload["neural_config"])
                model.load_state_dict(payload["state"])
                # CPU inference is deterministic and avoids device dependence in reporting.
                p = predict_nn(model, x, "cpu")
            elif spec["family"] == "xgboost":
                native_path = ROOT / payload["xgboost_model_path"]
                if sha256(native_path) != payload["xgboost_sha256"]:
                    raise ValueError("Native XGBoost checkpoint integrity mismatch")
                p = xgb_process({"operation": "predict", "x": x, "model_path": str(native_path)})
            else:
                p = payload["estimator"].predict_proba(x)[:, 1]
            np.savez_compressed(path, probability=p, label=y, row=test.index.to_numpy())
        metadata = {k: v for k, v in spec.items() if k not in ["settings", "config_sha"]}
        metadata.update(
            {k: r[k] for k in ["train_n", "train_positive", "train_negative", "train_prevalence"]}
        )
        metadata.update(
            {
                k: r["training"].get(k)
                for k in ["duration_seconds", "parameter_count", "best_epoch", "epochs_run"]
            }
        )
        metadata["validation_ap"] = r["validation_default"]["ap"]
        for operating, threshold in [("default", 0.5), ("validation_selected", r["threshold"])]:
            rows.append({**metadata, "operating": operating, **metrics(y, p, threshold)})
        for m in threshold_grid(y, p):
            grids.append({**metadata, **m})
    pd.DataFrame(rows).to_csv(AGG / "test_metrics.csv", index=False)
    pd.DataFrame(grids).to_csv(AGG / "threshold_grid.csv", index=False)
    write_json(
        AGG / "evaluation_complete.json",
        {
            "finished_utc": stamp(),
            "fitted_models_evaluated": len(dev["records"]),
            "development_lock_sha256": sha256(AGG / "development_lock.json"),
        },
    )
    print(f"EVALUATED {len(dev['records'])} locked fitted models", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--config", default="configs/benchmark.json")
    args = parser.parse_args()
    if args.stage == "develop":
        development(read_json(args.config))
    else:
        final_evaluation()


if __name__ == "__main__":
    main()
