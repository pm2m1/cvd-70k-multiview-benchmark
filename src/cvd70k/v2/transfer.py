"""Disjoint-role internal subgroup adaptation with full selector evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from cvd70k.data.sampling import sample_indices
from cvd70k.evaluation.metrics import metrics
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.config import candidates
from cvd70k.v2.data import development_data, partition_transfer
from cvd70k.v2.engine import (
    PRIVATE,
    TRIALS,
    fingerprint,
    fit_trial,
    predict_trial,
    probabilities,
    train_network,
)
from cvd70k.v2.models import MODULES, model_for
from cvd70k.v2.selector import (
    calibrated_mask,
    evidence,
    legacy_mask,
    matched_random,
    set_mask,
    snapshot_frozen,
    verify_frozen,
)

ROOT = Path.cwd()
AGG = ROOT / "results/v2/aggregate"
RECORDS = AGG / "transfer_trials"
MODELS = PRIVATE / "transfer_models"


def representation(model, prep, df):
    model.to("cpu").eval()
    x = torch.tensor(prep.transform(df))
    out = {n: [] for n in MODULES}
    with torch.no_grad():
        for part in x.split(2048):
            _, features = model(part, True)
            for name, value in features.items():
                out[name].append(value.numpy())
    return {n: np.concatenate(v) for n, v in out.items()}


def adapt(run, source_bundle, fit, stop, mask, cfg, seed, device, zero=True):
    path = MODELS / f"{run}.joblib"
    record_path = RECORDS / f"{run}.json"
    specification = {
        "run": run,
        "mask": mask,
        "config": cfg,
        "seed": seed,
        "epoch_zero_eligible": zero,
        "fit_membership_sha256": fingerprint(fit.index.tolist()),
    }
    if record_path.exists():
        record = read_json(record_path)
        if record["specification"] != specification or sha256(path) != record["checkpoint_sha256"]:
            raise ValueError("Stale adaptation " + run)
        return record
    prep = source_bundle["preprocessor"]
    model = model_for(prep, source_bundle["config"])
    model.load_state_dict(source_bundle["state"])
    set_mask(model, mask)
    snapshot = snapshot_frozen(model, mask)
    model, info = train_network(
        model,
        prep.transform(fit),
        fit.cardio.to_numpy(),
        prep.transform(stop),
        stop.cardio.to_numpy(),
        cfg,
        seed,
        device,
        zero=zero,
    )
    assert verify_frozen(model, snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"state": model.state_dict(), "preprocessor": prep, "config": source_bundle["config"]},
        path,
        compress=3,
    )
    rec = {
        "specification": specification,
        "training": info,
        "checkpoint_sha256": sha256(path),
        "frozen_verified": True,
        "fit_n": len(fit),
        "fit_positive": int(fit.cardio.sum()),
        "status": "completed",
    }
    write_json(record_path, rec)
    print(
        run, "completed epoch", info["best_epoch"], "trainable", info["parameter_count"], flush=True
    )
    return rec


def prepare_roles():
    parts = partition_transfer(development_data())
    role_table = pd.concat(
        [pd.DataFrame({"row": v.index, "role": k}) for k, v in parts.items()], ignore_index=True
    )
    path = PRIVATE / "transfer_roles.csv"
    if path.exists() and path.read_text() != role_table.to_csv(index=False):
        raise ValueError("Transfer roles changed")
    path.write_text(role_table.to_csv(index=False))
    return parts


def develop():
    if (AGG / "transfer_evaluation_access.json").exists():
        raise RuntimeError("Transfer evaluation already opened")
    torch.set_num_threads(2)
    cfg = read_json(ROOT / "configs/v2/study.json")
    parts = prepare_roles()
    source_cfg = {
        **candidates("multiview")[0],
        "epochs": 120,
        "patience": 16,
        "batch": 512,
        "stop": "ap",
    }
    adaptation_cfg = {**source_cfg, "epochs": 80, "patience": 12, "batch": 128, "stop": "ap"}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    conditions = [
        {"positive": p, "negative": p, "prevalence": 0.5, "design": "support"}
        for p in cfg["transfer"]["positive_budgets"]
    ]
    conditions.append(
        {"positive": 250, "negative": 750, "prevalence": 0.25, "design": "prevalence"}
    )
    manifest = []
    evidence_records = []
    for seed in cfg["transfer"]["draw_seeds"]:
        source_run = f"transfer_source_s{seed}"
        fit_trial(source_run, parts["source_train"], parts["source_stop"], source_cfg, seed, device)
        source_bundle = joblib.load(PRIVATE / "models" / f"{source_run}.joblib")
        source_model = model_for(source_bundle["preprocessor"], source_cfg)
        source_model.load_state_dict(source_bundle["state"])
        src = representation(source_model, source_bundle["preprocessor"], parts["source_selector"])
        for condition in conditions:
            positive, negative = condition["positive"], condition["negative"]
            fit_pool, selector_pool = parts["target_fit_pool"], parts["target_selector_pool"]
            fit = fit_pool.iloc[sample_indices(fit_pool.cardio, positive, negative, seed)]
            selector = selector_pool.iloc[
                sample_indices(selector_pool.cardio, positive, negative, seed + 10000)
            ]
            assert not set(fit.group) & set(selector.group)
            key = f"p{positive}_n{negative}_s{seed}"
            target = representation(source_model, source_bundle["preprocessor"], selector)
            ev = evidence(
                src,
                parts["source_selector"].cardio.to_numpy(),
                target,
                selector.cardio.to_numpy(),
                seed=seed,
            )
            logical, reasons = calibrated_mask(ev)
            threshold, _ = calibrated_mask(ev, "threshold")
            random, random_info = matched_random(source_model, logical, seed)
            masks = {
                "head_only": {n: False for n in MODULES},
                "full": {n: True for n in MODULES},
                "fixed": {n: n in ["fusion1", "fusion2"] for n in MODULES},
                "random_matched": random,
                "legacy_topk": legacy_mask(ev),
                "calibrated_threshold": threshold,
                "calibrated_logical": logical,
            }
            evidence_records.append(
                {
                    "key": key,
                    **condition,
                    "seed": seed,
                    "modules": ev,
                    "logical_mask": logical,
                    "logical_reasons": reasons,
                    "random_control": random_info,
                    "condition": "primary",
                }
            )
            base = {
                "key": key,
                **condition,
                "seed": seed,
                "target_fit_positive": positive,
                "target_fit_negative": negative,
                "target_stop_n": len(parts["target_stop"]),
                "target_stop_positive": int(parts["target_stop"].cardio.sum()),
            }
            manifest.append(
                {
                    **base,
                    "method": "source_only",
                    "kind": "source",
                    "run": source_run,
                    "target_fit_positive": 0,
                    "target_fit_negative": 0,
                    "target_stop_n": 0,
                    "target_stop_positive": 0,
                    "selector_positive": 0,
                    "selector_negative": 0,
                }
            )
            for family in ["multiview", "logistic", "xgboost"]:
                tc = {**adaptation_cfg, "family": family}
                if family == "logistic":
                    tc["C"] = 1.0
                if family == "xgboost":
                    tc.update(
                        xgb={
                            "max_depth": 3,
                            "learning_rate": 0.05,
                            "subsample": 0.9,
                            "colsample_bytree": 0.9,
                            "reg_lambda": 1.0,
                        },
                        epochs=600,
                        patience=30,
                        tree_stop="bce",
                    )
                run = f"transfer_target_{family}_{key}"
                fit_trial(
                    run,
                    fit,
                    parts["target_stop"],
                    tc,
                    seed,
                    device if family == "multiview" else "cpu",
                )
                manifest.append(
                    {
                        **base,
                        "method": "target_only" if family == "multiview" else "target_" + family,
                        "kind": "scratch",
                        "run": run,
                        "selector_positive": 0,
                        "selector_negative": 0,
                    }
                )
            for method, mask in masks.items():
                run = f"transfer_{method}_{key}"
                adapt(
                    run,
                    source_bundle,
                    fit,
                    parts["target_stop"],
                    mask,
                    adaptation_cfg,
                    seed,
                    device,
                )
                uses_selector = method in [
                    "random_matched",
                    "legacy_topk",
                    "calibrated_threshold",
                    "calibrated_logical",
                ]
                manifest.append(
                    {
                        **base,
                        "method": method,
                        "kind": "adapt",
                        "run": run,
                        "selector_positive": positive if uses_selector else 0,
                        "selector_negative": negative if uses_selector else 0,
                    }
                )
                if method in ["head_only", "full", "calibrated_logical"]:
                    for stop, zero, suffix in [
                        ("bce", True, "bce_zero"),
                        ("ap", False, "ap_nozero"),
                    ]:
                        ablation = {**adaptation_cfg, "stop": stop}
                        ab_run = f"transfer_{method}_{suffix}_{key}"
                        adapt(
                            ab_run,
                            source_bundle,
                            fit,
                            parts["target_stop"],
                            mask,
                            ablation,
                            seed,
                            device,
                            zero,
                        )
                        manifest.append(
                            {
                                **base,
                                "method": method + "_" + suffix,
                                "kind": "adapt",
                                "run": ab_run,
                                "selector_positive": positive if uses_selector else 0,
                                "selector_negative": negative if uses_selector else 0,
                            }
                        )
            cost_run = f"transfer_full_cost_matched_{key}"
            adapt(
                cost_run,
                source_bundle,
                pd.concat([fit, selector]),
                parts["target_stop"],
                masks["full"],
                adaptation_cfg,
                seed,
                device,
            )
            manifest.append(
                {
                    **base,
                    "method": "full_cost_matched",
                    "kind": "adapt",
                    "run": cost_run,
                    "target_fit_positive": positive * 2,
                    "target_fit_negative": negative * 2,
                    "selector_positive": 0,
                    "selector_negative": 0,
                }
            )
            if condition["design"] == "support" and positive in [25, 1000]:
                for name, gcap, mcap in [("geometry128", 128, 128), ("mmd512", None, 512)]:
                    ab_ev = evidence(
                        src,
                        parts["source_selector"].cardio.to_numpy(),
                        target,
                        selector.cardio.to_numpy(),
                        seed=seed,
                        mmd_cap=mcap,
                        geometry_cap=gcap,
                    )
                    ab_mask, why = calibrated_mask(ab_ev)
                    evidence_records.append(
                        {
                            "key": key,
                            **condition,
                            "seed": seed,
                            "modules": ab_ev,
                            "logical_mask": ab_mask,
                            "logical_reasons": why,
                            "condition": name,
                        }
                    )
                    run = f"transfer_logical_{name}_{key}"
                    adapt(
                        run,
                        source_bundle,
                        fit,
                        parts["target_stop"],
                        ab_mask,
                        adaptation_cfg,
                        seed,
                        device,
                    )
                    manifest.append(
                        {
                            **base,
                            "method": "logical_" + name,
                            "kind": "adapt",
                            "run": run,
                            "selector_positive": positive,
                            "selector_negative": negative,
                        }
                    )
            write_json(AGG / "transfer_evidence.json", evidence_records)
            write_json(AGG / "transfer_manifest.json", manifest)
    checkpoint_hashes = {
        r["run"]: sha256(
            (MODELS if r["kind"] == "adapt" else PRIVATE / "models") / f"{r['run']}.joblib"
        )
        for r in manifest
    }
    lock = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(ROOT / "docs/v2/INTERNAL_TRANSFER_PROTOCOL.md"),
        "study_sha256": sha256(ROOT / "configs/v2/study.json"),
        "manifest_sha256": sha256(AGG / "transfer_manifest.json"),
        "evidence_sha256": sha256(AGG / "transfer_evidence.json"),
        "roles_sha256": sha256(PRIVATE / "transfer_roles.csv"),
        "checkpoints": checkpoint_hashes,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                ROOT / "src/cvd70k/v2/selector.py",
                ROOT / "src/cvd70k/v2/transfer.py",
                ROOT / "src/cvd70k/v2/engine.py",
                ROOT / "src/cvd70k/v2/models.py",
            ]
        },
    }
    write_json(AGG / "transfer_lock.json", lock)
    print("TRANSFER DEVELOPMENT LOCKED", len(checkpoint_hashes), flush=True)


def evaluate():
    torch.set_num_threads(2)
    lock = read_json(AGG / "transfer_lock.json")
    for path, h in [
        ("docs/v2/INTERNAL_TRANSFER_PROTOCOL.md", lock["protocol_sha256"]),
        ("configs/v2/study.json", lock["study_sha256"]),
        ("results/v2/aggregate/transfer_manifest.json", lock["manifest_sha256"]),
        ("results/v2/aggregate/transfer_evidence.json", lock["evidence_sha256"]),
    ] + list(lock["source_sha256"].items()):
        if sha256(ROOT / path) != h:
            raise ValueError("Stale transfer lock " + path)
    manifest = read_json(AGG / "transfer_manifest.json")
    for r in manifest:
        path = (MODELS if r["kind"] == "adapt" else PRIVATE / "models") / f"{r['run']}.joblib"
        if sha256(path) != lock["checkpoints"][r["run"]]:
            raise ValueError("Checkpoint corrupted")
    write_json(
        AGG / "transfer_evaluation_access.json",
        {
            "opened_utc": datetime.now(timezone.utc).isoformat(),
            "lock_sha256": sha256(AGG / "transfer_lock.json"),
        },
    )
    parts = prepare_roles()
    test = parts["target_evaluation"]
    rows = []
    output = PRIVATE / "transfer_predictions"
    output.mkdir(exist_ok=True)
    cache = {}
    for r in manifest:
        run = r["run"]
        p = cache.get(run)
        if p is None:
            if r["kind"] != "adapt":
                p = predict_trial(run, test)
            else:
                bundle = joblib.load(MODELS / f"{run}.joblib")
                model = model_for(bundle["preprocessor"], bundle["config"])
                model.load_state_dict(bundle["state"])
                p = probabilities(model, bundle["preprocessor"].transform(test))
            cache[run] = p
            np.savez_compressed(output / f"{run}.npz", probability=p, row=test.index.to_numpy())
        if r["kind"] == "adapt":
            training = read_json(RECORDS / f"{run}.json")["training"]
        else:
            training = read_json(TRIALS / f"{run}.json")["training"]
        cost = (
            r["target_fit_positive"]
            + r["target_fit_negative"]
            + r["selector_positive"]
            + r["selector_negative"]
            + r["target_stop_n"]
        )
        rows.append(
            {
                **r,
                "total_target_labels": cost,
                "trainable_parameters": training["parameter_count"] if r["kind"] != "source" else 0,
                "best_epoch": training["best_epoch"],
                "fit_seconds": training["fit_seconds"] if r["kind"] != "source" else 0.0,
                **metrics(test.cardio.to_numpy(), p),
            }
        )
    pd.DataFrame(rows).to_csv(AGG / "transfer_metrics.csv", index=False)
    print("TRANSFER EVALUATION COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "evaluate"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        develop() if args.stage == "develop" else evaluate()
