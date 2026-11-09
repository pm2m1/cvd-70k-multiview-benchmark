"""Fit and lock first; evaluation is a separate command with manifest enforcement."""

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cvd70k.evaluation.metrics import metrics
from cvd70k.training.engine import set_seed
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.engine import probabilities
from cvd70k.v2.models import MultiView
from cvd70k.v2.selector import evidence, snapshot_frozen, verify_frozen
from cvd70k.v3.adaptation import Radial, anchors, drift, policy, source_penalty
from cvd70k.v3.design import (
    CFG,
    METHODS,
    NEW,
    PRIV,
    PUB,
    ROOT,
    balanced,
    digest,
    immutable,
    label_cost,
    prepare,
)
from cvd70k.v3.engine import checkpoint, state_hash, train
from cvd70k.v3.features import Representation
from cvd70k.v3.models import build

CAT = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def native_frame(df):
    # Schema-defined categories, never learned from evaluation records.
    x = df[["age_years", "ap_hi", "ap_lo", "height", "weight", "bmi", *CAT]].copy()
    for c in CAT:
        x[c] = x[c].fillna(-1).astype(int).astype(str)
    for c in x.columns.difference(CAT):
        x[c] = x[c].fillna(-999.0)
    return x


def tree_call(request):
    PRIV.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PRIV, prefix="native_") as folder:
        req = Path(folder) / "request.joblib"
        response = Path(folder) / "response.joblib"
        joblib.dump(request, req)
        subprocess.run(
            [sys.executable, "-m", "cvd70k.v3.tree_worker", str(req), str(response)],
            check=True,
            timeout=900,
        )
        return joblib.load(response)


def fit_predictor(run, fit, stop, cfg, seed, device):
    spec = dict(
        config=cfg,
        seed=seed,
        fit=digest(list(map(int, fit.index))),
        stop=digest(list(map(int, stop.index))),
    )

    def execute():
        prep = Representation(cfg["block"], cfg.get("scale", "standard")).fit(fit)
        if cfg["family"] == "neural":
            set_seed(seed)
            model = build(prep, cfg)
            used = "cpu" if cfg["architecture"] == "ft" or cfg.get("loss") == "rank" else device
            model, info = train(
                model,
                prep.transform(fit),
                fit.cardio.to_numpy(),
                prep.transform(stop),
                stop.cardio.to_numpy(),
                cfg,
                seed,
                used,
            )
            bundle = dict(state=model.state_dict(), preprocessor=prep, config=cfg)
        elif cfg["family"] == "logistic":
            import time

            from sklearn.linear_model import LogisticRegression

            start = time.perf_counter()
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
                prep.transform(fit), fit.cardio
            )
            info = dict(
                fit_seconds=time.perf_counter() - start,
                parameter_count=len(prep.names) + 1,
                best_epoch=int(model.n_iter_[0]),
                training_device="cpu",
            )
            bundle = dict(logistic=model, preprocessor=prep, config=cfg)
        else:
            path = PRIV / "native" / f"{run}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            request = dict(
                family=cfg["family"],
                config=cfg,
                seed=seed,
                x=prep.transform(fit),
                y=fit.cardio.to_numpy(),
                xs=prep.transform(stop),
                ys=stop.cardio.to_numpy(),
                native_path=str(path),
            )
            if cfg["family"] == "catboost":
                request.update(cat_x=native_frame(fit), cat_xs=native_frame(stop), cat_features=CAT)
            bundle, info = tree_call(request)
            bundle.update(preprocessor=prep, config=cfg, native_sha256=sha256(path))
        return bundle, info

    return checkpoint(run, spec, execute)


def predict(bundle, df):
    prep = bundle["preprocessor"]
    cfg = bundle["config"]
    if "native_path" in bundle:
        path = Path(bundle["native_path"])
        if sha256(path) != bundle["native_sha256"]:
            raise ValueError("Native checkpoint integrity")
        x = native_frame(df) if cfg["family"] == "catboost" else prep.transform(df)
        return tree_call(dict(action="predict", checkpoint=str(path), x=x))
    if "logistic" in bundle:
        return bundle["logistic"].predict_proba(prep.transform(df))[:, 1]
    model = build(prep, cfg)
    model.load_state_dict(bundle["state"])
    return probabilities(model, prep.transform(df))


def safe_scores(y, p):
    return {"ap": metrics(y, p)["ap"], "brier": metrics(y, p)["brier"]}


def lock(track, value):
    files = [
        ROOT / "configs/v3/study.json",
        ROOT / "docs/v3/RESEARCH_PROTOCOL.md",
        *sorted((ROOT / "src/cvd70k/v3").glob("*.py")),
    ]
    value.update(
        code={str(p.relative_to(ROOT)): sha256(p) for p in files},
        roles_sha256=sha256(PRIV / "roles.csv"),
    )
    immutable(PUB / f"{track}_lock.json", value)


def classification(device):
    df, parts, _ = prepare()
    specs = read_json(ROOT / "configs/v3/study.json")["candidates"]
    selected = {}
    records = []
    for k, roles in parts.items():
        fit, stop, select = [df.loc[roles[r]] for r in ["fit", "stop", "select"]]
        scores = {}
        predictions = {}
        for name, cfg in specs.items():
            ff, ss = fit, stop
            if cfg.get("strict"):
                ff = ff.dropna(subset=["age_years", "ap_hi", "ap_lo", "height", "weight", "bmi"])
                ss = ss.dropna(subset=["age_years", "ap_hi", "ap_lo", "height", "weight", "bmi"])
            run = f"class_o{k}_{name}_s101"
            bundle, rec = fit_predictor(run, ff, ss, cfg, 101, device)
            p = predict(bundle, select)
            predictions[name] = p
            scores[name] = safe_scores(select.cardio, p)
            records.append(dict(outer=k, candidate=name, run=run, **scores[name]))
        eligible = [
            n
            for n in specs
            if specs[n]["family"] == "neural"
            and n
            not in [
                "v1_reference",
                "v2_reference",
                "strict_reference",
                "shuffled_quality",
                "uniform_gate",
                "random_gate",
                "depth_control",
                "attention_no_identity",
                "ft_reference",
            ]
        ]
        neural = min(eligible, key=lambda n: (-scores[n]["ap"], scores[n]["brier"], n))
        tree = min(
            [n for n in specs if specs[n]["family"] in ["xgboost", "lightgbm", "catboost"]],
            key=lambda n: (-scores[n]["ap"], scores[n]["brier"], n),
        )
        for name in [neural, tree]:
            fit_predictor(f"class_o{k}_{name}_s202", fit, stop, specs[name], 202, device)
        # Weights fitted only on held-out inner selection predictions.
        bestview = min(
            ["v2_reference", "gated", "gated_regularized", "residual", "attention", "pairwise"],
            key=lambda n: (-scores[n]["ap"], n),
        )
        ensembles = {}
        for tag, names in {
            "tree_view": [tree, bestview],
            "tree_gated": [tree, "gated"],
            "tree_residual": [tree, "residual"],
            "tree_gated_residual": [tree, "gated", "residual"],
        }.items():
            pp = np.column_stack([predictions[n] for n in names])
            count = pp.shape[1]
            weights = [np.ones(count) / count]
            for a in range(11):
                for b in range(11 - a):
                    if count == 3:
                        weights.append(np.array([a, b, 10 - a - b]) / 10)
                if count == 2:
                    weights.append(np.array([a, 10 - a]) / 10)
            best = min(weights, key=lambda w: (-safe_scores(select.cardio, pp @ w)["ap"], tuple(w)))
            ensembles[tag] = dict(
                names=names, weights=best.tolist(), equal_weights=(np.ones(count) / count).tolist()
            )
        selected[k] = dict(neural=neural, tree=tree, ensembles=ensembles)
    write_json(PUB / "classification_inner_scores.json", records)
    lock(
        "classification",
        dict(
            selected=selected,
            checkpoints={
                p.stem: read_json(p)["checkpoint_sha256"]
                for p in (PUB / "trials").glob("class_*.json")
            },
        ),
    )
    print("CLASSIFICATION LOCKED", flush=True)


def source_model(bundle):
    model = MultiView(bundle["preprocessor"].dims, bundle["config"])
    model.load_state_dict(bundle["state"])
    return model


def representations(model, prep, df):
    model.eval()
    with torch.no_grad():
        _, features = model(torch.tensor(prep.transform(df)), representations=True)
    return {m: v.numpy() for m, v in features.items()}


def adapt(run, source_bundle, fit, stop, cfg, method, ev, seed, device):
    spec = dict(
        method=method,
        config=cfg,
        seed=seed,
        source=digest({k: v.numpy().tolist() for k, v in source_bundle["state"].items()}),
        fit=digest(list(map(int, fit.index))),
        stop=digest(list(map(int, stop.index))),
        evidence=digest(ev),
    )

    def execute():
        source = source_model(source_bundle)
        before = state_hash(source)
        model, mask, weights, _ = policy(source, method, ev, seed)
        reference = anchors(source)
        frozen = snapshot_frozen(source, mask)
        prep = source_bundle["preprocessor"]
        pen = (
            (lambda m: m.regularizer())
            if isinstance(model, Radial)
            else (lambda m: source_penalty(m, reference, weights))
        )
        projection = (lambda m: m.project()) if isinstance(model, Radial) else None
        model, info = train(
            model,
            prep.transform(fit),
            fit.cardio.to_numpy(),
            prep.transform(stop),
            stop.cardio.to_numpy(),
            cfg,
            seed,
            device,
            True,
            pen,
            projection,
        )
        gates = model.gates() if isinstance(model, Radial) else None
        target = model.materialize().cpu() if isinstance(model, Radial) else model
        verify_frozen(target, frozen)
        if state_hash(source) != before:
            raise ValueError("Source mutated")
        info.update(
            mask=mask,
            penalties=weights,
            gates=gates,
            drift=drift(source, target),
            source_state_sha256=before,
            source_verified=True,
        )
        return dict(
            state=target.state_dict(), preprocessor=prep, config=source_bundle["config"]
        ), info

    return checkpoint(run, spec, execute)


def transfer(device):
    df, _, parts = prepare()
    manifest = []
    choices = {}
    all_evidence = {}
    acfg = {
        **CFG,
        "block": "base",
        "architecture": "concat",
        "family": "neural",
        "epochs": 80,
        "patience": 12,
        "lr": 0.0003,
        "batch": 128,
        "weight_decay": 0.0,
    }
    scfg = {**CFG, "block": "base", "architecture": "concat", "family": "neural"}
    for k, roles in parts.items():
        seed = [101, 202, 303][k]
        source, rec = fit_predictor(
            f"source_o{k}",
            df.loc[roles["source_fit"]],
            df.loc[roles["source_stop"]],
            scfg,
            seed,
            device,
        )
        model = source_model(source)
        src = representations(model, source["preprocessor"], df.loc[roles["source_reference"]])
        stop = balanced(df.loc[roles["stop"]], 50, 6000 + k)
        select = balanced(df.loc[roles["select"]], 50, 7000 + k)
        for n in [25, 250, 1000]:
            fitting = balanced(df.loc[roles["fit"]], n, 4000 + k)
            selector = balanced(df.loc[roles["selector"]], n, 5000 + k)
            key = f"o{k}_p{n}"
            ep = PUB / "evidence" / f"{key}.json"
            if ep.exists():
                ev = read_json(ep)
            else:
                tgt = representations(model, source["preprocessor"], selector)
                ev = evidence(
                    src,
                    df.loc[roles["source_reference"]].cardio.to_numpy(),
                    tgt,
                    selector.cardio.to_numpy(),
                    seed,
                )
                write_json(ep, ev)
            all_evidence[key] = ev
            scores = {}
            costfit = pd.concat([fitting, selector, select])
            totalcost = label_cost(
                fitting=fitting, selector=selector, stopping=stop, model_selection=select
            )
            manifest.append(
                dict(
                    outer=k,
                    positive_budget=n,
                    method="source_only",
                    run=f"source_o{k}",
                    label_cost=label_cost(),
                    selection_workflow_cost=totalcost["total_distinct"],
                )
            )
            for method in METHODS:
                usefit = costfit if method.endswith("_cost") or method == "full_cost" else fitting
                run = f"transfer_{key}_{method}"
                if method.startswith("target_"):
                    cfg = {
                        **scfg,
                        "family": "logistic" if "lr" in method else "xgboost",
                        "variant": 0,
                        "rounds": 600,
                        "depth": 3,
                        "tree_lr": 0.03,
                        "l2": 3.0,
                        "child": 20,
                    }
                    bundle, rec = fit_predictor(run, usefit, stop, cfg, seed, device)
                else:
                    bundle, rec = adapt(run, source, usefit, stop, acfg, method, ev, seed, device)
                p = predict(bundle, select)
                scores[method] = safe_scores(select.cardio, p)
                uses_selector = method in [
                    "logical",
                    "random_matched",
                    "logical_sp_low",
                    "logical_sp_high",
                    "soft_evidence",
                    "soft_shuffled",
                ] or method.startswith("reg_")
                used = dict(fitting=usefit, stopping=stop)
                if uses_selector:
                    used["selector"] = selector
                manifest.append(
                    dict(
                        outer=k,
                        positive_budget=n,
                        method=method,
                        run=run,
                        label_cost=label_cost(**used),
                        selection_workflow_cost=totalcost["total_distinct"],
                        selection_ap=scores[method]["ap"],
                        selection_brier=scores[method]["brier"],
                    )
                )
            winner = min(NEW, key=lambda m: (-scores[m]["ap"], scores[m]["brier"], m))
            choices[key] = dict(method=winner, scores=scores, selection_workflow_cost=totalcost)
            # Retrospective comparator evidence uses only the dedicated selection role.
            if n == 25:
                small = balanced(df.loc[roles["stop"]], 10, 6000 + k)
                for method in ["head", "full_cost", winner]:
                    usefit = costfit if method == "full_cost" else fitting
                    run = f"smallstop_{key}_{method}"
                    adapt(run, source, usefit, small, acfg, method, ev, seed, device)
                    used = dict(fitting=usefit, stopping=small, model_selection=select)
                    if method == winner:
                        used["selector"] = selector
                    manifest.append(
                        dict(
                            outer=k,
                            positive_budget=n,
                            method="smallstop_" + method,
                            run=run,
                            label_cost=label_cost(**used),
                            selection_workflow_cost=label_cost(
                                fitting=fitting,
                                selector=selector,
                                stopping=small,
                                model_selection=select,
                            )["total_distinct"],
                        )
                    )
    write_json(PUB / "transfer_manifest.json", manifest)
    write_json(PUB / "transfer_choices.json", choices)
    write_json(PUB / "transfer_evidence.json", all_evidence)
    lock(
        "transfer",
        dict(
            choices=choices,
            manifest_sha256=sha256(PUB / "transfer_manifest.json"),
            checkpoints={
                p.stem: read_json(p)["checkpoint_sha256"]
                for p in (PUB / "trials").glob("*.json")
                if not p.stem.startswith("class_")
            },
        ),
    )
    print("TRANSFER LOCKED", flush=True)


def check_lock(track):
    value = read_json(PUB / f"{track}_lock.json")
    for f, h in value["code"].items():
        if sha256(ROOT / f) != h:
            raise ValueError("Locked implementation changed: " + f)
    if sha256(PRIV / "roles.csv") != value["roles_sha256"]:
        raise ValueError("Roles changed")
    for run, h in value["checkpoints"].items():
        if sha256(PRIV / "models" / f"{run}.joblib") != h:
            raise ValueError("Locked checkpoint changed")
    return value


def evaluate(track):
    value = check_lock(track)
    df, classification_parts, transfer_parts = prepare()
    output = []
    preds = {}
    immutable(
        PUB / f"{track}_evaluation_access.json",
        dict(
            lock_sha256=sha256(PUB / f"{track}_lock.json"), status="locked outer predictions opened"
        ),
    )

    def evaluate_one(k, name, run, subset, extra=None):
        bundle = joblib.load(PRIV / "models" / f"{run}.joblib")
        p = predict(bundle, subset)
        rec = read_json(PUB / "trials" / f"{run}.json")
        row = dict(
            outer=k,
            model=name,
            run=run,
            **(extra or {}),
            **metrics(subset.cardio.to_numpy(), p),
            parameter_count=rec.get("total_parameters", rec.get("parameter_count")),
            best_epoch=rec.get("best_epoch"),
            fit_seconds=rec.get("fit_seconds"),
        )
        output.append(row)
        return p

    if track == "classification":
        specs = read_json(ROOT / "configs/v3/study.json")["candidates"]
        for k, roles in classification_parts.items():
            subset = df.loc[roles["evaluation"]]
            pp = {}
            selections = value["selected"][str(k)]
            for name in specs:
                pp[name] = evaluate_one(k, name, f"class_o{k}_{name}_s101", subset)
            for typ in ["neural", "tree"]:
                name = selections[typ]
                p2 = evaluate_one(
                    k,
                    f"selected_{typ}_seed202",
                    f"class_o{k}_{name}_s202",
                    subset,
                    dict(selected_candidate=name),
                )
                p = (pp[name] + p2) / 2
                output.append(
                    dict(
                        outer=k,
                        model=f"selected_{typ}_seed_average",
                        selected_candidate=name,
                        **metrics(subset.cardio.to_numpy(), p),
                    )
                )
                pp[f"selected_{typ}_seed_average"] = p
            for name, ens in selections["ensembles"].items():
                for kind, weights in [("convex", ens["weights"]), ("equal", ens["equal_weights"])]:
                    p = np.column_stack([pp[n] for n in ens["names"]]) @ weights
                    tag = f"ensemble_{name}_{kind}"
                    pp[tag] = p
                    output.append(dict(outer=k, model=tag, **metrics(subset.cardio.to_numpy(), p)))
            # All methods compared on the same strict subset, without selecting by these outcomes.
            valid = (
                subset[["age_years", "ap_hi", "ap_lo", "height", "weight", "bmi"]]
                .notna()
                .all(axis=1)
                .to_numpy()
            )
            for name in ["strict_reference", "v1_reference", "v2_reference", "shuffled_quality"]:
                output.append(
                    dict(
                        outer=k,
                        model=name + "_strict_common",
                        **metrics(subset.cardio.to_numpy()[valid], pp[name][valid]),
                    )
                )
            preds[k] = dict(
                rows=subset.index.to_numpy(),
                y=subset.cardio.to_numpy(),
                groups=subset.group.to_numpy(),
                predictions=pp,
            )
    else:
        for entry in read_json(PUB / "transfer_manifest.json"):
            k = entry["outer"]
            n = entry["positive_budget"]
            method = entry["method"]
            subset = df.loc[transfer_parts[k]["evaluation"]]
            p = evaluate_one(
                k,
                method,
                entry["run"],
                subset,
                dict(
                    positive_budget=n,
                    target_label_cost=entry["label_cost"]["total_distinct"],
                    selection_workflow_cost=entry["selection_workflow_cost"],
                ),
            )
            key = f"o{k}_p{n}"
            if key not in preds:
                preds[key] = dict(
                    rows=subset.index.to_numpy(),
                    y=subset.cardio.to_numpy(),
                    groups=subset.group.to_numpy(),
                    predictions={},
                )
            preds[key]["predictions"][method] = p
            if method == value["choices"][key]["method"]:
                output.append(
                    {
                        **output[-1],
                        "model": "selected_new_policy",
                        "target_label_cost": entry["selection_workflow_cost"],
                    }
                )
                preds[key]["predictions"]["selected_new_policy"] = p
    pd.DataFrame(output).to_csv(PUB / f"{track}_metrics.csv", index=False)
    joblib.dump(preds, PRIV / f"{track}_outer_predictions.joblib")
    print(track, "EVALUATED", len(output), flush=True)


def probe():
    torch.set_num_threads(4)
    df, _, _ = prepare()
    x = balanced(df, 100, 890)
    stop = balanced(df.loc[~df.index.isin(x.index)], 50, 891)
    report = dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        mps_available=torch.backends.mps.is_available(),
    )
    for i, kind in enumerate(["concat", "attention", "ft"]):
        cfg = {
            **CFG,
            "family": "neural",
            "architecture": kind,
            "blocks": 1,
            "token_width": 16,
            "epochs": 2,
            "patience": 2,
        }
        device = "mps" if report["mps_available"] and kind != "ft" else "cpu"
        b, r = fit_predictor(f"smoke_{kind}", x, stop, cfg, 89 + i, device)
        report[kind] = dict(
            device=device,
            finite=bool(np.isfinite(predict(b, stop)).all()),
            best_epoch=r["best_epoch"],
        )
    write_json(PUB / "environment.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "probe",
            "classification",
            "transfer",
            "evaluate_classification",
            "evaluate_transfer",
        ],
    )
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    if a.stage == "probe":
        probe()
    elif a.stage.startswith("evaluate_"):
        evaluate(a.stage.removeprefix("evaluate_"))
    else:
        globals()[a.stage](a.device)
