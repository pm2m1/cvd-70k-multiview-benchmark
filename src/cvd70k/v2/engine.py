"""V2 fit/checkpoint engine; no dependency on historical test executors."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from cvd70k.data.sampling import oversample_indices
from cvd70k.training.engine import set_seed
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.data import Features
from cvd70k.v2.models import count, model_for

ROOT = Path.cwd()
PRIVATE = ROOT / "private/v2"
TRIALS = ROOT / "results/v2/aggregate/trials"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Stopper:
    def __init__(self, objective="ap", patience=16, zero=False):
        if objective not in ["bce", "ap", "auroc"]:
            raise ValueError(objective)
        self.objective, self.patience, self.zero = objective, patience, zero
        self.best = None
        self.state = None
        self.epoch = None
        self.bad = 0

    def update(self, record, model):
        epoch = record["epoch"]
        if epoch == 0 and not self.zero:
            return False
        value = record["validation_" + self.objective]
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite checkpoint objective")
        signed = -value if self.objective == "bce" else value
        delta = 1e-5 if self.objective == "bce" else 1e-4
        if self.best is None or signed > self.best + delta:
            self.best, self.bad, self.epoch = signed, 0, epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience


def probabilities(model, x, device="cpu", batch=4096):
    model.to(device).eval()
    output = []
    with torch.no_grad():
        for chunk in torch.as_tensor(x, device=device).split(batch):
            output.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(output)


def train_network(
    model,
    x,
    y,
    xv,
    yv,
    cfg,
    seed,
    device="cpu",
    fixed_epochs=None,
    intervention="standard",
    zero=False,
):
    set_seed(seed)
    model.to(device)
    indices = oversample_indices(y, seed) if intervention == "resampled" else np.arange(len(y))
    xx = torch.as_tensor(x[indices], device=device)
    yy = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
    weight = float((y == 0).sum() / (y == 1).sum()) if intervention == "weighted" else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_cls = torch.optim.Adam if cfg["optimizer"] == "adam" else torch.optim.AdamW
    optimizer = optimizer_cls(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs = fixed_epochs or cfg["epochs"]
    scheduler = None
    if cfg["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg["schedule"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
    stopper = Stopper(cfg["stop"], cfg["patience"], zero)
    curve = []
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()

    def monitor(epoch, training_loss):
        rec = {"epoch": epoch, "train_loss": training_loss, "lr": optimizer.param_groups[0]["lr"]}
        if xv is not None:
            p = probabilities(model, xv, device, batch=2048)
            rec.update(
                validation_bce=float(log_loss(yv, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
                validation_ap=float(average_precision_score(yv, p)),
                validation_auroc=float(roc_auc_score(yv, p)),
            )
        return rec

    first = monitor(0, None)
    curve.append(first)
    if xv is not None:
        stopper.update(first, model)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(xx), generator=generator).to(device).split(cfg["batch"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xx[idx]), yy[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(idx)
        rec = monitor(epoch, total / len(xx))
        curve.append(rec)
        if scheduler is not None:
            if cfg["schedule"] == "plateau":
                scheduler.step(rec.get("validation_bce", rec["train_loss"]))
            else:
                scheduler.step()
        if xv is not None and stopper.update(rec, model):
            break
    if xv is not None:
        model.load_state_dict(stopper.state)
    info = {
        "best_epoch": stopper.epoch if xv is not None else epochs,
        "epochs_completed": epoch,
        "parameter_count": count(model),
        "fit_seconds": time.perf_counter() - start,
        "curve": curve,
        "pos_weight": weight,
        "fit_positive": int((y[indices] == 1).sum()),
        "fit_negative": int((y[indices] == 0).sum()),
    }
    return model.to("cpu"), info


def tree_process(request):
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PRIVATE, prefix="tree_") as folder:
        req = Path(folder) / "request.joblib"
        response = Path(folder) / "response.joblib"
        request["response"] = str(response)
        joblib.dump(request, req)
        subprocess.run(
            [sys.executable, "-m", "cvd70k.v2.tree_worker", str(req)], check=True, timeout=600
        )
        return joblib.load(response)


def fit_trial(
    run_id, train, validation, cfg, seed, device, fixed_epochs=None, intervention="standard"
):
    spec = {
        "run_id": run_id,
        "config": cfg,
        "seed": seed,
        "fixed_epochs": fixed_epochs,
        "intervention": intervention,
        "train_membership_sha256": hashlib.sha256(train.index.to_numpy().tobytes()).hexdigest(),
        "validation_membership_sha256": hashlib.sha256(
            validation.index.to_numpy().tobytes()
        ).hexdigest()
        if validation is not None
        else None,
    }
    metadata = TRIALS / f"{run_id}.json"
    path = PRIVATE / "models" / f"{run_id}.joblib"
    if metadata.exists():
        record = read_json(metadata)
        if record["specification"] != spec or sha256(path) != record["checkpoint_sha256"]:
            raise ValueError("Stale/corrupt trial " + run_id)
        return record
    prep = Features(cfg["features"], cfg["scale"]).fit(train)
    x = prep.transform(train)
    y = train.cardio.to_numpy()
    xv = prep.transform(validation) if validation is not None else None
    yv = validation.cardio.to_numpy() if validation is not None else None
    family = cfg["family"]
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if family in ["multiview", "mlp", "ft_transformer"]:
        set_seed(seed)
        model = model_for(prep, cfg)
        model, info = train_network(
            model, x, y, xv, yv, cfg, seed, device, fixed_epochs, intervention
        )
        bundle = {"preprocessor": prep, "state": model.state_dict(), "config": cfg}
        if validation is not None:
            p = probabilities(model, xv)
    elif family == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier

        from cvd70k.v2.tabnet_support import AveragePrecision

        model = TabNetClassifier(
            n_d=cfg["decision_width"],
            n_a=cfg["decision_width"],
            n_steps=cfg["steps"],
            lambda_sparse=cfg["lambda_sparse"],
            optimizer_params={"lr": cfg["lr"]},
            seed=seed,
            device_name=device,
            verbose=0,
        )
        kwargs = {
            "X_train": x,
            "y_train": y,
            "max_epochs": fixed_epochs or cfg["epochs"],
            "patience": 0 if validation is None else cfg["patience"],
            "batch_size": cfg["batch"],
            "virtual_batch_size": 128,
            "num_workers": 0,
            "drop_last": False,
            "compute_importance": True,
        }
        if validation is not None:
            kwargs.update(eval_set=[(xv, yv)], eval_metric=[AveragePrecision])
        model.fit(**kwargs)
        importance = model.feature_importances_.tolist()
        # Local checkpoint only; established implementation retains its architecture/config.
        model.network.to("cpu")
        model.device = torch.device("cpu")
        if hasattr(model, "_optimizer"):
            for state in model._optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cpu()
        bundle = {"preprocessor": prep, "estimator": model, "config": cfg}
        history = dict(model.history.history)
        info = {
            "best_epoch": int(model.best_epoch + 1) if validation is not None else fixed_epochs,
            "epochs_completed": len(history["loss"]),
            "parameter_count": count(model.network),
            "fit_seconds": time.perf_counter() - start,
            "curve": [
                {
                    "epoch": i + 1,
                    "train_loss": float(v),
                    "validation_ap": float(history["val_0_ap"][i])
                    if "val_0_ap" in history
                    else None,
                }
                for i, v in enumerate(history["loss"])
            ],
            "aggregate_feature_importance": importance,
        }
        if validation is not None:
            p = model.predict_proba(xv)[:, 1]
    else:
        native = PRIVATE / "models" / f"{run_id}.native.joblib"
        response = tree_process(
            {
                "operation": "fit",
                "family": family,
                "config": cfg,
                "seed": seed,
                "x": x,
                "y": y,
                "xv": xv,
                "yv": yv,
                "fixed_epochs": fixed_epochs,
                "model_path": str(native),
            }
        )
        info = response["info"]
        bundle = {
            "preprocessor": prep,
            "native": str(native.relative_to(ROOT)),
            "native_sha256": sha256(native),
            "config": cfg,
        }
        if validation is not None:
            p = response["probability"]
    joblib.dump(bundle, path, compress=3)
    record = {
        "specification": spec,
        "training": info,
        "train_n": len(train),
        "train_positive": int(y.sum()),
        "checkpoint_sha256": sha256(path),
        "status": "completed",
    }
    if validation is not None:
        from cvd70k.evaluation.metrics import metrics

        record["validation"] = metrics(yv, p)
        pred = PRIVATE / "inner_predictions" / f"{run_id}.npz"
        pred.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred, probability=p)
    write_json(metadata, record)
    print(run_id, "completed", round(info["fit_seconds"], 2), "seconds", flush=True)
    return record


def predict_trial(run_id, df, device="cpu"):
    path = PRIVATE / "models" / f"{run_id}.joblib"
    record = read_json(TRIALS / f"{run_id}.json")
    if sha256(path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint mismatch")
    bundle = joblib.load(path)
    prep = bundle["preprocessor"]
    x = prep.transform(df)
    cfg = bundle["config"]
    if "state" in bundle:
        model = model_for(prep, cfg)
        model.load_state_dict(bundle["state"])
        return probabilities(model, x, device)
    if "estimator" in bundle:
        return bundle["estimator"].predict_proba(x)[:, 1]
    native = ROOT / bundle["native"]
    if sha256(native) != bundle["native_sha256"]:
        raise ValueError("Native checkpoint mismatch")
    return tree_process({"operation": "predict", "x": x, "model_path": str(native)})["probability"]
