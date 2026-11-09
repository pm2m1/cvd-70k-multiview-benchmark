"""Resumable V3 fitting with integrity-checked private checkpoints."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import hashlib
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn

from cvd70k.training.engine import set_seed
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.engine import Stopper, probabilities
from cvd70k.v3.design import PRIV, PUB, digest


def state_hash(model):
    h = hashlib.sha256()
    for n, v in model.state_dict().items():
        h.update(n.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def loss_value(logits, y, cfg):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if cfg.get("loss") == "focal":
        p = torch.sigmoid(logits)
        pt = torch.where(y == 1, p, 1 - p)
        return ((1 - pt) ** 2 * bce).mean()
    loss = bce.mean()
    if cfg.get("loss") == "rank":
        pos = logits[y == 1]
        neg = logits[y == 0]
        if len(pos) and len(neg):
            # At most batch_size paired comparisons, deterministic permutation stream.
            pairs = min(256, len(pos) * len(neg))
            ip = torch.arange(pairs, device=logits.device) % len(pos)
            inn = (torch.arange(pairs, device=logits.device) * 7) % len(neg)
            loss = loss + cfg["rank_weight"] * nn.functional.softplus(neg[inn] - pos[ip]).mean()
    return loss


def train(model, x, y, xs, ys, cfg, seed, device="cpu", zero=False, penalty=None, projection=None):
    set_seed(seed)
    model.to(device)
    xx = torch.as_tensor(x, device=device)
    yy = torch.tensor(y, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    stopper = Stopper(cfg["stop"], cfg["patience"], zero)
    curve = []
    generator = torch.Generator().manual_seed(seed)
    start = time.perf_counter()

    def monitor(epoch, loss):
        p = probabilities(model, xs, device, 2048)
        r = dict(
            epoch=epoch,
            train_loss=loss,
            validation_ap=float(average_precision_score(ys, p)),
            validation_auroc=float(roc_auc_score(ys, p)),
            validation_bce=float(log_loss(ys, np.clip(p, 1e-7, 1 - 1e-7))),
        )
        curve.append(r)
        return stopper.update(r, model)

    monitor(0, None)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        accum = 0.0
        for idx in torch.randperm(len(xx), generator=generator).to(device).split(cfg["batch"]):
            opt.zero_grad(set_to_none=True)
            loss = loss_value(model(xx[idx]), yy[idx], cfg)
            if (
                isinstance(getattr(model, "gates", None), nn.Parameter)
                and model.gates.requires_grad
            ):
                loss = (
                    loss + cfg.get("gate_penalty", 0.0) * ((model.gates.sigmoid() - 0.5) ** 2).sum()
                )
            if penalty is not None:
                loss = loss + penalty(model)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective")
            loss.backward()
            opt.step()
            if projection is not None:
                projection(model)
            accum += float(loss.detach()) * len(idx)
        if monitor(epoch, accum / len(xx)):
            break
    model.load_state_dict(stopper.state)
    return model.cpu(), dict(
        best_epoch=stopper.epoch,
        epochs_completed=epoch,
        curve=curve,
        parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        fit_seconds=time.perf_counter() - start,
        training_device=device,
        initial_checkpoint_eligible=zero,
    )


def checkpoint(run, spec, fit):
    record = PUB / "trials" / f"{run}.json"
    path = PRIV / "models" / f"{run}.joblib"
    if record.exists():
        rec = read_json(record)
        if rec["spec_hash"] != digest(spec) or sha256(path) != rec["checkpoint_sha256"]:
            raise ValueError("Resume integrity failed: " + run)
        return joblib.load(path), rec
    completed = len(list((PUB / "trials").glob("*.json")))
    if completed >= 400:
        raise RuntimeError("Registered hard fit budget exhausted")
    start = time.perf_counter()
    try:
        bundle, info = fit()
    except Exception as exc:
        write_json(PUB / "failures" / f"{run}.json", dict(run=run, error=repr(exc), spec=spec))
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    rec = dict(
        run=run,
        spec=spec,
        spec_hash=digest(spec),
        checkpoint_sha256=sha256(path),
        elapsed_seconds=time.perf_counter() - start,
        **info,
    )
    write_json(record, rec)
    print(
        run, "best", info.get("best_epoch"), "seconds", round(rec["elapsed_seconds"], 1), flush=True
    )
    return bundle, rec
