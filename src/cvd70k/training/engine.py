"""Deterministic in-memory PyTorch training with restored early stopping."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss
from torch import nn

from cvd70k.data.sampling import oversample_indices
from cvd70k.models.networks import FiveViewMLP, FlatMLP, parameter_count


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-5):
        if patience < 1:
            raise ValueError("Positive patience required")
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad, self.epoch, self.state = float("inf"), 0, None, None

    def update(self, loss, model, epoch):
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        if loss < self.best - self.min_delta:
            self.best, self.bad, self.epoch = loss, 0, epoch
            self.state = copy.deepcopy(model.state_dict())
        else:
            self.bad += 1
        return self.bad >= self.patience


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_loss(y, intervention, device):
    y = np.asarray(y)
    if intervention not in ["standard", "weighted", "resampled"]:
        raise ValueError(intervention)
    weight = float((y == 0).sum() / (y == 1).sum()) if intervention == "weighted" else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device)), weight


def build_model(kind, dims, cfg):
    if kind == "mlp":
        return FlatMLP(sum(dims), tuple(cfg["flat_hidden"]))
    return FiveViewMLP(dims, cfg["embedding"], tuple(cfg["fusion"]))


def predict_nn(model, x, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(x, device=device))).cpu().numpy()


def train_nn(x, y, xv, yv, kind, dims, cfg, seed, intervention="standard", device="cpu"):
    set_seed(seed)
    model = build_model(kind, dims, cfg).to(device)
    criterion, weight = make_loss(y, intervention, device)
    # Every original observation appears once; minority repeats augment it to 50/50.
    indices = oversample_indices(y, seed) if intervention == "resampled" else np.arange(len(y))
    xt = torch.as_tensor(x[indices], device=device)
    yt = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
    validation = torch.as_tensor(xv, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    stopper = EarlyStopping(cfg["patience"])
    curve = []
    start = time.perf_counter()
    # Scratch training: epoch 0 logged, but not eligible. No pretrained model here.
    p0 = predict_nn(model, xv, device)
    curve.append(
        {
            "epoch": 0,
            "train_objective": None,
            "validation_loss": float(log_loss(yv, np.clip(p0, 1e-7, 1 - 1e-7))),
            "validation_ap": float(average_precision_score(yv, p0)),
        }
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        total = 0.0
        for index in order.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[index]), yt[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(index)
        model.eval()
        with torch.no_grad():
            logits = model(validation)
            # Unweighted natural-prevalence validation BCE for every intervention.
            vl = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(yv, dtype=torch.float32, device=device)
            ).item()
            pv = torch.sigmoid(logits).cpu().numpy()
        curve.append(
            {
                "epoch": epoch,
                "train_objective": total / len(xt),
                "validation_loss": vl,
                "validation_ap": float(average_precision_score(yv, pv)),
            }
        )
        if stopper.update(vl, model, epoch):
            break
    model.load_state_dict(stopper.state)
    duration = time.perf_counter() - start
    return model, {
        "duration_seconds": duration,
        "best_epoch": stopper.epoch,
        "epochs_run": epoch,
        "parameter_count": parameter_count(model),
        "pos_weight": weight,
        "fit_examples_per_epoch": len(indices),
        "curve": curve,
    }
