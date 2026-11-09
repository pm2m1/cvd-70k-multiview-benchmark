"""Probability, threshold and cluster-paired bootstrap evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def calibration_bins(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    which = np.minimum((p * bins).astype(int), bins - 1)
    return [
        {
            "left": i / bins,
            "right": (i + 1) / bins,
            "n": int((which == i).sum()),
            "mean_probability": float(p[which == i].mean()) if (which == i).any() else None,
            "positive_fraction": float(y[which == i].mean()) if (which == i).any() else None,
        }
        for i in range(bins)
    ]


def metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Invalid labels, probabilities or threshold")
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    ece = sum(
        b["n"] * abs(b["mean_probability"] - b["positive_fraction"])
        for b in calibration_bins(y, p)
        if b["n"]
    ) / len(y)
    return {
        "ap": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "ece_10": float(ece),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": len(y),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
    }


def select_threshold(y, p, role="validation"):
    if role != "validation":
        raise ValueError("Threshold selection requires validation role")
    # Prespecified grid and balanced-accuracy objective; closest to .5 wins ties.
    candidates = np.arange(0.01, 1.0, 0.01)
    y, p = np.asarray(y), np.asarray(p)
    metrics(y, p)  # validate once, not once per threshold
    score = []
    for t in candidates:
        predicted = p >= t
        recall = predicted[y == 1].mean()
        specificity = (~predicted[y == 0]).mean()
        score.append(((recall + specificity) / 2, -abs(t - 0.5), -t, t))
    return float(max(score)[-1])


def threshold_grid(y, p):
    return [metrics(y, p, float(t)) for t in np.linspace(0, 1, 21)]


def paired_bootstrap(y, predictions, groups, reference, replicates=1000, seed=8021):
    """Same resampled feature-groups and paired seeds for every model.

    Resampling clusters handles unverified unique-person identifiers conservatively.
    Seed means are computed *within* each resample; this is not seed resampling.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in predictions}
    differences = {name: [] for name in predictions if name != reference}
    # AP via sorting once then weighted cumulative precision for each bootstrap.
    orders = {k: [np.argsort(-p, kind="stable") for p in v] for k, v in predictions.items()}

    def weighted_ap(p, order, w):
        yy, ww, pp = y[order], w[order], p[order]
        endpoint = np.r_[pp[1:] != pp[:-1], True]
        tp = np.cumsum(ww * yy)[endpoint]
        total = np.cumsum(ww)[endpoint]
        increments = np.diff(np.r_[0, tp])
        return float(
            np.sum(np.divide(tp, total, out=np.zeros_like(tp), where=total > 0) * increments)
            / max(1, np.sum(w * y))
        )

    for _ in range(replicates):
        gw = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = gw[inverse].astype(float)
        current = {}
        for name, ps in predictions.items():
            current[name] = np.mean(
                [weighted_ap(p, order, w) for p, order in zip(ps, orders[name], strict=True)]
            )
            values[name].append(current[name])
        for name in differences:
            differences[name].append(current[name] - current[reference])
    return {
        "replicates": replicates,
        "seed": seed,
        "resampling_unit": "modeled-feature group",
        "ap_ci": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in values.items()},
        "ap_difference_vs_" + reference: {
            k: {"mean": float(np.mean(v)), "ci95": np.quantile(v, [0.025, 0.975]).tolist()}
            for k, v in differences.items()
        },
    }
