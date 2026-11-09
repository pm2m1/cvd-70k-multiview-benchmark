"""Calibration, development-only operating points and OOF ensemble weights."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, log_loss

from cvd70k.evaluation.metrics import metrics


def logits(p):
    return logit(np.clip(p, 1e-7, 1 - 1e-7))


class Calibrator:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, y, p, role="calibration"):
        if role != "calibration":
            raise ValueError("Dedicated calibration role required")
        if self.kind == "none":
            return self
        if self.kind == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
            return self
        x = logits(p)
        if self.kind == "platt":
            fit = minimize(
                lambda a: log_loss(y, expit(a[0] * x + a[1]), labels=[0, 1]),
                [1.0, 0.0],
                bounds=[(1e-5, 20), (-20, 20)],
            )
            self.parameters = fit.x
        elif self.kind == "temperature":
            fit = minimize(
                lambda a: log_loss(y, expit(x / np.exp(a[0])), labels=[0, 1]),
                [0.0],
                bounds=[(-4, 4)],
            )
            self.parameters = fit.x
        else:
            raise ValueError(self.kind)
        if not fit.success and not np.isfinite(fit.fun):
            raise RuntimeError("Calibration failed")
        return self

    def transform(self, p):
        if self.kind == "none":
            return p
        if self.kind == "isotonic":
            return self.model.predict(p)
        if self.kind == "platt":
            return expit(self.parameters[0] * logits(p) + self.parameters[1])
        return expit(logits(p) / np.exp(self.parameters[0]))


def operating_points(y, p, role="calibration"):
    if role != "calibration":
        raise ValueError("Thresholds require calibration observations")
    y, p = np.asarray(y), np.asarray(p)
    grid = np.arange(0.01, 1, 0.01)
    statistics = []
    for t in grid:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        tn = int((~pred & (y == 0)).sum())
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        statistics.append(
            {
                "threshold": float(t),
                "ba": (recall + specificity) / 2,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )

    def best(rows, key):
        if not rows:
            return None
        return max(rows, key=lambda s: (s[key], -abs(s["threshold"] - 0.5), -s["threshold"]))[
            "threshold"
        ]

    return {
        "fixed_050": 0.5,
        "balanced_accuracy": best(statistics, "ba"),
        "f1": best(statistics, "f1"),
        "precision_075": best([s for s in statistics if s["precision"] >= 0.75], "recall"),
        "recall_080": best([s for s in statistics if s["recall"] >= 0.8], "precision"),
    }


def convex_weights(y, predictions, role="inner_oof"):
    if role != "inner_oof":
        raise ValueError("Ensemble weights require inner OOF predictions")
    names = list(predictions)
    matrix = np.column_stack([predictions[n] for n in names])
    if matrix.shape[0] != len(y) or np.isnan(matrix).any():
        raise ValueError("Incomplete OOF matrix")
    weights = []
    for ints in itertools.product(range(11), repeat=len(names) - 1):
        last = 10 - sum(ints)
        if last >= 0:
            weights.append(np.array([*ints, last]) / 10)
    chosen = max(weights, key=lambda w: average_precision_score(y, matrix @ w))
    return names, chosen


def calibration_diagnostic(y, p):
    x = logits(p)
    fit = minimize(
        lambda a: log_loss(y, expit(a[0] * x + a[1]), labels=[0, 1]),
        [1.0, 0.0],
        bounds=[(-20, 20), (-20, 20)],
    )
    return {"calibration_slope": float(fit.x[0]), "calibration_intercept": float(fit.x[1])}


def evaluated_metrics(y, p, threshold):
    return {**metrics(y, p, threshold), **calibration_diagnostic(y, p)}
