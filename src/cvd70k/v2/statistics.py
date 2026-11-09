"""Paired group bootstrap conditional on fitted models and fixed outer folds."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranked_cache(y, p):
    order = np.argsort(-p, kind="stable")
    pp = np.asarray(p)[order]
    ends = np.flatnonzero(np.r_[pp[1:] != pp[:-1], True])
    return np.asarray(y)[order], order, ends


def weighted_scores(cache, weights):
    y, order, ends = cache
    w = weights[order]
    positives = np.cumsum(w * y)[ends]
    negatives = np.cumsum(w * (1 - y))[ends]
    pos = np.diff(np.r_[0.0, positives])
    neg = np.diff(np.r_[0.0, negatives])
    total = positives + negatives
    P = positives[-1]
    N = negatives[-1]
    if P == 0 or N == 0:
        return np.nan, np.nan
    ap = (
        np.sum(np.divide(positives, total, out=np.zeros_like(positives), where=total > 0) * pos) / P
    )
    # Descending scores: each positive defeats negatives below and ties half.
    auc = np.sum(pos * (N - negatives + 0.5 * neg)) / (P * N)
    return float(ap), float(auc)


def paired_interval(y, a, b, groups, folds=None, replicates=1000, seed=9421):
    """Mean within-fold, within-seed metric difference A-B; fixed-fold bootstrap.

    Each draw resamples feature groups within each outer fold; no independence of
    folds is asserted. Seed predictions remain paired and are not independent people.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    folds = np.zeros(len(y), dtype=int) if folds is None else np.asarray(folds)
    a, b = [np.asarray(p) for p in a], [np.asarray(p) for p in b]
    if len(a) != len(b):
        raise ValueError("Paired seeds required")
    caches = []
    observed = []
    for fold in np.unique(folds):
        idx = np.flatnonzero(folds == fold)
        yy = y[idx]
        unique, inverse = np.unique(groups[idx], return_inverse=True)
        ac = [ranked_cache(yy, p[idx]) for p in a]
        bc = [ranked_cache(yy, p[idx]) for p in b]
        caches.append((unique, inverse, ac, bc))
        observed.append(
            np.mean(
                [
                    [
                        average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]),
                        roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx]),
                    ]
                    for pa, pb in zip(a, b, strict=True)
                ],
                axis=0,
            )
        )
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        values = []
        for unique, inverse, ac, bc in caches:
            gw = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
            weights = gw[inverse].astype(float)
            values.append(
                np.mean(
                    [
                        np.subtract(weighted_scores(ca, weights), weighted_scores(cb, weights))
                        for ca, cb in zip(ac, bc, strict=True)
                    ],
                    axis=0,
                )
            )
        samples.append(np.mean(values, axis=0))
    interval = np.nanquantile(samples, [0.025, 0.975], axis=0)
    point = np.mean(observed, axis=0)
    return {
        "ap_difference": float(point[0]),
        "auroc_difference": float(point[1]),
        "ap_ci95": interval[:, 0].tolist(),
        "auroc_ci95": interval[:, 1].tolist(),
        "replicates": replicates,
        "unit": "modeled-feature group, stratified by fixed outer fold",
        "scope": "conditional on fitted models; exploratory; not external validation or independent-fold inference",
    }
