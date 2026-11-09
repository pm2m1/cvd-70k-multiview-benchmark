"""Without-replacement class-conditional draws after split creation."""

from __future__ import annotations

import numpy as np


def feasible_plans(y, prevalences=(0.5, 0.25, 0.15, 0.0742)):
    y = np.asarray(y)
    p, n = int((y == 1).sum()), int((y == 0).sum())
    total = int(min(min(p / q, n / (1 - q)) for q in prevalences))
    positives = int(min(p, min(n * q / (1 - q) for q in prevalences)))
    plans = []
    for analysis in ["fixed_total", "fixed_positive"]:
        for q in prevalences:
            if analysis == "fixed_total":
                npos = int(round(total * q))
                nneg = total - npos
            else:
                npos = positives
                nneg = int(round(positives * (1 - q) / q))
            if npos > p or nneg > n:
                raise ValueError("Infeasible plan")
            plans.append(
                {
                    "analysis": analysis,
                    "requested_prevalence": q,
                    "positive": npos,
                    "negative": nneg,
                    "n": npos + nneg,
                    "actual_prevalence": npos / (npos + nneg),
                }
            )
    return plans


def sample_indices(y, positive, negative, seed):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0))
    if positive < 1 or negative < 1 or positive > len(pos) or negative > len(neg):
        raise ValueError("Infeasible class counts")
    # Same permutation seed yields nested class-specific subsets across conditions.
    indices = np.concatenate([pos[:positive], neg[:negative]])
    return rng.permutation(indices)


def oversample_indices(y, seed):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if min(len(pos), len(neg)) == 0:
        raise ValueError("Both classes required")
    size = max(len(pos), len(neg))

    def expand(indices):
        return np.concatenate([indices, rng.choice(indices, size - len(indices), replace=True)])

    return rng.permutation(np.concatenate([expand(pos), expand(neg)]))
