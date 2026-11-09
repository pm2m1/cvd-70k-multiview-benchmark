"""Independent DMV-MAT-inspired evidence and seven-module selection rules.

No MMD alignment loss. Full selector labels are disjoint from target fitting/stopping.
Mathematical baseline: the public Phase 1 calibrated logical policy.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.distance import cdist

from cvd70k.v2.models import MODULES


def validate_pair(source, target):
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != target.shape[1]
        or min(len(source), len(target)) < 2
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Finite matching representation matrices required")


def gaussian_kernel(x, bandwidths=(0.5, 1.0, 2.0, 4.0)):
    if any(b <= 0 for b in bandwidths):
        raise ValueError("Positive bandwidths required")
    distance = cdist(x, x, "sqeuclidean")
    return sum(np.exp(-distance / (2 * b * b)) for b in bandwidths) / len(bandwidths)


def multi_mmd(source, target, normalization="pooled", cap=128, seed=0):
    validate_pair(source, target)
    rng = np.random.default_rng(seed)
    x, y = source.astype(float), target.astype(float)
    if normalization == "source":
        center = x.mean(0)
        radius = max(float(np.sqrt(((x - center) ** 2).sum(1).mean())), 1e-8 * np.sqrt(x.shape[1]))
        x, y = (x - center) / radius, (y - center) / radius
    x = x[rng.permutation(len(x))[:cap]]
    y = y[rng.permutation(len(y))[:cap]]
    pooled = np.vstack([x, y])
    if normalization == "pooled":
        pooled -= pooled.mean(0)
        pooled /= max(float(np.sqrt((pooled**2).sum(1).mean())), 1e-8 * np.sqrt(pooled.shape[1]))
    k = gaussian_kernel(pooled)
    n, m = len(x), len(y)
    w = np.r_[np.full(n, 1 / n), np.full(m, -1 / m)]
    return max(0.0, float(w @ k @ w))


def permutation_mmd(source, target, permutations=199, cap=128, seed=0):
    validate_pair(source, target)
    if permutations < 9 or cap < 2:
        raise ValueError("Insufficient permutation/sample budget")
    rng = np.random.default_rng(seed)
    x = source[rng.permutation(len(source))[:cap]].astype(float)
    y = target[rng.permutation(len(target))[:cap]].astype(float)
    pooled = np.vstack([x, y])
    pooled -= pooled.mean(0)
    pooled /= max(float(np.sqrt((pooled**2).sum(1).mean())), 1e-8 * np.sqrt(pooled.shape[1]))
    kernel = gaussian_kernel(pooled)
    n, m = len(x), len(y)
    weights = np.full((permutations + 1, n + m), -1 / m)
    weights[0, :n] = 1 / n
    for j in range(1, permutations + 1):
        weights[j, rng.permutation(n + m)[:n]] = 1 / n
    values = np.maximum(np.sum((weights @ kernel) * weights, axis=1), 0)
    sd = float(values[1:].std(ddof=1))
    return {
        "mmd": float(values[0]),
        "domain_p": float((1 + (values[1:] >= values[0] - 1e-12).sum()) / (permutations + 1)),
        "domain_status": "valid" if sd > 1e-12 else "degenerate_null",
        "permutations": permutations,
        "mmd_source_n": n,
        "mmd_target_n": m,
        "null_mean": float(values[1:].mean()),
        "null_sd": sd,
    }


def discriminability(features, labels):
    counts = [int((labels == c).sum()) for c in [0, 1]]
    if min(counts) < 2:
        return None, "insufficient_class_samples"
    scatter = (
        sum(
            np.linalg.norm(features[labels == c] - features[labels == c].mean(0), axis=1).mean()
            for c in [0, 1]
        )
        / 2
    )
    if not np.isfinite(scatter) or scatter <= 1e-8:
        return None, "degenerate_scatter"
    between = np.linalg.norm(features[labels == 1].mean(0) - features[labels == 0].mean(0))
    return float(between / (scatter + 1e-8)), "valid"


def stratified_cap(features, labels, cap, seed):
    if cap is None or len(labels) <= cap:
        return features, labels
    rng = np.random.default_rng(seed)
    indices = []
    for c in [0, 1]:
        group = np.flatnonzero(labels == c)
        indices.extend(rng.permutation(group)[: cap // 2])
    return features[indices], labels[indices]


def relative_geometry(source, sy, target, ty, bootstraps=199, seed=0, geometry_cap=None):
    validate_pair(source, target)
    sy, ty = np.asarray(sy), np.asarray(ty)
    if (
        sy.shape != (len(source),)
        or ty.shape != (len(target),)
        or not np.isin(np.r_[sy, ty], [0, 1]).all()
    ):
        raise ValueError("Binary matching labels required")
    source, sy = stratified_cap(source, sy, geometry_cap, seed)
    target, ty = stratified_cap(target, ty, geometry_cap, seed + 1)
    ds, status_s = discriminability(source, sy)
    dt, status_t = discriminability(target, ty)
    result = {
        "source_discriminability": ds,
        "target_discriminability": dt,
        "R": None,
        "U": None,
        "lower": None,
        "source_positive": int(sy.sum()),
        "source_negative": int((sy == 0).sum()),
        "target_positive": int(ty.sum()),
        "target_negative": int((ty == 0).sum()),
        "geometry_status": "valid",
        "bootstrap_valid_fraction": None,
    }
    if ds is None or dt is None:
        result["geometry_status"] = f"source:{status_s};target:{status_t}"
        return result
    if min(int(sy.sum()), int((sy == 0).sum()), int(ty.sum()), int((ty == 0).sum())) < 5:
        result["geometry_status"] = "insufficient_class_samples"
        return result
    if ds < 0.1:
        result["geometry_status"] = "weak_source_geometry"
        return result
    s = source.astype(float)
    t = target.astype(float)
    sd = s[sy == 1].mean(0) - s[sy == 0].mean(0)
    td = t[ty == 1].mean(0) - t[ty == 0].mean(0)
    direction = sd / np.linalg.norm(sd)

    def scatter(x, y):
        return sum(np.linalg.norm(x[y == c] - x[y == c].mean(0), axis=1).mean() for c in [0, 1]) / 2

    ss, st = scatter(s, sy), scatter(t, ty)
    result["R"] = float((td @ direction / st) / (sd @ direction / ss))
    result["orientation_cosine"] = (
        float(td @ direction / np.linalg.norm(td)) if np.linalg.norm(td) > 1e-12 else None
    )
    rng = np.random.default_rng(seed)

    def contrast(x, y):
        projected = x @ direction
        means = []
        for c in [0, 1]:
            group = projected[y == c]
            means.append(group[rng.integers(len(group), size=(bootstraps, len(group)))].mean(1))
        return means[1] - means[0]

    bs = contrast(s, sy) / ss
    bt = contrast(t, ty) / st
    valid = bs > 0.02
    result["bootstrap_valid_fraction"] = float(valid.mean())
    if valid.mean() < 0.95:
        result["geometry_status"] = "uncertain_source_geometry"
        return result
    ratio = bt[valid] / bs[valid]
    result["lower"] = float(np.quantile(ratio, 0.05))
    result["U"] = float(np.quantile(ratio, 0.95))
    return result


def evidence(source, sy, target, ty, seed=0, mmd_cap=128, geometry_cap=None):
    return {
        name: {
            **permutation_mmd(source[name], target[name], cap=mmd_cap, seed=seed + i),
            **relative_geometry(
                source[name], sy, target[name], ty, seed=seed + 1000 + i, geometry_cap=geometry_cap
            ),
            "legacy_mmd": multi_mmd(source[name], target[name], "source", cap=128, seed=seed + i),
        }
        for i, name in enumerate(MODULES)
    }


def calibrated_mask(evidence, policy="logical"):
    if policy not in ["logical", "threshold"]:
        raise ValueError(policy)
    mask = {}
    reasons = {}
    for name, record in evidence.items():
        if record["geometry_status"] != "valid":
            mask[name] = False
            reasons[name] = "ABSTAIN:" + record["geometry_status"]
            continue
        r, u = record["R"], record["U"]
        if r is None or u is None or not np.isfinite([r, u]).all():
            raise ValueError("Invalid retention bounds")
        strong = u < 0.75 if policy == "logical" else r < 0.5 and u < 1
        corroborated = (
            record["domain_status"] == "valid" and record["domain_p"] <= 0.05 and r < 0.8 and u < 1
        )
        mask[name] = bool(strong or corroborated)
        reasons[name] = (
            "predictive_degradation"
            if strong
            else "domain_and_geometry"
            if corroborated
            else "SHARE"
        )
    return mask, reasons


def legacy_mask(evidence):
    def normalize(values):
        a = np.asarray(values)
        span = a.max() - a.min()
        return (a - a.min()) / span if span > 1e-12 else np.full(len(a), 0.5)

    names = sorted(evidence)
    m = normalize([evidence[n]["legacy_mmd"] for n in names])
    d = normalize([evidence[n]["target_discriminability"] or 0.0 for n in names])
    scores = dict(zip(names, 0.65 * m + 0.35 * (1 - d), strict=True))
    chosen = sorted(names, key=lambda n: (-scores[n], n))[: round(len(names) * 0.5)]
    return {n: n in chosen for n in names}


def set_mask(model, mask):
    if set(mask) != set(MODULES) or any(type(v) is not bool for v in mask.values()):
        raise ValueError("Complete Boolean module mask required")
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.selectable_modules().items():
        if mask[name]:
            for p in module.parameters():
                p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def snapshot_frozen(model, mask):
    return {
        n: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        for n, m in model.selectable_modules().items()
        if not mask[n]
    }


def verify_frozen(model, snapshot):
    import torch

    for name, state in snapshot.items():
        now = model.selectable_modules()[name].state_dict()
        for key, value in state.items():
            if not torch.equal(now[key].detach().cpu(), value):
                raise RuntimeError("Frozen parameter changed: " + name + ":" + key)
    return True


def matched_random(model, reference, seed):
    counts = {
        n: sum(p.numel() for p in m.parameters()) for n, m in model.selectable_modules().items()
    }
    budget = sum(counts[n] for n in counts if reference[n])
    names = list(counts)
    choices = list(itertools.product([False, True], repeat=7))
    errors = [
        abs(sum(counts[n] for n, b in zip(names, bits, strict=True) if b) - budget)
        for bits in choices
    ]
    indices = np.flatnonzero(np.array(errors) == min(errors))
    picked = int(np.random.default_rng(seed).choice(indices))
    result = dict(zip(names, choices[picked], strict=True))
    return result, {
        "candidate_count": len(indices),
        "parameter_error": min(errors),
        "equals_logical": result == reference,
    }


def external_gate(facts):
    required = [
        "source_identity",
        "target_identity",
        "permissions",
        "endpoint_crosswalk",
        "time_horizon",
        "baseline_features",
        "outcome_observation",
        "positive_counts",
        "missingness",
        "label_budget",
        "test_isolation",
        "cohort_overlap",
        "calibration_population",
    ]
    missing = [k for k in required if not facts.get(k)]
    return {
        "eligible": not missing,
        "missing": missing,
        "status": "ELIGIBLE" if not missing else "EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED",
    }
