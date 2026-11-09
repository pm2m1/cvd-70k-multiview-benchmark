"""Prospectively fixed candidates and label roles (no evaluation imports)."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.data import development_data, folds

ROOT = Path.cwd()
PUB = ROOT / "results/v3/aggregate"
PRIV = ROOT / "private/v3"
BASE = "2a090b4f50a02c2eda89b8ee303102d608f7a390"
CFG = dict(
    embedding=16,
    fusion=[64, 32],
    dropout=0.0,
    optimizer="adamw",
    lr=0.001,
    weight_decay=0.0001,
    schedule="constant",
    epochs=120,
    patience=16,
    batch=256,
    stop="ap",
    block="missing",
    architecture="concat",
    loss="bce",
)


def candidates():
    specs = [
        ("v1_reference", dict(block="base", epochs=60, patience=8, batch=1024, stop="bce")),
        ("v2_reference", {}),
        ("age_spline", dict(block="spline")),
        ("nonlinear", dict(block="nonlinear")),
        ("quantile", dict(scale="quantile")),
        ("shuffled_quality", dict(block="shuffled")),
        ("strict_reference", dict(block="base", strict=True)),
        ("gated", dict(architecture="gated")),
        ("gated_regularized", dict(architecture="gated", gate_penalty=0.01, lr=0.0003)),
        ("uniform_gate", dict(architecture="uniform")),
        ("random_gate", dict(architecture="random")),
        ("residual", dict(architecture="residual")),
        ("depth_control", dict(architecture="depth_control")),
        ("attention", dict(architecture="attention")),
        ("attention_no_identity", dict(architecture="attention_no_identity")),
        ("pairwise", dict(architecture="pairwise")),
        ("flat_pairwise_matched", dict(architecture="flat", match="pairwise")),
        ("flat_attention_matched", dict(architecture="flat", match="attention")),
        ("focal", dict(loss="focal")),
        ("rank_01", dict(loss="rank", rank_weight=0.1)),
        ("rank_001", dict(loss="rank", rank_weight=0.01)),
        (
            "ft_reference",
            dict(architecture="ft", blocks=2, token_width=32, dropout=0.1, epochs=60, patience=12),
        ),
    ]
    result = {name: {**CFG, **changes, "family": "neural"} for name, changes in specs}
    for family in ["xgboost", "lightgbm", "catboost"]:
        for variant in [0, 1]:
            result[f"{family}_{variant}"] = {
                **CFG,
                "family": family,
                "variant": variant,
                "rounds": 1200,
                "depth": 3 if variant == 0 else 5,
                "tree_lr": 0.03 if variant == 0 else 0.05,
                "l2": 3.0 if variant == 0 else 10.0,
                "child": 20 if variant == 0 else 60,
            }
    return result


METHODS = [
    "head",
    "full",
    "logical",
    "random_matched",
    "full_cost",
    "full_sp_low",
    "full_sp_high",
    "logical_sp_low",
    "logical_sp_high",
    "soft_uniform",
    "soft_evidence",
    "soft_random",
    "soft_shuffled",
    "soft_learned",
    "reg_geometry",
    "reg_mmd",
    "reg_combined",
    "reg_shuffled",
    "view_hemo",
    "view_metabolic",
    "view_background",
    "fusion_both",
    "target_lr",
    "target_xgb",
    "view_anthro",
    "view_behaviour",
    "fusion1_only",
    "fusion2_only",
    "target_lr_cost",
    "target_xgb_cost",
]
NEW = [
    "full_sp_low",
    "full_sp_high",
    "logical_sp_low",
    "logical_sp_high",
    "soft_uniform",
    "soft_evidence",
    "soft_learned",
    "reg_geometry",
    "reg_mmd",
    "reg_combined",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable manifest changed: {path}")
    else:
        write_json(path, value)


def isolated(parts):
    values = list(parts.values())
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            if set(a.group) & set(b.group):
                raise ValueError("Group role leakage")


def balanced(df, n, seed):
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.permutation(df.index[df.cardio == c])[:n] for c in [0, 1]])
    if len(ids) != 2 * n or df.loc[ids].cardio.sum() != n:
        raise ValueError("Infeasible class budget")
    return df.loc[ids]


def label_cost(**roles):
    all_ids = set()
    counts = {}
    for name, df in roles.items():
        ids = set(df.index)
        counts[name] = {
            "n": len(ids),
            "positive": int(df.cardio.sum()),
            "negative": int((df.cardio == 0).sum()),
        }
        all_ids |= ids
    return {
        "roles": counts,
        "total_distinct": len(all_ids),
        "role_sum": sum(v["n"] for v in counts.values()),
        "reused_label_count": sum(v["n"] for v in counts.values()) - len(all_ids),
    }


def prepare():
    PUB.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    df = development_data()
    outer = folds(df, 3, 3300)
    classification = {}
    transfer = {}
    counts = []
    memberships = []

    def register(track, k, parts):
        isolated(parts)
        for role, x in parts.items():
            counts.append(
                dict(
                    track=track,
                    outer=k,
                    role=role,
                    n=len(x),
                    positive=int(x.cardio.sum()),
                    negative=int((x.cardio == 0).sum()),
                )
            )
            memberships.extend(dict(track=track, outer=k, role=role, row=int(r)) for r in x.index)
        return {r: x.index.to_numpy() for r, x in parts.items()}

    for k in range(3):
        inside = df.loc[outer != k]
        inner = folds(inside, 5, 3400 + k)
        classification[k] = register(
            "classification",
            k,
            {
                "fit": inside.loc[inner < 3],
                "stop": inside.loc[inner == 3],
                "select": inside.loc[inner == 4],
                "evaluation": df.loc[outer == k],
            },
        )
    source = df.loc[df.age_years < 55]
    target = df.loc[df.age_years >= 55]
    sf = folds(source, 10, 7301)
    tf = folds(target, 3, 3310)
    for k in range(3):
        inside = target.loc[tf != k]
        role = folds(inside, 10, 3500 + k)
        transfer[k] = register(
            "transfer",
            k,
            {
                "source_fit": source.loc[sf < 8],
                "source_stop": source.loc[sf == 8],
                "source_reference": source.loc[sf == 9],
                "fit": inside.loc[role < 4],
                "selector": inside.loc[(role >= 4) & (role < 7)],
                "stop": inside.loc[role == 7],
                "select": inside.loc[role >= 8],
                "evaluation": target.loc[tf == k],
            },
        )
        for n in [25, 250, 1000]:
            balanced(df.loc[transfer[k]["fit"]], n, 4000 + k)
            balanced(df.loc[transfer[k]["selector"]], n, 5000 + k)
        balanced(df.loc[transfer[k]["stop"]], 50, 6000 + k)
        balanced(df.loc[transfer[k]["select"]], 50, 7000 + k)
    membership = pd.DataFrame(memberships)
    path = PRIV / "roles.csv"
    if path.exists() and path.read_text() != membership.to_csv(index=False):
        raise ValueError("Role memberships changed")
    path.write_text(membership.to_csv(index=False))
    immutable(
        PUB / "roles.json",
        {
            "eligible_partition": "V1 train only",
            "n": len(df),
            "sha256": sha256(path),
            "counts": counts,
        },
    )
    return df, classification, transfer


def preserve():
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE], text=True
    ).splitlines()
    for f in files:
        old = subprocess.check_output(["git", "show", f"{BASE}:{f}"])
        if (ROOT / f).read_bytes() != old:
            raise ValueError("Historical file altered: " + f)
    archive = Path.home() / "Downloads/archive(1).zip"
    expected = read_json(ROOT / "results/aggregate/dataset_audit.json")["archive_sha256"]
    if sha256(archive) != expected:
        raise ValueError("Archive changed")
    immutable(
        PUB / "historical_integrity.json",
        {
            "v1": "88c8020072fa3bf48d627bd3c2216d0b8e683181",
            "v2": BASE,
            "tracked_files_verified": len(files),
            "archive_sha256": expected,
            "tracked_manifest_sha256": digest({f: sha256(ROOT / f) for f in files}),
        },
    )


if __name__ == "__main__":
    preserve()
    prepare()
    immutable(
        ROOT / "configs/v3/study.json",
        {
            "candidates": candidates(),
            "methods": METHODS,
            "new_policies": NEW,
            "positive_budgets": [25, 250, 1000],
            "seeds": [101, 202, 303],
            "fit_limit": 400,
            "planned_fits": 375,
            "classification_fits": 90,
            "transfer_fits": 282,
            "smoke_allowance": 3,
            "primary": "P1000 development-selected new policy minus full_cost AP",
            "practical_ap_margin": 0.005,
            "bootstrap_replicates": 1000,
        },
    )
    print("V3 design and memberships registered")
