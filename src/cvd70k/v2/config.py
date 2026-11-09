"""Finite prespecified candidate library; no outer-fold-driven expansion."""

from __future__ import annotations

import copy


def candidates(family):
    base = {
        "family": family,
        "name": "v1_equivalent",
        "embedding": 16,
        "fusion": [64, 32],
        "hidden": [96, 64],
        "dropout": 0.0,
        "lr": 0.0003 if family == "mlp" else 0.001,
        "batch": 1024,
        "epochs": 60,
        "patience": 8,
        "stop": "bce",
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "schedule": "constant",
        "features": "base",
        "scale": "standard",
    }
    if family in ["mlp", "multiview"]:
        settings = [
            {},
            {"name": "epochs120", "epochs": 120},
            {"name": "epochs200", "epochs": 200},
            {"name": "patience16", "epochs": 120, "patience": 16},
            {"name": "patience24", "epochs": 200, "patience": 24},
            {"name": "ap_stop", "epochs": 120, "patience": 16, "stop": "ap"},
            {"name": "auroc_stop", "epochs": 120, "patience": 16, "stop": "auroc"},
            {"name": "batch256", "epochs": 120, "patience": 16, "stop": "ap", "batch": 256},
        ]
        core = {**base, **settings[7]}
        extras = [
            {"name": "width32", "embedding": 32, "fusion": [128, 64], "hidden": [192, 128]},
            {"name": "width64", "embedding": 64, "fusion": [256, 128], "hidden": [384, 256]},
            {
                "name": "regularized32",
                "embedding": 32,
                "fusion": [128, 64],
                "hidden": [192, 128],
                "dropout": 0.1,
                "lr": 0.0003,
                "weight_decay": 0.001,
                "schedule": "cosine",
            },
            {
                "name": "adam_plateau",
                "optimizer": "adam",
                "lr": 0.002,
                "weight_decay": 0.0,
                "dropout": 0.2,
                "schedule": "plateau",
            },
            {
                "name": "slow_regularized",
                "lr": 0.0001,
                "weight_decay": 1e-5,
                "dropout": 0.3,
                "batch": 512,
                "schedule": "cosine",
            },
            {"name": "fast_small_batch", "lr": 0.003, "batch": 128, "dropout": 0.1},
            {"name": "derived_features", "features": "derived"},
            {"name": "missing_indicators", "features": "missing"},
            {"name": "combined_features", "features": "both"},
            {"name": "robust_combined", "features": "both", "scale": "robust"},
        ]
        return [{**base, **c} for c in settings] + [{**core, **c} for c in extras]
    if family == "ft_transformer":
        return [
            {
                **base,
                "name": f"ft{i}",
                "epochs": 120,
                "patience": 16,
                "stop": "ap",
                "batch": 512,
                "lr": lr,
                "token_width": width,
                "blocks": blocks,
                "dropout": drop,
            }
            for i, (width, blocks, lr, drop) in enumerate(
                [(32, 2, 0.0003, 0.1), (64, 2, 0.0003, 0.2), (32, 3, 0.001, 0.1)]
            )
        ]
    if family == "tabnet":
        return [
            {
                **base,
                "name": f"tabnet{i}",
                "epochs": 120,
                "patience": 16,
                "stop": "ap",
                "batch": 1024,
                "lr": lr,
                "decision_width": width,
                "steps": steps,
                "lambda_sparse": sparse,
            }
            for i, (width, steps, lr, sparse) in enumerate(
                [(8, 3, 0.02, 0.001), (16, 5, 0.01, 0.0001)]
            )
        ]
    if family == "logistic":
        return [{**base, "name": f"C{c}", "C": c} for c in [0.1, 1.0]]
    if family == "random_forest":
        return [{**base, "name": f"depth{d}", "depth": d} for d in [8, None]]
    if family == "xgboost":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1711))
        result = []
        # Reproducible nonadaptive Optuna candidate schedule: no cross-outer information.
        for i in range(10):
            t = study.ask()
            hp = {
                "max_depth": t.suggest_int("max_depth", 2, 6),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample": t.suggest_float("subsample", 0.65, 1),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.7, 1),
                "reg_alpha": t.suggest_float("reg_alpha", 1e-5, 2, log=True),
                "reg_lambda": t.suggest_float("reg_lambda", 0.1, 20, log=True),
                "gamma": t.suggest_float("gamma", 0, 2),
            }
            study.tell(t, 0.0)
            result.append(
                {
                    **base,
                    "name": f"xgb{i}",
                    "xgb": hp,
                    "epochs": 1200,
                    "patience": 40,
                    "features": "both" if i % 2 else "base",
                    "tree_stop": "ap" if i % 3 == 2 else "bce",
                }
            )
        return result
    raise ValueError(family)


def configuration():
    families = [
        "logistic",
        "random_forest",
        "xgboost",
        "mlp",
        "multiview",
        "ft_transformer",
        "tabnet",
    ]
    return {
        "version": "2.0",
        "outer_folds": 3,
        "inner_folds": 3,
        "seeds": [101, 202, 303],
        "candidate_seed": 101,
        "families": families,
        "candidates": {f: candidates(f) for f in families},
        "transfer": {
            "age_boundary": 55,
            "positive_budgets": [25, 100, 250, 500, 1000],
            "draw_seeds": [101, 202, 303],
            "permutations": 199,
            "bootstraps": 199,
            "mmd_cap": 128,
            "alpha": 0.05,
            "min_class_count": 5,
            "min_source_discriminability": 0.1,
        },
        "bootstrap_replicates": 1000,
    }


def copy_configuration(value):
    return copy.deepcopy(value)
