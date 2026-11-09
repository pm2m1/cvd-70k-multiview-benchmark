"""Tabular native fits isolated from PyTorch/OpenMP; never imports Torch."""

from __future__ import annotations

import sys
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def negative_ap(y, p):
    return -average_precision_score(y, p)


def run(path):
    r = joblib.load(path)
    if r["operation"] == "predict":
        model = joblib.load(r["model_path"])
        response = {"probability": model.predict_proba(r["x"])[:, 1]}
    else:
        cfg = r["config"]
        family = r["family"]
        start = time.perf_counter()
        if family == "logistic":
            model = LogisticRegression(C=cfg.get("C", 1.0), max_iter=2000, random_state=r["seed"])
        elif family == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=10,
                max_depth=cfg["depth"],
                n_jobs=2,
                random_state=r["seed"],
            )
        elif family == "xgboost":
            params = {
                **cfg["xgb"],
                "n_estimators": r["fixed_epochs"] or cfg["epochs"],
                "n_jobs": 2,
                "random_state": r["seed"],
                "tree_method": "hist",
                "eval_metric": negative_ap if cfg.get("tree_stop") == "ap" else "logloss",
            }
            if r["xv"] is not None:
                params["early_stopping_rounds"] = cfg["patience"]
            model = XGBClassifier(**params)
        else:
            raise ValueError(family)
        if family == "xgboost" and r["xv"] is not None:
            model.fit(r["x"], r["y"], eval_set=[(r["xv"], r["yv"])], verbose=False)
        else:
            model.fit(r["x"], r["y"])
        if r["xv"] is not None:
            p = model.predict_proba(r["xv"])[:, 1]
        best = (
            int(getattr(model, "best_iteration", (r["fixed_epochs"] or cfg["epochs"]) - 1) + 1)
            if family == "xgboost"
            else 1
        )
        curve = []
        if family == "xgboost" and r["xv"] is not None:
            vv = next(iter(model.evals_result()["validation_0"].values()))
            curve = [{"epoch": i + 1, "validation_objective": float(v)} for i, v in enumerate(vv)]
        info = {
            "fit_seconds": time.perf_counter() - start,
            "best_epoch": best,
            "epochs_completed": len(curve) or best,
            "parameter_count": int(model.coef_.size + model.intercept_.size)
            if family == "logistic"
            else None,
            "curve": curve,
        }
        if family == "xgboost":
            model.set_params(eval_metric="logloss")
        joblib.dump(model, r["model_path"], compress=3)
        response = {"info": info}
        if r["xv"] is not None:
            response["probability"] = p
    joblib.dump(response, r["response"])


if __name__ == "__main__":
    run(sys.argv[1])
