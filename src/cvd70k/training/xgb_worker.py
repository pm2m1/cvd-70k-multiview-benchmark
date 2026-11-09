"""XGBoost process boundary: avoid a macOS PyTorch/OpenMP native-library conflict.

This module deliberately never imports torch or training.engine. Requests and outputs
contain local private arrays and are never included in the public release.
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
from xgboost import XGBClassifier


def run(request_path):
    request = joblib.load(request_path)
    if request["operation"] == "fit":
        model = XGBClassifier(**request["params"])
        model.fit(
            request["x"], request["y"], eval_set=[(request["xv"], request["yv"])], verbose=False
        )
        joblib.dump(model, request["model_path"], compress=3)
        probability = model.predict_proba(request["xv"])[:, 1]
        curve = [
            {"epoch": i + 1, "validation_loss": float(v)}
            for i, v in enumerate(model.evals_result()["validation_0"]["logloss"])
        ]
        joblib.dump(
            {
                "probability": probability,
                "curve": curve,
                "best_epoch": int(getattr(model, "best_iteration", model.n_estimators - 1) + 1),
            },
            request["response_path"],
        )
    elif request["operation"] == "predict":
        model = joblib.load(request["model_path"])
        np.save(request["response_path"], model.predict_proba(request["x"])[:, 1])
    else:
        raise ValueError("Unknown XGBoost worker operation")


if __name__ == "__main__":
    run(sys.argv[1])
