"""Post-lock descriptive CPU inference cost; no training or model selection."""

import time

import joblib
import torch

from cvd70k.utils.io import write_json
from cvd70k.v2.data import development_data
from cvd70k.v3.design import PRIV, PUB
from cvd70k.v3.runner import predict

torch.set_num_threads(4)
df = development_data()
private = joblib.load(PRIV / "classification_outer_predictions.joblib")
subset = df.loc[private[0]["rows"][:1000]]
names = [
    "v2_reference",
    "gated",
    "residual",
    "attention",
    "pairwise",
    "flat_attention_matched",
    "ft_reference",
    "xgboost_1",
    "lightgbm_1",
    "catboost_1",
]
rows = []
for name in names:
    path = PRIV / "models" / f"class_o0_{name}_s101.joblib"
    start = time.perf_counter()
    bundle = joblib.load(path)
    load = time.perf_counter() - start
    predict(bundle, subset)
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        predict(bundle, subset)
        timings.append(time.perf_counter() - start)
    import numpy as np

    rows.append(
        dict(
            model=name,
            n=len(subset),
            checkpoint_load_seconds=load,
            median_predict_seconds=float(np.median(timings)),
            scope="CPU preprocessing+model reconstruction+prediction; native trees include isolated process startup; no trained-weight distribution",
        )
    )
write_json(PUB / "inference_profile.json", rows)
print("Profiled", len(rows), "locked predictors")
