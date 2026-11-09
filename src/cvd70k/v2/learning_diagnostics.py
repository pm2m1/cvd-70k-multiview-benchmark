"""Three development-only refits with train AP instrumentation.

The frozen engine's validation-probability boundary is wrapped only in this separate,
single-threaded diagnostic process. Training-mode predictions, optimizer steps, RNG calls,
checkpoint choices and original checkpoints are not changed. Extra train predictions run
in eval/no-grad mode and are reduced immediately to aggregate metrics. This avoids rewriting
already frozen engines or inventing historical curves that were never saved.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import joblib
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cvd70k.training.engine import set_seed
from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2 import engine
from cvd70k.v2.data import Features, prepare_folds
from cvd70k.v2.models import model_for


@contextmanager
def training_metric_monitor(x, y, records):
    original = engine.probabilities

    def monitored(model, xv, device="cpu", batch=4096):
        pv = original(model, xv, device, batch)
        pt = original(model, x, device, batch)
        records.append(
            {
                "train_ap": float(average_precision_score(y, pt)),
                "train_auroc": float(roc_auc_score(y, pt)),
            }
        )
        return pv

    engine.probabilities = monitored
    try:
        yield
    finally:
        engine.probabilities = original


def main():
    torch.set_num_threads(2)
    root = Path.cwd()
    cfg = read_json(root / "configs/v2/study.json")
    selections = read_json(root / "results/v2/aggregate/optimization_selections.json")
    df, parts = prepare_folds()
    results = []
    for outer, part in parts.items():
        c = selections[str(outer)]["selected"]["multiview"]
        settings = cfg["candidates"]["multiview"][c]
        pool = df.loc[part["fit"]]
        train = pool.iloc[part["inner"] != 0]
        val = pool.iloc[part["inner"] == 0]
        path = root / "private/v2/models" / f"learning_diagnostic_o{outer}.joblib"
        meta = root / "results/v2/aggregate" / f"learning_diagnostic_o{outer}.json"
        if meta.exists():
            old = read_json(meta)
            if sha256(path) != old["checkpoint_sha256"]:
                raise ValueError("Diagnostic checkpoint mismatch")
            results.append(old)
            continue
        prep = Features(settings["features"], settings["scale"]).fit(train)
        x = prep.transform(train)
        y = train.cardio.to_numpy()
        set_seed(101)
        model = model_for(prep, settings)
        extra = []
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        with training_metric_monitor(x, y, extra):
            model, info = engine.train_network(
                model, x, y, prep.transform(val), val.cardio.to_numpy(), settings, 101, device
            )
        assert len(extra) == len(info["curve"])
        for curve, diagnostics in zip(info["curve"], extra, strict=True):
            curve.update(diagnostics)
        joblib.dump(
            {"state": model.state_dict(), "preprocessor": prep, "config": settings},
            path,
            compress=3,
        )
        original = joblib.load(
            root / "private/v2/models" / f"inner_o{outer}_j0_multiview_c{c}_s101.joblib"
        )
        difference = max(
            float((model.state_dict()[k] - value).abs().max())
            for k, value in original["state"].items()
        )
        record = {
            "outer": outer,
            "candidate": c,
            "seed": 101,
            "scope": "development-only diagnostic refit, excluded from hyperparameter selection",
            "training": info,
            "checkpoint_sha256": sha256(path),
            "max_weight_difference_vs_original_inner_checkpoint": difference,
        }
        write_json(meta, record)
        results.append(record)
    write_json(root / "results/v2/aggregate/learning_diagnostics.json", results)
    print(
        "Three genuine diagnostic refits complete; weight differences:",
        [r["max_weight_difference_vs_original_inner_checkpoint"] for r in results],
    )


if __name__ == "__main__":
    main()
