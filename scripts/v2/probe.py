"""V2 device and third-party fit smoke checks, only V1 train rows."""

import copy

import torch

from cvd70k.utils.io import write_json
from cvd70k.v2.config import candidates
from cvd70k.v2.data import development_data
from cvd70k.v2.engine import fit_trial, predict_trial

if __name__ == "__main__":
    torch.set_num_threads(2)
    df = development_data().iloc[:1000]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    results = {
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": device,
        "families": {},
    }
    for family in ["multiview", "ft_transformer", "tabnet", "xgboost"]:
        cfg = copy.deepcopy(candidates(family)[0])
        cfg.update(epochs=2, patience=2)
        try:
            r = fit_trial("smoke_" + family, df.iloc[:800], df.iloc[800:], cfg, 101, device)
            p = predict_trial("smoke_" + family, df.iloc[800:])
            assert len(p) == 200 and torch.isfinite(torch.tensor(p)).all()
            results["families"][family] = {
                "status": "passed",
                "training_device": device,
                "best_epoch": r["training"]["best_epoch"],
            }
        except Exception as error:
            results["families"][family] = {
                "status": "failed",
                "error": repr(error),
                "training_device": device,
            }
    write_json("results/v2/aggregate/device_probe.json", results)
    print(results)
