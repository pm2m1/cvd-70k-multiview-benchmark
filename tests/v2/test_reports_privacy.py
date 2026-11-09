import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cvd70k.v2 import reporting


def audit_module():
    path = Path("scripts/v2/audit_release.py")
    spec = importlib.util.spec_from_file_location("v2_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publication_privacy_checks():
    audit = audit_module()
    for name, data in [
        ("private/v2/pred.npz", b"a"),
        ("results/v2/raw.csv", b"age,cardio\n55,1\n"),
        ("results/v2/weights.pt", b"a"),
    ]:
        with pytest.raises(ValueError):
            audit.check_artifact(name, data)
    audit.check_artifact("results/v2/aggregate/metrics.csv", b"model,ap,auroc\nfixture,.7,.8\n")


def test_report_and_figure_reproducible(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "model": ["invented", "invented"],
            "ap": [0.7, 0.8],
            "auroc": [0.8, 0.9],
            "brier": [0.2, 0.1],
            "balanced_accuracy": [0.7, 0.75],
            "recall": [0.6, 0.7],
        }
    )
    assert reporting.table(frame) == reporting.table(frame)
    monkeypatch.setattr(reporting, "FIG", tmp_path)

    def render():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0.2, 0.8])
        ax.set_title("Invented fixture")
        reporting.save(fig, "fixture")
        return (tmp_path / "fixture.png").read_bytes()

    assert render() == render()
    assert (tmp_path / "fixture.pdf").stat().st_size > 1000
