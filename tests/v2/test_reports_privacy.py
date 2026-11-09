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


def test_transfer_metadata_preserves_evaluation_metrics(tmp_path):
    from cvd70k.utils.io import write_json
    from cvd70k.v2.metadata import normalize_transfer_metadata

    aggregate = tmp_path / "results/v2/aggregate"
    aggregate.mkdir(parents=True)
    (tmp_path / "private/v2").mkdir(parents=True)
    path = aggregate / "transfer_metrics.csv"
    frame = pd.DataFrame(
        [
            {
                "key": "p25_n25_s101",
                "method": "logical",
                "run": "fixture",
                "positive": 2,
                "prevalence": 0.5,
                "ap": 0.73123456789,
                "auroc": 0.75,
            }
        ]
    )
    frame.to_csv(path, index=False)
    write_json(
        aggregate / "transfer_manifest.json",
        [
            {
                "key": "p25_n25_s101",
                "method": "logical",
                "run": "fixture",
                "positive": 25,
                "negative": 25,
                "prevalence": 0.5,
            }
        ],
    )
    normalize_transfer_metadata(tmp_path)
    output = pd.read_csv(path)
    assert output.positive.iloc[0] == 2 and output.positive_budget.iloc[0] == 25
    assert output.ap.iloc[0] == pytest.approx(frame.ap.iloc[0])
    before = path.read_bytes()
    normalize_transfer_metadata(tmp_path)
    assert before == path.read_bytes()
