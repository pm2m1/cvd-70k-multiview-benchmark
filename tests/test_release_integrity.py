from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd70k.utils.io import read_json, sha256


def test_public_aggregate_consistency():
    path = Path("results/aggregate/test_metrics.csv")
    if not path.exists():
        pytest.skip("No completed benchmark in this development checkout yet")
    frame = pd.read_csv(path)
    assert len(frame) == 1360 and frame.run_id.nunique() == 680
    assert (frame[["tn", "fp", "fn", "tp"]].sum(axis=1) == frame.n).all()
    np.testing.assert_allclose(frame.recall, frame.tp / (frame.tp + frame.fn))
    np.testing.assert_allclose(frame.balanced_accuracy, (frame.recall + frame.specificity) / 2)
    assert frame.groupby("run_id").operating.nunique().eq(2).all()
    assert frame.groupby("run_id").ap.nunique().eq(1).all()
    im = frame[frame.analysis.isin(["fixed_total", "fixed_positive"])]
    assert im.run_id.nunique() == 600
    for _, group in im.groupby(
        ["analysis", "requested_prevalence", "family", "intervention", "operating"]
    ):
        assert len(group) == 5 and group.seed.nunique() == 5
    fixed = im[im.analysis == "fixed_total"]
    assert fixed.train_n.nunique() == 1
    fixed = im[im.analysis == "fixed_positive"]
    assert fixed.train_positive.nunique() == 1
    assert im.n.nunique() == 1 and im.positive.nunique() == 1


def test_release_lock_checksums():
    path = Path("results/aggregate/development_lock.json")
    if not path.exists():
        pytest.skip("Development has not locked yet")
    lock = read_json(path)
    assert sha256("results/aggregate/development.json") == lock["development_sha256"]
    assert sha256("docs/experiment_protocol.md") == lock["protocol_sha256"]
    assert sha256("configs/benchmark.json") == lock["config_sha256"]
    for name, expected in lock["source_sha256"].items():
        assert sha256(name) == expected
    dev = read_json("results/aggregate/development.json")
    expected = max(dev["validation_mean_ap"], key=dev["validation_mean_ap"].get)
    assert dev["validation_selected_winner"] == expected


def test_private_paths_ignored():
    ignore = Path(".gitignore").read_text()
    assert "data/**" in ignore and "private/" in ignore
    assert "*.zip" in ignore and "*.joblib" in ignore and ".venv/" in ignore
