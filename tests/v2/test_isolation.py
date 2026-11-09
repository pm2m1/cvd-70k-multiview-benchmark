from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd70k.data.pipeline import clean, feature_groups
from cvd70k.utils.io import read_json, sha256
from cvd70k.v2.data import partition_transfer


def test_synthetic_transfer_roles_isolated(frame):
    frame.loc[50:, "age"] = 22000 + np.arange(50)
    df, _ = clean(frame)
    df["group"] = feature_groups(frame)
    parts = partition_transfer(df)
    for a, aa in parts.items():
        for b, bb in parts.items():
            if a != b:
                assert not set(aa.group) & set(bb.group)
    source = pd.concat([v for k, v in parts.items() if k.startswith("source")])
    target = pd.concat([v for k, v in parts.items() if k.startswith("target")])
    assert source.age_years.max() < 55 and target.age_years.min() >= 55


def test_v1_input_and_v2_roles_real_data():
    if not Path("data/raw/cardio_train.csv").exists():
        pytest.skip("Private original dataset not distributed")
    audit = read_json("results/aggregate/dataset_audit.json")
    assert sha256("data/processed/splits.csv") == audit["split_file_sha256"]
    eligible = pd.read_csv("data/processed/splits.csv").query("split == 'train'").row
    folds = pd.read_csv("private/v2/folds.csv")
    assert set(folds.row) == set(eligible)
    if Path("private/v2/transfer_roles.csv").exists():
        roles = pd.read_csv("private/v2/transfer_roles.csv")
        assert set(roles.row) == set(eligible) and roles.row.is_unique


def test_completed_target_label_accounting():
    path = Path("results/v2/aggregate/transfer_metrics.csv")
    if not path.exists():
        pytest.skip("Transfer evaluation not completed yet")
    df = pd.read_csv(path)
    total = (
        df.target_fit_positive
        + df.target_fit_negative
        + df.selector_positive
        + df.selector_negative
        + df.target_stop_n
    )
    np.testing.assert_array_equal(df.total_target_labels, total)
    logical = df[df.method == "calibrated_logical"]
    assert set(logical.positive) == {25, 100, 250, 500, 1000}
    assert logical.groupby(["positive", "negative"]).seed.nunique().eq(3).all()
