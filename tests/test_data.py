import zipfile

import numpy as np
import pandas as pd
import pytest

from cvd70k.data.pipeline import (
    Preprocessor,
    audit_dataset,
    clean,
    extract_csv,
    feature_groups,
    inspect_zip,
    load_csv,
    make_splits,
    validate_schema,
)
from cvd70k.data.sampling import feasible_plans, oversample_indices, sample_indices


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_loading(frame, tmp_path, delimiter):
    path = tmp_path / "table.csv"
    frame.to_csv(path, sep=delimiter, index=False)
    loaded, sep = load_csv(path)
    assert sep == delimiter
    pd.testing.assert_frame_equal(loaded, frame)


def test_zip_inspection_and_extraction(frame, tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("table.csv", frame.to_csv(index=False))
    assert inspect_zip(path)["entries"][0]["name"] == "table.csv"
    csv, _ = extract_csv(path, tmp_path / "raw")
    assert len(load_csv(csv)[0]) == len(frame)


@pytest.mark.parametrize("name", ["../escape.csv", "/absolute.csv", "folder\\escape.csv"])
def test_unsafe_zip(name, tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, "a")
    with pytest.raises(ValueError):
        inspect_zip(path)


@pytest.mark.parametrize(
    "column,value", [("cardio", 2), ("gender", 3), ("gluc", 8), ("age", "bad")]
)
def test_schema_rejects(frame, column, value):
    frame[column] = frame[column].astype(object) if isinstance(value, str) else frame[column]
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_schema(frame)


def test_duplicates_contradictions_isolated(frame):
    duplicate = frame.iloc[[0]].copy()
    duplicate["id"] = 1000
    duplicate["cardio"] = 1 - duplicate.cardio
    df = pd.concat([frame, duplicate], ignore_index=True)
    groups = feature_groups(df)
    assert groups[0] == groups[-1]
    split = make_splits(df, 9)
    assert split.split.iloc[0] == split.split.iloc[-1]
    for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]:
        assert not set(split.loc[split.split == a, "group"]) & set(
            split.loc[split.split == b, "group"]
        )
    assert audit_dataset(df)["contradictory_raw_groups"] == 1
    pd.testing.assert_frame_equal(split, make_splits(df, 9))


def test_cleaning_and_bmi_parents(frame):
    frame.loc[0, "height"] = 50
    frame.loc[1, "ap_lo"] = 1200
    frame.loc[2, ["ap_hi", "ap_lo"]] = [70, 80]
    minimal, flags = clean(frame)
    strict, _ = clean(frame, "strict")
    assert len(minimal) == len(frame)
    assert len(strict) == len(frame) - 3
    assert np.isnan(minimal.loc[0, "bmi"])
    assert np.isnan(minimal.loc[1, "ap_lo"])
    assert np.isnan(minimal.loc[2, "ap_hi"]) and np.isnan(minimal.loc[2, "ap_lo"])
    assert flags.any(axis=1).sum() == 3


def test_train_only_imputation_scaling(frame):
    train, _ = clean(frame)
    train.loc[0, "height"] = np.nan
    prep = Preprocessor().fit(train)
    baseline = prep.imputer.statistics_.copy()
    val = train.copy()
    val["weight"] = 199.0
    transformed = prep.transform(val)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(baseline, prep.imputer.statistics_)
    assert transformed.shape == (len(val), 16)
    assert prep.columns.count("id") == 0
    with pytest.raises(ValueError):
        Preprocessor().fit(val, role="test")
    with pytest.raises(ValueError):
        Preprocessor().transform(val)


def test_missing_and_boundary_handling(frame):
    frame.loc[0, "height"] = np.nan
    frame.loc[1, ["height", "weight", "ap_hi", "ap_lo"]] = [120, 30.0, 80, 40]
    minimal, _ = clean(frame)
    strict, _ = clean(frame, "strict")
    assert 0 not in strict.index and 1 in strict.index
    assert np.isfinite(Preprocessor().fit(minimal).transform(minimal)).all()


def test_sampling_feasibility_and_fixed_counts():
    y = np.r_[np.ones(2500), np.zeros(2400)]
    plans = feasible_plans(y)
    fixed = [p for p in plans if p["analysis"] == "fixed_total"]
    positives = [p for p in plans if p["analysis"] == "fixed_positive"]
    assert len(set(p["n"] for p in fixed)) == 1
    assert len(set(p["positive"] for p in positives)) == 1
    for plan in plans:
        a = sample_indices(y, plan["positive"], plan["negative"], 3)
        b = sample_indices(y, plan["positive"], plan["negative"], 3)
        np.testing.assert_array_equal(a, b)
        assert len(np.unique(a)) == len(a)
        assert int(y[a].sum()) == plan["positive"]
        assert abs(y[a].mean() - plan["requested_prevalence"]) < 0.001
    with pytest.raises(ValueError):
        sample_indices(y, 3000, 1, 1)


def test_oversampling_training_only():
    y = np.r_[np.ones(3), np.zeros(10)]
    idx = oversample_indices(y, 7)
    assert len(idx) == 20
    assert y[idx].sum() == 10
    assert set(idx) == set(range(13))
