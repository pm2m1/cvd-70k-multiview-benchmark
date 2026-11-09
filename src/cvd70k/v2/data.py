"""V2 data access: only the immutable V1 training population is eligible."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

from cvd70k.data.pipeline import Preprocessor, clean, feature_groups
from cvd70k.utils.io import read_json, sha256, write_json


def development_data(root=Path.cwd()):
    root = Path(root)
    audit = read_json(root / "results/aggregate/dataset_audit.json")
    raw = root / "data/raw/cardio_train.csv"
    split = root / "data/processed/splits.csv"
    if sha256(raw) != audit["csv_sha256"] or sha256(split) != audit["split_file_sha256"]:
        raise ValueError("V1 input integrity failure")
    membership = pd.read_csv(split)
    permitted = membership.loc[membership.split == "train", "row"].to_numpy()
    # Read the authorized file, restrict to V1 train BEFORE any transform/statistics.
    df = pd.read_csv(raw, sep=";").iloc[permitted].copy()
    assert set(df.index) == set(permitted)
    x, _ = clean(df)
    x["group"] = feature_groups(df)
    return x


def folds(df, n=3, seed=711):
    result = np.full(len(df), -1)
    for k, (_, test) in enumerate(
        StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
            df, df.cardio, df.group
        )
    ):
        result[test] = k
    for group in np.unique(result):
        assert not set(df.loc[result == group, "group"]) & set(df.loc[result != group, "group"])
    return result


def prepare_folds(root=Path.cwd()):
    root = Path(root)
    df = development_data(root)
    outer = folds(df, 3, 711)
    assignment = pd.DataFrame({"row": df.index, "outer": outer})
    all_folds = {}
    counts = []
    for k in range(3):
        outer_train = df.loc[outer != k]
        calibration_fold = folds(outer_train, 5, 800 + k)
        calibration = outer_train.index[calibration_fold == 4]
        fit = outer_train.index[calibration_fold != 4]
        inner = folds(df.loc[fit], 3, 900 + k)
        all_folds[k] = {
            "outer_test": df.index[outer == k].to_numpy(),
            "calibration": calibration.to_numpy(),
            "fit": fit.to_numpy(),
            "inner": inner,
        }
        assignment.loc[assignment.row.isin(calibration), f"role_{k}"] = "calibration"
        assignment.loc[assignment.row.isin(fit), f"role_{k}"] = "inner_pool"
        for name, idx in [
            ("outer_test", df.index[outer == k]),
            ("calibration", calibration),
            ("inner_pool", fit),
        ]:
            yy = df.loc[idx, "cardio"]
            counts.append(
                {
                    "outer": k,
                    "role": name,
                    "n": len(idx),
                    "positive": int(yy.sum()),
                    "negative": int((yy == 0).sum()),
                }
            )
        for j in range(3):
            yy = df.loc[fit[inner == j], "cardio"]
            counts.append(
                {
                    "outer": k,
                    "role": f"inner_validation_{j}",
                    "n": len(yy),
                    "positive": int(yy.sum()),
                    "negative": int((yy == 0).sum()),
                }
            )
    p = root / "private/v2/folds.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_text() != assignment.to_csv(index=False):
            raise ValueError("V2 folds changed")
    else:
        p.write_text(assignment.to_csv(index=False))
    write_json(
        root / "results/v2/aggregate/folds.json",
        {
            "eligible_v1_partition": "train only",
            "n": len(df),
            "seeds": {"outer": 711, "calibration": 800, "inner": 900},
            "fold_file_sha256": sha256(p),
            "counts": counts,
        },
    )
    return df, all_folds


class Features:
    """Fixed deterministic blocks; all imputation/scaling fitted on fold training."""

    def __init__(self, block="base", scale="standard"):
        if block not in ["base", "derived", "missing", "both"]:
            raise ValueError(block)
        self.block, self.scale = block, scale
        self.base = Preprocessor()
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = RobustScaler() if scale == "robust" else StandardScaler()
        self.names = None
        self.dims = None

    def raw(self, df):
        base = self.base.encode(df)
        parts = []
        names = []
        dims = []
        offset = 0
        for name, cols in self.base.views.items():
            values = [base[:, offset : offset + len(cols)]]
            nn = cols.copy()
            offset += len(cols)
            if self.block in ["derived", "both"]:
                additions = {
                    "background": {"age_bmi": df.age_years * df.bmi},
                    "haemodynamic": {
                        "pulse_pressure": df.ap_hi - df.ap_lo,
                        "map": (df.ap_hi + 2 * df.ap_lo) / 3,
                        "age_sbp": df.age_years * df.ap_hi,
                    },
                    "anthropometric": {"bmi_squared": df.bmi**2},
                }.get(name, {})
                for col, value in additions.items():
                    values.append(value.to_numpy()[:, None])
                    nn.append(col)
            if self.block in ["missing", "both"]:
                original = base[:, offset - len(cols) : offset]
                for j, col in enumerate(cols):
                    values.append(np.isnan(original[:, j : j + 1]).astype(float))
                    nn.append(col + "_missing")
            part = np.column_stack(values)
            parts.append(part)
            names.extend(nn)
            dims.append(part.shape[1])
        self.names, self.dims = names, dims
        return np.column_stack(parts)

    def fit(self, df, role="train"):
        if role != "train":
            raise ValueError("Train-only feature fit")
        self.scaler.fit(self.imputer.fit_transform(self.raw(df)))
        return self

    def transform(self, df):
        return self.scaler.transform(self.imputer.transform(self.raw(df))).astype(np.float32)


def partition_transfer(df):
    # Fixed boundary selected from recorded age and feasibility, not performance.
    source = df[df.age_years < 55].copy()
    target = df[df.age_years >= 55].copy()
    assert not set(source.group) & set(target.group)
    sf = folds(source, 10, 1201)
    tf = folds(target, 10, 1202)
    parts = {
        "source_train": source.loc[sf < 8],
        "source_stop": source.loc[sf == 8],
        "source_selector": source.loc[sf == 9],
        "target_fit_pool": target.loc[tf < 4],
        "target_selector_pool": target.loc[(tf >= 4) & (tf < 7)],
        "target_stop": target.loc[tf == 7],
        "target_evaluation": target.loc[tf >= 8],
    }
    for name, a in parts.items():
        for other, b in parts.items():
            if name != other:
                assert not set(a.group) & set(b.group)
    return parts
