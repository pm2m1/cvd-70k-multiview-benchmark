"""Auditable loading, plausibility masking and immutable group-isolated splits."""

from __future__ import annotations

import csv
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from cvd70k.utils.io import sha256, write_json

RAW_FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
CATEGORIES = {
    "gender": [1, 2],
    "cholesterol": [1, 2, 3],
    "gluc": [1, 2, 3],
    "smoke": [0, 1],
    "alco": [0, 1],
    "active": [0, 1],
}
VIEWS = {
    "background": ["age_years", "gender"],
    "haemodynamic": ["ap_hi", "ap_lo"],
    "anthropometric": ["height", "weight", "bmi"],
    "metabolic": ["cholesterol_1", "cholesterol_2", "cholesterol_3", "gluc_1", "gluc_2", "gluc_3"],
    "behaviour": ["smoke", "alco", "active"],
}


def inspect_zip(archive):
    with zipfile.ZipFile(archive) as z:
        entries = []
        for item in z.infolist():
            p = PurePosixPath(item.filename)
            if (
                p.is_absolute()
                or ".." in p.parts
                or "\\" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
                or item.flag_bits & 1
                or item.file_size > 100_000_000
                or item.file_size / max(1, item.compress_size) > 200
            ):
                raise ValueError("Unsafe ZIP entry")
            entries.append(
                {
                    "name": item.filename,
                    "bytes": item.file_size,
                    "compressed_bytes": item.compress_size,
                    "crc": item.CRC,
                }
            )
        if sum(x["bytes"] for x in entries) > 200_000_000 or z.testzip() is not None:
            raise ValueError("Unsafe size or corrupt archive")
    return {"archive_sha256": sha256(archive), "entries": entries}


def extract_csv(archive, destination):
    audit = inspect_zip(archive)
    files = [e for e in audit["entries"] if e["name"].lower().endswith(".csv")]
    if len(files) != 1 or len(audit["entries"]) != 1:
        raise ValueError("Review unexpected archive contents before extraction")
    dest = Path(destination) / Path(files[0]["name"]).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        content = z.read(files[0]["name"])
    if dest.exists() and dest.read_bytes() != content:
        raise ValueError("Refusing to overwrite a different raw dataset")
    dest.write_bytes(content)
    return dest, audit


def load_csv(path):
    with open(path) as f:
        delimiter = csv.Sniffer().sniff(f.read(8192), delimiters=",;\t").delimiter
    df = pd.read_csv(path, sep=delimiter)
    validate_schema(df)
    return df, delimiter


def validate_schema(df):
    if set(df) != set(RAW_FEATURES + ["id", "cardio"]):
        raise ValueError("Unexpected schema")
    for col in df:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Non-numeric column: {col}")
        if np.isinf(df[col].to_numpy()).any():
            raise ValueError(f"Infinite values: {col}")
    if df.cardio.isna().any() or not df.cardio.isin([0, 1]).all():
        raise ValueError("Target must be observed binary 0/1")
    for col, values in CATEGORIES.items():
        if not df[col].dropna().isin(values).all():
            raise ValueError(f"Unexpected category: {col}")


def clean(df, mode="minimal"):
    """Fixed research quality bounds, NOT diagnostic or optimized thresholds."""
    if mode not in ["minimal", "strict"]:
        raise ValueError(mode)
    x = df.copy()
    x["age_years"] = x.age / 365.25
    checks = {
        "age_outside_18_100": ~x.age_years.between(18, 100),
        "height_outside_120_220_cm": ~x.height.between(120, 220),
        "weight_outside_30_200_kg": ~x.weight.between(30, 200),
        "systolic_outside_80_250": ~x.ap_hi.between(80, 250),
        "diastolic_outside_40_150": ~x.ap_lo.between(40, 150),
        "systolic_not_above_diastolic": x.ap_hi <= x.ap_lo,
    }
    flags = pd.DataFrame(checks, index=x.index)
    x.loc[flags.iloc[:, 0], "age_years"] = np.nan
    x.loc[flags.iloc[:, 1], "height"] = np.nan
    x.loc[flags.iloc[:, 2], "weight"] = np.nan
    x.loc[flags.iloc[:, 3] | flags.iloc[:, 5], "ap_hi"] = np.nan
    x.loc[flags.iloc[:, 4] | flags.iloc[:, 5], "ap_lo"] = np.nan
    # Derive BMI only after invalid raw parents have been masked.
    x["bmi"] = x.weight / (x.height / 100) ** 2
    flags["bmi_outside_10_70"] = x.bmi.notna() & ~x.bmi.between(10, 70)
    x.loc[flags.bmi_outside_10_70, "bmi"] = np.nan
    if mode == "strict":
        keep = ~flags.any(axis=1) & x[RAW_FEATURES + ["bmi"]].notna().all(axis=1)
        x = x.loc[keep].copy()
    return x, flags


def feature_groups(df):
    """Hash full modeled feature vectors after fixed masking; no label or ID."""
    x, _ = clean(df)
    cols = [
        "age_years",
        "gender",
        "height",
        "weight",
        "bmi",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]
    return pd.util.hash_pandas_object(x[cols], index=False).to_numpy()


def make_splits(df, seed=20261109):
    groups = feature_groups(df)
    # Twenty approximately stratified, disjoint folds: 14 train, 3 val, 3 test.
    cv = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, indices) in enumerate(cv.split(df, df.cardio, groups)):
        fold[indices] = k
    split = np.where(fold < 14, "train", np.where(fold < 17, "validation", "test"))
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(groups[split == a]) & set(groups[split == b])
    return pd.DataFrame({"row": np.arange(len(df)), "group": groups, "split": split})


def audit_dataset(df):
    minimal, flags = clean(df)
    strict, _ = clean(df, "strict")
    raw_groups = df.groupby(RAW_FEATURES, dropna=False).cardio.agg(["size", "nunique"])
    model_groups = pd.DataFrame({"group": feature_groups(df), "y": df.cardio})
    grouped = model_groups.groupby("group").y.agg(["size", "nunique"])
    return {
        "shape": list(df.shape),
        "columns": list(df),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "class_counts": df.cardio.value_counts().to_dict(),
        "prevalence": df.cardio.mean(),
        "id_unique": bool(df.id.is_unique),
        "duplicates_including_id": int(df.duplicated().sum()),
        "duplicate_labeled_rows_excluding_id": int(df.duplicated(RAW_FEATURES + ["cardio"]).sum()),
        "feature_identical_groups_raw": int((raw_groups["size"] > 1).sum()),
        "contradictory_raw_groups": int((raw_groups["nunique"] > 1).sum()),
        "modeled_duplicate_groups": int((grouped["size"] > 1).sum()),
        "modeled_contradictory_groups": int((grouped["nunique"] > 1).sum()),
        "modeled_group_count": len(grouped),
        "flags_overlapping": flags.sum().to_dict(),
        "flagged_rows_union": int(flags.any(axis=1).sum()),
        "minimal_retained": len(minimal),
        "strict_retained": len(strict),
        "strict_excluded": len(df) - len(strict),
        "postmask_missing": minimal.isna().sum().to_dict(),
        "ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in df},
        "categories": {c: df[c].value_counts().sort_index().to_dict() for c in CATEGORIES},
    }


class Preprocessor:
    """Explicit categorical encoding; imputation/scaling learned only in fit(train)."""

    def __init__(self, anthropometry="all"):
        if anthropometry not in ["all", "bmi_only", "parents_only"]:
            raise ValueError(anthropometry)
        self.views = {k: v.copy() for k, v in VIEWS.items()}
        if anthropometry == "bmi_only":
            self.views["anthropometric"] = ["bmi"]
        elif anthropometry == "parents_only":
            self.views["anthropometric"] = ["height", "weight"]
        self.columns = [c for cols in self.views.values() for c in cols]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.fitted = False

    def encode(self, df):
        x = df.copy()
        x["gender"] = x.gender - 1  # no unverified mapping of numeric code to sex
        for c in ["cholesterol", "gluc"]:
            for level in [1, 2, 3]:
                x[f"{c}_{level}"] = np.where(x[c].isna(), np.nan, (x[c] == level).astype(float))
        return x[self.columns].to_numpy(dtype=np.float64)

    def fit(self, df, role="train"):
        if role != "train":
            raise ValueError("Preprocessing fit is restricted to training")
        x = self.imputer.fit_transform(self.encode(df))
        self.scaler.fit(x)
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise ValueError("Not fitted")
        return self.scaler.transform(self.imputer.transform(self.encode(df))).astype(np.float32)


def prepare(root, archive, seed=20261109):
    root = Path(root)
    path, info = extract_csv(archive, root / "data/raw")
    df, delimiter = load_csv(path)
    info.update(audit_dataset(df))
    info.update({"delimiter": delimiter, "csv_sha256": sha256(path), "split_seed": seed})
    dest = root / "data/processed/splits.csv"
    splits = make_splits(df, seed)
    if dest.exists():
        old = pd.read_csv(dest)
        if not old.equals(splits):
            raise ValueError("Frozen splits differ; use a new workspace/protocol")
    else:
        splits.to_csv(dest, index=False)
    info["split_file_sha256"] = sha256(dest)
    info["partitions"] = {}
    for mode in ["minimal", "strict"]:
        x, _ = clean(df, mode)
        info["partitions"][mode] = {}
        for part in ["train", "validation", "test"]:
            indices = splits.index[splits.split == part].intersection(x.index)
            y = x.loc[indices, "cardio"]
            info["partitions"][mode][part] = {
                "n": len(y),
                "positive": int(y.sum()),
                "negative": int((y == 0).sum()),
                "prevalence": float(y.mean()),
            }
    write_json(root / "results/aggregate/dataset_audit.json", info)
    return df, splits, info
