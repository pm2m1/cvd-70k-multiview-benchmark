"""Lossless separation of condition metadata from similarly named metric fields."""

import csv
from pathlib import Path

import pandas as pd

from cvd70k.utils.io import read_json, sha256, write_json


def read_exact_csv(path):
    # Use Python's correctly rounded float conversion without deprecated pandas options.
    frame = pd.read_csv(path)
    with open(path, newline="") as stream:
        raw = list(csv.DictReader(stream))
    for column in frame.select_dtypes(include="floating").columns:
        frame[column] = [float(row[column]) if row[column] else float("nan") for row in raw]
    return frame


def normalize_transfer_metadata(root=Path.cwd()):
    root = Path(root)
    path = root / "results/v2/aggregate/transfer_metrics.csv"
    frame = read_exact_csv(path)
    manifest = read_json(root / "results/v2/aggregate/transfer_manifest.json")
    keys = ["key", "method", "run"]
    mapping = pd.DataFrame(
        [
            {
                **{k: r[k] for k in keys},
                "positive_budget": r["positive"],
                "negative_budget": r["negative"],
                "training_prevalence": r["prevalence"],
            }
            for r in manifest
        ]
    )
    if "positive_budget" in frame:
        check = frame.merge(mapping, on=keys, suffixes=("", "_manifest"), validate="one_to_one")
        assert (check.positive_budget == check.positive_budget_manifest).all()
        return
    before_sha = sha256(path)
    before = path.read_bytes()
    backup = root / "private/v2/transfer_metrics_executor_original.csv"
    if backup.exists() and backup.read_bytes() != before:
        raise ValueError("Preserved original export differs")
    backup.write_bytes(before)
    fixed = frame.merge(mapping, on=keys, validate="one_to_one", sort=False)
    assert len(fixed) == len(frame)
    fixed.to_csv(path, index=False)
    after = read_exact_csv(path)
    pd.testing.assert_frame_equal(frame, after[frame.columns])
    write_json(
        root / "results/v2/audit/evaluation_metadata_normalization.json",
        {
            "reason": "Condition positive/prevalence field names collided with evaluation metric fields. Add explicit budget and training-prevalence columns from frozen manifest.",
            "original_export_sha256": before_sha,
            "normalized_export_sha256": sha256(path),
            "original_columns_and_all_metric_values_preserved_exactly": True,
            "refits_performed": 0,
            "rows": len(frame),
            "frozen_manifest_sha256": sha256(root / "results/v2/aggregate/transfer_manifest.json"),
        },
    )
