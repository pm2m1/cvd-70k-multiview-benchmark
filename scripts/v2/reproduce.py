"""Fresh private V2 workspace; preserve every published V1/V2 historical artifact."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cvd70k.utils.io import read_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--v1-splits", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    audit = read_json(root / "results/aggregate/dataset_audit.json")
    if (
        sha256(args.raw_csv) != audit["csv_sha256"]
        or sha256(args.v1_splits) != audit["split_file_sha256"]
    ):
        raise ValueError("Original V1 dataset/split hashes required")
    dest = root / "private/v2/reproductions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True, exist_ok=False)
    for name in ["src", "scripts", "docs", "configs"]:
        shutil.copytree(
            root / name, dest / name, ignore=shutil.ignore_patterns(".venv", "__pycache__")
        )
    shutil.copytree(root / "results/aggregate", dest / "results/aggregate")
    for name in ["pyproject.toml", "uv.lock", ".gitignore"]:
        shutil.copy2(root / name, dest / name)
    for name in [
        "data/raw",
        "data/processed",
        "private/v2",
        "results/v2/audit",
        "results/v2/aggregate",
        "results/v2/figures",
        "results/v2/reports",
    ]:
        (dest / name).mkdir(parents=True, exist_ok=True)
    os.symlink(args.raw_csv.expanduser().resolve(), dest / "data/raw/cardio_train.csv")
    os.symlink(args.v1_splits.expanduser().resolve(), dest / "data/processed/splits.csv")
    env = {**os.environ, "PYTHONPATH": str(dest / "src")}
    # Device probe must select validated fallbacks; existing production study config is copied.
    for cmd in [
        ["scripts/v2/audit_v1.py"],
        ["scripts/v2/probe.py"],
        ["-m", "cvd70k.v2.optimization", "develop"],
        ["-m", "cvd70k.v2.transfer", "develop"],
        ["-m", "cvd70k.v2.learning_diagnostics"],
        ["-m", "cvd70k.v2.optimization", "evaluate"],
        ["-m", "cvd70k.v2.transfer", "evaluate"],
        ["-m", "cvd70k.v2.reporting"],
    ]:
        subprocess.run([sys.executable, *cmd], cwd=dest, env=env, check=True)
    print(dest / "docs/v2/RESEARCH_OVERVIEW.md")


if __name__ == "__main__":
    main()
