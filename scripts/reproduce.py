"""Run a fresh full benchmark without overwriting the published locked results."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def create_reproduction(root):
    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = root / "private/reproductions" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["src", "scripts", "configs", "docs"]:
        shutil.copytree(
            root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ["pyproject.toml", "uv.lock", ".python-version"]:
        shutil.copy2(root / name, destination / name)
    for name in [
        "data/raw",
        "data/processed",
        "results/aggregate",
        "results/figures",
        "results/reports",
    ]:
        (destination / name).mkdir(parents=True, exist_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve(strict=True)
    destination = create_reproduction(Path(__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": str(destination / "src")}
    commands = [
        ["scripts/prepare.py", "--archive", str(archive)],
        ["scripts/probe_hardware.py"],
        ["scripts/run.py", "develop"],
        ["scripts/run.py", "evaluate"],
        ["scripts/report.py"],
    ]
    print(f"Fresh reproduction workspace: {destination}", flush=True)
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=destination, env=env, check=True)
    print(f"Complete: {destination / 'docs/results.md'}", flush=True)


if __name__ == "__main__":
    main()
