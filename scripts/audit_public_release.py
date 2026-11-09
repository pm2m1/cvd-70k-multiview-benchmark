"""Fail closed on unexpected staged artifacts before public Git publication."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

from cvd70k.utils.io import write_json


def audit_staged(root):
    root = Path(root)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"], cwd=root, check=True, capture_output=True
    )
    names = [name for name in result.stdout.decode().split("\0") if name]
    if not names:
        raise ValueError("No staged files to review")
    allowed_roots = {"src", "tests", "scripts", "configs", "docs", "results", ".github"}
    allowed_files = {
        ".gitignore",
        ".python-version",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "data/README.md",
    }
    allowed_csv = {"test_metrics.csv", "threshold_grid.csv", "condition_summary.csv"}
    secret = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16})"
    )
    forbidden_cols = {
        "id",
        "row",
        "group",
        "age",
        "height",
        "weight",
        "cardio",
        "label",
        "probability",
        "prediction",
        "participant_id",
    }
    total = 0
    for name in names:
        relative = Path(name)
        path = root / relative
        if relative.parts[0] not in allowed_roots and name not in allowed_files:
            raise ValueError(f"Unexpected public path: {name}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 25_000_000:
            raise ValueError(f"Unexpected file type/size: {name}")
        suffix = path.suffix.lower()
        if suffix in {
            ".zip",
            ".npz",
            ".npy",
            ".pt",
            ".pth",
            ".joblib",
            ".pkl",
            ".parquet",
            ".env",
            ".pem",
            ".key",
            ".log",
        }:
            raise ValueError(f"Private artifact staged: {name}")
        if suffix == ".pdf" and not name.startswith("results/figures/"):
            raise ValueError(f"Unexpected report PDF: {name}")
        if suffix == ".csv":
            if path.name not in allowed_csv or not name.startswith("results/aggregate/"):
                raise ValueError(f"Unreviewed CSV: {name}")
            with path.open() as f:
                columns = set(next(csv.reader(f)))
            if columns & forbidden_cols:
                raise ValueError(f"Participant-level columns: {name}")
        if suffix not in {".png", ".pdf"}:
            if secret.search(path.read_text()):
                raise ValueError(f"Credential pattern: {name}")
        total += path.stat().st_size
    return {
        "staged_file_count": len(names),
        "staged_bytes": total,
        "reviewed_paths": names,
        "participant_data_staged": False,
        "learned_artifacts_staged": False,
        "credential_patterns_found": False,
        "release_scope": "Original source/tests/docs and non-reconstructive statistical outputs; dataset licence remains Unknown",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    audit = audit_staged(root)
    write_json(root / "results/aggregate/public_release_audit.json", audit)
    print({k: v for k, v in audit.items() if k != "reviewed_paths"})
