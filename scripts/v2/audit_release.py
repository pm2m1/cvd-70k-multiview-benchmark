"""Audit staged V2 artifacts while verifying every original V1 tracked file."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from pathlib import Path

from cvd70k.utils.io import write_json

BASE = "88c8020072fa3bf48d627bd3c2216d0b8e683181"
BANNED = {
    ".zip",
    ".npz",
    ".npy",
    ".joblib",
    ".pt",
    ".pth",
    ".pkl",
    ".parquet",
    ".env",
    ".pem",
    ".key",
    ".log",
}
FORBIDDEN = {
    "id",
    "row",
    "group",
    "age",
    "gender",
    "height",
    "weight",
    "cardio",
    "label",
    "probability",
    "prediction",
    "participant_id",
}


def check_artifact(name, content):
    p = Path(name)
    if p.suffix.lower() in BANNED or name.startswith(("private/", "data/raw/", "data/processed/")):
        raise ValueError("Private artifact " + name)
    if p.suffix == ".csv":
        columns = set(next(csv.reader(io.StringIO(content.decode()))))
        if columns & FORBIDDEN:
            raise ValueError("Participant columns " + name)
        if not name.startswith("results/v2/"):
            raise ValueError("Unexpected CSV " + name)
    if p.suffix not in [".png", ".pdf"]:
        text = content.decode()
        if re.search(
            r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16})", text
        ):
            raise ValueError("Credential pattern " + name)
    if len(content) > 25_000_000:
        raise ValueError("Unreviewed large artifact " + name)


def audit(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    original = git("ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    changed = []
    for name in original:
        if (root / name).read_bytes() != git("show", f"{BASE}:{name}"):
            changed.append(name)
    if changed:
        raise ValueError("V1 historical files changed " + str(changed))
    staged = git("diff", "--cached", "--name-only", BASE).decode().splitlines()
    allowed = (
        "configs/v2/",
        "src/cvd70k/v2/",
        "scripts/v2/",
        "tests/v2/",
        "results/v2/",
        "docs/v2/",
    )
    for name in staged:
        if not name.startswith(allowed) and name != ".github/workflows/v2.yml":
            raise ValueError("Unversioned change " + name)
        check_artifact(name, git("show", f":{name}"))
    return {
        "v1_commit": BASE,
        "v1_tracked_files_verified": len(original),
        "v1_changed_files": changed,
        "staged_files_reviewed": len(staged),
        "reviewed_paths": staged,
        "private_data_or_weights_staged": False,
        "credentials_found": False,
        "licence_status": "Kaggle Unknown; original code, aggregate research and synthetic fixtures only",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    result = audit(root)
    write_json(root / "results/v2/audit/publication_audit.json", result)
    print({k: v for k, v in result.items() if k != "reviewed_paths"})
