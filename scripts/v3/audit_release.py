"""Check staged V3 changes against frozen history and public-file constraints."""

import json
import re
import subprocess
from pathlib import Path

from cvd70k.v3.design import BASE, preserve


def check_artifact(name, content):
    p = Path(name)
    allowed = (
        any(
            name.startswith(x)
            for x in [
                "configs/v3/",
                "src/cvd70k/v3/",
                "scripts/v3/",
                "tests/v3/",
                "results/v3/",
                "docs/v3/",
            ]
        )
        or name == ".github/workflows/v3.yml"
    )
    if not allowed:
        raise ValueError("Not a V3 path: " + name)
    if p.suffix.lower() in [
        ".zip",
        ".npz",
        ".npy",
        ".parquet",
        ".pt",
        ".pth",
        ".pkl",
        ".joblib",
        ".bin",
        ".csv.gz",
    ]:
        raise ValueError("Private artifact extension: " + name)
    if p.suffix == ".csv":
        head = content.split(b"\n", 1)[0].decode().lower().split(",")
        if any(
            c in head
            for c in [
                "row",
                "row_id",
                "patient_id",
                "participant_id",
                "prediction",
                "probability",
                "cardio",
                "ap_hi",
                "ap_lo",
                "height",
                "weight",
            ]
        ):
            raise ValueError("Participant-like columns: " + name)
    if p.suffix in [".json", ".md", ".py", ".toml", ".yml", ".lock", ".sh", ".csv"]:
        text = content.decode()
        if re.search(
            r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----",
            text,
        ):
            raise ValueError("Potential credential: " + name)
        if re.search(r"/(?:Users|home)/[A-Za-z0-9._-]+/", text):
            raise ValueError("Machine-local path: " + name)
    return True


def audit(staged=True):
    preserve()
    args = ["git", "diff", "--cached" if staged else "HEAD", "--name-only", BASE]
    paths = subprocess.check_output(args, text=True).splitlines()
    for name in paths:
        content = (
            subprocess.check_output(["git", "show", ":" + name])
            if staged
            else Path(name).read_bytes()
        )
        check_artifact(name, content)
    return {
        "paths_checked": len(paths),
        "historical_preservation": True,
        "participant_artifacts": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
