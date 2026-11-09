"""Prepare an isolated fresh worktree so public trial records cannot masquerade as caches.

The public release deliberately omits weights and row predictions. Its completed trial
metadata therefore cannot resume locally. This prepares empty V3 outputs in a NEW
worktree; the original checkout, V1/V2 artifacts and local source data remain untouched.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def prepare_fresh(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError("Fresh reproduction destination must not exist")
    if (
        not (source / "data/raw/cardio_train.csv").exists()
        or not (source / "data/processed/splits.csv").exists()
    ):
        raise ValueError("Prepare the privately downloaded audited V1 raw file and splits first")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(destination), commit], cwd=source, check=True
    )
    # Only brand-new worktree generated V3 outputs are cleared. Published evidence
    # and every historical file remain in the original checkout and Git history.
    shutil.rmtree(destination / "results/v3")
    (destination / "results/v3/aggregate").mkdir(parents=True)
    (destination / "data").mkdir(exist_ok=True)
    for folder in ["raw", "processed"]:
        (destination / "data" / folder).symlink_to(
            source / "data" / folder, target_is_directory=True
        )
    return destination


def main():
    p = argparse.ArgumentParser()
    p.add_argument("destination", type=Path)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    destination = prepare_fresh(Path.cwd(), args.destination)
    print("Fresh isolated workspace:", destination)
    if args.run:
        subprocess.run(["sh", "scripts/v3/run.sh"], cwd=destination, check=True)
    else:
        print(
            "Execute sh scripts/v3/run.sh from this new workspace to refit all registered experiments."
        )


if __name__ == "__main__":
    main()
