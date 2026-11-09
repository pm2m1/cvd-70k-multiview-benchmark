"""Inspect and extract an authorized archive; freeze split membership."""

import argparse
from pathlib import Path

from cvd70k.data.pipeline import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    _, _, audit = prepare(Path.cwd(), args.archive)
    print({k: audit[k] for k in ["shape", "class_counts", "strict_retained", "partitions"]})
