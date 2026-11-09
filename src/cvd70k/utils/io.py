"""Small strict-JSON and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (Path, np.ndarray)):
            return str(x) if isinstance(x, Path) else x.tolist()
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()
