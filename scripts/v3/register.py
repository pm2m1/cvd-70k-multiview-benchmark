"""Record a fresh execution prospectively, preserving any existing registration."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cvd70k.utils.io import read_json, sha256, write_json


def main():
    root = Path.cwd()
    pub = root / "results/v3/aggregate"
    path = pub / "execution_manifest.json"
    hashes = {
        "protocol_sha256": sha256(root / "docs/v3/RESEARCH_PROTOCOL.md"),
        "configuration_sha256": sha256(root / "configs/v3/study.json"),
        "lock_sha256": sha256(root / "configs/v3/uv.lock"),
    }
    if path.exists():
        old = read_json(path)
        if any(old[k] != value for k, value in hashes.items()):
            raise ValueError("Registered scientific inputs changed")
        print("Existing execution registration preserved")
        return
    dependencies = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", "configs/v3/.venv/bin/python"], text=True
    )
    write_json(
        path,
        dict(
            registered_utc=datetime.now(timezone.utc).isoformat(),
            base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            **hashes,
            source_sha256={
                str(p.relative_to(root)): sha256(p) for p in (root / "src/cvd70k/v3").glob("*.py")
            },
            budget=375,
            hard_ceiling=400,
            dependencies=dependencies.replace(str(root), "<PROJECT_ROOT>").splitlines(),
        ),
    )
    print("Fresh execution registered before model fitting")


if __name__ == "__main__":
    main()
