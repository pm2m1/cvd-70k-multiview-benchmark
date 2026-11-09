# Reproduction and execution inventory

From repository root:
```sh
uv sync --project configs/v3 --frozen
uv run --project configs/v3 pytest -q
sh scripts/v3/run.sh
```

Supply the original privately downloaded ZIP and recreate V1 audited raw file/splits following V1 README before real-data execution. V3 validates the immutable V1 raw/split hashes and restricts eligibility before statistics. Do not download an alternative dataset. Local role memberships are private/v3/roles.csv; source/adapted checkpoints and per-row scores remain private/v3. Historical V1/V2 files are never written by V3. Checkpoint resume verifies specification and SHA-256; evaluation requires source/config/protocol/role/checkpoint locks. Trials and failures have separate records. Successful models are never overwritten to repair failures.

Completed successful registered predictor fits: 375. Failed trial attempts retained: 3. Summed model-fit runtime: 607.7 seconds (not wall-clock because tracks overlap). Three probe fits are included; synthetic pytest optimization is not counted as research fits. Environment and dependency versions: execution_manifest.json/environment.json and the separate configs/v3/uv.lock. MPS is validated for multiview/attention; deterministic ranking-loss indexing and FT run CPU. Native tree libraries execute in isolated CPU processes. Corrections are in execution_corrections.json.

Publication contains no raw data, per-row predictions, trained weights or private preprocessing. Kaggle metadata still reports Unknown licence; source-data redistribution rights are not assumed. Public aggregates cannot substitute for the source dataset or trained models. CI runs synthetic tests without private data; local real-data tests additionally verify memberships, hashes and numerical outputs.

## Fresh public checkout versus local resume

Published trial records are evidence, not sufficient resume caches: weights are intentionally private. With the privately audited raw data/splits available, use:

```sh
uv run --project configs/v3 python scripts/v3/reproduce.py ~/Desktop/cvd70k-v3-reproduction --run
```

The destination must not exist. This creates a detached worktree at the current commit, clears only its newly created V3 generated outputs and links the original local raw/processed directories for reading. The original checkout and all V1/V2 evidence stay untouched. The wrapper performs375 new registered fits; it was not invoked during the completed study to avoid unnecessary duplicate fitting. Direct `sh scripts/v3/run.sh` is for an originating workspace with compatible private caches or the prepared fresh worktree. Missing weights cannot be represented as completed local fits.
