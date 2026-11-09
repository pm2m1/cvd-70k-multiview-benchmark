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
