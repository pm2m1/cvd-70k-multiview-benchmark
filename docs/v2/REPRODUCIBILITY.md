# Reproduce V2 without overwriting historical evidence

Python3.12; separate locked environment `configs/v2/pyproject.toml` and `configs/v2/uv.lock`.
The original root pyproject/uv.lock and V1 files remain unchanged. All new learned artifacts,
row assignments and predictions live under ignored `private/v2/`. Kaggle raw data stay in
ignored `data/raw/`; obtain them yourself under applicable terms. Official metadata remains
Unknown licence; raw redistribution is not authorized here. External UCI files are private
feasibility input only. No Phase1 private participant/checkpoint access.

```sh
uv sync --project configs/v2 --frozen
uv run --project configs/v2 ruff check .
uv run --project configs/v2 ruff format --check .
uv run --project configs/v2 pytest -q
```

Before a fresh study, run the original preparation command on the authorized exact archive
in a **new local checkout**, checking its expected hash. Existing assignments are never
regenerated in place. V2 eligibility is the original V1 train partition only. Full V2 stages:

```sh
uv run --project configs/v2 python scripts/v2/audit_v1.py
uv run --project configs/v2 python scripts/v2/probe.py
uv run --project configs/v2 python -m cvd70k.v2.optimization develop
uv run --project configs/v2 python -m cvd70k.v2.transfer develop
uv run --project configs/v2 python -m cvd70k.v2.optimization evaluate
uv run --project configs/v2 python -m cvd70k.v2.transfer evaluate
uv run --project configs/v2 python -m cvd70k.v2.reporting
```

Released access/lock records are immutable historical outputs. Use `scripts/v2/reproduce.py`
for a new timestamped ignored run workspace rather than replacing them. It copies V1 data
instructions/artifacts and new code/configuration, verifies data/assignment hashes, and uses
local raw-data/split references without publishing them. It does not retrain/evaluate the
V1 test. The full workflow is finite and resumes completed trial IDs only when config,
membership and checkpoint hashes match. Do not alter configurations halfway through a run.
Inner selections and all refits lock before outer inference; transfer masks/checkpoints lock
before target evaluation. Both development studies finish before either evaluation opens.

MPS is validated for original MLP/five-view operations. Official FT embeddings lack a
strictly deterministic MPS backward; TabNet hit a CPU/MPS mismatch. Both passed CPU fallbacks.
No arbitrary nondeterminism workaround was enabled. The installed TabPFN code/model versions
were inspected, but authenticated licence acceptance was not verified, so no weights were
requested. Do not disable sample guards or upload weights. Native XGBoost runs in a separate
process to avoid the known macOS Torch/OpenMP conflict. MPS/CPU/two-worker timings and library
versions are recorded; cross-device bitwise reproducibility is not guaranteed.

Main model seed list101/202/303; outer711; calibration800+outer; inner900+outer; source/target
role seeds1201/1202. Target paired class permutations seed and seed+10000 for separate roles.
Source geometry floor.1, target/source minimum5/class, permutation/bootstrap199, Monte Carlo
p denominator200, logical thresholds U<.75 OR(p≤.05 AND R<.8 AND U<1), invalid→SHARE.
Epochzero is eligible for all main adaptation models; the explicitly labeled AP/nozero
engineering ablation is deliberately unprotected, never the main policy.

Conditional uncertainty uses1,000 within-fold/group resamples for predictive comparisons,
and group resamples on a fixed target evaluation for transfer. Shared folds, people and source
fits remain dependent. Trials/curves are aggregate model records; no row score is published.
CPU CI uses invented fixtures and released aggregate integrity checks, with private-data tests
skipped when data are absent. New V2 workflow installs only the separate environment.

Before outer evaluation, three additional selected-five-view diagnostic refits (one inner
fold per outer fold, seed101) record training AP/AUROC, which the primary engine did not log.
Run `uv run --project configs/v2 python -m cvd70k.v2.learning_diagnostics` after optimization
development. A single-process eval/no-grad monitor records train metrics without changing
optimizer/checkpoint logic; actual weight differences from original inner checkpoints are
reported. These are three measurement refits, not additional hyperparameter candidates.
