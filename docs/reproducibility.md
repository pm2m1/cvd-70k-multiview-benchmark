# Reproducibility

Use Python 3.12 and uv. All resolved versions are in `uv.lock`. On macOS XGBoost also needs
a working OpenMP runtime (the completed machine had Homebrew libomp). There is no raw-data
download in the pipeline. Obtain authorized data from the linked Kaggle source first.

```sh
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python scripts/reproduce.py --archive '/absolute/path/to/archive(1).zip'
```

The last command executes the **full** benchmark, all five seeds, both sampling analyses,
strict/BMI sensitivities, locked final evaluation, bootstrap, figures and reports. It
creates a timestamped ignored workspace under `private/reproductions/`, copying source,
configuration and protocol there and using the same environment. It preserves the published
results and their original development/test-access records. Read its new `docs/results.md`.
Do not train over the released `test_access.json` in place; that guard is intentional.

For a brand-new workspace with no prior results, the explicit stages are:

```sh
uv run python scripts/prepare.py --archive '/absolute/path/to/archive(1).zip'
uv run python scripts/probe_hardware.py
uv run pytest -q
uv run python scripts/run.py develop
uv run python scripts/run.py evaluate
uv run python scripts/report.py
```

Archive checks: path traversal, absolute paths, Windows backslash paths, symlinks,
encryption, per-entry/total decompressed byte caps, ratio cap, CRC integrity and expected
single CSV. Extraction refuses overwriting different bytes. CSV delimiter/schema are
verified, not assumed from the file extension.

Split membership is immutable, saved locally in `data/processed/splits.csv`. Public audit
and lock record its hash. Preparation is deterministic but full 20-fold grouped assignment
can take tens of seconds. Two-candidate searches use seed 101. Training resumes completed
runs with matching config/checkpoint hashes; it refuses further development after final
test access. Runtime metadata distinguishes completed fits from the initial failed native
attempt. All final models were fitted and development locked before test predictions.

PyTorch uses deterministic algorithms, explicit Python/NumPy/Torch seeds, CPU-generated
minibatch permutations and no stochastic dropout. MPS support is tested via actual finite
optimization and CPU gradient agreement; reliable MPS is preferred as requested. The tiny
hardware probe also reports timing, and CPU can be faster for this model. CPU inference
is used for neural final predictions. Cross-device/library bitwise equivalence is not
promised; CI tests deterministic CPU repeats on invented fixtures.

XGBoost and this macOS PyTorch build crash when fitted in the same process. A tested,
standalone `cvd70k.training.xgb_worker` process performs all XGBoost fitting and prediction;
it imports no Torch. Both request arrays and fitted models stay under ignored `private/`.
The workaround preserves model specification and validation-only early stopping. The
initial native failure occurred before test opening. The pipeline completed after this
fix; no failed fit is counted as a completed experiment.

Neural `.joblib` bundles contain CPU state tensors and fitted preprocessing. They are
private rather than distributed. Never load pickle/joblib files from untrusted sources.
Model checkpoint hashes and local inference-code hashes are part of the development lock.
Reporting reruns may read existing predictions without retraining or revising thresholds.
CSV metrics are aggregate per fit, with no participant rows or probabilities.

Primary bootstrap has 1,000 paired feature-group draws across five seed predictions;
imbalance intervals compare multiview and flat MLP in all 24 paired conditions. Rare-condition
intervention comparisons also use paired group draws. CI conditions on fitted models and
this single held-out population. Software versions, hardware, aggregate audit, exact
counts, effective weights, thresholds, epochs, curves and durations are machine readable.

CI uses the `macos-15` ARM64 runner with libomp to match the completed environment and
avoid installing unused Linux CUDA dependencies. Runner labels are documented by
[GitHub](https://docs.github.com/en/actions/reference/runners/github-hosted-runners). CI uses
only invented fixtures and released aggregates; it does not obtain participant data.
