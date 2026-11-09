#!/bin/sh
set -eu
uv sync --project configs/v3 --frozen
uv run --project configs/v3 python -m cvd70k.v3.design
uv run --project configs/v3 python -m cvd70k.v3.runner probe
uv run --project configs/v3 python -m cvd70k.v3.runner classification
uv run --project configs/v3 python -m cvd70k.v3.runner transfer
uv run --project configs/v3 python -m cvd70k.v3.runner evaluate_classification
uv run --project configs/v3 python -m cvd70k.v3.runner evaluate_transfer
uv run --project configs/v3 python scripts/v3/analyze.py
