#!/bin/sh
set -eu
uv sync --project configs/v3 --frozen
uv run --project configs/v3 python -m cvd70k.v3.design
uv run --project configs/v3 python scripts/v3/register.py
uv run --project configs/v3 python scripts/v3/audit_history.py
research_device=$(uv run --project configs/v3 python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')")
uv run --project configs/v3 python -m cvd70k.v3.runner probe
uv run --project configs/v3 python -m cvd70k.v3.runner classification --device "$research_device"
uv run --project configs/v3 python -m cvd70k.v3.runner transfer --device "$research_device"
uv run --project configs/v3 python -m cvd70k.v3.runner evaluate_classification
uv run --project configs/v3 python -m cvd70k.v3.runner evaluate_transfer
uv run --project configs/v3 python scripts/v3/profile_inference.py
uv run --project configs/v3 python scripts/v3/analyze.py
uv run --project configs/v3 python scripts/v3/refine_figures.py
