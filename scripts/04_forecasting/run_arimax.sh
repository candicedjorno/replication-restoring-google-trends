#!/bin/bash
set -e

CONDA_BIN="$HOME/miniconda3/bin/conda"
ENV_NAME="py38-main"
ENV_FILE="environment-py38.yml"
PYTHON_SCRIPT="scripts/04_forecasting/arimax.py"

if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
	"$CONDA_BIN" env create -f "$ENV_FILE"
else
	"$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python "$PYTHON_SCRIPT"
