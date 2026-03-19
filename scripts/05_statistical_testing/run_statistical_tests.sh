#!/bin/bash
set -e

CONDA_BIN="$HOME/miniconda3/bin/conda"
ENV_NAME="Renv433"
ENV_FILE="environment-R.yml"
RSCRIPT_FILE="scripts/05_statistical_testing/statistical_tests.R"

if ! "$CONDA_BIN" run -n "$ENV_NAME" Rscript -e "sessionInfo()" >/dev/null 2>&1; then
	"$CONDA_BIN" env create -f "$ENV_FILE"
else
	"$CONDA_BIN" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" Rscript "$RSCRIPT_FILE"
