#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: data_toolkit/setup.sh"
    echo "Installs the data-toolkit dependency group into .venv using uv."
    exit 0
fi

if ! command -v uv > /dev/null; then
    echo "Error: uv is required but was not found in PATH"
    exit 1
fi

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_PATH="$WORKDIR/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    uv venv --python 3.10 "$VENV_PATH"
fi

cd "$WORKDIR"
uv sync --python "$PYTHON_BIN" --no-default-groups --group data-toolkit

echo "Data toolkit dependencies installed. Activate with: source .venv/bin/activate"
