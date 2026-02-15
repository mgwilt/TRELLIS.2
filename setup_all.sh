#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: ./setup_all.sh"
    echo "Runs the full TRELLIS.2 setup with uv, including all optional extensions."
    exit 0
fi

if [ ! -x "$ROOT_DIR/setup.sh" ]; then
    echo "Error: $ROOT_DIR/setup.sh not found or not executable."
    exit 1
fi

exec "$ROOT_DIR/setup.sh" \
    --new-env \
    --basic \
    --flash-attn \
    --nvdiffrast \
    --nvdiffrec \
    --cumesh \
    --o-voxel \
    --flexgemm
