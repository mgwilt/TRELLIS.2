#!/usr/bin/env bash
set -euo pipefail

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="$WORKDIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ] ; then
    echo "Error: $PYTHON_BIN not found."
    echo "Run ./setup_all.sh first."
    exit 1
fi

if command -v nvidia-smi > /dev/null ; then
    if [ -z "${CUDA_HOME:-}" ] && command -v nvcc > /dev/null ; then
        NVCC_BIN=$(readlink -f "$(command -v nvcc)")
        NVCC_ROOT=$(dirname "$(dirname "$NVCC_BIN")")
        if [ -f "$NVCC_ROOT/include/cuda_runtime.h" ] ; then
            export CUDA_HOME="$NVCC_ROOT"
        fi
    fi

    if [ -z "${CUDA_HOME:-}" ] ; then
        for d in /nix/store/*-cuda-merged-* /usr/local/cuda /usr/local/cuda-* ; do
            if [ -d "$d" ] && [ -f "$d/include/cuda_runtime.h" ] ; then
                export CUDA_HOME="$d"
                break
            fi
        done
    fi

    if [ -n "${CUDA_HOME:-}" ] ; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export CPATH="$CUDA_HOME/include:${CPATH:-}"
        if [ -d "$CUDA_HOME/lib64" ] ; then
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        elif [ -d "$CUDA_HOME/lib" ] ; then
            export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
        fi
    fi

    if [ -z "${TRITON_LIBCUDA_PATH:-}" ] ; then
        for d in \
            /run/opengl-driver/lib \
            /nix/store/*-graphics-drivers/lib \
            /nix/store/*-nvidia-x11-*/lib \
            "${CUDA_HOME:-}/lib64" \
            "${CUDA_HOME:-}/lib" ; do
            if [ -f "$d/libcuda.so.1" ] || [ -f "$d/libcuda.so" ] ; then
                export TRITON_LIBCUDA_PATH="$d"
                break
            fi
        done
    fi

    if [ -n "${TRITON_LIBCUDA_PATH:-}" ] ; then
        export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:${LD_LIBRARY_PATH:-}"
    else
        echo "[WARN] TRITON_LIBCUDA_PATH could not be auto-detected."
    fi
fi

export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRELLIS_MODEL_DIR="${TRELLIS_MODEL_DIR:-$WORKDIR/models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$WORKDIR/.hf/hub}"

cd "$WORKDIR"
if [ "${1:-}" = "--check" ] ; then
    exec "$PYTHON_BIN" - <<'PY'
import importlib
mods = ["torch", "flash_attn", "nvdiffrast.torch", "cumesh", "flex_gemm", "o_voxel"]
for mod in mods:
    importlib.import_module(mod)
    print(f"OK {mod}")
PY
fi
exec "$PYTHON_BIN" app.py "$@"
