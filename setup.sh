#!/usr/bin/env bash
set -euo pipefail

TEMP=$(getopt -o h --long help,new-env,basic,flash-attn,cumesh,o-voxel,flexgemm,nvdiffrast,nvdiffrec -n 'setup.sh' -- "$@")

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
FLASHATTN=false
CUMESH=false
OVOXEL=false
FLEXGEMM=false
NVDIFFRAST=false
NVDIFFREC=false
ERROR=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --cumesh) CUMESH=true ; shift ;;
        --o-voxel) OVOXEL=true ; shift ;;
        --flexgemm) FLEXGEMM=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --nvdiffrec) NVDIFFREC=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a new uv virtual environment (.venv)"
    echo "  --basic                 Install base TRELLIS.2 dependencies"
    echo "  --flash-attn            Install flash-attention"
    echo "  --cumesh                Install cumesh"
    echo "  --o-voxel               Install o-voxel"
    echo "  --flexgemm              Install flexgemm"
    echo "  --nvdiffrast            Install nvdiffrast"
    echo "  --nvdiffrec             Install nvdiffrec"
    exit 0
fi

if ! command -v uv > /dev/null; then
    echo "Error: uv is required but was not found in PATH"
    exit 1
fi

WORKDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_PATH="$WORKDIR/.venv"
PYTHON_BIN="$VENV_PATH/bin/python"

create_venv() {
    if [ -d "$VENV_PATH" ]; then
        rm -rf "$VENV_PATH"
    fi
    uv venv --python 3.10 "$VENV_PATH"
}

ensure_venv() {
    if [ ! -x "$PYTHON_BIN" ]; then
        uv venv --python 3.10 "$VENV_PATH"
    fi
}

if command -v nvidia-smi > /dev/null; then
    PLATFORM="cuda"
elif command -v rocminfo > /dev/null; then
    PLATFORM="hip"
else
    echo "Error: No supported GPU found"
    exit 1
fi

cd "$WORKDIR"

if [ "$PLATFORM" = "cuda" ] ; then
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] ; then
        export TORCH_CUDA_ARCH_LIST="8.9"
        echo "[CUDA] TORCH_CUDA_ARCH_LIST not set, defaulting to $TORCH_CUDA_ARCH_LIST"
    fi
    if [ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ] ; then
        export FLASH_ATTN_CUDA_ARCHS="80"
        echo "[CUDA] FLASH_ATTN_CUDA_ARCHS not set, defaulting to $FLASH_ATTN_CUDA_ARCHS"
    fi

    if [ -z "${CUDA_HOME:-}" ] ; then
        if command -v nvcc > /dev/null; then
            NVCC_BIN=$(readlink -f "$(command -v nvcc)")
            NVCC_ROOT=$(dirname "$(dirname "$NVCC_BIN")")
            if [ -f "$NVCC_ROOT/include/cuda_runtime.h" ] ; then
                export CUDA_HOME="$NVCC_ROOT"
            fi
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

        CUDA_LIBCUDA_RUNTIME_DIR=""
        for d in \
            /run/opengl-driver/lib \
            /nix/store/*-graphics-drivers/lib \
            /nix/store/*-nvidia-x11-*/lib \
            "$CUDA_HOME/lib64" \
            "$CUDA_HOME/lib" ; do
            if [ -f "$d/libcuda.so.1" ] || [ -f "$d/libcuda.so" ] ; then
                CUDA_LIBCUDA_RUNTIME_DIR="$d"
                break
            fi
        done

        CUDA_LIBCUDA_DIR="$CUDA_LIBCUDA_RUNTIME_DIR"
        if [ -z "$CUDA_LIBCUDA_DIR" ] ; then
            for d in \
                "$CUDA_HOME/lib/stubs" \
                "$CUDA_HOME/lib64/stubs" ; do
                if [ -f "$d/libcuda.so" ] ; then
                    CUDA_LIBCUDA_DIR="$d"
                    break
                fi
            done
        fi

        if [ -n "$CUDA_LIBCUDA_DIR" ] ; then
            export LD_LIBRARY_PATH="$CUDA_LIBCUDA_DIR:${LD_LIBRARY_PATH:-}"
            export LIBRARY_PATH="$CUDA_LIBCUDA_DIR:${LIBRARY_PATH:-}"
            export LDFLAGS="-L$CUDA_LIBCUDA_DIR ${LDFLAGS:-}"
            echo "[CUDA] Using libcuda from $CUDA_LIBCUDA_DIR"
        else
            echo "[CUDA] Warning: libcuda.so not found in common locations."
        fi

        if [ -n "$CUDA_LIBCUDA_RUNTIME_DIR" ] ; then
            export TRITON_LIBCUDA_PATH="$CUDA_LIBCUDA_RUNTIME_DIR"
            echo "[CUDA] Using TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH"
        fi

        echo "[CUDA] Using CUDA_HOME=$CUDA_HOME"
    else
        echo "[CUDA] Warning: CUDA_HOME not set and could not be auto-detected."
    fi
fi

if [ "$NEW_ENV" = true ] ; then
    create_venv
fi

if [ "$BASIC" = true ] ; then
    ensure_venv
    uv sync --python "$PYTHON_BIN" --no-default-groups
fi

if [ "$FLASHATTN" = true ] ; then
    ensure_venv
    if [ "$PLATFORM" = "cuda" ] ; then
        uv sync --python "$PYTHON_BIN" --no-default-groups
        uv pip install --python "$PYTHON_BIN" psutil wheel packaging ninja
        uv pip install --python "$PYTHON_BIN" --no-build-isolation flash-attn==2.7.3
    elif [ "$PLATFORM" = "hip" ] ; then
        echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
        uv sync --python "$PYTHON_BIN" --no-default-groups
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/flash-attention
        git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
        cd /tmp/extensions/flash-attention
        git checkout tags/v2.7.3-cktile
        GPU_ARCHS=gfx942 uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/flash-attention
        cd "$WORKDIR"
    else
        echo "[FLASHATTN] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFRAST" = true ] ; then
    ensure_venv
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/nvdiffrast
        git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
        uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/nvdiffrast
    else
        echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$NVDIFFREC" = true ] ; then
    ensure_venv
    if [ "$PLATFORM" = "cuda" ] ; then
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/nvdiffrec
        git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec
        uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/nvdiffrec
    else
        echo "[NVDIFFREC] Unsupported platform: $PLATFORM"
    fi
fi

if [ "$CUMESH" = true ] ; then
    ensure_venv
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/CuMesh
    git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive
    uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/CuMesh
fi

if [ "$FLEXGEMM" = true ] ; then
    ensure_venv
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/FlexGEMM
    git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive
    uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/FlexGEMM
fi

if [ "$OVOXEL" = true ] ; then
    ensure_venv
    if [ ! -f "$WORKDIR/o-voxel/third_party/eigen/Eigen/Dense" ] ; then
        echo "[OVOXEL] Eigen headers missing. Initializing submodule..."
        if git -C "$WORKDIR" rev-parse --is-inside-work-tree > /dev/null 2>&1 ; then
            git -C "$WORKDIR" submodule update --init --depth 1 o-voxel/third_party/eigen || true
        fi
    fi
    if [ ! -f "$WORKDIR/o-voxel/third_party/eigen/Eigen/Dense" ] ; then
        echo "[OVOXEL] Submodule unavailable. Cloning Eigen directly..."
        rm -rf "$WORKDIR/o-voxel/third_party/eigen"
        git clone --depth 1 https://gitlab.com/libeigen/eigen.git "$WORKDIR/o-voxel/third_party/eigen"
    fi
    if [ ! -f "$WORKDIR/o-voxel/third_party/eigen/Eigen/Dense" ] ; then
        echo "[OVOXEL] Error: Eigen headers were not found after setup."
        exit 1
    fi
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/o-voxel
    cp -r "$WORKDIR/o-voxel" /tmp/extensions/o-voxel
    uv pip install --python "$PYTHON_BIN" --no-build-isolation /tmp/extensions/o-voxel
fi

echo "Done. Activate with: source .venv/bin/activate"
