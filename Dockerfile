FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=1
ARG INSTALL_FLASH_ATTN=1
ARG INSTALL_NVDIFFRAST=1
ARG INSTALL_NVDIFFREC=1
ARG INSTALL_CUMESH=1
ARG INSTALL_FLEXGEMM=1
ARG INSTALL_OVOXEL=1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    USE_NINJA=0 \
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    FORCE_CUDA=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CPLUS_INCLUDE_PATH=/usr/include/eigen3 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    git wget ca-certificates sudo \
    build-essential cmake ninja-build pkg-config \
    libjpeg-dev libpng-dev libeigen3-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && ln -sf python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# setup.sh uses nvidia-smi to detect CUDA; provide a stub during build
RUN if ! command -v nvidia-smi >/dev/null 2>&1; then \
      printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi; \
    fi

WORKDIR /app

RUN python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install -r /tmp/requirements-docker.txt

RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then pip install flash-attn==2.7.3; fi

RUN if [ "${INSTALL_NVDIFFRAST}" = "1" ]; then \
      mkdir -p /tmp/extensions && \
      git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
      pip install /tmp/extensions/nvdiffrast --no-build-isolation; \
    fi

RUN if [ "${INSTALL_NVDIFFREC}" = "1" ]; then \
      mkdir -p /tmp/extensions && \
      git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
      pip install /tmp/extensions/nvdiffrec --no-build-isolation; \
    fi

RUN if [ "${INSTALL_CUMESH}" = "1" ]; then \
      mkdir -p /tmp/extensions && \
      git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
      pip install /tmp/extensions/CuMesh --no-build-isolation; \
    fi

RUN if [ "${INSTALL_FLEXGEMM}" = "1" ]; then \
      mkdir -p /tmp/extensions && \
      git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
      pip install /tmp/extensions/FlexGEMM --no-build-isolation; \
    fi

COPY o-voxel /tmp/o-voxel
RUN if [ "${INSTALL_OVOXEL}" = "1" ]; then pip install /tmp/o-voxel --no-build-isolation; fi

COPY trellis2 /app/trellis2
COPY app.py example.py /app/
COPY assets /app/assets

EXPOSE 7860
CMD ["python", "app.py"]
