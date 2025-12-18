FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" \
    FORCE_CUDA=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    git wget ca-certificates sudo \
    build-essential cmake ninja-build pkg-config \
    libjpeg-dev libpng-dev \
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
COPY . /app

RUN python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies and native extensions
RUN bash -lc ". ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"

EXPOSE 7860
CMD ["python", "app.py"]
