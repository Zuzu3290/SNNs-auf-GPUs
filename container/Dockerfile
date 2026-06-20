# SNN CUDA Kernel Build Environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# CUDA toolkit (nvcc + headers) via NVIDIA's apt repo
RUN apt-get update && apt-get install -y wget gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN nvcc --version

WORKDIR /workspace

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision==0.25.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source is mounted at `docker run` time, not baked in — see container/docker.yaml
# COPY . .

CMD ["/bin/bash"]
