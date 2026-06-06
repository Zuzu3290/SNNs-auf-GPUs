# SNN CUDA Kernel Build Environment
# Uses PyTorch official image with CUDA 12.1 and cuDNN8
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install build essentials and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    cmake \
    ninja-build \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA and cuDNN installation
RUN echo "CUDA Version:" && nvcc --version && \
    echo "CUDA Path:" && ls -la /usr/local/cuda/lib64/ | grep cublas

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (will be mounted in Docker compose or run command)
# COPY . .

# Build command (will be executed via docker run)
CMD ["/bin/bash"]
