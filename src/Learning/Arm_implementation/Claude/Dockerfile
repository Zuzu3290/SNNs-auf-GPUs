# =============================================================================
# docker/Dockerfile  —  Event Camera SNN Pipeline Container
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): compile C + ARM ASM
#   Stage 2 (runtime): minimal image with compiled binaries + Python env
#
# Build:
#   docker build -t event-camera-snn:latest -f docker/Dockerfile .
#
# Run (GPU):
#   docker run --rm --gpus all \
#     --device=/dev/bus/usb \          # for real cameras
#     -v $(pwd):/workspace \
#     event-camera-snn:latest make run
#
# Run (CPU / mock only):
#   docker run --rm event-camera-snn:latest make run
#
# For ARM64 (e.g. Jetson Orin):
#   docker buildx build --platform linux/arm64 \
#     -t event-camera-snn:arm64 -f docker/Dockerfile .
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder (compiles C and ARM assembly)
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    cmake \
    ninja-build \
    gcc-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Build C and ARM assembly (native arch)
RUN make dirs asm c 2>&1 | tail -20

# For cross-compilation to ARM64 from x86 host:
# RUN make dirs asm c CROSS=1 2>&1 | tail -20


# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libusb-1.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash pipeline
RUN usermod -aG plugdev pipeline

WORKDIR /workspace

# Copy compiled binaries from builder
COPY --from=builder /build/build ./build

# Copy Python sources
COPY requirements.txt .
COPY pipeline.py .
COPY src/ ./src/
COPY asm/ ./asm/
COPY include/ ./include/
COPY scripts/ ./scripts/
COPY Makefile .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        numpy \
        snntorch \
        matplotlib \
    # PyTorch installed separately for CUDA version matching
    && pip3 install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cu121

# udev rules for cameras (available when host /dev mounted)
COPY docker/udev/ /etc/udev/rules.d/

# Switch to non-root for runtime
USER pipeline

# Default command: run mock pipeline
CMD ["make", "run"]

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="Event Camera SNN Pipeline"
LABEL org.opencontainers.image.description="Event camera to SNN pipeline with ARM NEON kernels"
LABEL org.opencontainers.image.authors="pipeline@example.com"


# =============================================================================
# ARM64-native variant (Jetson / Raspberry Pi 5)
# Build: docker buildx build --platform linux/arm64 -t event-camera-snn:arm64 .
# =============================================================================
# If building natively on ARM64:
#   FROM nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3  (Jetson JetPack 5.x)
#   or
#   FROM arm64v8/ubuntu:22.04
