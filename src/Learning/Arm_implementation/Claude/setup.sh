#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh  —  Full system setup for Event Camera SNN Pipeline
# =============================================================================
# Installs all system-level dependencies:
#   - Build tools (gcc, binutils for ARM, make)
#   - CUDA toolkit
#   - libcaer (iniVation cameras)
#   - Metavision SDK stub + udev rules
#   - Python 3.10+ with venv
#
# Run once on a fresh Ubuntu 22.04 / Debian 12 machine:
#   sudo bash scripts/setup.sh
#
# Tested on:
#   - Ubuntu 22.04 (x86_64, aarch64)
#   - Debian 12 (aarch64 — Raspberry Pi 5, Jetson AGX Orin)
# =============================================================================

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[SETUP]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Root check ───────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || error "Run as root: sudo bash scripts/setup.sh"

ARCH=$(uname -m)
DISTRO=$(. /etc/os-release && echo "$ID")
info "Arch: $ARCH  |  Distro: $DISTRO"

# =============================================================================
# 1. Base build tools
# =============================================================================
info "Installing base build tools..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential \
    make \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    libusb-1.0-0-dev \
    libboost-all-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev
ok "Base tools installed"

# =============================================================================
# 2. ARM cross-compilation tools (skip if already on ARM)
# =============================================================================
if [[ "$ARCH" == "x86_64" ]]; then
    info "Installing ARM64 cross-compiler (aarch64-linux-gnu)..."
    apt-get install -y --no-install-recommends \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        binutils-aarch64-linux-gnu
    ok "ARM cross-compiler installed: $(aarch64-linux-gnu-gcc --version | head -1)"
else
    ok "Running native ARM — cross-compiler not needed"
fi

# =============================================================================
# 3. CUDA Toolkit (skip on non-NVIDIA systems)
# =============================================================================
info "Checking for NVIDIA GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "GPU found: $GPU"

    if ! command -v nvcc &>/dev/null; then
        info "Installing CUDA toolkit 12.x..."
        if [[ "$DISTRO" == "ubuntu" ]]; then
            wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
            dpkg -i cuda-keyring_1.1-1_all.deb
            apt-get update -qq
            apt-get install -y cuda-toolkit-12-3
            rm cuda-keyring_1.1-1_all.deb

            # Add to PATH
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/environment
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/environment
        else
            warn "CUDA auto-install only supported on Ubuntu — install manually from:"
            warn "  https://developer.nvidia.com/cuda-downloads"
        fi
        ok "CUDA toolkit installed"
    else
        ok "CUDA already installed: $(nvcc --version | head -1)"
    fi
else
    warn "No NVIDIA GPU detected — CUDA step skipped"
    warn "  For Jetson devices: CUDA is pre-installed via JetPack"
fi

# =============================================================================
# 4. libcaer — iniVation DAVIS/DVS camera driver
# =============================================================================
info "Installing libcaer (iniVation cameras)..."

LIBCAER_VERSION="3.3.14"
LIBCAER_URL="https://github.com/inivation/libcaer/archive/refs/tags/${LIBCAER_VERSION}.tar.gz"

if ! pkg-config --exists libcaer 2>/dev/null; then
    apt-get install -y --no-install-recommends \
        libserialport-dev libopencv-dev

    TMP=$(mktemp -d)
    info "Downloading libcaer ${LIBCAER_VERSION}..."
    curl -sL "$LIBCAER_URL" | tar xz -C "$TMP"
    cmake -S "$TMP/libcaer-${LIBCAER_VERSION}" \
          -B "$TMP/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_OPENCV=OFF \
          -G Ninja
    ninja -C "$TMP/build"
    ninja -C "$TMP/build" install
    ldconfig
    rm -rf "$TMP"
    ok "libcaer ${LIBCAER_VERSION} installed"
else
    ok "libcaer already installed: $(pkg-config --modversion libcaer)"
fi

# =============================================================================
# 5. udev rules for event cameras
# =============================================================================
info "Installing udev rules for event cameras..."

# iniVation DAVIS / DVS cameras
cat > /etc/udev/rules.d/65-inivation.rules << 'EOF'
# iniVation event cameras
SUBSYSTEM=="usb", ATTR{idVendor}=="152a", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", MODE="0666", GROUP="plugdev"
EOF

# Prophesee cameras
cat > /etc/udev/rules.d/66-prophesee.rules << 'EOF'
# Prophesee event cameras
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTR{idVendor}=="03fd", MODE="0666", GROUP="plugdev"
# EVK4
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", ATTR{idProduct}=="00f3", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger
ok "udev rules installed"

# Add current user to plugdev
if [[ -n "${SUDO_USER:-}" ]]; then
    usermod -aG plugdev "$SUDO_USER"
    ok "Added $SUDO_USER to plugdev group"
fi

# =============================================================================
# 6. Prophesee Metavision SDK (stub install — full SDK requires registration)
# =============================================================================
info "Metavision SDK setup..."
warn "Prophesee Metavision SDK requires registration at:"
warn "  https://docs.prophesee.ai/stable/installation/index.html"
warn "After downloading, run:"
warn "  sudo apt install ./metavision-sdk-*.deb"
warn ""
warn "For now, installing open-source Metavision stub (metavision-sdk-core)..."

# The open-source components are available via pip
# (full HAL/driver requires the proprietary installer)
pip3 install --quiet metavision-sdk-core 2>/dev/null && \
    ok "metavision-sdk-core (open-source) installed via pip" || \
    warn "metavision-sdk-core pip install failed — install SDK manually"

# =============================================================================
# 7. Python 3.10+ check
# =============================================================================
info "Checking Python version..."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [[ $PY_MAJOR -lt 3 || ($PY_MAJOR -eq 3 && $PY_MINOR -lt 9) ]]; then
    info "Python $PY_VER is too old — installing Python 3.11..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.11 python3.11-venv python3.11-dev
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
else
    ok "Python $PY_VER OK"
fi

# =============================================================================
# 8. Summary
# =============================================================================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  System setup complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "  Next steps:"
echo "    make env         # create Python venv + install packages"
echo "    make all         # build C and ARM assembly"
echo "    make run         # run pipeline with mock camera"
echo ""
echo "  For real cameras:"
echo "    iniVation:   plug in USB, run: make run-pyaer"
echo "    Prophesee:   install SDK, run: make run-mv"
echo ""
[[ -n "${SUDO_USER:-}" ]] && \
    echo -e "${YELLOW}  NOTE: Log out and back in for plugdev group to take effect.${NC}" && echo ""
