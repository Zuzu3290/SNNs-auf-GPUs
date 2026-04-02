# =============================================================================
# Event Camera SNN Pipeline — Master Makefile
# =============================================================================
# Targets:
#   make all          — build everything (C + ARM asm + Python check)
#   make asm          — assemble ARM modules only
#   make c            — compile C modules only
#   make python       — verify Python environment
#   make run          — run full pipeline (mock camera)
#   make run-pyaer    — run with iniVation hardware
#   make run-mv       — run with Prophesee hardware
#   make test         — run all unit tests
#   make docker       — build Docker image
#   make clean        — remove all build artifacts
#   make env          — create Python venv + install deps
#   make setup        — full system setup (apt + Python + build)
# =============================================================================

# ---------------------------------------------------------------------------
# Toolchain configuration
# ---------------------------------------------------------------------------
CC          := gcc
AS          := as
LD          := ld
PYTHON      := python3
PIP         := pip3

# ARM cross-compile (set CROSS=1 to target ARM from x86)
ifdef CROSS
  CC  := aarch64-linux-gnu-gcc
  AS  := aarch64-linux-gnu-as
  LD  := aarch64-linux-gnu-ld
endif

ARCH        := $(shell uname -m)
OS          := $(shell uname -s)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
SRC_DIR     := src
ASM_DIR     := asm
INC_DIR     := include
BUILD_DIR   := build
OBJ_DIR     := $(BUILD_DIR)/obj
BIN_DIR     := $(BUILD_DIR)/bin
LIB_DIR     := $(BUILD_DIR)/lib
VENV_DIR    := .venv

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
CFLAGS      := -O2 -Wall -Wextra -std=c11 -I$(INC_DIR)
CFLAGS_ARM  := -march=armv8-a+simd -mfpu=neon-fp-armv8

# ARM assembler flags
ASFLAGS     := --warn --fatal-warnings

ifdef DEBUG
  CFLAGS    += -g -DDEBUG -O0
endif

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
C_SRCS      := $(wildcard $(SRC_DIR)/**/*.c) $(wildcard $(SRC_DIR)/*.c)
ASM_SRCS    := $(wildcard $(ASM_DIR)/*.s)

C_OBJS      := $(patsubst $(SRC_DIR)/%.c,  $(OBJ_DIR)/%.o, $(C_SRCS))
ASM_OBJS    := $(patsubst $(ASM_DIR)/%.s,  $(OBJ_DIR)/asm_%.o, $(ASM_SRCS))

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: all asm c python run run-pyaer run-mv test docker clean env setup \
        lint format help dirs info

# ---------------------------------------------------------------------------
# Default
# ---------------------------------------------------------------------------
all: dirs asm c python
	@echo ""
	@echo "  ✓ Build complete"
	@echo "  Run:  make run"
	@echo ""

# ---------------------------------------------------------------------------
# Create build directories
# ---------------------------------------------------------------------------
dirs:
	@mkdir -p $(OBJ_DIR)/camera $(OBJ_DIR)/driver $(OBJ_DIR)/buffer \
	           $(OBJ_DIR)/kernel $(OBJ_DIR)/snn $(OBJ_DIR)/output \
	           $(BIN_DIR) $(LIB_DIR)

# ---------------------------------------------------------------------------
# ARM Assembly
# ---------------------------------------------------------------------------
asm: dirs $(ASM_OBJS)
	@echo "  [ASM] ARM assembly complete — $(words $(ASM_OBJS)) object(s)"

$(OBJ_DIR)/asm_%.o: $(ASM_DIR)/%.s
	@echo "  [AS]  $<"
	$(AS) $(ASFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# C compilation
# ---------------------------------------------------------------------------
c: dirs $(C_OBJS)
	@echo "  [C]   C compilation complete — $(words $(C_OBJS)) object(s)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "  [CC]  $<"
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# ---------------------------------------------------------------------------
# Link the C pipeline runner
# ---------------------------------------------------------------------------
$(BIN_DIR)/pipeline_c: $(C_OBJS) $(ASM_OBJS)
	@echo "  [LD]  linking pipeline_c"
	$(CC) $(CFLAGS) -o $@ $^ -lpthread -lm

# ---------------------------------------------------------------------------
# Python environment check
# ---------------------------------------------------------------------------
python:
	@echo "  [PY]  Checking Python environment..."
	@$(PYTHON) scripts/check_env.py
	@echo "  [PY]  Python OK"

# ---------------------------------------------------------------------------
# Run targets
# ---------------------------------------------------------------------------
run: all
	@echo "  [RUN] Starting pipeline (mock camera)..."
	$(PYTHON) pipeline.py --backend mock --n_frames 20

run-pyaer: all
	@echo "  [RUN] Starting pipeline (iniVation / pyaer)..."
	$(PYTHON) pipeline.py --backend pyaer

run-mv: all
	@echo "  [RUN] Starting pipeline (Prophesee / Metavision)..."
	$(PYTHON) pipeline.py --backend metavision

run-arm: $(BIN_DIR)/pipeline_c
	@echo "  [RUN] Starting C/ARM pipeline..."
	$(BIN_DIR)/pipeline_c --backend mock --frames 20

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: all
	@echo "  [TEST] Running unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "  [TEST] Running C tests..."
	@for t in $(wildcard tests/*.c); do \
	  $(CC) $(CFLAGS) -o $(BIN_DIR)/test_$$(basename $$t .c) $$t -lm && \
	  $(BIN_DIR)/test_$$(basename $$t .c) && \
	  echo "    PASS: $$t"; \
	done

test-asm: asm
	@echo "  [TEST] ARM assembly tests..."
	$(PYTHON) tests/test_asm_kernels.py

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
env:
	@echo "  [ENV] Creating Python venv at $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt
	@echo "  [ENV] Done. Activate with: source $(VENV_DIR)/bin/activate"

# ---------------------------------------------------------------------------
# Full system setup (run once on a fresh machine)
# ---------------------------------------------------------------------------
setup:
	@echo "  [SETUP] Running system setup..."
	bash scripts/setup.sh
	$(MAKE) env
	$(MAKE) all
	@echo "  [SETUP] System ready."

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker:
	@echo "  [DOCKER] Building image..."
	docker build -t event-camera-snn:latest -f docker/Dockerfile .

docker-run:
	docker run --rm --gpus all \
	  --device=/dev/bus/usb \
	  -v $(PWD):/workspace \
	  event-camera-snn:latest \
	  make run

docker-shell:
	docker run --rm -it --gpus all \
	  -v $(PWD):/workspace \
	  event-camera-snn:latest bash

# ---------------------------------------------------------------------------
# Lint / format
# ---------------------------------------------------------------------------
lint:
	$(PYTHON) -m flake8 src/ pipeline.py --max-line-length=100
	$(PYTHON) -m mypy src/ --ignore-missing-imports

format:
	$(PYTHON) -m black src/ pipeline.py
	clang-format -i $(wildcard $(SRC_DIR)/**/*.c) $(wildcard $(INC_DIR)/*.h)

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	@echo "  [CLEAN] Removing build artifacts..."
	rm -rf $(BUILD_DIR)
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "  [CLEAN] Done."

clean-all: clean
	rm -rf $(VENV_DIR)
	docker rmi event-camera-snn:latest 2>/dev/null || true

# ---------------------------------------------------------------------------
# Info / help
# ---------------------------------------------------------------------------
info:
	@echo "  Arch:    $(ARCH)"
	@echo "  OS:      $(OS)"
	@echo "  CC:      $(CC)"
	@echo "  AS:      $(AS)"
	@echo "  Python:  $(shell $(PYTHON) --version)"
	@echo "  CUDA:    $(shell nvcc --version 2>/dev/null | head -1 || echo 'not found')"

help:
	@echo ""
	@echo "  Event Camera SNN Pipeline — Build System"
	@echo ""
	@echo "  make all          Build everything"
	@echo "  make asm          Assemble ARM modules only"
	@echo "  make c            Compile C modules only"
	@echo "  make run          Run pipeline (mock camera)"
	@echo "  make run-pyaer    Run with iniVation camera"
	@echo "  make run-mv       Run with Prophesee camera"
	@echo "  make run-arm      Run native C/ARM binary"
	@echo "  make test         Run all tests"
	@echo "  make env          Create Python venv"
	@echo "  make setup        Full system setup"
	@echo "  make docker       Build Docker image"
	@echo "  make clean        Remove build artifacts"
	@echo "  make info         Show toolchain info"
	@echo ""
	@echo "  Options:"
	@echo "    CROSS=1         Cross-compile for ARM (aarch64)"
	@echo "    DEBUG=1         Debug build"
	@echo ""
