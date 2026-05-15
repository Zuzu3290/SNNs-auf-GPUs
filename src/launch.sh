#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Multi-Node DDP Launch Script
#
# All tunables are read from environment variables with sane defaults.
# Pass extra flags (e.g. --epochs 20) as positional arguments.
#
# Usage:
#   # Node 0
#   MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash scripts/launch.sh --epochs 20
#
#   # Node 1
#   MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash scripts/launch.sh --epochs 20
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12355}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "──────────────────────────────────────────"
echo "  Nodes        : ${NNODES}"
echo "  GPUs/node    : ${NPROC_PER_NODE}"
echo "  Node rank    : ${NODE_RANK}"
echo "  Master addr  : ${MASTER_ADDR}"
echo "  Master port  : ${MASTER_PORT}"
echo "──────────────────────────────────────────"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_DIR}/train.py" "$@"

# ──────────────────────────────────────────────────────────────────────
# Quick single-node test (uncomment to use):
#
#   torchrun --standalone --nproc_per_node=1 "${PROJECT_DIR}/train.py" \
#       --epochs 2 --batch_size 32
# ──────────────────────────────────────────────────────────────────────

## Read to know how to launch scripts with DDP. 
# https://towardsdatascience.com/building-a-production-grade-multi-node-training-pipeline-with-pytorch-ddp/
## 