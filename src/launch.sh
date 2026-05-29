#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SNN Training Launcher
#
# Usage:
#   ./launch.sh              # defaults to norse
#   ./launch.sh norse
#   ./launch.sh torch
#   ./launch.sh sj
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:-norse}"

case "$MODEL" in
    norse|torch|sj) ;;
    *)
        echo "Unknown model: '$MODEL'"
        echo "Valid options: norse | torch | sj"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "──────────────────────────────────────────"
echo "  Model backend  : ${MODEL^^}"
echo "  Script         : $SCRIPT_DIR/learning/main.py"
echo "──────────────────────────────────────────"

python "$SCRIPT_DIR/learning/main.py" --model "$MODEL"
