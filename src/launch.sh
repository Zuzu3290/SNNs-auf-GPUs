#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SNN Training Launcher
# Framework is set via training.framework in SNN_module.yaml
#
# Usage (local):
#   ./launch.sh
#
# Usage (Colab):
#   !bash /content/SNNs-auf-GPUs/src/launch.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Make both the project root (for 'skeleton') and src/ (for 'learning', 'compiler')
# importable without any sys.path manipulation inside Python files.
export PYTHONPATH="$PROJECT_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "──────────────────────────────────────────"
echo "  Script     : $SCRIPT_DIR/learning/main.py"
echo "  PYTHONPATH : $PYTHONPATH"
echo "──────────────────────────────────────────"

python3 "$SCRIPT_DIR/learning/main.py"
