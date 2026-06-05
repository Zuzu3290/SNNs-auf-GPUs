#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SNN Training Launcher
# Framework is set via training.framework in configuration/SNN_module.yaml
#
# Usage (local):
#   ./launch.sh
#
# Usage (Colab):
#   !bash /content/SNNs-auf-GPUs/launch.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make project root (skeleton, event_data_workflow) and src/ (learning, compiler) importable.
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "──────────────────────────────────────────"
echo "  SNN Training Launcher"
echo "  Framework set via:"
echo "  configuration/SNN_module.yaml"
echo "  PYTHONPATH : $PYTHONPATH"
echo "──────────────────────────────────────────"

python3 "$PROJECT_ROOT/src/learning/main.py"
