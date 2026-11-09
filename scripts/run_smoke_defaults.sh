#!/usr/bin/env bash
# Базовый прогон SkipGCN + CF-MSR на всех variant×task (дефолтные HP).
# Возобновляемый: после выключения продолжает с первого незавершённого.
# FAILED/RUNNING runs в MLflow удаляются автоматически.
#
# Usage:
#   bash scripts/run_smoke_defaults.sh
#   bash scripts/run_smoke_defaults.sh --list
#   bash scripts/run_smoke_defaults.sh --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

echo "=== Smoke defaults (resumable): 2×4×7 ==="
exec "${PYTHON}" scripts/batch_runner.py smoke "$@"
