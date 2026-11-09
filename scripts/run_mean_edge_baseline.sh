#!/usr/bin/env bash
# Классификатор «средний вес ребра» на 4 variant × 7 task, test outer 0.
# Сравнение вариантов: ROC-AUC [95% CI] (кластерный bootstrap по субъектам),
# парные ΔAUC, Romano–Wolf / Holm / BH. Без обучения GNN, ~2–4 мин.
#
# Usage:
#   bash scripts/run_mean_edge_baseline.sh
#   bash scripts/run_mean_edge_baseline.sh --task EMOTION --task MOTOR
#   bash scripts/run_mean_edge_baseline.sh --no-cache
#   N_BOOTSTRAP=20000 REFERENCE=none bash scripts/run_mean_edge_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
REFERENCE="${REFERENCE:-correlation}"
ALPHA="${ALPHA:-0.05}"

echo "=== Mean-edge baseline: 4 variant × 7 task, эталон ${REFERENCE} ==="
exec "${PYTHON}" scripts/mean_edge_baseline.py \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --reference "${REFERENCE}" \
  --alpha "${ALPHA}" \
  "$@"
