#!/usr/bin/env bash
# Логистическая регрессия на характеристиках графов (4 variant × 7 task, test outer 0):
#   block_means      — средние веса внутри/между блоками узлов;
#   strength_summary — сводки узловых сил s, s+, s−, |s| и энтропия |s|;
#   strength_vector  — вектор узловых сил;
#   all              — все три группы вместе.
# C подбирается по GroupKFold (по субъектам) на train pool. ROC-AUC [95% CI] —
# кластерный bootstrap по субъектам, парные ΔAUC и поправки на множественность.
# Первый запуск ~5–10 мин (чтение 46 тыс. графов), затем признаки берутся из кэша.
#
# Usage:
#   bash scripts/run_graph_features_logreg.sh
#   bash scripts/run_graph_features_logreg.sh --task EMOTION --task MOTOR
#   bash scripts/run_graph_features_logreg.sh --feature-set block_means+strength_summary
#   bash scripts/run_graph_features_logreg.sh --partition-csv conf/yeo7.csv   # mmp_id,network
#   N_SPATIAL=8 N_BOOTSTRAP=20000 REFERENCE=none bash scripts/run_graph_features_logreg.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
REFERENCE="${REFERENCE:-correlation}"
ALPHA="${ALPHA:-0.05}"
N_SPATIAL="${N_SPATIAL:-6}"
N_JOBS="${N_JOBS:-1}"

echo "=== Graph features + LogReg: 4 variant × 7 task, блоков коры ${N_SPATIAL}×2, эталон ${REFERENCE} ==="
exec "${PYTHON}" scripts/graph_features_logreg.py \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --reference "${REFERENCE}" \
  --alpha "${ALPHA}" \
  --n-spatial "${N_SPATIAL}" \
  --n-jobs "${N_JOBS}" \
  "$@"
