#!/usr/bin/env bash
# Классификатор «средний вес top-n рёбер»: ROC-AUC(n) для 4 variant × 7 task.
# По умолчанию outer fold 0–3: знак по train pool своего fold, test всех fold объединяются,
# CI — один кластерный bootstrap по субъектам. Графики + таблицы в artifacts/top_n_edge_baseline/.
# Первый запуск ~20–40 мин (чтение графов по fold), затем скоры берутся из кэша.
#
# Usage:
#   bash scripts/run_top_n_edge_baseline.sh
#   bash scripts/run_top_n_edge_baseline.sh --task EMOTION --task MOTOR
#   bash scripts/run_top_n_edge_baseline.sh --n-values 1 5 10 50 100 500 1000
#   bash scripts/run_top_n_edge_baseline.sh --replot       # только графики из CSV
#   bash scripts/run_top_n_edge_baseline.sh --no-cache
#   N_BOOTSTRAP=5000 bash scripts/run_top_n_edge_baseline.sh
#
# Язык подписей рисунков (оси, заголовок, легенда): PLOT_LANG=ru|en|both (по умолчанию ru).
# Рисунки сохраняются в PNG и SVG; английский набор — файлы с суффиксом _en.
#   PLOT_LANG=en   bash scripts/run_top_n_edge_baseline.sh --replot
#   PLOT_LANG=both bash scripts/run_top_n_edge_baseline.sh --replot
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
N_BOOTSTRAP="${N_BOOTSTRAP:-2000}"
PLOT_LANG="${PLOT_LANG:-ru}"

case "${PLOT_LANG}" in
  ru|en|both) ;;
  *) echo "PLOT_LANG должен быть ru, en или both, получено: ${PLOT_LANG}" >&2; exit 2 ;;
esac

echo "=== Top-n edge baseline: 4 variant × 7 task, B=${N_BOOTSTRAP}, plot lang=${PLOT_LANG} ==="
exec "${PYTHON}" scripts/top_n_edge_baseline.py \
  --n-bootstrap "${N_BOOTSTRAP}" --plot-lang "${PLOT_LANG}" "$@"
