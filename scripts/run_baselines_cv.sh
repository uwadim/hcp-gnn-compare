#!/usr/bin/env bash
# Базовые классификаторы на outer fold 0–3 с ОДНИМ отчётом на классификатор:
#   mean_edge — «средний вес ребра»; top_n — «средний вес top-n рёбер»;
#   logreg    — логистическая регрессия на признаках графов.
# Модель каждого fold обучается на train pool этого fold, test-предсказания всех fold
# объединяются, доверительные интервалы считаются по всем fold (pooled — основной,
# macro — среднее по fold; для mean_edge и logreg также AUC по каждому fold).
# Выход: artifacts/baselines_cv/{mean_edge,top_n,logreg}/ (report.md, CSV).
# Без обучения GNN; чтение графов ~5–10 мин на fold и baseline (далее кэш).
#
# Env: FOLDS="0 1 2 3" BASELINES="mean_edge top_n logreg"
#      N_BOOTSTRAP — если не задан, берётся умолчание скрипта baseline
#
# Usage:
#   bash scripts/run_baselines_cv.sh
#   FOLDS="0 1 2" BASELINES="mean_edge" bash scripts/run_baselines_cv.sh --task WM
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
FOLDS="${FOLDS:-0 1 2 3}"
BASELINES="${BASELINES:-mean_edge top_n logreg}"
OUT_ROOT="artifacts/baselines_cv"

read -r -a fold_args <<< "${FOLDS}"
bootstrap_args=()
if [[ -n "${N_BOOTSTRAP:-}" ]]; then
  bootstrap_args=(--n-bootstrap "${N_BOOTSTRAP}")
fi

for baseline in ${BASELINES}; do
  case "${baseline}" in
    mean_edge) script="scripts/mean_edge_baseline.py" ;;
    top_n)     script="scripts/top_n_edge_baseline.py" ;;
    logreg)    script="scripts/graph_features_logreg.py" ;;
    *) echo "Неизвестный baseline: ${baseline} (mean_edge|top_n|logreg)" >&2; exit 2 ;;
  esac
  echo "=== ${baseline}: outer fold [${FOLDS}], один отчёт ==="
  "${PYTHON}" "${script}" \
    --outer-fold "${fold_args[@]}" \
    --out-dir "${OUT_ROOT}/${baseline}" \
    "${bootstrap_args[@]}" \
    "$@"
done
