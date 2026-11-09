#!/usr/bin/env bash
# Stage 2 (варианты): SkipGCN × 4 variant × 7 task на test outer 0 + отчёт.
# Обучение resumable (FINISHED пропускаются); порядок «задача → 4 варианта».
# Отчёт: ROC-AUC [95% CI] по субъектам, парные ΔAUC; поправка на
# множественность — Romano–Wolf step-down (основная), Holm, BH.
# Семейства: пары с REFERENCE (основное) и остальные (вторичное).
#
# Env: MODEL=skipgcn N_BOOTSTRAP=10000 REFERENCE=correlation ALPHA=0.05
#
# Usage:
#   bash scripts/run_stage2_variants.sh                # обучение + отчёт
#   bash scripts/run_stage2_variants.sh --list         # статусы 28 прогонов
#   bash scripts/run_stage2_variants.sh --dry-run      # план возобновления
#   bash scripts/run_stage2_variants.sh --report-only  # только отчёт
#   bash scripts/run_stage2_variants.sh --variant ensemble_full
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-skipgcn}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
REFERENCE="${REFERENCE:-correlation}"   # эталон основного семейства; none — одно семейство
ALPHA="${ALPHA:-0.05}"

report_only=false
train_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--report-only" ]]; then
    report_only=true
  else
    train_args+=("${arg}")
  fi
done

if [[ "${report_only}" == false ]]; then
  echo "=== Stage2 variants (resumable): ${MODEL} × 4 variant × 7 task ==="
  "${PYTHON}" scripts/batch_runner.py stage2_variants --model "${MODEL}" "${train_args[@]}"
  for arg in "${train_args[@]}"; do
    [[ "${arg}" == "--list" || "${arg}" == "--dry-run" ]] && exit 0
  done
fi

echo "=== Отчёт: сравнение вариантов (${MODEL}), Romano–Wolf / Holm / BH, эталон ${REFERENCE} ==="
exec "${PYTHON}" scripts/compare_variants.py \
  --model "${MODEL}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --reference "${REFERENCE}" \
  --alpha "${ALPHA}"
