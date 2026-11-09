#!/usr/bin/env bash
# Проверка утечки: аудит построения ensemble + SkipGCN на перемешанных метках.
#
# Цель: ячейки ensemble_univariate / ensemble_full × {EMOTION, MOTOR, SOCIAL},
# где stage-2 дал ROC-AUC = 1.000.
#
# 1) Статический аудит (без GPU): subjects ↔ splits.json, corr(edge, label).
# 2) Обучение с shuffle_labels (resumable). По умолчанию scope=all → ожидание AUC≈0.5.
#
# Usage:
#   bash scripts/run_leak_check.sh
#   bash scripts/run_leak_check.sh --audit-only
#   bash scripts/run_leak_check.sh --train-only
#   bash scripts/run_leak_check.sh --scope train   # метки перемешаны только на train
#   bash scripts/run_leak_check.sh --list
#   bash scripts/run_leak_check.sh --dry-run
#   SCOPE=train bash scripts/run_leak_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SCOPE="${SCOPE:-all}"
MODEL="${MODEL:-skipgcn}"

audit_only=false
train_only=false
train_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit-only) audit_only=true; shift ;;
    --train-only) train_only=true; shift ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --scope=*) SCOPE="${1#*=}"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    *) train_args+=("$1"); shift ;;
  esac
done

if [[ "${SCOPE}" != "all" && "${SCOPE}" != "train" ]]; then
  echo "scope должен быть all или train, получено: ${SCOPE}" >&2
  exit 2
fi

if [[ "${train_only}" == false ]]; then
  echo "=== Аудит построения ensemble (univariate/full × EMOTION/MOTOR/SOCIAL) ==="
  "${PYTHON}" scripts/audit_ensemble_construction.py \
    --variant ensemble_univariate --variant ensemble_full \
    --task EMOTION --task MOTOR --task SOCIAL
  [[ "${audit_only}" == true ]] && exit 0
fi

echo "=== Leak check train: ${MODEL}, shuffle_labels_scope=${SCOPE}, 2×3 ячейки ==="
exec "${PYTHON}" scripts/batch_runner.py leak_check \
  --model "${MODEL}" \
  --scope "${SCOPE}" \
  "${train_args[@]}"
