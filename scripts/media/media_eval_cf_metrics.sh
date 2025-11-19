#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"


P_VALUES=("1" "2.5" "5")

RETRIEVAL_JSON="${PROJECT_ROOT}/results/media/retrieval_results.json"
OUT_BASE_DIR="${PROJECT_ROOT}/results/media"

PYTHON_PY="${SCRIPT_DIR}/media_eval_cf_metrics.py"

for k in "${K_VALUES[@]}"; do
  for p in "${P_VALUES[@]}"; do
    "${PYTHON_BIN}" "${PYTHON_PY}" \
      --retrieval_json "${RETRIEVAL_JSON}" \
      --cf_gt_json "${PROJECT_ROOT}/dataset/user_cf_media_gt_${p}p.json" \
      --out_dir "${OUT_BASE_DIR}/eval/added_only_p${p}" \
      --ks "1,5,10,15,20,25,50" \
      --debug_users 50 \
      --use_image_title_norm_from_results \
      --prefer_added_only
  done
done

