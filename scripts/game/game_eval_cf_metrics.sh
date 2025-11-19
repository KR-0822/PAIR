#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"

RETRIEVAL_JSON="${RETRIEVAL_JSON:-${PROJECT_ROOT}/inference/game/results_game/retrieval_results.json}"
CF_GT_BASE="${CF_GT_BASE:-${PROJECT_ROOT}/data/game/cf_gt}"
OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/eval/game}"

P_VALUES=("1" "2_5" "5_0")

for p in "${P_VALUES[@]}"; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/game_eval_cf_metrics.py" \
    --retrieval_json "${RETRIEVAL_JSON}" \
    --cf_gt_json "${CF_GT_BASE}/user_cf_game_gt_${p}p.json" \
    --out_dir "${OUT_BASE}/implicit/${p}p" \
    --ks "1,5,10,20,25,50" \
    --debug_users 50 \
    --prefer_added_only \
    --use_image_appid_from_results
done

