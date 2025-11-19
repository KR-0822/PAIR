#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd )"

PYTHON_BIN="${PYTHON_BIN:-python}"

TEST_JSON="${TEST_JSON:-${PROJECT_ROOT}/dataset/user_split/game/test.json}"
IMG_EMB_DIR="${IMG_EMB_DIR:-${PROJECT_ROOT}/data/game/image_embeddings_e5v}"

OUT_BASE="${OUT_BASE:-${PROJECT_ROOT}/results/baseline_e5v/game}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-42}"

mkdir -p "${OUT_BASE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/game_eval_e5v_baseline.py" \
  --test_file "${TEST_JSON}" \
  --image_embedding_dir "${IMG_EMB_DIR}" \
  --output_dir "${OUT_BASE}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --target_policy "last" \
  --skip_on_mismatch \
  --allowed_subs "Preferred Game Name;Preferred Game Genres;Multiplayer Preference;Gaming Frequency"
