#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/select_v2_params.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --judged_glob "runs/${RUN_ID}/api_judge/${MODEL_KEY}/calibration/**/*.api_judge.jsonl" \
  --default_l_star "${L_STAR_FALLBACK}"

