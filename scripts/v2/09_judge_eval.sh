#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/rejudge_with_api.py \
  --input "runs/${RUN_ID}/generations/${MODEL_KEY}/eval/generations.jsonl" \
  --base_root "runs/${RUN_ID}/generations/${MODEL_KEY}/eval" \
  --output_root "runs/${RUN_ID}/api_judge/${MODEL_KEY}/eval" \
  --api_base "${API_BASE}" \
  --api_key_env "${API_KEY_ENV}" \
  --judge_model "${JUDGE_MODEL}" \
  --summary_mode steering \
  --overwrite

