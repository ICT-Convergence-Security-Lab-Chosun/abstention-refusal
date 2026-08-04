#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/summarize_v2_results.py \
  --run_id "${RUN_ID}" \
  --judged_glob "runs/${RUN_ID}/api_judge/${MODEL_KEY}/eval/**/*.api_judge.jsonl"

