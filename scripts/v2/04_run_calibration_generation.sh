#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

"${PYTHON_BIN}" -u src/steering/run_v2_campaign.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --jobs_jsonl "runs/${RUN_ID}/jobs/${MODEL_KEY}.calibration.jobs.jsonl" \
  --output_dir "runs/${RUN_ID}/generations/${MODEL_KEY}/calibration" \
  --max_length "${MAX_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --seed "${SEED}"

