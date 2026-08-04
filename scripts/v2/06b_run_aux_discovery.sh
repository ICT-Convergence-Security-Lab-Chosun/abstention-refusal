#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/v2_defaults.env

BUILD_ARGS=(
  --run_id "${RUN_ID}"
  --model "${MODEL_KEY}"
  --mode aux_discovery
  --selection_json "runs/${RUN_ID}/selection/${MODEL_KEY}.selection.json"
  --discovery_layers "${DISCOVERY_LAYERS}"
)
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  BUILD_ARGS+=(--sample_limit "${SAMPLE_LIMIT}")
fi

"${PYTHON_BIN}" -u src/steering/build_v2_jobs.py "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" -u src/steering/run_v2_campaign.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --jobs_jsonl "runs/${RUN_ID}/jobs/${MODEL_KEY}.aux_discovery.jobs.jsonl" \
  --output_dir "runs/${RUN_ID}/generations/${MODEL_KEY}/aux_discovery" \
  --max_length "${MAX_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --seed "${SEED}"

"${PYTHON_BIN}" -u src/steering/rejudge_with_api.py \
  --input "runs/${RUN_ID}/generations/${MODEL_KEY}/aux_discovery/generations.jsonl" \
  --base_root "runs/${RUN_ID}/generations/${MODEL_KEY}/aux_discovery" \
  --output_root "runs/${RUN_ID}/api_judge/${MODEL_KEY}/aux_discovery" \
  --api_base "${API_BASE}" \
  --api_key_env "${API_KEY_ENV}" \
  --judge_model "${JUDGE_MODEL}" \
  --summary_mode steering \
  --overwrite

"${PYTHON_BIN}" -u src/steering/select_v2_params.py \
  --run_id "${RUN_ID}" \
  --model "${MODEL_KEY}" \
  --judged_glob "runs/${RUN_ID}/api_judge/${MODEL_KEY}/**/*.api_judge.jsonl" \
  --default_l_star "${L_STAR_FALLBACK}"
