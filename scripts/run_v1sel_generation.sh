#!/usr/bin/env bash
# Generate the steered outputs for the stance-selected replacement items (see
# src/analysis/build_v1sel_jobs.py). Writes to a SEPARATE directory per model so the frozen
# matched_flip generations are never touched; the recompute step reads both.
set -uo pipefail
PKG="${ABST_PKG:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=python3
# Qwen-3.5 is multimodal (AutoModelForMultimodalLM) and only loads under transformers 5.x
PY_QWEN35="${PY_QWEN35:-python3}"
J=$PKG/runs/v1sel_jobs
# instruction_common.default_cache_env() derives its cache root from PACKAGE_ROOT.parents[2],
# which resolves to /hf_cache for this checkout location. It uses setdefault, so exporting every
# variable it touches is what actually pins the frozen model cache.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
export TRANSFORMERS_CACHE=$HOME/.cache/transformers
export XDG_CACHE_HOME=$HOME/.cache/xdg_cache
export TRITON_CACHE_DIR=$HOME/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor
export TMPDIR=$HOME/.cache/tmp
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
cd "$PKG"

run () {  # model_key run_id gpu
  local mk=$1 rid=$2 gpu=$3
  [ -f "$J/$mk.jobs.jsonl" ] || { echo "skip $mk (no jobs)"; return; }
  local od="$PKG/runs/$rid/generations/$mk/matched_flip_v1sel"
  mkdir -p "$od"
  echo ">> $mk on GPU$gpu -> $od"
  local py="$PY"; [ "$mk" = "qwen35_9b" ] && py="$PY_QWEN35"
  CUDA_VISIBLE_DEVICES=$gpu "$py" -u src/steering/run_v2_campaign.py \
    --run_id "$rid" --model "$mk" \
    --jobs_jsonl "$J/$mk.jobs.jsonl" --samples_jsonl "$J/$mk.samples.jsonl" \
    --output_dir "$od" \
    --generation_batch_size 1 --max_new_tokens 256 --max_length 2048 --seed 42 \
    2>&1 | grep -E "^\[(PLAN|JOB|DONE)\]|Error|Traceback"
}

# 24 GB cards: Gemma-3-12B in bf16 nearly fills one on its own, so models run one at a time
# rather than three to a GPU. Only ~78 generations total, so serial costs little.
run llama31_8b_instruct v2_full_llama31_8b 0
run qwen35_9b           v2_full_qwen35_9b  1
run gemma3_12b          v2_full_gemma3_12b 2
run qwen25_7b_instruct  v2_full_qwen       3
run gemma2_9b_it        v2_full_gemma      0
echo "== all done =="
