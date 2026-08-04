#!/usr/bin/env bash
# Generate steered outputs for the stance-selected suppression replacements
# (src/analysis/build_supp_v1sel_jobs.py). Separate output directory; frozen generations untouched.
# Batch size per model matches that model's frozen matched_baseline manifest (gemma3 ran at 1).
set -uo pipefail
PKG="${ABST_PKG:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=python3
PY_QWEN35="${PY_QWEN35:-python3}"
J=$PKG/runs/supp_v1sel_jobs
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

run () {  # model_key run_id gpu batch
  local mk=$1 rid=$2 gpu=$3 bs=$4
  [ -f "$J/$mk.jobs.jsonl" ] || { echo "skip $mk"; return; }
  local od="$PKG/runs/$rid/generations/$mk/matched_flip_supp_v1sel"
  mkdir -p "$od"
  local py="$PY"; [ "$mk" = "qwen35_9b" ] && py="$PY_QWEN35"
  echo ">> $mk GPU$gpu batch=$bs"
  CUDA_VISIBLE_DEVICES=$gpu "$py" -u src/steering/run_v2_campaign.py \
    --run_id "$rid" --model "$mk" \
    --jobs_jsonl "$J/$mk.jobs.jsonl" --samples_jsonl "$J/$mk.samples.jsonl" \
    --output_dir "$od" \
    --generation_batch_size "$bs" --max_new_tokens 256 --max_length 2048 --seed 42 \
    2>&1 | grep -E "^\[(PLAN|DONE)\]|Error|Traceback|OutOfMemory"
}

# four cards, four models at once; the fifth follows on the card that frees first
run gemma3_12b          v2_full_gemma3_12b 0 1 &
run llama31_8b_instruct v2_full_llama31_8b 1 8 &
run qwen35_9b           v2_full_qwen35_9b  2 8 &
run qwen25_7b_instruct  v2_full_qwen       3 8 &
wait
run gemma2_9b_it        v2_full_gemma      1 8
echo "== suppression generation done =="
