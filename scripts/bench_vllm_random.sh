#!/usr/bin/env bash
# Upstream-style vLLM serving bench (random ISL/OSL). Same CLI shape as
# `vllm bench serve --dataset-name random` used in vLLM docs and 70B papers.
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen72-awq}"
TOKENIZER="${TOKENIZER:-$MODEL}"
OUT="${RESULT_DIR:-./results/vllm-random}"
mkdir -p "$OUT"

run_cell() {
  local isl="$1" osl="$2" conc="$3" n="$4"
  local name="isl${isl}_osl${osl}_c${conc}_n${n}"
  echo "=== $name ==="
  vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --endpoint /v1/chat/completions \
    --dataset-name random \
    --random-input-len "$isl" \
    --random-output-len "$osl" \
    --random-range-ratio 0 \
    --num-prompts "$n" \
    --max-concurrency "$conc" \
    --ignore-eos \
    --save-result \
    --result-dir "$OUT" \
    --result-filename "${name}.json" \
    | tee "$OUT/${name}.log"
}

# Short / balanced / prefill-heavier / decode-heavier / batched.
run_cell 128 128 1 8
run_cell 512 128 1 8
run_cell 2048 128 1 8
run_cell 8192 128 1 4
run_cell 128 256 1 8
run_cell 512 128 4 16
