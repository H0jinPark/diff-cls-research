#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# OUTPUT_DIR must not already contain files. Output/logging is handled by the caller.
MODEL_ID="${MODEL_ID:-google-bert/bert-large-uncased}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/overnight-$(date +%Y%m%d-%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-13 42 77}"
CONTRASTIVE_EPOCHS="${CONTRASTIVE_EPOCHS:-3}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Started: $(date -Is)"
echo "Output: $OUTPUT_DIR | model: $MODEL_ID | batch: $BATCH_SIZE | epochs: $EPOCHS | seeds: $SEEDS"
trap 'result=$?; echo "Finished: $(date -Is) | exit code: $result"' EXIT
python scripts/check_cuda.py
read -r -a SEED_ARGS <<< "$SEEDS"
python scripts/paper_comparison.py \
  --splits data/splits/tratz-coarse-lexical --source data/benchmarks/panic \
  --output "$OUTPUT_DIR" --model "$MODEL_ID" --epochs "$EPOCHS" \
  --seeds "${SEED_ARGS[@]}" --batch-size "$BATCH_SIZE" --lr 0.00002 \
  --max-length 32 --device cuda --contrastive-epochs "$CONTRASTIVE_EPOCHS"
