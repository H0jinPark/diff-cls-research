#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Override with environment variables if desired; CL is deliberately disabled.
MODEL_ID="${MODEL_ID:-google-bert/bert-large-uncased}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/bert-large-$(date +%Y%m%d-%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-13 42 77}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

python scripts/check_cuda.py
echo "Model: $MODEL_ID | output: $OUTPUT_DIR | batch: $BATCH_SIZE | epochs: $EPOCHS"
# Seeds are whitespace-separated integers. The output directory must be new.
read -r -a SEED_ARGS <<< "$SEEDS"
python scripts/compare_dev.py \
  --splits data/splits/tratz-coarse-lexical --output "$OUTPUT_DIR" \
  --model "$MODEL_ID" --modes diff_cls cls_cls pair_interaction \
  --contrastive-epochs 0 --epochs "$EPOCHS" --seeds "${SEED_ARGS[@]}" \
  --batch-size "$BATCH_SIZE" --lr 0.00002 --max-length 32 --device cuda
