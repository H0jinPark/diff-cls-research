#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# This pinned version matches the local experiments discussed in this project.
PANIC_COMMIT=9f7556eab0daf98698dba6f4b13b4668b45ba928
PANIC_DIR="${PANIC_DIR:-data/benchmarks/panic}"
OUTPUT_DIR="${OUTPUT_DIR:-data/splits/tratz-coarse-lexical}"

if [[ ! -e "$PANIC_DIR" ]]; then
  mkdir -p "$(dirname "$PANIC_DIR")"
  git clone --no-checkout https://github.com/vered1986/panic.git "$PANIC_DIR"
  git -C "$PANIC_DIR" checkout --detach "$PANIC_COMMIT"
elif [[ ! -d "$PANIC_DIR/.git" ]]; then
  echo "Existing source is not a Git clone: $PANIC_DIR" >&2
  exit 1
fi

actual_commit="$(git -C "$PANIC_DIR" rev-parse HEAD)"
if [[ "$actual_commit" != "$PANIC_COMMIT" ]]; then
  echo "Existing PANIC clone is at $actual_commit, expected $PANIC_COMMIT. No files changed." >&2
  exit 1
fi

# Verify the three inputs themselves, even when the clone already exists.
python - "$PANIC_DIR" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1]) / 'classification/data/tratz_coarse_grained_lexical'
expected = {
    'train.tsv': '70cc6072814b37aff7d0e03164fca864917fd18b3eed2bef97b380ac8fe8d4a2',
    'val.tsv': 'ebebef9b165633bdc0c7216651eea441a8ee828d4947e318f2997854da968124',
    'test.tsv': 'a89395fab79b4594a08f7cb853fcdc313b73592d329dfdd8b4c01d01d49b579f',
}
for name, digest in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f'Source hash mismatch or missing file: {path}')
print('Verified source commit and SHA-256 hashes.')
PY

python -m diffcls.cli prepare-tratz --source "$PANIC_DIR" \
  --granularity coarse --split-type lexical --output "$OUTPUT_DIR"
