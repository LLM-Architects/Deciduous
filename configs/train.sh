#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash configs/train.sh TRAIN TEMPERATURE EVALUATION JEV_REFERENCE PROBABILITY_DIAGNOSTIC RUN_DIR CHECKPOINT_DIR

All input files are explicit JSONL paths. Relative paths are resolved from the
repository root. RUN_DIR and CHECKPOINT_DIR must not exist. This starts one epoch
from the pinned pretrained base with a fresh optimizer; it does not resume or
initialize from a released AutoJev checkpoint. Input types are defined in
src/autojev/types.py; recipe metadata is in configs/training.json.
EOF
  exit 0
fi
if [[ "$#" -ne 7 ]]; then
  printf '%s\n' 'Expected seven paths. Run bash configs/train.sh --help.' >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
train_file=$1
temperature_file=$2
evaluation_file=$3
jev_reference=$4
probability_file=$5
run_dir=$6
checkpoint_dir=$7
for input_file in "$train_file" "$temperature_file" "$evaluation_file" "$jev_reference" "$probability_file"; do
  if [[ ! -f "$input_file" ]]; then
    printf 'Missing input file: %s\n' "$input_file" >&2
    exit 2
  fi
done
if [[ -e "$run_dir" || -e "$checkpoint_dir" ]]; then
  printf '%s\n' 'Use fresh run and checkpoint directories; existing artifacts will not be replaced.' >&2
  exit 2
fi
git rev-parse --is-inside-work-tree >/dev/null
export AUTOJEV_EVENTS="$run_dir/events.jsonl"
export PYTHONUNBUFFERED=1

exec env -u PYTHONPATH -u VIRTUAL_ENV uv run --locked autojev-train \
  --train "$train_file" \
  --temperature "$temperature_file" \
  --development "$evaluation_file" \
  --reference "$jev_reference" \
  --public "$probability_file" \
  --run "$run_dir" --output "$checkpoint_dir" \
  --base-model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --epochs 1 --seed 20260920 --lr 2e-6 --weight-decay 0.01 \
  --batch-size 32 --effective-batch-size 256 \
  --token-budget 8192 --max-length 8192 --cpu-threads 32 \
  --eval-every 50 --public-eval-every 50 --resume-every 50
