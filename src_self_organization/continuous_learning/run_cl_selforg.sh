#!/usr/bin/env bash
set -euo pipefail
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-100}"
BIRTH_BUDGET="${BIRTH_BUDGET:-100}"
REPLAY_PER_CLASS="${REPLAY_PER_CLASS:-20}"
for seed in $SEEDS; do
  for method in static activity full; do
    python -u -m src_self_organization.continuous_learning.train_cl_selforg \
      --method "$method" --seed "$seed" --epochs-per-task "$EPOCHS" \
      --task1-max-neurons 400 --birth-budget "$BIRTH_BUDGET" \
      --replay-per-class "$REPLAY_PER_CLASS"
  done
done
