#!/usr/bin/env bash
# Run from anywhere: bash src_self_organization/continuous_learning/train_cl.sh [start|stop]
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"
ROOT="${OUTPUT_ROOT:-data/so/continual/cifar10}"
LOG_DIR="$ROOT/logs"
CONTROL="$ROOT/.run_control"
PID_FILE="$CONTROL/main.pid"
CHILD_FILE="$CONTROL/child.pid"
MODELS=(static20 static80 full no_birth no_death)
SEEDS=(0 1 2 3 4)
EPOCHS_PER_TASK="${EPOCHS_PER_TASK:-20}"
mkdir -p "$LOG_DIR" "$CONTROL"

stop_run() {
  if [[ -f "$CHILD_FILE" ]]; then
    child="$(cat "$CHILD_FILE")"
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
  fi
  if [[ -f "$PID_FILE" ]]; then
    parent="$(cat "$PID_FILE")"
    [[ "$parent" == "$$" ]] || kill -TERM "$parent" 2>/dev/null || true
  fi
}
if [[ "${1:-start}" == stop ]]; then
  stop_run
  echo 'Stop requested.'
  exit 0
fi
if [[ "${1:-start}" != start ]]; then
  echo "Usage: bash $0 [start|stop]" >&2
  exit 2
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running (PID $(cat "$PID_FILE"))" >&2
  exit 1
fi
printf '%s\n' "$$" > "$PID_FILE"
child=''
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
  rm -f "$PID_FILE" "$CHILD_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

completed=0
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    result="$ROOT/$model/seed_$seed/summary.pt"
    if [[ -f "$result" ]]; then
      echo "[SKIP] $model seed=$seed (summary exists)"
      completed=$((completed+1))
      continue
    fi
    log="$LOG_DIR/${model}_seed${seed}.log"
    echo "[START] $model seed=$seed ($((completed+1))/25) -> $log"
    python -u -m src_self_organization.continuous_learning.train_cl \
      --model "$model" --seed "$seed" \
      --epochs-per-task "$EPOCHS_PER_TASK" \
      --output-dir "$ROOT" > "$log" 2>&1 &
    child=$!
    printf '%s\n' "$child" > "$CHILD_FILE"
    if ! wait "$child"; then
      echo "[FAILED] $model seed=$seed; see $log" >&2
      exit 1
    fi
    child=''
    : > "$CHILD_FILE"
    completed=$((completed+1))
    echo "[DONE] $model seed=$seed ($completed/25)"
  done
done
echo 'All 25 continual-learning experiments completed.'