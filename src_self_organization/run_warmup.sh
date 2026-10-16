#!/usr/bin/env bash
# Run five seeds concurrently for each warmup, N0=80.
# Usage: bash src_self_organization/run_warmup.sh [start|stop]
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
WARMUPS=(50000 100000 200000 400000 800000)
N0=80
SEEDS=(0 1 2 3 4)
DATASET="${DATASET:-fashionmnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/neurogenesis/warmup_ablations}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUN_DIR="${OUTPUT_ROOT}/.run_control"
PID_FILE="$RUN_DIR/main.pid"
CHILD_FILE="$RUN_DIR/children.pid"
stop_all() {
  if [[ -f "$CHILD_FILE" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    done < "$CHILD_FILE"
  fi
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    [[ "$pid" == "$$" ]] || kill -TERM "$pid" 2>/dev/null || true
  fi
}
if [[ "${1:-start}" == stop ]]; then
  stop_all
  echo 'Stop requested.'
  exit 0
fi
if [[ "${1:-start}" != start ]]; then
  echo "Usage: bash $0 [start|stop]" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Already running with PID $old_pid; use bash $0 stop" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" > "$PID_FILE"
: > "$CHILD_FILE"
children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  rm -f "$PID_FILE" "$CHILD_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Dataset=%s | N0=%s | total=%s runs\n' "$DATASET" "$N0" "$((${#WARMUPS[@]} * ${#SEEDS[@]}))"
for warmup in "${WARMUPS[@]}"; do
  echo "=== Warmup=$warmup: launching ${#SEEDS[@]} seeds in parallel ==="
  children=()
  : > "$CHILD_FILE"
  for seed in "${SEEDS[@]}"; do
    log="$LOG_DIR/warmup_${warmup}_N${N0}_seed${seed}.log"
    (
      set -o pipefail
      python -u -m src_self_organization.train_neurogenesis_ablation \
        --dataset "$DATASET" \
        --hidden_neurons "$N0" \
        --seed "$seed" \
        --ablation warmup \
        --value "$warmup" \
        --output_dir "$OUTPUT_ROOT" \
        2>&1 | tee "$log" | sed -u "s/^/[warmup=$warmup N0=$N0 seed=$seed] /"
    ) &
    pid=$!
    children+=("$pid")
    echo "$pid" >> "$CHILD_FILE"
    echo "Started warmup=$warmup seed=$seed PID=$pid"
  done
  failed=0
  for pid in "${children[@]}"; do
    wait "$pid" || failed=1
  done
  children=()
  : > "$CHILD_FILE"
  if (( failed )); then
    echo "ERROR: at least one seed failed for warmup=$warmup; see $LOG_DIR" >&2
    exit 1
  fi
  echo "=== Completed warmup=$warmup ==="
done
echo 'All 25 experiments completed.'
