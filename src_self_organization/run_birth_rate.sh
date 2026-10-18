#!/usr/bin/env bash
# Birth-threshold ablation: N0=20, 4 thresholds x 5 seeds = 20 runs.
# Usage: bash src_self_organization/run_birth_threshold.sh [start|stop]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

THRESHOLDS=(0.15 0.25 0.35 0.45)
N0=20
SEEDS=(0 1 2 3 4)
DATASET="${DATASET:-fashionmnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/neurogenesis/birth_ablations}"
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

case "${1:-start}" in
  stop) stop_all; echo 'Stop requested.'; exit 0 ;;
  start) ;;
  *) echo "Usage: bash $0 [start|stop]" >&2; exit 2 ;;
esac

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

printf 'Dataset=%s | N0=%s | thresholds=%s | seeds=%s | total=%d runs\n' \
  "$DATASET" "$N0" "${THRESHOLDS[*]}" "${SEEDS[*]}" \
  "$((${#THRESHOLDS[@]} * ${#SEEDS[@]}))"

for threshold in "${THRESHOLDS[@]}"; do
  echo "=== Birth threshold=$threshold: launching ${#SEEDS[@]} seeds in parallel ==="
  children=()
  : > "$CHILD_FILE"
  for seed in "${SEEDS[@]}"; do
    log="$LOG_DIR/birth_${threshold}_N${N0}_seed${seed}.log"
    (
      set -o pipefail
      python -u -m src_self_organization.train_neurogenesis_ablation \
        --dataset "$DATASET" \
        --hidden_neurons "$N0" \
        --seed "$seed" \
        --ablation birth \
        --value "$threshold" \
        --output_dir "$OUTPUT_ROOT" \
        2>&1 | tee "$log" | sed -u "s/^/[birth=$threshold N0=$N0 seed=$seed] /"
    ) &
    pid=$!
    children+=("$pid")
    echo "$pid" >> "$CHILD_FILE"
    echo "Started birth=$threshold N0=$N0 seed=$seed PID=$pid"
  done
  failed=0
  for pid in "${children[@]}"; do
    wait "$pid" || failed=1
  done
  children=()
  : > "$CHILD_FILE"
  if (( failed )); then
    echo "ERROR: at least one run failed for birth=$threshold; see $LOG_DIR" >&2
    exit 1
  fi
  echo "=== Completed birth threshold=$threshold ==="
done

echo 'All 20 birth-threshold experiments completed.'
