#!/usr/bin/env bash
# Connectivity (beta) ablation: 5 N0 x 4 beta x 5 seeds = 100 runs.
# Usage: bash src_self_organization/run_connectivity.sh [start|stop]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

N0_VALUES=(10 20 40 80 160)
BETA_VALUES=(0.1 0.2 0.4 0.8)
SEEDS=(0 1 2 3 4)
DATASET="${DATASET:-fashionmnist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/neurogenesis/beta_ablation}"
LOG_DIR="$OUTPUT_ROOT/logs"
RUN_DIR="$OUTPUT_ROOT/.run_control"
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

printf 'Dataset=%s | N0=%s | beta=%s | seeds=%s | total=%d\n' \
  "$DATASET" "${N0_VALUES[*]}" "${BETA_VALUES[*]}" "${SEEDS[*]}" \
  "$((${#N0_VALUES[@]} * ${#BETA_VALUES[@]} * ${#SEEDS[@]}))"

# For each beta/seed, launch the five N0 configurations in parallel.
for beta in "${BETA_VALUES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== beta=$beta seed=$seed: launching ${#N0_VALUES[@]} N0 runs ==="
    children=()
    : > "$CHILD_FILE"
    for n0 in "${N0_VALUES[@]}"; do
      log="$LOG_DIR/beta_${beta}_N${n0}_seed${seed}.log"
      (
        set -o pipefail
        python -u -m src_self_organization.train_neurogenesis_ablation \
          --dataset "$DATASET" \
          --hidden_neurons "$n0" \
          --seed "$seed" \
          --ablation beta \
          --value "$beta" \
          --output_dir "$OUTPUT_ROOT" \
          2>&1 | tee "$log" | sed -u "s/^/[beta=$beta N0=$n0 seed=$seed] /"
      ) &
      pid=$!
      children+=("$pid")
      echo "$pid" >> "$CHILD_FILE"
      echo "Started beta=$beta N0=$n0 seed=$seed PID=$pid"
    done
    failed=0
    for pid in "${children[@]}"; do
      wait "$pid" || failed=1
    done
    children=()
    : > "$CHILD_FILE"
    if (( failed )); then
      echo "ERROR: at least one run failed for beta=$beta seed=$seed; inspect $LOG_DIR" >&2
      exit 1
    fi
    echo "=== Completed beta=$beta seed=$seed ==="
  done
done

echo 'All 100 connectivity experiments completed.'