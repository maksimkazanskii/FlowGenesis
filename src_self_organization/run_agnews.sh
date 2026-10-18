!/usr/bin/env bash
# AG News N0 ablation: N0=80,160,320,640; seeds=0..4.
# Usage: bash src_self_organization/run_agnews.sh [start|stop]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

N0_VALUES=(80 160 320 640)
SEEDS=(0 1 2 3 4)
MAX_PARALLEL="${MAX_PARALLEL:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/so/n0_ablation/agnews}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CONTROL_DIR="$OUTPUT_ROOT/.run_control"
PID_FILE="$CONTROL_DIR/launcher.pid"
CHILD_FILE="$CONTROL_DIR/children.pid"
MODULE="src_self_organization.generalization_news.train_neurogenesis_n0_ablation"
PYTHON_BIN="${PYTHON_BIN:-python}"

stop_all() {
  if [[ -f "$CHILD_FILE" ]]; then
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill -TERM "$pid" 2>/dev/null || true
      pkill -TERM -P "$pid" 2>/dev/null || true
    done < "$CHILD_FILE"
  fi
  if [[ -f "$PID_FILE" ]]; then
    local launcher
    launcher="$(cat "$PID_FILE")"
    kill -TERM "$launcher" 2>/dev/null || true
  fi
}

case "${1:-start}" in
  stop)
    if [[ ! -f "$PID_FILE" ]]; then
      echo "No recorded AG News launcher ($PID_FILE)."
      exit 0
    fi
    stop_all
    echo "Stop requested."
    exit 0
    ;;
  start) ;;
  *) echo "Usage: bash $0 [start|stop]" >&2; exit 2 ;;
esac

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" "$CONTROL_DIR"
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Already running with PID $old_pid; use: bash $0 stop" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" > "$PID_FILE"
: > "$CHILD_FILE"
active=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${active[@]-}"; do
    kill -TERM "$pid" 2>/dev/null || true
    pkill -TERM -P "$pid" 2>/dev/null || true
  done
  rm -f "$PID_FILE" "$CHILD_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Rebuild PID file after each batch; all experiments in one batch are awaited.
run_batch() {
  local failed=0 pid n0 seed log summary
  active=()
  : > "$CHILD_FILE"
  for pair in "$@"; do
    n0="${pair%%:*}"
    seed="${pair##*:}"
    summary="$OUTPUT_ROOT/neurons_${n0}/seed_${seed}/summary.pt"
    if [[ -s "$summary" ]]; then
      echo "SKIP completed: N0=$n0 seed=$seed ($summary)"
      continue
    fi
    log="$LOG_DIR/agnews_N${n0}_seed${seed}.log"
    echo "START N0=$n0 seed=$seed -> $log"
    (
      exec "$PYTHON_BIN" -u -m "$MODULE" \
        --dataset agnews --hidden_neurons "$n0" --seed "$seed" \
        > "$log" 2>&1
    ) &
    pid=$!
    active+=("$pid")
    echo "$pid" >> "$CHILD_FILE"
  done
  for pid in "${active[@]-}"; do
    if ! wait "$pid"; then
      echo "FAILED process PID=$pid (check logs)" >&2
      failed=1
    fi
  done
  active=()
  : > "$CHILD_FILE"
  return "$failed"
}

pairs=()
for seed in "${SEEDS[@]}"; do
  for n0 in "${N0_VALUES[@]}"; do
    pairs+=("$n0:$seed")
  done
done

printf 'AG News N0 ablation | %s experiments | concurrency=%s\n' "${#pairs[@]}" "$MAX_PARALLEL"
for ((i=0; i<${#pairs[@]}; i+=MAX_PARALLEL)); do
  batch=("${pairs[@]:i:MAX_PARALLEL}")
  run_batch "${batch[@]}" || exit 1
done
echo "All AG News experiments completed."ag