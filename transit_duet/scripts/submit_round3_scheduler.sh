#!/usr/bin/env bash
# Submit TransitDuet round-3 jobs through the scheduler.
#
# Defaults are intentionally dry-run only. To actually submit:
#   DRY_RUN=0 DISPATCH=1 STAGE=train bash scripts/submit_round3_scheduler.sh
#
# Useful stages:
#   STAGE=train        main timetable runs + variants + fixed-grid + baselines
#   STAGE=independent  independent-assembly runs after H_fixed_timetable exists
#   STAGE=eval         per-checkpoint eval for learned methods
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCHEDULER="${SCHEDULER:-/home/erzhu419/.claude/skills/scheduler/scheduler.py}"
SCHEDULER_PY="${SCHEDULER_PY:-python3}"
PY="${PY:-python3}"
SCHEDULER_CWD="${SCHEDULER_CWD:-$ROOT_DIR}"

STAGE="${STAGE:-train}"
EPISODES="${EPISODES:-300}"
SEEDS="${SEEDS:-42 123 456 789 1001 1002 1003 1004 1005 1006}"
EVAL_EPS="${EVAL_EPS:-49,99,149,199,249,299}"
N_EVAL="${N_EVAL:-20}"
DEVICE="${DEVICE:-cpu}"
CPU_PER_TASK="${CPU_PER_TASK:-2}"
RAM_MB="${RAM_MB:-2048}"
PROJECT="${PROJECT:-TransitDuet}"
PRIORITY="${PRIORITY:-normal}"
DRY_RUN="${DRY_RUN:-1}"
DISPATCH="${DISPATCH:-0}"
ALLOW_DUPLICATE="${ALLOW_DUPLICATE:-0}"

NODE_MODE="${NODE_MODE:-require}"
NODES="${NODES:-node001 node002 node003 node004 node005 node006}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu}"

MAIN_EXP=H_timetable_v4
TIMETABLE_VARIANTS=(H_hiro H_timetable_continuous H_timetable_no_context)
INDEPENDENT_ASSEMBLY=H_timetable_v4_independent_assembly
FIXED_TIMETABLE_GRID=(H_fixed_timetable_300 H_fixed_timetable_330 H_fixed_timetable H_fixed_timetable_390 H_fixed_timetable_420)
COUPLING_VARIANTS=(H_tpc H_haar)
BASELINES=(fixed ga cmaes)
RULE_BASELINES=(rule_daganzo rule_xuan)

if [[ ! -f "$SCHEDULER" ]]; then
  echo "scheduler not found: $SCHEDULER" >&2
  exit 2
fi

case "$NODE_MODE" in
  auto|preferred|require|hpc) ;;
  *) echo "NODE_MODE must be auto, preferred, require, or hpc" >&2; exit 2 ;;
esac

read -r -a NODE_ARR <<< "$NODES"
if [[ "$NODE_MODE" != "auto" && "${#NODE_ARR[@]}" -eq 0 ]]; then
  echo "NODES must not be empty when NODE_MODE=$NODE_MODE" >&2
  exit 2
fi

echo "TransitDuet scheduler submission"
echo "  root       : $ROOT_DIR"
echo "  stage      : $STAGE"
echo "  episodes   : $EPISODES"
echo "  seeds      : $SEEDS"
echo "  device     : $DEVICE"
echo "  cpu/ram    : ${CPU_PER_TASK} cores / ${RAM_MB} MB"
echo "  placement  : $NODE_MODE${NODES:+ ($NODES)}"
echo "  dry_run    : $DRY_RUN"
echo "  dispatch   : $DISPATCH"
echo

CURRENT_NODE=""
choose_node() {
  if [[ "$NODE_MODE" == "auto" ]]; then
    CURRENT_NODE=""
    return
  fi
  CURRENT_NODE="${NODE_ARR[$((node_idx % ${#NODE_ARR[@]}))]}"
  node_idx=$((node_idx + 1))
}

submit_task() {
  local desc="$1"
  local cmd="$2"
  local signature="$3"
  local result_dir="$4"
  choose_node
  local node="$CURRENT_NODE"

  echo "[$((submitted + 1))] $desc -> ${NODE_MODE}${node:+:$node}"
  echo "    $cmd"
  submitted=$((submitted + 1))

  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  mkdir -p "$result_dir"
  args=(
    "$SCHEDULER" submit
    --project "$PROJECT"
    --description "$desc"
    --cmd "$cmd"
    --cwd "$SCHEDULER_CWD"
    --signature "$signature"
    --vram 0
    --cpu "$CPU_PER_TASK"
    --ram-mb "$RAM_MB"
    --priority "$PRIORITY"
    --result-dir "$result_dir"
    --local-result-dir "$result_dir"
    --allow-no-resume
    --allow-cpu-training
    --cpu-training-justification "TransitDuet bus-control experiments are CPU-friendly event simulations and were requested to be dispatched through scheduler nodes."
    --env "OMP_NUM_THREADS=$CPU_PER_TASK" "MKL_NUM_THREADS=$CPU_PER_TASK" "OPENBLAS_NUM_THREADS=$CPU_PER_TASK"
  )
  if [[ "$NODE_MODE" == "preferred" ]]; then
    args+=(--preferred-node "$node")
  elif [[ "$NODE_MODE" == "require" ]]; then
    args+=(--require-node "$node")
  elif [[ "$NODE_MODE" == "hpc" ]]; then
    args+=(--require-node "$node" --slurm-partition "$SLURM_PARTITION")
  fi
  if [[ "$ALLOW_DUPLICATE" == "1" ]]; then
    args+=(--allow-duplicate)
  fi
  local output
  if ! output="$("$SCHEDULER_PY" "${args[@]}" 2>&1)"; then
    echo "$output"
    exit 1
  fi
  echo "$output"
  local task_id
  task_id="$(printf '%s\n' "$output" | awk '/^submitted t[0-9]+/ {print $2; exit}')"
  if [[ -n "$task_id" ]]; then
    submitted_task_ids+=("$task_id")
  fi
}

node_idx=0
submitted=0
submitted_task_ids=()

if [[ "$STAGE" == "train" ]]; then
  exps=("$MAIN_EXP" "${TIMETABLE_VARIANTS[@]}" "${FIXED_TIMETABLE_GRID[@]}" "${COUPLING_VARIANTS[@]}")
  for seed in $SEEDS; do
    for exp in "${exps[@]}"; do
      cmd="$PY -u runner_v3.py --config configs_ablation/${exp}.yaml --episodes $EPISODES --seed $seed"
      result="$ROOT_DIR/logs/${exp}_seed${seed}"
      submit_task "TransitDuet train ${exp} seed ${seed}" "$cmd" \
        "TransitDuet/train/${exp}/seed${seed}/episodes${EPISODES}" "$result"
    done
    for method in "${BASELINES[@]}"; do
      lower_warmup=20
      [[ "$method" == "fixed" ]] && lower_warmup=0
      cmd="$PY -u run_upper_comparison.py --method $method --episodes $EPISODES --lower_warmup $lower_warmup --seed $seed"
      result="$ROOT_DIR/logs/upper_${method}_seed${seed}"
      submit_task "TransitDuet baseline ${method} seed ${seed}" "$cmd" \
        "TransitDuet/baseline/${method}/seed${seed}/episodes${EPISODES}" "$result"
    done
    for variant in "${RULE_BASELINES[@]}"; do
      cmd="$PY -u run_baseline_rule.py --upper $variant --episodes $EPISODES --seed $seed"
      result="$ROOT_DIR/logs/baseline_${variant}_seed${seed}"
      submit_task "TransitDuet rule baseline ${variant} seed ${seed}" "$cmd" \
        "TransitDuet/rule/${variant}/seed${seed}/episodes${EPISODES}" "$result"
    done
  done
elif [[ "$STAGE" == "independent" ]]; then
  for seed in $SEEDS; do
    cmd="$PY -u runner_v3.py --config configs_ablation/${INDEPENDENT_ASSEMBLY}.yaml --episodes $EPISODES --seed $seed"
    result="$ROOT_DIR/logs/${INDEPENDENT_ASSEMBLY}_seed${seed}"
    submit_task "TransitDuet train ${INDEPENDENT_ASSEMBLY} seed ${seed}" "$cmd" \
      "TransitDuet/train/${INDEPENDENT_ASSEMBLY}/seed${seed}/episodes${EPISODES}" "$result"
  done
elif [[ "$STAGE" == "eval" ]]; then
  exps=("$MAIN_EXP" "${TIMETABLE_VARIANTS[@]}" "$INDEPENDENT_ASSEMBLY" "${FIXED_TIMETABLE_GRID[@]}" "${COUPLING_VARIANTS[@]}")
  for exp in "${exps[@]}"; do
    cmd="$PY scripts/per_ckpt_eval.py --exp $exp --config configs_ablation/${exp}.yaml --seeds \"${SEEDS// /,}\" --eps $EVAL_EPS --n_eval $N_EVAL --device $DEVICE"
    result="$ROOT_DIR/logs/eval_per_ckpt/${exp}"
    submit_task "TransitDuet eval ${exp}" "$cmd" \
      "TransitDuet/eval/${exp}/eps${EVAL_EPS}/n${N_EVAL}" "$result"
  done
else
  echo "Unknown STAGE=$STAGE. Use train, independent, or eval." >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run only; no scheduler tasks submitted."
  exit 0
fi

echo
echo "Submitted $submitted tasks."
if [[ "$DISPATCH" == "1" ]]; then
  if [[ "${#submitted_task_ids[@]}" -eq 0 ]]; then
    echo "No submitted task ids captured; refusing broad dispatch." >&2
    exit 3
  fi
  echo "Dispatching submitted TransitDuet tasks only..."
  dispatch_args=("$SCHEDULER" dispatch)
  for task_id in "${submitted_task_ids[@]}"; do
    dispatch_args+=(--task-id "$task_id")
  done
  "$SCHEDULER_PY" "${dispatch_args[@]}"
fi
