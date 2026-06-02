#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-100}"
LAST_K="${LAST_K:-50}"
UPPER_WARMUP_EPS="${UPPER_WARMUP_EPS:-10}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKERS="${WORKERS:-${SCHEDULEURM_CPU_WORKERS:-1}}"
THREADS="${THREADS:-1}"
JOB_START="${SCHEDULEURM_CPU_SHARD_START:-${SCHEDULEURM_CPU_START:-0}}"
JOB_END="${SCHEDULEURM_CPU_SHARD_END:-${SCHEDULEURM_CPU_END:-600}}"
SHARD_LABEL="${SCHEDULEURM_CPU_SHARD_INDEX:-manual}"
GEN_TOTAL=360
BASE_TOTAL=240

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-start)
      JOB_START="$2"
      shift 2
      ;;
    --job-end)
      JOB_END="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --worker-threads)
      THREADS="$2"
      shift 2
      ;;
    --shard-label)
      SHARD_LABEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

SEEDS="7,11,17,23,31,37,42,43,53,61,71,83,97,109,123,127,149,456,789,2026"
GEN_CONFIGS="F_freqduet_gen_highnoise_nofreq_hiro,F_freqduet_gen_highnoise_rawhistory_hiro,F_freqduet_gen_highnoise_allfreq_hiro,F_freqduet_gen_highnoise_main_hiro,F_freqduet_gen_highnoise_nopromotion_hiro,F_freqduet_gen_highnoise_noleakage_hiro,F_freqduet_gen_odshift_nofreq_hiro,F_freqduet_gen_odshift_rawhistory_hiro,F_freqduet_gen_odshift_allfreq_hiro,F_freqduet_gen_odshift_main_hiro,F_freqduet_gen_odshift_nopromotion_hiro,F_freqduet_gen_odshift_noleakage_hiro,F_freqduet_gen_rushshift_nofreq_hiro,F_freqduet_gen_rushshift_rawhistory_hiro,F_freqduet_gen_rushshift_allfreq_hiro,F_freqduet_gen_rushshift_main_hiro,F_freqduet_gen_rushshift_nopromotion_hiro,F_freqduet_gen_rushshift_noleakage_hiro"
BASE_CONFIGS="F_freqduet_terminal_main_hiro,F_freqduet_gen_highnoise_main_hiro,F_freqduet_gen_odshift_main_hiro,F_freqduet_gen_rushshift_main_hiro"

clamp() {
  local value="$1"
  local low="$2"
  local high="$3"
  if (( value < low )); then
    echo "$low"
  elif (( value > high )); then
    echo "$high"
  else
    echo "$value"
  fi
}

JOB_START="$(clamp "$JOB_START" 0 "$((GEN_TOTAL + BASE_TOTAL))")"
JOB_END="$(clamp "$JOB_END" 0 "$((GEN_TOTAL + BASE_TOTAL))")"
if (( JOB_END < JOB_START )); then
  JOB_END="$JOB_START"
fi

GEN_START="$(clamp "$JOB_START" 0 "$GEN_TOTAL")"
GEN_END="$(clamp "$JOB_END" 0 "$GEN_TOTAL")"
BASE_START="$(clamp "$((JOB_START - GEN_TOTAL))" 0 "$BASE_TOTAL")"
BASE_END="$(clamp "$((JOB_END - GEN_TOTAL))" 0 "$BASE_TOTAL")"
WORKERS="$(clamp "$WORKERS" 1 "$((JOB_END - JOB_START > 0 ? JOB_END - JOB_START : 1))")"

export PYTHONPATH="${PYTHONPATH:-.}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

echo "FreqDuet promoted ep100 batch shard=${SHARD_LABEL} combined=[${JOB_START},${JOB_END}) gen=[${GEN_START},${GEN_END}) base=[${BASE_START},${BASE_END}) workers=${WORKERS} threads=${THREADS}"

if (( GEN_END > GEN_START )); then
  "$PYTHON_BIN" -u scripts/run_freqduet_ablation.py \
    --configs "$GEN_CONFIGS" \
    --seeds "$SEEDS" \
    --episodes "$EPISODES" \
    --last-k "$LAST_K" \
    --logs-dir "logs_generalization_promoted_ep${EPISODES}_wu10" \
    --out-dir "results_freqduet/generalization_promoted_ep${EPISODES}_wu10/shards/summary_${SHARD_LABEL}" \
    --workers "$WORKERS" \
    --worker-threads "$THREADS" \
    --upper-warmup-eps "$UPPER_WARMUP_EPS" \
    --job-start "$GEN_START" \
    --job-end "$GEN_END" \
    --skip-existing \
    --no-aggregate
fi

if (( BASE_END > BASE_START )); then
  "$PYTHON_BIN" -u scripts/run_freqduet_external_baselines.py \
    --configs "$BASE_CONFIGS" \
    --variants fixed_headway,rule_holding,rule_mpc \
    --seeds "$SEEDS" \
    --episodes "$EPISODES" \
    --last-k "$LAST_K" \
    --logs-dir "logs_external_baselines_promoted_ep${EPISODES}" \
    --out-dir "results_freqduet/external_baselines_promoted_ep${EPISODES}/shards/summary_${SHARD_LABEL}" \
    --workers "$WORKERS" \
    --worker-threads "$THREADS" \
    --job-start "$BASE_START" \
    --job-end "$BASE_END" \
    --skip-existing \
    --no-aggregate
fi

mkdir -p "results_freqduet/hpc_promoted_ep${EPISODES}_r1/shards"
touch "results_freqduet/hpc_promoted_ep${EPISODES}_r1/shards/${SHARD_LABEL}_${JOB_START}_${JOB_END}.done"
echo "FREQDUET_PROMOTED_EP${EPISODES}_BATCH_DONE shard=${SHARD_LABEL} combined=${JOB_START}-${JOB_END}"
