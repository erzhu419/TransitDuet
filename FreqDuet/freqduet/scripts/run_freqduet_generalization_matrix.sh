#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-100}"
LAST_K="${LAST_K:-50}"
WORKERS="${WORKERS:-8}"
THREADS="${THREADS:-1}"
UPPER_WARMUP_EPS="${UPPER_WARMUP_EPS:-10}"
SEEDS="${SEEDS:-7,11,17,23,31,37,42,43,53,61,71,83,97,109,123,127,149,456,789,2026}"
LOGS_DIR="${LOGS_DIR:-logs_generalization_ep${EPISODES}_wu10}"
OUT_DIR="${OUT_DIR:-results_freqduet/generalization_ep${EPISODES}_wu10}"

CONFIGS="$(
  IFS=,
  echo \
F_freqduet_gen_highnoise_nofreq_hiro,\
F_freqduet_gen_highnoise_rawhistory_hiro,\
F_freqduet_gen_highnoise_allfreq_hiro,\
F_freqduet_gen_highnoise_main_hiro,\
F_freqduet_gen_highnoise_nopromotion_hiro,\
F_freqduet_gen_highnoise_noleakage_hiro,\
F_freqduet_gen_odshift_nofreq_hiro,\
F_freqduet_gen_odshift_rawhistory_hiro,\
F_freqduet_gen_odshift_allfreq_hiro,\
F_freqduet_gen_odshift_main_hiro,\
F_freqduet_gen_odshift_nopromotion_hiro,\
F_freqduet_gen_odshift_noleakage_hiro,\
F_freqduet_gen_rushshift_nofreq_hiro,\
F_freqduet_gen_rushshift_rawhistory_hiro,\
F_freqduet_gen_rushshift_allfreq_hiro,\
F_freqduet_gen_rushshift_main_hiro,\
F_freqduet_gen_rushshift_nopromotion_hiro,\
F_freqduet_gen_rushshift_noleakage_hiro
)"

python3 scripts/run_freqduet_ablation.py \
  --configs "${CONFIGS}" \
  --seeds "${SEEDS}" \
  --episodes "${EPISODES}" \
  --last-k "${LAST_K}" \
  --workers "${WORKERS}" \
  --worker-threads "${THREADS}" \
  --upper-warmup-eps "${UPPER_WARMUP_EPS}" \
  --logs-dir "${LOGS_DIR}" \
  --out-dir "${OUT_DIR}" \
  "$@"
