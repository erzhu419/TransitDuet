#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-40}"
LAST_K="${LAST_K:-20}"
WORKERS="${WORKERS:-8}"
THREADS="${THREADS:-1}"
UPPER_WARMUP_EPS="${UPPER_WARMUP_EPS:-10}"
SEEDS="${SEEDS:-42,123,456,789,2026}"
LOGS_DIR="${LOGS_DIR:-logs_step8_aligned_wu10}"
OUT_DIR="${OUT_DIR:-results_freqduet/step8_aligned_wu10}"

CONFIGS="$(
  IFS=,
  echo \
F_freqduet_terminal_aligned_nofreq_hiro,\
F_freqduet_terminal_aligned_rawhistory_hiro,\
F_freqduet_terminal_aligned_lf_upper_hiro,\
F_freqduet_terminal_aligned_hf_lower_hiro,\
F_freqduet_terminal_aligned_allfreq_hiro,\
F_freqduet_terminal_aligned_swapped_hiro,\
F_freqduet_terminal_aligned_nopromotion_hiro,\
F_freqduet_terminal_aligned_noleakage_hiro,\
F_freqduet_terminal_promotion_conservative_spline2dir_waitattr_hiro
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
