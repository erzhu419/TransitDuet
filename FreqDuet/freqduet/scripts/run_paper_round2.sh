#!/usr/bin/env bash
# run_paper_round2.sh
# ===================
# One-shot reproducibility entrypoint for the round-2 TransitDuet submission.
# Pins the unified protocol the paper claims:
#   * runner_v3 (HIRO goal-conditioned coupling) for all H_hiro / H_hiro_no_*
#   * runner_v3 with coupling_mode {haar, channels} for the alternative-coupling rows
#   * runner_v2 NOT used anywhere (silent channels-mode fallback bug, fixed in round 2)
#   * 300 episodes per seed, seeds 42/123/456
#   * elastic fleet [8, 16] sampled per episode
#   * demand noise sigma = 0.15
#   * per-checkpoint validation evaluation: 6 checkpoints * 20 eval episodes,
#     best per seed is selected, 3-seed mean +/- std reported.
#
# Usage:
#   bash scripts/run_paper_round2.sh                 # run full pipeline
#   bash scripts/run_paper_round2.sh --quick         # 50-episode sanity check
#   bash scripts/run_paper_round2.sh --tier 1        # main results only
#   bash scripts/run_paper_round2.sh --tier 2        # ablations only
#   bash scripts/run_paper_round2.sh --skip-train    # only re-run eval + figures
#   bash scripts/run_paper_round2.sh --skip-eval     # only re-run training
#
# Final outputs:
#   logs/<exp>_seed<N>/                              training history + ckpts
#   logs/eval_per_ckpt/<exp>/{<exp>_per_ckpt.csv}    held-out eval, 3-seed best
#   paper/figures/{training_curves,ablation_bars,...}.pdf

set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=300
SEEDS=(42 123 456)
EVAL_EPS=49,99,149,199,249,299
N_EVAL=20

QUICK=0
TIER=0
SKIP_TRAIN=0
SKIP_EVAL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)      QUICK=1; EPISODES=50; EVAL_EPS=9,19,29,39,49; shift ;;
    --tier)       TIER="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval)  SKIP_EVAL=1; shift ;;
    *)            echo "unknown arg: $1"; exit 2 ;;
  esac
done

H_HIRO_ABLATIONS=(H_hiro H_hiro_no_tpc H_hiro_no_holdfb H_hiro_no_csbapr \
                  H_hiro_no_hindsight H_hiro_no_morl \
                  H_hiro_fixed_fleet H_hiro_no_demand_noise)
COUPLING_VARIANTS=(H_haar H_tpc)
BASELINES=(fixed ga cmaes)

echo "=========================================================="
echo " TransitDuet round-2 reproducibility pipeline"
echo " EPISODES=$EPISODES  TIER=$TIER  QUICK=$QUICK"
echo " SEEDS=${SEEDS[*]}"
echo "=========================================================="

## ---------- 1. Training ----------
if [[ $SKIP_TRAIN -eq 0 ]]; then
  python scripts/launcher.py --tier "$TIER" --episodes "$EPISODES" \
    $([[ $QUICK -eq 1 ]] && echo --quick)
else
  echo "[skip] training"
fi

## ---------- 2. Per-checkpoint held-out eval ----------
if [[ $SKIP_EVAL -eq 0 ]]; then
  if [[ $TIER -ne 2 ]]; then
    for exp in H_hiro "${COUPLING_VARIANTS[@]}"; do
      python scripts/per_ckpt_eval.py \
        --exp "$exp" --config "configs_ablation/$exp.yaml" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    done
  fi
  if [[ $TIER -ne 1 ]]; then
    for exp in "${H_HIRO_ABLATIONS[@]:1}"; do  # skip first (=H_hiro, already done)
      python scripts/per_ckpt_eval.py \
        --exp "$exp" --config "configs_ablation/$exp.yaml" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    done
  fi

  ## Mirror eval outputs into logs_remote/ for figure scripts (which read there).
  mkdir -p logs_remote/eval_per_ckpt
  for exp in "${H_HIRO_ABLATIONS[@]}" "${COUPLING_VARIANTS[@]}"; do
    [[ -d "logs/eval_per_ckpt/$exp" ]] || continue
    rm -rf "logs_remote/eval_per_ckpt/$exp"
    cp -r "logs/eval_per_ckpt/$exp" "logs_remote/eval_per_ckpt/"
  done
else
  echo "[skip] per-ckpt eval"
fi

## ---------- 3. Figures (training curves + ablation bars + Pareto) ----------
python scripts/make_result_figures.py 2>&1 | tee logs/round2_figures.log

## ---------- 4. Aggregate composite-cost table ----------
python scripts/composite_score.py --logs logs --last-k 30 \
  --out results_remote/round2_composite.csv

echo "=========================================================="
echo " Done. Key outputs:"
echo "  - logs/eval_per_ckpt/<exp>/<exp>_per_ckpt.csv"
echo "  - paper/figures/{training_curves,ablation_bars,pareto_frontier,generalization}.pdf"
echo "  - results_remote/round2_composite.csv"
echo "=========================================================="
