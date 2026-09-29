#!/usr/bin/env bash
# run_paper_round3.sh
# ===================
# Single reproducibility entrypoint for the round-3 TransitDuet submission.
# Goes from an empty `logs/` to every table/figure used in the paper, with no
# manual reach-into-archive steps.
#
# Stages (each can be skipped via flags below):
#   1. train      Train H_timetable_v4 main + variants + ablations + baselines.
#   2. eval_main  Per-checkpoint held-out eval (6 ckpts/exp x 20 eval episodes; pick best by composite).
#   3. eval_base  Validation-best eval for fixed/GA/CMA-ES baselines.
#   4. pareto     Pareto frontier from the validation-best main checkpoint per seed.
#   5. genrl      Cross-sigma + demand-shift generalisation eval.
#   6. agg        Aggregate all numbers into machine-readable CSVs.
#   7. figs       Regenerate all paper figures from the CSVs.
#
# Outputs land at:
#   logs/<exp>_seed<N>/                              training history + ckpts
#   logs/eval_per_ckpt/<exp>/<exp>_per_ckpt.csv      per-ckpt eval CSVs
#   logs/<exp>_seed<N>/pareto_frontier.json          Pareto JSONs (main policy)
#   logs/eval_generalization/{cross_sigma,demand_shift}/*.json
#   logs_remote/eval_per_ckpt/<exp>/                 mirror for figure scripts
#   results_remote/round3_composite.csv              aggregated composite table
#   paper/figures/{training_curves,ablation_bars,pareto_frontier,generalization,...}.pdf
#
# Usage:
#   bash scripts/run_paper_round3.sh                 # full pipeline (~14h on 1 GPU)
#   bash scripts/run_paper_round3.sh --quick         # 50-episode sanity sweep
#   bash scripts/run_paper_round3.sh --tier 1        # main+baselines only
#   bash scripts/run_paper_round3.sh --tier 2        # ablations only
#   bash scripts/run_paper_round3.sh --skip train    # skip a single stage
#   bash scripts/run_paper_round3.sh --only pareto   # only run a single stage
#
# Compared to run_paper_round2.sh this script adds stages eval_base,
# pareto, genrl, agg explicitly so that figures are not silently
# generated from stale archived data.

set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=300
SEEDS=(42 123 456 789 1001 1002 1003 1004 1005 1006)
EVAL_EPS=49,99,149,199,249,299
N_EVAL=20
GEN_SIGMAS=0.5,1.0,1.5,2.0,3.0
GEN_NEPS=10
PARETO_NEPS=5

QUICK=0
TIER=0
SKIP_LIST=()
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; EPISODES=50; EVAL_EPS=9,19,29,39,49; GEN_NEPS=3; PARETO_NEPS=2; shift ;;
    --tier)  TIER="$2"; shift 2 ;;
    --skip)  SKIP_LIST+=("$2"); shift 2 ;;
    --only)  ONLY="$2"; shift 2 ;;
    *)       echo "unknown arg: $1"; exit 2 ;;
  esac
done

stage_active() {
  local s="$1"
  if [[ -n "$ONLY" ]]; then
    [[ "$ONLY" == "$s" ]]
    return
  fi
  for x in "${SKIP_LIST[@]}"; do [[ "$x" == "$s" ]] && return 1; done
  return 0
}

MAIN_EXP=H_timetable_v4
TIMETABLE_VARIANTS=(H_hiro H_timetable_continuous H_timetable_no_context)
INDEPENDENT_ASSEMBLY=H_timetable_v4_independent_assembly
FIXED_TIMETABLE_GRID=(H_fixed_timetable_300 H_fixed_timetable_330 \
                      H_fixed_timetable H_fixed_timetable_390 \
                      H_fixed_timetable_420)
H_HIRO_ABLATIONS=(H_hiro_no_tpc H_hiro_no_holdfb H_hiro_no_csbapr \
                  H_hiro_no_hindsight H_hiro_no_morl \
                  H_hiro_fixed_fleet H_hiro_no_demand_noise)
COUPLING_VARIANTS=(H_haar H_tpc)
BASELINES=(fixed ga cmaes)
RULE_BASELINES=(rule_daganzo rule_xuan)

echo "=========================================================="
echo " TransitDuet round-3 reproducibility pipeline"
echo " EPISODES=$EPISODES  TIER=$TIER  QUICK=$QUICK"
echo " SEEDS=${SEEDS[*]}  ONLY=${ONLY:-<all>}  SKIP=${SKIP_LIST[*]:-<none>}"
echo "=========================================================="

## ------------------------- 1. train -------------------------
if stage_active train; then
  echo ">>> stage 1/7: train"
  python scripts/launcher.py --tier "$TIER" --episodes "$EPISODES" \
    $([[ $QUICK -eq 1 ]] && echo --quick)
  if [[ $TIER -ne 2 ]]; then
    echo ">>> stage 1/7b: train classical holding rule baselines"
    for variant in "${RULE_BASELINES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python -u run_baseline_rule.py \
          --upper "$variant" --episodes "$EPISODES" --seed "$seed"
      done
    done
    echo ">>> stage 1/7c: train independent-assembly timetable upper"
    for seed in "${SEEDS[@]}"; do
      python -u runner_v3.py \
        --config "configs_ablation/${INDEPENDENT_ASSEMBLY}.yaml" \
        --episodes "$EPISODES" --seed "$seed" --gpu
    done
  fi
fi

## ----------------- 2. eval_main (per-ckpt) ------------------
if stage_active eval_main; then
  echo ">>> stage 2/7: per-ckpt held-out eval (${MAIN_EXP} + variants + ablations)"
  if [[ $TIER -ne 2 ]]; then
    for exp in "$MAIN_EXP" "${TIMETABLE_VARIANTS[@]}" "$INDEPENDENT_ASSEMBLY" "${FIXED_TIMETABLE_GRID[@]}" "${COUPLING_VARIANTS[@]}"; do
      python scripts/per_ckpt_eval.py \
        --exp "$exp" --config "configs_ablation/$exp.yaml" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    done
  fi
  if [[ $TIER -ne 1 ]]; then
    for exp in "${H_HIRO_ABLATIONS[@]}"; do
      python scripts/per_ckpt_eval.py \
        --exp "$exp" --config "configs_ablation/$exp.yaml" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    done
  fi
fi

## --------------------- 3. eval_base ------------------------
if stage_active eval_base; then
  echo ">>> stage 3/7: baseline eval (fixed + GA + CMA-ES)"
  for method in "${BASELINES[@]}"; do
    if [[ "$method" == "fixed" ]]; then
      python scripts/eval_fixed_baseline.py \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    else
      python scripts/eval_baseline.py --method "$method" \
        --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
        --eps "$EVAL_EPS" --n_eval "$N_EVAL"
    fi
  done
  for variant in "${RULE_BASELINES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python eval_rule_baseline.py \
        --variant "$variant" --seed "$seed" --n_eval "$N_EVAL"
    done
  done
fi

## ----------------------- 4. pareto -------------------------
if stage_active pareto; then
  echo ">>> stage 4/7: Pareto frontier from validation-best ${MAIN_EXP} ckpts"
  python scripts/eval_pareto_hiro.py \
    --exp "$MAIN_EXP" --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
    --n_eval "$PARETO_NEPS"
fi

## ----------------------- 5. genrl --------------------------
if stage_active genrl; then
  echo ">>> stage 5/7: generalisation eval (cross-sigma + demand-shift)"
  python scripts/generalization_eval.py --mode all \
    --exp "$MAIN_EXP" --sigmas "$GEN_SIGMAS" \
    --seeds "$(IFS=,; echo "${SEEDS[*]}")" \
    --n_eps "$GEN_NEPS"
fi

## ------------------------- 6. agg --------------------------
if stage_active agg; then
  echo ">>> stage 6/7: aggregate composite-cost CSV + mirror everything figure scripts read"
  python scripts/composite_score.py --logs logs --last-k 30 \
    --out results_remote/round3_composite.csv

  ## Mirror everything figure scripts (make_result_figures.py) currently
  ## read from logs_remote/. This is what closes the "from empty logs/ to
  ## paper figures" loop --- without these copies the figure script silently
  ## skips Pareto / training_curves.
  mkdir -p logs_remote/eval_per_ckpt logs_remote/eval_generalization

  ## (a) per-ckpt eval CSVs (Table I/II + Fig 3)
  for exp in "$MAIN_EXP" "${TIMETABLE_VARIANTS[@]}" "$INDEPENDENT_ASSEMBLY" "${FIXED_TIMETABLE_GRID[@]}" "${H_HIRO_ABLATIONS[@]}" "${COUPLING_VARIANTS[@]}"; do
    [[ -d "logs/eval_per_ckpt/$exp" ]] || continue
    rm -rf "logs_remote/eval_per_ckpt/$exp"
    cp -r "logs/eval_per_ckpt/$exp" "logs_remote/eval_per_ckpt/"
  done

  ## (b) main / timetable-variant training diagnostics (Fig 2 training curves)
  ##     and Pareto JSONs (Fig 4 + Table III).
  for exp in "$MAIN_EXP" "${TIMETABLE_VARIANTS[@]}" "$INDEPENDENT_ASSEMBLY" "${FIXED_TIMETABLE_GRID[@]}" "${H_HIRO_ABLATIONS[@]}" "${COUPLING_VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
      src="logs/${exp}_seed${s}"
      [[ -d "$src" ]] || continue
      mkdir -p "logs_remote/${exp}_seed${s}"
      for f in diagnostics.csv history.json pareto_frontier.json; do
        [[ -f "$src/$f" ]] && cp "$src/$f" "logs_remote/${exp}_seed${s}/"
      done
    done
  done

  ## (c) baseline training histories (Fig 2 training curves).
  for method in "${BASELINES[@]}"; do
    for s in "${SEEDS[@]}"; do
      src="logs/upper_${method}_seed${s}"
      [[ -d "$src" ]] || continue
      mkdir -p "logs_remote/upper_${method}_seed${s}"
      [[ -f "$src/history.json" ]] && cp "$src/history.json" "logs_remote/upper_${method}_seed${s}/"
    done
  done

  ## (d) generalization JSONs (Fig 7 + Table V).
  for sub in cross_sigma demand_shift; do
    [[ -d "logs/eval_generalization/$sub" ]] || continue
    rm -rf "logs_remote/eval_generalization/$sub"
    cp -r "logs/eval_generalization/$sub" "logs_remote/eval_generalization/"
  done
fi

## ------------------------ 7. figs --------------------------
if stage_active figs; then
  echo ">>> stage 7/7: regenerate figures"
  python scripts/make_result_figures.py 2>&1 | tee logs/round3_figures.log
  python scripts/make_mechanism_figures.py 2>&1 | tee logs/round3_mechanism_figures.log
fi

echo "=========================================================="
echo " Done. Key outputs:"
echo "  - logs/eval_per_ckpt/<exp>/<exp>_per_ckpt.csv"
echo "  - logs/${MAIN_EXP}_seed<N>/pareto_frontier.json"
echo "  - logs/eval_generalization/{cross_sigma,demand_shift}/*.json"
echo "  - results_remote/round3_composite.csv"
echo "  - paper/figures/*.pdf"
echo "=========================================================="
