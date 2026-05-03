# scripts/ — TransitDuet paper pipeline

This directory contains the reproducibility pipeline for the round-3+
TransitDuet submission. The single entrypoint is
[`run_paper_round3.sh`](run_paper_round3.sh); everything else is invoked
from there in a fixed stage order.

## Active scripts (used by `run_paper_round3.sh`)

| Script | Purpose | Used by stage |
|---|---|---|
| `launcher.py` | Train all 8 H_hiro / 2 coupling-variant / 3 baseline runs ✕ 3 seeds | `train` |
| `per_ckpt_eval.py` | 6-checkpoint × 20-eval validation eval for H_hiro + variants + ablations (`runner_v3`, `--config configs_ablation/<exp>.yaml`) | `eval_main` |
| `eval_baseline.py` | Validation eval for GA + CMA-ES baselines | `eval_base` |
| `eval_fixed_baseline.py` | Validation eval for the Fixed-headway baseline | `eval_base` |
| `eval_pareto_hiro.py` | Pareto frontier (9 fleets × 5 eval) from validation-best H_hiro ckpts; writes `logs/H_hiro_seed*/pareto_frontier.json` | `pareto` |
| `generalization_eval.py` | Cross-σ + demand-shift generalisation eval (`runner_v3` + validation-best ckpts); writes `logs/eval_generalization/{cross_sigma,demand_shift}/*.json` | `genrl` |
| `composite_score.py` | Aggregate composite-cost CSV (per-episode formula, then 3-seed mean ± std) | `agg` |
| `make_result_figures.py` | Fig 2 (training curves) + Fig 3 (ablation bars) + Fig 4 (Pareto) + Fig 7 (generalisation) | `figs` |
| `make_mechanism_figures.py` | Fig 5 (θ evolution) + Fig 6 (λ convergence) + δ-utilisation, all from H_hiro diagnostics | `figs` |

All paper protocol parameters (`--eps 49,99,149,199,249,299`, `--n_eval 20`,
`--seeds 42,123,456`) are passed by `run_paper_round3.sh` and are also the
defaults of each individual script, so a single-script invocation also
matches the paper.

## Deprecated (do not use; kept for provenance only)

* [`../_archive_pre_round2/scripts/aggregate.py`](../_archive_pre_round2/scripts/aggregate.py) — Round-1 ablation aggregator that reads legacy
  `A_full / B_no_holding_feedback / ...` directory names and the old
  `composite_score.py` semantics (per-method averaging instead of
  per-episode aggregation). Replaced by `composite_score.py` + the
  per-checkpoint validation flow.
* [`../_archive_pre_round2/eval_generalization/demand_shift_G/`](../_archive_pre_round2/eval_generalization/demand_shift_G/) — round-1
  demand-shift JSONs (`G_in_dist_seed*.json`, `G_ood_noisy_seed*.json`).
  Round-3 replaces these with `demand_in_dist_seed*.json` and
  `demand_ood_noisy_seed*.json` written by the rewritten
  `generalization_eval.py`.
* [`../_archive_pre_round2/logs_remote/A_full_seed*/`](../_archive_pre_round2/logs_remote/) — channels-mode `runner_v2.py` baseline runs that
  preceded the HIRO main result. Not referenced by the round-3 paper or
  any active script.

If you are re-running the pipeline from scratch on a fresh checkout, only
the *Active scripts* section above is in scope. The archive directory is
read-only history.
