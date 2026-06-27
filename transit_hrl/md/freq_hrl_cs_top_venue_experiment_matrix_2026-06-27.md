# Freq-HRL CS Top-Venue Experiment Matrix

Date: 2026-06-27

Purpose: make the eight reviewer-critical experiments explicit, executable, and claim-gated. This prevents the manuscript from silently counting protocol smoke tests as top-venue empirical evidence.

- experiment count: `8`
- blocker count: `5`
- strong learned status: `registered_executable`
- learned cross-stress scenarios: `0`
- same-agency status: `external_truth_not_control_linked`
- venue-grade pairs: `3`

| id | experiment | current_status | claim_gate | artifact | paper_table | priority |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | strong learned RL baselines | registered_executable | PPO-family learned baselines need paired Sharpe/return/FocusScore; SAC/TD3 must remain explicit limitations until implemented. | transit_hrl/results/strong_learned_baseline_validation_latest/summary.json | main_baseline_table | 1 |
| E2 | learned-baseline cross-stress regime | registered_executable | At least four stress regimes with paired learned-policy rows. | transit_hrl/results/strong_learned_baseline_validation_latest/summary.json | stress_generalization_table | 2 |
| E3 | complete ablation main table | supported | Baseline matrix must include heuristic ablations plus learned rows when present. | transit_hrl/results/baseline_ablation_matrix_latest/summary.json | main_ablation_table | 3 |
| E4 | parameter-budget fair comparison | registered_executable | Freq-HRL, flat PPO, and generic HRL PPO must share state/action dimensions and parameter counts. | transit_hrl/results/strong_learned_baseline_validation_latest/parameter_budget.csv | parameter_budget_appendix | 4 |
| E5 | sensitivity and robustness | registered_executable | Report stress-registered sensitivity; do not claim universal hyperparameter robustness. | transit_hrl/results/sensitivity_robustness_matrix_latest/summary.json | robustness_appendix | 5 |
| E6 | runtime and sample efficiency | registered_executable | Report environment steps, iterations, elapsed seconds, and held-out objective proxy. | transit_hrl/results/strong_learned_baseline_validation_latest/sample_efficiency.csv | sample_efficiency_appendix | 6 |
| E7 | same-agency real Transit | boundary_registered | Current public truth-source coverage is not a full same-agency deployment loop unless the gate says supported. | transit_hrl/results/agency_demand_onboard_coverage_latest/summary.json | real_data_boundary_table | 7 |
| E8 | larger L2/L3 order-book replay | partial_scale | Finance/data-mining venues should see at least 20 venue-grade symbol-session pairs or a limitation. | transit_hrl/results/order_book_lobster_venue_grade_multisymbol/summary.json | order_book_scale_appendix | 8 |

## Scheduler Use

The companion CSV `cs_top_venue_scheduler_manifest.csv` contains the command for each row. Commands may be dispatched on CPU nodes; raw third-party data remain outside the committed repository unless explicitly licensed and compact.
