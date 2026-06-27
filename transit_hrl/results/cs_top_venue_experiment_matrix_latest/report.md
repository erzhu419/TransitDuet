# CS Top-Venue Experiment Matrix

This is an experiment readiness matrix, not a claim of completion. Rows only upgrade paper claims when their artifact gates pass.

- experiment count: `8`
- blocker count: `5`
- strong learned status: `registered_executable`
- learned stress scenarios: `0`

| id | experiment | current_status | claim_gate | paper_table | priority |
| --- | --- | --- | --- | --- | --- |
| E1 | strong learned RL baselines | registered_executable | PPO-family learned baselines need paired Sharpe/return/FocusScore; SAC/TD3 must remain explicit limitations until implemented. | main_baseline_table | 1 |
| E2 | learned-baseline cross-stress regime | registered_executable | At least four stress regimes with paired learned-policy rows. | stress_generalization_table | 2 |
| E3 | complete ablation main table | supported | Baseline matrix must include heuristic ablations plus learned rows when present. | main_ablation_table | 3 |
| E4 | parameter-budget fair comparison | registered_executable | Freq-HRL, flat PPO, and generic HRL PPO must share state/action dimensions and parameter counts. | parameter_budget_appendix | 4 |
| E5 | sensitivity and robustness | registered_executable | Report stress-registered sensitivity; do not claim universal hyperparameter robustness. | robustness_appendix | 5 |
| E6 | runtime and sample efficiency | registered_executable | Report environment steps, iterations, elapsed seconds, and held-out objective proxy. | sample_efficiency_appendix | 6 |
| E7 | same-agency real Transit | boundary_registered | Current public truth-source coverage is not a full same-agency deployment loop unless the gate says supported. | real_data_boundary_table | 7 |
| E8 | larger L2/L3 order-book replay | partial_scale | Finance/data-mining venues should see at least 20 venue-grade symbol-session pairs or a limitation. | order_book_scale_appendix | 8 |
