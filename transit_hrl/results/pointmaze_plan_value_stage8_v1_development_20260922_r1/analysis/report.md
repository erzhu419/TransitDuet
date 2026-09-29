# PointMaze Plan-Value Stage-8 Qualification

Protocol: `pointmaze_plan_value_qualification_stage8_v1`
Independent optimizer roots: 8
Decision: **stage9_not_authorized**

| Qualification contrast | ISE improvement [95% CI] | Status |
|---|---:|---|
| plan_refresh_vs_stale | 7.5735 [6.1774, 8.9695] | supported |
| plan_integrity_vs_perturbed | 0.4908 [0.4140, 0.5677] | supported |
| current_regime_information | 0.0255 [-0.1096, 0.1607] | inconclusive |
| oracle_timing_same_budget | -0.7686 [-0.8890, -0.6482] | contradicted |
| oracle_timing_100ms_delay_cost | -0.6639 [-0.7524, -0.5754] | contradicted |
| oracle_timing_250ms_delay_cost | -0.9638 [-1.0847, -0.8429] | contradicted |
| oracle_timing_500ms_delay_cost | -0.5704 [-0.6982, -0.4426] | contradicted |
| combined_oracle_reference | -0.7431 [-0.9563, -0.5298] | contradicted |

Oracle schedules are privileged task-qualification references, not deployable candidates. Non-stale schedule contrasts preserve the fixed upper-call budget.
