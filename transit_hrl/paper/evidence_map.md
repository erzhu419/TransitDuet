# Active Manuscript Evidence Map

Date: 2026-11-20. Main manuscript: [manuscript.md](manuscript.md).
Compact numerical source: [plan_validity_evidence.json](plan_validity_evidence.json).
Core figures and traceability: [figure evidence audit](figure_evidence_audit.md).

## Central argument

Paired keep/renew supervision improves fixed-call waypoint timing on PointMaze. Uniform clock-grid acquisition works without hidden event labels, and the same approach improves on fixed clocks in derived polygonal Reacher. The CEM experiment measures the trade-off between planner and scoring time, while Transit tests a separate local-to-episode question.

## Main evidence and comparators

The saved identifiers `temple_gate` and `deliberation_option_critic` refer
to the learned binary and cost-difference gates, respectively. They are
kept in result files for traceability.

| Manuscript location | Evidence | Unit and exact comparison | Interpretation |
| --- | --- | --- | --- |
| Method / Table 1 / Figures 1 and 4 | Stage 9 independent confirmation | Eight fresh optimizer roots, 16 held-out paths/root; shared controller, history trigger versus fixed/random-offset/current-only timing | All five frozen gates pass; central independent-seed confirmation |
| Results 5.1 / Table 8 | [Frozen timing portfolio](../results/pointmaze_stage9_timing_portfolio_20261112/summary.json) | Eight frozen builds, 64 fresh paired paths/root, 24 calls and shared CPU/steps/latency caps | 5.81% lower ISE than selected distance threshold; 21.56% lower than selection-based portfolio |
| Results 5.1 | [Distance diagnostic](../results/pointmaze_stage9_distance_diagnostic_20261112/summary.json) & [Matched feature ablation](../results/pointmaze_stage9_scale_ablation_20261112/summary.json) | Eight frozen builds, 32 and 64 fresh paired paths/root | 7.28% lower ISE than distance cut; 100 ms short history improves ISE 6.44% over current-only, while 500 ms degrades 5.85% |
| Results 5.2 / Table 2 / Figure 2a | [Clock-grid query ablation](../results/pointmaze_stage9_unprivileged_queries_20261118/summary.json) | Eight frozen roots, 64 paired paths/root; 192 query pairs/root | Uniform queries beat periodic (32.36%), random offsets (36.59%) and distance (6.42%). No detectable difference from event-stratified sampling; no noninferiority test |
| Results 5.3 / Table 3 / Figure 5a | [Timing baselines and MLP](../results/pointmaze_route_b_strong_baselines_20261118/summary.json) | Eight roots, 64 paired paths/root; ridge and MLP use the same 192 query pairs/root | Ridge beats native periodic (36.65%), learned binary (33.80%), distance (6.91%), cost difference (4.38%), and MLP (21.05%); not comparisons with original TEMPLE or Option-Critic |
| Results 5.4 / Table 4 / Figure 2b-d | [Multi-budget tracking](../results/pointmaze_stage9_budget_pareto_20261118/summary.json) | Eight frozen roots, 64 fresh paired paths/root across \(B \in \{12, 15, 20, 24, 30, 40\}\) | At matched call counts ridge improves tracking; with 20 or 24 calls it beats periodic timing with 24 or 30, but loses to the distance gate at those higher counts |
| Results 5.4 / Table 9 / Figure 5c,d | [CEM planner and control timing](../results/pointmaze_route_b_expensive_cem_planner_20261118/summary.json) | Eight roots, 32 paired paths/root per budget; 3 x 64 candidates per call | Ridge offers better tracking with fewer calls in two comparisons. Planner savings of 18.38 and 31.08 ms are offset by scoring; the measured control-time intervals do not establish a speedup |
| Results 5.5 / Table 5 / Figure 5b | [Polygonal Reacher confirmation](../results/reacher_polygonal_route_b_confirmation_20261118/summary.json) | Eight fresh roots, 64 paired paths/root; 192 query pairs/root, 10 ms steps | Beats random offsets (6.44%), selected clock (5.00%, 8/8 roots), shared periodic (4.27%) and learned binary (3.86%). Distance, cost-difference and native-periodic contrasts inconclusive |
| Results 5.5 / Table 10 | [Reacher applicability](../results/reacher_timing_applicability_20261112/summary.json) & [Cart-pole applicability](../results/cartpole_timing_applicability_20261112/summary.json) | Paired development paths, matched call counts, plus [cart-pole factorial](../results/cartpole_waypoint_factorial_20261112/summary.json) | Separates waypoint freshness from timing opportunity; identifies execution-policy dependence |
| Appendix A / Tables 6-7 / Figure 3 | [Stage298](../results/native_transit_service_confirmation_stage298_20261019_r1/report/result.json), [Stage301](../results/native_transit_service_comparison_stage301_20261020_r1/report/result.json), [Stage304](../results/native_transit_closed_loop_confirmation_stage304_20261021_r2/report/result.json) | Two 400-prefix single-intervention cohorts and 320 complete two-decision scenes | Joint single-intervention gains replicate; Stage304 fails the cost/wait and wait-retention criteria, with rare peak-fleet changes contributing to the cost difference |
