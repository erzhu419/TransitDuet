# Stage52 Status

Implementation `b398d85fb6`; preflight and full preregistrations frozen and pushed in `61c2e6de25` before native outcomes. Protocol: `freq_hrl_stage52_plan_alignment_protocol_2026-09-30.md`.

- Unit `t107100` on node004 passed 24 tests, including causal reference construction, unchanged default joint rollout, Stage51 regression tests, full-budget reconstruction and signed CI aggregation.
- Native preflight `t107134` and qualification `t107137` on node005 passed. Compact summary: `results/pointmaze_plan_alignment_stage52_preflight_20260930_r1/qualification_summary.json`. Observed 14400 steps, 48 audits, 216 upper calls, 14400 lower calls, 56 plan fits and 56 audit fits; zero gate calls, optimizer updates or extra verification simulation.
- Full run `pointmaze_plan_alignment_stage52_full_20260930_r1`: `t107139/t107143` started on node005, `t107140/t107144` on node006, `t107141/t107145` on node001 and `t107142/t107146` on node004. Placement is dynamic within node001-node006, with no hard node pins. Each root reserves 9 CPU/12 GiB and uses eight persistent native workers.
- Full frozen budget: 3686400 native steps, 3072 audits, 55296 upper calls, 17408 plan fits and 17408 audit fits. Raw trajectories remain on servers; pull compact JSON only.

Full CIs are pending. Do not tune the window, gain, period, mode or root roster from preflight outcomes. The adoption gate is positive deterministic return CIs for curve versus held target, reverse curve and frozen at both periods, with Bonferroni correction across all eight primary endpoints.

Next: after all eight tasks finish, run `scripts/analyze_pointmaze_plan_alignment_stage52.py --run-name pointmaze_plan_alignment_stage52_full_20260930_r1` as a 2-CPU scheduler task, save the compact qualification summary and retain every signed endpoint. This analytic plan diagnostic is not learned Freq-HRL confirmation; identical inference call budgets do not imply identical FLOPs. Stage51's negative and inconclusive findings remain unchanged.
