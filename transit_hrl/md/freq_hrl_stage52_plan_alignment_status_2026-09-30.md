# Stage52 Status

Implementation `b398d85fb6`; preflight and full preregistrations frozen and pushed in `61c2e6de25` before native outcomes. Protocol: `freq_hrl_stage52_plan_alignment_protocol_2026-09-30.md`.

- Unit `t107100` on node004 passed 24 tests, including causal reference construction, unchanged default joint rollout, Stage51 regression tests, full-budget reconstruction and signed CI aggregation.
- Native preflight `t107134` and qualification `t107137` on node005 passed. Compact summary: `results/pointmaze_plan_alignment_stage52_preflight_20260930_r1/qualification_summary.json`. Observed 14400 steps, 48 audits, 216 upper calls, 14400 lower calls, 56 plan fits and 56 audit fits; zero gate calls, optimizer updates or extra verification simulation.
- Full run `pointmaze_plan_alignment_stage52_full_20260930_r1`: `t107139/t107143` started on node005, `t107140/t107144` on node006, `t107141/t107145` on node001 and `t107142/t107146` on node004. Placement is dynamic within node001-node006, with no hard node pins. Each root reserves 9 CPU/12 GiB and uses eight persistent native workers.
- Full frozen budget: 3686400 native steps, 3072 audits, 55296 upper calls, 17408 plan fits and 17408 audit fits. Raw trajectories remain on servers; pull compact JSON only.

The first full attempt ended with seven audit failures (exit 1) and only `t107144` complete (exit 0); it is not a completed eight-root performance result. Each failure differed in one float32 reference element because the auditor reassociated `velocity * age * dt` as `velocity * (age * dt)`. Absolute differences were at most `4.440892e-16`. The audit now reproduces the executed multiply order without changing the controller or relaxing exact equality. Unit `t107169` passed all 25 tests, including native-target cancellation and rejection of a one-ULP mutation.

Recovery run: `pointmaze_plan_alignment_stage52_full_20260930_r2`, same complete eight-root roster, checkpoints, evaluation seeds, gains, window, periods, modes, endpoints and CI method. All eight roots rerun into a separate directory; original logs and partial trajectories remain remote. The first attempt submitted 1264 full-horizon paths across opened groups, reconstructing 1516800 native steps (1250 saved raw paths; 14 rejected before saving). The recovery budget is another 3686400 steps, not a free retry: 5203200 combined native steps if completed.

Full CIs are pending. The adoption gate remains positive deterministic return CIs for curve versus held target, reverse curve and frozen at both periods, with Bonferroni correction across all eight primary endpoints.

Next: qualify r2 on the server using the unchanged analyzer, save the compact summary and retain every signed endpoint. This analytic plan diagnostic is not learned Freq-HRL confirmation; identical inference call budgets do not imply identical FLOPs. Stage51's negative and inconclusive findings remain unchanged.
