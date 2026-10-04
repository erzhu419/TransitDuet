# Stage107 Native Preflight

Run: `pointmaze_optional_plan_stage107_preflight_20261004_r1`. Tasks `t132280` (node005) and `t132281` (node001) both done/exit0. Nine local focused tests passed. Both preflight and full protocols were frozen and pushed before native execution.

Mechanical gate passed: source-cell reaggregation exactly matches official qualification;240 episodes/72,000 steps,12 lower-only updates,324 upper calls,504 OLS/ridge evaluations each. Upper,std,values,Adam,source and decoder/forecaster remain frozen. Exact KL range0.001000021-0.001000161;old-logp maximum difference0.000012636. Native measured loop wall61.01s;sampled process-tree peak1412MiB. Qualification RAM was unmeasured,not zero.

Six primary descriptive differences are near zero and mostly negative:period50 learned-minus-blind -0.000513,minus-forecast +0.000000611,minus-own-blinded -0.000428;period100 -0.000250,-0.000043,-0.000323. One root/H300 does not establish performance; no reward gate,tuning or cohort selection follows this preflight.

Artifacts: [compact summary](../results/pointmaze_optional_plan_stage107_preflight_20261004_r1/compact_summary.json),[terminal roster](../results/pointmaze_optional_plan_stage107_preflight_20261004_r1/task_roster.json),[frozen full protocol](../results/pointmaze_optional_plan_stage107_full_20261004_r1/preregistration.json). Only compact JSON was pulled;no checkpoints or raw native traces.

Full study dispatched unchanged: training `t132282-t132289` running,two roots each on node001/004/005/006;qualification `t132290` waits for all8 roots. All node001-006 remain eligible with no node pin. Cost52,224 new episodes/62,668,800 steps,384 lower-mean updates. See [dispatch roster](../results/pointmaze_optional_plan_stage107_full_20261004_r1/task_roster.json). Interpret all20 corrected contrasts and require all6 primary positive lower CI bounds. Stage106 negative evidence and Stage67 critic HOLD remain unchanged.

Terminal follow-up: all9 tasks done/exit0; [full result](freq_hrl_stage107_optional_plan_result_2026-10-04.md) is not_supported. The dispatch sentence above records the earlier launch snapshot.
