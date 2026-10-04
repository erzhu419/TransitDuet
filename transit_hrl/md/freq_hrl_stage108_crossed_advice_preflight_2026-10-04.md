# Stage108 Native Preflight

`t134477/t134478` both done/exit0 on node002. Five focused local tests passed. Both preflight and full protocols were frozen and pushed before native execution.

Mechanical gate passed:64 episodes/19,200 steps,144 upper calls,168 OLS/ridge fits each,16 matched upper-noise checks. Source-cell reaggregation exactly matches official qualification;max innovation mismatch2.384e-7. All source/std/value/Adam weights unchanged;zero policy updates,critic/forecaster fitting,checkpoint or raw-trace writes. Native wall21.86s. RAM was unmeasured,not zero;the later scheduler completion timestamp is not native runtime.

The4 primary descriptive means are tiny and mixed:50 learned-minus-forecast -0.00003667,minus-noise -0.00000404;100 +0.00001725,-0.00000166. This1-root/H300 run admits the full study mechanically,not by reward.

See [compact evidence](../results/pointmaze_crossed_advice_stage108_preflight_20261004_r1/compact_summary.json) and [frozen full protocol](../results/pointmaze_crossed_advice_stage108_full_20261004_r1/preregistration.json). Next:unchanged8-root fixed-policy execution,4,096 episodes/4,915,200 steps;all20 contrasts and4 primary corrected CI bounds. No Stage107 pooling or rescue tuning.

Full dispatch: `t134562-t134569` running on node001 (40 requested CPU total,no node pins,all node001-006 eligible); `t134570` waits for all8 roots. See [roster](../results/pointmaze_crossed_advice_stage108_full_20261004_r1/task_roster.json). This is scheduler placement,not a protocol change.

Terminal follow-up: all9 tasks done/exit0; [full result](freq_hrl_stage108_crossed_advice_result_2026-10-04.md) is not_supported. The dispatch sentence above records the earlier launch snapshot.
