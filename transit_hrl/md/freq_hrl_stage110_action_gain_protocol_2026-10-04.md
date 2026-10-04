# Stage110 Native Directional Gain

Stage109 measured a weak learned-mean command channel. This step tests actual closed-loop reward sensitivity before training an upper policy for the current lower. All8 Stage107 final learned-hint lowers and their fixed Stage106 uppers are reused without donor selection. All392 flat feedback features,task,reward,lower Gaussian std and per-step noise are unchanged.

Use the existing full-scale decoder(alpha1),not a scale sweep. Upper runs deterministically at its current mean:mean plus/minus0.25 in each of4 latent coordinates at every scheduled renewal. Controls are unchanged mean,causal forecast,zero residual(exact forecast identity),and blind four-zero advice;12 variants total. Lower stays stochastic. Independently paired panels A/B each have8 fresh scenarios/root/period. Preflight uses2/panel,H300 and only checks mechanics. Full H1200/all8 roots is frozen before preflight.

Report36 equal-root pooled-panel endpoints:9 mean/directional reward gains over forecast,4 central slopes,4 curvatures,and forecast-minus-blind,per period. Bootstrap65,536 with seed(110,110110),Bonferroni36. A detectable direction requires a positive corrected gain CI,a matching nonzero corrected slope CI,and matching equal-root slope signs in both independent panels. Report both periods or partial/unsupported;do not choose a production action. Retain all36 endpoints and all roots.

Full budget:3,072 episodes/3,686,400 steps,41,472 upper calls. Scheduler dynamic node001-006,pref3CPU/3GiB,full5CPU/4GiB with2/4 workers. Pair checks compare initial feedback,entire exogenous measurement stream,and actual standardized lower innovations;zero residual must reproduce every forecast command and return. Temporary arrays stay inside workers and are discarded. No training,critic fits,checkpoint or raw-trace writes;pull compact JSON only. Source preparation remains an inherited cost.

## Limitations

This is a whole-episode upper-bias direction diagnostic,not a local action-conditioned Q,learned-policy advantage,or frequency-superiority claim. It changes deployment from old stochastic/attenuated upper execution and is a separate development protocol;do not pool it with Stage108. Gain magnitude must be reported even when statistically detectable. Stage67 critic HOLD remains unchanged.

Six focused tests passed. Native preflight t134593 on node005 passed all96 episodes/28,800 steps,including8 forecast/zero command identities and88 actual lower-innovation pairs;max innovation mismatch1.192e-6. [Preflight compact](../results/pointmaze_action_gain_stage110_preflight_20261004_r1/compact_summary.json). The previously frozen full cohort is admitted mechanically without inspecting reward signs for selection.

Full t134596-t134604 is complete,all exit0;the registered directional gain gate is not_supported. [Result and next step](freq_hrl_stage110_action_gain_result_2026-10-04.md).
