# Stage105: Faster-Regime Conditioning Transfer

Stage104 did not support joint superiority at both periods, but lower conditioning improved both update orders. This follow-up changes one environment parameter: hidden regime dwell from0.8-1.6s to0.4-0.8s. All other physical, observation, reward and disturbance parameters stay fixed. Both learning and evaluation use this shifted distribution.

All eight original Stage96 U0/L0 initializers and Stage97 decoder/forecaster remain fixed. Each of two fresh joint learners uses eight simultaneous mean updates per actor (preflight:two), separate upper/lower A/B pools, matched per-actor scenario/noise rosters and nominal call-weighted KL0.001/update. Only conditioned lower credit shares upper innovations; evaluation uses the original independent noise mapping. No Stage98-104 trained weights, source refits or checkpoint selection.

Periods50/100, four final policies:joint-independent, joint-conditioned, original base, zero-plan. Full evaluation uses32 fresh paired paths/policy/period. All12 pairwise reward contrasts share equal-root bootstrap65536 and Bonferroni12 correction, seed(105,105105). All four primary lower bounds must exceed zero:conditioned-minus-independent and conditioned-minus-base at both periods. Every root, period and negative result is retained; no pooling with previous stages or sequential seed extension.

Preflight:one root, horizon300,160 native episodes /48,000 steps,16 mean updates,72 extra upper replay forwards, no checkpoints. Mechanical eligibility only, no return admission. Full:8 roots, horizon1200,34,816 episodes /41,779,200 steps,512 mean updates,73,728 extra upper replay forwards and32 final server-only checkpoints. Source preparation is a separate inherited cost.

Scheduler:dynamic node001-006, no required node, preflight3 CPU/3GB, full9 CPU/8GB; qualifier1 CPU/2GB. Only completion markers and compact summaries are pulled. Full starts only after native preflight qualification passes.

Implementation checks:5 focused tests plus3 unchanged Stage102 regressions passed in74.503s. Fixtures verify actual rollout task arguments, regime-driver change with identical nuisances, independent evaluation, source freezes, exact budget, final-only checkpoint writes and the four-primary/all12 gate. These are not native performance evidence.

## Scope
This tests adaptation across one registered regime-timescale shift using original initializers, not new initializations, zero-shot transfer or cross-domain generality. Std, values, Adam, forecaster and decoder stay frozen:it remains MC actor-mean learning, not full actor-critic or frequency superiority. Common-noise credit adds replay cost; realized trajectory KL is not matched. Stage67 critic HOLD and all prior negative results remain unchanged.
