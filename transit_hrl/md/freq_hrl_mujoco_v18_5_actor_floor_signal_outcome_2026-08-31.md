# MuJoCo v18.5 Causal Actor-Floor Signal Outcome

## Decision

Status: `actor_floor_signal_stops_debt_feedback_direction`.

V18.5 completed both frozen H16 causal signal candidates on all 120 unchanged
paths without reading a correction target. None of the 12 preregistered
candidate-score assessments passed the complete discrimination rule. A
target-fitted FIR debt-feedback screen is therefore not authorized on this
reused panel.

## Best Signal

The best assessment was H16-hold mean floor-power excess. Its global rank AUC
was 0.9545 and its Hopper-conditioned AUC was 0.8442, so the signal contains
real information and is not merely an environment classifier. The unresolved
v17.14 Hopper OOD-chirp path ranked fourth globally.

The complete gate nevertheless failed. Only 4/7 actor-floor paths appeared in
the global top 7 and 5/7 appeared in the top 14; the registered rule required
at least 6/7 in the top 14. All seven appeared only by the top 28. A broad
feedback gate would therefore also act on many reference-feasible paths, which
is exactly where previous mechanisms lost the action trust region.

## Post-Hoc Boundary

After the frozen decision, combining hold and damped path aggregates can be
explored descriptively. Such a combination was not one of the preregistered six
scores. It cannot retroactively make v18.5 positive or authorize another tuned
screen on the same 120 paths. A dual-forecast mechanism would need one fixed,
theory-specified form and untouched fresh validation; otherwise the MuJoCo
actor-floor extension should close as negative development evidence.

## Efficiency

Scheduleurm task `t86055` completed on node003 with 16 single-threaded workers.
Worker runtime was 296.5 seconds, scheduler runtime was 310.3 seconds, and peak
RAM was 811 MB. Only the 264 KB preregistration and summary directory was
synchronized.

## Claim Boundary

V18.5 is a reused-path signal diagnostic. It does not establish a correction
policy, reward preservation, fresh-seed generalization, learned control,
leakage no-tradeoff, or manuscript support.
