# Cart-Pole Waypoint Factorial

[Frozen protocol](../protocols/plan_validity_cartpole_waypoint_factorial_2026-11-12.json).
The preceding qualification changed LQR position weight, fixed offset and
path bank together. On one fresh common bank of 32 paths, cross position
weights 10 and 30 with offsets 5 and 20, keeping the first call at zero.
In every cell run the original current and one-call-old waypoint conditions
with identical disturbances and 24 planner calls. No clock search, feedback
tuning, event-policy rerun or learned policy fitting.

Let D = ISE(lagged) - ISE(current). Compare the gain effect averaged over
clocks, the clock effect averaged over gains, and their difference in
differences on each whole path. Report all four cells, both path halves and
nominal paired 95% t intervals for whole-episode and late-option ISE.
Paths are the paired units; the eight treatments are not eight independent
samples. This diagnostic cannot change the original qualification.

Budget: 256 episodes, 307,200 rollout steps, 6,144 calls and 20
linearization steps. Two scheduler workers, one per LQR configuration.

## Completed Result

[Summary](../results/cartpole_waypoint_factorial_20261112/summary.json) and
[exit records](../results/cartpole_waypoint_factorial_20261112/completion.json).
t214885-t214886 exited 0 at the exact budget above, 37.18 CPU seconds and zero
policy fits. Only 7,748 final summary bytes returned.

| Timely versus lagged ISE gain | Offset 5, weight 10 | Offset 20, weight 10 | Offset 5, weight 30 | Offset 20, weight 30 |
| --- | ---: | ---: | ---: | ---: |
| Whole episode | -2.68% | -2.63% | +7.08% | +7.41% |
| Latter option halves | -3.08% | -2.18% | +6.80% | +8.17% |

All four whole-episode cell intervals exclude zero and both path halves
retain their cell's sign. On absolute whole-episode D, increasing position
weight changes D by +0.02458 [0.02299, 0.02618]. The offset main effect is
+0.00055 [-0.00479, 0.00589]; the interaction is +0.00096
[-0.00092, 0.00284]. These are nominal path-paired 95% intervals. The same
ordering holds for late ISE; the late weight-10/offset-20 cell interval crosses
zero. The sign reversal follows the feedback configuration at both tested
clocks, not a reversal caused by choosing 5 versus 20.

Weight 30 is not the better absolute controller: at both offsets its current
and lagged means are higher than their weight-10 counterparts. Do not promote
it for having a larger relative freshness gain. Preserve the failed parent
qualification and keep this designed task as a timing diagnostic. No Stage9
transfer or further gain search was released.
