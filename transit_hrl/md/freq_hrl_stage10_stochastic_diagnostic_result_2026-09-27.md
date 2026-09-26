# Stage-10 Stochastic-Termination Diagnostic Result

Date: 2026-09-27

Run: `pointmaze_termination_stochastic_v2_diagnostic_20260926_r1`

Scheduler tasks `t100835` and `t100836` completed. Both compact results have
16 held-out paths, four Bernoulli draws per path, and 24 upper calls per
rollout. Deterministic rows match Stage-10 v1 exactly; fixed rows match after
excluding wall-clock timing fields. All scored rows are finite, causal, and
respect the one-call-per-50-step-bin budget.

| Root | Mode | Mean episode ISE | Mean return | Early calls/episode |
|---|---|---:|---:|---:|
| 209011 | Fixed | 1.587736 | 899.557 | 0 |
| 209011 | Stage-9 candidate | 1.143830 | 931.692 | 13.750 |
| 209011 | Deterministic PPO | 1.691883 | 890.746 | 0 |
| 209011 | Stochastic PPO | 1.781158 | 887.454 | 17.766 |
| 209061 | Fixed | 1.447218 | 915.771 | 0 |
| 209061 | Stage-9 candidate | 0.838439 | 968.832 | 11.875 |
| 209061 | Deterministic PPO | 1.424981 | 913.717 | 0.562 |
| 209061 | Stochastic PPO | 1.380837 | 923.253 | 17.578 |

Stochastic termination is active, but not consistently competent: it improves
over fixed on one root and degrades on the other. The sampled policy takes
more early calls than the Stage-9 candidate while remaining worse on both
roots. Its early-call rate is similar in bins where the candidate calls early
and where the candidate waits, so merely lowering the deterministic threshold
is not a sufficient repair. The registered diagnosis therefore points to
training credit and conditional action quality, not only deployment.

These are two revealed development roots. The four draws per path are policy
repetitions, not independent optimizer roots; no superiority interval or paper
claim is licensed.
