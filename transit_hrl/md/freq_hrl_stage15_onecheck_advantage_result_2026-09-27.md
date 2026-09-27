# Stage-15 One-Check Advantage Result

Tasks `t101096/101097` completed. Both roots passed seed separation, 96
training pairs, exact Stage-12 reference replay, 16 evaluation paths and
24-call budgets. Additional fitting used 240,000 steps/root; reference and
candidate evaluations each used 19,200 steps/root.

| Root | Fixed ISE | Stage-9 ISE | Stage-12 ISE | Stage-15 ISE | Stage-15 return |
|---|---:|---:|---:|---:|---:|
| 209011 | 1.587736 | 1.143830 | 1.144676 | 1.587736 | 899.557 |
| 209061 | 1.447218 | 0.838439 | 1.050864 | 1.093164 | 945.143 |

**Frozen development gate failed on both roots.** Root 209011 is exactly
fixed-at-bin-start planning on every evaluation path. Root 209061 retains
conditional timing (10.0625 early calls/path), but loses in ISE to Stage-12
and Stage-9. Its return improves over Stage-12; this does not change the
registered ISE decision.

The full-episode regression selected Ridge alpha 10000/1000. Leave-one-path-
out MSE is 0.003775/0.004412, compared with 0.003764/0.004617 for predicting
each fold's training mean. Root 209011 has positive predictions on all 96
fit samples and all 96 out-of-fold samples. This explains the loss of
conditional action discrimination; it is not a rollout or budget failure.

Do not adopt Stage-15, tune its threshold, iterate it further, or add roots.
Stage-9 remains the performance reference. The next training redesign should
separate near-term action credit from estimated continuation value and
establish randomized action coverage, rather than use another full-episode
regression followed by an unrestricted greedy replacement. That redesign
is not implemented or validated by this result.
