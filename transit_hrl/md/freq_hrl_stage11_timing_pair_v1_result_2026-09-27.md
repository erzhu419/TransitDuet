# Stage-11 Timing-Pair Development Result

Tasks `t100902/903` completed. Both roots passed the registered exact-prefix,
one-upper-call-per-bin, fixed-controller replay, and 230,400 paired primitive
step checks. Each has 96 branch-fit pairs and 16 held-out evaluation paths.

| Root | Fixed ISE | Stage-9 ISE | Stage-11 ISE | Stage-11 return |
|---|---:|---:|---:|---:|
| 209011 | 1.587736 | 1.143830 | 1.713864 | 891.965 |
| 209061 | 1.447218 | 0.838439 | 0.945610 | 953.106 |

**Development gate failed.** Stage-11 loses to Stage-9 on both roots and to
fixed planning on one. The full-episode same-budget label has grouped
cross-validation MSE / target variance of 0.914 and 1.007, compared with
0.400 and 0.471 for the Stage-9 short-window label. The full-episode target
is therefore weakly predictable by the frozen causal feature model on these
training paths. This motivates an isolated short-window, same-budget label
test; it does not license threshold tuning or fresh-root confirmation.
