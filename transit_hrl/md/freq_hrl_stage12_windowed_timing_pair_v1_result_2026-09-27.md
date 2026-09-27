# Stage-12 Windowed Timing-Pair Development Result

Tasks `t100944/945` completed. Both roots have 96 paired branch-fit
opportunities, 16 held-out evaluation paths, 230,400 paired replay steps,
identical causal prefixes, and exactly one upper call per 50-step bin in each
arm. Fixed-controller replay matches Stage-9/11.

| Root | Fixed ISE | Stage-9 ISE | Stage-12 ISE | Stage-12 return | Stage-12 early calls/path |
|---|---:|---:|---:|---:|---:|
| 209011 | 1.587736 | 1.143830 | 1.144676 | 925.908 | 8.875 |
| 209061 | 1.447218 | 0.838439 | 1.050864 | 938.525 | 5.000 |

**Development gate failed.** Stage-12 beats fixed planning on both roots but
does not beat Stage-9 on either. Its ISE disadvantage to Stage-9 is 0.000846
and 0.212425, respectively. No fresh-root confirmation or threshold retuning
is authorized by the frozen protocol.

The 50-step target is not simply uninformative on the static branch-fit
distribution: across the 96 pairs/root, its correlation with the paired
full-episode ISE advantage is 0.758/0.610. The top quartile of leave-one-path-
out scores has mean 50-step advantage +0.051/+0.072 and mean full-episode
advantage +0.058/+0.070. This is a branch-fit diagnostic, **not** a deployed
policy result. The closed-loop mismatch remains unresolved; fewer early calls
alone do not establish the cause.
