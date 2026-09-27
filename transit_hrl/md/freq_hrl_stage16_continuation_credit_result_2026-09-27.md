# Stage-16 Continuation-Credit Result

Tasks `t101193/101194` completed. Each root has 96 paired interventions,
eight path-disjoint value fits (512 updates), 16 evaluation paths/mode and
24 upper calls/episode. Additional sampling: 240,000 fit + 57,600 evaluation
steps/root, beyond inherited supervision and controller reconstruction.
Only two compact JSON files (1,743,822 bytes total) were retrieved.

| Root | Stage-9 ISE | Stage-12 ISE | Short-only ISE | Bootstrap ISE |
|---|---:|---:|---:|---:|
| 209011 | 1.143830 | 1.144676 | 1.181110 | 1.240951 |
| 209061 | 0.838439 | 1.050864 | 0.954915 | 1.041449 |

**The frozen gate failed.** Bootstrap loses to short-only on both roots
and to Stage-9 on both. Both triggers retain conditional timing; this is
not the constant-action collapse seen in Stage-15 root 209011.

Out-of-path tail-contrast MSE is 0.002667/0.002917 versus the zero prediction's
0.002280/0.002959: relative skill -16.96%/+1.40%. Predicting separate suffix
values has not established useful action-contrast accuracy. Do not adopt
Stage-16 or rescue it with thresholds, longer fitting, or extra roots.

Next: isolate the critic objective using the cached endpoint pairs. Compare
paired-difference loss with absolute-value loss on identical training states,
architecture, initialization and normalization. Qualify on whole held-out
paths before any new controller rollout; no new environment sampling is needed.
