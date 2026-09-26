# Stage-10 Full-Return Credit Result

Date: 2026-09-27

Run: `pointmaze_termination_mc_credit_v3_development_20260927_r1`

Tasks `t100891` and `t100892` completed. Both roots have 16 held-out paths
and four fixed-seed policy draws per path. Controller and fixed-plan episode
rows replay exactly, and every rollout makes one upper call in each of the
24 bins. Trigger training and checkpoint selection use the frozen 230,400
primitive steps per root.

| Root | Fixed ISE | Old stochastic ISE | MC-credit stochastic ISE | MC-credit return | MC early calls |
|---|---:|---:|---:|---:|---:|
| 209011 | 1.587736 | 1.781158 | 1.800670 | 886.868 | 17.781 |
| 209061 | 1.447218 | 1.380837 | 1.484142 | 917.471 | 18.031 |

**Development gate failed.** Full-return credit increased ISE against both
the old stochastic policy and fixed planning on both roots. Deterministic
selection again chose a nearly deadline-only actor; changing GAE lambda
alone did not produce conditional timing control.

The two roots were already revealed, and four policy draws do not create
additional independent roots. This result authorizes no superiority claim or
fresh-root extension under the frozen gate. The next development step is to
test action-aligned counterfactual timing credit, not another deployment
threshold adjustment.
