# Stage99 Fresh Lower: Full Result

All nine tasks t128854-t128862 completed with exit0 on node004/005/006. Server-only native qualification and the frozen 26-contrast aggregate matched the official result. All 64 registered final donor files exist.

## Primary Result
All four registered common-versus-independent contrasts are positive on each of eight new teacher roots and have strictly positive familywise-corrected CI lower bounds. conditioning_confirmation = supported.

| Registered endpoint | Mean reward difference | Corrected CI |
| --- | ---: | --- |
| 50/source_common_minus_source_matched | 0.295608 | [0.140810, 0.454251] |
| 50/joint_common_minus_joint_fixed | 0.290242 | [0.143615, 0.450776] |
| 100/source_common_minus_source_matched | 1.753319 | [0.896175, 2.786084] |
| 100/joint_common_minus_joint_fixed | 1.720775 | [0.875816, 2.739318] |

The unchanged 65,536 equal-root bootstrap uses the Stage93 seed and Bonferroni correction over all26 contrasts, without old-cohort pooling. Total: 16 positive, 4 negative and 6 inconclusive endpoints.

## Other Results
Common-noise lower policies trained under UJ are slightly worse than common-noise lowers trained under U0 under both evaluation uppers (four negative endpoints). The final joint-common compositions do not have CI-supported improvement over the original Stage98 joint policy. All results and donors are retained.

Exact cost: 38,912 native episodes / 46,694,400 steps, 512 lower-mean updates, 147,456 extra upper-replay forwards and 64 final checkpoint writes. Per-root native runtime: 1213.30-1249.89 s. Source, upper, std, values, Adam and decoder freezes passed. The saved compact JSON is 79355 bytes before pretty-printing; no trajectories or checkpoints were transferred to local disk.

## Next Step
Repeat the unchanged staged-upper rule from Stage94/95 using these new teachers and the registered U0-trained independent/common lower donors. Train both uppers from U0, preserve all fixed-joint/matched and crossed-actor controls, freeze the lower policies and use a separate fresh sampling roster and 28-contrast family.

Scope: fixed-upper conditional lower MC mean learning on the same native task with new teacher initializations, not full actor-critic training, unseen-task generalization or frequency-superiority proof.
