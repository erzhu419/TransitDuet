# Stage93: partial fresh-sample replication, global gate not passed

t126019-t126027 all done/exit0; mechanical gate passed. Eight roots,38,912 episodes /46,694,400 steps; wall1333-1378s/root, peak4672-4913MiB. Pulled150,793-byte compact JSON only; checkpoints remain server-side.
All four lowers were freshly trained from L0 on new samples; independent controls were not reused from Stage90/91. U0 and final Stage88 UJ stayed frozen. Evaluation used32 fresh paired seeds/period/root and the unchanged normal RNG path.

| Primary: shared minus independent upper-noise training | period50: mean [CI]; positive roots | period100: mean [CI]; positive roots |
| --- | --- | --- |
| Matched frozen U0 | +0.34439 [+0.00893,+0.62536];7/8 | +1.17188 [+0.32637,+2.17136];8/8 |
| Matched frozen UJ | +0.33546 [-0.00684,+0.62101];7/8 | +1.14987 [+0.31566,+2.13350];8/8 |

Three of four primary endpoints are positive supported; fixed-UJ period50 is inconclusive. The preregistered all-four confirmation is not_supported: Stage92's global conditioning-benefit result was not fully confirmed on fresh training/evaluation samples. Do not replace this gate with supported secondary endpoints or pool stages.
All26 equal-root bootstrap65536 / Bonferroni26 CIs were independently recomputed, together with evaluation-mean contrasts, root/total costs and freeze metadata;16 positive,5 negative,5 inconclusive. The512 lower updates,64 checkpoints,16 donor loads,64 initialization checks,192 compositions and147,456 replay forwards match preregistration; all gradients used64 episodes/update and all evaluations excluded noise replay.
Descriptive first-update covariance trace ratios average0.663/0.664 at50 (14/16 root/batches lower) and0.367/0.365 at100 (16/16 lower), for U0/UJ. Reduced covariance did not guarantee passing every performance CI.
Secondary results favor the U0-trained common lower over the UJ-trained common lower at both evaluation uppers and periods: all four registered opposite-direction contrasts are negative supported. Direct UJ-minus-U0 contributions with either common lower remain positive supported at both periods. Base-minus-zero at100 remains negative supported.
Next: stop the all-four replication without seed/radius rescue or root exclusion. Separately preregister staged lower training under U0 followed by independent-noise upper training with that lower frozen, against matched staged independent-lower controls. Shared-upper-noise lower credit must not be applied directly to upper updates.

## Limitations
Same eight teachers, not teacher-population replication; dense task-return differences are small relative to return levels. Crossing-zero CI is neither equivalence nor harm. This is fixed-std/decoder MC mean learning, not full actor-critic, joint HRL or frequency superiority; shared replay adds compute and is not uniform variance dominance. Stage92's original result is retained, Stage91 remains unsupported, Stage67 HOLD and earlier negatives remain unchanged.
