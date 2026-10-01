# Stage81 Result and Next Step

Preflight t120491/492 and full t120517-525 all exited0; code47e54192a4,
16 focused tests passed.8 roots,8192 episodes,9,830,400 native steps;
192 parameter-part checks and all exact-KL checks passed. Sources/Adam/decoder
and Fisher radius.001 stayed frozen. Dynamic compute246-272s/root,4.5-4.8GB peak.

Mean-only positive directions improve source reward, Bonferroni62 root-bootstrap CI:

| Period / Actor | Mean plus minus source | Std plus minus source |
| --- | --- | --- |
| 50 / upper | +.05590 [.02470,.08856] | -.00018 [-.00332,.00270] |
| 50 / lower | +.69743 [.43876,.88893] | +.00081 [-.00900,.01247] |
| 100 / upper | +.11348 [.06352,.19649] | +.00513 [-.00596,.02290] |
| 100 / lower | +.81766 [.00747,1.55836] | -.00460 [-.03023,.01848] |

All four mean-plus versus mean-minus and mean-plus versus std-plus contrasts
are positive, as are full-plus versus source; full-plus versus mean-plus is
inconclusive. Mean-only std is unchanged; full std shifts stay below.127%,
std-only shifts reach3.20%. Gains support mean-learning, not just noise shrinkage.

## Limitations
Gains are small and teacher-initialized. Lower50 beats zero residual by+.70524
CI [.30707,.98309]; other mean-plus/zero comparisons are inconclusive.
100 source trails zero by-.58651 CI [-1.31131,-.01660]. Subspace gains are not
additive or joint HRL evidence; scenario MC needs independent simulator replicates.
Stage67 HOLD remains. No actor adoption, optimizer step or checkpoint write.

Next: joint upper/lower mean-only directions with radius-matched single-level
controls, frozen std/decoder and independent evaluation before iterative training.
[Compact evidence](../results/pointmaze_actor_parts_stage81_full_20261002_r1/qualification_compact.json).
