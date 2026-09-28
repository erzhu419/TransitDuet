# Stage 45: training-objective actor acceptance readout

Implementation `7d3d6bda9c`; pre-outcome full freeze `15cecfd33c`.
Scheduler `t104071`:18 focused tests pass. Native preflight `t104072` and
aggregate `t104073` pass. Full `t104074`-`t104081`:eight roots, all exit0,
474-487s/root, dynamically node001/004/005/006. Offline aggregate `t104086`
passes exact first-batch pairing, rollback/accounting and independent statistics.

## Results

Sampled lower, deterministic upper/gate;16 pre-registered sampled-return
contrasts,65536 paired-root bootstrap draws,Bonferroni16. Zero positive,one
negative,fifteen inconclusive. Final accepted-update effects:

| Source method | Accepted minus vanilla: mean [CI] | Accepted minus frozen: mean [CI] |
|---|---|---|
| intrinsic_sham | -1.49111 [-4.07064,0.79453] | -5.26214 [-8.76267,-1.11608] |
| intrinsic_clock | -0.78578 [-3.62892,0.99694] | -3.80987 [-8.94228,0.48087] |
| task_sham | 0.59535 [-4.75339,5.43772] | -1.04226 [-5.16138,3.18534] |
| task_clock | 0.12893 [-4.02836,5.28269] | 1.47627 [-1.79134,5.57843] |

Accepted treatment rejects133/512 updates, restoring actor and full Adam while
keeping critic updates. Both treatments execute40960 actor/40960 critic steps;
35640 actor steps are retained. Rejection does not save optimization cost.
Method15052800 steps/370484 upper/528557 gate calls,12544 offline trajectory
audits,zero extra environment verification. All16 endpoints and all roots are
in the [71KB compact qualification](../results/pointmaze_actor_acceptance_stage45_v1_full_20260929_r1/qualification_summary.json).

## Limitations And Next Step

Reused development roots give conditional intervention evidence. Training
objective monotonicity is operationally enforced but does not establish native
learning benefit or noninferiority; intrinsic-sham accepted learning harms
return versus frozen. Keep this intervention experimental, not a shared-core
repair. Next isolate a full-episode task-credit estimator versus original GAE
with paired updates and frozen utility control; reward/option-cut switching
alone was already tested in Stage38. Do not extend seeds or tune acceptance.
