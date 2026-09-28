# Stage 44: signed microstep native readout

Implementation `9d1bf4b503`; full pre-outcome freeze `2c47eb2f48`.
Scheduler `t103802`:23 regressions pass in149.561s. Native preflight `t103805`
and aggregation `t103813` pass. Full `t103815`-`t103822`: eight roots, all exit0,
99-101s/root, dynamic node001/004/005/006. Offline aggregation `t103825` exits0,
paired rosters/accounting and independent16-endpoint bootstrap pass.

## Results

Sampled lower, deterministic upper/gate. Adjusted Bonferroni-16 paired-root
percentile CIs, 65536 draws. One negative endpoint,15 inconclusive,0 positive.

| Source update | +1/16 minus zero: mean [CI] | Full minus +1/16: mean [CI] |
|---|---|---|
| intrinsic_sham | -0.01981 [-0.07107, 0.02493] | -0.02425 [-1.15233, 1.25419] |
| intrinsic_clock | -0.01134 [-0.04368, 0.02958] | 0.30312 [-0.75807, 1.35798] |
| task_sham | 0.23238 [-0.06691, 1.04996] | -1.00451 [-1.61020, -0.27475] |
| task_clock | 0.25914 [-0.05122, 1.08614] | -0.78720 [-1.65525, 0.16331] |

Task-sham full displacement is worse than the positive microstep;7/8 root
differences are negative. All four signed slopes and negative-microstep versus
zero intervals cross0. The registered positive-micro/slope plus negative-full
conjunction is not met. Full-minus-zero and deterministic results are retained
as descriptive values in the [compact qualification](../results/pointmaze_signed_microstep_stage44_v1_full_20260929_r1/qualification_summary.json).
Method3993600 steps/99113 upper/140230 gate calls,3328 offline native trace
audits. Zero optimizer/extra environment verification steps. Only53KB is local.

## Limitations And Next Step

Reused development weights give conditional diagnosis, not confirmation or a
learner repair. Microstep benefit over no update is unproven. Task-sham positive
micro gain is concentrated in root310089 (+2.09549); its mean upper-call count
also changes by1/16. These paired outcomes do not identify the causal mediator.
Preserve all roots and scales. Next freeze a training-batch-only acceptance/
rollback intervention for the original full-batch actor objective, with a
paired native utility control. Do not choose a learning rate from these returns.
