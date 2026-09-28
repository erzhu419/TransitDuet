# Stage 46: full-episode task actor credit readout

Implementation `98f5ddca0d`; pre-outcome freeze `dbed0be1ab`.
Combined65 checks pass:64 in `t104090`, corrected comparator in `t104097`.
`t104093` reran stale tests before explicit tests-directory staging was added.
Native preflight `t104102` and qualification `t104103` pass. Full
`t104104`-`t104111`:eight roots, all exit0,238-245s/root on node001/004/005/006.
Offline `t104112` passes paired batches/critic updates, accounting and statistics.

## Results

Eight registered sampled-return contrasts,65536 equal-root paired bootstrap
draws,Bonferroni8: all inconclusive,zero positive or negative. Final effects:

| Source | MC minus GAE: mean [CI] | MC minus frozen: mean [CI] | GAE minus frozen: mean [CI] |
|---|---|---|---|
| task_sham | 1.97747 [-3.58147,8.54266] | 0.26808 [-3.54884,3.96834] | -1.70939 [-5.49782,1.45545] |
| task_clock | 1.08525 [-5.04823,9.17061] | 0.26808 [-3.54884,3.96834] | -0.81717 [-6.10018,3.10080] |

First MC-minus-GAE intervals also cross0. MC deployment means match exactly
across critic variants at both snapshots/modes, as expected when actor credit
does not use the critic. They are not two independent MC learning successes.
Descriptive first-batch GAE/MC normalized inner products average-0.11812 and
-0.12953; these are estimator differences, not a causal explanation of utility.
Method7680000 steps/182092 upper/267012 gate calls,20480 actor/20480 critic
steps,6400 offline trace audits,zero extra native verification. Only the
[45KB compact summary](../results/pointmaze_episode_credit_stage46_v1_full_20260929_r1/qualification_summary.json)
is local; all raw trajectories/weights stay remote. All eight endpoints remain.

## Limitations And Next Step

Reused development roots give conditional evidence. Neither relative learning
benefit nor benefit over frozen is supported; no noninferiority was registered.
Full-episode credit alone is not an established repair. Next freeze a causal
state-conditioned full-task baseline with episode-held-out fits, against this
same-time LOO estimator and GAE, retaining a frozen utility reference. This
tests a variance-reduction hypothesis, not an identified failure cause. No
seed extension, checkpoint selection or post-outcome tuning follows this run.
