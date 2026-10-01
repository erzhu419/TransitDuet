# Stage80 Same-Scenario Cross-Fit Credit

Stage79 fresh MC directions had no supported reward improvement. Test whether
removing exogenous scenario variation improves credit under the frozen decoder.

- Full Stage78 alpha/envelope, Stage55 actors/forecaster; no new scale calibration.
- Same eight roots and periods50/100; two disjoint batches of16 scenarios each,
  two independent policy/lower-noise rollouts per scenario. Environment seed fixed.
- Verify initial physical state and all exogenous history columns within each pair.
- Same trajectories feed two baselines: opposite-batch mean reward-rate control,
  and other independent same-scenario time-aligned MC return candidate.
- Undiscounted native episode objective at both levels; upper constant cost restored.
- Unnormalized raw loss gradients, no entropy; average replicate gradients before
  estimating covariance across scenario groups. Shared scenarios are not extra samples.
- Separate upper/lower, rate/scenario directions, fixed Fisher radius0.001 and
  symmetric +/- candidates. Decoder alpha unchanged for every nonzero variant.
- Ten variants: bounded source, zero residual and four +/- direction pairs.
-32 new independent evaluation seeds per root/period with common action noise.
-38 reward and8 credit contrasts,65536 equal-root bootstrap draws, Bonferroni46.
  Credit endpoints: scenario-minus-rate cross-batch cosine and log variance ratio.
- Full:6144 native episodes,7,372,800 steps; same rollout/score data for both methods.
- Preflight: first full root, H300,2 scenarios/batch,2 replicates,4 evaluation seeds;
  mechanical checks only, not favorable reward screening.
- Scheduler dynamic node001-node006,3CPU/3GB preflight and9CPU/8GB full.

Limitations: this is an offline training control variate using another rollout's
future, never an actor input or online critic. Teacher-initialized fixed-period
diagnostic, not joint HRL/frequency superiority. No actor adoption, checkpoint/trace
writes or reward-selected scale/radius; Stage67 HOLD remains. Pull compact JSON only.
