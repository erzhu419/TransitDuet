# Stage79 Fresh Native Credit Under the Feasible Decoder

Stage78 satisfied all historical conditional response budgets, yet period100
bounded-minus-zero reward remained negative. Freeze that decoder and test whether
fresh native task-return gradients point toward better control.

- Source: full Stage78 saved alpha/envelope, full Stage55 BC actors and forecaster.
- Roots: 310011, 310023, 310037, 310049, 310061, 310073, 310089, 310101; periods50/100.
- Two fresh independent credit batches of16 episodes each per root/period.
- Undiscounted native episode MC for both levels; restore upper constant call cost.
- Cross-fit baseline: remaining primitive steps times the opposite batch reward rate.
- Raw MC loss gradient, all actor parameters; no entropy or advantage normalization.
- Separate upper/lower symmetric perturbations at fixed Fisher radius0.001;
  exact raw Gaussian KL must pass the existing [0.5,2] nominal-radius check.
- Six variants: bounded source, zero residual, upper+/-, lower+/-. Same alpha throughout.
-32 independent evaluation seeds per root/period, paired environment/lower noise and
  standardized upper Gaussian noise. Updated upper proposals need not match.
-18 reward contrasts,65536 equal-root bootstrap draws, Bonferroni18. Tracking,
  cross-batch cosines and conditional gradient-noise estimates are descriptive.
- Full budget:3584 native episodes,4,300,800 steps; no critic/optimizer updates.
- Preflight: first full root, H300,2+2 credit and4 evaluation seeds per period;
  mechanical qualification only, no favorable-reward prerequisite for full dispatch.
- Scheduler dynamic node001-node006 placement,3CPU/3GB preflight,9CPU/8GB full.
- Training batches live only in server memory. Pull completion markers/compact JSON.

Limitations: teacher-initialized fixed-period direction test, not joint HRL training
or frequency-superiority evidence. Candidate actors are not certified by the source
historical command bound. No actor adoption or alpha/radius retuning; Stage67 HOLD stays.
