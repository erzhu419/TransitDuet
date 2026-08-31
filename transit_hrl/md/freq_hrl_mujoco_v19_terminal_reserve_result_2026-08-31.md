# MuJoCo v19 Terminal-Reserve Development Result

## Status

The frozen v19 panel is complete and **not supported** for candidate
selection. All 180 registered cells finished. No consistency coefficient was
selected, and the v19 optimizer, rollout, selection, and evaluation roots are
retired.

This is a development result. It is not confirmatory or manuscript evidence.

## Frozen panel

- Environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5.
- Arms: capacity-matched raw context, projected reserve without consistency,
  and fixed consistency coefficients 0.01, 0.03, and 0.10.
- Replicates: 12 fresh optimizer roots per environment and arm.
- Training: 128 PPO iterations, 512 transitions per training path.
- Heldout evaluation: 8 seeds crossed with five disturbance conditions.
- Checkpoint selection used reward-only selection paths. Heldout terminal
  metrics did not select checkpoints.

## What held

The terminal-reserve mechanism itself was valid for the raw projected arm and
the 0.01 and 0.03 consistency arms:

- zero heldout certificate violations;
- realized upper and lower prefix powers stayed within the frozen budgets;
- recursive fallback stayed within its registered bound;
- projection convergence met the 0.95 gate.

The 0.10 arm missed the global validity gate because its minimum cell-level
projection convergence rate was 0.9443. This is a numerical convergence
failure, not evidence that the prefix certificate is invalid.

## What failed

No fixed coefficient passed both the correction and reward gates.

| Coefficient | Worst-environment correction reduction | Supported environments | Result |
|---:|---:|---:|---|
| 0.01 | -0.1305 | 0/3 | correction and reward gates failed |
| 0.03 | 0.0306 | 0/3 | pooled direction was favorable but its CI crossed zero; reward failed |
| 0.10 | 0.1439 | 1/3 | Hopper correction improved, but validity and reward gates failed |

Hopper is the binding physical tradeoff. Its mean reward was 171.31 for the
unprojected context arm and 134.34 for projected reserve without consistency.
The 0.03 and 0.10 consistency arms reduced component correction, but mean
reward fell further to 118.04 and 116.97. HalfCheetah also lost reward at the
largest coefficient. Walker2d was comparatively insensitive because the
projector changed very few total actions.

## Training-horizon diagnostic

Only numeric learning-curve summaries were read on the compute nodes; full
training histories and checkpoints remained server-side.

For the 12 Hopper projected-reserve runs, the selected checkpoint iteration
had mean 119.67 and median 123, with 6/12 selections at iteration 127. From
iteration 95 to 127, the selection score improved in 12/12 runs, with mean
change +0.1008 and median change +0.0931. The matched raw arm improved in only
5/12 runs over the same tail.

Therefore 128 iterations truncated constrained learning. This does not rescue
v19, but it justifies a fresh long-horizon development panel.

## v20 decision

v20 must not tune on or reuse v19 roots. It tests the following precommitted
mechanism changes:

1. extend training to 384 iterations;
2. delay consistency until reward learning has stabilized, then linearly ramp
   it from 0 to its target coefficient;
3. separate the consistency correction from the PPO reward step;
4. project and backtrack the consistency gradient so the same-minibatch PPO
   reward surrogate and any active native leakage surrogate do not worsen;
5. retain the exact terminal-reserve certificate and capacity-matched raw and
   projected controls.

Passing v20 would authorize a seed-disjoint confirmation freeze. Failing v20
would reject actor-to-projector consistency as the current solution to the
Hopper tradeoff; it would not justify another coefficient search on these
development roots.
