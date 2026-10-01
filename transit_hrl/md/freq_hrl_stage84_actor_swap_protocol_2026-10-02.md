# Stage84: fixed final-checkpoint actor swaps

Stage83 improved native reward but joint learning lost to lower-only at equal budget.
The next test isolates the direct learned-upper effect from the lower training difference.

## Frozen design

- Read Stage83 full round-eight checkpoints on the server; no further training or selection.
- U0 is source upper, UJ is joint-trained upper; L0 is source lower, LJ is joint-trained lower, LL is lower-only-trained lower.
- Evaluate U0/LJ, UJ/LJ, U0/LL, UJ/LL, U0/L0, UJ/L0 and source zero-residual control.
- Keep Stage78 alpha/envelope, values, both log-std, forecaster, source weights and optimizers unchanged.
- Same environment and upper/lower action-noise seeds across all seven variants; fresh Stage84 namespace.
- Eight fixed roots, periods 50/100, horizon1200, 32 evaluation seeds per root/period.
- Preflight uses the same full checkpoints, one root, horizon300, four seeds; mechanical checks only.
- Full budget: 3,584 episodes, 4,300,800 native steps, 32 checkpoint reads, zero updates/writes/traces.
- Scheduler dynamic node001-006 placement, 9 CPU/8192MiB per full task, eight rollout workers.

## Analysis and decision

All 22 reward/interaction contrasts use one equal-root bootstrap family (65,536 draws, Bonferroni22).
Primary upper effects hold LJ, LL or L0 fixed. Interaction is the upper effect with LJ minus that with LL.
Lower contrasts hold U0 or UJ fixed; joint/lower-only and zero comparisons use fresh paired evaluations.
If upper effects are positive but joint loses to lower-only, target lower learning/budget interference next.
If upper effects are inconclusive or harmful, retain the negative result rather than expand iterations or tune allocation.

## Limitations

Fixed checkpoint interventions are not counterfactual retraining or a flat-RL comparison.
Teacher initialization, fixed decoder/std and independent MC route remain; Stage67 critic-route HOLD and the closed frequency-superiority claim remain unchanged.
