# Stage96 Fresh Teacher Build

- t128396-t128404: all done, exit 0. Eight registered new roots retained; native total 25,984 episodes / 31,180,800 steps.
- Fixed-final controller384, critic warmup16 and BC64 budgets passed. All 32 final checkpoints passed source/period identity and frozen-component checks; no historical teacher, forecaster or decoder loaded.
- Equal-root initial/final controller diagnostic means: 506.363 -> 877.524; improvement at 8/8 roots. This is an initializer diagnostic, not a staged-route CI test.
- Final BC command MSE across both periods and all roots: 0.003899-0.005501. No admission threshold or checkpoint selection was applied.
- Wall time: 720.3-742.7 s/root; peak RAM: 4,318-4,444 MiB. Retrieved approximately 27KB compact JSON; checkpoints and labels remain server-only.

Next: reconstruct BC states on these new labels and freeze the first-feasible bounded decoder before native probes. Then rebuild fresh joint/lower donors and run the four registered staged-upper endpoints, retaining zero-plan and matched-independent controls. New-cohort staged performance has not been tested.
