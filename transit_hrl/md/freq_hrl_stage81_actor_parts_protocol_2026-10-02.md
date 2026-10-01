# Stage81 Mean Learning Versus Exploration Scale

Stage80 scenario-credit directions improved reward but did not isolate mean
learning from reduced Gaussian exploration. Test the parameter blocks directly.

- Full Stage78 decoder alpha/envelope and Stage55 actors/forecaster unchanged.
- Same eight roots and periods50/100; fresh Stage81 seeds. Two disjoint batches
  of16 scenarios, two independent action-noise rollouts per scenario.
- Same scenario cross-fit undiscounted task-MC gradient for all three parts.
- Full: all actor parameters. Mean-only: log-std bit-exact frozen. Log-std-only:
  entire mean network bit-exact frozen. Other actor and source/Adam unchanged.
- Each direction independently matches fixed Fisher KL0.001, with symmetric +/-
  candidates and the existing exact raw Gaussian KL check. No radius/alpha search.
-14 variants: bounded source, zero residual, upper/lower full/mean/log-std +/-.
-32 independent evaluation seeds per root/period, common environment/action noise.
-62 reward contrasts,65536 equal-root bootstrap draws, Bonferroni62; all contrasts
  reported. Parameter gradients, actual std shifts and tracking are descriptive.
- Full8192 native episodes,9,830,400 steps;1024 credit,7168 evaluation episodes.
- Preflight first full root H300,2 scenarios/batch,2 replicates,4 evaluation seeds;
  mechanical only, not reward screening. Full runs after mechanical pass.
- Scheduler dynamic node001-node006,3CPU/3GB preflight,9CPU/8GB full.

Limitations: radius-matched subspace directions are not an additive decomposition
of one full update's reward. Teacher-initialized fixed-period diagnostics, not joint
HRL/frequency superiority. Other-rollout future is only a training baseline, never
an actor input. No optimizer/critic training, actor adoption, traces or checkpoints;
Stage67 HOLD remains. Pull completion markers and compact JSON only.
