# Stage-10 Learned-Termination Development Protocol

Date: 2026-09-26

Protocol: `pointmaze_learned_termination_stage10_v1_development`

Algorithm revision: `efaba47e42e40684f9d11db8be951ea0932049d8`

## Question

Does the confirmed Stage-9 causal plan-validity trigger outperform a learned
termination actor trained directly on the external tracking objective, under
the same upper-call and approximately matched extra-transition budgets?
This is a **development** comparison against already revealed Stage-9
confirmation paths, not a second independent confirmation.

## Frozen Design

Reproduce the same goal-conditioned upper/lower PPO controller at Stage-9
confirmation roots `209011, 209023, 209037, 209049, 209061, 209073,
209089, 209101`. Fixed-schedule replay must exactly match the stored Stage-9
episode rows before pairing outcomes. Trigger fitting uses the same eight
branch-fit seeds and controller-selection seeds; the 16 Stage-9 trigger-eval
seeds remain held out from trigger training and selection.

The baseline is a 64-hidden-unit Bernoulli actor with a separate value head.
It receives all 37 causal plan features plus the within-bin offset, with no
regime label or future input. The upper/lower policies stay frozen. At offsets
0, 5, 10, 15, and 20, the actor may terminate the current option and call
the planner; offset 25 is a forced deadline. Every mode makes exactly one
upper call in every 50-step bin (24 calls per episode). The trigger receives
only negative external tracking ISE, not lower intrinsic reward.

One warm-up rollout per branch-fit path fixes feature normalization. Eighteen
PPO iterations each use the same eight branch-fit paths; checkpoints at
iterations 4, 8, 12, 16, and 18 are selected by mean ISE on the eight
controller-selection paths. This is 192 additional 1200-step rollouts, or
230,400 primitive steps per root including warm-up and selection, below the
stored Stage-9 branch-fit replay count for every root. Controller training
and evaluation are otherwise identical. No checkpoint artifact is retained.

The primary comparison is paired learned-termination minus Stage-9 candidate
episode tracking ISE. The development gate requires two root-level 95%
Student-t interval lower bounds above zero: the learned termination arm beats
fixed planning, **and** the Stage-9 candidate beats learned termination.
Return and trigger activation are reported, not substituted for the primary
endpoint. The root is the statistical unit; no root extension or post-hoc
hyperparameter tuning is permitted. A pass would motivate fresh-seed
confirmation of this baseline comparison, not a cross-domain claim.

Preflight root `208001` uses the previously frozen Stage-9 software path,
two controller and two termination iterations, a 300-step horizon, and only
checks replay, learning update, budget, analyzer, and compact result handling.
Each cell uses one CPU, 1536 MB, and dynamic placement on node001-node006;
only `result.json` is synchronized.
