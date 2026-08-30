# MuJoCo v14.28 domain-general restoration portfolio preflight

## Decision context

The v14 series has isolated two complementary mechanisms. Function-preserving
router adaptation was stable in HalfCheetah and Hopper but reversed direction
in Walker. Orthogonal actor-bias restoration produced full-rank, cross-role
stable directions in every environment and independently improved Hopper and
Walker, but its HalfCheetah design selection did not transfer. A single fixed
mechanism is therefore the wrong domain-general abstraction.

## Frozen portfolio

Every environment receives the same candidate registry and the same selector.
There is no environment-name branch.

- Five orthogonal paired-FD actor updates use output-bias RMS steps `1e-7`,
  `1e-6`, `1e-5`, `3e-5`, and `1e-4` at the baseline router strength `0.5`.
- Ten function-preserving router candidates use strengths `0.0` through `1.0`
  at `0.1` resolution, excluding the baseline `0.5`; actor parameters remain
  frozen for these candidates.
- The pooled selector first minimizes reward violations, then frequency merit,
  worst frequency violation, actor step, and router strength among eligible
  candidates.

The orthogonal direction retains the v14.27 protocol: order-eight Hadamard
output-bias interventions at RMS `0.25`, two independent roots per row and mode,
upper eight-decision and lower 32-decision native cost targets, full-rank least
squares, positive critic R2/action relevance, and positive overall/every-mode
train-versus-holdout direction cosine.

## Two-fold selection

Thirty-two fresh design roots are crossed with four disturbance modes. The
first and second 16-root blocks are frozen design folds, each containing 64
paths. A candidate must independently satisfy the unchanged reward floor,
minimum frequency-merit reduction, and worst-violation funnel in both folds and
in the pooled 128-path design. This specifically prevents a candidate like the
v14.27 HalfCheetah actor update from advancing on one favorable aggregate.

Only one pooled-best eligible candidate enters 32-root, 128-path untouched
validation. The same eligibility rule is reapplied unchanged.

## Independent roles

Critic train and holdout each use 16 fresh roots. Design and validation each use
32 further roots. All 96 role roots are mutually disjoint and absent from
v14.20-v14.27. The portfolio therefore contains 320 train intervention paths,
320 holdout intervention paths, 128 baseline design paths, 1,920 candidate
design paths, and at most 256 validation paths per environment.

## Execution contract

The three cells are HalfCheetah, Hopper, and Walker at optimizer seed
`4196455150`. Each requests 24 CPU cores and 16 GB RAM. Scheduler placement is
dynamic across `node001-node006`, with `require_node=None`; no Slurm or
login-node execution is permitted.

## Evidence boundary

This is an adaptive mechanism preflight, not confirmatory evidence. It advances
only if all three independent validation cells pass. Success would support the
portfolio selector for integration into the shared Freq-HRL restoration core;
it would not by itself constitute a paper-level multi-seed efficacy claim.
