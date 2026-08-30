# Freq-HRL MuJoCo v14.19 Bidirectional Router Adapter

## Motivation

The frozen v14.18 screen rejected a single larger router strength. Strengths
above `0.5` improved all six HalfCheetah/Hopper cells but worsened all three
Walker2d cells. All candidates preserved the reward guard, so the remaining
question is whether a deployment-aligned selector can adapt routing direction
without changing executed behavior.

This design was written after all v14.18 outcomes were inspected. It is an
adaptive mechanism-development screen and cannot be used as confirmatory paper
evidence.

## Frozen screen

- Anchors: the same nine completed v14.17 anchor checkpoints.
- Router strengths: `0.0, 0.1, ..., 1.0`.
- Actor gains: fixed at `1.0`.
- Comparator: router strength `0.5`.
- Guard: the frozen v14.17 16-path, four-mode, five-endpoint mode-CVaR guard.
- Scheduler: nine independent one-core `scheduleurm` tasks, dynamically placed
  over `node001-node006`, with no required node and no Slurm.

## Adapter rule

Every cell uses the same grid and the same selector. A candidate is eligible
only when it has zero reward violations and strictly lower frequency-violation
merit than strength `0.5`. Eligible candidates are ordered by:

1. lower frequency-violation merit;
2. lower worst frequency violation;
3. fewer frequency violations;
4. smaller distance from `0.5`; and
5. smaller strength.

Environment identity and optimizer seed are not selector inputs. There is no
manual environment-specific coefficient.

## Mechanism gate

The bidirectional adapter is supported only if the identical rule selects an
eligible candidate in all nine cells. A failure in any cell rejects router-only
adaptation. Passing permits implementation in the training core, followed by a
fresh-seed continuation screen; it does not itself establish a performance or
statistical claim.
