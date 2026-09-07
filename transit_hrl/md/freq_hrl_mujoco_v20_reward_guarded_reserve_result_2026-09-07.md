# MuJoCo v20 Reward-Guarded Reserve Development Result

## Status

The frozen v20 panel is complete and **not supported** for candidate
selection. All 144 registered cells finished with unique scheduler signatures.
No candidate was selected, and every v20 optimizer, rollout, selection, and
evaluation root is retired.

This is a development result. It is not confirmatory or manuscript evidence.

## Execution audit

- Scheduler records: 144 archived tasks, all `done`, with no duplicate
  signature.
- Node distribution: node001 24, node002 25, node003 25, node004 22,
  node005 24, and node006 24.
- Local synchronization used the frozen small-result path for all 144 cells.
  The synchronized directory was 19 MiB and contained no file larger than
  1 MiB. Full training histories and checkpoints remained server-side.

## Frozen panel

- Environments: HalfCheetah-v5, Hopper-v5, and Walker2d-v5.
- Arms: capacity-matched unprojected context, projected terminal reserve
  without consistency, delayed scalarized consistency 0.10, and delayed
  reward-guarded consistency 0.10.
- Replicates: 12 fresh optimizer roots per environment and arm.
- Training: 384 PPO iterations and 512 transitions per training path.
- Consistency schedule: zero for the first half, linear ramp over the next
  quarter, and full coefficient for the final quarter.
- Heldout evaluation: eight seeds crossed with five disturbance conditions.
- Checkpoint selection started only after the consistency ramp reached its
  target. Heldout terminal metrics did not select checkpoints.

## What held

The zero-consistency terminal-reserve baseline passed every registered
mechanism validity gate:

- zero heldout certificate violations;
- upper and lower realized prefix powers within their frozen budgets;
- recursive fallback below 0.05;
- minimum cell-level projection convergence above 0.95.

The reward guard also executed as implemented. Across the guarded candidate,
the lower and upper consistency steps had acceptance rates 0.9573 and 0.9626.
Their maximum same-minibatch reward-surrogate deltas were negative, active
native-constraint deltas were zero, and mean consistency-loss deltas were
-1.62e-6 and -1.85e-6, respectively.

These facts validate the certificate and local guard behavior. They do not
establish heldout reward preservation.

## What failed

Neither candidate passed the complete preregistered gate.

| Candidate | Worst environment component reduction | Pooled total reduction | Component support | Result |
|---|---:|---:|---:|---|
| delayed scalarized 0.10 | 19.36% | 33.97% | 2/3 | reward, burden, and validity gates failed |
| delayed reward-guarded 0.10 | -10.13% | 3.07% | 0/3 | correction, reward, burden, and validity gates failed |

The scalarized arm reduced component correction in all three environment point
estimates, but HalfCheetah's confidence interval crossed zero. Hopper remained
the binding physical burden: mean total correction RMS was 0.2548 and mean
action-change rate was 0.5614, above the registered 0.25 and 0.50 limits.

The guarded arm did not produce a useful actor adaptation. Although almost all
guarded steps were accepted, each accepted raw-gradient step changed the
consistency objective only at approximately 1e-6 scale. Its component
correction point estimate worsened in HalfCheetah and Walker2d, and no
environment had CI-supported component reduction.

Both candidate arms also narrowly missed the 0.95 minimum cell-level numerical
projection-convergence gate: 0.9474 for scalarized and 0.9489 for guarded.
Both retained zero certificate violations, so this is a solver convergence
failure rather than a counterexample to the terminal-reserve invariant.

Mean heldout reward exposed the larger physical tradeoff:

| Environment | Raw context | Reserve 0.00 | Scalarized 0.10 | Guarded 0.10 |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 2175.31 | 1494.42 | 1566.20 | 1538.74 |
| Hopper-v5 | 248.60 | 161.03 | 158.03 | 155.61 |
| Walker2d-v5 | 298.30 | 167.20 | 203.75 | 162.69 |

Scalarized consistency improved projected reward in HalfCheetah and Walker2d,
but no candidate met the per-environment reward noninferiority rule against
both the projected reserve and unprojected raw references.

## Decision

V20 rejects the detached actor-to-projector consistency objective with a
same-minibatch raw-gradient correction as the solution to the terminal-reserve
reward tradeoff. Another coefficient or schedule screen using these roots is
not authorized.

The terminal-reserve certificate remains a valid runtime mechanism. The next
algorithmic experiment must change how feasible actions enter policy learning,
not merely change the consistency coefficient. A justified next mechanism is
reward-selective feasible-action learning: train the actor toward certified
actions in proportion to their within-rollout reward advantage, retain exact
projected execution, and compare it against both uniform consistency and the
capacity-matched raw policy on fresh roots. It requires a new frozen
development protocol before any new heldout result is read.
