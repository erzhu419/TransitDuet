# MuJoCo v23 Causal Upper-Target Development

## Decision

V22 showed a large correction reduction but failed the joint reward, solver
validity, and Hopper burden gates. Post-result diagnosis found a separate
training-label problem: one upper action was trained against the average of all
projected lower-step targets in its macro interval. Later targets depend on
later lower proposals and states, so that average is neither the target
available when the upper action was chosen nor a single causal upper decision.

V23 tests the algorithm revision that assigns the upper action only the first
same-observation projected target. The old `macro_mean` path remains unchanged
as a matched control. This is a development screen, not a retry of v22 and not
confirmatory evidence.

## Frozen Mechanism

The algorithm revision is
`72643f24204543a3655dde395717bfdbb60ffd2a`. It adds the explicit modes:

- `macro_mean`: average projected targets over the macro, preserving v22;
- `decision_time`: use the first projected target associated with the upper
  action's decision observation.

No coefficient, budget, projector threshold, checkpoint rule, or training
horizon changes between the matched full-consistency arms.

## Arms And Matrix

1. `terminal_reserve_consistency_000`: projected zero-consistency baseline.
2. `terminal_reserve_macro_mean_uniform_010`: unchanged v22 label control.
3. `terminal_reserve_lower_only_decision_time_010`: lower consistency only,
   isolating interference from the upper consistency objective.
4. `terminal_reserve_decision_time_uniform_010`: causal upper-target candidate.

The matrix has 3 environments, 4 arms, and 4 fresh optimizer roots: 48 cells.
Each cell uses four fresh train roots, four fresh checkpoint-selection roots,
and eight fresh heldout roots crossed with standard, low-frequency,
high-frequency, mixed, and OOD-chirp conditions. All 20 roots were generated
once from NumPy seed `230091`, checked against earlier MuJoCo script literals,
and frozen before execution.

The shared settings remain 512 iterations, upper period 16, hidden width 64,
PPO clip 0.10, delayed-linear consistency after 50% warmup and 25% ramp,
checkpoint eligibility from iteration 383, terminal windows 8/32, upper HF RMS
budget 0.075, and lower LF RMS budget 0.0475.

## Frozen Advancement Rule

Every arm must pass certificate and prefix checks, projection convergence at
least 0.95, and recursive fallback at most 0.05. Consistency diagnostics must
show exactly the intended active levels.

An advancing candidate must also:

- remain within -5% reward of both zero and old controls in every environment;
- beat the old control on at least 8/12 optimizer-root pairs and at least 2/4
  roots in every environment;
- improve mean reward over the old control in at least two environments;
- reduce both component and total correction by at least 5% versus zero in at
  least two environments;
- regress neither correction metric by more than 5% versus zero or old control
  in any environment;
- keep Hopper mean total correction RMS at or below 0.25.

The full `decision_time` arm has fixed selection priority. The lower-only arm
can advance only if the full arm fails and the lower-only arm independently
passes every gate. No result-dependent threshold or parameter changes are
allowed.

## Execution Contract

- Scheduler: scheduleurm only.
- Nodes: `node001` through `node006`, with no required node.
- Per cell: one CPU core and 1536 MiB RAM.
- Synchronized: `cell_summary.json`, `evaluation_rows.csv`, and
  `server_artifact_location.json`.
- Server only: checkpoint and full training history.

## Claim Boundary

Four optimizer roots can authorize a fresh-root confirmation but cannot support
a manuscript, superiority, no-tradeoff, cross-domain, or deployment claim. A
failed gate retires these roots for this mechanism and parameterization.
