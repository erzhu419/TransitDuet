# MuJoCo v24 Policy-Mean Upper-Target Development

## Decision

V23 removed future-state contamination from the upper projection target, but
its first same-state target depended on one stochastic lower-action sample.
Upper consistency MSE increased in every environment and the candidate failed
the reward/correction gate. V24 tests one new estimator: project the lower
policy distribution mean from the same actor forward pass and the same causal
projector state.

This is a fresh-root development screen. It is not a retry on v23 roots and is
not confirmatory evidence.

## Frozen Mechanism

Revision `8e3b571185a84d0adb00307421a89c0f38a81412` adds
`decision_policy_mean`. At each upper decision it:

1. obtains the sampled lower action and distribution mean in one actor pass;
2. previews terminal-reserve projection with the mean without advancing any
   projector history, energy, or step counter;
3. uses the previewed upper component as the upper consistency target;
4. executes the independently sampled lower action through the unchanged
   projector.

The projector, budgets, policy capacity, lower target, objective coefficients,
schedule, and checkpoint rule remain unchanged. The old `macro_mean` arm is a
non-adoptable hindsight diagnostic because it uses later states in a macro.

## Arms And Matrix

1. `terminal_reserve_consistency_000`: projected zero-consistency baseline.
2. `terminal_reserve_macro_mean_uniform_010`: hindsight diagnostic.
3. `terminal_reserve_decision_time_uniform_010`: v23 stochastic causal control.
4. `terminal_reserve_policy_mean_uniform_010`: v24 deterministic causal
   candidate.

The matrix has 3 environments, 4 arms, and 4 fresh optimizer roots: 48 cells.
Each cell uses four fresh train roots, four fresh checkpoint-selection roots,
and eight fresh heldout roots crossed with standard, low-frequency,
high-frequency, mixed, and OOD-chirp conditions. All 20 roots were generated
once from NumPy seed `240091`, checked against earlier MuJoCo script literals,
and frozen before execution.

Shared settings remain 512 iterations, upper period 16, hidden width 64, PPO
clip 0.10, delayed-linear consistency after 50% warmup and 25% ramp, checkpoint
eligibility from iteration 383, terminal windows 8/32, upper HF RMS budget
0.075, and lower LF RMS budget 0.0475.

## Frozen Advancement Rule

Every arm must have zero certificate violations, recursive fallback at most
0.05, and realized prefix power within both budgets. Dykstra's step-tolerance
convergence rate is reported as a numerical diagnostic, not substituted for the
separate certificate feasibility check.

The policy-mean candidate must also:

- activate policy-mean targets only in its own arm and record positive
  sampled-versus-mean target delta during training;
- reduce upper consistency MSE versus the first-sample control by at least 5%
  in two environments, with no environment regressing more than 5%;
- remain within -5% reward of zero, first-sample, and hindsight controls in
  every environment;
- beat first-sample reward on at least 8/12 paired roots, at least 2/4 per
  environment, and improve mean reward in at least two environments;
- reduce component and total correction versus zero by at least 5% in two
  environments, with neither metric regressing more than 5% versus any control;
- keep Hopper mean total correction RMS at or below 0.25.

The candidate advances only if every gate passes. No result-dependent
coefficient, threshold, budget, seed, or checkpoint change is permitted.

## Execution Contract

- Scheduler: scheduleurm only.
- Nodes: `node001` through `node006`, with no required node.
- Per cell: one CPU core and 1536 MiB RAM.
- Synchronized: `cell_summary.json`, `evaluation_rows.csv`, and
  `server_artifact_location.json`.
- Server only: checkpoint and full training history.

## Claim Boundary

A pass authorizes fresh-root confirmation only. Four optimizer roots cannot
support manuscript superiority, no-tradeoff, cross-domain, or deployment
claims. A failed gate retires these roots and this parameterization.
