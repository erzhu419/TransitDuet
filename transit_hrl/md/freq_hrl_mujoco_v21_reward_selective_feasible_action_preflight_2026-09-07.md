# MuJoCo v21 Reward-Selective Feasible-Action Preflight

## Decision

V20 is closed as a negative development result. Its projected raw-gradient
guard executed correctly but changed the consistency loss only at approximately
1e-6 scale, while uniform scalarized consistency reduced correction and improved
projected reward in two environments. V21 therefore changes the learning
objective rather than screening another coefficient or guard step size.

V21 is a development preflight. It cannot supply manuscript, confirmatory,
generalization, superiority, or no-tradeoff evidence.

## Mechanism

For transition `i`, let `a_i^cert` be the detached inverse-tanh action
corresponding to the component executed by the terminal-reserve certificate,
`mu_theta(s_i)` the Gaussian actor mean, and `A_i` the normalized SMDP GAE
already used by PPO. The uniform v20 consistency loss is

```text
L_uniform = alpha mean_i ||mu_theta(s_i) - a_i^cert||^2.
```

The v21 reward-selective weight is

```text
q_i = exp(clip(A_i / T, -log C, log C))
w_i = q_i / mean_j q_j
L_selective = alpha mean_i w_i ||mu_theta(s_i) - a_i^cert||^2.
```

The frozen preflight uses `T=1`, `C=5`, and `alpha=0.10`. Advantages and
weights are detached. Unit-mean normalization keeps the minibatch-average
regularization scale matched to the uniform arm, so the comparison isolates
reward selectivity instead of silently increasing the effective coefficient.

The terminal-reserve controller remains the runtime authority. The environment
executes certified components and their sum; the weighting changes only actor
learning. It does not relax a prefix budget or replace the recursive-feasibility
certificate.

## Arms

All four arms have identical terminal-reserve state capacity, hidden size,
training paths, checkpoint window, and heldout paths.

1. `raw_context_v21_preflight`: capacity-matched context, no projection.
2. `terminal_reserve_consistency_000`: projected execution, no consistency.
3. `terminal_reserve_delayed_uniform_010`: projected execution and delayed
   uniform consistency.
4. `terminal_reserve_delayed_reward_selective_010`: the same projected
   execution, coefficient, and schedule with unit-mean reward-selective weights.

Consistency is zero for the first half of 512 iterations, ramps linearly over
the next quarter, and remains at 0.10 for the final quarter. Checkpoints before
iteration 383 are ineligible.

## Fresh-root panel

- Environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5.
- Optimizer roots: 4.
- Train roots: 4.
- Crossed checkpoint-selection roots: 4.
- Heldout roots: 8, each crossed with standard, low-frequency,
  high-frequency, mixed, and OOD-chirp conditions.
- Total cells: 48.
- Per task: one CPU core, 1536 MiB RAM.
- Scheduler: scheduleurm on node001-node006 with no fixed node.
- Server-only artifacts: checkpoint and full training history.
- Synchronized artifacts: cell summary, evaluation rows, and server artifact
  location.

The 20 role roots were generated once from NumPy generator seed `210031` and
checked against every integer literal in earlier MuJoCo scripts before freeze.

## Advancement gates

V21 advances only if every gate passes:

- every projected cell passes certificate, prefix-power, convergence, and
  fallback validity;
- reward-selective reward exceeds uniform in at least two environments;
- the optimizer-root pooled normalized reward delta is positive;
- no environment loses more than 5% reward relative to uniform;
- component and total correction each improve by at least 5% over reserve in
  at least two environments;
- neither correction metric regresses more than 5% from uniform in any
  environment;
- candidate physical burden remains below the frozen correction and
  action-change limits;
- both levels report active weighted updates in every candidate cell, unit mean
  weights, nontrivial maximum weights, and finite positive raw and weighted MSE.

Bootstrap intervals are descriptive because four optimizer roots are a
mechanism preflight, not a statistical confirmation. The advancement decision
uses the frozen directional and bounded-regression gates above.

## Stopping rule

Any failed gate stops this weighting mechanism. These roots cannot be reused to
tune temperature, clip, coefficient, schedule, or thresholds. A pass authorizes
a separately committed multi-seed development protocol, not confirmation.

## Frozen source

The algorithm revision is
`f215434f404c4adee1203e206b0a414d21369994`. Scheduler scripts, analysis, and
tests may be committed afterward, but the `freq_hrl/` source tree must remain
identical to that revision for dispatch.
