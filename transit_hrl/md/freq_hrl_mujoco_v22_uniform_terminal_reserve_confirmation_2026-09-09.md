# MuJoCo v22 Uniform Terminal-Reserve Confirmation

## Decision

V21 rejected positive reward-advantage weighting. Its unchanged uniform control
nonetheless improved four-root mean reward and reduced terminal-reserve
correction in all three environments. V22 is a fresh-root confirmation of that
already fixed uniform mechanism; it does not change its coefficient, schedule,
training horizon, checkpoint window, actor architecture, or projector.

The registered primary question is whether uniform certified-action consistency
reduces certificate intervention magnitude without degrading projected-policy
reward. Reward superiority is secondary and cannot be inferred from merely
passing the noninferiority gate.

## Frozen Mechanism

The candidate is exactly the v21 uniform control:

```text
upper and lower consistency coefficient = 0.10
weighting = uniform
training iterations = 512
warmup = first 50%
linear ramp = next 25%
eligible checkpoint window = iterations 383 through 511
terminal reserve windows = upper 8, lower 32
upper HF RMS budget = 0.075
lower LF RMS budget = 0.0475
```

The frozen algorithm revision is
`f215434f404c4adee1203e206b0a414d21369994`. No `freq_hrl/` source change is
authorized for this run.

## Arms And Matrix

1. `raw_context_v22_confirmation`: capacity-matched, unprojected reference.
2. `terminal_reserve_consistency_000`: projected execution without actor
   consistency.
3. `terminal_reserve_delayed_uniform_010`: unchanged v21 uniform candidate.

The matrix contains 3 environments, 3 arms, and 32 fresh optimizer roots, for
288 independent cells. Every cell uses the same four fresh training roots, four
fresh crossed checkpoint-selection roots, and eight fresh heldout roots crossed
with standard, low-frequency, high-frequency, mixed, and OOD-chirp conditions.
All 48 role roots were generated once from NumPy seed `220091`, checked against
earlier MuJoCo script literals, and frozen before execution.

## Registered Statistics

Environment effects use paired bootstrap ratio-of-means estimators across the
32 optimizer roots. This avoids the v21 mean-of-root-ratios instability in which
one small baseline reward could reverse the sign of the environment summary.
Environment-specific lower bounds use a Bonferroni family confidence of
98.33%; pooled root-paired intervals use 95% confidence.

The joint confirmation requires all of the following:

- both projected arms pass every certificate, prefix-power, convergence, and
  recursive-fallback check;
- the candidate reward lower bound is above a -5% noninferiority margin in
  every environment and in the pooled paired analysis;
- pooled component and total correction lower bounds both exceed 5% reduction;
- component and total correction point estimates each improve by at least 5%
  in at least two environments;
- no environment-specific correction lower bound is below -5%;
- candidate mean total correction RMS is at most 0.25 in every environment;
- uniform consistency updates are active and report exactly unit weights, while
  the zero-consistency baseline reports no active consistency updates.

Reward-superiority intervals and root wins are reported as secondary outcomes.
They do not replace the primary correction-plus-noninferiority gate.

## Execution Contract

- Scheduler: scheduleurm only.
- Nodes: `node001` through `node006`, no fixed-node binding.
- Per cell: one CPU core and 1536 MiB RAM.
- Synchronized: `cell_summary.json`, `evaluation_rows.csv`, and
  `server_artifact_location.json`.
- Server only: checkpoint and full training history.

Any failed primary gate rejects the joint confirmation. These roots cannot be
reused to tune the consistency coefficient, schedule, thresholds, or
checkpoint rule.

## Claim Boundary

A supported result establishes fresh-root MuJoCo evidence that the unchanged
uniform mechanism reduces terminal-reserve correction while preserving
projected reward. It does not by itself establish reward superiority,
cross-domain superiority, universal frequency separation, or deployment
validity.
