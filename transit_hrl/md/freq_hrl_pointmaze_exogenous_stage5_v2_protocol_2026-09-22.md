# PointMaze Exogenous-Control Stage-5 V2 Protocol

Date: 2026-09-22

## Purpose

Stage-5 V1 learned relative to its untrained policy but failed the registered
absolute-success gate because optimizer stability was insufficient. A bounded
development screen selected eight training rollout roots; checkpoint-rank
changes had no effect. V2 independently tests that repair.

## Frozen Design

The runtime algorithm remains revision
`a4a64730a9542e690650d3a39856a5324ec51fcc` and runtime protocol
`pointmaze_exogenous_control_stage5_v1`. The evidence protocol is
`pointmaze_exogenous_control_stage5_v2_confirmation`.

V2 uses eight new optimizer roots. Each root has eight new training paths,
16 new checkpoint-selection paths, and 16 new held-out evaluation paths.
Neither optimizer roots nor environment paths overlap V1, the stability
screen, or the separate preflight. Flat PPO and HRL receive the same eight-path
training interaction budget.

The task, 300-step horizon, external target and force process, 134-dimensional
states, matched parameter budget, reward, PPO implementation, 768 iterations,
96-iteration checkpoint interval, and success-then-return rank remain fixed.
Frequency routing, promotion, leakage losses, and legacy projectors remain
disabled.

## Registered Gate

Using optimizer-root means as the statistical unit, all three conditions must
hold:

- the lower endpoint of the HRL tracking-success 95% t interval is at least
  0.50;
- HRL final-minus-untrained tracking success has a positive 95% interval; and
- HRL final-minus-untrained episode return has a positive 95% interval.

HRL-versus-flat effects are reported but do not gate substrate admission.
Failure of any conjunct blocks frequency-routing experiments; roots cannot be
added post hoc.

## Execution

A separate optimizer root and separate role seeds are reserved for a two-cell
software preflight. The formal matrix contains 16 single-core cells and may
run dynamically on node001 through node006. Only `result.json` is synchronized;
checkpoints and rollout CSV files are not retained.

## Claim Boundary

Preflight can authorize execution only. A passing formal V2 would establish
that ordinary HRL reliably learns the separate-exogenous substrate and would
admit a subsequent fresh frequency-routing attribution experiment. It would
not by itself prove hierarchy superiority, selective frequency assignment, or
domain-general Freq-HRL.

The two-cell preflight completed and passed all registered execution checks.
See `freq_hrl_pointmaze_exogenous_stage5_v2_preflight_2026-09-22.md`.
