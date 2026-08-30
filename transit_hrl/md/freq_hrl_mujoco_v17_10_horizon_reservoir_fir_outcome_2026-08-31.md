# MuJoCo v17.10 Horizon-Reservoir FIR Outcome

## Decision

Status: `horizon_reservoir_fir_stopped_before_fresh_path_access`.

The frozen selection returned to the zero-reserve v17.9 behavior: 120/120 valid
and upper-feasible paths, 48/81 recovered failures, and 32/32 preserved
baseline-feasible Walker2d paths. The strict recovery gate failed, so no fresh
path was accessed.

## Reservoir Diagnostic

The 82-step reservoir was the strongest recovery diagnostic. Its W48,
ridge-`1e-3` candidate recovered 63/81 failures and preserved 32/32 Walker2d
paths while all 120 endpoints met the upper budget. Recovery was 40
HalfCheetah, 15 Hopper, and 8 Walker2d. It was not eligible: only 113/120 paths
maintained the causal envelope throughout, and Hopper remained below its 24/33
gate.

The full reservoir makes all future credit immediately available. Some paths
spend that credit early and later encounter a physical component box that
cannot return to the envelope. A final router-only test may release only a
fraction of remaining credit and repay it linearly. Failure there ends this
filter family; further progress would require actor-level total-action change.

## Claim Boundary

This is reused-path grouped development evidence with fixed total action. It
does not establish fresh-seed generalization, reward improvement, closed-loop
learning, leakage no-tradeoff, or manuscript support.
