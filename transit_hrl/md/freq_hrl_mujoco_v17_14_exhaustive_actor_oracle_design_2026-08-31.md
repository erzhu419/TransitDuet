# MuJoCo v17.14 Exhaustive Actor Oracle Design

## Motivation

V17.13 evaluated 48 globally prefiltered candidates with exact responsibility
oracles. Those candidates all used gain 0.5 or 1.0 and recovered at most 3/7
actor-floor paths. The result blocks fresh validation but does not close the
900-member linear FIR design because 852 candidates lacked exact outcomes.

## Exhaustive Remainder

V17.14 introduces no new hyperparameter. It validates the registered v17.13
summary, subtracts its 48 exact-oracle candidate identifiers from the original
900-grid, and evaluates every one of the 852 remaining candidates. The new
outcomes are then merged with the 48 prior outcomes into one exact 900-candidate
frontier using the unchanged v17.13 selection order and advancement gate.

Each candidate remains an eight-fold leave-one-seed-out causal adapter. Cached
path-level sufficient statistics preserve the exact weighted-ridge solution.
Full-horizon responsibility oracles run in a bounded process pool with one
numerical thread per worker. Progress is emitted every 25 candidates so the
scheduler can distinguish active computation from a stalled process.

## Decision Rule

If any candidate satisfies all registered recovery, preservation, target,
trust-region, and post-clipping action gates, the best candidate may advance to
a separately frozen closed-loop fresh-seed validation. If no candidate passes,
the frozen 900-member linear causal FIR adapter grid is closed and the next
mechanism must be a state-conditioned learned actor policy.

## Claim Boundary

This remains reused-path development. Exhaustive evaluation of the frozen grid
does not establish reward improvement, online learning, fresh-seed
generalization, leakage no-tradeoff, or manuscript support.
