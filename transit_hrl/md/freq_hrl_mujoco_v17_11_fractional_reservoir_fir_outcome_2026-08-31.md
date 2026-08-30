# MuJoCo v17.11 Fractional-Reservoir FIR Outcome

## Decision

Status: `fractional_reservoir_fir_stops_router_only_development`.

The frozen selector chose the 80-step, 0.75-borrow, W64 candidate. It kept all
120 paths numerically and physically valid, met the endpoint upper budget on
all 120, improved mean lower power in every environment, and preserved all 32
baseline-feasible Walker2d paths. It recovered 62/81 oracle-recoverable
failures: 40/40 HalfCheetah, 14/33 Hopper, and 8/8 Walker2d. The pre-registered
65/81 total and 24/33 Hopper gates therefore failed, and no fresh path was
accessed.

## Router Boundary

The failure is not an artifact of the selection ordering. Among the 20
candidates that were valid and upper-feasible on all 120 paths, 62/81 was the
largest recovery count and 14/33 was the largest Hopper count. The best Hopper
diagnostic reached only 16/33 while already losing envelope feasibility on one
path. No candidate in the 40-member frozen panel passed the complete gate.

V17.6 established that a full-horizon noncausal split exists for 33 Hopper
failures, while v17.9-v17.11 show that post-processing a fixed total-action
trajectory cannot recover enough of them under causal physical and frequency
constraints. Continuing to tune the router would reuse the same rejected paths
without addressing the limiting variable. Router-only development is closed;
the next mechanism must change the actor's total action under endpoint
frequency constraints.

## Claim Boundary

This is grouped reused-path development evidence with fixed total action. It
does not establish fresh-seed generalization, reward improvement, closed-loop
learning, leakage no-tradeoff, or manuscript support.
