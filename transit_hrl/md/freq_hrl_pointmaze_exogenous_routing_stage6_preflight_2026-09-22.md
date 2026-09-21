# PointMaze Exogenous Frequency-Routing Stage-6 Preflight

Date: 2026-09-22

Run: `pointmaze_exogenous_routing_stage6_v1_preflight_20260922_r1`

Tasks `t100042`-`t100049` completed without failure on node003-node006,
with two cells on each node. The eight methods used reserved optimizer root
`184001`, which is excluded from the formal matrix.

The audit passed all method identities, paired role seeds, identical external
paths, runtime consistency, 134-dimensional state shapes, equal capacity and
initialization within each architecture, causal observability fields, finite
PPO updates, upper/lower decision boundaries, disabled legacy mechanisms, and
the result-only artifact contract. Flat models have 67,973 trainable parameters
and hierarchical models have 68,424, matching the Stage-5 capacity contract.

The one-root, two-iteration analysis is intentionally `not_supported`: a
single optimizer root has an unbounded Student-t interval and cannot satisfy
the preregistered performance gate. This run is software evidence only. It
authorizes the frozen 64-cell Stage-6 development matrix and supplies no
performance claim.
