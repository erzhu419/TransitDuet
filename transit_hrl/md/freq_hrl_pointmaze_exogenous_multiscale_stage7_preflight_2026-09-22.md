# PointMaze Exogenous Multiscale Stage-7 Preflight

Date: 2026-09-22

Run: `pointmaze_exogenous_multiscale_stage7_v1_preflight_20260922_r1`

Tasks `t100120`-`t100123` completed without failure on node003-node006, one
cell per node. The four factorial methods used reserved optimizer root `194001`,
which is excluded from confirmation.

The audit passed method identity, exact seed pairing, 134-dimensional state
shapes, architecture-matched capacity and initialization, causal external-path
visibility, finite PPO updates, fixed-horizon rows, runtime identity, and the
result-only artifact contract. The Stage-7 analyzer accepted the complete
factorial and rejected no protocol field.

The one-root analysis is necessarily `not_supported` because its Student-t
intervals are unbounded. This preflight supplies software evidence only and
authorizes the fixed 64-cell, 16-root confirmation. It does not contribute a
performance observation or permit sequential root extension.
