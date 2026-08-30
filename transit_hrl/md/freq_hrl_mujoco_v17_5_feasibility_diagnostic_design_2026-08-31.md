# MuJoCo v17.5 Feasibility Diagnostic Design

## Question

The v17.4 streaming projection improved lower-frequency drift and joint merit,
but HalfCheetah and Hopper still exceeded the fixed lower budget. The next
algorithmic decision depends on whether this residual is avoidable by a better
projection or is the minimum physical violation induced by the frozen learned
total action and component bounds.

## Diagnostic

The diagnostic reuses the rejected v17.4 checkpoints and the same 40 accessed
evaluation paths per environment. It is development-only and cannot contribute
confirmatory evidence. For each checkpoint it runs strength-one routing under:

1. `causal_streaming_audit_projection` from v17.4;
2. `causal_feasibility_normalized_audit_projection` from v17.5.

Before interpretation, the refactored v17.4 implementation must reproduce every
legacy reward, executed-action, and latent-policy trace hash, with the selected
numeric metrics agreeing within `1e-12`. A mismatch invalidates the diagnostic.

The v17.5 router reports the physical action interval, upper HPF8 budget
interval, lower LPF32 budget interval, minimum unavoidable violation, and excess
regret over that minimum. The decision rule is fixed before replay:

- continue projection work only if v17.4 has maximum excess regret above
  `1e-7` and v17.5 removes at least 75% of it;
- move to learned-policy feasibility constraints if excess regret is at most
  `1e-7` but the unavoidable floor exceeds `1e-7`;
- stop this line if the legacy replay is not exact.

## Execution And Artifacts

The three source checkpoints exist only on `node003`. The scheduleurm tasks are
therefore hard-bound to that node for data locality; this is not a general
placement policy. Source staging excludes `.server_artifacts`. Only
`diagnostic_summary.json` and `diagnostic_rows.csv` are synchronized locally;
checkpoints and training histories remain server-only.

## Claim Boundary

This replay chooses the next implementation target. It is not a fresh test, a
performance validation, or publication evidence. Any learned-policy change that
follows must be frozen and evaluated on fresh optimizer and environment seeds.
