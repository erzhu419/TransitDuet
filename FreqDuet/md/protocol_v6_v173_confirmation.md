# V173: frozen V172 independent confirmation

Source `0151b7c719`; map `a5b45df520`, protocol `f5d152f0bc`. Preparation
`t210203` passed 25 server tests. Earlier `t210196` prepared only the superseded
reporting entry; no collection tasks use its source `7914534858`.

- Exactly days3152001-3152032 and samplers1711000000-1711000031, shared across all 12 actors and old/handoff modes: 384 pairs/768 paths. No replacements or outcome-dependent additions.
- Active collection: `t210266` plus `t210268`-`t210314`: 48 shards, eight actor/day pairs each, two workers/CPU cores, explicit high priority and manual dispatch restricted to node001-node006.
- Aggregate `t210315` is also high/manual. Its 48 local wait files are tiny completion receipts automatically synced from each shard, not remote paths mistaken for local dependencies. It cannot aggregate a partial matrix.
- Preserve the audited raw query schema and capture pipeline; interval lookup adds hour2/hour13/other from query time alone. Predictions are saved before labels. Low/unseen/native-unsupported requests remain abstentions and stay in the denominator.
- Keep V172, V167 coarse/conditional maps, V151, V146, the 3600s window, tails0.1/0.1 and controller/actor/critic/dual unchanged. No fitting, map updates or policy switch.
- Apply all V171 gates in both modes/all six scopes. Coverage lower95>=0.8, each tail upper95<=0.1, finite fraction lower95>=0.8, paired score upper95<=0; strict score improvement in all/clearance. Missing finite labels/comparators remain inconclusive.
- Query-to-episode-to-actor/day reduction; crossed bootstrap5000/seed20260803 shares resampled demand days across actors. Store full actor/day marginals remotely; report coarse-primary and V167-conditional secondary comparisons on the candidate-finite mask.
- Also report in-sample calibration versus fresh-confirmation hour13 moving-launch tails, support and scores. Use exact saved calibration ETA labels, shared resampled actors and independent cohort-day axes; migration is descriptive, not an extra admission gate.

Launch repair: initial tasks `t210204`-`t210252` were stopped after the first four
failed before rollout: the output-relative runtime proof was absent. The V173
root now links runtime_optimization_20261014 to the existing verified V168 proof
directory. No runtime/model changes; same days, samplers and source resumed.
Two real paths in `t210266` completed capture, frozen lookup and clock readback
before restoring the other 47 shards; no outcome statistics were used to select them.

Next: let all 48 shards finish, then retrieve only the aggregate summary/status.json
and logs. Full trajectories, residuals, predictions and actor/day tables stay on
the servers. Execution has started; scientific confirmation is still pending.
