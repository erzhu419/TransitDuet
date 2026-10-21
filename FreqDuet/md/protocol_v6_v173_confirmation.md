# V173: frozen V172 independent confirmation

Source `0151b7c719`; map `a5b45df520`, protocol `f5d152f0bc`. Preparation
`t210203` passed 25 server tests. Earlier `t210196` prepared only the superseded
reporting entry; no collection tasks use its source `7914534858`.

- Exactly days3152001-3152032 and samplers1711000000-1711000031, shared across all 12 actors and old/handoff modes: 384 pairs/768 paths. No replacements or outcome-dependent additions.
- Completed collection: `t210266` plus `t210268`-`t210314`: 48 shards, all exit0/zero retries, eight actor/day pairs each, two workers/CPU cores, high/manual on node001-node006.
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

Queue follow-up: all 48 local receipts were present. Original aggregate `t210315`
ran on node004 and completed; no replacement computation or collection was needed.
Only its 112,092-byte summary was retrieved. All 768 paths/3,993,600 queries and
12 mode/scope crossed grids are complete; no finite unresolved labels/missing references.

V171 joint decision: **not admitted**, with partial positive evidence:
- All/active/hour2 pass in both modes; every coverage, lower-tail and finite-support gate passes.
- Hour2 coverage is 82.668%/82.715% versus matched frozen V167 75.308%/75.339%. Interval-score delta versus V167 is -43.296 [-49.419,-37.045] / -43.215 [-49.302,-36.905] seconds (95% crossed CI). All12 actor and all32 day score marginals improve.
- Clearance fails upper-tail upper95 (10.750%/10.864%) and strict coarse-score improvement (delta upper95 +0.394/+0.500s).
- Hour13 and hour13 moving-launch fail upper-tail upper95. Moving-launch mean upper misses are 9.599%/9.652%, but upper95 is 10.948%/10.995%, above10%; this is failure to confirm the tail bound, not mean misses proven above10%.
- Moving-launch score is worse than matched V167 by +0.470 [0.136,0.815] / +0.693 [0.327,1.095] seconds. Whole-day score still improves versus V167 by -3.218/-3.212s; the hour2 gain does not establish a clearance improvement.

Next: retain the hour2 result; do not promote the full map or switch policy.
Only read-only diagnosis of clearance/moving-launch interval bounds and day-level
tail variation is warranted next. These32 confirmation days are now consumed;
do not tune/re-gate on them. All maps and other frozen items remain unchanged.
