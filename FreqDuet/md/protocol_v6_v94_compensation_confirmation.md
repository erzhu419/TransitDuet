# V94: frozen compensation-rule confirmation

Freeze V93 source `0561d3776e`, its seven-bin laws, observable prefix classifier,
zero split threshold, executable targets and original joint gate. No new fit,
actor/critic/dual update, terminal-time replay or main-policy switch.

Use all twelve existing V34 main-training seeds. The primary validation cohort
is the ten seeds other than V93's 12011/933013; report the original two and all
twelve separately. These are archived runs, previously present in V34/V35,
not twelve newly trained policies or a test of training convergence.

Use sixteen fresh demand days 1941001-1941016 and three matched logical-event
samplers. Replay three regimes: B200 late; unchanged V93 preserve; original
no-gain-only replacement of both groups. There are 576 cells and 1,728 full-day
paths. Preserve-minus-late is the sole primary control contrast. Both-minus-
preserve remains the locked mechanism contrast; both-minus-late is a control,
not a fallback winner. Report passenger/journey, arrival CV-squared, readiness,
completion, actual holding and terminal dispatch gaps, plus covariance and
mean-normalization/boundary changes with exact reconstruction.

Average repeats within each training-seed/day cell, then reuse the existing
crossed training/evaluation bootstrap with 5,000 draws. Apply the unchanged
passenger-improvement and CV/readiness/completion nonharm gate to the ten-actor
primary cohort. Report actor-specific day intervals and separate original-two
and all-twelve crossed results. Do not count samplers as independent policies.

Smoke uses consumed V93 days and two additional actors, checks exact B200
endpoints and full sampling/executed-holding paths, runs all three regimes,
and tests covariance identities and crossed inference before opening fresh
days. All tests, replays and statistics use scheduler on node001-node006 in
the existing dedicated CPU env. Saved timing primitives and checkpoints remain
on-server; only small markers and summaries return locally. No controller is
admitted by this experiment, including if the diagnostic gate passes.

Frozen source `9c6c0a1e3f`; all twelve archived lower-39/lower-199/upper-199
and runner-199 files are present on-server. Smoke `t172994` is queued for two
CPU slots. All six allowed nodes are online but live planning reports zero
free CPU, so no local fallback or forced oversubscription is used. Register
eight resource-bounded 72-cell shards, 24 workers each: `t172995-t173002`.
Their exact smoke marker gates fresh-day access; aggregate `t173003` waits
for all eight shard markers. Tests and smoke have not run while capacity is
unavailable. Only marker/summary directories are configured for retrieval.
