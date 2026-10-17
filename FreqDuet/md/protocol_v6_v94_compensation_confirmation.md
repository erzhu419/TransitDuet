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

Smoke `t172994` subsequently passed four tests but stopped before any rollout:
the nested smoke output directory was incorrectly used as the runtime root,
so the unchanged optimization-proof loader searched the wrong parent. Separate
the canonical runtime root from smoke-only pair/record output and add a path
regression test. Keep the proof loader, action laws and statistical gate intact.

Repair source `30ea4ce32b`: replacement smoke `t173054` passes five tests and
all eight full-day/native-check paths, exit 0/retry 0. The 161-byte real smoke
marker was retrieved. Old queued tasks are cancelled; replacement shards
`t173055-t173062` retain the original 576-cell/1,728-path design, 24 workers
and explicit 16 GB RAM each. Aggregate `t173063` waits for their eight markers.
The original failure is resolved by this replacement, not retried unchanged.

All replacement tasks completed with exit 0/retry 0. Retrieved only the 178,197-
byte summary. The primary ten-actor crossed intervals pass the original joint
gate: journey -0.33574 min [-0.51058,-0.20565], arrival CV-squared -0.002287
[-0.004352,-0.000393], readiness -98.25 s [-126.37,-71.84], and holding -13.35
s/trip [-16.73,-10.04]. Completion counts are unchanged. All ten actor-specific
journey/readiness intervals are negative; nine CV-squared means are negative,
but only five actor-specific CV-squared intervals exclude zero.

Both-minus-preserve saves another 0.29874 min [-0.44211,-0.18177] and 91.99 s
readiness [-119.76,-66.30], with CV-squared -0.001061 [-0.003786,+0.001586].
Five actor CV means improve and five worsen. This does not reproduce V93's
positive mean CV difference, nor establish nonharm. Its release/holding cross
term increases +0.004423 [+0.002561,+0.006316], but this allocation alone is
not total harm. Preserve reduces raw arrival variance -393.64 s2
[-700.84,-112.36], despite a +0.000526 CV mean-normalization contribution.
The all-twelve crossed result also passes; the original-two fresh-day CV
interval crosses zero. Keep both facts, not a universal per-actor claim.

Server-side axis review (16,655 bytes including the V95 reference budget):
the ten-actor averaged journey/readiness effects are negative on all 16 days;
CV-squared improves on 13/16. Both-minus-preserve improves CV on 9 days and
worsens it on 7, versus five/five at the actor axis. These are descriptive
directions, not extra independent samples or substitutes for crossed intervals.

Next: V95 tests a compensation-blind mixture, with its scalar fixed only from
V93 observable B200 holding budgets. This separates selector evidence from
simply reducing intervention intensity, subject to measured closed-loop dose
differences. V94 is a replicated archived-policy intervention result, not a
repair of training convergence or a fixed-headway comparison. Main stays frozen.
