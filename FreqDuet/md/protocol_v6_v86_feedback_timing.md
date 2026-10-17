# V86: passive feedback timing audit

V85 suppresses the full-day amplification under taped continuation, but saved
endpoints do not distinguish along-route correction from return/dispatch costs.
Existing per-cell logs contain no temporal ledger. No controller candidate,
new cutoff, actor/critic/dual update or main-policy change follows this result.

Replay the existing 45s intervention only: all 24 consumed prefixes, eight
matched futures and the same three continuation regimes, 576 labelled paths.
Zero/45 action-response inference remains V85's endpoint result; this audit
targets fixed-45 regime differences. Every replay must match its V84/V85
metrics and local witnesses, including original native whole paths in smoke.

Sample exact existing system credit every 300s, plus target and day end:
waiting, onboard, dispatch backlog, executed vehicle/passenger holding, launched
count and readiness-delay sum. Preserve cleared-tail accounting. Record arrival
headway moments through each sample timestamp and existing return/launch ledgers, with bus and trip
identity; changing a bus's next trip is not a matched-trip latency comparison.
CV is cumulative, not an additive exposure. Prefix launch means omit pending
trips, so retain counts and backlog instead of claiming prefix readiness safety.

Compare recovery minus each control on 0-15, 15-30, 30-60 minutes and the later
tail, using paired eight-future intervals within each prefix. Recovery and taped
exposures must agree through 15 minutes. This locates recorded cost changes;
it does not identify natural mediation or establish fresh-day generalization.

Tests, instrumentation smoke, replays and aggregation all use scheduler on
node001-node006. Keep detailed timelines/ledgers server-side, pull small summary
logs/markers only. Inspect whether cost divergence tracks subsequent holding
or return/launch changes before choosing a control-credit modification.

Source `2e72a9fbd9`; smoke `t166406` passes six tests, four native path checks,
three existing 45s endpoint/witness checks and pre-release exposure identity.
An earlier observer attached before protocol initialization; it failed before
simulation and its five queued dependents were cancelled. Initialization now
occurs on the first physical record after reset, with a regression test.
Shards `t166407-t166410` cover `[0,144)`, `[144,288)`, `[288,432)`, `[432,576)`
on node002/004/005/006, 72 workers each. Aggregate `t166411` waits for their exact
four markers. No timing result or controller advantage is claimed at dispatch.

All 576 paths and aggregation completed exit 0/retry 0, reproducing the original
endpoints. Read-only review `t166432` passes nine tests. Recovery-minus-tape
passenger effects average +82.57 person-s at 15-30min, -44.32 at 30-60min, and
-1,054,921.40 after 60min; tail waiting/onboard contributions are -997,548.32 /
-57,373.08 person-s. Tail passenger and cumulative CV intervals are negative in
15/24 prefixes each, positive in none. Executed holding has no clear tail change;
passenger holding exposure falls in 11/24 conditional intervals. Readiness and
backlog tail intervals all include zero. Target-vehicle next-trip identities
match in every pair, but other vehicle assignments can change.

Recovery-minus-closed passenger costs increase in 8/24 early 0-15min intervals,
with no negative intervals; all full-day passenger/CV/readiness intervals remain
unresolved in V85. Feedback suppresses late waiting amplification, not simply
total holding, and no stable readiness mediation is established. These are
consumed-prefix conditional descriptions, not a new policy win. Next separate
the early/late actors' mean logit preference from their centered state response,
without suppressing feedback or updating actor/critic/dual weights.
