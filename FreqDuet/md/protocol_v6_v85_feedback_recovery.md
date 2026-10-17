# V85: locate lower feedback recovery

V84 replaces the V83 concentrated closed-loop harms by weak, mixed-sign effects.
Recorded-lower responses are larger and more repeatable but can help or harm.
First report signal/noise and paired continuation differences from existing
V84 summaries on the server. No new fitting or controller admission.

Reuse all 24 fixed prefixes and eight matched futures. Compare one physical
0s versus 45s intervention, replaying that future's native lower commands for
900s after intervention, then restoring the frozen projected lower controller.
The two endpoint interventions are a diagnostic contrast, not a restriction
of the deployed seven-bin action set. Keep the frozen upper reacting normally.
Evaluate costs through day end, not just the 900s observation interval.

There are 384 labelled arms. Reuse V84's corresponding closed-loop and all-day
recorded-lower endpoints. Report paired differences of the 45-minus-0 response
against both controls. This tests sensitivity to early feedback recovery;
it is not another value-selector dataset or a cutoff sweep for promotion.

Smoke verifies native complete-path identity for both actors, unchanged causal
prefix and matched logical uniforms, zero pre-release tape mismatches and actual
execution of both recorded/recovered phases. All tests, audits, rollouts and
aggregation use scheduler on node001-node006. Main, actors, critics and dual
remain frozen; no default environment changes or local experiments. Keep tapes
and trajectories on servers and pull only small summaries/markers.

Source `158ebc7df9`; smoke/audit `t166316` passes five V84 and three V85 tests,
both native complete-path checks and the changed-action timed-release check.
Shards `t166317-t166320` passed their smoke dependency and cover `[0,97)`,
`[97,192)`, `[192,288)`, `[288,384)` on
node002/004/005/006, 49/48/48/48 workers. Aggregate `t166321` waits for those
exact markers. No V85 outcome has been used to change a controller.

All 384 labels and aggregation completed exit 0/retry 0. Read-only review
`t166390` (source `a2ea14a985`) passes two tests and reproduces original response
means while separately computing same-action regime differences. For action
45s, recovery-minus-recorded passenger/CV intervals are negative in 15/24
prefixes and positive in none. Recovery-minus-closed intervals contain zero in
all 24 for passenger, CV and readiness. Readiness versus recorded is likewise
unresolved. These are conditional eight-future, unadjusted descriptive intervals,
not equivalence tests or independent training-seed confirmation.

The first recovered decision occurs 0-40s after the 900s deadline; all 328
executed non-cached arms have zero pre-release command mismatches. Later feedback
can suppress taped-path amplification, but full-day endpoints cannot locate
when waiting/onboard costs accrue or attribute them to vehicle turnover.
Keep the main policy and actors/critics/dual frozen. V86 adds passive temporal
credit and return/launch records to existing 45s paths, not another controller.
