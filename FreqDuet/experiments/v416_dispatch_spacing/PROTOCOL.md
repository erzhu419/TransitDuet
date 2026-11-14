# V416: Dispatch-spacing protection under a queued timetable

Keep V415 hourly running, the native capped continuation, 12 buses, and the
existing 0/5/10 s holding menu. No change to published release times or terminal
authority. Compare a candidate to the same-state capped reference in each of
the existing demand scenarios. Only activate when a direction has at least two
unlaunched trips whose fixed release times have arrived. Protect the adjacent
gaps from the last actual launch, through the queued trips, and to the first
following planned trip if modeled. The first edge uses the published target;
internal edges use the difference of their two public clocks.

For every protected, modeled edge, require
`candidate_gap <= max(published_gap, reference_gap)`. This allows movement
within the public gap and improvement of an existing long gap, but rejects a
larger excess even when total launch delay falls. All original admissibility
conditions still apply; if no alternative survives, use the original capped
reference. No added rollouts per query. It is a model-side constraint on long
gaps, not an actual-dispatch guarantee or a minimum-gap terminal hold.

**Current release:** one zero-traffic server check. Verify both backlogged
directions, observed boundary and next-plan edges, no change outside backlog,
rejection of a cheaper but spacing-worse action, and exact original selection
when the constraint is inactive. Include the V415 real-kernel clock fixture.
No real traffic, replay, training, or promotion is released.

**Proposed traffic check:** after a separately bounded release, first run one
old-day engineering path with the constraint disabled and require exact V415
hourly parity (actions, physical clocks, random events and endpoints).
Then compare pooled V406, frozen hourly V415 and hourly plus V416 on four
fresh slow development days (12 paths, at most four day-pair jobs in parallel).
The old-day path is for equivalence only. No retries or training.
Compare guarded vs hourly for dispatch spacing, then guarded vs pooled for
retained readiness and public-delay benefit; both contrasts must include
journey and actual arrival CV2. Keep 2550004 as a consumed diagnosis, never
as a release or success gate. An unchanged policy because few queries
activate is a negative result for this mechanism, not grounds to widen it
on the same days.

**Preflight result.** Scheduler task `t215047` finished on node004 (high,
manual dispatch). All 13 checks passed with zero physical starts. In the
controlled two-scenario fixture the original capped selector chose 5 s; the
guard vetoed it in the second scenario and selected 10 s. Both took six model
evaluations. Inactive behavior matched exactly. The frozen hourly kernel's
clock checks also passed; no claim about actual dispatch behavior is made yet.

[Server preflight summary](review/preflight/results.json) |
[Scheduler receipt](submission_preflight.json)

**Bounded traffic release.** Freeze old day 2550001 for one guard-disabled
equivalence path; fresh days 2560001--2560004 for the three arms. The budget
is 13 starts. Before release, check unused day metadata and the pinned
runtime. Fresh jobs depend on exact old-day equivalence; aggregation depends
on all four day receipts. Keep 5,200 common random-event keys and draws per
day and the same fleet, roster and legal commands. All tasks use high priority
and manual scheduler dispatch, with no automatic retry.

**Development screen.** Relative to hourly V415, require lower mean actual
dispatch CV2, improvement on at least three days, and no increase in mean
positive gap excess over the public interval. Relative to pooled V406,
require lower mean readiness delay and no increase in public launch delay.
Journey and actual arrival CV2 must not worsen in mean versus either
comparator. Report equal-day paired differences and descriptive 95% intervals
from all 256 day bootstrap draws. This four-day screen is not confirmation
or evidence of statistical noninferiority; passing only justifies a separate
fresh confirmation. Report activation and veto counts without widening scope.

[Frozen traffic manifest](traffic_protocol.json)

**Dispatch.** Zero-traffic task `t215048` passed on node004 after checking
28 manifests. The server confirmed one start for `t215049`, the old-day
engineering comparison. Jobs `t215050`--`t215053` each own one fresh day
and wait for the engineering receipt; `t215054` waits for all four summaries.
The engineering run reuses the V415 runtime identity in its raw case receipt.
No fresh traffic path had started at dispatch verification.

[Traffic preflight receipt](review/traffic_preflight/results.json) |
[Traffic submission](submission_traffic.json)

**Completion.** Tasks `t215049`--`t215054` finished without retry on node004.
The guard-disabled engineering path matched the 12 archived V415 streams,
endpoints and predictor counts. The aggregate accounts for all 13 starts,
including 5,200 paired random events on each of the four development days.
No training or change to the main policy.

| V416 minus comparator, equal-day mean [descriptive 95% interval] | Hourly V415 | Pooled V406 |
| --- | ---: | ---: |
| Actual dispatch CV2 | -0.00405 [-0.00669, -0.00043] | +0.00148 [-0.00288, +0.00449] |
| Positive gap excess (s per dispatch gap) | -0.635 [-1.688, +0.419] | -0.302 [-1.392, +0.675] |
| Readiness delay (s/trip) | -4.886 [-12.779, +1.171] | -8.974 [-14.780, -4.706] |
| Public launch delay (s/day) | -1280.25 [-3348.0, +306.75] | -2351.25 [-3872.25, -1233.0] |
| Journey (s/person) | -5.154 [-11.356, -1.976] | -3.955 [-11.344, +1.230] |
| Actual arrival CV2 | -0.00417 [-0.00695, -0.00140] | +0.00110 [-0.00440, +0.00508] |

Five of six development checks pass. Compared with hourly V415, journey and
arrival CV2 improve on all four days, while dispatch CV2 improves on three.
The journey gain mostly comes from waiting (-4.94 s/person), with almost
unchanged mean holding (+0.076 s/trip). Compared with pooled V406, readiness
improves on all four days, but arrival CV2 worsens on three. The maximum
actual dispatch gap is also larger on all four days (+24 s on average).
Public delay and readiness express the same lateness here because eligible
and public times coincide.

The guard was active on 7,384 of 20,800 queries and rejected at least one
otherwise admissible option on 62 queries (84 scenario vetoes). These are
not counts of root actions changed by the guard. The two hourly paths
ultimately differed at 644 actual dispatch clocks and 2,679 common
bus/trip/stop holding commands, with no schedule edits. Median elapsed time
was 29.1 min/path versus 28.5 min for V415.

**Decision.** Retain the positive incremental effect of the guard relative
to hourly V415 in this four-day screen; do not promote V416 or claim joint
improvement over pooled V406. If continuing, the missing comparison is
pooled plus the same frozen guard, which would separate spacing protection
from the benefit of hourly forecasts when both controllers are protected.
No new paths are released by this readout.

[Engineering acceptance](review/engineering/results.json) |
[Frozen aggregate and daily contrasts](review/aggregate/results.json)
