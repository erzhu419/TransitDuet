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
