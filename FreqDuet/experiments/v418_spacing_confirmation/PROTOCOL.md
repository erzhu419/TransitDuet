# V418: Independent test of hourly forecasts under the frozen spacing guard

V417 found faster readiness but weaker regularity for hourly versus pooled
forecasts under the same guard on four development days. Test whether the
frozen hourly-plus-guard policy actually improves circulation at preserved
service quality on independent slow days. Do not train or tune anything.

Use the three frozen controllers: pooled V406 without the guard, pooled
with the V416 guard, and hourly with the same guard. Both guarded arms use
exactly the previously executed selector, forecast and fallback code.
Keep demand scale 1.2, road speed input -0.5 m/s, 12 vehicles, 262 fixed
public trips and the 0/5/10 s command menu. The sixteen demand days in
the manifest have not been used to choose policies or thresholds.
The pooled guarded arm isolates what is added by the hourly predictor;
the unguarded arm is the original V416 comparison.

**Acceptance, frozen before launch.** Use equal-weight within-day
differences (hourly guarded minus comparator) with 20,000 paired day
bootstrap draws and two-sided percentile 95% intervals. Against **each**
pooled comparator require the upper interval endpoint to be strictly
below zero for mean readiness delay and at or below zero for actual
nonterminal arrival CV2, restricted passenger journey, and public
departure delay. The acceptable mean arrival-CV2 increase is **0**; no
corridor-specific positive tolerance was established independently.
This is an experimental non-worsening rule, not an operator service
standard. Readiness and public delay summarize the same lateness under
the fixed roster. Report actual dispatch CV2, mean positive excess over
360 s, maximum launch gap, waiting, onboard, tail journey, passengers
not boarded, holding and activity of the guard as diagnostics.
Pooled guarded minus pooled unguarded is a separate descriptive contrast,
not a third candidate selected from the confirmation outcomes. No result
changes the historical V415/V416 verdicts or promotes a policy automatically.

**Release.** Zero-traffic server check first: inspect the frozen bindings,
paid acceptance and unused-day metadata. Use no new engineering path.
Then run 16 day jobs with three sequential arms each, at most 16 jobs
concurrent; 48 full-day starts total. Require an identical generated
passenger count and public roster and all 5,200 event-uniform keys and
values matched within every day. All tasks use scheduler high and manual
dispatch on node001--node006, no retries or replacement days. Incomplete
panels cannot pass. The raw paths stay on the server; sync only compact
receipts and the final aggregate. The main policy remains unchanged.

[Frozen manifest](protocol.json)

**Dispatch.** Zero-traffic task `t215065` passed on node004, with no overlap
found in the checked case locations and 29 old manifests. After explicit
approval of the 48-start ceiling, all 16 day jobs `t215067`--`t215082`
started; first-arm startup receipts were verified on node002 (4), node003
(4) and node004 (8). Aggregate `t215083` waits for all 16 day summaries.
No outcome is available at dispatch.

[Preflight receipt](review/preflight/results.json) |
[Confirmation submission](submission_confirmation.json)

**Completion.** All 16 day tasks and aggregate `t215083` finished with exit
code 0 and no retries. The aggregate accounts for 48 starts on 16 distinct
scenario tapes, with 5,200 matched random-event keys and values on each day.
The 16 synced day summaries match the aggregate. No training or policy change.

| Hourly + guard minus comparator | Pooled V406: mean [95% interval] | Pooled + same guard: mean [95% interval] |
| --- | ---: | ---: |
| Journey (s/person) | -1.79 [-4.98, +1.44] | -3.11 [-5.59, -0.65] |
| Readiness delay (s/trip) | -2.42 [-6.06, +1.59] | -4.57 [-7.13, -2.07] |
| Public launch delay (s/day) | -632.88 [-1588.46, +415.32] | -1198.25 [-1869.01, -542.54] |
| Actual arrival CV2 | +0.00412 [-0.00067, +0.00871] | +0.00052 [-0.00503, +0.00566] |

The joint gate fails: zero of four conditions pass against unguarded
pooled V406 and three of four against pooled with the same guard. Against
the matched guard, journey and readiness improve on 12/16 days. Mean
waiting (-2.58 s/person) and onboard time (-0.53 s/person) both decrease;
journey P95 changes by -20.18 s [-34.02, -7.21]. Arrival CV2 improves on
5/16 days against either comparator; neither interval establishes
non-worsening, and neither establishes a positive mean loss.

Pooled with the guard versus unguarded pooled has mean changes of +1.32
s/person in journey, +2.16 s/trip in readiness and +0.00360 in arrival CV2.
Their intervals cross zero. This descriptive contrast does not reproduce
the development-day regularity improvement seen when adding the guard to
pooled forecasts. The new panel has no unguarded hourly arm and cannot
independently confirm the V417 factorial interaction.

The hourly guard was active on 28,947 of 83,200 queries, with at least one
veto on 224 queries (pooled guard: 29,583 active, 194 with a veto). These
are not counts of executed actions changed. The longest-gap and mean
positive-excess contrasts cross zero against both comparators. Median
single-path runtime was 38.4 min (hourly guard), 36.5 min (pooled guard)
and 34.6 min (unguarded pooled).

**Decision.** Report the independently supported journey and readiness
benefits against the matched-guard comparator, not a joint improvement
at preserved regularity or superiority over unguarded V406. Do not
promote V418, relax the margin, or reuse these days to tune the guard.
No further traffic is released by this readout.

[Independent contrasts, gates and daily evidence](review/aggregate/results.json)
