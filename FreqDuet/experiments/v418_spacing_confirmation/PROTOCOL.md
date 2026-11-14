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
