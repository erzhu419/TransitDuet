# V409: Holding Dose and Action Authority

Retain the V408 corridor, 1.2x demand, 12 buses, fixed public timetable and
corrected V406. Add one variant: V406's current action enumeration uses only
original legal bins at or below 10 s. Forecasts, continuation (native fixed10),
constraints, objective, execution, and fallback are unchanged. Test the cap
as a decision restriction, not as a clip applied after optimization.

Use only V407 heavy development days 2470201/2 for calibration. Reuse their
full V406, fixed10 and fixed15 paths; run capped V406 once per day. Fixed0 has
exactly zero commanded holding by construction. For each contrast, choose the
unique whole-day mixture weight that matches the candidate's mean development
holding seconds/trip: fixed10/15 for full V406, fixed0/10 for capped V406.
No test-day matching, extrapolation, or within-path interpolation. If the full
target is unbracketed, stop before confirmation. If the capped target is
unbracketed, report that contrast as not identified and retain the fixed10
and full-vs-capped contrasts. Do not change candidate or cap endpoints.

Sixteen new independent heavy demand days x five arms, 80 confirmatory starts
and two development starts. Use paired random streams and equal day weights.
For each matched contrast first require the 95% paired day-bootstrap interval
of holding difference to lie within +/-3 s/trip (a day-average engineering
resolution smaller than the minimum nonzero command). This matches the mean
total dose, not each trip or query. Then require journey interval upper bound
strictly below zero and CV2, readiness and public delay upper bounds <=0.
Full V406 vs its mixture is primary; capped vs its mixture is the prespecified
mechanism contrast. Full vs capped isolates removal of the high bins from the
controller, including its consequent closed-loop response; it is not a
fixed-dose contrast. Report pure-rule contrasts, waiting, onboard, tails,
unboarded, executed bins, fallback, and runtime without using them to promote.

No traffic before zero-traffic preflight and no confirmation before frozen
calibration. Failed starts count; no auto-retry, replacement day, V408
calibration, training, promotion, or paper-mainline change. All computation
through scheduler high/manual on node001--006; only small summaries sync.
