# V401: completed, development screen failed

Completed at 04:51 on 2026-11-10 (Asia/Shanghai): 64 training collections,
24 development paths, 1,536 updates per learner and 92 charged starts including
the two preserved engineering failures.

At both retentions, explicit minus implicit/manual is exactly zero on every
day for all traffic endpoints. All 16 learned evaluation paths selected joint
at every one of their 47 publications. Holding histograms and slow summaries
also match manual joint. Thus screen_passed=false: journey did not improve.

The controller was active: 3,370--4,985 supported queries and 969--1,180 holding
overrides per path. Training budget/joint counts were 158/1,346 (explicit) and
156/1,348 (implicit), becoming predominantly joint after the first batch.

All arms share the rho=.25 minus .50 tradeoff: journey -2.255 s/person,
readiness -3.656 s/trip, public delay -952.5 s/day, CV squared +0.001126.
This supports budget sensitivity, not learned-upper or explicit-state benefit.
Typical completed path time was 20--22 minutes; development maximum 27.3 minutes.

Close the candidate under its frozen protocol: no confirmation expansion,
tuning or main-policy switch. One learner seed/four development days do not
establish general ineffectiveness; historical results remain unchanged.

Evidence: [gate](review02/aggregate/results.json), [readout](review02/readout01/results.json).
Task t214088 read JSON only, with zero replay/optimization; about 27 KB retrieved.
