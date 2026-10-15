# V44/V45: propagation audit and completed-window calibration

## Protocol
V44 retains V43 and frozen U200/L200 control. Trace arrival/ready/action/departure
at 2/6/10 h for eight of 64 paths; match trip/stop/stage and retain missing counts.
V45 issues 15/30-min forecasts every 15 min, from 1 to 13.5 h (51 origins/day).
Per day/horizon, retain 16 nonoverlapping windows ending strictly before issue;
require eight for correction. Mean raw residual shifts the point; past
prequential nonconformity scores expand nominal 90% bands using ceil(.9*(n+1)).
Only holding, APC onboard exposure and dispatch backlog are calibrated. Waiting
exposure remains raw. No across-seed pooling, gate tuning or controller change.
Keep V42/V43 common-clock thresholds: holding MAE <= historical, absolute bias
<=5%, passenger MAE <=105% of historical, mature holding coverage >=80%.

## Evidence
V44 `c1a005ca3b`: 23 tests/smoke `t155545`; eight exact V35 replays
`t155546-t155549`; aggregate `t155557` / `7f78c88577`. Travel MAE 36.0/35.5 s,
service 3.87/3.80 s, holding pipeline 12.6/12.9 s. At 15-30 min, arrival clocks
err by 110/87 s and follower gaps by 107/90 s. No nonzero requested actions
lack predecessor-departure evidence;
geometry and target headways align. Acquisition smokes `t155530/t155535` failed
on recorder API differences; their retries were cancelled.

V45 recorded source `718b366ee2`: smoke `t155591`, 12 days `t155592-t155596`.
Eight exact archived controls plus four fresh-demand controls (actors
12011/933013 x demand 19511021/19511027), each matched to an unobserved replay.
Final observable aggregation `t155600` / `dfd8cb4fef`: 29 tests pass; raw common
forecasts equal V43; observable-channel recomputation equals recorded results.
All computation used scheduler on node001-node006; only small summaries return.

Common-clock 30-min holding MAE, seconds (warmup included):

| Cohort | Raw | Calibrated | Historical |
| --- | ---: | ---: | ---: |
| Development | 783 | 378 | 337 |
| Reused demand | 690 | 269 | 333 |
| Fresh demand | 617 | 327 | 207 |

Fresh-demand all-origin holding MAE is 739 -> 498 s (historical 484); residual
bias -237 s, about -8.3%. Onboard MAE worsens: all origins 58,870 -> 61,723
person-seconds; common clocks 52,626 -> 63,333. Mature holding coverage is
99.3% over all origins, but average band width is 3,355 s. All joint screens fail.
Canonical results: `protocol_v6_v45_completed_calibration/observable_summary`.

## Limitations And Next Step
Original records also contain simulator-only waiting calibration; operational
aggregation discards it. Only three independent observable channels are reused,
with strict chronological replay and equality checks. Endpoint calibration does
not fix transition physics or validate counterfactual action response. No main
policy promotion. Next model completed-leg travel-time distributions and their
cross-vehicle/temporal dependence, preserving headway variation inside rollouts,
then validate holding/passenger coupling on new contexts without retuning gates.
