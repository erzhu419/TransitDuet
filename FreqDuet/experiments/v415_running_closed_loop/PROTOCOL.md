# V415: Entry-hour running in closed loop

V414 passed the independent five-minute launch-clock screen on eight slow
days. Test whether its only model change, full-link running time conditioned
on the simulated entry hour, improves delivered service. Reuse the frozen
V406 capped controller from V412/V414: 12 physical buses, high demand (1.2),
road-speed input -0.5 m/s, same public schedule, native masks and executed
bins {0, 5, 10}. Keep APC inputs, stress scenarios, partial links, committed
commands, bilateral continuation, objective and admissibility unchanged.
Both reference and alternative evaluations use the same forecast within an
arm. The pooled arm uses the original code. The hourly arm uses V414's
strictly completed and available prefix, three same-hour visits and pooled
fallback. No public-slot shadow observer runs in either arm.

**Release.** Zero-traffic preflight verifies source binding, unused day
metadata, completed-prefix/clock semantics, exact pooled results for all
three actions, and a real kernel timing difference on a supported synthetic
state. One engineering path on old day 2520208 uses the detached kernel with
pooled forecasts. Require exact old live streams and endpoints, excluding
callback wall time; failure stops release. Eight new days then run both arms
with identical generated passenger counts and all 5,200 random-event keys and
values paired. One job per pair, at most eight concurrent. Total cap is 17
traffic starts, including engineering; failure consumes any started slot and
is not retried. No training or change to the main policy. Scheduler high,
manual dispatch on node001--node006; retain raw data on the server.

**Decision.** Report hourly minus pooled per demand day and the equal-day
mean with paired 95% percentile-bootstrap intervals. Require the upper bound
strictly below zero for vehicle readiness delay per trip, and at or below zero
for public launch delay per day, passenger journey per passenger and actual
arrival CV2. The two operating delays are closely related under a fixed
timetable, not independent evidence. Retain waiting/onboard, tail journey,
holding dose, action differences, predictor usage and callback latency for
interpretation. A complete screen that fails the joint gate is not a policy
promotion; incomplete pairs are not a result. This does not change V412's
original transfer verdict or establish an LF/HF effect.

**Dispatch.** Zero-traffic preflight `t215034` passed on node004, including
the exact capped-action comparisons and a check of 27 archived manifests.
Engineering task `t215035` started once on node004. The eight paired jobs
`t215036`--`t215043` are registered behind the engineering receipt; aggregate
`t215044` waits for all eight paired summaries. All were submitted high with
automatic requeue disabled. No new-day control result exists yet.

[Preflight receipt](review/preflight/results.json) |
[Pipeline submission](submission_pipeline.json)

**Completion.** Engineering, all eight paired days, and aggregate `t215044`
finished with exit code 0 and no retries. The engineering path matched 12
archived streams and the endpoints; only callback wall time was excluded.
All 17 starts are accounted for, with 5,200 matched event keys and uniforms
per new day. No training or main-policy change.

| Hourly minus pooled, eight equal-weight days | Mean | Paired 95% interval | Days improved |
| --- | ---: | ---: | ---: |
| Readiness delay (s/trip) | -2.189 | [-7.311, +1.455] | 4/8 |
| Public launch delay (s/day) | -573.5 | [-1915.4, +381.3] | 4/8 |
| Journey (s/person) | +1.272 | [-1.251, +5.029] | 4/8 |
| Actual arrival CV2 | +0.006608 | [-0.003764, +0.024028] | 3/8 |

All four frozen gates fail. Day 2550004 accounts for most of the apparent
readiness benefit (-17.813 s/trip and -4667 s/day public delay), while journey
and CV2 worsen (+13.051 s/person, +0.064519). For influence assessment only,
the other seven days average +0.043 s/trip readiness, +11.3 s/day public delay,
-0.411 s/person journey and -0.001665 CV2. The gate retains all eight days.

Holding changed by -0.448 s/trip [-0.804, -0.093], but increased mean
waiting (+1.347 s/person) outweighed the small onboard reduction (-0.075).
The two paths differ in 5,501 of 38,560 common same-bus/trip/stop commands and
in 1,456 actual launch clocks out of 2,096, with no schedule edits. These are
closed-loop path differences, not same-state action effects. Hourly means
were used in 16.2% of full-link calls. Median runtime across days rose from
34.7 to 37.4 min per path and median daily callback p95 from 0.79 to 0.92 s.

**Decision.** Keep V414 as a validated forecast and stop V415 promotion.
Better factual launch-clock prediction did not establish better control.
If pursuing the mechanism, first read the existing paired paths for the
day-2550004 divergence in actual launch intervals, holding and downstream
headways. Those archives can localize the difference, but cannot identify
unexecuted actions' effects. No retuning or additional traffic is released.

[Engineering receipt](review/engineering/results.json) |
[Frozen aggregate](review/aggregate/results.json)
