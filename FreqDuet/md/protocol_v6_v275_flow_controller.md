# V275: closed-flow evidence and executable continuation

One complete controller revision following the V268--V274 review. Keep V107,
V272, V274 and the original TransitDuet archives unchanged. No actor, dual,
training, fleet, action, timetable, horizon, cost-weight or beam changes.

Demand uses the last three positive-duration, nonoverlapping clearance blocks
with strict completion and availability clocks. Counts include intervening full
visits. Their count/time ratio is a constant-rate future point extrapolation,
not an identified future bound. Unclosed intervals remain censored. Initialize
queue predictions from the last closed anchor and subsequent visible APC.

For each proposal, simulate both its full plan and only its current action with
baseline continuation. Both must satisfy the existing predicted CV2, physical
return, next-launch and public-delay constraints; fallback passenger cost cannot
increase. Minimize the worse passenger cost. Both evaluations share the original
128-simulation budget; beam4, variables6. Record future proposals and their
subsequent actual command receipts, changed requests and cancellation reasons.

Freeze days2075001--2075004 after server metadata non-consumption checks. Three
arms: fixed360+bilateral025, flow/APC+fallback, flow/homogeneous-cost+fallback.
One actor anchor12011, sampler27011001; 12 unique full days, not independent
future replications. Server synthetic checks precede starts. Eight single-process
units maximum, high/manual scheduler, node001--006. Each slot starts once;
failure consumes its slot without retry, replacement or retuning. Raw records
remain server-side; local transfers are small JSON/log receipts only.

Use four paired days and10000 shared bootstrap resamples(seed2750201). Against
BOTH controls require passenger-cost upper95<0 and actual CV2/readiness/public
total-delay upper95<=0, without extra unfinished trips on any day. Report day
values, action counts, executed H, support/fallback, proposal realization and
latency. A failed gate closes this frozen candidate; a pass only permits proposing
independent confirmation, not switching main or claiming frequency superiority.

Control source894ad14707: t211773 passed83 server synthetic tests on node004.
Its day inventory then descended into old cell/log trees; no physical paths
started. The user then approved stopping only t211773; scheduler cancellation
confirmed one running task stopped,zero related tasks and zero kill failures.
Replacement check t211783 passed84 tests and7527 structured metadata files,
with zero day collisions and zero raw trace reads. It uses index-only entry
f3f15604a5 with unchanged894ad14707 control source and original12-path budget;
high/manual,node004. Its small passed receipt now releases the collection gate.
High/manual units t211774--t211781 launched on node001/node004. The launch
snapshot confirms p0--p7 started,zero completed paths and zero failures;the four
remaining homogeneous paths follow within the same eight sequential units.
Aggregate t211782 waits for eight small unit receipts. Only those receipts and
the final summary sync locally. Main/training stay frozen;no traffic conclusion yet.

## Completed development result

All twelve paths and scheduler aggregate t211782 completed successfully; no failed
paths or retries. All 62,400 commands were verified, fully executed, and service
cleared with twelve physical vehicles per path. Matched event uniforms had zero
mismatches. Only the 39 KB result JSON and small execution records were retrieved.

APC minus comparator; intervals are the frozen four-day paired bootstrap 95% CI:

| Metric | fixed360+bilateral025 | homogeneous+fallback |
| --- | --- | --- |
| Passenger journey, seconds/person | -3.451 [-5.333,-1.355] | -1.235 [-3.580,1.045] |
| Actual CV2 | -0.004766 [-0.006317,-0.003215] | 0.000588 [-0.000143,0.001554] |
| Readiness delay, seconds/trip | 4.563 [1.898,6.320] | -4.611 [-6.019,-3.202] |
| Public launch delay, total seconds/day | 1195.50 [497.25,1655.75] | -1208 [-1577,-839] |
| Executed holding, seconds/trip | 8.888 [4.761,13.483] | -4.594 [-5.830,-3.053] |

Versus bilateral, journey and CV2 improved on all four days, but readiness and
public delay worsened on all four. Waiting fell 103,325.5 person-seconds/day while
onboard time rose 30,549.25: this is not a universal holding reduction. APC used
more 30/45-second actions on every day. Versus homogeneous, passenger cost won
two days and lost two; the APC-specific passenger advantage is not established.

APC supported 17,554/20,800 queries (84.4%), with 2,287 actual same-state overrides;
support ranged 60.2--94.9% by day. Of 223 future non-baseline proposals, all reached
their query, 55 matched the subsequent actual command and 168 differed; none were
cancelled for unsupported prefix. These are command matches,not causal benefit
attribution. Callback mean was 2.02--3.14s, p95 4.24--4.97s, max 15.90s.

Decision: both frozen joint gates failed;close this candidate without expansion,
retuning or independent confirmation. Keep this genuine passenger/regularity
development signal. Model-level fallback nonharm did not ensure full-day fleet
nonharm;the current summaries do not separate prediction error from downstream
propagation. Evidence is four development days, one actor, no independent futures;
no frequency-superiority claim. Main policy and training remain unchanged.
