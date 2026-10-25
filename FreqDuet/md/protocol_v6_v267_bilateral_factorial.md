# V267: matched V107 bilateral upper-regime pilot

Keep the V107 bilateral025 law and native seven bins unchanged. Cross B200
versus the existing exact360-second fixed schedule with bilateral025/zero.
The fixed schedule is a requested timetable, never forced actual dispatch.
Original12 vehicles,physical readiness,demand tape,clocks and checkpoints stay
the same. Record public-clock total dispatch lateness,not only eligibility delay.

Use the first four actors in frozen V107 order(12011/12049/12071/12097),and its
first four consumed days2071001--2071004. This is attribution,not new-day
confirmation. Forty unique full-day paths:32 learned upper paths and8 fixed
day/holding paths shared across actors,subject to fixed bilateral actor-invariance.
Do not count shared fixed paths as independent actor repetitions.

Six smoke paths:four first-actor/day arms,reused formally if accepted;one
second-actor fixed bilateral path;one existing external fixed/zero reference.
Require exact cached V107 endpoints/action counts/coverage,matched contracts,
full fixed-bilateral action/timing equality across the two actors,and external
fixed-zero parity. Only engineering validity releases36 remaining paths in
eight shards(max8 concurrent). No outcome-quality screen. At most42 starts;
failures consume their slots,no automatic retry,replacement or budget extension.

Report passenger time,actual CV2,readiness,original-public total delay,holding,
completion,upper contrasts at each lower,holding contrasts at each upper and
their difference-in-differences. Use existing5000-draw crossed actor/day paired
bootstrap and actor/day means. Keep both signs and crossing-zero uncertainty;
no winner selection,noninferiority margin,training or main-policy switch.
All tests/replays/analysis via scheduler high/manual on node001--node006;
only small markers and the final summary may return locally.

Launch snapshot:source25ed615133,smoke t211575 on node004 exited0. Seven synthetic tests
and all six full-day checks pass;the four matrix corners are reusable. Cached
V107 endpoints/action counts/coverage are exact,fixed bilateral actor traces
match,and fixed zero matches the original external baseline. No outcome gate.
Remaining36 paths:t211577,t211579--t211585,all running high/manual with no
auto-requeue,four each on node004/node005. Snapshot:30 total starts,20 formal
paths saved,zero completed shards. No efficacy conclusion or policy admission.

Completion (2026-10-25):all eight shards exited0;40 formal paths plus2 engineering
checks used exactly42 starts,no retries. High/manual aggregate t211587 exited0.
Only the46,928-byte summary returned locally. All trips launched and completed.
The fixed paths remain shared by day,not four independent actor repetitions.

Candidate minus control,95% crossed actor/day bootstrap intervals:

| Contrast | Journey,min | Actual CV2 | Readiness,s | Public launch delay,total s |
| --- | --- | --- | --- | --- |
| B200 bilateral - B200 zero | +0.6506 [0.5197,0.7821] | -0.144870 [-0.168088,-0.117955] | +82.90 [74.06,94.92] | +21694.75 [19376.03,24854.37] |
| Fixed bilateral - fixed zero | +0.6453 [0.5170,0.7736] | -0.145806 [-0.168640,-0.118838] | +82.31 [73.48,94.27] | +21565.25 [19250.50,24698.00] |
| B200 bilateral - fixed bilateral | +0.0010 [-0.0048,0.0066] | -0.000066 [-0.000423,0.000247] | +0.50 [0.10,0.92] | +355.13 [30.91,682.00] |

Holding at B200 adds156.78s/trip. Waiting changes by-157450.81 person-s
[-325850.63,10224.88],but onboard time increases925905.31 person-s
[906360.81,941988.31]. Fixed-upper holding has the same tradeoff.
Every actor mean and every day mean shows more journey/readiness delay and
lower CV2 with holding. The upper journey/CV2 contrasts and their interactions
cross zero;readiness/public-delay interaction is positive,not a coordination gain.

Decision:retain V107's improvement over the old learned B200 lower;this pilot
does not establish a learned-upper or frequency-separation benefit. Do not expand
this B200 matrix or select a winner. Next method work should change the slow-plan/
holding resource coupling,using fixed+bilateral as the matched strong comparator,
and require realized dispatch changes plus independent-future joint service gains.
No new replay,training or policy change is authorized by this aggregation.

Limitations:four consumed days and four archived actors form an attribution
pilot,not independent confirmation or equivalence. Launch delay is summed across
trips;readiness is the existing mean metric. Full contrasts,components and actor/
day means are in results_freqduet/protocol_v6_v267_bilateral_factorial/summary/summary.json.
