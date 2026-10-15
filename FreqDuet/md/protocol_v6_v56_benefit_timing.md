# V56: timing of the existing joint-action benefits

Freeze the original V37 12 contexts, U200/L200 actors, B200 deployment state,
seven action bins, fleet and verified optimized runtime. Apply the existing V38
`select_branches` rule unchanged to the archived 49-pair bank. Compare reference,
first-only, second-only and joint arms; deduplicate equal action pairs. The second
decision remains the next eligible same-stop/direction different-vehicle event,
which can change with the first action. This is development mechanism diagnosis,
not a new action search, training experiment or deployable policy.

Reuse the four existing detailed dominant-context traces after matching actions,
decision identities and V37 endpoints. Replay only missing unique arms, at most
48 full days including smoke; smoke's completed arm is reused by the formal run.
Every replay must reproduce all recorded V37 endpoints. A mismatch stops that
arm and is retained in the failure ledger; it cannot enter the final aggregation.

Compare cumulative waiting/onboard person-seconds, dispatch-backlog trip-seconds,
holding commands and launch readiness at common reference-clock cuts: first and
second decisions, first +15/+30 minutes, each intervened reference trip's terminal
return, both reference returns, and day end. Return cuts never follow a candidate's
changed return clock. Missing reference returns stop accounting rather than being
silently replaced by day end. Report induced second-decision identities separately.

Tick-integrated exposure is a timing diagnostic, not the restricted journey
estimator. Keep exact archived daily journey/CV/readiness outcomes in a separate
table. Report short/full exposure sign reversals, short-exposure/daily-journey
sign reversals, and disagreement between the two daily metrics separately. Include
all 12 contexts, both singles, joint and factorial interactions; no outcome-based
context exclusion or threshold tuning. Descriptive counts are not independent
policy-confirmation statistics.

Tests, preparation, one end-to-end smoke, missing-arm batches and aggregation all
run through scheduler on node001-node006. Only completion markers and small result
tables return locally; raw traces, evaluation cells and checkpoints stay remote.
If true short-window effects frequently reverse by day end, the next candidate
must address tail/cross-trip credit. Otherwise compare action-difference prediction
against these matched outcomes before choosing which prediction component to fix.
V55 is closed as a failed calibration candidate; its pending alignment audit only
explains the negative result. No retrospective attenuation or main-policy switch.

## Execution

Implementation commit `2235bb48c4`; frozen diagnostic source directory
`source/v56_timing_20261015` under the server-side V56 result root. Scheduler
`t162169` runs V37/V38 regressions, seven timing tests and plan preparation.
`t162170` waits for its small preparation marker, then validates cached dominant
accounting and one missing intervened full-day arm. Both allow only node001-node006.
Preparation passed: 32 unique arms, four cached traces and 28 missing replays.
Smoke `t162170` and retry `t162185` failed before any new replay because the audit
incorrectly required equal service-clearance clocks. Retry `t162186` was cancelled.
All four cached arms have `service_cleared`, zero unserved passengers, and end
clocks 50684/50708/50688/50455 s. Commit `4995dc915f` uses the fixed configured
evaluation horizon instead; after verified service clearance only remaining
unserved waiting exposure can increase. Truncated traces still fail. Replacement
`t162188` passed 26 tests, then exposed the second loader error: this dataset
derives its clock through EpisodeProtocol and has no JSON `max_time` field. It
was cancelled before another retry. Commit `bcd5768e3a` initializes the original
frozen environment protocol directly. Replacement `t162189` runs the original
regressions, twelve timing tests and full-day smoke from
`source/v56_protocol_clock_20261015`. All 27 tests and full-day smoke passed:
the replayed seed12011/eval45007/t36000 first-only arm exactly matches V37,
and all four cached dominant arms pass fixed-clock exposure accounting. The
smoke arm is reusable, leaving 27 new full-day replays in the 28-item batch plan.
Submit missing-arm CPU batches only after smoke passes, using the missing-arm
count above; aggregate only after all completion markers. Earlier failed attempts
remain in scheduler history; the result's `failed_arm_attempts` counts replay-arm
failures specifically, not pre-replay smoke failures.

Formal batches `t162200-t162203` use scheduler-planned workers 4/8/8/8 on
node002/node004/node005/node006. The 28-item missing-arm plan reuses one completed
smoke arm, so at most 27 new days are run. Aggregation waits for the four small
`exports/batch_*/done.json` markers; raw traces and checkpoints stay server-side.

## Result

All batches and aggregation `t162204` completed: 12 contexts, 32 unique arms,
four cached traces, 28 new full-day replays including smoke, zero failed replay
arms, and exact V37 endpoints throughout. Joint journey improves in 9/12 contexts;
the other three are noops. In 5/9 nonzero contexts the 30-min passenger-exposure
change is harmful despite a beneficial full day. The 15-min and both-reference-
return cuts each also have five reversals, but not necessarily the same contexts.
No daily exposure/journey sign disagreement occurs in these four-arm comparisons.

In the dominant context, joint exposure is -632 person-s at 15 min, +3032 at
30 min, +4318 at both reference returns, and -246160 at day end; the post-return
part is -250478 person-s. First-only looks beneficial at 30 min (-2123 person-s)
but harms the full day (+12207); joint adds holding initially yet reduces full-
day commanded holding by 1405 s. Thus a perfect short-window forecast can still
choose the wrong action; the mechanism is not explained by immediate readiness
alone. Mean hindsight joint deltas are journey -0.042168 min, CV -0.003250 and
readiness -5.491094 s. These are development-bank opportunities, not learned-policy
improvements. Next V57 compares realized trip credit with full-day outcomes using
the existing V36 bank, before specifying a cross-trip system-credit candidate.
