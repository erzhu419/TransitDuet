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
