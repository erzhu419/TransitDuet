# V224: prepare-time identity/phase handoff and passive budget

Seal only. Current budget:0 archive scans,0 replay paths,0 training,0 new futures.
The executable preregistration contains field lists,slot IDs and acceptance
rules. Its only command runs synthetic contract tests and writes protocol.json;
it has no collector or simulation entrypoint. Main and V223 law stay frozen.

## Prepare record

Immediately after native `_prepare_for_action` returns,copy its cached evidence
before another vehicle updates. Record prepare/read/pre_action clocks,stop,
trip/bus/direction,occurrence,native gaps/validity/sources/target and three roles:

- Current:the prepared trip and physical bus,not a later query-time lookup.
- Front:the SAME cached predecessor's completed same-stop/direction departure.
  The recorder has trip/time only;resolve physical bus from the completed launch
  ledger,not the bus's current trip after reuse. Require F=prepare-departure.
- Follower:the cached native predicted physical bus/trip and the EXACT AVL
  snapshot used by that estimate. Capture original tick-start or inline-fallback
  sampling provenance;never rerun the estimator or replace it with later AVL.

Use a sidecar monotonic observation order for native AVL-freeze,launch,departure,
prepare and query events. Preserve native order,including same-second events.
Do not add events to V221's existing EventClock or shift its legacy event_seq.
V223's shadow launch/prepare sequence inputs both use the sidecar order,never
mixed counters. Bind each query to its exact prepared occurrence;retain repeated
calls and unused prepares. New data live separately from the legacy payload.

For EACH role copy its immutable launch-phase provenance:the SAME admission's
native/proposed/APPLIED plan,eligibility ticks,actual plan/clock and reserved/
actual bus. Latch nonzero only if the intended plan actually executes on that
reserved bus at the coordinated tick. All launch records and matched front
departures must precede prepare. Replanning overwrite,cancellation,unknown or
undelivered plans remain unsupported. Unmodified launched trips have explicit0,
including late launches. V107's unapplied shadow proposals are not phase offsets.
Never backfill from future actual neighbours,completed full-day labels or late
onboard observations. Missing phase is not0;each field is present with a reason
for legitimate absence. A missing recording is failure,not legitimate abstention.
Only supported/known-zero rows enter V223's phase resolver;other statuses pass
None even if an unsupported row happens to contain equal plan clocks.

## Reserved passive budget (not launched)

Reserve8 logging-enabled full-day paths:V107 and upper-only for each fixed
(actor,day)=(12037,2151001),(12037,2151002),(933083,2151001),(933083,2151002).
Reuse the matching8 V221 archives as references;no additional reference runs.
Preserve original sampler187000000+actor+100*day and original source2b0115af7b,
V218 source_b09088f636,V211 c45b2d0a93 and optimized V9 runtime dependencies.
Do not substitute current mutable runtime code. At most2 workers later.

First pair:c0_v107/c0_upper. Both must pass before the remaining6 start. Every
started attempt consumes a slot,including failures;no auto retry,replacement
context,extra futures or enlarged budget. Existing complete files can be reused
for analysis recovery. Collection requires a subsequent explicit authorization.

Only the EXISTING V221 upper-only intervention executes;all lower actions stay
bilateral025. Compute V223 shadow requests without applying them. Original12
vehicles,seven bins,upper projection/caches/EMA/cadence and all frozen models,
calibrations,duals and main are unchanged. No gain fitting or bin forcing.

## Acceptance and decision

Require exact recursive equality of EVERY legacy field/key against its archived
arm:no ignored fields,rounding,tolerance or resorting. This includes policy
inputs/probabilities,upper/lower actions,uniform identities,event sequence,
plans/tasks/responses,arrival/departure/launch/return clocks,issued/executed
holding,physical/resource ledgers and every endpoint,including passenger time,
actual CV2,readiness,total public dispatch delay and completion. Wall time is
not a simulation clock and need not match. Native RNG state/call counts and
estimator/policy/planner call counts must be unchanged BEFORE/AFTER each
observer-only body. This does not require unavailable historical RNG states
or estimator counters;archived action-uniform and callback traces stay exact.

Every executed query must join one correct prepare record with complete fields,
strict-prefix identities/phases and exact cached evidence. Report each arm and
context,all support/abstention reasons,known-zero versus unavailable phases and
native/shadow raw requests/bins. Legitimate unsupported states are retained.
All equality AND recording gates are required;no aggregate hides a failed path.

Shadow bin differences are not executed action authority or service benefit.
Zero differences stop intervention progression;nonzero differences only justify
a separately budgeted actual-action smoke. Independent confirmation stays shut.
All later execution uses scheduler high/manual,node001--006. Raw archives and
sidecars stay remote;only small per-path checks/counts return locally.

Sealed source282f0d884b,t211242(high/manual,node006),exit0:8 synthetic contract
tests passed. Server protocol.json is frozen;only the small status/log returned.
Started paths,archive scans,training and new futures are all0;launch_authorized
is false. No collector was attached and runtime equivalence remains untested.

## Authorized implementation

The user subsequently authorized the8 passive slots,with c0 gating the other6.
Keep the sealed protocol unchanged;record this approval in a separate execution
authorization. Frozen V9 builds immutable AVL snapshots INLINE in its native
follower update,not through the newer live tree's tick-freeze API. Copy those
actual estimator arguments and returned bus identity. The old snapshot lacks
trip IDs:join the selected physical bus to trip/direction copied at entry to
that SAME native update,not at query. Record this provenance explicitly. Its
prepare argument supplies the pre-action clock;do not add a missing native bus
attribute or claim it was present. No estimator,control rule or clock changes.
