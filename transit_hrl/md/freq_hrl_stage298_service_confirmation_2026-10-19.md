# Stage298: Fresh Service-Protected Confirmation

Run: `native_transit_service_confirmation_stage298_20261019_r1`.

Lock Stage297's joint cutoff 0.002181045736174378 before new scenes. Reuse
the same critic/raw/lower weights and P8/S8 proposals; no refitting or new
calibration. The risk-only calibrated family selected full base, so base is
also its control. Main arms: raw, old247, base, legacy risk, joint, same-count
service ranking and expected count-matched thinning.

Use 400 fresh prefixes, 40 in each root/regime group, test replicas 12-51.
Ten excluded pilots use pilot replica1. Scene and future seeds are disjoint
from Stage296. This expands Stage297's reserved first 120 before any fresh
capture, addressing prefix uncertainty rather than adding futures to old
states. Six probes/profile, 16 fixed profiles; profiles are not new seeds.

Freeze model/cutoff -> native pilots -> qualify -> 400 captures -> 4,800 P/S
tasks -> seal actions/weights -> independent E32. Budgets: 430 pilot episodes,
400 captures, 268,800 P/S episodes; sparse E32 at most 537,600 episodes. Native
jobs use 1 CPU/1,536 MiB, offline 1 CPU/3 GiB, unpinned node001-node006.

Primary is unchanged: cost/wait versus raw must pass both prefix and paired
MC CI95 and improve in both E16 halves; service/wait must improve, retaining
at least 50% of base waiting gain pooled and in both halves. Extra cost value
versus base, thinning and service ranking are separate two-interval claims.
Average profiles and probes before 400-prefix stratified bootstrap5000 seed298.

Service ranking takes the lowest causal S8 service scores at exactly the
joint intervention count within each profile/root/regime/probe. It is a
frozen cohort-budget control, not an online policy; its actions already
occur in base, so it adds no rollout. All matching is sealed before E32.

Validation: 55 focused/regression tests pass, including protocol-specific
receipt/seed validation, renamed-arm numerical equivalence, retained-gain
rejection and exact groupwise service-control counts.
Shared sampling defaults and Stage296 inference remain regression-tested;
only protocol paths/seeds and arm names are supplied by the new experiment.
This remains finite-MC single-intervention validation with fixed lower, not
end-to-end learned HRL. Full panels, models and trajectories stay server-side.

Execution: source `c762765aa1`; freeze t189917/node006 completed done/exit0.
All ten pilots t189919-t189928 passed; official pilot-report t189937/node006
qualified 430 episodes and 60 exact replay identities, peak RSS 687.37 MiB,
with pilot labels excluded (3,084 scalar report bytes pulled this turn).
All 400 captures t189939-t190338 completed done/exit0 with zero retries.
Successful exit records were reconciled before proceeding; no capture was
rerun. Placement: node006 136, node004 104, node005 83, node001 52, node002 25.
The 4,800 P/S jobs t190356-t195155 are submitted through scheduler, unpinned
over node001-node006, for 268,800 native episodes. Initial native process
t190357/node006 is alive at 100% CPU / 578 MiB; the dispatch receipt records
5 running, 123 launching and 4,672 queued jobs at that snapshot.
E32 remains undispatched;
actions and matching weights will be sealed only after all P/S jobs succeed.
No new fits or performance conclusions are introduced by this sampling step.
