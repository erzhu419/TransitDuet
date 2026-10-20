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
with pilot labels excluded (3,084 scalar bytes in the preflight report).
All 400 captures t189939-t190338 completed done/exit0 with zero retries.
Successful exit records were reconciled before proceeding; no capture was
rerun. Placement: node006 136, node004 104, node005 83, node001 52, node002 25.
The 4,800 P/S jobs t190356-t195155 are submitted through scheduler, unpinned
over node001-node006, for 268,800 native episodes. Initial native process
t190357/node006 is alive at 100% CPU / 578 MiB; the dispatch receipt records
5 running, 123 launching and 4,672 queued jobs at that snapshot.
E32 was not dispatched at that initial sampling snapshot.

Follow-up (2026-10-20): all 4,800 P/S jobs completed done/exit0, with zero
retries and 268,800 native episodes. Seal t195343/node004 passed the full
receipt/menu/seed checks and froze all actions and matching weights before
test E32 access. Only its 12,009-byte summary/index was pulled; full panels,
actions, models and trajectories remain server-side. The sparse union has
914 active states, requiring 3,656 E32 jobs / 108,672 native episodes.
Evaluation t195344-t198999 is submitted through scheduler with unchanged
policy and inference rules. The first physical E32 check, t195353/node001,
is alive at 97.3% CPU / 582 MiB; its dispatch snapshot records 10 running,
118 launching and 3,528 queued jobs. No new fits or performance conclusions
are introduced; the primary and matched-control tests await independent E32.

Independent E32 completion (2026-10-20): all 3,656 tasks completed done/exit0
with zero retries. Official report t199244/node001 qualified 108,672 native
episodes / 29,248 unique state-futures; only 18,437 report bytes were pulled.
The registered joint-transfer gate passes. Cost/wait versus raw have both
CI95 upper bounds below zero; pooled / first16 / last16 waiting retention
is 92.37% / 97.62% / 87.17%, exceeding the registered 50% point floor.

Below, estimates and CI endpoints are multiplied by 1e6; waiting uses minutes.
Negative differences favor the joint rule.

| Comparator | Metric | Mean | Prefix CI95 | Conditional MC CI95 |
| --- | --- | ---: | --- | --- |
| raw | cost | -202.006 | [-288.844, -119.627] | [-265.551, -138.460] |
| raw | wait | -1182.737 | [-1658.035, -781.291] | [-1299.168, -1066.305] |
| thinning | cost | -25.592 | [-52.490, 2.744] | [-48.385, -2.800] |
| thinning | wait | -246.449 | [-376.541, -133.691] | [-293.894, -199.004] |
| service ranking | cost | -12.153 | [-38.628, 10.337] | [-26.931, 2.626] |
| service ranking | wait | 15.392 | [-11.078, 44.784] | [-0.186, 30.970] |

First16 / last16 cost versus raw: -225.068 / -178.943; wait: -1244.139 /
-1121.335 in the same scaled units. Both halves also favor joint over
thinning on cost/wait. Versus service ranking, cost favors joint in both
halves but waiting is worse in both. Relative to full base, joint costs
11.301 more and waits 97.669 more (scaled); no extra fleet saving is supported.
Thus the service-preserving rule transfers, but learned fleet information
has not established additional cost value over strong matched controls.
The all-state waiting reduction is about 0.071 seconds per single intervention.

Next: Stage299 adds paired retention-margin/half/group inference without
changing this primary result. Then define a deployable service-only control
on old fit/calibration data, keeping Stage298 retired from policy tuning.
Stage300 preserves the original risk/joint candidates in the simultaneous
calibration family and adds service-only cutoffs from old fit quantiles.
Only base/service candidates can be selected; the existing joint stays
frozen. New simulation admission requires a nontrivial service subset with
better development cost than base. The four-module test set passes 28 tests.

Stage299 audit t199245/node001 completed done/exit0 without new fits or
simulation (43,954 report bytes). The paired waiting-floor margin is -542.534
in the scaled units: prefix CI [-784.631, -341.923], MC [-619.030, -466.038].
Both E16 margins also have both CI upper bounds below zero. Each half's
cost/wait versus raw has both CIs below zero. Against service ranking, only
last16 cost has both CIs below zero; first16 and pooled cost do not. Waiting
versus service ranking is slightly worse in both halves, with CIs crossing zero.
Fixed-stratum leave-one-prefix point ranges remain negative: cost versus raw
[-210.417, -187.664], waiting [-1201.383, -1043.087]. Group means are
descriptive: cost favors joint in 10/10 root/regime groups, waiting in 9/10;
hour-burst/persistent-shift groups supply 85.83% of pooled waiting benefit.

Stage300 t199248/node001 completed done/exit0. Its 52-candidate development
family selected service cutoff 0.0022712083333332397: 142/169 base changes,
112.68% of base waiting gain, cost -0.000461337 versus base -0.000427096.
This admits prospective confirmation, not a performance claim.
Stage301 locks this cutoff and the unchanged joint rule on 400 fresh test
prefixes, replicas52-91. Its critic-increment claim requires joint cost
versus service-only to have both CIs below zero and both E16 differences
negative, together with the original joint-transfer gate. Native sampling
reuses the qualified Stage298 path; the new arm only retains supported base
actions, so no redundant native pilot or reference fitting is added.
The six-module focused/regression suite passes 45 tests.
