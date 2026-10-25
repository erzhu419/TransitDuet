# V269: cooperative holding allocation on an event graph

## Decision Change

V268 is closed. Do not expand its coefficient search or change its acceptance.
The new decision is a joint multi-vehicle,multi-stop set of future event times
and native holding commands,not a residual added to each bilateral request.
Optimize modeled passenger waiting plus onboard time with separate regularity,
vehicle-readiness and original-public dispatch-delay constraints. Preserve the
original12-vehicle pool,seven holding bins,boarding/service order and legal masks.
Published dispatch remains exact360 in the first integration;no new terminal,
speed,boarding or assignment authority is introduced.

## Why The Tradeoff Might Improve

Regularity depends on differences between event phases. Passenger cost also
depends on WHERE a delay is spent. Jointly moving a necessary delay past an
alighting point can preserve downstream phases,per-vehicle total holding,return
times and next launches,while allowing existing passengers to finish earlier.
An independent uniform cut does not preserve these coupled event phases.
This is an allocation hypothesis,not a claim that recorded H is removable slack.

The constructed three-vehicle,two-interior-control-position certificate uses upstream
commands(30,20,10) and downstream(10,10,10). Moving to(20,10,0)/(20,20,20)
retains per-vehicle totals(40,30,20). Forty passengers per vehicle alight between
positions;the later control is empty. Intermediate headway moments are preserved,
downstream arrivals are unchanged,and existing-passenger exposure falls1200
person-s. Fixed before/after boundary events are included,so this is not a
moving-window or CV-denominator artifact. Future demand is absent at the
alighting control;downstream fluid waiting is evaluated over fixed boundaries.

## Model And Causality

Use nodes for arrival A,service completion Q,departure D,return R and next launch L:
Q=A+s;D=Q+h;A_next=D+tau;h in{0,5,10,15,20,30,45} at legal query nodes.
Executed events and pending irreversible commands are fixed,not optimization
variables. Future arrivals are re-ordered when needed;do not freeze a hindsight
neighbor. At fixed-public dispatch,L=max(P,R+1) in the single-supply-chain case,
not merely L>=P and L>=R+1. The extra1 reflects dispatch-before-return ordering.
Fleet-wide assembly must use the current causal FIFO pool and event ordering,
never the eventual vehicle assignment as an input or a free extra vehicle.

The slow planner proposes feasible phase trajectories through the same vehicle's
next release,using LF demand and completed-prefix service/run information. The
fast controller updates the affected graph with observed AVL/APC/HF innovations
and executes ONLY its current legal first command;future decisions are recourse.
Every later query uses the updated graph too,not an old bilateral continuation
that cancels the planned allocation. No post-terminal holding node is invented.
Do not assume the existing small shared-scalar upper learned useful authority.
The fixed-public integration tests allocation first;frequency benefit is separate.

Current load and position are available from the existing APC/AVL query contract.
Future load,boarding cutoff,capacity,left-behind passengers,travel and service are
predictions,not privileged simulator fields. In particular,earlier arrival can
miss later boarders when doors do not reopen:waiting cannot be replaced by just
onboard_load*H. The actual full-day once-only passenger ledger remains the final
objective. Prediction constraints are model constraints,not actual-CV guarantees.

## Bounded Next Step

This stage seals the design and checks four constructed cases on the server:
useful same-amount reallocation,no cheaper position,a saturated later control,
and already committed upstream actions. Enumerate all7^6 native allocations in
this small model and all universal-offset action aliases;do not fit parameters
or read V268 outcomes. Failure does not authorize fixture tuning to claim success.
One high/manual CPU scheduler task;zero new traffic paths,training or confirmation.

Next implement the causal event-graph assembler and bounded discrete rolling
solver before traffic execution. Reuse current recording interfaces;do not start
another instrumentation lineage. Freeze horizon-to-next-launch,forecasts,solver
limits and failure behavior before a small complete-day matched pilot. Controls:
fixed+unchanged bilateral,and the SAME graph/constraints with homogeneous rather
than APC/location-specific onboard costs. Record actual H amounts and exposures;
equal model H does not establish equal realized H. Reject forecasts or allocations
that require hidden terminal waits,extra stock or changing executed commands.
Freeze causal forecast inputs before the pilot;do not repair them on evaluation
outcomes or turn a failed pilot into a new calibration/recording series.
Only repeatable actual passenger/CV2/readiness/public-delay joint benefit merits
fresh independent confirmation,then matched-budget frequency ablations.

## Limitations

The six-control exact solver is a mechanism reference,not a production controller.
The constructed model does not establish real-world opportunity frequency,
forecast reliability,full-day service superiority or novel literature priority.
Future field inputs must be causal;fixture future quantities are never runtime
features. No traffic integration or new-path budget is released by this test.
Main and all frozen historical experiments remain unchanged.

Run (2026-10-25):source22cb0a6acb,high/manual t211613 on node004,exit0;
7 tests passed. Each constructed case examines117649 native allocations.
The alighting case selects(20,10,0)/(20,20,20):model passenger exposure-1200
person-s,CV2/waiting/readiness/public delay unchanged,per-vehicle H and subsequent
return/launch clocks exact. Its14 universal-offset aliases yield no joint gain.
Through/saturated/committed cases retain the reference. Only5642 summary bytes
returned locally. Traffic paths,new fits and main-policy changes:all0.
