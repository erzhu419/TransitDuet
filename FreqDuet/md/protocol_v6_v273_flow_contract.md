# V273: flow evidence and executable-prefix design

V272 stays closed. No V272 forecast,threshold,weight,window or solver repair;
no old-day fitting,replay,training or main switch. This stage uses constructed
cases only and releases0 physical paths. Original12 vehicles/seven bins remain.

## Observation Change

Native boarding serves all waiting passengers until capacity;there is no within-day
abandonment. With post-boarding queue Q and boarding B,new arrivals satisfy
`A_k = B_k + Q_k - Q_(k-1)`. APC slack certifies Q=0;full does not identify Q.
Clear-to-clear identifies A=B;clear-to-full gives A>=B;full-to-clear gives
0<=A<=B;full-to-full gives only A>=0. A zero-seat bus boarding0 is not zero demand.
Between TWO clearance events,total arrivals equal intervening boardings,including
full visits. This aggregate count is an interval label,not an instantaneous rate
or an observed HF burst. An unfinished censored block has no finite future upper.

Extend the existing CompletedPrefix in memory,not a new recorder lineage:retain
the first visit,actual arrival order,bus/trip,APC counts and BOTH availability and
completion clocks. Legacy visits/solver inputs stay unchanged. A visit with
either clock not strictly before query breaks the label chain;never drop it and
bridge across it. Same-clock arrivals keep native bus order and have0 new arrivals
between boardings,not a fabricated positive duration. No queue/destination oracle.

## Decision Change

Any later controller must distinguish first-action execution from a proposed
future allocation. Planned later cuts/additions cannot fund today's action or
certify regularity unless execution is secured within original permissions.
Evaluate fallback continuation with ONLY the current action changed,as well as
the proposed full plan;unidentified demand remains unknown rather than clipped.
This is a design condition,not an online gate implemented in this stage. H is
an actuator counter,never slack. Model feasibility still is not actual no-harm.

One high/manual server task checks flow identities,strict prefixes,tied clocks,
legacy bridge tests,and two loss-of-future-action counterexamples. No new labels
are consumed. Next specify the native passive-equivalence budget and an independent
development cohort before traffic integration;do not retest on V272's four days.
After connectivity recovered,high/manual task t211737 ran source717483c80d on
node004 and exited0:all21 synthetic tests passed(14 new and7 legacy bridge).
The clearance-block example recovered90 arrivals;the full-plan recourse example
had CV2 delta0 but executing only its first action increased model CV2 by0.000128.
Small output:summary/status.json. No real-data reads,physical paths,fits,training
or main switch. Synthetic acceptance does not replace native passive equivalence.
Limitations:the identities require the complete native arrival roster;no demand
forecast or finite uncertainty envelope,controller or traffic benefit is claimed.
