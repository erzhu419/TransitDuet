# V104: frozen-lower system action credit pilot

## Frozen experiment

Keep B200 lower/upper actors, native critics, alpha, dual, deployment and main
unchanged. Fit a diagnostic model only. Three separate cost outputs: once-only
full-day waiting + onboard person-seconds; actual arrival headway CV squared;
actual mean fleet readiness delay in seconds. No native reward or goal weighting.

Replay the 20 V101 native references without intervention to recover their
existing causal SystemStateEncoder inputs. Append the native seven-bin law.
Passenger exposure already incurred is a label-only baseline, never an input;
subtract completed-prefix CV squared from its endpoint label, and use terminal
readiness directly. These common baselines cancel in every action difference.
No latent queues, day/seed ID or future observations enter the model.

Train on V101's 40 consumed prefixes, four paired futures and all seven actions.
For each goal define V = sum(p_native * Y), A = Y - V. V and A have independent
parameters; the three goals also have separate networks. Free categorical heads,
not a quadratic response. Same 16-hidden-unit Tanh action architecture and
initialization, 256 full-prefix Adam updates at 0.0003 for separated A and direct
Q. Train-only standardization: A uses contrast scale, Q full-outcome scale, V
level scale. Thus this compares a parameterization/scaling package, not centering
alone. Zero A is the no-action-signal control. No fit-budget/seed/penalty sweep.

Freeze the saved model before generating fresh reference or outcome labels.
Validation: actors 12037/933083, demand days 2041001--2041004, first eligible
seven-bin bilateral contexts after 7200/21600 seconds, four new event-coupled
futures, seven bins. 8 references + 448 full-day single-action continuations.
These actors are represented in development; demand days and futures are new.

## Decisions and gates

Report value-level squared error separately from all 21 action-pair difference
MSE/MAE per goal, contrast noise variance / four futures, and observed contrast
energy minus noise. Average futures/time within actor/day; use existing crossed
actor/day bootstrap. Never count seven actions as independent samples.

For each model use one identical LP over seven action probabilities: minimize
predicted passenger change subject to nonpositive predicted CV and readiness
changes. Normalize solver rows with development contrast scales; native p is
feasible by centering. No entropy temperature, dual or penalty tuning. Evaluate
its exact mixture against all seven full nonlinear simulator continuations.
Record the law to expose endpoint saturation. Predicted feasibility is not a
physical guarantee; this does not update the actor or claim a full-day policy.

Credit pilot passes only if separated A reduces paired-difference MSE against
both direct Q and zero for all three goals (paired 95% CI upper bound < 0).
Service gate: passenger CI upper < 0 and CV/readiness/unlaunched/incomplete upper
<= 0. Both gates are required before considering a larger independent check;
neither alone authorizes a main-policy switch. No post-hoc rescue on these days.

## Why this differs from the failed paths

V59 updated an actor before system values were calibrated: no actor update here.
V75 reduced absolute prediction error but misranked actions: paired differences,
free seven-action heads, zero control and realized action effects are primary.
V78 used locally linearized policy gradients: this fits full endpoint action
labels and measures nonlinear single-action outcomes, not a policy step.

Server scheduler only, node001--node006, dedicated CPU environment. Smoke uses
consumed V101 day 0: tests, native full-path no-op and both extreme actions must
reproduce cached labels/timing exactly. Keep model/trajectories on server; pull
only small completion markers and final summary. This small 2-actor/4-day pilot
can reject a candidate, but cannot establish durable policy superiority.
