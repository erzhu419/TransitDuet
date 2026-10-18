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
alone. V is a shared diagnostic benchmark, not an input to either action head or
the solver. Zero A is the no-action-signal control. No fit-budget/seed/penalty sweep.

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

## Scheduler record

Physical/model code: `dc6121729d`; fit-only development-error reporting:
`a2cc8a773f`. Smoke `t179203` on node001 passed: five tests, exact native/no-op
full paths, and cached 0/45-second labels and event timings. Earlier assertions were
repaired (zero versus absent gradient; JSON anchor-key conversion), with no
fresh demand days used and no scientific gate changed.

Reference recovery `t179188--t179191` -> fit `t179192` -> fresh references
`t179193--t179196` -> 448 arms `t179197--t179201` -> aggregate `t179202`.
Arms use five scheduler-planned shards, 13/15/6/9/9 workers, eligible on
node001--node006 only. Each phase waits for its predecessors' small markers.
All formal shards and aggregate finished with exit code 0. Both preregistered
gates failed; main and all native learners remain frozen.

## Outcome and next decision

Two actors, four new demand days, 16 prefixes, four paired futures per prefix.
Separated A significantly reduces all three action-difference MSEs versus the
matched-budget direct Q. This does not establish useful credit: versus zero A,
CV and readiness MSE are significantly worse; passenger MSE is inconclusive.

| Cost | Separated A MSE | Zero A MSE | Actual one-action change versus native law [95% CI] |
| --- | ---: | ---: | --- |
| Passenger person-seconds | 6.766e8 | 5.218e8 | +1347 [-12692, +15438] |
| Actual CV squared | 1.163e-6 | 1.023e-6 | +0.000102 [-0.000470, +0.000692] |
| Mean readiness seconds | 12.354 | 8.707 | +0.820 [-0.725, +2.554] |

Predicted passenger change was -10508 person-seconds, not the realized +1347.
Training-label MSE improvement did not generalize beyond zero A. Noise-corrected
action energy has intervals crossing zero for all three costs.

Cached review `t179253`, source `f08fb7c0e8`, passed three tests on node004;
zero new rollouts/updates. A two-future system-cost teacher evaluated on the
other two futures (then swap) also fails joint service: passenger -1248
[-7424, +3114], CV squared -0.0000071 [-0.000173, +0.000158], readiness +0.829
[-0.016, +1.329]. Its selection optimism intervals exclude zero for all three
costs. Thus cross-day network fitting alone cannot explain the failure; noisy
action selection is already visible without a learned state-to-value mapping.
This conditional teacher is exploratory, not a deployable policy or value bound.

Decision: stop promotion of this fit, retain the V/A separation implementation
and negative evidence. Do not choose direct Q from its favorable point estimates
or tune a rescue gate on these consumed days. Next validate frozen action
contrasts on independent continuation labels before any actor update or larger
training matrix. Summary: `summary/status.json`; cached review: `review/summary.json`.
