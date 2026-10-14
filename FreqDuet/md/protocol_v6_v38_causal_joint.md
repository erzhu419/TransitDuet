# V38: fleet mechanism and observable joint scoring

V37 development context (train 933013, scenario 45007, target 7200 s) contributes
about 90% of the selected incremental journey gain. Actor actions are (0, 0);
the selected pair is (15, 45), at 7206/7759 s, buses 7/8, trips 38/40, stop 3.
Both unilateral changes worsen journey by about 0.01 min. Joint journey changes
by -0.202022 min and total realized holding by -1405 s. These are development
counterfactuals, not a validated policy or evidence against fixed-headway.

## Mechanism audit

Replay the four factorial arms, keeping the original checkpoints and deployment
background. Instrumentation must reproduce every archived V37 outcome. Record
terminal arrivals, physical vehicle reuse, published/actual launches, readiness,
and holding commands. Verify the command sum against its episode metric rather
than equating commanded and realized time. Compare tick-integrated waiting,
onboard and backlog exposure at common clock cuts: before first action, between
actions, through the reference target vehicles' returns, and afterward.
The factorial interaction describes policy-path nonadditivity; it does not
identify an exclusive causal mediator.

Four matched replays completed (`t153255-t153258`, source `d262164d53`).
Aggregation `t153264` verifies command/realized holding equality in these arms.
There is no vehicle reassignment. Trip 38 returns at 9120 s in both reference
and joint; trip 40 returns at 9557/9584 s, delaying its next departure by 27 s.
Before both reference returns, joint waiting/onboard exposure increases by
1174/3144 person-seconds; afterward it decreases by 170221/80257 person-seconds.
There are 1423 changed subsequent holding commands and -8025 backlog trip-seconds
after that cut. The evidence supports a later policy-path cascade, not an
immediate fleet-readiness benefit. Only the 40205-byte mechanism summary returns.

Use current APC/AVL, observed past movement and holding, terminal inventory,
route geometry and the currently published timetable for new scorer inputs.
Exclude simulator-wide stochastic road speeds, latent queues, future vehicle
returns and realized later decisions. The timetable/ETA margin is a proxy,
not an assigned next-trip commitment or guaranteed available holding budget.

## Scorer development boundary

The first stage may use only its own current snapshot. The second stage may
refresh the snapshot and use the actually executed first action. A first-stage
target must include continuation by its fitted second-stage policy, not an
oracle-selected second action. Predict journey, CV and readiness separately;
do not add a fleet opportunity penalty on top of full-day rollout labels.

Collect 84 first-action prefixes for the same 12 contexts and reuse V37's 588
pair outcomes. Keep contexts grouped, exclude held-out training seeds from
fitting, and cross-fit second-stage continuation targets inside the training
fold. This is a small development replay, not independent confirmation. Freeze
the score before any new confirmatory cohort; preserve failed selections and
actual constraint violations instead of reporting only hindsight-feasible rows.

The prototype uses fixed ridge 10 with signed linear/quadratic actions and
state/action interactions. Second-stage targets are full-day incremental outcomes
relative to the same first-prefix incumbent continuation, exactly zero for its
noop. First-stage labels use inner-held-context second-stage selections. Retain
causal joint, first-only and second-only selections for all 12 contexts. Use
predicted non-increase of CV/readiness, then minimum journey, to select actions;
report actual violations rather than treating predictions as hard guarantees.

All computation and tests use scheduler on node001-node006. Only small exports
and summaries return locally; checkpoints and detailed traces stay on server.

## Development outcome

All 84 snapshot replays (`t153259-t153263`, 19/8/19/19/19 workers) exactly
match V37. Scorer source `2e11f970ea` passes 14 tests in `t153310` and retains
all contexts. Causal joint selection gives journey +0.016058 min, CV +0.000517,
readiness +5.158715 s; 0/12 joint improvements and 7/12 actual constraint
violations. First-only/second-only journey deltas are +0.007614/+0.003820 min.
The prototype is rejected, not promoted. No gate or ridge tuning on this bank.
Next work needs a fleet-pool availability/service-commitment representation and
short-horizon causal rollout with a separately validated tail value, rather
than treating a local ETA margin or a small fitted regression as a hard budget.
