# MuJoCo v18.2 State-Conditioned Actor Outcome

## Decision

Status: `state_conditioned_actor_stops_before_fresh_path_access`.

V18.2 evaluated every member of the frozen 16-candidate state-conditioned MLP
grid with eight leave-one-seed-out folds and exact full-horizon responsibility
oracles. No candidate passed the complete reused-panel gate, so no fresh path
was accessed and this MLP screen is closed.

## Frontier

The selected one-step, two-layer, width-32 model preserved all 113
reference-feasible paths and changed the executed action on all seven
actor-floor paths. It recovered only 3/7 actor-floor paths, however, versus
6/7 for the v17.14 linear FIR frontier. Fourteen candidates recovered two paths
and two candidates recovered three; none recovered more.

Target normalized MSE was 0.9524, above the frozen 0.75 limit. The maximum
reference-feasible correction RMS was 0.00340, within the 0.01 trust region.
The selected model recovered 0/2 held paths for seed 2802248628 and 3/5 for
seed 294864529. This reversal matters: training on the five positive paths from
seed 294 did not transfer to either seed-280 failure, while the two seed-280
positives transferred to three seed-294 paths. The sparse target support is not
consistent across seed groups.

## Efficiency

Scheduleurm task `t85969` completed on node003 with 16 single-threaded workers.
Scheduler runtime was 48.2 seconds and measured worker runtime was 18.1 seconds.
The scheduler did not sample peak RAM before this short task completed, so no
measured peak-memory value is claimed.

## Next Mechanism

Increasing MLP width, path weight, proposal history, or output cap did not
improve the frozen frontier. The next development step must use causal online
frequency-budget feedback or a feasibility projection that adapts from the
current prefix, rather than another supervised state-MLP parameter sweep.

## Claim Boundary

This is a negative grouped reused-path development result. It does not establish
reward improvement, online learning, fresh-seed generalization, leakage
no-tradeoff, or manuscript support for a state-conditioned actor.
