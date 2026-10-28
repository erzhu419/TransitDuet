# Completed Native Pilot

Decision: **PILOT_RECIPE_HOLD**. t212077-t212079 all completed successfully;
their3 distinct attempts plus the9 retained originals complete the12 logical
cells. Registered parameters/seeds and hardware profiles match. Within each of
the4 task/root groups, controller-selection ISE curves and selected iterations
match across all3 methods. Retrieved246,922 bytes of new public JSON, no models.

## Results

Mean full-episode ISE on timing-selection paths,2 roots/task; lower is better:

| Task | History renewal value | Simple threshold | Learned termination |
| --- | ---: | ---: | ---: |
| PointMaze | 1.153528 | 1.188560 | 1.393070 |
| Reacher | 4.649214 | 4.521588 | 4.507493 |

PointMaze retains a17.20% mean advantage over termination, but the2.95% mean
advantage over threshold is mixed by root (-10.19%/+3.34%). Reacher is worse
than threshold in both roots (+3.94%/+1.83%); against termination the differences
are +6.92%/-0.036%. No stable cross-task superiority is demonstrated.

All4 stopping heads are noncollapsed and state-conditional. Reacher1972220000
nevertheless fails competence: its4.896113 ISE exceeds the best constant clock's
4.749286 (derived from the saved readiness gain of -0.146827). Therefore the
registered task eligibility is PointMaze=true, Reacher=false, blocking recipe lock.

Saved-weight/cache readout, without fitting or simulation, finds path-CV MSE
reductions versus a pathwise constant predictor of61.27%/58.37% in PointMaze,
but only7.81%/25.14% in Reacher. Selected ridge alphas are10/10 versus1000/100.
The weakness is not solely the comparator: the candidate's local-value prediction
also transfers less effectively across Reacher paths in its fitting bank.

## Budget And Next Step

Completed pilot measurements cost72,297.630 CPU seconds (20.08 CPU hours) and
55,147,390 environment steps. Known campaign costs including preflights/runtime
gate failures are72,545.526 CPU seconds. Maximum cell p99 latency is2.79ms;
21 deadline misses occurred among115,200 decisions. The3 earlier interrupted
CPU/step costs remain unknown, so lifecycle-budget admission is still blocked.

Keep all artifacts and the failed eligibility result. The next bounded proposal
is a frozen-artifact local-to-episode diagnostic using the pilot's reserved
development evaluation paths, not confirmation roots: distinguish weak local
prediction from mismatch between50-step credit and full-episode timing value.
Measure same-state episode renewal headroom as well, to separate limited timing
opportunity from a learning failure.
No threshold repair, extra fitting, simulation or scheduler submission was made
in this analysis. No confirmation task was launched.

Limitations: these are exposed selection-path results with only2 roots/task,
not independent superiority/CI evidence. CV selected alpha and is not a fresh
test. The readout does not establish whether representation, training coverage
or credit horizon is the causal bottleneck. C1/C2 remain unestablished.

Sources: [merged gate](../results/plan_validity_dev_20261028_r2/analysis/recovery_preempt1_result_20261029.json),
[paired readout](../results/plan_validity_dev_20261028_r2/analysis/complete_pilot_readout_20261029.json),
[saved-fit readout](../results/plan_validity_dev_20261028_r2/analysis/cached_fit_readout_20261029.json),
[completion receipt](../results/plan_validity_dev_20261028_r2/analysis/recovery_preempt1_completion_20261029.json).
