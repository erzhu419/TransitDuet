# Native Pilot Readout

Campaign: `plan_validity_dev_20261028_r2`. Scheduler t212043-t212054 has9 done
and3 failed tasks. Retrieved9 public JSONs,737,891 bytes total, without models
or fitting caches. The existing analyzer correctly returns `incomplete_or_failed`
with3 missing measurements; confirmation remains blocked.

## Development Results

Full-episode ISE on the8 timing-selection paths per root; lower is better:

| Task / root | History renewal value | Simple threshold | Learned termination |
| --- | ---: | ---: | ---: |
| PointMaze1972110000 | 0.991994 | 1.104563 | 1.165331 |
| PointMaze1972120000 | 1.315062 | 1.272557 | 1.620809 |
| Reacher1972210000 | 4.404082 | missing | 4.118874 |
| Reacher1972220000 | missing | 4.806159 | missing |

PointMaze history beats termination in both roots (-14.9%/-18.9% ISE), but
threshold is mixed (-10.2%/+3.34%); the pooled threshold difference is -2.95%.
Reacher history is6.92% worse than termination in the sole complete paired
comparison. Do not compare its unmatched root2 threshold as though paired.

All3 available termination instances are behaviorally noncollapsed, conditional
on state at a common clock, and improve on their selected best constant clock
(selection ISE gains0.439757/0.211667/0.217079). Their registered development
eligibility passes; the missing fourth instance is unresolved. PointMaze and
Reacher controllers improve on initialization; all3 history fitting banks cover
the6 query categories with nonzero waypoint changes.

## Budget And Interruption

The9 completed measurements cost56,844.486 CPU seconds (15.79 CPU hours), with
41,354,146 environment steps. Controller builds dominate cost, followed by
branch supervision/search, not ridge fitting. Observed p99 latency is1.63-2.79ms,
below the10ms cap,
with21 deadline misses among86,400 decisions. Including both preflights and the
earlier runtime-gate failures gives57,092.381 known measured CPU seconds.

t212050/t212052/t212054 were killed by priority preemption for t212022 and
later relaunched into their existing outputs. They exit with `FileExistsError`;
only unfinished private `running` reports remain, with no matching processes.
Interrupted CPU/step costs are unknown, not zero. Scheduler priority preemption
calls `evict_to_queue` independently of `no_auto_requeue`; the non-resumable
workers correctly refuse to overwrite outputs. No algorithm/scheduler code changed.

## Decision

Retain all9 results and the original recipe. Resolve preemption/relaunch handling
and interrupted-cost accounting before separately registering the3 missing
same-seed cells; do not retrain successful cells. Then check the complete pilot
before considering recipe lock. Numeric budgets/risk thresholds and independent
full-episode/stress reliability evidence remain pending. This analysis launched
no tasks and did not access confirmation paths.

Limitations: selection-path results are exposed development measurements, not
independent evaluation or cross-root CI evidence. C1/C2 and stable cross-task
superiority are not established. Sources: the campaign's
[`analysis/cost_pilot_gate.json`](../results/plan_validity_dev_20261028_r2/analysis/cost_pilot_gate.json),
[`cost_pilot/`](../results/plan_validity_dev_20261028_r2/cost_pilot/) public records,
and [`analysis/scheduler_completion.json`](../results/plan_validity_dev_20261028_r2/analysis/scheduler_completion.json).

## Recovery

`preempt1` remeasures only t212050/t212052/t212054's logical cells, with identical
training/selection/fitting seeds and parameters. New attempt directories and
cost IDs retain the interrupted originals; the other9 cells are not resubmitted.
These3 finite, non-resumable measurements use high priority and loss protection,
without hard/preferred nodes: they are excluded from normal-priority victims
and cannot themselves trigger pinned-node priority preemption. Resource eviction
or process failure remains possible. The scheduler's existing missing-prerequisite
preemption fix passed9 tests; its working tree was not changed. Project tests:28 passed.

Input: [`inputs/recovery_preempt1.json`](../results/plan_validity_dev_20261028_r2/inputs/recovery_preempt1.json).
After completion, run the existing analyzer with `--recovery-bundle` pointing at
that input to combine the frozen9 originals and the3 declared attempts. It keeps
interrupted costs unknown and `lifecycle_costs_complete=false`, even if all12
logical measurements complete. Numeric budget admission remains blocked.
No confirmation task is authorized by this recovery.
