# Stage180: Expanded Coverage and Head-Normalized Value Learning

Stage179's negative mean total cost is driven by two fleet reductions in
root401; root397 and both roots' wait still worsen. The first bank has only
one negative fleet label, the second has 27. Fleet/nonfleet paired target
RMS has squared ratio 73.3 under the shared training scale.

Join all first/second registered scene states: 120 states, 720 labels, 20
groups. The second bank is now development training data, not a test set.
Compare joint_native with paired_heads, primary fixed to paired_heads.
Only paired-MSE head scaling changes; use each training fold's paired RMS.
Restore physical units before native-cost regret and inference. No added
wait reward, cost weights, feature changes, optimizer tuning or actor updates.
Both use the same 96-16-2 network, seed178, learning rate .003, weight decay
.01 and 1,500 steps per fit. All callbacks in a scene remain together.

Twenty scene, five regime and two lower-root cross-fits plus one full fit
per normalization: two unbound scheduler CPU tasks on node001-006,
56 fits/84,000 updates and zero new native episodes. Primary must improve
cost and not worsen wait for both roots in all three splits before a
separate third-scene frozen transfer screen.
Run native_transit_value_balance_stage180_fit_20261011_r1, revision
e9eecde210, tasks t143135/t143136; node001-006 allowed, no node binding.
Eighteen focused tests pass: training-only head scales, native regret units,
serialized inference, all-label/group retention, held-out separation, matched
budgets and unchanged previous value-model tests. Local runs are unit tests;
the actual 56 fits run through the scheduler.

## Result and Next Step

Both tasks completed with exit0; 56 fits/84,000 updates qualified. Only two
result JSONs (1,269,869 bytes) were pulled. Primary cost / wait deltas:

| Split | root397 cost | root401 cost | Overall wait, min |
|---|---:|---:|---:|
| Scene | +0.0290310 | -0.0124084 | +0.0185583 |
| Regime | +0.0157201 | -0.0048973 | +0.0240000 |
| Lower root | +0.0032832 | -0.0060662 | +0.0258583 |

Normalization improves scene cost by 0.0091296 relative to joint_native,
but worsens lower-root transfer by 0.0037512. All six root/split wait means
remain positive. The registered gate fails: no third-scene collection or
native learned-action deployment is authorized by this result.

The primary fits development cost well (-0.0385940), but scene-selected
predicted fleet delta averages -0.185532 versus realized -0.006944. Of
720 fleet labels, 664 are zero. Wait-feasible hindsight candidates still
improve cost in 73/120 states: the action pool has joint-improvement room.
Next: absolute future fleet-cost supervision above the irreversible causal
peak, plus a directly learned wait head and matched constrained selection.
Keep the current labels, grouped splits, physical cost weights and unused
third/fourth action-label scenes. This is a new development experiment,
not a reinterpretation of the failed Stage180 primary.

## Limitations

The normalization change addresses loss imbalance, not proven calibration
or a physical fleet bound. Cross-fits and one-action labels are development
evidence; repeated learned upper actions and complete HRL remain unverified.
Third/fourth registered scenes' action labels remain unused.
