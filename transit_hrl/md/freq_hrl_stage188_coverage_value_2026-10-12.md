# Stage188: Expanded Grouped Value Validation

Stage187 r2 finished480 tasks, all exit0. Qualify the full cohort on a CPU
node and retain its timetables/labels on shared server storage. Combine120
unchanged old states with480 fresh states: 100 groups/600 states/3,600 labels.

Fixed tail_summary versus timetable_sequence, floor_absolute, proper-MSE,
seed178,100 updates per fit and unchanged optimizer. Each mode has100
leave-scene,5 leave-regime,2 leave-lower-root and1 full fit:216 fits/21,600
updates. Both selectors share each fit. No inner tuning, native or actor updates.

Keep the same rootwise cost<0/wait<=0 gate across all three held-out splits.
Report the old120-state subset separately against Stage184/186; comparing
new600-state averages against old120-state averages would mix sample changes
with training-coverage effects. Parallelize folds through scheduler, one CPU/
3GB each, node001-006 unbound. Full bank/models remain server-only.

Eight focused tests pass. Next: qualify the bank, run all matched folds and
decide from held-out effects rather than training losses or hindsight oracle.

Run native_transit_coverage_value_stage188_fit_20261012_r1, fit code251473b2db.
t144065 completed exit0 on node006: all480 collections and100 combined groups
qualified, bank34,907,152 bytes server-only, compact receipt4,658 bytes.
Posthoc joint-improvable states:397=201/300,401=191/300. These are future-outcome
oracle diagnostics. The216 fit tasks are t144082-t144297.
Node-side report qualification uses complete fold exports, not delayed task
status writeback; only its compact report directory is synchronized.

t144335 qualified all216 fits/21,600 updates. Both joint gates remain false.
Wait-feasible held-out means (negative is better):

| Mode/Split | Cost397 | Cost401 | Wait397,min | Wait401,min |
|---|---:|---:|---:|---:|
| Scalar/scene | -0.000399 | -0.008760 | -0.000980 | -0.001783 |
| Scalar/regime | +0.001834 | -0.008389 | +0.005820 | +0.001843 |
| Scalar/lower root | +0.001837 | -0.002207 | -0.006010 | -0.005093 |
| Sequence/scene | +0.003560 | -0.006356 | -0.004253 | +0.006777 |
| Sequence/regime | +0.006675 | -0.008278 | +0.000123 | +0.003217 |
| Sequence/lower root | +0.000433 | -0.006808 | -0.005763 | +0.005783 |

Scalar improves some matched old-state effects but does not close all splits.
Sequence gains are mixed. Mean fit wall time: scalar2.07s,sequence51.74s.
Next: cached cost-ranking versus wait-mask error decomposition, not more seeds.

## Limitations

The action labels still describe one forced action followed by original
greedy continuation, not repeated learned upper control. Six callbacks are
correlated within each scene; two frozen lower roots remain. Additional
development coverage is not independent confirmation or end-to-end HRL.
