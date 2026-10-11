# Stage185: Full Preaction Timetables

Recover all six complete candidate curves and both directions' pending
departures, including current scheduled times, commitment flags and lower
headway goals. Keep every event in a variable-length six-channel sequence;
trip IDs join records but are not network inputs.

Twenty passive replays: two already-labelled scenes per root/regime,
six callbacks each. Original greedy selector, frozen lower and duration
models are unchanged. Require exact native source reproduction and match
the existing 120 causal states/occupancy contexts; reuse all 720 labels.

Scheduler only, node001-006 unbound, one CPU/3GB per replay. Full observer
and raw artifacts stay server-side; sync only compact context JSON.
Budget: 20 episodes/1,227,600 ticks, no branch reruns or learning updates.
After qualification, compare a learned full-sequence encoder on the same
labels/groups before expanding independent development-scene coverage.

Fifteen focused tests pass: six new capture/sequence/join/scheduler tests
and nine existing passive-context/commitment regressions.

## Result and Next Step

t143310-t143329 all completed exit0 on node001/004/005/006. The merge
qualified all twenty original controls, 120 states and unchanged 720 labels.
Event sequences contain 45-215 pending departures. Only 6,705,434 bytes
of context JSON were synced; no full observer, raw data or checkpoint pulls.
The known candidate3/4 alias differs at 55 tentative departures by up to
66 seconds in the recovered full representation, despite equal prefixes.
Next: matched full-sequence value training, then independent development
scenes for wait-value generalization.

## Limitations

This step repairs missing causal input data, not policy performance.
Scene coverage remains twenty development groups; third/fourth action-label
scenes remain untouched. Encoder performance is pending.
