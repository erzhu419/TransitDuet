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

## Limitations

This step repairs missing causal input data, not policy performance.
Scene coverage remains twenty development groups; third/fourth action-label
scenes remain untouched. Encoder performance is pending.
