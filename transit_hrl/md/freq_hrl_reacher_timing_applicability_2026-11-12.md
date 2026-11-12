# Reacher Timing Applicability Precheck

[Frozen protocol](../protocols/plan_validity_reacher_timing_applicability_2026-11-12.json).
Reuse both pilot controllers on the unchanged Reacher task. On each old
timing-selection set, reproduce the best of six fixed offsets and keep the
worst as a timing-sensitivity reference. Then evaluate five schedules or
waypoint conditions on 24 new development paths per root.

- Waypoint authority: at the same best fixed clock, compare the issued
  waypoint with the previous call's absolute waypoint. Both execute the same
  first waypoint, make 24 calls and run the same lower feedback controller.
  Check whole-episode ISE and the latter half of every option.
- Timing sensitivity: compare the old-selection best and worst fixed clocks.
- Timing opportunity: align a call to the first reachable regime switch
  inside each bin, or use its best fixed offset. Compare this privileged
  schedule with the best clock and with a seven-bin rotation of the noninitial
  offsets. The rotation preserves each path's offset histogram and call count.

Each gate contrast requires at least 2% relative improvement, positive effects
in both path halves and a positive lower nominal paired 95% bound within
each frozen root. Both authority contrasts and both event contrasts must pass
on both roots. A failed event test rules out a stable gain for this alignment
rule, not all adaptive timing opportunities. The event schedule uses hidden
simulator information and is not a deployable baseline.

Fixed cost: 403,200 steps, 336 full episodes and 8,064 calls, including old
clock reproduction. No fits, controller changes, critic work, or original
confirmation access. This is a two-root development screen, not transfer
confirmation or a cross-root significance test.

## Completed Result

[Root summaries](../results/reacher_timing_applicability_20261112/summary.json)
and [exit records](../results/reacher_timing_applicability_20261112/completion.json).
Both t214879-t214880 exited 0. Old selection ISEs reproduced, selecting
offset 15 on both roots. The complete budget was 403,200 steps, 8,064 calls,
297.92 CPU seconds and zero fits. Only 5,496 final summary bytes returned.

| ISE improvement (positive is better) | Root 1972210000 | Root 1972220000 |
| --- | ---: | ---: |
| Current vs lagged waypoint, whole episode | +12.30% | +6.02% |
| Current vs lagged waypoint, latter halves of options | +10.88% | +4.85% |
| Event aligned vs selected best fixed offset | -0.22% | +1.99% |
| Event aligned vs rotated offsets | +0.30% | +1.41% |
| Selection-best vs selection-worst fixed offset | +0.77% | -2.23% |

Both waypoint contrasts pass on both roots. For whole-episode waypoint gain,
the within-root nominal 95% intervals are [0.34391, 0.87789] and
[0.09163, 0.51306] in normalized ISE units. No timing contrast passes;
each timing interval crosses zero. On the second root, the old selection
worst clock has lower new-path ISE than the selection best at the point
estimate. Event alignment moves about four calls per episode versus the
clock and seven to eight versus the rotation, so the schedules are not
identical.

The frozen lower controllers benefit from useful, timely waypoints throughout
an option. The tested event-alignment rule does not demonstrate reliable
extra full-episode timing value. Keep Reacher as a reported transfer boundary:
no critic restart, no formal transfer, and no change to either prior HOLD.
This does not show that all adaptive timing rules lack value.
