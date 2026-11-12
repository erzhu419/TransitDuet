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
