# Plan-Validity Paper: Submission Readiness

Date: 2026-11-20. [Manuscript](manuscript.md) | [Review PDF](submission.pdf) |
[Reproduction instructions](README.md).

## Supported claims

- **Original PointMaze confirmation.** Eight independent optimizer roots pass
  all five registered gates. At 24 calls, the timing rule reduces ISE by
  34.94% relative to periodic planning. Follow-ups on eight separate frozen
  builds favor the rule over distance and a timing portfolio under shared caps.
- **Clock-grid queries.** Training without hidden event annotations beats
  periodic, random-offset and distance timing on held-out PointMaze paths.
  Ridge also beats two learned timing gates and the raw-input MLP at the same
  192 paired queries. The acquisition study did not establish noninferiority
  to event stratification; the predictor comparison did not isolate features.
- **Second plant.** After retraining on derived polygonal Reacher, selected
  fixed-clock ISE improved 5.00% across eight roots (nominal p = 0.00937;
  Holm p = 0.0656 across seven ridge timing contrasts). The driver reuses
  PointMaze's disturbance mechanism. Adaptive gate contrasts are mixed.
- **Call counts.** The matched-count ridge benefits do not carry over to
  every cross-budget comparison: 20- and 24-call ridge schedules beat the
  24- and 30-call periodic schedules, but lose to the respective distance
  gates at those higher counts.
- **Compute trade-off.** Two CEM comparisons improve ISE with fewer calls
  and less planner time, but scoring more than offsets the planner savings.
  Against distance gates with more calls, fewer-call ridge is less accurate
  and costs more upper-loop time at both tested points. The measured control
  times do not establish a net speedup.
- **Waiting value.** Earlier development isolates opposite-signed episode
  gains despite positive 50-step gains. A next-check waiting target did not
  displace the frozen trigger.
- **Scope.** The smooth Reacher and cart-pole screens separate waypoint
  freshness from timing opportunity. Transit Stage304 remains a negative
  episode result for a different action selector.
- **Timing mechanism.** Online timing beats the yoked-clock replay by
  37.42%, with a positive adjusted interval and both path halves favorable.
  Pairing versus matched-step single-arm supervision is inconclusive after
  adjustment despite a favorable 3.29% point estimate.

## Before submission

The learned binary and cost-difference gates are budgeted adaptations, not
reproductions of the original TEMPLE or Option-Critic algorithms. Retain the
fixed-call timing claim on the tested planning and execution hierarchies.
The [TMLR submission build](tmlr/README.md) uses the official anonymous
style and packages reproducible analysis as an anonymous ZIP, not native
simulation or controller training. The authors complete the private
OpenReview declarations before upload. A stronger
cross-task claim needs an independent comparison with a selected strong
timing gate under matched total compute, not more periodic controls.
