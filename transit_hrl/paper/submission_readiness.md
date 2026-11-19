# Plan-Validity Paper: Submission Readiness

Date: 2026-11-19. [Manuscript](manuscript.md) | [Review PDF](submission.pdf) |
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
- **Second plant.** On derived polygonal Reacher, eight fresh roots support
  a 5.00% ISE improvement against the selected fixed clock. The distance,
  cost-difference and native-periodic comparisons are inconclusive.
- **Compute trade-off.** Two CEM comparisons improve ISE with fewer calls
  and less planner time, but scoring more than offsets the planner savings.
  The measured control-time intervals do not establish a net speedup.
- **Scope.** The smooth Reacher and cart-pole screens separate waypoint
  freshness from timing opportunity. Transit Stage304 remains a negative
  episode result for a different action selector.

## Before submission

The learned binary and cost-difference gates are budgeted adaptations, not
reproductions of the original TEMPLE or Option-Critic algorithms. Retain the
fixed-call timing claim on the tested planning and execution hierarchies.
The artifact reproduces analysis and presentation from saved evidence, not
native simulation or controller training. Apply the chosen venue's template,
page limits and anonymity requirements before uploading.
