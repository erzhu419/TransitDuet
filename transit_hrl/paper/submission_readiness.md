# Plan-Validity Paper: Submission Readiness

Date: 2026-11-12. This is the manuscript decision, not an experiment release.

## What the paper can claim

The [main result](manuscript.md#51-plan-validity-timing-improved-held-out-pointmaze-control)
is improved PointMaze episode tracking and return at exactly 24 planner calls
with shared controller weights and eight independent optimizer roots.
On a different set of eight frozen builds, the
[timing portfolio comparison](../md/freq_hrl_stage9_timing_portfolio_2026-11-12.md#result)
also supports better tracking against a selected distance threshold and a
learned-termination/best-clock portfolio under shared measured compute caps.
The subsequent [matched feature study](../md/freq_hrl_stage9_scale_ablation_2026-11-12.md#completed-result)
supports short history, not the tested longer target window or a general
benefit from frequency separation. These path tests reuse the eight follow-up
controllers and do not add independent training replications.

## What bounds the claim

- **Timing baselines.** The [all-learned comparison](../md/freq_hrl_stage9_strong_timing_2026-11-11.md#build-outcome)
  remains on HOLD before test access: one of eight termination heads missed
  best-clock competence. No root was dropped. The portfolio compares against
  the selection rule with a clock fallback, not a learned-only comparator,
  a named-method reproduction, or methods with their own training schedules.
- **Same-algorithm transfer.** In [Reacher](../md/freq_hrl_reacher_timing_applicability_2026-11-12.md#completed-result),
  fresh waypoints help, but the tested event alignment does not establish a
  timing gain. In [cart-pole](../md/freq_hrl_cartpole_timing_applicability_2026-11-12.md#completed-result),
  visible-command timing helps, but waypoint freshness depends on the feedback
  configuration, as the [crossed comparison](../md/freq_hrl_cartpole_waypoint_factorial_2026-11-12.md#completed-result)
  shows. Neither qualifies as transfer confirmation. The two-task C1/C2 HOLD
  is unchanged; neither a Reacher critic search nor another cart-pole gain or
  window sweep is needed to report this result.
- **Total compute.** The portfolio charged supervision, training, selection
  and deployment under common CPU, offline-step and latency caps. This is
  stronger than matching call counts, but does not equalize realized cost or
  establish performance across budgets. Simulator event metadata still helps
  select training queries; the deployed signal itself is causal.
- **Episode reliability.** PointMaze establishes mean episode performance, not
  a general tail-risk guarantee. [Stage304](../md/freq_hrl_stage304_closed_loop_and_paper_evidence_2026-10-21.md)
  did not confirm cost/wait improvement, statistical half-base wait retention,
  or extra cost benefit from the joint selector. Its Transit action search is
  not Stage9, and differences in profiles, callbacks and scenes preclude a
  causal attribution to repetition alone. The one-call-per-bin guarantee is
  exact; local branch value does not guarantee episode benefit or fleet safety.

## Next submission work

Close the draft around fixed-budget renewal timing on PointMaze. Present the
two task screens as applicability evidence and Transit as a separate test of
local-to-episode reliability. Freq-HRL targets hierarchies with useful slow
plans and fast execution; it does not compete with PPO or SAC as a learning
backend. The [source map](evidence_map.md) and [figure audit](figure_evidence_audit.md)
identify the compact sources for the main claims. The remaining package work
is venue formatting, an artifact manifest, and a reproducible code/data release.

A stronger conference claim would need a prospectively qualified second task
using the same renewal target, fit and call contract, plus a competent timing
baseline comparison with full costs. The
[unified preregistration draft](../md/freq_hrl_unified_preregistration_2026-10-22.md)
and [episode reliability protocol](../md/freq_hrl_episode_reliability_protocol_2026-10-22.md)
retain the prepared paths and their conditions; they are not completed evidence.
No new run is released.
