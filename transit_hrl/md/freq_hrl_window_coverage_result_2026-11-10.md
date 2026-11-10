# W Coverage Result

All four main tasks t214228-t214231 exited 0. Frozen W reproduction and 96-pair
budgets passed. Main cost: 5,570.881 CPU seconds, 6,681,600 simulator steps;
133,983 public bytes retrieved, no controller/termination updates. Decision: HOLD.

## Complete Episodes

Mean ISE on eight new paths/root; lower is better:

| Task | Dense DF2 | Dense DF8 | Wide DF2 | Wide DF8 | History | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointMaze | 1.098746 | 1.147743 | 1.099215 | 1.067007 | 0.992463 | 1.024637 |
| Reacher | 4.664463 | 4.601352 | 4.523516 | 4.530526 | 4.510577 | 4.453356 |

Reacher coverage gains are 3.02%/1.54% at DF2/DF8. DF2 improves both roots and
both episode halves; all eight per-root coverage 95% path CIs still cross zero.
PointMaze DF8's 7.03% pooled gain is driven by root1 and reverses between pooled
episode halves. No new arm beats pooled history or threshold in either task.
Reacher wide DF2 is 0.29% worse than history; PointMaze wide DF8 is 7.51% worse.

On common held states, wide DF8 reduces W MSE by 25.6%/7.5% on PointMaze and
46.5%/14.7% on Reacher. Reacher dense DF2 predicts better than dense DF8 but
has worse episode ISE on both roots. Lower complexity is not a general repair.

## Mechanism And Next Step

Cached attribution added no fit/simulation (30,223 derived bytes). Reacher
root2's DF2 coverage contrast has W gains +0.00456/+0.00254, G gains
-0.01343/-0.01702, yet episode ISE difference -0.15402. Single-intervention
credit under old history does not directly predict repeated-policy episode gain.

The [sampler](../freq_hrl/experiments/plan_validity_window_coverage.py#L28) labels
only bin starts; the [executor](../freq_hrl/experiments/plan_validity_transfer.py#L133)
also checks offsets 5/10/15/20. The [legacy head](../freq_hrl/experiments/pointmaze_timing_baselines.py#L54)
ignores supplied offset/remaining fractions. Wider prefix coverage does not
fill this decision-phase gap; it is not yet proven to be the sole cause.

Next: match budgets for bin-start/all-reachable-check coverage and absent/present
clock conditioning; fix controllers, W target and controls, reserve another held
set. Then test continuation consistency. No new experiment jobs were submitted.

Limits: two roots/task; descriptive unadjusted within-root CIs; cached effects
are not additive policy gains. Coverage trades prefixes against future averaging,
not equal CPU. Cross-cohort MSE changes are not paired improvements. Termination
competence and unknown interrupted costs still block confirmation.

Sources: [frozen summary](../results/plan_validity_window_coverage_20261110/summary.json),
[cache readout](../results/plan_validity_window_coverage_20261110/cache_readout.json),
[protocol](freq_hrl_window_coverage_2026-11-10.md).
