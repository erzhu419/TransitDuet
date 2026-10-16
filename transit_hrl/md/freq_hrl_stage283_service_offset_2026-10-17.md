# Stage283: Frozen Service Offset Versus Slopes

- Run: `native_transit_service_offset_stage283_20261017_r1`.
- Reuse Stage282 coefficients/predictions. Split each uncut service correction into its fitted intercept on nonoriginal actions and the remaining feature slopes; both original-action corrections stay zero. No geometry, normalization, penalty, radius or coefficient changes.
- Evaluate offset-only and slopes-only using the same training-only service clip, all three frozen risks and unchanged feasibility. Replay all 24 old controllers; report the old clipped/MC controls plus 18 new ablations. Freeze actions before evaluation.
- Attribute uncut MSE change to offset/slopes energies and all three signed cross terms; preserve bias contributions and assigned/opposite16 plus fixed8+8. Clipped components are not assumed additive. This is a fixed-parameter contrast, not a uniquely identified causal explanation.
- Partition each controller's change from its own unchanged-risk control into unchanged, original-to-other, other-to-original and other-to-other. Report cost/fleet/service/wait contributions with the full-state denominator, including zero-action-change states.
- Same 32 cells / 2048 queries / 84480 repeated exposures; train, retained, excluded, full-held and prefix-head-held, every view/bank. Primary mixed excluded/opposite and held/opposite; compact switch detail for those two populations.
- One scheduler CPU / 3072 MiB, any node001-node006 without pin. Zero new fits, raw/risk training, simulations or actor updates; full query ledgers stay remote, scalar report only.
- Verification: five tests passed for frozen split/anchors/clip nonadditivity, signed MSE terms and denominators, all four switch types, complete cell/group replay with no fits or early outcome access, and completed-parent scheduler gate. Both cold CLI startups passed.
- Completed: `t166182`, source `9061d35100`, node004, exit 0, no retries, 113.26 s. All 32 cells / 84,480 repeated exposures qualified. Maximum replay/identity/partition error `4.45e-16`. Local scalar JSON 3,508,025 bytes, compressed transfer 460,373 bytes; no model or query ledger pulled.

## Results

Mixed pooled observed-context service, unchanged raw risk. MSE and cost multiplied by `1e6`; waiting in minutes. Cost/wait are relative to raw, not to another learned-risk controller. Each tuple is `(service MSE, cost delta, wait delta)`.

| Population | Raw MSE | Full clipped | Offset-only clipped | Slopes-only clipped |
| --- | ---: | --- | --- | --- |
| Train / assigned16 | 71.492 | (49.352, -146.396, -0.000923698) | (66.263, -71.233, -0.000428678) | (53.313, +23.935, -0.000314453) |
| Train / opposite16 | 71.492 | (54.607, -126.581, -0.000679460) | (66.784, -81.664, -0.000388900) | (58.325, -2.747, -0.000150814) |
| Prefix excluded / opposite16 | 71.492 | (74.834, -0.292, +0.000202051) | (70.693, -104.535, -0.000356641) | (76.042, +121.207, +0.000521159) |
| Outer held / opposite16 | 111.846 | (111.813, -143.179, -0.000499837) | (113.863, -133.505, -0.000435124) | (110.892, -53.283, -0.000590430) |

- Prefix transfer is not principally a harmful uniform offset: uncut offset changes MSE by `-0.799e-6`, slopes by `+6.417e-6`, interaction by `-0.276e-6`. Slopes' raw-error cross term shrinks from `-20.352e-6` train/opposite to `-0.998e-6` excluded, while their energy rises `5.850e-6 -> 7.415e-6`. Lost residual alignment, not merely correction amplitude, explains the failure. All three maps' excluded slopes-only MSE exceeds raw; all their uniform offsets coincide because training normalization centers the slopes.
- Outer transfer differs: offset changes uncut MSE by `+2.017e-6`, slopes by `-1.742e-6`, interaction by `-1.218e-6`. Removing offset improves held MSE, but does not give the best cost. Prefix54 heads on the same held states have full/slopes-only MSE `112.891/111.926e-6`, versus raw `111.846e-6`; the full60 result must not be mistaken for a training-size-matched comparison.
- Under raw risk, excluded full service creates 28 other-to-other switches contributing `+62.028e-6` cost and `+0.000300260` min wait, cancelling improvements from original-to-other switches. Offset makes no other-to-other switches. Slopes-only also creates 39 other-to-original switches with `+93.718e-6` cost and `+0.000322168` min wait. Both threshold and ranking changes matter.
- These are not stable cost wins: excluded offset cost is negative in the first evaluation half and positive in the second; mixed-b held offset cost is positive in both banks, and its bank1 wait is positive. Full service also harms mixed-b held cost in both banks. Excluded offset wait improves in both halves and all four mixed-view/bank cells. No controller is promoted.
- Next: Stage284 separates each frozen correction into its causal query-state eligible-action mean and zero-mean action contrasts; compares level-only, contrast-only and cached-offset-plus-contrast, with exact level/contrast MSE and empirical pair-ranking diagnostics. This resolves the remaining ambiguity: feature slopes contain state-level calibration as well as action ranking. Zero new fits or simulations.
- Limitations: fixed-parameter, reused-development single-intervention diagnosis, not unique causal attribution, new-prefix confirmation, seed CI or full HRL validation. MC remains an information benchmark. Stage253/248/169 stay closed; defaults unchanged.
