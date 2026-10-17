# Stage284: Frozen State-Level Calibration Versus Action Contrasts

- Run: `native_transit_service_centering_stage284_20261017_r1`; parent Stage283 completed `t166182`.
- For all three cached maps, center the uncut correction over the raw-feasible nonoriginal actions of each query state. Compare level-only, zero-mean contrast-only, and saved uniform offset plus contrasts. Retain full/offset/slopes controls and all three frozen risks. Original-action correction stays zero; use the same training radius after decomposition. No label-dependent calibration or selected winner.
- Freeze every action before outcome access. Replay all 42 Stage283 choices and every old cell outcome/service metric, plus all compact group metrics/outcomes. Evaluate the same 32 cells / 2048 queries / 84,480 repeated exposures, train/retained/excluded/full60-held/prefix54-held, assigned/opposite16 and fixed8+8; primary mixed excluded/opposite and held/opposite.
- For every service, compute `MSE = state-level error MSE + within-state contrast error MSE` with the unchanged state-balanced eligible-action denominator. This is different from squared pooled bias. Compute pair inversion within nonoriginal eligible actions, state-balanced; omit empirical truth ties and count predicted ties as half errors. Report eligible and rank-state counts, all views/banks and evaluation halves.
- Report cost/fleet/service/wait and original/other switch contributions, relative to each own frozen-risk control as well as raw. Clipped component predictions are not additive; uncut contrasts preserve nonoriginal ranking but clipping can change it.
- One scheduler CPU / 3072 MiB, any node001-node006, no pin. Reuse server caches; zero new fits, raw/risk training, episodes or actor updates. Only scalar report returns locally.
- Verification: four new tests plus five Stage283 regression tests passed (6.91 s), covering prediction-only decomposition/anchors/feasibility/clip, exact MSE and ranking denominators/ties, full cell/group replay without fits or early labels, and completed-parent/unpinned scheduler admission. Both cold CLI startups passed.
- Completed: `t166209`, source `661b701b10`, node006, exit 0, no retries, 206.73 s. All 32 cells / 84,480 repeated exposures qualified; old cell/group replay error zero, maximum identity/partition error `4.45e-16`. Scalar report 7,229,561 bytes, compressed transfer 1,111,181 bytes; no models, predictions or query ledgers pulled.

## Results And Next Step

Observed-context full clipped service versus raw, mixed pooled/opposite16. Level/contrast errors and cost deltas multiplied by `1e6`; rank is eligible nonoriginal pair inversion percent, not total-cost ranking. Cost/wait use unchanged raw risk and are relative to raw.

| Population / model | Level MSE | Contrast MSE | Rank % | Cost delta | Wait delta / min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train / raw | 56.302 | 15.189 | 46.076 | 0 | 0 |
| Train / full | 40.644 | 13.962 | 45.111 | -126.581 | -0.000679460 |
| Prefix excluded / raw | 56.302 | 15.189 | 46.076 | 0 | 0 |
| Prefix excluded / full | 58.774 | 16.059 | 47.591 | -0.292 | +0.000202051 |
| Outer held / raw | 87.312 | 24.534 | 48.147 | 0 | 0 |
| Outer held / full | 88.037 | 23.776 | 47.687 | -143.179 | -0.000499837 |

- Training gain is 92.7% level-error reduction. Prefix excess MSE is 74.0% level and 26.0% contrast; both are worse in both fixed evaluation halves. Thus state-level calibration transfer is the larger MSE failure, but within-state contrast/ranking also fails on excluded prefixes. All three full clipped maps worsen both excluded MSE components and pooled inversion rate; the observable map is not uniformly worse than the action-derived controls.
- Keeping only the predicted state level restores raw ranking exactly but yields excluded cost `+0.054e-6`, wait `-0.000025423` min, service `+5.479e-6`; it does not recover the uniform-offset result (`-104.535e-6`, `-0.000356641` min). Centered-only excluded correction yields cost `+35.529e-6`, wait `+0.000199870` min; offset-plus-centered yields `-34.565e-6`, `-0.000105143` min, still weaker than offset alone.
- Outer contrast transfers modestly in the pooled mean: full contrast MSE `-0.758e-6` and inversion `-0.460` percentage points, offsetting level MSE `+0.725e-6`. Centered-only improves contrast MSE but increases cost `+44.853e-6` (fleet `+46.115e-6`), with wait only `-0.000021257` min. Its 15 other-to-other switches help, but original/other switches reverse the cost benefit. Better service ranking alone does not solve threshold/risk control.
- The result survives the training-size comparison: prefix54 heads on the same outer states have full level/contrast MSE `89.008/23.883e-6` versus raw `87.312/24.534e-6`; centered-only total MSE `111.277e-6` but cost only `-13.407e-6`. Assigned training full errors are `35.719/13.633e-6`, rank `42.921%`, versus opposite training `40.644/13.962e-6`, `45.111%`; future-label noise and prefix transfer are distinct effects.
- No stable winner: excluded level-only cost flips between evaluation halves; mixed-b held full/centered costs are positive in both banks, and held pair ranking improvement is not present in both halves. State-level squared error is not squared pooled bias; service inversion ignores risk and pair margin. Neither component alone closes the control claim. Defaults unchanged.
- Next Stage285 fits the eligible-action mean service residual directly: two 9-parameter state-level heads, causal observed context versus action-mean coordinates, with feature-only matched effective df. Test each level alone and with all three frozen action-contrast heads. This tests whether a direct pure-state level representation transfers better than deriving the level from action interactions; 1408 small fits, no raw/risk/contrast training or new simulation.
- Limitations: reused-development frozen single-intervention analysis; evaluation-mean ranking is a noisy MC diagnostic, not true/oracle ordering or a new-prefix/seed-CI/full-HRL validation. Stage253/248/169 stay closed; defaults unchanged.
