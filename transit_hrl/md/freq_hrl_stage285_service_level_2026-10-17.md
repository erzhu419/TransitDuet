# Stage285: Directly Learned State-Level Service Calibration

- Run `native_transit_service_level_stage285_20261017_r1`; qualified parent Stage284 `t166209`.
- Fit only the training-assigned16 eligible-nonoriginal action mean of the service residual. Retain full60/prefix54 indices, causal cached Linear9/basis/context geometry, and all existing query populations. No held/opposite labels enter fits, normalization, penalty, radius or choices.
- Compare two 9-parameter level heads: eligible-action mean of frozen Linear9 coordinates versus eight pure observed-context tanh coordinates. Normalize only their training-state features. Match both effective df to the smaller feature-only df at ridge `0.01` by increasing penalties, without labels or held selection. Reproduce the old training intercept from the mean target.
- Cross each learned level with no action contrast and with each of the three frozen uncut centered action contrasts. Add first, then apply the unchanged Stage282 radius; original correction zero. Reuse all three risks, raw gate/candidates/ties. Freeze all choices before evaluation; retain uniform offset, derived level, centered/full and MC controls.
- Same 32 cells / 704 contexts / 2048 queries / 84,480 repeated exposures. Primary mixed excluded/opposite and full60-held/opposite; report training, retained, prefix54-held, all views/banks and fixed8+8. Compare level/contrast MSE, empirical pair ranking, cost/fleet/service/wait and original/other switch contributions; no selected winner.
- Replay all 69 old choices and cell metrics/outcomes/geometry plus every compact parent group value. Compact the new report only after replay; full predictions/models/outcome ledgers remain on server.
- Budget: 1408 small 9-parameter fits; zero new raw/risk/basis/contrast fits, episodes or actor updates. One scheduler CPU / 3072 MiB, any node001-node006, no pin. Pull only scalar summaries.
- Verification: four tests passed (6.93 s): matched feature-only df/target means, frozen contrasts/anchors/clip, full cell/group replay with assigned-only fit indices/futures and no raw/contrast refit, and completed-parent/unpinned scheduler admission. Both cold CLI startups passed.
- Completed: `t166265`, source `32e908dbc3`, node004, exit 0, no retries, 262.20 s; all 1408 head fits and 32 cells qualified. Effective-df mismatch `3.96e-12`, old cell/group replay error zero. Scalar report 5,974,145 bytes, compressed transfer 928,761 bytes; no models or query ledgers pulled.

## Results

Mixed pooled, opposite16, unchanged raw risk. State-level MSE and cost multiplied by `1e6`; cost/wait relative to raw.

| Population | Level head | Level MSE | Cost delta | Wait delta / min |
| --- | --- | ---: | ---: | ---: |
| Train | Uniform | 51.595 | -81.664 | -0.000388900 |
| Train | Action mean | 43.353 | -301.520 | -0.001240690 |
| Train | Observed state | 44.902 | -273.804 | -0.001235482 |
| Prefix excluded | Uniform | 55.504 | -104.535 | -0.000356641 |
| Prefix excluded | Action mean | 66.298 | -146.094 | -0.000363086 |
| Prefix excluded | Observed state | 68.737 | -201.328 | -0.000740983 |
| Outer held | Uniform | 89.329 | -133.505 | -0.000435124 |
| Outer held | Action mean | 92.236 | -182.904 | -0.000907324 |
| Outer held | Observed state | 91.979 | -257.069 | -0.000762337 |

- Matched df does not establish superior calibration: observed-state MSE is worse than action mean in training/prefix exclusion, marginally better on outer held, and worse than uniform on both held populations. Assigned-training level MSE is uniform/action/observed `51.073/39.118/39.602e-6`; the training-size-matched outer values are `89.568/94.408/93.706e-6`. Level-only service ranking stays raw (`46.076%` training/excluded; `48.147%` held).
- Nevertheless observed level-only improves pooled cost/wait over uniform in all three risks on excluded and outer held. Against action mean, it improves both excluded metrics, but trades better held cost for worse held wait. These are control point estimates, not a calibration or representation dominance claim.
- Adding the observed frozen contrast weakens excluded cost/wait to `-133.275e-6 / -0.000647168` and worsens inversion to `47.157%`. Outer cost/wait improve to `-275.123e-6 / -0.000910612`, inversion `47.890%`. All three contrast additions weaken excluded cost/wait under raw risk; outer action-context contrast has stronger cost but not uniformly stronger waiting/ranking. No universal benefit from adding contrasts.
- Relative to uniform, excluded observed-level cost is worse in the first8 (`+30.766e-6`) and better in the last8 (`-224.353e-6`); outer both halves improve. Mixed-b held costs remain positive in both banks. Two held level-only other-to-other records have zero cost/fleet/service/wait contribution. No controller is promoted.
- Next Stage286 freezes these policies: compare error at the fixed best nonoriginal candidate with global error; attribute paired improvements to prefixes and fleet events. Collapse repeated model/view/fold uses of each state-future before MC variance; report banks separately, never a pooled seed CI. No new fitting or simulation.
- Limitations: reused-development supervised single-intervention calibration, not unseen-prefix confirmation, training-seed CI, deployment or full HRL. MC is an information benchmark. Stage253/248/169 closed; defaults unchanged.
