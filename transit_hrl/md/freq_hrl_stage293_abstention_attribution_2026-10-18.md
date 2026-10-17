# Stage293: inference bridge and prefix attribution

Stage292 reproduced waiting gains but failed the registered joint criterion.
Cost versus matched thinning passes conditional MC, not prefix uncertainty.
Stage291 also used rematched thinning in bootstrap, unlike Stage292's frozen
weights: compare the inference conventions before attributing the CI change.

Reuse old8/new64 panels and frozen actions. Cross both future banks with fixed
and screen-only rematched bootstrap on identical 5,000 prefix resamples (seed292).
Reproduce published means and Stage292 fixed CIs; alternate intervals cannot
replace the registered primary decision. Describe root/regime/prefix contributions,
leave-one-prefix influence and MC/latent-prefix variance without pruning states.

One scheduler CPU/3 GiB analysis; no fits, simulations, held rows or new policies.
Only the small scalar report is pulled. Raw prefix records remain on the server.
The entrypoint reads frozen JSON/NumPy schemas directly, avoiding the historical
native-stage recursive import chain. 22 focused/regression tests and two cold
CLI startup checks passed.

Run: `native_transit_abstention_attribution_stage293_20261018_r1`.

## Result and next decision

Task t175903/node002 exited zero; all frozen means and Stage292 fixed CIs
reproduced. Pulled 22,646 scalar-report bytes; no new fits or simulations.
Cost versus matched thinning, multiplied by 1e6:

| Bank | Mean | Fixed prefix CI95 | Rematched prefix CI95 |
| --- | --- | --- | --- |
| Old8 | -193.09 | [-387.02, -11.65] | [-238.94, -16.57] |
| New64 | -101.70 | [-231.36, +32.30] | [-145.73, +19.12] |

The loss of prefix support is not explained by the bootstrap switch: old8
passes under both conventions, new64 under neither. Fleet accounts for 85.3%
of the point-cost advantage reduction. Latent prefix variation accounts for
72.4% of analytic cost variance versus thinning (57.0% versus raw), descriptively.
The largest waiting-gain prefix, 397/hour_burst/2, provides 47.0% of wait gain
versus raw but increases cost; it provides 91.5% of extra wait gain versus
thinning. Both hour-burst groups improve waiting but increase total cost.
Removing any single prefix leaves pooled cost points negative: versus raw
[-217.85, -106.04], versus thinning [-140.39, -63.22]. This is limited positive
evidence, not a one-prefix sign artifact or a passed joint claim.

Next budget: genuinely new prefix coverage with the frozen four controls and
all five regimes retained. No more futures/head/threshold searches on this cohort.
No additional sampling was dispatched in this diagnostic stage.

Limitations: post-confirmation diagnosis, not independent confirmation or new-prefix
transfer. Variance subtraction is descriptive; signed estimates can be negative.
