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

Limitations: post-confirmation diagnosis, not independent confirmation or new-prefix
transfer. Variance subtraction is descriptive; signed estimates can be negative.
