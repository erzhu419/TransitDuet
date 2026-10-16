# Stage263: Matched Future Forcing

Reuse Stage249 labels and Stage260/262 models; no new fits, simulation or selection.
For each anchor bank/prefix, keep coefficients, basis, reliability, lambda and
the Stage262 prefix KKT inverse fixed. Reconstruct unshrunk target means instead
of exchanging reliability-multiplied cached targets.

Cross future-target bank and cached GLS-precision bank (2x2). At the same model
point/operator, target swap + precision swap + interaction = joint swap.
Anchor data response = zero-target design/prior + two-bank target mean + target
half-contrast; retain cross terms, full/low/rest blocks and both banks.

t165687: node006, exit 0, no retries, 50.90 seconds; seven tests passed.
32 cells / 1,280 primary records qualified. Cached target and Stage262 data RMS
reproduced exactly; decomposition residual <= 1.43e-14. Pulled 1,924,470-byte
scalar report plus a small posthoc concentration summary; raw ledgers stay remote.

Mixed removed: 12 bank-cells / 120 active prefix-models per family.
Below: function-response RMS, not outcome costs; swaps use the same KKT inverse.

| Region / model | Anchor data | Target swap | Precision swap | Anchor/other cosine |
|---|---:|---:|---:|---:|
| Excluded / free_old | .005370 | .015036 | .002271 | -.543 |
| Excluded / shrunk | .005695 | .017361 | .006036 | -.134 |
| Held / free_old | .001935 | .004620 | .001049 | -.122 |
| Held / shrunk | .001957 | .004169 | .001105 | -.027 |

Target changes exceed precision changes and persist outside the low block.
Mixed all-category held target/anchor RMS ratios are also 2.26 / 2.70.
Target-only held response cosines are .726 / .568: targets have shared direction,
but cancellation with the design/prior term leaves an unstable smaller response.
Posthoc: only 24/120 / 22/120 held response dots are negative. Grouping by removed
training prefix, 397/913970203 contributes 77.1% / 55.5% of held target-swap squared
response. The pooled pattern is concentrated, not universal direction reversal.

Limitations: conditional directional stability, not ground-truth signal/noise,
accuracy or utility evidence. Anchor model/key/calibration depend on its bank;
exposures are not seeds. Only cell-training labels are used from the server-side
archive; held causal features do not entail held-outcome evaluation.
Stage253/248/169 stay closed; no default/deployment change.

Next: cached service/risk target-swap attribution and 8+8 future-half stability
for all prefixes, highlighting the high-influence ones. Keep the same operator
and cap; distinguish fleet-tail labels from service errors before fitting or gates.
