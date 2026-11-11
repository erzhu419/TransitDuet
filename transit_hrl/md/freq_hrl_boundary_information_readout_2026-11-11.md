# Frozen Boundary Information E Readout

After t214694 froze all eight Dense/Wide DF-3 models, release only the reserved
new-prefix prediction readout. Keep the original four roots, C0, model geometry,
coefficients and all controls unchanged. Do not select a model from training MSE.

Each root uses its eight registered root+68000..68007 prefixes, three sampled
checks/prefix, four windows/check and eight suffix indices E20-27/window.
Reuse observe's outcome-independent 617/839 check selection and the 11401/11402
window/suffix recipes. All six predictors (zero, constant, causal_t, frozen base,
Dense, Wide) see the same windows and labels. E labels never affect predictions.
The acquisition storage key A_replicas is shared with the existing sampler;
these rows contain only E20-27 and are not training data.

Eight 1CPU/2GiB scheduler shards, four prefixes each: 926,400 steps/shard,
1,852,800/root and 7,411,200 total, including every factual and paired full
1,200-step replay. No fitting or extra precheck simulation. Private rows and
predictions remain remote; public progress and summaries alone are retrieved.

Tasks t214699-t214706 launched on node004/node005/node006 by unpinned scheduler
placement. All eight passed the global model-freeze gate and sampled their first
state: 78,000 counted steps each, 624,000 total, no reported errors at the launch
snapshot. The [launch receipt](../results/plan_validity_boundary_information_20261111/readout_release.json)
retains identities, permissions and progress; only 1,705 status bytes were pulled.

Report MSE for all six controls and paired contrasts base/zero, Dense/zero,
Wide/zero, Dense/base, Wide/base and Wide/Dense separately for each root.
For a contrast p versus q, use p^2-q^2-2(p-q)*mean(E); the shared squared-label
term cancels. Report conditional suffix MC SE, individual prefix contrasts and
descriptive prefix dispersion separately. Check E20-23/E24-27 and windows
0-1/2-3 halves. Prefixes/windows are not independent fitted-policy seeds.

## Limits And Next Step

The [release manifest](../protocols/plan_validity_boundary_information_readout_2026-11-11.json)
freezes the whole readout before E access. Twenty focused tests passed, including
global-freeze-before-native-work, label-independent predictions and budget closure.
Compare Dense and Wide on this common E bank to distinguish label information
from coverage; do not compare their different own-bank training MSE as a ranking.
Stop for failed provenance or budget closure and retain partial paid rows. Read
all four roots before another decision. This evaluates boundary information at b,
not deployable earlier-t credit, timing-policy gains or complete-episode Freq-HRL.
Original confirmation and performance claims remain on HOLD.
