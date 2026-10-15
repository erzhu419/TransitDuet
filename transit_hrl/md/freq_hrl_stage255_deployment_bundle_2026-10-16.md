# Stage255: Fixed-Past Deployment Bundle

Stage254 found both validation optimism and an incremental-task mismatch between
prefix-refit OOF and full deployment. Compare the existing prefix-refit control
with inner learning conditional on the frozen full past bundle: reference,
feature coordinates, full TSVD basis, historical initialization, scale and RMS cap.
Keep the GLS/shrinkage losses, candidate grid and selector unchanged, including
historical fold score scales. Read both full model families from Stage251 verbatim.

Use the same prefix-excluded eight training futures per inner fit and new2/new8
validation from the other training-bank half. Old A2 cannot validate this arm:
the past bundle saw those labels. Freeze every full action before opposite-bank
evaluation. Report both banks, four views, seen/held outcomes and the
validation / inner independent / full independent bridge, separately for each
family, selector and filter. Raw-action and predicted-wait gate disagreement must
be zero for the aligned bundle. Reproduce the cached control actions/outcomes.

One successful scheduler preflight, then 31 remaining single-core fit tasks and
one report, unbound on node001-node006. 10,240 small inner solves; zero full/raw
fits, native episodes or actor updates. Pull scalar reports/receipts only.

Limitations: this is incremental cross-fitting conditional on a pretrained
bundle, not OOF validation of the entire learner. It changes the whole bundle,
not just the reference. Inner/full still differ in training prefixes and future
count. Stage249 is reused development data, with no fresh confirmation or seed CI.
Stage253/248 outcomes and Stage169 reserves are not read. No default change.

## Results And Next

All 33 scheduler records completed exit 0, zero retries (node004/005/006).
10,240 inner solves; 69,120 control action-state exposures and all control metrics
reproduced exactly. Six focused tests passed. Only a 1,807,585-byte scalar report
and the small preflight receipt were pulled; no raw data or models.

Mixed held-prefix new8 / pooled-safe development means relative to full raw:

| Family | Prefix-Refit Cost / Wait (min) | Fixed-Bundle Cost / Wait (min) |
|---|---:|---:|
| free_old | -0.000336602 / -0.000546061 | -0.000013383 / +0.000004069 |
| shrunk | -0.000278643 / -0.000405599 | -0.000056902 / -0.000142057 |

Aligned raw/gate disagreement is exactly zero. Seen-prefix wait correction-transfer
gaps shrink from +0.000275993/+0.000445508 to +0.000045964/+0.000033333,
but validation utility and admission also shrink: new8 admits 14->6 / 15->7 of
32 mixed bank-cell exposures. Smaller gaps do not establish better performance.
New2 improves pooled held cost/wait for both families, but remains an exploratory
mean, not grounds to select it after seeing these outcomes. Results vary by view.

Next: use cached per-cell keys/outcomes to separate lost admission from changed
key quality. Then compare point and uncertainty-aware selection on both frozen
bundles; do not assume more conservative filtering will recover rejected gains.
