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
