# V259: S-bank collector, synthetic admission only

Frozen V253/V254: 32 S units (4 contexts x 8 futures), at most7 physical paths
per unit and224 total. Native first, then every legal nonalias bin in bin order;
no forced-native, masked replacement, extra reference or outcome-based filtering.
All E roots remain metadata-only and have no collection entrypoint here.

Reuse V257's six-observer/V255 chain, prefix snapshot, actual execution matcher,
finish and full branch payload. Bind actor/day/prefix sampler/root to each sealed
S unit without changing V257 or V238. S0/native supplies each context's prefix
for later S futures; c0/S0 uses the accepted smoke pre-command prefix. Only the
pre-command prefix is shared across roots, never their final tape ID/outcomes.

Retain all legal outcomes plus frozen R action differences, actual holding
direct/later-same-cycle/other-cycle closure and FIRST actual same-bus next-launch
accounting. Missing or changed obligations stay explicit; they never select
anchors, drop outcomes or become zero benefit. No offline action choice yet.

This stage runs server synthetic tests only. The real entrypoint requires a
separate source-bound selection_authorization.json; this stage does not write it.
Every claimed path burns its slot before runtime construction, including failure;
no retries or unit/root replacement. Scheduler high priority, manual server
dispatch, small summaries only. Main/V238, S/E starts and training stay unchanged.

Server acceptance: t211483, source357e60fd70, node004/high/manual dispatch;
14 tests passed,0 failures/errors,backend exit0. The formal collection directory
and authorization file are absent; real S/E starts remain0. Tests cover all four
contexts, cross-S prefixes, disjoint derived streams, legal execution and224-slot
accounting; incomplete banks cannot seal contrasts or open E.
Earlier t211481 also passed14 tests/exit0 but its expected negative-test traceback
triggered scheduler failure scanning. Only that fixture now captures its expected
logs and verifies the saved failure; production error reporting is unchanged.
Next: separate explicit S collection authorization; E stays closed until all S
units and the locked contrasts are sealed. No smoke-based tuning or policy change.
