# V260: frozen S-bank collection

2026-10-24: the user's "continue next step" authorizes S collection after
V259 synthetic acceptance t211483. Collector source357e60fd70 and the
V253/V254 protocol/manifest remain unchanged; no smoke-based tuning.

Seal the exact source-bound selection_authorization.json: 32 fixed units,
at most7 paths each/224 total, at most8 workers. Native first, all legal
nonnative bins retained; aliases and masked bins are not physical starts.
Failures consume slots, with no automatic retry, substitution or extra path.

First dispatch c0-c3/S0 through scheduler/high to the server. Accept their
full-unit prefix, future and actual-execution checks before dispatching the
remaining28 units. Preserve every actual start and failure server-side.
Only small status/summary files return locally; raw records remain remote.

After all32 units, aggregate S before separately sealing offline contrasts.
E remains closed; no training, online selector or main/V238 policy change.
