# V248 independent passive PREPARE observer

Implements V247 `e74d381e62` without changing its mapping or two-path budget.
The separate sixth payload records launch baselines, an O(1) completed-departure
index, PREPARE counter samples, mapped components and exact query joins.
Native seq and V224 order remain separate; same-front identity persists across
stations. Later queries and physical-bus reuse cannot overwrite earlier samples.

Synthetic tests use actual V224/V230/V235/V241 observers with a synthetic callback
producer. They compare all five original payloads and check delayed/repeated
queries, plan changes, unused PREPAREs, missing front support, open/zero-query
cycles, hook returns, RNG/native-call invariance and detach restoration.

Source `b83f1c5db4`: scheduler high/manual task `t211400` on node004, exit 0,
**25 tests passed** (15 frozen mapping tests plus 10 callback-chain tests).
Both synthetic native arms preserve all five original payloads exactly.
No real archive scan, collection, replay, training, online selector or shadow
action execution occurred. V238/main and both reserved real-path slots are unchanged.

Next: after synthetic acceptance, obtain separate approval for c0/V107 real
passive equality preflight. Upper-only still requires every first-path gate to pass.
Synthetic equality does not establish real-path equality or controller efficacy.
