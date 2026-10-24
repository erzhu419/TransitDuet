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

Server verification pending. No real archive scan, collection, replay, training,
online selector or shadow action execution is authorized. V238/main are unchanged.

Next: after synthetic acceptance, obtain separate approval for c0/V107 real
passive equality preflight. Upper-only still requires every first-path gate to pass.
Synthetic equality does not establish real-path equality or controller efficacy.
