# V401: budget-conditioned upper implementation

The V350 law and integration were copied here; historical modules are unchanged.
Both training arms have 181 state channels. `explicit` observes rho in the final
channel; `implicit` receives zero there and still observes remaining allowances.
Retention belongs to each manager. The old quota, timetable, reward and physical
action semantics are retained. This is an explicit-budget information contrast,
not a claim that the implicit arm has no budget information.

Server check `t213963`, scheduler high/manual on node004: 9/9 contracts passed.
No physical path, native Runner construction, optimization or candidate training
was performed. See [the small receipt](review01/checks01/results.json).

The user approved the separate 90-path ceiling in `authorization.json`.
Collection, checkpoint updates and the file-gated scheduler workflow are now
implemented. Entry task `t213965` passed fresh checkpoint round-trips, replay
budget checks and 47 exact old-actor reference calls, with zero optimizer steps.
The corrected metadata03 task verifies/reserves the four development days.
The first two metadata tasks were stopped after unnecessary traversal of raw
S/E payloads and V75 native-run copies caused NFS waits; their receipts remain.
Producer metadata stays in scope. No traffic path or update was consumed.
Two old-policy engineering paths must pass before 64 matched training paths and
24 development paths. The two learners each receive 32 training days and 1,536
updates; neither receives extra training or warm-starts from V400. Detailed
assignments and stop conditions are frozen in [PROTOCOL.md](PROTOCOL.md) and
[protocol.json](protocol.json). Multi-seed/confirmation remains outside this budget.

Synthetic success does not establish runtime equivalence or a service benefit.
V400, its negative harmonic-journey gate and the main policy remain unchanged.
