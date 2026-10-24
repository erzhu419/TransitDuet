# V256: real original-future passive/no-op acceptance

Actor12037,day2151001,sampler402112137,c0/V107. Reuse V249's six accepted
reference files and frozen runtime/observers;append V255 `9f4afff2f3` only.
Dispatch observer-only first;its complete exact pass admits native no-op.

Initial source `fb30e41bae`:interface task `t211437` passed14 tests;observer
`t211438` failed before runner construction because nested output misplaced
the frozen runtime proof lookup. That failed start and original records remain.
The user explicitly approved ONE extra observer start,not a free retry.

Source `aa8616bcfb` uses sibling outputs under the original results parent.
Repair integration `t211439` passed16 tests. Separate repair authorization and
integration records preserve old ones. Budget:original2 + approved observer1 =3.
No automatic requeue,extra references,context/future substitution or more starts.

Observer repair `t211440` passed;only then native no-op `t211445` was submitted.
Both backend exit sentinels are0. Both preserve all six old payloads,actions,
timing,endpoints,native calls and RNG exactly;new records are complete.
Each has5,200 queries,5,240 PREPAREs and40 unused PREPAREs. Same first anchor:
7218s,query index661/event1487,cycle[11,35,1261],native bin1=5s,
PREPARE/query H=15s. Observer records0 commands;no-op records1 unchanged command.
Both have0 action changes and0 future switches. All tasks use high/manual,node004.

Final:3/3 starts consumed,including1 startup failure;0 remaining. No new future,
smoke,formal matrix,training or policy/V238 change. Raw data stay remote;local
receipt:`freqduet/results_freqduet/protocol_v6_v256_passive_pulse/summary/receipt.json`.

Next:separate approval before any reserved smoke or S/E bank. This proves
passive/native-no-op equivalence,not control benefit or a deployable selector.
