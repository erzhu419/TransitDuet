# V250 saved compensation and supply audit

Only the two accepted V249 paths (actor 12037, day 2151001). Keep PREPARE/query
H and their clocks separate; close forward gap F to launch + H + nonholding
elapsed components. Retain every legal candidate relative to native, including
unknown rear evidence. Attach first-next same-bus labels only after the causal
state pass. V238 remains `b4b1d00203`; main is unchanged.

## Results
Source `5a72961382`, task `t211408`, node004, high/manual, one attempt, exit 0.
9 tests passed; one read per saved path; H/component accounts close. No replay,
fit, training, online selector or shadow execution. Full outputs remain remote.

V107 H: 40,215 s = 38,375 s cross-PREPARE accumulation + 1,840 s return tail.
675 intervals cross plan writes with positive prior H; the ledger persists.
Of 2,820 positive actions, 2,371 have local benefit versus zero and 449 have
unknown evidence; none has a local-nonharm lower bin. Covariance with F:
launch +2,659.25, H -1,803.44, other elapsed +6,009.66 s2 (sum = F variance).

At first own-due exposure, 106 cycles already carry 14,965 s H; 2,110 s follows
in 154 positive actions. All 106 associations match first next same-bus launches,
return-bound after the posted plan with zero extra wait beyond the native tick.
All 154 actions have local compensation benefit and no local-nonharm cut.
1,296 own-future queries also have other due obligations: future association
must not be read as globally pressure-free.

Upper: 106 matched/return-bound cycles, 14,940 s prior H, 2,130 s later H.
Versus V107: H -5 s, passenger +5,673 person-s, CV2 +0.0000982053, public delay
+89 s. Of 250 matched chains, next launches: 6 earlier, 28 later, 216 unchanged;
12 final returns have no next label.

Next: design earlier causal supply linkage and explicit compensation tradeoffs.
Overlap exists, but harmless H removal is not identified; 87.6% of H on these
cycles predates own-due exposure. A strict-local-nonharm cut guard is a no-op.

## Limitations
One consumed actor/day, not independent efficacy or mediation evidence.
Position links are not reservations; H and observed waits are not slack.
Query-label sums are repeated exposures, not unique dispatch totals; first
own-due labels are cycle-unique. No query-level statistical replication.
