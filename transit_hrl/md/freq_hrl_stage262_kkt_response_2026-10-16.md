# Stage262: Data Removal Versus RMS Multiplier

Reuse Stage260/261 caches; no fitting, simulation or evaluation panels.
t165651: node006, exit 0, no retries, 34.21 seconds; eight tests passed.
32 cells / 640 cached fits / 1,280 primary records reproduced within 1.17e-15.
Only the 2,607,378-byte scalar report was pulled; raw ledgers stay remote.

For A = H_prefix + mu_prefix I, q = grad L_prefix(u_full) - grad L_full(u_full),
d = (mu_prefix - mu_full) u_full: data = -pinv(A) q, dual = -pinv(A) d.
Retain null displacement and numerical residual; keep all six cross terms.
Maximum KKT / prediction decomposition residual: 6.58e-14 / 4.17e-17.

Mixed removed group: 12 bank-cells / 120 active prefix-models per family.
Cost-response RMS values below describe function changes, not outcome costs:

| Region / model | Actual response | Data component | Dual component |
|---|---:|---:|---:|
| Excluded / free_old | 0.00521148 | 0.00536953 | 0.00073777 |
| Excluded / shrunk | 0.00550525 | 0.00569499 | 0.00100167 |
| Held / free_old | 0.00182863 | 0.00193510 | 0.00057913 |
| Held / shrunk | 0.00183770 | 0.00195739 | 0.00070757 |
| Held rest / free_old | 0.00117442 | 0.00121101 | 0.00022989 |
| Held rest / shrunk | 0.00115433 | 0.00124053 | 0.00042274 |

Data forcing dominates excluded/held remaining-direction response in both banks.
Mixed removed data/dual cross terms are negative in all regions and low/rest
blocks; held cross / actual squared response is -22.0% / -28.3% (signed ratios,
not shares). Mixed all-category held also cancels. Null/residual RMS <= 5.03e-14
for removed held. Multiplier change is not the main amplification source here.

Limitations: endpoint-KKT algebraic attribution, not separately fitted causal
interventions or utility evidence; component RMS values do not add. Frozen
calibration saw excluded labels; exposures are not seeds and removed bank sets
are not paired. Root397/root401 shrunk excluded cross terms are positive.
Stage253/248/169 stay closed; no new selection, deployment or default change.

Next: keep the cap fixed and compare same-prefix data-forcing directions across
the two cached independent future banks under matched model/calibration. Isolate
target variability from weighted-design sensitivity before new fitting.
