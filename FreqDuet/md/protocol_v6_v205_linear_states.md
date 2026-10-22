# V205: execute the sealed V204 linear diagnostic

Use original64 saved V199 joint moments only,with V202 daily/V203 point-and-draw
references and V204 inventory. Two fixed keys,9 fixed query-time states,old/handoff,
clearance/hour13_moving_launch. Keep V199 masks/bins/19-day threshold,V191 factors,
original denominators and equal12 actors/day. No raw archives or new data.

One pooled WITHIN-cell/day standardized projection per mode/scope,shared across
keys,actors,bins and halves;recompute in each of the unchanged10000 crossed draws.
Use V195 pinv rcond1e-10. Zero-variance and collinear state coefficients are not
individually identifiable;no state variation gives unknown,not a zero effect.

Compute full parent/day means from ALL9 cells before support exclusion. Expand
V203's9 terms as bilinear matrices,then evaluate state-state,the two mixed pairs,
and residual-residual under the common operator. Their sum must reproduce EVERY
V203 point/draw part,endpoint and mix. Retain signed cancellation. Unsupported
cells receive only the ORIGINAL aggregate JY contribution,no state partition.

The V20416 two-sided conditional-change contrasts use alpha.05/16;all other
intervals are diagnostic95%. These are bootstrap uncertainty summaries,not a
new ETA interval fit. Source-clock validity is inherited from V194/V199.

Six server tests cover scalar/matrix matched draws,all4 pair additivity,
draw-specific operators,within-day versus parent centering,unsupported exclusion,
collinear/constant states and nominal weight scaling. Run high/manual on node004;
only the small status summary comes back locally. No confirmation,replay,training,
map/predictor/window/quantile or main-policy change.

Limitations:reused development data,additive linear co-movement only. Mixed or
residual covariance does not identify a conditional-law change or causal control
effect;the V204 nonlinear joint-moment/support gap remains. No policy promotion.

## Result and next decision

t210954/source09530a00dc completed high/manual on node004,exit0;6 tests passed.
All4 cases have10000 valid draws. V203 parts/endpoints/mix reproduce(max2.13e-14);
the4 state/residual pairs add back(max6.06e-13). Support remains315/864 cells.
Only157779-byte status summary fetched;detailed arrays stay on the server.
Clearance point rank9(draws8-9);moving-launch rank7,with predicted-entry and
known-command indicators constant and not individually identifiable there.

Point ranges across old/handoff,NOT confidence intervals,nominal weighted s^2:

| Key / scope | State-state | Mixed sum | Residual-residual | Original |
| --- | --- | --- | --- | --- |
| 7 / clearance | .384 to .484 | -1.586 to -1.514 | .101 to .485 | -.929 to -.717 |
| 7 / moving-launch | .210 to .267 | -1.539 to -1.537 | 2.245 to 2.248 | .916 to .977 |
| 12 / clearance | -.482 to -.443 | .485 to .552 | 2.260 to 2.414 | 2.331 to 2.456 |
| 12 / moving-launch | -.240 to -.208 | 1.172 to 1.216 | 1.509 to 1.647 | 2.485 to 2.611 |

All16 primary family intervals cross zero;all residual-change95% intervals also
cross zero. State projection does not absorb key12's positive change at the
point estimate. Residual between-day contribution is positive at the point
estimate,not a confirmed conditional-law shift. Large raw-cell and parent-center
terms still cancel;they are not independent explanatory gains.

Decision:retain this diagnostic,all frozen items and existing support. No basis
for a new state/interval correction. If continuing,locate actor/day contributions
to the residual uncertainty from saved moments/draws,not by adding features,
changing bins or refitting on confirmation data. No such additional test run here.
