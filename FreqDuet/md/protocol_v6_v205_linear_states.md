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
