# V203: cell means versus parent/day centering

Read V202 saved daily moments/reference draws ONLY,original64 days and two
fixed keys. V202 raw/centered values already contain V191 factors;do not apply
weights twice. Preserve V199 masks,bins,19-day threshold,original denominators
and equal12 actors/day. Full parent/day means use ALL9 cells before exclusion.

At each supported actor/key/bin,let j,y be daily cell means,pJ,pY the full
parent/day means. Under this CELL's daily mass,not parent daily mass:
Cov(j-pJ,y-pY)=Cov(j,y)-Cov(j,pY)-Cov(pJ,y)+Cov(pJ,pY).
Split pooled mean-products into the same4 signed products. Daily within-cell
covariance is unchanged. Apply the SAME V198 average-mass times half-difference
to every part;all parts plus unchanged mix must reproduce V202. Low-support
mass/cross change stays aggregate;no conditional decomposition or renormalizing.

Reuse10000 shared-actor/independent-half-day draws,seed20260803;compare V202
points and EACH matched draw. Preserve full signed covariance allocation among
the4 between,4 mean-product,and within/raw-cell/centering groups. These views
overlap and must not be added together. New95% intervals diagnostic only.

Five server tests cover common-shift cancellation,cell-only covariance,weight
scaling,V202 reproduction and selected-parent centering errors. High/manual
scheduler,node004;summary only local. No V199/V194 reads,raw trajectory scan,
interval/predictor fit,new confirmation,replay,training or frozen-policy change.

Limitations:algebraic parts of one fixed estimand,not alternate algorithms or
evidence to remove centering. Reused development days/keys;no new gate,causal
claim or extrapolation beyond supported cells. Bootstrap uncertainty channels
and day-mean covariance are different quantities.
