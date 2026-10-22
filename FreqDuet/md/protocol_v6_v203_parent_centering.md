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

## Result and next decision

t210948/sourcec26aa1f39b completed high/manual-dispatch on node004,exit0.
Five tests passed;V202 points and individual draws reproduced(max error1.60e-14).
Only78465-byte summary fetched:freqduet/results_freqduet/protocol_v6_v203_parent_centering/audit_c26aa1f39b/summary/status.json.

Key12 between-cell contribution8.86-10.12s^2 is offset by parent-centering
terms-8.69 to-7.69,leaving1.17-1.44. Full conditional raw-cell contribution
11.12-12.28 is offset by centering-9.67 to-8.79,leaving2.33-2.61.
Key7 clearance:raw-cell3.41-3.64 plus centering-4.36 to-4.34 gives-.93 to-.72;
moving-launch:raw-cell3.88 plus centering-2.96 to-2.91 gives.92-.98.
All are components of the SAME fixed estimand,not results of removing centering.

Centering's signed conditional-variance allocation is negative:Key7 about-32%
to-58%,Key12-110% to-118%. Raw-cell terms and centering are strongly dependent;
their own variances cannot be added as independent noise. All new group-change
95% intervals cross zero. No centering implementation error or basis for a
predictor/control change was found.

Decision:keep centering and all frozen items. The V202 daily mean algebra is
now accounted for;more rearrangement cannot provide independent evidence.
Further attribution to observable operating states requires statistics beyond
these8 saved daily fields and a separately specified evidence design,not an
extra correction fitted to these64 development days. No such work started here.
