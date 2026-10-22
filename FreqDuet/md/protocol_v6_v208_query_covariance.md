# V208: query-observable covariance test and availability

Original64 saved statistics only;V199 masks/19-day-per-half threshold,V191
weights,all12 actors/64 days and frozen controller/predictors unchanged.

Freeze two strictly completed query-time signals:running surprise RMS and
global service-prefix residual mean. Test their within-cell/day covariance
with the full-parent-centered residual product,using SAVED V205 point beta.
Preserve within-cell and relative-cell/parent contributions;all9 cells enter
parent means. Retrospective centering is diagnostic,not a deployable input.

Seal16 two-sided associations and both-half/both-mode replication before the
availability audit. Reuse nominal V207 fixed mass coefficients,without fitting.
Existing second moments provide every term EXCEPT sum(w*S*residualJ*residualY).
Only two additional aligned scalar sums are needed,not a high-order tensor.

Audit server NPZ headers,small support arrays and saved operator admission.
Do not rescan trajectories,estimate associations,fit intervals,start confirmation,
replay or train. Tests distinguish identical second moments with opposite
third moments,and verify centering,relative-mean decomposition and no-payload
header inspection. High/manual scheduler on node004;small status JSON only.
