# V245: read-only V244 holding propagation

Read the six V244 stored paths only. Reproduce CV2 and actual holding endpoints,
then close event times and headway covariance using launch,direct adjustment,
native state response,and remaining elapsed time. Keep changed-neighbour support
and CV mean normalization explicit. Compare both supply/blind closed-loop pairs
and the two saved requests on each identical post-anchor state.

Local restoration labels freeze completed departure neighbours;they are geometry,
not replayed action values. No per-query independence/CI or causal promotion claim.
No V238/controller edits,tuning,new futures,replays,training or formal matrix.
Run synthetic accounting tests and analysis through scheduler high/manual on the
server. Detailed diagnostics remain there;return only a small summary.

Completed:high/manual node004 tasks t211377/39cd832b46 and t211378/8050fd604e,
both exit0. Six-path endpoint/clock/covariance closure passed;5200 matched headway
edges,no neighbour changes. Lower-supply vs V107:direct -20920s,native response
+6665s,net -14255s. Local fixed-departure geometry labels1270/1436 cuts as lost
compensation;45->10s is116/117. CV2 delta+.019365 includes holding variance
-.022969 but holding/remaining-travel-clock cross+.022647,normalization+.000364.
Supply vs blind retains605s(630s with upper),slightly improves CV2,but costs
journey time;all262 actual launches and vehicle identities are identical in
both supply/blind pairs. This smoke's large service gain is also present blind.
Next:do not open the formal matrix or tune V238. Separately preregister a rule
that distinguishes necessary disturbance compensation from supply-chain excess;
cumulative H alone has not made that distinction. Any new evaluation needs
separate authorization. Limits:single consumed future,geometric/covariance
accounting rather than per-action closed-loop causal effects;remaining time also
contains unobserved terminal/clock effects,not separately measured pure running.
