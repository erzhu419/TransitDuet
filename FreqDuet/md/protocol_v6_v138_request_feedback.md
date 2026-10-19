# V138: read-only request feedback

Use the same six admitted V137 pairs only. No replay, new days, training,
parameter fitting or main-policy change. Tests and analysis run via scheduler.

For stable same-vehicle/front/follower identities and prediction source, split
the continuous bilateral request into realized forward/follower timing and
forecast-error changes. Further split timing into launch, cumulative holding
and other elapsed. Use the original clipping secant and retain discrete-command
residuals. Keep identity/source/evidence changes in separate request buckets.
Reconcile command and passenger-holding sums; account separately for command
clock relocation across 15-minute bins. Align with saved system passenger time,
backlog and matched vehicle-return/launch accounting. Full details stay remote.

Realized follower labels include future feedback, not deployable current state.
These identities and selected-path time alignments are not per-action causal
values or new independent policy evidence. Fetch only the small final summary.
