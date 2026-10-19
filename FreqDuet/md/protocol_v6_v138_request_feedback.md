# V138: read-only request feedback

Use the same six admitted V137 pairs only. No replay, new days, training,
parameter fitting or main-policy change. Tests and analysis run via scheduler.

For stable same-vehicle/front/follower identities and prediction source, split
the continuous bilateral request into realized forward/follower timing and
forecast-error changes. Further split timing into launch, cumulative holding
and other elapsed. Use the original clipping secant and retain discrete-command
residuals. Keep identity/source/evidence changes in separate request buckets.
For source switches with unchanged identities, also report the timing/error
identity separately; this is not a same-state source intervention.
Reconcile command and passenger-holding sums; account separately for command
clock relocation across 15-minute bins. Align with saved system passenger time,
backlog and matched vehicle-return/launch accounting. Full details stay remote.

Realized follower labels include future feedback, not deployable current state.
These identities and selected-path time alignments are not per-action causal
values or new independent policy evidence. Fetch only the small final summary.

Outcome: sources ecf08d43d8/61df8f20b4, t195269/t195270 on node006, both exit0;
6 tests pass. Final summary178378 bytes; no raw paths/checkpoints pulled.
All seven-bin commands reproduce; no changed-action identity mismatch bucket.
12071/2351015: direct-path holding-75s, other stops+510s, passenger+22103person-s.
Other-stop allocation: stable timing+130.27s, stable error+38.56s, rounding+194.22s,
source switches+119.70s, remaining evidence/forward-only requests+27.25s.
Source switches are predicted <-> not_causally_moving (raw ETA fallback), not
follower-ID changes. Trip219/stop8: realized forward/follower changes+2/+4s,
command5->45s; clipped error allocation37.87s vs timing0.23s and rounding1.90s.
12143/2351015: other stops+310s, passenger+20309person-s; stable cumulative
holding allocation+166.35s despite stable error-9.50s. Delay+0.9313s/trip includes
prior holding+0.5752 and prior launch+0.3130s/trip. In contrast12071/2351007
reduces holding195s and launch delay1.1107s/trip, yet waiting+18779person-s
outweighs onboard-6576person-s. Rebound is not a universal loss explanation.

Next: prioritize read-only fallback-trigger/stop-phase analysis on saved paths,
then service timing versus demand. Do not tune bins/gain on these pairs or
promote the forecast/control package; no additional replay or training started.
