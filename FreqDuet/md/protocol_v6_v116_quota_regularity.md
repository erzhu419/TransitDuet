# V116: read-only V115 compensation and regularity diagnosis

Read the eight completed funded and eight bilateral025 paths from V115. No
simulation, fitting, new demand days, actor/dual updates or main-policy change.
Run tests and analysis through scheduler on node001--node006; retrieve only the
small final JSON and logs if needed. Frozen V115 data and source stay unchanged.

On each funded visited prefix, B is the quarter-rule shadow, P the seven-bin
current pulse without funding cap, A the executed funded action. Exact command
chain: A-B = (P-B)+(A-P). Disjoint removed baseline holding is
`negative_cut=max(B-P,0)` and `quota_cut=max(min(B,P)-A,0)`. Funding that only
withholds a positive pulse addition is reported separately. Negative here means
the response increment `-0.75*delta_R<0`, not the raw residual innovation sign.

For each cut, move only that recorded departure by the restored seconds, keeping
all other completed departure times fixed. Measure exact finite-day CV2 change,
squared-gap change, reordered
neighbours and boundary events. Also calculate simultaneous negative/quota/all-cut
and all-shadow shifts, with actual sorting and finite-day CV2 normalization.
These geometry shadows identify locally useful removed compensation; they are
not simulator counterfactuals and cannot establish passenger or full-day effects.

Reproduce recorded holding and actual arrival CV2 before analysis. For actual
departures, exactly decompose CV2 differences into arrival timing, nonholding
service dwell and current executed holding. For actual arrivals, decompose release, accumulated quarter
shadows, negative/positive pulse responses, quota cuts and other elapsed time.
The current action is excluded from its own arrival. Components use covariance
with realized headway and include the separate mean/boundary normalization term.
They describe observed trajectory propagation, not identified causal mediation.
Keep initial services without lower decisions in the actual metric and require
their recorded holding to be zero. Do not invent control decisions for them.

Use existing crossed actor/day bootstrap intervals on the eight consumed cells;
report actor/day patterns. Decide whether evidence points to locally destructive
cuts, follower evidence, or history/other-timing propagation before proposing
another controller. Do not tune gains, quotas or admission gates on these paths.
