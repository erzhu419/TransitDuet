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

For negative baseline cuts and all funding-withheld extras `P-A`, move only that
recorded departure by the restored seconds, keeping
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

## Run record

Initial source `2f0999559a`, task `t188792`, stopped at a missing decision on an
initial uncontrolled service. Corrected source `913be92a37`, task `t189877` on
node006, passes six tests and reproduces holding/actual arrival CV2 on all 16 paths.
Only its 149,322-byte JSON returned. This first pass finds zero funding cuts to
baseline authorization, so extend the local geometry to all funding-withheld
positive extras, not only baseline cuts. Same consumed paths, no replay or tuning.

## Outcome and next step

Final source `f8e6f34995`, task `t189887` on node006: six tests and all 16 paths
pass. Its 256,306-byte JSON plus the first summary total 405,628 bytes locally.
Negative responses remove 50,255 s baseline holding; 36,210 s (72.1%) is locally
useful when individually restored. Quota removes zero baseline holding but
withholds 117,895 s pulse extras; 92,495 s (78.5%) is locally useful. Simultaneous
geometry CV2 changes are -0.000750 [-0.000820,-0.000681] for negative restoration
and -0.002389 [-0.002476,-0.002328] for quota restoration, negative in all eight cells.

Actual funded-minus-bilateral arrival CV2 remains +0.001313. Mean normalization
is only +0.000035; departure accounting is arrival timing +0.001182, current
holding +0.000181, service dwell -0.000074 and normalization +0.000030. Arrival
accounting includes release +0.000587 and net signed pulse/quota +0.000579.
Most cuts occur with valid follower evidence; all 20 stops are represented.

Keep main frozen. Do not loosen quota or restore all holding from geometry
shadows. Next read the same paths to reconstruct decision-time forward gaps and
follower-ready estimates from residual/release/executed-holding coordinates;
align errors with beneficial/harmful cuts and test whether delta_R duplicates
information already present in D. This precedes any causal allocation redesign.
