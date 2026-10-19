# V142: fixed finite-sample rank intervals

V140/V141 points are copied unchanged, not regenerated. Same saved V137 paths,
strict completed labels; no replay, training or action selection. Source418a3f001a,
t195284/node006 exit0,9 tests pass; only41496-byte summary fetched.

Keep each tail probability0.1 and the original1h/all-past window rule. Replace
linear interpolation by ranks r=floor((n+1)/10), s=n+1-r. n<9 has no finite
upper rank: upper=null with explicit status, never counted as coverage success.
No-survivor cases stay missing; physical lower support0 is not an estimate.

Service:3974/7327 finite(54.24%);3154 insufficient-rank,199 no-survivor. On the
same3974 requests, coverage70.61->85.56%, mean width10.44->15.32s. Ready:
436/492 finite(88.62%),56 insufficient-rank; paired coverage81.42->92.66%,
width15.11->22.05s. Frozen point MAE remains V14046.00692/V14145.23681s.

Rank reference: [Dunn (2021), Theorem4.2.1 and Appendix D](https://robinmdunn.github.io/files/RobinDunn-dissertation.pdf#page=78).

Limitations: hour13 finite coverage75.51%(n98), hour0 has only6 finite cases.
Exchangeability is not established for adaptive completed-window bus data;
rank theory is not a field-calibration guarantee. These are current-stage, not
full-ETA intervals, on consumed selected paths; no independent/control claim.

Next: retain the fixed rank implementation for passive use. Remaining gaps are
low-support availability and independent calibration, not point-mean tuning.
Do not change windows/tails or launch new paths/control trials in this stage.
