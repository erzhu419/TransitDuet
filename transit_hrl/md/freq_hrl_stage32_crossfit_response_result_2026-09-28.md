# Stage-32 Cross-Fitted Response Result

Tasks `t101748/101749` completed on node006/node005 at `6ff7076cd4`.
34 tests pass. Independent server recomputation matches causal prefixes,
plans, path-held-out fits, kernels, predictions, labels, CIs and accounting.
Primary joint qualification fails: root 209011 fails both gates, 209061 passes.

| Root | Linear history MSE | Cross-fit history MSE | Change | Changed decisions |
| --- | --- | --- | --- | --- |
| 209011 | 0.057826986 | 0.060472795 | +4.58% | 11/120 |
| 209061 | 0.040437514 | 0.039669045 | -1.90% | 1/120 |

Cross-fit minus linear history decision benefit (control minus candidate ISE):
209011 -0.005574, descriptive path CI [-0.013866, 0.002569];
209061 +0.000586, CI [0, 0.001759]. No stable positive correction claim.
All other decision-control means are positive, but multiple CIs cross zero.
Retrospective scoring of the unchanged linear history against its original
six learned controls plus constant decisions passes its old two gates on
both new path rosters. This is supporting development evidence, not a
replacement for Stage-31's failed result or a newly registered primary win.

Costs: 620 reused fit/240 fresh pairs, 339600 new primitive steps, 240 new/620
reused proposal calls, 238 linear+14 kernel solves (1260 RHS), zero controller,
motion or physical updates/reconstruction. Audit adds 224 linear+14 kernel
verification solves, separately counted. Synced 1791886 JSON bytes only.
Next: stop this kernel-correction route; freeze the original linear forecast
response for qualification on new optimizer roots before deployment.
Limitations: local paired decisions with shared history-aware candidates,
reused optimizer roots and descriptive intervals; no episode reward or
planning-cost savings. Preserve negative preflight and failed primary gates.
