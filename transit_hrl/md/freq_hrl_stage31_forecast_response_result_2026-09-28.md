# Stage-31 Forecast-to-Plan Response Result

Tasks `t101745/101746` completed on node006/node005 at `073d377fa6`; 24 tests
passed. Both decision-mean gates pass; prediction and joint gates fail.
Benefit = control minus history ISE at equal-call step 150; descriptive path CI:

| Control | Root 209011 mean [CI] | Root 209061 mean [CI] |
| --- | --- | --- |
| Current forecast | 0.03420 [0.00740, 0.06373] | 0.01752 [-0.02180, 0.04889] |
| Shuffled forecast | 0.03375 [0.00896, 0.05887] | 0.00904 [-0.01809, 0.03762] |
| Lag-one extrapolation | 0.04279 [0.00512, 0.08338] | 0.01243 [-0.00213, 0.02865] |
| Zero forecast | 0.06944 [0.03530, 0.10120] | 0.02211 [-0.02702, 0.07283] |
| Raw current | 0.14665 [0.05919, 0.23874] | 0.07810 [0.02126, 0.14192] |
| Raw history | 0.12157 [0.04879, 0.19604] | 0.09024 [0.03122, 0.15362] |
| Always keep | 0.13474 [0.07194, 0.19587] | 0.06302 [0.01624, 0.10582] |
| Always renew | 0.08092 [0.03885, 0.13012] | 0.06900 [0.02341, 0.11427] |

History rate MSE 0.079441794/0.083735621 beats current 0.079974714/0.097143113,
but loses to shuffled 0.079045097 (209011) and lag-one 0.080622758 (209061).
Retain failed joint qualification and negative preflight without retuning.
Accounting: 620 reused fit/240 fresh pairs, 347900 new steps, 860 proposal calls,
14 critic fits/solves (70 RHS), zero controller/motion/physical updates or
reconstruction. Server recomputation matched inputs, plans, labels, designs,
predictions, metrics, CIs and budgets. Synced 1325855 JSON bytes, no raw/ckpt.
Next: address plan-response approximation and decision-quality mismatch before
new-root qualification; do not extend this completed screen or promote deployment.
Limitations: reused-root local decisions, not episode reward or cost savings;
four root-209061 CIs cross zero, with no simultaneous confirmation. All views
share a history-aware candidate. Stage-9 remains the performance reference.
