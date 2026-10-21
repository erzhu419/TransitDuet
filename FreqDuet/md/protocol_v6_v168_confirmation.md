# V168: independent confirmation of frozen V167 maps

Evaluation source `8466f300ab`, map `e6fc851ec1`, protocol `4a6d783667`.
High/manual preparation `t208543` passed 17 tests. All 48 shards
`t209824`-`t209871` completed on node001/node002/node004/node005/node006,
zero retries; aggregation `t209872` ran on node006.
Reserved days2852001-2852032, samplers1600000032-1600000063: 768 paths,
3,993,600 queries. Every actor/day/mode/scope cell is complete; finite unresolved
labels and missing matched coarse references are zero. Bounds precede labels.
Statistics use the frozen 5,000-draw crossed bootstrap, seed20260803.

Whole-day and active gates pass in both modes. Whole-day old/handoff coverage:
0.826554/0.826411. Paired interval-score differences versus the same-mask coarse
reference: -2.743 [-3.027,-2.454] / -2.735 [-3.022,-2.450] seconds, 95% crossed CI.
All finite-fraction lower bounds pass. The full joint gate is **not admitted**:
- hour2: coverage0.749483/0.749434; above-upper rate0.220570/0.220672.
- clearance: above-upper one-sided upper95=0.104256/0.105779; score-delta
  upper95=+0.161349/+0.258261, so neither required guarantee is confirmed.
- hour13: above-upper upper95=0.104971/0.105569, despite better mean score.

Read-only actor/day marginals (`t209874`, source `b9d60d28a7`) show hour2
undercoverage in all 12 actor marginals, 29/32 day marginals, and above-upper
rates >0.1 in all 32 day marginals, for both modes. Whole-day coverage >0.8 and
score delta <0 hold in every actor/day marginal (descriptive, not separate CIs).
The matched coarse reference also undercovers hour2 (~0.749), whereas its hour13
above-upper upper95 (~0.0956/0.0959) passes: shared hour2 failure and conditional
hour13 tail risk need distinct diagnosis. No causal component is identified yet.

Only the 41,058-byte status JSON and small report log were fetched. An inline
report launch (`t209873`) hit SSH argument length and was canceled before analysis;
the short-file report replaced it without rerunning confirmation or inference.
V167 maps, V151/V146, window3600s, tails0.1/0.1 and main policy remain unchanged.
Next: read-only hour2 running/service/residence and calibration-versus-confirmation
residual diagnosis, plus hour13 upper-bound contraction versus coarse; no tuning
or re-admission on these now-consumed confirmation days.
