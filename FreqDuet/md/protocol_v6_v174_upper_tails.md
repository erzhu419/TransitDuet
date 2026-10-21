# V174: read-only V173 upper-tail diagnosis

Task t210451 (high/manual, node005), source caeaad8fed, exit 0: 7 tests passed.
All 768 saved paths reproduced the targeted V173 endpoints and crossed CIs.
Exact labels, candidate-finite masks and episode/actor/day weights were retained;
only the 132,198-byte summary was fetched. No fitting, replay or frozen changes.

Numbers are old / handoff; probability differences are percentage points.

| Comparison | Mean upper-bound change, s | Upper-miss change, pp |
| --- | --- | --- |
| Clearance vs coarse | +0.013 / -0.115 | +0.753 / +0.840 |
| Clearance vs V167 | 0 / 0 (all bounds identical) | 0 / 0 |
| Hour13 moving-launch vs coarse | -3.030 / -3.030 | +1.406 / +1.448 |
| Hour13 moving-launch vs V167 | +0.629 / +0.569 | -0.109 / -0.033 |

Both coarse-relative miss increases exclude zero. Moving-launch adds
2.715/2.796 pp new misses while repairing 1.310/1.349 pp. Clearance inherits
V167's problem; nearly unchanged mean bounds conceal conditional misallocation.
V167-relative moving-launch miss differences include zero, but score worsens
+0.470 [0.136,0.815] / +0.693 [0.327,1.095] seconds, mainly through increased
lower-tail penalties, not systematic upper-bound contraction.

Shared-day effects account descriptively for about 94% of upper-miss cell
variance; actor means account for at most 0.12%. Moving-launch day-mean misses
span 2.57%-18.32% / 2.87%-18.17%. This is not causal attribution or proof of drift.

Next: retain hour2's finding, stop full-map promotion. Develop any new tail
method on calibration days under a separate protocol with fresh confirmation;
do not tune on consumed days 3152001-3152032. No further experiment started.
