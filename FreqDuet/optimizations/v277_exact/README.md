# Detached exact-computation candidate

Only this new directory is added. Existing controllers, historical results and
running V277 paths are unchanged. No traffic replay, training or main switch.

The candidate projects only the nearest eligible physical follower and caches
identical native-rule inputs within one query. The native rule, clocks, beam,
128 logical evaluations, action set, constraints and arrival metric are reused.

Server task `t212094`, node004, high/manual: seven regression tests and four
complete solver comparisons passed with exact output equality.

| Query workload | Previous optimized seconds | Candidate seconds | Speedup |
| --- | ---: | ---: | ---: |
| Late native | 4.463 | 1.815 | 2.46x |
| Late arrival CV2 | 4.503 | 1.975 | 2.28x |
| Midday native | 7.633 | 3.455 | 2.21x |
| Midday arrival CV2 | 7.740 | 3.540 | 2.19x |

These are synthetic query timings against the prior4888d751ab optimization,
not complete-day timings or independent traffic-benefit evidence. Raw profiles
remain remote; `benchmark_summary.json` is the small local receipt. Its
`frozen_s` measures the old search with the previously optimized kernel.

Next: isolated same-path complete-day equivalence and timing against a saved
reference before adoption. No such replay has been started or substituted.
