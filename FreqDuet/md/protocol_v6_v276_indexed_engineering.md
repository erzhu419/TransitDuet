# V276 indexing: engineering equivalence and performance

Keep control source7023e2e784 and all current physical paths untouched. Optimize
only historical arrival/departure lookup and route-coordinate construction.
Use the same native rule functions, clocks, tie order, physical simulator,
128-evaluation budget, beam4, variables6, objectives and acceptance gate.

Check complete model outputs, all seven current bins and two-stop transfers,
same-tick fleet order, first-tie identity, committed commands, candidate isolation
and the full solver/fallback result. Profile a synthetic12-vehicle,21-stop-per-
direction state with a late-day-sized historical prefix. Compare identical fixed
plans and the full solver; warm imports before timing. These are engineering
measurements,not traffic efficacy or guaranteed full-day speedups.

All checks/profile jobs use scheduler high/manual on node001--006. No real
rollout,replay,training,main-policy change or cancellation is authorized here.
Keep cProfile files server-side and fetch only compact JSON/logs. After a pass,
report the measured gain and obtain a decision on replacing the three remaining
physical paths;any replacement needs explicit additional start accounting.
