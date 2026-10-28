# V276 indexing: engineering equivalence and performance

Keep control source7023e2e784 and all current physical paths untouched. Optimize
only historical arrival/departure lookup, route-coordinate construction and
repeated native-rule object/mask allocation. Reuse the same native functions.
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

The user subsequently approved replacing at most three still-unfinished paths
and the old aggregate wait,conditional on final verification. Reuse completed
original pairs;do not alter their files. Maximum physical allowance becomes
13 original starts plus3 replacements,not three new cases. The first searchable
query checks every optimized kernel against frozen7023e2e784 before continuing;
these extra engineering evaluations do not supply additional search candidates.
Keep128 search evaluations,all seeds and the original acceptance gate unchanged.

Completed validation: t211922/source4a216af00c passed 19 checks. Full 128-evaluation
synthetic search fell from 75.898s to 6.018s (12.612x); index construction 0.00264s.
The earlier profile attributed about 78% of frozen kernel time to two history scans.
All model fields and full solver output matched exactly;real-day speed remains
to be measured. Old t211829/t211831/t211832 and queued t211837 were cancelled
with zero related cancellations or kill failures. Replacements t211923--t211925
run on node001/node004,aggregate t211926 waits for their receipts. Each of the
three real first searches already passed 128 frozen/indexed kernel comparisons.
Reuse original pairs0--7 and9; no training, gate change or main-policy promotion.

Completion:all three replacements and aggregate t211926 finished without an
extra retry. Actual full-day replacement wall times were7.153h(p8),6.700h(p10)
and10.604h(p11);the reused unsupported p9 took0.245h. Corresponding indexed
callback p95 values were8.647s,8.566s and8.882s. First-search maxima include
the explicit frozen-kernel verification overhead. These are measured runtimes,
not a same-completed-path full-day speedup ratio. The original frozen runs
were stopped before completion. The original traffic gate remains unchanged;
see the completed development readout in the native-continuation protocol.
