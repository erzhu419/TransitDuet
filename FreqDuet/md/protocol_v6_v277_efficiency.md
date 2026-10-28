# V277 efficiency: detached profile, no live replacements

The20-path matrix/source b9909ddc03 stays untouched:no cancellation,replacement,
hot patch,extra traffic start,training or main switch. Profiles ran high/manual
on servers;only small JSON returned locally,cProfile files stayed remote.

Live readout:eight paths completed(fixed50--52s,V2756588--11785s),eight128-call
paths running,seven-bin arms still behind references in their sequential shards.
One representative query made53760 native rule calls and repeated39/128 plans.
AVL/native ETA dominated,not the CV2 overlay. Full lookahead may contain26
published trips versus12 in the old timing fixture.

Development-only fixes:skip the discarded legacy follower/action computation and
unused local departure history when native continuation is active;memoize exact
command dictionaries inside one FallbackGraph query. Keep128 logical evaluations,
beam4,variables6,all scores/ties/gates and independent query state unchanged.

Source4888d751ab/t212088 passed68 checks plus exact complete solver comparisons:

| Corrected-CV workload | Frozen seconds | Optimized seconds | Kernel calls |
| --- | --- | --- | --- |
| Late regression,12 plans | 7.270 | 4.221 | 128 -> 89 |
| Midday,26 plans | 11.431 | 7.029 | 128 -> 90 |

The seven-bin arm kept exact outputs too(0.392->0.339s;0.618->0.551s).
Limitations:synthetic solves,not full-day speedup or traffic benefits;no live
application. Next:one path per queued task under the same concurrency cap,
and choose the seven-bin method only after the current traffic comparison.
