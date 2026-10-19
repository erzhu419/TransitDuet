# V145: fresh independent validation of frozen intervals

Authorized after V144 found no admissible archive. Register demand seeds
2451001--2451032 and sampler1450000000+i (i=0..31), one sampler per day shared
by every actor and old/handoff arm. Twelve existing B200 actors x32 days x2
modes =768 new frozen-control paths; no training or main-policy switch.
Check the V144 server usage ledger and prior source generators before collecting.

Keep V140/V141 point formulas,3600s/all-past window,0.1 tails and V142 ranks.
V137's full passive recorder is reused; terminal clocks are stored alongside
each new path. First reproduce one consumed old/handoff pair: actions, event
arrays, terminal clocks, every saved point and every saved interval bound.
New-day collection requires that preflight from the same execution source.

Eligibility is fixed before future labels are scored. Keep missing same-vehicle
outcomes and unsupported intervals; report their denominators separately.
Episode averaging precedes actor/day averaging. Use the existing5000 crossed
bootstrap draws, fifth-percentile lower bound >=0.8 in both kinds and modes.
Incomplete support is inconclusive, not a zero or pass; unresolved finite labels
cannot certify coverage. Report hour13/rest and the five V143 support classes.
No coverage readout is exported until all768 registered paths are present.

All collection, smoke and statistics run through scheduler on node001--006.
Raw traces, labels, shadows and checkpoints stay remote; only small status
summaries are fetched. Scientific validation and data collection are separate
from control admission; main policy stays frozen even if interval coverage passes.

Execution source6eaa4a389b: preflight t195290/node006 exit0,36 tests pass;
both consumed modes reproduce actions, event arrays, terminal clocks, saved
points and rank bounds. Initial t195289 stopped before replay because the smoke
subdirectory lacked the existing runtime-equivalence proof reference; its
server-side link was restored without changing models or forecast parameters.
Formal paths submitted as t195292--t195296 using scheduler's live CPU plan
(16/13/12/5/15 workers); t195297 waits for all768 cells and aggregates remotely.
All768 cells and t195292--t195297 completed,zero retries. Full paths remain
remote; only99155 bytes of three summary JSON files were fetched this analysis.

## Results

Episode-first, actor/day crossed5000-draw estimates; percentages and95% CIs:

| Arm | Service | Ready | Service hour13 |
| --- | --- | --- | --- |
| old | 85.54 [84.99,86.07] | 92.74 [91.77,93.74] | 68.55 [64.98,72.12] |
| handoff | 85.54 [84.99,86.07] | 92.79 [91.75,93.85] | 68.55 [64.93,72.18] |

Both global kinds/arms pass the frozen one-sided lower-bound80% gate.
Service finite availability remains55.2%;75% of unsupported requests lose
survivors(mean pool9.43 becomes4.99--5.00). Ready availability is89.6%.
All outcomes resolve. Hour13 service misses80% for all12 actor margins and
28/32 day margins in each arm; these margins are descriptive, not44 tests.
Ready hour13 has only272/281 contributing cells out of384: inconclusive.

Read-only audits t195312/cbd61026d2 and t195314/a862f0c775,node006 exit0,
respectively5/6 tests pass. All768 clocks stop new demand at46980s; hour13
starts46800s. Within hour13, pooled query coverage before cutoff is88.07/86.87%,
after cutoff64.07/64.55%;940/985 and916/964 lower misses occur after cutoff.
The finite hour13 supports are9--18, so bounds already span empirical min/max.
This post-hoc boundary diagnosis does not replace the registered hour13 gate.

Next: separate clearance-phase service-distribution calibration from survivor
scarcity; retain explicit abstention. Design distinct calibration/confirmation
days, with V145 now consumed. Point/window/tails and main remain frozen; no
new replay/training or controller promotion in this analysis.
