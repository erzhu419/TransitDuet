# V185: frozen V184 independent confirmation

One V183 cohort only:32 days3452001-3452032,samplers1831000000-1831000031,
12 actors,old/handoff;384 pairs/768 paths. V1842447e29d07 parameters and V182
protocol remain unchanged. No refit, threshold/window/quantile changes, extra
days, replacement, successful-subset admission or policy switch.

Use the admitted passive collector and equivalent optimized runtime. Before
collection, test causal completion-prefix replay against V179 on existing
consumed V166 preflight archives, then check reserved IDs remain unconsumed.
Only first field-complete eligible query per physical episode contributes;
label availability strictly precedes query time,3600s window,>=9 episodes.
Calculate completed component residuals after completion; current query labels
are attached only after interval predictions. Parameters never update.

Retain native/V172/prefix/parameter low-support abstention; V172 upper and
non-target intervals unchanged. Score candidate/V172 on the same candidate mask,
also report V172 own mask. Original episode/query denominators preserved.

Two modes x six frozen scopes:5000 crossed actor/shared-demand-day bootstrap
draws,seed20260803. Original one-sided gates: coverage lower95>=.8,each tail
upper95<=.1,finite fraction lower95>=.8; paired V172 score upper95<0 for
all/clearance,<=0 elsewhere. All must pass; missing evidence is inconclusive.
Save actor/day marginals and all negative results; controller never admitted.

High/manual scheduler,node001-node006 only.48 shards x8 pairs,4 workers each;
prepare and aggregation separately scheduled. Dependencies use only small
prepare/shard status receipts synced locally; raw/CSV/checkpoints stay server-side.

Launch: source9c36b3b3cf,high/manual prepare t210738,node005;6 tests and8742
consumed-query prefix comparisons passed exactly.45 roots/120223 artifact names/
20783 sampler receipts:zero consumed-cohort conflicts. Only423-byte preparation
summary and small logs pulled. High/manual shards t210739-t210786 all observed
running on node001/004/005/006,192 workers; aggregate t210787 waits for48 small
completion receipts. First old/handoff paths passed full readback. No partial
scientific readout or tuning; evaluate once all768 assigned paths are complete.

Completed:48 shards/768 paths,all12 actor/day readout grids complete (384 cells
each),no finite unresolved labels or missing references. Aggregate t210787,node004;
5000 crossed draws. Gate:not_admitted. Coverage/support/lower-tail gates pass in
both modes/all scopes. Upper-tail upper95 fails clearance,hour2,hour13 and moving-launch;
hour2 mean exceedance10.45%,upper95~11.3%. Score gates fail all/active/clearance/
hour13/moving-launch. Overall paired score:old+0.0208s,CI95[-0.0207,0.0668];
handoff+0.0188s,[-0.0191,0.0621]. Clearance:old+0.5725s,[-1.2707,2.5245];
handoff+0.5308s,[-1.1316,2.3711]. No confirmed score gain,not uniform significant
degradation. V184/V172/V151/V146 unchanged,map updates/training0,main unchanged.
Pulled76433-byte final summary and11444-byte completion receipts;no raw/CSV/ckpt.
Next:stop this candidate's promotion,retain its negative confirmation and frozen
maps. These32 days are consumed; no retuning or replacement cohort for this candidate.
