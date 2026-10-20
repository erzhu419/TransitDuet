# V162: V161 calibration capture only

Collect V160/V161's 32 calibration days, 2851001-2851032, all 12 archived
actors and old/handoff: 768 frozen paths. Reuse the reserved matched samplers.
Confirmation days are not enumerated by the collector. No training or interval
fit; V151, main policy, V146 3s/0s, 3600s and .1/.1 tails stay unchanged.

Add passive ready-time snapshots: physical identities, AVL position/speed,
callback order, and counts/clocks of observed events. Query load stays absent.
V151 points, native stage/support, condition keys and component predictions
are reconstructed from the strict completed prefix and written for every lower
query before a separate outcome pass attaches same-vehicle labels and joint
running/service/residence residuals. Opaque, invalid and low-support rows remain.
Raw live snapshots and reconstructed predictor support are distinguished.

Admission first replays two consumed V145 paths and requires exact endpoints,
actions, event arrays, terminal clocks, and archived V151 forecasts/components.
Only after admission may calibration shards run through scheduler on nodes
001-006. Traces, records, support and residual files stay server-side; only
small preflight/progress/completion summaries come back locally.

Admission passed on node006 (`t208239`, source `25a2eddeaf`): nine tests,
10,400 query records, 8,742 archived V151 forecasts reproduced across both
consumed paths; all original actions/timing/terminal clocks remained exact.
The initial launch lacked the existing runtime receipts; these were restored
server-side and the private runtime path was made explicit. No fresh day ran
during the failed launches. The local preflight summary is 600 bytes.
Calibration is dispatched as 48 shards of eight actor/day pairs, eight workers
per shard, restricted to node001-node006. Confirmation and interval fit stay off.
Scheduler IDs: `t208240-t208287`; each requests eight CPU cores and no GPU.
Scheduling update: unfinished FreqDuet jobs use high priority and manual
dispatch. The calibration batch has no failed retries; the old preflight
startup failures are separate historical tasks. No data recollection required.

Next: audit the completed calibration metadata and support, without fitting
intervals or opening confirmation until separately authorized.
