# V166: passive admission and fixed 64-day calibration capture

Source `20d914af03`; high/manual preflight `t208350` on node002 passed
15 server unit tests and both consumed actor12011/day2451001 paths.
Original actions, event arrays, endpoints, deployment contract and terminal clocks
are exact. Against V162's consumed-path artifacts, all 10,400 queries, residuals
and decision traces plus 10,480 ready labels and live snapshots match exactly.
Query state/support/identity and query/policy/label clocks pass readback;
8,742 archived V151 forecasts and their components reproduce.

V165 source `4a6d783667` assigns days2951001-2951064 and samplers
1650000000-1650000063: 12 actors x 64 days x old/handoff = 1,536 paths.
Capture reuses V162's passive instrumentation and query-before-label passes.
All 96 eight-pair shards `t208351`-`t208446` were submitted with high priority
and explicitly passed to manual dispatch, allowed only on node001-node006.
Root: `protocol_v6_v166_fresh_capture/capture_20d914af03` under server results.

No confirmation collection, interval fitting or training is authorized here.
V151, V146, window, tails and main policy remain unchanged; old V162 development
days are not part of this cohort. Only the 4,619-byte preflight summary was
downloaded locally. Next: after all shards finish, admit the complete saved
cohort's metadata and clocks before constructing any frozen joint maps.
