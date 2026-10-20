# V166: fixed 64-day capture and readback

Capture source `20d914af03`; V165 protocol `4a6d783667` assigns days
2951001-2951064 and samplers1650000000-1650000063, excluding old V162 days.
High/manual preflight `t208350` on node002 passed 15 tests: original endpoints,
actions, arrays and terminal clocks exact; 10,400 saved queries/traces/residuals,
10,480 ready labels/snapshots and 8,742 V151 forecasts/components reproduce.
All 96 high/manual shards `t208351`-`t208446` finished with zero retries:
12 actors x 64 days x old/handoff = 1,536 paths on node001-node006 only.

High/manual readback `t208474` on node006, source `c990e3c99b`, passed 11 tests
and all 1,536 paths; zero failed paths. Each actor has 128 paths, each day 24.
All 7,987,200 queries and 8,048,640 live snapshots have complete state/support
and valid query/policy/label clocks; late policy load never enters query state.
All 6,687,180 eligible requests have resolved same-physical-follower labels.
Joint native-supported vectors: 6,267,583. Explicit native abstentions: 419,597;
invalid follower evidence: 1,300,020 requests, retained outside the denominator.
Old/handoff have 2,825/2,820 keys with at least 19 days (5,623/5,622 observed keys).
These are descriptive support counts, not interval fitting or release admission.

Raw artifacts remain on server. Only 4,619-byte preflight and 24,193-byte
readback JSON summaries were downloaded. No replay, fitting, training,
confirmation collection or frozen V151/V146/window/tail/main-policy change.
Next: separately authorize V161 joint-map construction on these 64 days only;
seal maps and evaluate frozen support admission before opening confirmation.
