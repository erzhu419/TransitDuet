# V172: V171 temporal joint-map construction

Implementation `a5b45df520`; scheduler `t210167`, high priority, four CPU cores,
manually dispatched to node006. All 22 tests passed and all 24 actor/mode maps sealed.

- Use only the admitted V166 64 calibration days2951001-2951064, 12 actors, old/handoff: 1536 existing paths. No new collection or replay.
- Apply the single V171 candidate: original key plus hour2/hour13/other. Recompute equal-day/episode/query masses within each new key using the unchanged V163 estimator and ranks; preserve complete running/service/residence vectors.
- Keep V167 conditional and coarse maps unchanged. Copy the coarse reference without refitting; retain the original conditional bounds separately for matched secondary comparisons.
- Require at least19 distinct days per new key. Low-support and unseen keys abstain; no coarse fallback into candidate, no dropping native abstentions.
- Fitted distinct-day support and every actor/day/scope cell must reproduce the sealed V171 precheck. Read back stored weights, finite masks and bounds before sealing the model.
- V151, V146, the 3600s native window, tails0.1/0.1, controller/actors/critic/dual and main policy remain unchanged. The 32 reserved confirmation days are not started or read.

Accepted: 6,267,583 candidate atom weights recomputed, 5,559,359 changed; maximum
key-mass normalization error 6.82e-14. Of 17,469 keys, 8,518 are finite and 8,951
retain low-support abstention. All 419,597 native abstentions and 44,788 key-support
query abstentions remain in the denominator. All 9,216 support cells match V171;
both modes/all six scopes pass. Hour2 finite support is 90.18%/90.27%; hour13
moving-launch is 98.66%/98.79% (old/handoff). Saved array readback passed.

Only summary/status.json (85,873 bytes) was retrieved from
freqduet/results_freqduet/protocol_v6_v172_temporal_intervals/seal_a5b45df520/.
Next: keep this map sealed and await separate authorization for the reserved
32-day confirmation; no collection, replay or training was started here.
This seal is not independent coverage, tail-performance or policy evidence.
