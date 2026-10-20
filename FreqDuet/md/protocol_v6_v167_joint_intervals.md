# V167: fresh 64-day joint-map seal

High/manual task `t208522` on node006, source `e6fc851ec1`, passed 12 tests.
Only V166 days2951001-2951064 were fitted: 1,536 paths, 6,267,583 joint atoms;
no old V162 atoms. V163 estimator `35ae15b185` and V161/V165 rules are unchanged.
Sealed 24 chunks: 11,245 candidate keys, 5,645 finite keys and 48 coarse keys.
Finite requests: 6,234,813; native abstentions: 419,597; low-day-key abstentions:
32,770. Unseen/low-support keys abstain, without coarse pooling or zero fill.

Actor/day-equal finite fractions; required >=0.8 for every mode/scope:

| Scope | old | handoff |
| --- | --- | --- |
| all | 0.932366 | 0.932361 |
| active | 0.933383 | 0.933404 |
| clearance | 0.893785 | 0.892871 |
| hour2 | 0.920688 | 0.920798 |
| hour13 | 0.876436 | 0.874864 |

Support admission passed. All 7,680 actor/day/mode/scope cells are nonempty
with finite estimates; matched coarse references missing: zero. The frozen
rule requires the >=0.8 mean, not >=0.8 in every individual cell.
Maps/atoms/model remain under server `protocol_v6_v167_joint_intervals/seal_e6fc851ec1`;
only the 9,439-byte status JSON was downloaded, not CSVs, atoms or checkpoints.
V151, V146, native window3600s, tails0.1/0.1 and main policy are unchanged.
No new rollouts, training or confirmation. Old V163 seal remains unchanged.
This is calibration support admission, not independent coverage or control gain.
Next: separately authorize the reserved 32 confirmation days, apply these maps
once without updating them, then run the frozen crossed validation gates.
