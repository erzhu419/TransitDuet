# Stage63 Paired Episode Credit

Stage62 found substantial option/episode advantage disagreement and nearly constant upper values. Isolate lower episode credit with consistent lower critic calibration, leaving upper repair for a separate experiment. Freeze the same eight roots, periods50/100, zero_train/joint_ppo, Stage55 clones, Stage57 calibration/first training archives and inherited PPO settings. Root310001 is preflight only. No new environment paths or KL/seed/fit-budget sweep.

Option control must exactly reproduce Stage60 backtracking diagnostics. Episode treatment starts from the same lower actor/value/Adam, calibrates only its lower value on episode-continuing GAE with the same16 iterations/eight paths and nominal value steps (preflight2/two), then performs one guarded lower PPO update. Reconstruct source actions/log probabilities once; evaluate each treatment's own scalar old values. Episode boundaries survive concatenation, while artificial option terminals are removed. Upper calibration/update runs once and final upper actor/value/Adam states are shared exactly. KL0.02, shrink sequence and guard semantics are unchanged.

Probe critic MC fit and actual pre-actor GAE alignment on the first training batch, which is disjoint from critic calibration. Full native prerequisite: every active actor moves under the KL budget, and every episode critic has positive episode-MC EV and lower episode-MC MSE than the option critic on this probe. Preflight is mechanical qualification only. Failed cases remain visible; no native reward experiment starts automatically.

Full cost:4352 reconstructed archive episodes,5222400 lower/78336 upper calls plus5222400 extra episode-critic scalar calls;1536 critic calibration calls,80 observed first updates,1616 core PPO GAE/176 extra diagnostic GAE calls,160 distribution/160 value diagnostic passes,64 probe MC calls and64 candidate checkpoints. Nominal optimizer counts derive from source config; count all guard candidates, interpolation and rollback work separately. All archives/checkpoints remain remote; only compact JSON is pulled. Scheduler dynamic node001-node006,9CPU/12GB full,2CPU/4GB preflight/tests/qualification; source/protocol and preregistration committed before full outcomes.

## Limitations

This isolates credit plus its necessary lower critic recalibration, not credit alone with a stale critic. A shared upper path is valid for this frozen first batch, not later on-policy training where lower trajectories diverge. Reused teacher-initialized roots and calibration-probe MC fit are development evidence, not native reward, frequency-specific, OOD or full-training confirmation. Positive EV is a prerequisite, not proof of a calibrated value or a performance gain.

## Execution

Implementation committed as eef5006ecf. Scheduler t116620/node004 passed five Stage63 tests plus the existing Stage46 regression (six tests,54.120s). Actual archive preflight t116623/node006 and qualification t116626/node001 passed exact Stage60 control reproduction, shared upper states, critic probe criteria and complete cost accounting. Full archive comparison is next; native evaluation remains gated.
