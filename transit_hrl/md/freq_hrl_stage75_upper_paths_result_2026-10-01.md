# Stage75 result: residual velocity is the dominant execution cost

All t118808..t118816 finished done0. Eight roots, two periods, four frozen-clone interventions, 2048 native episodes /2,457,600 steps /36,864 upper calls. Every root passed sampling, pairing, budget and frozen model/Adam checks. Root computation took61.21..62.07 seconds, observed peak RAM4169..4216MiB. Only completion markers and about59KB of compact JSON were pulled; no native traces or checkpoints.

## Results

All24 registered Bonferroni-corrected root-bootstrap endpoints support harm: all12 reward contrasts are negative, all12 tracking-error contrasts positive, with the same sign in all8 roots for every endpoint. The intervals below use all8 equal-weight roots, 65,536 bootstrap draws and the frozen24-endpoint family.

| Reward contrast | Period50 mean [CI] | Period100 mean [CI] |
| --- | --- | --- |
| Reference only, base velocity | -24.57 [-28.71,-19.72] | -20.15 [-25.32,-14.89] |
| Velocity only, base reference | -116.00 [-127.23,-106.86] | -38.05 [-45.16,-33.28] |
| Both paths, normal minus zero | -158.41 [-172.13,-147.35] | -73.37 [-83.44,-64.63] |
| Factorial interaction | -17.84 [-22.21,-13.76] | -15.17 [-18.49,-11.90] |

Normal-minus-zero tracking error increases by1.8836 [1.6834,2.1235] at period50 and1.0702 [0.9472,1.2793] at period100. Equal-root base rewards are961.28/867.39; normal rewards802.87/794.03. Reference-only and velocity-only changes both hurt, and their negative reward interaction compounds the cost.

Descriptive residual-velocity energy is19.3571 at period50 versus4.8091 at period100 (12-second episodes; RMS1.2701/0.6331). The production derivative uses physical time correctly; an equal position coefficient scale generates larger velocity perturbations at shorter horizons. The causal intervention identifies the velocity pathway's cost, not its training-support explanation.

## Next

Audit the teacher-clone's historical velocity-context support and its actor sensitivity, then preregister a physically calibrated upper residual/velocity budget using only historical training data, not reward-selected scales. This replaces further lower-gradient tuning as the immediate priority. Source policies remain frozen; Stage67 HOLD is unchanged.

## Limits

These are sampled zero-mean upper teacher clones, not jointly trained upper policies. Mixed paths are mechanistic interventions, not deployable plans. This establishes a native execution failure and its pathway attribution, not frequency superiority or a repaired HRL learner. Full per-episode JSON remains server-side; local `qualification_compact.json` retains every root effect, all24 corrected intervals and descriptive mode summaries.
