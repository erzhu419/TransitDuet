# Stage64 Continuing Critic Targets

Stage63 reduces MSE but produces almost constant continuing values. Separate bootstrap-target and output-scale effects with a fixed2x2:gae_raw,mc_raw,gae_normalized,mc_normalized. Use the same eight Stage55 clones, periods50/100, zero_train/joint_ppo, Stage57 calibration archives and disjoint first-training probe; root310001 is preflight only. All actors, upper values and their Adam remain frozen. No new native paths, hyperparameter/seed sweep or architecture change.

Raw GAE value-only training must exactly reproduce Stage63 pre-actor episode-MC probe metrics. The slim critic-only loop preserves source loss coefficient, gradient clipping, Adam, LR, epochs, minibatch order and nominal value steps. MC targets are the existing discounted complete-episode returns; count this additional supervised calibration explicitly.

Both normalized arms use one fixed mean/std from the first calibration episode-MC batch only. Rebase the last linear head to preserve initial public predictions (float32 tolerance2e-4); source clone value Adam must be empty, not reset. Train/export in explicitly different units. Public predictions/bootstrap stay in reward units; checkpoint stores normalized training weights/Adam, mean/std and public weights, not a native-ready full policy. Probe values/returns never affect training, normalization or arm choice.

Freeze mc_normalized as the candidate before results. Full critic prerequisite:every case EV>=0.10 (at least10% return variance explained) and MSE below gae_raw. The stronger EV floor excludes Stage63's mean-only repair. Keep all roots, including310037. A pass still needs a separate guarded actor/native trial; no native trial starts automatically. Report saturation fractions, hidden variability, gradient clipping and all four fits as mechanism diagnostics, not performance evidence.

Full fixed costs:4352 archive episodes,5222400 lower/78336 upper reconstructions and15667200 extra critic scalar calls;2048 value calibration calls,1024 GAE target calls,544 MC target calls,128 probe GAE calls,614400 initialization prediction rows,2432 representation batches and128 server-only critic checkpoints. Value optimizer/forward counts derive from source config (four matched arms); MC-supervised steps reported separately. Dynamic scheduler node001-node006,9CPU/12GB full and2CPU/4GB tests/preflight/qualification. Only compact JSON/logs pulled locally.

## Limitations

Fixed-policy retrospective critic calibration does not establish on-policy reward gains, frequency attribution, OOD generalization or full-training stability. Activation saturation is measured rather than assumed causal. Unit reparameterization changes value optimization, not just a printed loss scale.

## Execution

Implementation aa2f0103aa; scheduler t116702/node005 passed four focused tests in45.286s. Raw critic/Adam equivalence, normalized public predictions, frozen actor/upper and actual GAE/MC/optimizer accounting passed. Actual archive preflight is next; native evaluation remains HOLD.
