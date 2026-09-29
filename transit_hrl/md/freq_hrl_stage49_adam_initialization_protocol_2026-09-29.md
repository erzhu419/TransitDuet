# Stage 49: actor Adam initialization under fixed episode KL

Stage48 holds the training KL budget but does not establish learning utility.
Test GAE/full-task time-LOO MC crossed with inherited/fresh actor Adam, all four
under the same maximum training-episode KL **0.1** and13-trial LR backoff.
Fresh means clear actor Adam state exactly once at branch start; keep actor
weights and optimizer parameter groups. Never reset critic Adam; retain exactly
the first original GAE critic update. Subsequent actor Adam states accumulate
normally. No objective gate, evaluation access, new warmup or auxiliary baseline.

Task-clock Stage42 first GAE update:checkpoint17 (pre3), original PPO settings.
This fixed source already exists; no new warmup. Exact first native
batches/rewards and first critic/Adam update must match across four arms. Record
source/start/end per-parameter Adam step counts and validate round continuity.
Trials restart independent actor/critic/Adam copies with identical shuffle;
accepted actual scaled-LR state retained. Rejected executions are charged.

Sixteen rounds x eight paths,H1200; pre two x four,H300. First/fixed-final only;
sampled lower primary,deterministic descriptive; upper/gate networks fixed.
Roots310011/310023/310037/310049/310061/310073/310089/310101; pre310001.
Fresh base11500000+index*10000,pre11490000;train1..128/pre1..8,eval3001..3016/pre3001..3002.
SeedSequence49/root/env/49017 policy,49019 lower noise; shuffle49/root/round.

Nine registered sampled-return effects: fresh-MC minus inherited-MC first/final,
fresh-GAE minus inherited-GAE final,fresh-MC minus fresh-GAE final,reset-credit
difference-in-differences final,and each of four final arms minus frozen.
Equal-root paired bootstrap65536 draws,seed(49,49049),two-sided Bonferroni9.
Reset repair requires positive final fresh-MC minus inherited-MC AND frozen;
credit-specific reset evidence also requires positive registered interaction.

Full7680000 native steps/6400 audits;960000/root:614400 train+345600 eval.
Pre20400/68. Nominal full20480 actor/20480 critic steps,max266240 each with
13 trials; executed/retained counts and KL checks separate,zero extra native
verification. Dynamic scheduler node001-node006,9CPU/12GiB per root,pre2CPU/4GiB.
Only compact JSON local; raw/weights server-only. No tuning,exclusions,seed extension.

Version2 replaces version1 before any native run. Unit `t104184` passes24/26;
the pipeline correctly catches empty actor Adam at critic-only warmup16/pre2.
All nine source records have zero warmup actor steps;first learning performs40
actor steps/full or4/pre. Reset there would be a null intervention. The other
unit error is fixture upper-call accounting, corrected without a protocol change.
Version1 freeze `7804d9b00b` is retained,not scientific evidence.

## Limitations

A common KL upper bound is not equal realized displacement or a population
trajectory bound. One-time reset changes first/second moments and Adam step
counter together; it does not isolate them. Reused roots are development only.
