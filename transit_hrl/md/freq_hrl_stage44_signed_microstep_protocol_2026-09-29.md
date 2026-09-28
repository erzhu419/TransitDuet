# Stage 44: signed microsteps along the first lower update

Stage 43 did not establish a universal wrong task direction or a finite-update
return loss. It did find 8/32 updates reducing their original PPO objectives.
This read-only diagnostic distinguishes local response from full displacement;
it does not change the learner or claim a repair.

## Frozen Design

Use all four Stage-42 methods, checkpoints 16/17 (preflight 2/3), original eight
roots 310011/310023/310037/310049/310061/310073/310089/310101. Preflight 310001.
For each actual lower actor displacement, including log standard deviation,
evaluate alpha = +1/16, -1/16, 1, plus a shared alpha = 0 reference. Zero and full
actor endpoints are exact; upper/gate and all critics stay at each before
checkpoint. Before actors and fixed networks must match across methods.
The one microstep fraction is fixed here, not selected from evaluation returns.

Fresh native streams: 10900000 + root-index*10000 + 3001..3016; preflight
10893001..10893002. Upper/gate deterministic; both deterministic and sampled
lower evaluated. Sampled lower noise uses SeedSequence(44,root,env,44019)+step.
Same paths and noise across all 13 policies. No training-batch reconstruction,
optimizer steps, selection, root exclusions, budget extensions or scale search.

Primary: four comparisons per method in sampled deployment: positive micro
minus zero, negative micro minus zero, (positive-minus-negative)/(2/16), full
minus positive micro. Equal-root paired bootstrap, 65536 draws, seed(44,44044),
two-sided Bonferroni-16 percentile intervals. Deterministic effects and full
minus zero are descriptive. Positive micro/signed slope together with negative
full-minus-micro can support bounded finite-step attenuation, not its cause.
The central difference is finite, not an exact derivative.

## Budget And Limits

Full: 8 roots * 13 policies * 2 modes * 16 paths * 1200 = 3993600 native steps,
3328 offline trace audits. Preflight: 15600 steps, 52 audits. No extra environment
verification steps. Scheduler only, dynamic node001-node006, 9 CPUs/12 GiB per
full root, 8 persistent workers; preflight 2 CPUs/4 GiB. Raw traces and source
weights stay remote; only compact JSON is local. Reused development weights
make this conditional diagnosis, not independent confirmation or paper utility.
