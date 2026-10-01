# Stage64 Value Targets Result

Four tests t116702 passed; actual preflight t116706 and qualification t116709 exited0. Preflight mc_normalized EV0.06329-0.07342 failed the unchanged0.10 floor with only2 warmup iterations. Full preregistration662c7b4032 froze16 iterations, eight roots, both periods/arms and mc_normalized before outcomes.

Full t116713-t116720 and qualification t116775/node006 all exited0. Every32 case passes exact Stage63 raw-GAE reproduction, frozen actor/upper/Adam, initialization and accounting. The preregistered candidate passes EV>=0.10 and MSE below gae_raw in32/32 cases; no roots or failures were removed.

| Treatment | Probe EV range | EV mean | MSE reduction vs gae_raw |
| --- | --- | --- | --- |
| gae_raw | -0.009651 to0.012309 | 0.000946 | reference |
| mc_raw | 0.000180 to0.002767 | 0.001284 | -0.18% to1.02% |
| gae_normalized | 0.149445 to0.393737 | 0.248532 | 75.76% to85.11% |
| mc_normalized | 0.491022 to0.735686 | 0.609815 | 87.22% to94.82% |

MC alone does not restore variance fitting. Fixed normalization does; MC further improves EV and MSE over normalized GAE in32/32 cases. Root310037 now has candidate EV0.540068-0.625502 in all four cases. Raw GAE second-layer Tanh saturation is87.19%-99.88%, versus32.86%-57.84% for mc_normalized; clipping falls from100% to22.34%-44.69%. This supports a scale/representation bottleneck, without isolating saturation as its sole cause.

Full cost:4352 archived episodes,5222400 lower/78336 upper reconstructions,15667200 extra critic scalar calls,20480 value optimizer/forward steps per treatment (81920 total;40960 MC-supervised). All128 critic checkpoints remain remote. No actor updates/native steps/forecaster fits; only103017-byte compact JSON pulled in addition to logs.

## Next

Preregister a paired guarded first-actor-update/native trial using the same eight roots and clone control. Retain the normalized training weights/Adam/frame, export public reward-unit predictions for GAE, and verify unit-consistent continuation before native sampling. Freeze KL guard, path/evaluation budgets and root-level reward contrasts; no automatic full-training launch or post-hoc arm selection.

## Limitations

These are held-out fixed-policy critic-fit results, not native reward gains, frequency-specific attribution, OOD evidence or full-training stability. The critic prerequisite is passed; actor/native performance remains untested. Stage63's negative results and the failed preflight fit remain retained.
