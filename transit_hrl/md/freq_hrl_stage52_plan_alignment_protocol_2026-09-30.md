# Stage52: Native Plan Alignment

Stage51 found current-target feedback headroom but negative waypoint-feedback utility. This intervention separates anchor semantics and option-phase prediction under identical renewal budgets; it is not another optimizer variant.

## Frozen Controls

- Source: unchanged Stage42 task_clock warmup16 (warmup2 preflight), same eight roots as Stage51; new evaluation paths only. No actor, critic, standard-deviation or optimizer updates.
- Fixed renewal periods 50 and 100 primitive steps, with the source upper actor called once per renewal in every arm. Gate disabled in every arm. No counterfactual schedule replay or hidden event access.
- Six arms: frozen source lower; Stage51 LQR feedback on original learned waypoint; LQR on the target observed at renewal and held; LQR on that same target anchor plus a causal linear reference curve; opposite-velocity curve control; and current-target feedback at every primitive step as a diagnostic.
- Curves use NumPy least squares on target position versus past relative time, using at most 64 valid observations ending at renewal. At time zero velocity is zero. The intercept is not used: all target-hold/curve arms share the currently observed anchor. Between renewals only option age advances; no new target measurements enter those plans. Evaluated references are clipped to the same environment goal bounds. No route map, future regime, gain/window search or velocity feedforward.
- Upper anchors remain separate from phase-evaluated lower references. All feedback arms retain Stage51 gains, action clipping and source Gaussian standard deviation. Frozen lower retains its original network and original decoded waypoint.
- Full: eight roots, 16 fresh paths, two lower execution modes, two periods and six arms. Preflight: one root, two paths, 300-step episodes. Primary mode is deterministic; sampled execution and reference-error integrals are descriptive.

## Endpoints And Budget

Four return differences at each fixed period: target hold minus waypoint; curve minus hold; curve minus reverse; curve minus frozen. Eight fixed endpoints, equal-root paired bootstrap, 65536 draws, two-sided Bonferroni correction. Phase utility requires curve improvement over hold, reverse and frozen at both periods; no period, mode or checkpoint selection.

Per full root: 460800 native steps, 384 trajectory audits, 6912 upper calls, 460800 lower calls, zero gate calls, 2176 plan regressions and 2176 audit regressions. Full total: 3686400 steps, 3072 audits, 55296 upper calls, 17408 plan and 17408 audit regressions. Preflight: 14400 steps, 48 audits, 216 upper calls and 56 plan plus 56 audit regressions. No extra verification simulation or training. Each full root uses eight persistent workers plus parent, 9 CPU/12 GiB, dynamically placed through scheduler on node001-node006. Raw trajectories remain remote; only compact JSON is pulled.

## Limitations

Analytic plan/feedback controls do not establish learned Freq-HRL improvement or frequency-responsibility separation. Reused roots provide conditional development evidence, not independent confirmation. Curves add explicitly counted regression computation; equal calls are not equal FLOPs. Linear extrapolation may fail at turns or regime changes. A successful curve supports a plan-interface candidate for subsequent learned training, not immediate adoption.
