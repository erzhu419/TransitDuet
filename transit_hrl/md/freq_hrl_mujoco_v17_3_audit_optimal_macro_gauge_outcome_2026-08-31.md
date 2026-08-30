# MuJoCo v17.3 Audit-Optimal Macro Gauge Outcome

## Decision

`audit_optimal_macro_gauge_preflight_not_supported`

This is valid development evidence, not confirmatory evidence. All three
scheduleurm tasks (`t85435`--`t85437`) completed on `node003` with dynamic
placement and no node binding. The synchronized bundle contains only cell
summaries, evaluation CSV files, and server artifact locations; checkpoints and
training histories remain on the worker.

## Paired Mechanics

Each environment trained one strength-zero policy with the full v17.3 policy
state, then evaluated the same frozen checkpoint at strengths zero and one on
the same 40 paths. Across all 120 pairs:

- reward, executed-action, and latent-policy trace hashes matched exactly;
- numeric reward and latent-policy metrics matched exactly;
- router and responsibility reconstruction passed the frozen `1e-7` RMS gate;
- component projection remained below the registered `0.25` mean-rate bound;
- transition counts satisfied the asynchronous hierarchy contract.

The causal gauge is therefore function-preserving on the frozen panel. Its
realized frequency allocation is not supported.

## Frozen Frequency Result

| Environment | Upper-HPF8 reduction | Lower-LPF32 reduction | Joint-merit reduction | Decision |
|---|---:|---:|---:|---|
| HalfCheetah-v5 | -1030.89% | -48.28% | -256.36% | not supported |
| Hopper-v5 | -94.73% | +66.52% | +56.48% | not supported |
| Walker2d-v5 | -733.49% | +72.65% | +72.51% | not supported |

Hopper and Walker2d demonstrate that directly optimizing a causal complement
can remove substantial slow structure from the lower coordinate. That gain is
not sufficient: realized upper-HF power increased in every environment, and
HalfCheetah also worsened lower-LF power and normalized joint merit. The frozen
rule required at least ten-percent upper, lower, and joint improvement in every
environment, so no leakage-active multiseed expansion is permitted.

## Diagnosis

The implementation optimizes an affine HPF8/LPF32 objective over a persistence
forecast for the current macro interval. The registered audit, however, is
computed over complete realized episodes and spans consecutive macro plans.
Independent interval solutions can therefore be smooth internally while
introducing large plan increments at interval boundaries. The upper-HPF8
failure in all three environments is direct evidence of this boundary mismatch.

HalfCheetah exposes a second mismatch. Its long, nonterminating trajectories
make the persistence forecast least representative of the realized policy
sequence. The gauge moved both fast boundary error and forecast residual into
the responsibility coordinates, increasing every registered frequency
endpoint while preserving the plant path exactly.

## Next Mechanism

The next development mechanism must optimize the same streaming audit state
used for evaluation, not a reset finite-horizon surrogate. In particular it
must:

1. carry exact HPF8 and LPF32 filter state across macro boundaries;
2. penalize the realized first-step increment from the previous upper plan;
3. choose the entire next macro curve under bounded component projection;
4. use a causal forecast that is updated by realized total action;
5. expose the complete filter and plan state to the policy before any learned
   constraint experiment.

Fresh development roots are required. v17.3 paths cannot be reused for
selection.

## Claim Boundary

Allowed: v17.3 validates exact pathwise function preservation, shows large
lower-LPF32 reductions in Hopper and Walker2d, and rejects the current
finite-horizon audit-optimal gauge under the frozen cross-environment rule.

Forbidden: v17.3 validates cross-environment frequency separation, leakage
no-tradeoff, reward improvement, learned constraint improvement, optimizer-seed
robustness, fresh-seed confirmation, or a final Freq-HRL algorithm.
