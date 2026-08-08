# MuJoCo v11 Outcome And v12 Confirmatory Protocol

Date: 2026-08-08

## Evidence Boundary

The v11 matrix is development evidence. Its held-out paths were inspected and
cannot be reused as confirmatory evidence. The v12 design below was written
before any v12 training, safety-selection, or held-out path was evaluated.

## v11 Registered Decision

The complete 36-cell matrix used algorithm revision
`8e47614f1005d8a064a3d6691a0ca6e5bb311ee4` and source manifest
`002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7`.
All cells and serialized checkpoints passed the source and completeness audit.

The exact one-branch invariance gate passed in every environment:

| Environment | Return difference | Raw-action difference | Raw-drift difference | Responsibility-drift reduction |
| --- | ---: | ---: | ---: | ---: |
| HalfCheetah-v5 | 0.0 | 0.0 | 0.0 | 63.99% |
| Hopper-v5 | 0.0 | 0.0 | 0.0 | 53.62% |
| Walker2d-v5 | 0.0 | 0.0 | 0.0 | 85.10% |

Frozen parameter hashes also matched for every paired optimizer replicate.
This establishes the intended structural result: changing only the causal
responsibility attribution leaves the unconstrained learned policy and
environment path unchanged while reducing responsibility-level lower LF
drift.

The registered same-method safe-selector gate failed:

| Environment | Transfer minus additive return | Drift reduction | Gate |
| --- | ---: | ---: | --- |
| HalfCheetah-v5 | +137.215 | 51.15% | pass |
| Hopper-v5 | -20.970 | -60.68% | fail |
| Walker2d-v5 | -15.779 | 89.60% | fail |

Therefore v11 is recorded as `canonical_state_gate_failed`; its safe-selector
result must not be described as confirmatory support.

## Corrected Method Identification

The failed complete gate compared two separately selected constrained methods:
an additive safe selector and a transfer safe selector. Their responsibility
cost states intentionally differ, so they need not select the same branch.
That comparison remains a useful mechanism ablation, but it is not the primary
comparison for the full algorithm.

v12 freezes the actual method-level comparison:

1. **baseline:** canonical additive Freq-HRL with no leakage constraint;
2. **full method:** canonical causal responsibility transfer with the
   trajectory-safe leakage selector.

The full-method selector contains an internal no-leakage branch. For every
environment and optimizer replicate, that internal baseline checkpoint must
match the external additive baseline parameter SHA-256 exactly. Thus any
selected constrained branch is evaluated against the exact fallback policy,
not against a separately tuned baseline.

## Frozen v12 Design

The source-preserving runtime uses the unchanged v11 algorithm revision and
manifest above. Runtime, launcher, specification, and analysis bytes are
committed before dispatch and recorded independently in every result.

- environments: HalfCheetah-v5, Hopper-v5, Walker2d-v5;
- independent optimizer replicates: 24 per environment and arm;
- cells: 72 per arm, 144 total;
- training disturbances: standard, low-frequency, high-frequency, mixed;
- held-out disturbances: the four training families plus OOD chirp;
- held-out paths: 8 seeds by 5 disturbances, or 40 per cell;
- training, checkpoint-selection, safety-selection, held-out, and optimizer
  seed namespaces are pairwise disjoint;
- training budget and all algorithm hyperparameters remain fixed at v11.

The exact seed registries and numeric settings are executable constants in
`scripts/mujoco_v12_confirmatory_spec.py`. The launcher exposes no
hyperparameter or seed override.

## Primary Statistical Gate

The inference unit is an independent optimizer replicate. Paths are averaged
within replicate before paired inference. There are six primary statistical
gates: return and responsibility-drift for each of three environments.

Family-wise alpha is 0.05. Each one-sided gate therefore uses the Bonferroni
confidence level `1 - 0.05/6 = 0.991666...`, with 50,000 paired cluster
bootstrap draws.

For every environment, all conditions must hold:

1. the family-wise lower bound of full-minus-baseline return plus the frozen
   2% noninferiority margin is at least zero;
2. the family-wise lower bound of relative `LowerLFDriftAbs` reduction is at
   least 10%;
3. maximum responsibility reconstruction RMS is at most `1e-7`;
4. source, runtime, checkpoint, seed-role, row-matrix, and internal/external
   baseline audits are valid.

The full claim advances only if all three environments pass. Return
superiority is exploratory; the preregistered return claim is noninferiority.
No threshold, seed, environment, branch, or metric may be changed after v12
held-out access.

## Execution Boundary

Only `node001` through `node006` are eligible. Every cell requests one physical
CPU core, 2 GB RAM, dynamic placement, and no hard node binding. The stale
`jtl110cpu` task records are excluded from the run and from all evidence or
dependency manifests.
