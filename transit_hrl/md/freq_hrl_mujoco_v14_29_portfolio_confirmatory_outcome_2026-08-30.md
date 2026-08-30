# MuJoCo v14.29 portfolio confirmation outcome

## Frozen execution

- Run: `mujoco_v14_29_mechanism_portfolio_confirmatory_20260830_r1`
- Scheduler tasks: `t84930..t84977`
- Frozen algorithm revision: `fc7fa8d8c1e55325af9cb32efece3e0cfc2bbd3c`
- Freq-HRL source manifest: `02f3ba95376021dff0aa11f30d46dd6159e63b55a1d2678d6011ea350745af39`
- Qualified anchor bank: `mujoco_v14_29_fresh_anchor_bank_20260830_r1`
- Statistical unit: one fresh optimizer-seed anchor

All 48 cells ran through `scheduleurm` with dynamic placement across
node001-node006, 24 CPU cores and 16 GB declared per cell, and
`require_node=None`. All tasks completed; none failed or were cancelled.

## Confirmatory result

The frozen analyzer returned `mechanism_portfolio_confirmed`. Results are:

| Environment | Supported | Fail/abstain | Success rate | Wilson 95% CI | Gate |
|---|---:|---:|---:|---:|---|
| HalfCheetah-v5 | 16/16 | 0 | 1.0000 | [0.8064, 1.0000] | pass |
| Hopper-v5 | 16/16 | 0 | 1.0000 | [0.8064, 1.0000] | pass |
| Walker2d-v5 | 15/16 | 1 | 0.9375 | [0.7167, 0.9889] | pass |

Every environment exceeds the preregistered requirement that the two-sided
95% Wilson lower bound be strictly above 0.5. The only unsuccessful cell was
Walker2d-v5 optimizer seed `2978317753`: no candidate passed the frozen design
gate, so the cell abstained and was counted as a failure.

## Selected mechanisms

- HalfCheetah-v5 selected 16 function-preserving router transactions: 12 at
  strength 0.6 and 4 at strength 0.7.
- Hopper-v5 selected 16 function-preserving router transactions: 9 at strength
  0.6 and 7 at strength 0.7.
- Walker2d-v5 selected 6 function-preserving router transactions at strength
  0.6 and 9 paired finite-difference actor transactions; one cell abstained.
- Walker actor steps were `1e-6` once, `1e-5` twice, and `1e-4` six times.

Every selected router matched the paired baseline exactly on executed-action,
reward, and latent-policy traces on all design and validation paths. All 47
supported cells had zero reward-floor violations. Validation frequency-merit
reduction ranged from 38.04% to 60.00% in HalfCheetah-v5, 33.70% to 60.00% in
Hopper-v5, and 0.053% to 43.54% among supported Walker2d-v5 cells.

## Claim boundary

This experiment confirms the preregistered guarded restoration portfolio across
fresh optimizer seeds, conditional on the frozen validation-path panel. It
supports reliable responsibility restoration with a reward floor under this
MuJoCo stress protocol.

It does not establish reward improvement: router transactions preserve physical
behavior by construction, while actor transactions are admitted only when the
held-out reward floor is not violated. It also does not show that every
environment is restored solely by function-preserving routing, because most
supported Walker2d-v5 cells selected actor transactions. Rollout paths remain
paired stress conditions rather than independent statistical replicates.
