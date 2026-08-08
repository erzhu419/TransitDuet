# Freq-HRL MuJoCo v13 Behavioral Confirmatory Outcome

Date: 2026-08-09

## Decision

The frozen v13 decision is `confirmatory_primary_gate_failed`.

The experiment completed all 144 cells and 5,760 registered held-out paths:

- three environments;
- 24 paired optimizer replicates per environment and arm;
- 40 held-out paths per cell;
- family-wise alpha 0.05 across 12 one-sided statistical gates;
- one deterministic responsibility-reconstruction integrity gate per
  environment.

Ten of the 12 statistical gates passed. HalfCheetah failed the raw lower-action
low-frequency reduction gate, and Hopper failed the absolute upper
high-frequency RMS budget. The global behavioral claim therefore fails even
though every return-noninferiority and responsibility-drift gate passed.

## Frozen Results

| Environment | Return delta [95% CI]; family-wise NI lower | Responsibility LF reduction; family-wise lower | Raw lower LF reduction; family-wise lower | Upper HF RMS; family-wise upper | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| HalfCheetah-v5 | +96.059 [32.206, 176.069]; 26.808 | 74.89%; 65.84% | 5.22%; -7.75% | 0.0634; 0.0799 | failed raw-lower gate |
| Hopper-v5 | +13.025 [3.820, 24.158]; 5.927 | 73.15%; 57.02% | 58.04%; 32.03% | 0.1068; 0.1178 | failed upper-HF gate |
| Walker2d-v5 | +23.406 [11.618, 36.420]; 13.522 | 90.37%; 87.05% | 38.39%; 15.52% | 0.0240; 0.0263 | passed |

Selected full-method branches were:

- HalfCheetah: no-leakage 15, projected 5, scalarized 4;
- Hopper: no-leakage 9, projected 15;
- Walker2d: no-leakage 11, projected 3, scalarized 10.

Maximum responsibility reconstruction RMS was `8.26e-9`, `1.85e-8`, and
`2.07e-8`, respectively, below the frozen `1e-7` integrity limit.

Return superiority is exploratory. The registered return endpoint was
noninferiority, and no threshold or endpoint was changed after v13 result
access.

## Diagnosis

The v12 selector optimizes a return floor and `LowerLFDriftAbs`, which is a
responsibility-space quantity after causal transfer. Its no-leakage fallback
can therefore lower attributed responsibility drift while preserving the raw
lower policy exactly. This is useful for responsibility assignment but cannot
guarantee a reduction in `RawLowerLFDriftAbs`.

The same selector has no explicit upper high-frequency safety endpoint. In
v13, full-method upper HF RMS was 1.07, 1.20, and 1.42 times the matched
baseline value in HalfCheetah, Hopper, and Walker, respectively. Hopper crossed
the prospective absolute budget.

The next development version must therefore change the algorithm rather than
relax the analysis:

1. constrain raw lower LF behavior in addition to transferred responsibility;
2. add an upper plan-continuity or upper-HF safety constraint;
3. select branches from independent safety seeds using return,
   responsibility drift, raw lower drift, and upper HF endpoints jointly;
4. treat all v13 outcomes as development information and use fresh seeds for
   any later confirmatory version.

## Integrity

- frozen algorithm revision:
  `8e47614f1005d8a064a3d6691a0ca6e5bb311ee4`;
- frozen source manifest:
  `002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7`;
- frozen runtime revision:
  `3718a7ea5623be8746eb8a88272237cd30e939bf`;
- launcher SHA-256:
  `0e66f11316e1f2bd77181413371711a9e429af9fd17a8b9349cc5eabd44905df`;
- runtime SHA-256:
  `fc83c752f0cab17f37207291dffe6db6989a87b55cc4f26441219cdfaa91bf06`;
- specification SHA-256:
  `09cd6525789bf785aa834150cd8a1547a2ba4628e0480d592c0685a367c38f27`;
- analyzer SHA-256:
  `bb8baad824525b50bc7646121d4dac95270f97e086b11de8cf546cc17d270916`;
- decision SHA-256:
  `1cfb52b4e403756a3c947161f44df44aab7a57954832de01936b78f9d3bb7e0b`;
- paired rows SHA-256:
  `a839d442667162d395825b3d276b896169fd72b6a0e578870f0462dc4249f216`;
- environment gates SHA-256:
  `552522b3da6a10c0abf37d0dda6c9f88dfee6b6d4b7027493edddcaa5845987c`.

The immutable result artifacts remain under
`transit_hrl/results/mujoco_v13_behavioral_confirmatory_*_20260808_r1` in the
detached v13 worktree. No `jtl110cpu` task contributed to this experiment.

## Claim Boundary

Supported wording:

> Under fresh seeds, the full method met family-wise return noninferiority and
> reduced responsibility-space lower LF drift in all three MuJoCo environments;
> raw lower LF reduction and the upper HF budget did not jointly replicate.

Disallowed wording:

> MuJoCo v13 confirms that Freq-HRL universally improves external frequency
> behavior without tradeoff.
