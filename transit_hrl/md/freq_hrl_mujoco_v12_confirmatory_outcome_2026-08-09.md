# Freq-HRL MuJoCo v12 Confirmatory Outcome

Date: 2026-08-09

## Decision

The pre-registered v12 comparison is `confirmatory_supported`.

The comparison was frozen before held-out access:

- baseline: additive responsibility with `freq_hrl_no_leakage`;
- full method: causal LF transfer with `freq_hrl_safe_selector`;
- environments: `HalfCheetah-v5`, `Hopper-v5`, and `Walker2d-v5`;
- 24 independent optimizer replicates per environment and arm;
- 40 held-out paths per cell (8 seeds by 5 disturbance modes);
- 144 complete cells and 5,760 held-out evaluation rows;
- family-wise alpha 0.05 across six primary gates;
- primary return gate: 2% noninferiority margin;
- primary leakage gate: at least 10% reduction in `LowerLFDriftAbs`;
- maximum responsibility reconstruction RMS: `1e-7`.

All three environments passed return noninferiority, the minimum leakage
reduction gate, and responsibility reconstruction. The internal no-leakage
checkpoint trained inside every full-method selector exactly matched the
corresponding external baseline checkpoint.

## Primary Results

| Environment | Return delta, full minus baseline [95% CI] | Family-wise NI lower bound | Responsibility LF-drift reduction [95% CI] | Family-wise drift lower bound | Selected branches |
| --- | ---: | ---: | ---: | ---: | --- |
| HalfCheetah-v5 | 118.704 [53.010, 198.837] | 50.200 | 89.60% [79.10%, 94.26%] | 76.77% | no-leakage 14; projected 3; scalarized 7 |
| Hopper-v5 | 20.241 [5.193, 38.992] | 6.251 | 78.23% [69.90%, 85.28%] | 67.89% | no-leakage 8; projected 16 |
| Walker2d-v5 | 31.402 [17.650, 46.177] | 20.262 | 89.56% [85.13%, 93.55%] | 84.15% | no-leakage 9; projected 7; scalarized 8 |

Return superiority is exploratory. The registered return claim was
noninferiority; no threshold was changed after result access.

## Independent Raw-Table Audit

An independent read of the paired replicate table and all cell CSV files
verified:

- exactly 24 unique optimizer seeds per environment;
- no overlap among training, checkpoint-selection, safety-selection, and
  held-out evaluation seeds;
- exactly 40 registered held-out condition rows in every cell;
- complete environment, disturbance, and seed coverage in both arms;
- no duplicate cell identity;
- positive drift reduction in all 72 paired replicates.

The paired return distribution contains exact zeros when the safe selector
falls back to the checkpoint-identical no-leakage branch. As a robustness
diagnostic, one-sided paired Wilcoxon p-values were 0.002531, 0.001892, and
0.000327 for HalfCheetah, Hopper, and Walker2d, respectively. These diagnostics
were not substituted for the frozen primary analysis.

## Integrity

- frozen algorithm revision: `8e47614f1005d8a064a3d6691a0ca6e5bb311ee4`;
- frozen algorithm manifest: `002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7`;
- confirmatory runtime revision: `2a68860607ba95d59da518653d3d78e9aca051ce`;
- launcher SHA-256: `2b394855fae5862aa59077e7803cf937692f52f108a17d38be2beccff073293a`;
- runtime SHA-256: `b5e740423ee1343c066ecd575d9168ad7771911936c85344889f3068fa07e73c`;
- spec SHA-256: `114efc29eaf34a73e5ed24c1d4bb57b2bef240abd075128d03b001e9d281a0c4`;
- decision SHA-256: `548df0b336181ee7e46795cc27874b5f34bf73d49fddd78e266ea627493c52d2`;
- paired-row SHA-256: `041007d91fca72b811b04c3c63cb9fbdc2a84b443bf31a1d084553215037eb83`.

The analysis artifacts are under
`transit_hrl/results/mujoco_v12_confirmatory_analysis_20260808_r1` in the
frozen v12 worktree. `jtl110cpu` tasks were not used by this experiment or its
analysis.

## Claim Boundary

Supported wording:

> Across three continuous-control environments, the pre-registered full
> Freq-HRL method reduced lower-controller low-frequency responsibility drift
> while satisfying a family-wise return-noninferiority gate against its
> checkpoint-matched additive-responsibility baseline.

The positive return intervals may be reported as exploratory superiority.

Do not rewrite `LowerLFDriftAbs` as physical action smoothing. It is a
responsibility-space leakage measure. The exact no-leakage fallback preserves
raw policy behavior while causal transfer changes responsibility attribution;
raw-action spectral claims require a separately registered external behavioral
metric.
