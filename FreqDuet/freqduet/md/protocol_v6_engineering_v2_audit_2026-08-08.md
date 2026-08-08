# Protocol V6 engineering-v2 audit (2026-08-08)

## Evidence boundary

- Frozen source commit: `95f406a49fabd075014ee6eeb33d3309d2f0c475`.
- Run: `protocol_v6_engineering_ep40_s4_e4_v2`.
- Protocol: 40 training episodes, four train seeds, four disjoint evaluation
  seeds, and common-random-number frozen evaluation.
- Completed evidence: 44 of 48 unique train shards. The four missing shards
  are all `F_freqduet_protocol_v6_nofreq_hiro`.
- The 44-run aggregate is provisional. It excludes `nofreq` and must not be
  used by the locked V6 promotion decision.

## Failure diagnosis

The four `nofreq` shards fail deterministically at episode 30, when the upper
policy first becomes active:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x18 and 19x64)
```

V6 disabled frequency features but inherited `upper.state_dim`; unlike V5,
the runner did not derive the upper state dimension from the environment. The
four retries therefore had no scientific value and were cancelled. `nofreq`
remains mandatory because it is the direct control for the frequency claim.

## Provisional effect result

The current V6 main is not promotable even before repairing `nofreq`:

- Main restricted passenger journey: `19.3297 min`.
- `noguard - main`: `-1.7248 min`, bootstrap 95% CI
  `[-2.7278, -0.9147]`.
- `noguard` also reduces restricted wait by `1.1129 min`, denied-trip rate by
  `0.2042`, and holding exposure by `0.6230 passenger-min/generated`.
- `noguard` increases headway CV by `0.0575`, so the old guard trades a large
  passenger-level loss for a narrower regularity improvement.
- `rawhistory`, `allfreq`, `upperonly`, `loweronly`, `swapped`, `nobudget`,
  `noloadcost`, `waitonlycredit`, and `CSAC` are statistically tied with main
  on the primary endpoint at this development budget. Several trend slightly
  better than main.

The guard is active on about `76.5%` of main decisions. The learned policy
requests about `43.8 s` on average, while the executed action averages
`16.8 s`; the no-guard policy learns only `11.2 s`. The critic is trained on
projected actions, but the actor and target backup optimize unprojected action
probabilities. This creates an out-of-support action-value loop rather than a
coherent constrained policy.

## Repair under test

The engineering-v3 candidate makes the causal action limit an auditable lower
observation and applies a state-dependent categorical mask inside policy
sampling, target backup, actor loss, entropy targeting, and deterministic
evaluation. The existing execution guard remains as a final invariant check.
It also derives V6 no-frequency upper dimensions from the environment and adds
a regression that executes the first post-warmup upper decision.

The new configs are explicitly exploratory:

- `F_freqduet_protocol_v6_maskguard_hiro`
- `F_freqduet_protocol_v6_maskguard_nofreq_hiro`

They cannot enter the locked V6 validator without the explicit
`--allow-experimental` switch.

## Promotion gate

Do not rename or promote the candidate unless all of the following hold:

1. All requested shards finish with homogeneous source and analysis hashes.
2. `nofreq` crosses upper warmup without a state-dimension failure.
3. Post-policy execution-guard adjustment is numerically zero apart from
   floating-point tolerance.
4. The masked policy materially closes the journey gap to `noguard` without
   losing the latter's holding and denied-dispatch improvements.
5. A follow-up four-train-seed, four-evaluation-seed matrix confirms the pilot
   direction with paired intervals.

If the masked candidate fails these gates, repeated retries are not justified.
The next valid choices are to remove the hard deficit guard from the main
method or redesign the lower objective and causal action set.
