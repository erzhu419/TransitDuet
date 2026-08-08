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

## Engineering-v3 pilot outcome (2026-08-09)

The exploratory run `protocol_v6_maskguard_ep40_s4_e2_v3` completed all 16
train shards and 32 frozen-evaluation rollouts at source commit
`a34167c9e1e97d641acfba7b103ea6f9f14cdfb9`. The aggregate is strict-complete,
uses four train seeds and two evaluation seeds, verifies common random numbers,
and contains only checkpoint-39 evaluations.

The engineering gates passed:

- All four corrected `maskguard_nofreq` shards crossed upper warmup and
  completed without a state-dimension failure.
- The masked configs enabled the policy mask on every rollout and exposed a
  mean of about `3.62` feasible lower actions.
- Their post-policy execution-guard adjustment is exactly `0.0 s`; the old
  main still requires `26.79 s` of mean adjustment and activates its guard on
  about `75.8%` of decisions.

The efficacy gate failed. Frozen means are:

| Config | Journey (min) | Wait (min) | Headway CV | Holding (passenger-min/generated) | Denied-trip rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| old main | 17.6611 | 5.0333 | 0.1935 | 1.7963 | 0.8822 |
| no guard | 16.5366 | 4.5038 | 0.2579 | 1.1937 | 0.7013 |
| mask guard | 17.5248 | 4.9588 | 0.2009 | 1.7243 | 0.8812 |
| mask guard, no frequency | 17.6221 | 5.0465 | 0.1998 | 1.7372 | 0.8807 |

Relative to mask guard, no guard improves restricted journey by `0.9882 min`
(paired bootstrap 95% CI `[0.4679, 1.5123]`), wait by `0.4550 min`, holding by
`0.5306 passenger-min/generated`, and denied-trip rate by `0.1799`. It worsens
headway CV by `0.0571`. Restricted service cost remains statistically tied.
Mask guard closes only about `12.1%` of the old-main-to-no-guard journey gap.

The frequency control is now executable, but this small pilot provides only a
weak journey result: no-frequency minus mask-guard journey is `+0.0974 min`
with 95% CI `[-0.0054, 0.3046]`. Its wait increase is `+0.0878 min` with 95%
CI `[0.0068, 0.2583]`.

## Decision

Retain the state-dimension fix, policy-action consistency implementation, and
regression coverage. Do not promote `maskguard`, rename it as main, or spend a
four-evaluation-seed confirmation budget on it. The next algorithmic candidate
must start from the coherent no-guard action semantics and recover regularity
through the lower objective or a soft causal constraint, not by projecting or
hard-masking the deployed action.

## Engineering-v4 soft-regularity preregistration (2026-08-09)

Engineering-v4 keeps the `noguard` execution semantics: every discrete action
sampled by the policy is the action sent to the environment. It adds an
optional CMDP cost computed from an immutable action-time tuple
`(matched predecessor departure gap, target headway, action)`. The predicted
post-hold departure headway is `gap + action`; squared normalized deviation
outside a 2% tolerance is capped at one and scaled by the configured dose.
Missing matched-departure evidence produces zero additional cost and is logged,
not imputed from a later bus state. The module never clips, replaces, or masks
an action.

The same source also makes the Lagrange dual statistic use the TPC-weighted
replay occupancy already used by the critics and actor. Consequently all
references must be rerun from this source; old `noguard` outputs cannot be
mixed into the new comparison.

The exploratory screen is a dose-response design:

- regularity cost weight: `0`, `0.25`, `0.5`, or `1.0`;
- per-decision cost limit: `0.35` or `0.30`;
- dual learning rate: `3e-4` or `1e-3`;
- references: locked V6 `main` and same-source `noguard`;
- budget: 40 training episodes, train seeds `503,521,541,557`, and frozen
  common-random-number evaluation seeds `41011,41017`.

The one-episode implementation smoke produced 5,200 lower transitions. It
reported zero execution-guard adjustment, 65.7% valid matched-departure
evidence, and an unscaled weight-one regularity-cost mean of about `0.356`.
This smoke is a scale check only and is not efficacy evidence.

No screened variant is promotable from this pilot alone. A variant may advance
to disjoint-seed confirmation only if it satisfies all of these gates:

1. Every requested shard and frozen rollout is complete and hash-homogeneous.
2. Mean execution adjustment remains exactly zero and evidence validity is
   reported for every rollout.
3. Restricted passenger journey is no more than `0.15 min` worse than
   same-source `noguard` and remains better than the old hard-guard main.
4. Headway CV improves over same-source `noguard` by at least `0.02` in mean,
   without reversing its holding and denied-dispatch gains by more than 10%.
5. Selection uncertainty and all 16 screened variants are retained in the
   negative-results appendix; only a later disjoint-seed run can support an
   efficacy claim.

## Engineering-v4 dispatch record (2026-08-09)

- Frozen source: `9685558e1985d9c5fb25fe3803dd0f058c3de716` in detached
  worktree `FreqDuet-v6-engineering-v4-snapshot`.
- Run: `protocol_v6_softreg_ep40_s4_e2_v4`.
- Scheduler tasks: `t74910` through `t74981`, one config/seed job per shard,
  round-robin pinned to `node001` through `node006`.
- Initial launch: 72 of 72 tasks reached `running`. Four transient jump-host
  SSH key-exchange failures were retried without duplicating a shard.
- Result transfer: `summary` scope only. Training logs and final checkpoints
  remain on the HPC filesystem and are not automatically copied to WSL.
- The regularity module and runner SHA256 values were identical on all six
  nodes: `b64e5f864142008afe31a91cee1cb19298a81af9776af461db0dea1557d127d5`
  and `9b2bde0bd64ec1e6c08e059a17824dfd921f3e53eed68ef624895de354b57cd2`.
- The first weight-0.25 shard completed episode zero with 5,200 transitions,
  zero action adjustment, 64.4% valid evidence, regularity-cost mean `0.0945`,
  replay batch-cost mean `0.7501` against limit `0.35`, and lambda `1.009`.
  This remains implementation evidence only.
