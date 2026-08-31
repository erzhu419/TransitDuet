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

## Engineering-v4 outcome (2026-08-09)

The matrix completed all 72 train shards and all 144 frozen rollouts. The
strict aggregate verifies checkpoint 39, common random numbers, four train
seeds, two disjoint evaluation seeds, one scenario contract, 72 run manifests,
and clean source commit `9685558e1985d9c5fb25fe3803dd0f058c3de716`.
All no-guard and soft variants have exactly zero execution adjustment. Enabled
regularity variants report matched-departure evidence on at least 99.96% of
their evaluation decisions.

No candidate passes the preregistered efficacy gate:

- `softreg_w05_c035_l1e3` has the best mean headway direction among enabled
  regularity variants, but improves CV by only `0.01254`, below the required
  `0.02`, while worsening restricted journey by `0.79557 min` with crossed
  bootstrap 95% CI `[0.15762, 1.36735]`.
- `softdual_c030_l3e4` is the closest joint tradeoff: journey changes by
  `-0.02742 min`, headway CV by `-0.00964`, holding exposure by
  `+0.01306 passenger-min/generated`, and denied-trip rate by `+0.00716`.
  It misses the CV threshold by `0.01036` and does not enable the new
  regularity term; its behavior comes from changing the shared cost limit and
  dual learning rate.
- The best journey row, `softdual_c035_l1e3`, changes journey by
  `-0.18221 min` but worsens CV by `+0.02008`.

Therefore engineering-v4 is a valid negative result. It is not promoted and
does not justify a disjoint-seed confirmation. The complete 16-row screen and
selection uncertainty remain part of the negative-results record.

## Engineering-v5 causal incremental preregistration (2026-08-09)

The v4 diagnosis identifies two structural confounds. First, its regularity
cost uses a matched action-time departure gap that is not exposed to the
policy state. Second, that cost is mixed with the existing safety cost under a
single critic and Lagrange multiplier, so changing the limit changes more than
regularity. Engineering-v5 leaves the same-source `noguard` cost limit and dual
learning rate unchanged and makes regularity a separate incremental reward.

The new state evidence is restricted to deployable, same-time observations:

- exact matched predecessor departure gap and a validity flag;
- for the two-sided arm only, the nearest physically active same-direction
  follower's current AVL spatial gap, converted to an ETA with its observed
  journey-average speed and causal speed fallbacks, plus a validity flag.

The incremental objective compares the local regularity loss before and after
the sampled action. Positive reward is assigned only when the sampled action
reduces that loss; worsening actions receive negative reward. The two-sided
loss predicts that holding increases the predecessor gap and decreases the
follower gap. It never projects, clips, or replaces the policy action.

The exploratory screen contains same-source `main` and `noguard`, context-only
controls `departctx` and `avlctx`, and forward/two-sided reward weights
`0.5, 1, 2, 4`. All use the existing discrete action set, previous-action
feature, cost limit `0.5`, and dual learning rate `1e-4`.

A candidate may advance only if:

1. every requested shard is strict-complete and hash-homogeneous;
2. execution adjustment is zero and all evidence coverage is reported;
3. an AVL two-sided row has at least 50% valid follower evidence in every
   rollout;
4. restricted journey is no more than `0.15 min` worse than same-source
   `noguard` and remains better than the old hard-guard main;
5. headway CV improves over same-source `noguard` by at least `0.02`, without
   reversing its holding and denied-dispatch gains by more than 10%; and
6. only a later disjoint-seed confirmation can support an efficacy claim.

## Engineering-v5 smoke and dispatch record (2026-08-09)

Commit `e8b825e74d430340ecccf25f991a50d6242537c1` passed the complete test suite:
263 tests passed, one was skipped, and 33 subtests passed. A four-job
integration smoke then completed on `node001` and `node002` as scheduler tasks
`t75984` through `t75987`.

The smoke is not efficacy evidence. It verifies that:

- main, same-source `noguard`, `fwdadv_w2`, and `avlbal_w2` all train,
  checkpoint, and complete frozen evaluation;
- both new candidates retain exactly zero execution adjustment;
- the weight-two incremental reward remains within its configured
  `[-0.5, 0.5]` bound;
- `avlbal_w2` reports 62.60% valid same-time follower evidence in frozen
  evaluation, above the preregistered 50% engineering gate; and
- the untrained/one-episode policies still have negative mean regularity
  improvement, so the smoke cannot be interpreted as learned control.

The full exploratory run is
`protocol_v6_incremental_ep40_s4_e2_v5`. It contains 12 configs, four train
seeds `503,521,541,557`, two frozen evaluation seeds `41011,41017`, and 40
training episodes. The 48 one-job shards are scheduler tasks `t76004` through
`t76051`, round-robin pinned to `node001` through `node006` with eight shards
per node. All 48 reached running state. Heavy checkpoints and logs remain on
the HPC filesystem; only later aggregate summaries will be synchronized.

## Engineering-v5 exploratory outcome (2026-08-09)

All 48 train shards and 96 frozen rollouts completed without a traceback. The
strict aggregate verifies checkpoint 39, common random numbers, four train
seeds, two evaluation seeds, one scenario contract, 48 run manifests, and
clean source commit `e8b825e74d430340ecccf25f991a50d6242537c1`. Its frozen
artifact hashes are:

- matrix manifest: `327510397f0cf6dcd059eca25a9bcbf8b755e81914e1af1209da1468f83aaaf1`;
- per-evaluation rows: `5db29b7bb2637327bd4cefd080fb1a2191ae0ca2d1e47f813ae8cd5da87e3a04`;
- summary: `e1dca5d773bc30aedf7a871e2977788071f04d8fdc7eaf6538d57dce0af5d6b8`;
- paired deltas: `3b4ec19c58dd01cbfc00b9d5a597d29da4a11c048b30a5a9b1e1b5e32265d797`.

The fail-closed `freqduet-v6-incremental-selection-v1` audit evaluates every
preregistered AVL weight. `avlbal_w4` is the unique passing row:

- restricted journey delta versus `noguard` is `-0.52630 min`, crossed
  bootstrap 95% CI `[-0.87956, -0.17679]`;
- headway CV delta is `-0.02160`, 95% CI `[-0.04875, -0.00324]`;
- holding changes by `-11016.88 vehicle-s` and denied dispatch events by
  `-37177.63`, so neither no-guard gain is reversed;
- execution adjustment is exactly zero in every rollout;
- minimum predecessor evidence coverage is 100%, and minimum same-time AVL
  follower coverage is 83.44%; and
- regularity baseline-minus-post loss is positive in every frozen rollout.

The lower weights do not pass: `avlbal_w05` misses the CV and holding gates,
`avlbal_w1` worsens CV, and `avlbal_w2` improves CV by only `0.01648`. This
screen is selection evidence only. It does not establish efficacy.

## Engineering-v6 independent-confirmation preregistration (2026-08-09)

The confirmation matrix is locked before launch to `main`, `noguard`, the
context-only `avlctx`, and selected `avlbal_w4`. It uses 40 train episodes,
disjoint training seeds `601,619,641,659`, and disjoint frozen evaluation
seeds `42011,42017,42023,42029`. No additional candidate or weight may be
added after launch.

Confirmation succeeds only if all shards remain strict-complete and
hash-homogeneous, `avlbal_w4` preserves zero execution adjustment and at least
50% follower evidence in every rollout, and the same journey, headway-CV,
holding, and denied-dispatch gates used for selection pass again. The result
must be reported as a negative confirmation if any required gate fails; the
exploratory estimate cannot be substituted for it.

## Engineering-v6 confirmation dispatch record (2026-08-09)

- Frozen source: clean commit
  `fa20bbbb854989794c207ea2c8319e54c5ce16d5` in detached worktree
  `FreqDuet-v6-engineering-v6-snapshot`.
- Run: `protocol_v6_avlbal_w4_confirm_ep40_s4_e4_v6`.
- Scheduler tasks: `t76801` through `t76816`, one config/seed job per shard,
  round-robin pinned to `node001` through `node006`.
- All 16 tasks reached `running`, emitted a confirmation-stage run manifest,
  and completed episode zero. Their source commit, clean flag, and scenario
  hash `7c88934b754c1c22fc0f1695c9732fa8d4f2294db6acd8051cce6852de9c4718`
  are homogeneous.
- The four episode-zero `avlbal_w4` rows retain exactly zero execution
  adjustment and report 53.77% to 58.60% valid same-time follower evidence.
  Their untrained incremental reward is still negative and is not efficacy
  evidence.
- Result synchronization is disabled for train shards. Heavy checkpoints and
  logs remain on the HPC filesystem; only the later strict aggregate will be
  synchronized.

## Engineering-v6 independent-confirmation outcome (2026-08-09)

All 16 train shards and all 64 frozen rollouts completed without a traceback.
The strict aggregate verifies confirmation-stage manifests, checkpoint 39,
common random numbers, four disjoint train seeds, four disjoint evaluation
seeds, clean source commit `fa20bbbb854989794c207ea2c8319e54c5ce16d5`,
and the launch-time source and scenario fingerprints. Its frozen artifact
hashes are:

- matrix manifest: `1a9e90023f2dc0b847934d2693292cceeec9e1602731202a11b9f9b11fbc8d00`;
- per-evaluation rows: `fb82b6481669c2b157a1d9fa736646a645023e0f28718807d8656768fc288d39`;
- summary: `8c476b0398e35bc09a53f8888c63c459aa1cd3943773f421104c81fc1b82f81d`;
- paired deltas: `ed21730c1fbdeac28da31ffc1d7ef2ea713d923dde78a76361ad79a510b8f253`.

The preregistered candidate does not confirm. Against same-source `noguard`,
`avlbal_w4` improves restricted journey by `-1.10516 min`, crossed bootstrap
95% CI `[-2.63355, -0.18993]`, holding by `-12651.56 vehicle-s`, and denied
dispatch events by `-43223.63`. It retains exactly zero execution adjustment,
100% predecessor evidence, at least 82.79% follower evidence, and positive
baseline-minus-post regularity loss in every rollout. However, headway CV
improves by only `-0.01436`, 95% CI `[-0.02799, 0.00124]`, below the locked
`-0.02` threshold. `freqduet-v6-incremental-selection-v2` therefore returns
`no_pass`; the exploratory estimate is not substituted and the candidate is
not promoted.

The matched controls identify the structural source of the failure. Raw
two-sided AVL context alone changes CV by `+0.03076` versus `noguard`; adding
the weight-four regularity objective recovers `-0.04512` versus that raw
context, but only `-0.01436` net versus `noguard`. The reward mechanism is
active and directionally useful; the four-dimensional raw-gap state is the
measured variance/confounding cost.

## Engineering-v7 compact-causal-state preregistration (2026-08-09)

Engineering-v7 preserves the no-guard action semantics and the complete
frozen action-time evidence used by the reward. It changes only what the
policy must learn from that evidence. The four raw predecessor/follower gap
and validity inputs are replaced by two deployable sufficient statistics:

- `regularity_hold_target_norm`: the clipped analytic minimizer
  `clip((follower_gap - forward_gap) / 2, 0, action_cap) / action_cap` of the
  same symmetric two-sided local objective; and
- `regularity_hold_target_valid`: joint availability of the matched
  predecessor departure and same-time physical follower AVL estimate.

The feature is an observation, not an action override. The categorical policy
still chooses from the unchanged action bins; execution adjustment must remain
exactly zero. Invalid evidence emits target zero and validity zero. The reward
continues to evaluate the full raw causal context after the sampled action.

The exploratory matrix is locked to `main`, `noguard`, raw `avlctx`, raw
`avlbal_w4`, compact context-only `avlcompact`, and compact regularity weights
`2, 4, 6, 8`. It uses 40 episodes, four new training seeds
`701, 719, 743, 761`, and two new frozen evaluation seeds `43011, 43017`.

A compact weight may advance only if the fail-closed v3 audit finds exactly
one candidate satisfying every prior engineering-v5 gate and both matched
mechanism gates: CV must improve over compact context-only by at least `0.01`,
and restricted journey may be no more than `0.15 min` worse than compact
context-only. Selection evidence remains insufficient for efficacy; any
selected weight requires another disjoint-seed confirmation.

## Engineering-v7 smoke and dispatch record (2026-08-09)

Clean detached commit `66d15cfd64c110c91b114e1bcf2e137782390fb0`
passed the complete local suite: 274 tests passed, one was skipped, and 33
subtests passed. Scheduler tasks `t78137` through `t78140` then completed a
one-episode integration smoke on `node001` through `node004` for `main`,
`noguard`, compact context-only, and compact weight eight.

All four smoke arms produced an episode-zero checkpoint, frozen evaluation
CSV, evaluation manifest, diagnostics, and a clean run manifest without a
traceback. Their source fingerprint is
`d69b94df107c3f675843126457ee14b940f73c057cea6e28e6fd7134769beada` and
their scenario contract is
`45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17`.
The compact weight-eight arm retains zero execution adjustment. Its frozen
evaluation reports 98.44% matched-predecessor evidence and 81.77% same-time
follower evidence; its observed reward remains within the configured
`[-2, 2]` bound. This is implementation evidence only.

The full exploratory run is
`protocol_v6_compactstate_ep40_s4_e2_v7`. It contains nine configs, four
training seeds `701, 719, 743, 761`, two frozen evaluation seeds
`43011, 43017`, and 40 training episodes. Its 36 one-job shards are scheduler
tasks `t78150` through `t78185`, round-robin hard-pinned with six tasks each to
`node001` through `node006`. Three shards required a second transient launch
attempt; all 36 subsequently reached running state.

Before episode zero, all 36 launch manifests were present with unique
config/seed keys, exploratory stage, clean commit `66d15cfd64`, and one common
source and scenario fingerprint. Heavy checkpoints and logs remain on the HPC
filesystem. No efficacy inference is permitted until all 72 frozen rollouts
are strict-aggregated and evaluated by the preregistered matched-context gate.

All 36 episode-zero diagnostics subsequently completed with 5,200 transitions
per shard and no traceback. Every compact arm retains exactly zero execution
adjustment. Weights `2, 4, 6, 8` remain within their respective observed
reward bounds `+/-0.5, +/-1, +/-1.5, +/-2`. Minimum same-time follower
coverage at this training checkpoint is 44.92%; this is below the final 50%
gate but is not the preregistered frozen-evaluation quantity. It is recorded
without being promoted to either a pass or a failure.

## Engineering-v7 exploratory outcome (2026-08-09)

All 36 training shards and all 72 checkpoint-39 frozen rollouts completed
without a traceback. The strict aggregate verifies the exploratory label,
common random numbers, four training seeds, two evaluation seeds, clean commit
`66d15cfd64c110c91b114e1bcf2e137782390fb0`, source fingerprint
`d69b94df107c3f675843126457ee14b940f73c057cea6e28e6fd7134769beada`,
and scenario contract
`45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17`.
The frozen aggregate hashes are:

- matrix manifest: `cd9ed92389a8048b48a5f49167ae1f9d9a5687b785fbee9cd009891c68b5ab8a`;
- per-evaluation rows: `725a4e4a405131a1d607400e0403d12445f1ed155754a76f0020053bb14911f0`;
- summary: `4a8358881f63fb65d9bc95bcab89896234d1887dba5785ca094d6c4bb3875ea7`;
- paired deltas: `8409959c98db2d786403dcd6c686c9b19404a8a4c97ee9cf871b28df54b1ca4e`.

The v3 matched-context gate returns `ambiguous_multiple_passes`, not a selected
candidate. Compact weight two improves restricted journey by `-0.35381 min`
versus `noguard`, crossed bootstrap 95% CI `[-0.58728, -0.02593]`, and headway
CV by `-0.02530`, 95% CI `[-0.03940, -0.01190]`. Relative to compact
context-only, its journey and CV deltas are `-0.48574 min` and `-0.01248`.
Compact weight four improves restricted journey by `-0.54800 min`, 95% CI
`[-0.84175, -0.25889]`, and headway CV by `-0.02409`, 95% CI
`[-0.04952, 0.00268]`. Relative to compact context-only, its journey and CV
deltas are `-0.67994 min` and `-0.01126`.

Both weights pass every locked absolute and matched-context gate, retain zero
execution adjustment, exceed 83.32% same-time follower coverage, and improve
the measured regularity loss in every rollout. Weight six fails the locked
matched-context CV threshold (`-0.00887` versus required `-0.01`). Weight eight
fails both the net CV and matched-context CV thresholds. No result from this
matrix is promoted as independent confirmation.

## Engineering-v8 compact primary confirmation preregistration (2026-08-09)

The minimum effective dose, compact weight two, is the primary candidate.
Compact weight four is retained as a prespecified sensitivity arm because it
also passed exploration, but it cannot rescue a failed weight-two primary
claim. The controls are hard `main`, same-semantics `noguard`, and compact
context-only. No weight or threshold will be selected from the confirmation
data.

The independent matrix is locked to 40 episodes; training seeds
`809, 827, 853, 877, 907, 929`; and frozen evaluation seeds
`44011, 44017, 44023, 44029`. It therefore contains 30 one-job training shards
and 120 frozen rollouts. All training and evaluation seeds are disjoint from
engineering-v7. Source and scenario fingerprints must remain identical to the
v7 exploratory matrix; the confirmation source must be clean and the manifest
must be explicitly labelled `confirmation` with independent confirmation
enabled.

The fail-closed `audit_protocol_v6_compact_confirmation.py` first verifies
this lineage, then applies the unchanged v3 absolute and matched-context gates
separately to weight two and weight four. The primary claim is eligible only
when weight two alone returns `unique_pass`. Weight four is reported as
sensitivity evidence regardless of outcome and cannot alter the primary
decision.

## Engineering-v8 dispatch record (2026-08-09)

Clean detached commit `54f4a8e66763059274b2d6d5c9f4b2bc5e7ad92a` in
`FreqDuet-v6-engineering-v8-snapshot` passed the complete local suite: 278
tests passed, one was skipped, and 33 subtests passed. The five-arm config
validator also passed without permitting an exploratory config.

The confirmation run is
`protocol_v6_compact_primary_confirm_ep40_s6_e4_v8`. Scheduler tasks `t78895`
through `t78924` were submitted as 30 one-job shards and hard-pinned
round-robin to `node001` through `node006`, five tasks per node. All 30 reached
running state and are visible to the scheduler.

Before episode zero, the shared HPC result tree contained 30 unique run
manifests and 30 diagnostics files, with six manifests for each locked arm.
Every manifest reports confirmation stage, 40 episodes, checkpoint 39, the
six preregistered training seeds, the four preregistered evaluation seeds,
clean commit `54f4a8e667`, and the isolated
`freqduet-cpu-py310` interpreter. Their common model-source fingerprint is
`d69b94df107c3f675843126457ee14b940f73c057cea6e28e6fd7134769beada`,
their common scenario contract is
`45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17`,
and their common launch-analysis fingerprint is
`5f9be9647ce2f6053fccd15b63195c649b188a652aa42a62bcd35b8be1a1b9a8`.
No traceback was present. Episode-zero checkpoints were not yet available at
this launch audit, so this record is dispatch and provenance evidence only.

## Engineering-v8 independent-confirmation outcome (2026-08-09)

Scheduler tasks `t78895` through `t78924` all completed. Each of the 30
training shards contains 40 diagnostic episodes, checkpoint-39 upper, lower,
and runner states, and a four-seed frozen evaluation with no traceback. The
node001 strict aggregate task `t79246` verified 120 unique rollouts, common
random numbers, all run manifests, the clean preregistered source and disjoint
seed sets. Its frozen artifact hashes are:

- matrix manifest: `baff736d7b176668b2895be989e73518872a91690e111a145c02de5998f827ec`;
- per-evaluation rows: `96f36a168106d852c1d859b592ffbc277083c743e8212abec352f3ecb3b7421d`;
- summary: `c273a1bf496e5a5e9a28bb0741322fea26c7b6e685dd02dae64d271f6e861acd`;
- paired deltas: `f8732b7fabf32dcdcc4711392c5fe26316d22979a9bb8e08796060b49bb8ef79`;
- confirmation gate: `a72da292c10a990552db8291a43f92c0c80097217db30153ecfbdee5a1ae8cd6`.

The preregistered primary compact weight two confirms. Against same-source
`noguard`, it improves headway CV by `-0.02231`, crossed bootstrap 95% CI
`[-0.03805, -0.00750]`, restricted journey by `-0.26266 min`, 95% CI
`[-0.83661, 0.17494]`, holding by `-455.21 vehicle-s`, and denied dispatch
events by `-22751.00`. Against compact context-only, its headway CV and journey
deltas are `-0.02677` and `-0.08539 min`. All 12 locked effect, mechanism, and
completeness gates pass across 24 frozen rollouts. Execution adjustment is
exactly zero, same-time follower coverage is at least 83.10%, and regularity
loss improves in every rollout.

The weight-four sensitivity arm does not confirm. Its CV delta versus
`noguard` is only `-0.01448`, 95% CI `[-0.03178, 0.00111]`, so it fails the
locked `-0.02` net-improvement gate even though its journey point estimate is
lower. The final gate therefore returns `primary_confirmed` for weight two and
`no_pass` for weight four, without sensitivity rescue.

The claim boundary is deliberate: compact weight two confirms a statistically
clear headway-regularity gain while satisfying the locked journey no-harm and
mechanism conditions. Its journey interval crosses zero, so this experiment
does not establish a statistically significant journey-time reduction.

Direct CLI execution of the frozen audit initially exposed a module-path
bootstrap defect before any result was read. Running the unchanged frozen
audit with `PYTHONPATH=.` produced the gate above. The maintained script now
adds the project root to `sys.path`, with a clean-environment subprocess test;
no threshold, metric, input artifact, or decision logic changed.

## Engineering-v8 promotion decision (2026-08-09)

`F_freqduet_protocol_v6_avlcompact_w2_hiro` is promoted as the confirmed lower
regularity design. Historical `F_freqduet_protocol_v6_main_hiro` remains
immutable. The new canonical candidate
`F_freqduet_protocol_v6_confirmed_main_hiro` is an exact behavioral alias of
the confirmed row, differing only in config name and protocol role. It must
next undergo fresh-seed long-training and external fixed-headway validation;
the 40-episode confirmation is not relabeled as a final multi-domain paper
matrix.

## Engineering-v9 confirmed-main long-training preregistration (2026-08-09)

The next gate tests whether the independently confirmed 40-episode mechanism
survives unrestricted 200-episode learning. The learned matrix is
`protocol_v6_confirmed_main_ep200_s8_e8_v9` and contains exactly historical
hard-guard `main`, same-semantics `noguard`, compact context-only, and the new
confirmed-main alias. It uses 200 episodes, eight fresh training seeds
`12011, 12037, 12049, 12071, 12097, 12109, 12143, 12161`, and eight fresh
frozen evaluation seeds
`45007, 45013, 45053, 45061, 45077, 45119, 45131, 45137`. This gives 32
one-job training shards and 256 frozen rollouts. No policy or critic freeze is
enabled.

The fail-closed `audit_protocol_v6_confirmed_longtrain.py` binds the parent V8
confirmation gate hash, exact configs and seeds, 200-episode checkpoint,
clean source, model-source fingerprint, and scenario contract. It reuses all
12 V8 effect and mechanism gates and adds three long-training requirements:

- the headway-CV improvement interval versus `noguard` must lie fully below
  zero;
- the restricted-journey delta 95% CI upper bound versus `noguard` must be at
  most `+0.15 min`;
- at least 75% of training-seed mean headway-CV deltas must be negative.

The same clean commit will run direct-scenario external baselines for all eight
evaluation seeds and exactly `fixed_headway`, `rule_holding`, and `rule_mpc`,
as required by the V6 external-comparison contract. This first long-training
gate remains on the base terminal scenario. Multi-domain aliases and the final
paper matrix advance only if the confirmed main survives this test.

## Engineering-v9 dispatch and external-provenance closure (2026-08-09)

Clean detached commit `673355cae10640b2737b6d51d541a9685059c342` in
`FreqDuet-v6-engineering-v9-snapshot` passed the complete pre-dispatch suite:
283 tests passed, one was skipped, and 33 subtests passed. The learned matrix
was submitted as scheduler tasks `t79401` through `t79432`, one config/seed
pair per shard across `node001` through `node006`. Task `t79428` had one
transient SSH launch failure and then launched successfully under the same
task ID. All 32 run manifests are parseable and bind confirmation stage, 200
episodes, checkpoint 199, the preregistered seeds, clean commit `673355cae1`,
the isolated `freqduet-cpu-py310` interpreter, model-source fingerprint
`d69b94df107c3f675843126457ee14b940f73c057cea6e28e6fd7134769beada`,
and scenario contract
`45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17`.
At the dispatch audit all shards were running at episode 6 of 199; no frozen
evaluation artifact existed, so no long-training effect is inferred here.

The first external submission, tasks `t79437` through `t79460`, completed all
24 config/method/seed jobs but exposed a fail-closed launcher defect: the
staged source has no `.git` directory and the external submitter had not
forwarded the frozen Git environment, so every run manifest recorded
`git.commit=unavailable`. These artifacts are provenance-invalid and are
excluded from every comparison and paper table; their values were not used to
select or modify the policy.

The maintained external submitter now forwards the same source commit, branch,
and tracked-dirty fields as the learned-matrix submitter and has a direct
dry-run regression test. The complete post-fix suite passes with 284 tests,
one skip, and 33 subtests. To preserve exact learned/external source identity,
the corrected run used the unchanged clean V9 snapshot with the repaired
environment contract under a new run name,
`protocol_v6_confirmed_main_external_s8_v9r1`. Scheduler tasks `t79461`
through `t79484` all completed. Its 24 manifests cover exactly eight direct
scenario seeds and `fixed_headway`, `rule_holding`, and `rule_mpc`; all report
commit `673355cae1`, `tracked_dirty=false`, the same model-source and scenario
fingerprints as the learned matrix, and no parse error or traceback.

The strict external aggregate reports `strict_complete=true` and verified all
run manifests. Its frozen artifact hashes are:

- per-seed rows: `cfca02acb8946db07b6b3b8d805f096c91f2c6b139b4d9e27b68329d9946519d`;
- summary CSV: `3e87025270c73888916fa99e71d5913a61a996c287c1f4465f627127c519cd65`;
- summary manifest: `37208935911fc5dc3356c0c62fcb850a9b29f31ac86af9f7409490ef26b7699d`.

This closes external-baseline provenance only. Learned-versus-external paired
effects remain unavailable until all 32 learned tasks complete, synchronize,
strict-aggregate, and pass the preregistered V9 long-training audit.

## Engineering-v9 long-training outcome (2026-08-10)

All scheduler tasks `t79401` through `t79432` completed. Each of the 32 shards
contains 200 diagnostic episodes, checkpoint-199 upper, lower, and exact runner
state, and all eight frozen evaluations. This gives 256 unique common-random-
number rollouts with no traceback. Aggregate task `t79641` exited with code zero
and wrote the complete result set, but the scheduler classified its custom
`V9 aggregate complete` tail as a missing success marker. The artifacts were
therefore synchronized and audited directly; this scheduler label is not used
as evidence either for or against the method.

The strict aggregate verifies all run manifests, the preregistered seed grid,
confirmation stage, checkpoint 199, clean commit `673355cae1`, source
fingerprint `d69b94df107c3f675843126457ee14b940f73c057cea6e28e6fd7134769beada`,
and scenario contract
`45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17`.
Its frozen artifact hashes are:

- matrix manifest: `272abd5aa35ccf3202803fa29605b81ed416c3217cfca67ba6f39e4a6ba9af04`;
- per-evaluation rows: `60ef82f3a650fa72faac4af4df0a9d56fa37e87d6ec8cc2bc2c1cdfac0adbc5f`;
- summary: `dfcc293ad15b75a0336fd31f5972e2a1dd68e3bb7af7960db26a3d0bfc4eeea9`;
- paired deltas: `070719eb5f2da9dfc964dfec2033bcbc89bd790b9963c9ada4fbc0789f2b1f0b`.

The preregistered V9 gate returns `longtrain_not_confirmed`. Relative to
same-semantics `noguard`, confirmed-main improves restricted journey by
`-1.24238 min`, 95% CI `[-2.20444, -0.53635]`, holding by
`-9862.89 vehicle-s`, and denied dispatch events by `-75868.08`. Its headway-CV
delta is only `-0.00911`, 95% CI `[-0.02785, 0.00572]`; this misses the locked
`-0.02` effect threshold and does not exclude zero. Seven of eight training-seed
CV deltas are negative, and all causal evidence and zero-adjustment mechanism
checks pass, but those conditions cannot rescue the failed primary gate.
Against compact context-only, the CV delta is `-0.00777`, also below the locked
`-0.01` matched-context threshold.

The valid external comparison remains scientifically useful but does not change
that decision. Learned minus fixed-headway restricted service cost is
`-0.12194`, 95% CI `[-0.18035, -0.05026]`, and headway CV is `-0.21928`, but
restricted journey is `+2.47025 min`, 95% CI `[1.87431, 3.18799]`. The learned
policy is therefore a different service-quality tradeoff, not a passenger-time
winner over strong fixed headway. It is clearly better than rule holding and
rule MPC on restricted journey and service cost. The valid external comparison
manifest hash is
`0741c4050aae1f46be59134ba2cf847dc6cb32c6b5f695e3473a8ff5499ad76f`.

Training trajectories locate the V9 failure after about episode 100. The
two-sided reward mechanism remains active, fully causal, and improves its local
loss, but the candidate CV advantage decays while the long-horizon reward critic
continues training. This is consistent with dilution of a small additive local
credit, not failure of the harmonic decomposition or action-time evidence.

## Engineering-v10 analytic action-dual preregistration (2026-08-10)

Engineering-v10 preserves the complete V9 environment, upper planner, compact
causal state, categorical holding bins, and `noguard` execution semantics. It
adds no action projection, fallback, or post-policy clipping. Instead, the
lower actor receives an independent constrained objective derived from the
same two-sided local regularity loss. If `a*` is the clipped analytic balancing
action, the action-dependent loss is exactly
`((a - a*) / target_headway)^2`, up to an action-independent constant. The
actor evaluates this term for every discrete action and an independent dual
enforces its conditional expectation only where both predecessor departure and
same-time follower AVL evidence are valid. The existing safety cost critic and
dual are unchanged. Frozen evaluations directly report actual action-target
cost, target action, absolute error, and evidence coverage.

The exploratory matrix is locked to historical hard `main`, `noguard`, compact
context-only, confirmed-main, three action-dual-only limits `0.0005, 0.001,
0.002`, and the same three limits combined with the confirmed weight-two
incremental reward. All use regularity dual learning rate `1e-3`, initial dual
`1.0`, and bounds `[0.001, 20]`. The matrix uses 40 episodes, fresh training
seeds `13001, 13007, 13033, 13049`, and fresh evaluation seeds
`46021, 46027, 46049, 46061`. None overlaps engineering-v5 through v9.

A variant may advance to a disjoint 200-episode confirmation only if the strict
aggregate is complete and homogeneous, execution adjustment remains zero,
same-time causal evidence is at least 50% in every rollout, and actual frozen
action-target cost satisfies its configured limit. It must improve CV versus
`noguard` by at least `0.02` with the interval fully below zero, preserve the
locked journey, holding, and dispatch no-harm conditions, improve CV versus
compact context-only by at least `0.01`, and improve CV versus current
confirmed-main by at least `0.005` without worsening journey by more than
`0.15 min`. The exploratory priority is the weakest passing intervention:
dual-only before reward-plus-dual and, within each family, larger cost limit
before smaller. Any selected row still requires disjoint-seed long-training
confirmation; this screen cannot replace V9 or support a paper claim.

The pre-freeze one-episode integration smoke ran compact context-only and
action-dual limit `0.001` under the same train and evaluation seeds. Both
completed training, exact checkpointing, frozen evaluation, and strict local
aggregation. Before meaningful learning they produced identical actions and
outcomes, as expected from matched initialization. The action-dual row reports
zero execution adjustment, frozen causal evidence coverage `0.68115`, actual
action-target cost `0.00460`, and dual `1.03046`; the cost still violates its
limit at episode zero. The context-only control records the same passive
action-target audit while leaving the objective disabled. These values verify
scale, direction, and instrumentation only, not efficacy or constraint
satisfaction after training.

The frozen V10 source is clean detached commit
`3ae72c3debebdcc15695a17398a60237cfb3d475` at
`FreqDuet-v6-engineering-v10-snapshot`. The preregistered run name is
`protocol_v6_actiondual_ep40_s4_e4_v10`. Scheduler tasks `t79679` through
`t79718` cover exactly the 40 config-by-training-seed jobs, one job per shard,
round-robin hard-pinned to `node001` through `node006`. All 40 were running
after dispatch on 2026-08-10. Task `t79711` had one transient SSH handshake
failure on its first node003 launch attempt and then launched successfully on
the same node; it was not duplicated. Runtime inspection of a historical
control, a new action-dual variant, and the retried shard showed live CPU and
memory use. These launch observations establish execution health only. No V10
effectiveness, constraint-satisfaction, or promotion conclusion is valid until
all shard summaries synchronize, strict aggregation succeeds, and the locked
action-dual screen gate is evaluated.

## Engineering-v10 action-dual outcome (2026-08-10)

All scheduler tasks `t79679` through `t79718` completed with exit code zero and
a success marker. The remote shared workspace contains exactly 40 run
manifests and 40 diagnostic sets with no traceback. Strict local aggregation
used only each run manifest plus the hash-locked frozen evaluation CSV and
evaluation manifest, avoiding transfer of 6.1 GB of suppressed training
artifacts. It verifies all 40 runs and 160 unique common-random-number
rollouts. The frozen aggregate hashes are:

- matrix manifest: `6cd03c6ef9368cf9ec27eb2e6cb8e86686abfe14e09dd817e016140565d31e2f`;
- per-evaluation rows: `bd9aa47fea93bf07f9b6bec971bf0521bc47dc1c97d29253bfff07d86b670caf`;
- summary: `ba1301ee04a843d8db8d841beb635c410d20bd8e8a91aa79c33742cccb194ceb`;
- paired deltas: `9b7ad4eed5f291fd853671aa24027629d05f8ee181b71c4ff90ad866348ba976`.

The preregistered V10 screen returns `no_pass`. All six candidates preserve
the no-guard execution semantics, have causal evidence coverage above 0.823,
and keep their duals finite, but none satisfies its actual frozen action-cost
limit and none has a CV interval fully below zero. The dual-only limit-0.001
row has the largest mean CV improvement over `noguard` (`-0.05020`) and also
improves over current confirmed-main (`-0.00688`), but its journey delta versus
confirmed-main is `+0.58775 min`; holding increases by `10699.69 vehicle-s`
and denied dispatches by `50683.19` versus `noguard`. The reward-plus-dual
limit-0.0005 row preserves the passenger and resource gains and improves CV
versus `noguard` by `-0.03373`, but its CV interval has upper bound `+0.02993`,
it is `+0.00959` worse than current confirmed-main, and its maximum rollout
mean action cost is `0.00290` against a `0.0005` limit. It is therefore not a
confirmation candidate.

The training traces identify a structural optimization conflict. Across V10,
the regularity dual grows from 1.0 to approximately 2.3--3.6 and the replay
action cost declines toward 0.0023--0.0026. At the same time, the categorical
SAC temperature rises from approximately 0.05 to 0.077 because its target
remains 98% of the seven-action maximum entropy. The actor is thus asked to
concentrate on a causal holding target and remain nearly uniform over the
holding alphabet at the same states. This explains the action-cost plateau and
seed-sensitive deterministic argmax without changing the causal-decomposer
or execution conclusions.

## Engineering-v11 conditional-entropy preregistration (2026-08-10)

Engineering-v11 preserves the V10 action-dual loss, confirmed weight-two
incremental reward, compact causal state, fixed seven-bin action alphabet, and
zero-adjustment `noguard` execution. It adds an independent entropy
temperature only where `regularity_hold_target_valid` is true. Valid-evidence
states use a lower registered target entropy, while all other states retain
the original SAC temperature and 0.98 target fraction. The conditional
temperature is included in exact training and deployment checkpoints and in
the policy target backup, actor objective, diagnostics, and provenance
contract. It cannot alter the action after policy selection.

The exploratory matrix is locked to historical main, `noguard`, compact
context-only, current confirmed-main, the V10 reward-plus-dual controls at cost
limits 0.001 and 0.002, and six conditional-entropy candidates crossing those
two limits with valid-state target fractions 0.25, 0.50, and 0.75. It uses 40
episodes, fresh training seeds `14009, 14029, 14057, 14071`, and fresh frozen
evaluation seeds `47017, 47041, 47059, 47087`. None overlaps V5--V10.

A V11 candidate must pass strict provenance and completeness checks, retain
zero execution adjustment and at least 50% causal evidence in every rollout,
satisfy its actual action-cost limit, use a lower valid-state temperature than
the ordinary temperature, reduce valid-state policy entropy by at least 0.10
nats versus its same-limit V10 control, and reach its registered entropy target
within a 0.15-nat tolerance. The existing journey, holding, denied-dispatch,
CV-versus-`noguard`, context, and current-main gates remain unchanged. It must
also improve CV by at least 0.005 versus its same-limit V10 control without
worsening journey by more than 0.15 min. Selection prefers the weakest passing
intervention: cost limit 0.002 before 0.001 and target fraction 0.75 before
0.50 before 0.25. The screen remains non-claim-eligible and any selected row
requires disjoint 200-episode confirmation.

The two-episode integration smoke compared the unchanged V10 limit-0.002 row
with the V11 target-fraction-0.50 row under identical train and evaluation
seeds. Both completed training, `freqduet-lower-training-v6` checkpointing,
frozen restoration, evaluation, and strict aggregation. Their actions and
outcomes remain identical at this horizon. The V11 valid-state temperature
independently moved from 0.05 to `0.04912`, while its ordinary temperature was
`0.04934`; the V10 control has no split temperature. Both report causal
evidence `0.65462`, action cost `0.00442`, and zero execution adjustment. These
observations validate the mechanism and instrumentation only, not efficacy.

The frozen V11 source is clean detached commit
`1a1b87517e1f9a8c1e3fd3ba51d80b24ca048eb8` at
`FreqDuet-v6-engineering-v11-snapshot`. The preregistered run name is
`protocol_v6_conditional_entropy_ep40_s4_e4_v11`. Scheduler tasks `t80258`
through `t80305` cover exactly the 48 config-by-training-seed jobs, one job per
shard, with eight shards hard-pinned to each of `node001` through `node006`.
The scheduler watcher launched all 48 on 2026-08-10; direct process telemetry
showed live CPU and memory use and no retries at the first post-launch audit.
These observations establish execution health only. No V11 effectiveness,
constraint-satisfaction, or confirmation conclusion is valid until every task
has a zero exit status and success marker, all frozen artifacts are
synchronized, strict aggregation succeeds, and the locked conditional-entropy
screen is evaluated.

## Engineering-v11 conditional-entropy outcome (2026-08-10)

All scheduler tasks `t80258` through `t80305` completed with exit code zero,
without a retry or synchronization error. The shared remote workspace contains
exactly 48 nonempty run manifests, diagnostics files, frozen-evaluation CSVs,
and frozen-evaluation manifests, with no detected traceback. Strict aggregation
used only the 144 run-manifest and hash-locked frozen-evaluation artifacts. It
verifies 48 runs and 192 unique common-random-number rollouts. The frozen
aggregate hashes are:

- matrix manifest: `6e33bcd06a4b7615b40485025fa733da5349fe1a2439bd2261592a7e645c1bba`;
- per-evaluation rows: `3d36096473fb6333e60d7b1f25609ccc072063e5a9a3964cdd79e6b90e4e9c69`;
- summary: `063ec09886a3eca62eb06fad4b6a36038771e1aefa45bcf07e0002ca9e127fb7`;
- paired deltas: `1d0f1a55c29e987a34572f98ee626c72ef5cb575ecbac0341195ca7eda949470`.

The preregistered V11 screen returns `no_pass`. Conditional entropy works as
implemented: every candidate uses a lower valid-state temperature, reduces
valid-state entropy by more than 0.10 nats versus its matched V10 control, keeps
zero execution adjustment, and significantly improves CV versus `noguard`.
However, all six candidates violate their frozen action-cost limit. The
limit-0.002, target-0.25 row improves journey versus `noguard` by `-0.53165 min`
and CV by `-0.03713` with CI upper bound `-0.01075`, but its mean and maximum
rollout action costs are `0.00271` and `0.00300`; it is also `+0.00351` worse in
CV than its same-limit no-entropy control. The limit-0.001, target-0.50 row
improves CV versus its same-limit control by `-0.01130` while keeping journey
within `+0.13510 min`, but its mean and maximum action costs are `0.00249` and
`0.00271`, and its frozen valid-state entropy remains above the registered
tolerance. Neither row is eligible for confirmation.

The synchronized training traces show a numerical-conditioning failure rather
than an infeasible discrete target. At episode 39 the regularity penalty is only
approximately `0.0055--0.0085`, against a lower policy loss of approximately
`1.8--1.9`; the dual continues to grow and action cost continues to decline,
but the raw normalized-headway squared cost is too small to condition the actor
within the 40-episode screen. This motivates an equivalent rescaling of the
constraint, not a looser limit or an execution-time guard.

## Engineering-v12 normalized-constraint preregistration (2026-08-10)

Engineering-v12 preserves the V11 compact causal state, weight-two incremental
reward, seven action bins, conditional valid-state entropy, analytic two-sided
target, cost limits, and zero-adjustment `noguard` execution. It changes only
the Lagrangian numerical parameterization from `cost <= limit` to the exactly
equivalent dimensionless inequality `cost / limit <= 1`. The actor and dual use
the same scaled cost and residual. Historical configurations default to
`raw_cost_v1`; the new mode is explicit in configuration, diagnostics, training
and deployment checkpoint contracts, and provenance. Frozen diagnostics also
record the nearest-bin oracle action cost so the screen can verify that the
registered limit lies above the discretization floor.

The exploratory matrix is locked to historical main, `noguard`, compact
context-only, current confirmed-main, no-entropy controls at limits 0.001 and
0.002, the two promising V11 controls (target 0.50 at limit 0.001 and target
0.25 at limit 0.002), and six dimensionless candidates crossing initial duals
`0.05, 0.10, 0.20` on those two routes. It uses 40 episodes, fresh training
seeds `15013, 15031, 15053, 15077`, and fresh frozen evaluation seeds `48017,
48041, 48059, 48083`; repository audit finds no prior use of these seeds.

A V12 candidate must pass strict provenance and completeness checks; retain
zero execution adjustment and at least 50% causal evidence in every rollout;
report the exact ratio mode, initial dual, and unit scaled limit; keep the
nearest-bin oracle below one quarter of the cost limit; satisfy the original
unscaled action-cost limit in every rollout; and reduce mean action cost by at
least 0.00020 versus its matched V11 entropy control. The V11 entropy, journey,
CV, holding, denied-dispatch, context, and current-main gates remain. It must
also improve CV by at least 0.005 versus its matched V11 control without
worsening journey by more than 0.15 min. Selection prefers the weakest initial
dual, then the looser 0.002 limit. The screen remains non-claim-eligible and any
selected row requires disjoint 200-episode confirmation.

The two-episode pre-freeze integration smoke compared the unchanged V11
limit-0.002, target-0.25 control with the V12 ratio-scaled initial-dual-0.10
candidate under identical train and evaluation seeds. Both completed training,
exact checkpointing, frozen restoration, evaluation, common-random-number
verification, and strict aggregation. The V12 training checkpoint is
`freqduet-lower-training-v7` and records the exact ratio mode, unit scaled
limit, initial dual, and conditional-entropy contract. At episode one the V11
control reports raw/scaled cost `0.00499`, penalty `0.00530`, and dual `1.06169`;
V12 reports raw cost `0.00469`, scaled cost `2.34723`, penalty `0.24889`, and
dual `0.10614`. Its nearest-bin oracle floor is approximately `7.2e-6`.
Frozen execution records zero guard adjustment and an oracle floor of
approximately `6.1e-6` for both rows. Their actions and outcomes remain
identical at this horizon. These observations validate conditioning,
instrumentation, and restoration only, not efficacy.

## Engineering-v12 normalized-constraint dispatch (2026-08-10)

The frozen V12 source is clean detached commit
`7fee57e65693afd22c7617765b10be2769c60fcf` at
`FreqDuet-v6-engineering-v12-snapshot`. A pre-dispatch inventory found no
existing scheduler signature or remote run directory for the locked run name
`protocol_v6_normalized_constraint_ep40_s4_e4_v12`. Scheduler tasks `t81852`
through `t81907` cover exactly the 56 config-by-training-seed jobs, one job per
shard. The hard placement is ten shards each on `node001` and `node002`, and
nine shards each on `node003` through `node006`.

The first dispatch launched 50 tasks. Six launch attempts encountered a
transient SSH key-exchange closure before a remote process was created:
`t81857`, `t81870`, `t81872`, `t81880`, `t81889`, and `t81898`. The scheduler
watcher subsequently launched those same task records; no replacement or
duplicate shard was submitted. The post-recovery audit found all 56 tasks in
`running` state, each with a live remote PID and approximately one fully active
CPU core. Task execution retry counts remained zero.

Remote process and artifact inspection through the scheduler route compared
the V11 entropy control in shard 25 with the first V12 normalized candidate in
shard 33. Both execute from the mapped V12 snapshot and both have already
written a nonempty run manifest and episode diagnostics. The normalized
manifest records source commit `7fee57e65693afd22c7617765b10be2769c60fcf`
with `tracked_dirty=false`. Its early diagnostics report
`cost_limit_ratio_v1`, initial dual `0.05`, scaled limit `1.0`, raw action cost
approximately `0.0050`, scaled cost approximately `5.0`, regularity penalty
approximately `0.24--0.26`, and nearest-bin oracle cost below `1.2e-5`. The
matched raw-cost control reports a penalty near `0.005` at comparable raw cost.
This establishes source lineage, live execution, and the intended numerical
conditioning only. It does not establish action-cost satisfaction, control
benefit, or eligibility for confirmation; those require all 56 tasks to finish
successfully, strict frozen aggregation, and the locked V12 screen.

## Engineering-v12 normalized-constraint outcome (2026-08-30)

The original scheduler records `t81852` through `t81907` remained disconnected
from the compute nodes for approximately 16 days and were eventually marked
failed from empty scheduler logs. Each record was requeued exactly once as one
of `t84027` through `t84086`. All 56 retry records completed with exit code zero
and a `DONE` marker. The shared result directory contains exactly 56 nonempty
run manifests, diagnostics files, checkpoints, frozen evaluation CSVs, and
frozen evaluation manifests, with no detected traceback. The strict aggregate
verifies all 56 runs, 224 unique common-random-number rollouts, checkpoint 39,
the clean frozen source commit, and the preregistered config and seed grid.

The locked V12 screen returns `no_pass`. Ratio scaling fixes the V11 numerical
conditioning failure: mean frozen absolute action cost falls from approximately
`0.00243` in the matched V11 controls to `0.00059--0.00088` in all six V12
candidates, although one candidate still violates its per-rollout maximum
limit. It does not recover the control tradeoff. The limit-0.002,
initial-dual-0.05 candidate has the strongest CV result (`-0.04295` versus
`noguard` and `-0.00719` versus its V11 control), but worsens journey by
`+0.8510 min` versus `noguard` and fails the holding, denied-dispatch, journey,
and conditional-entropy gates. The limit-0.001, initial-dual-0.05 candidate
keeps a significant CV gain versus `noguard` (`-0.02622`) but worsens journey
by `+0.4237 min`, reverses CV relative to its matched V11 control, and also
fails the holding and denied-dispatch gates.

Training traces explain the tradeoff. Ratio-scaled candidates incur a
regularity penalty of approximately `0.13--1.01` in episode zero, compared with
approximately `0.005` for the raw-cost controls. They rapidly concentrate on
the analytic two-sided balancing action. This improves local headway CV but
forces holding even where its passenger and dispatch cost outweighs the
additional regularity gain. The next objective must therefore constrain only
action-dependent regularity regret relative to zero holding, leaving actions
that already improve on the zero-hold baseline to the reward critic. Further
increases in the absolute-target dual are ruled out by this result.

The strict aggregation command itself exited successfully but printed no
scheduler success marker, causing three unnecessary automatic retries before
the retry cap. The matrix runner now prints `DONE aggregate` after a successful
aggregate. This changes scheduler observability only; it does not alter model,
simulation, or analysis results.

## Engineering-v13 zero-hold regret preregistration (2026-08-30)

V13 replaces the failed absolute-target constraint, not the causal state or
action semantics. Let `d` be the compact causal two-sided balancing action,
`a` the discrete holding action, and `h` the target headway. V12 constrained
`((a-d)/h)^2`. V13 constrains only
`max(((a-d)/h)^2 - (d/h)^2, 0)`, after applying the same bounded cost
contract. Zero holding therefore has zero regret. Actions at least as close to
the balancing target as zero holding also have zero regret and remain ordered
by the learned passenger, load, and dispatch value. Only actions that make the
local two-sided regularity term worse than no intervention are penalized.

The implementation preserves the seven discrete holding bins, compact causal
AVL/APC target features, historical demand prior, conditional entropy target
`0.25`, and zero execution adjustment. Frozen evaluation reports absolute
target cost, zero-hold baseline cost, and positive regret separately. The
training objective and gate use regret; absolute target cost is descriptive and
cannot make a candidate pass.

The exploratory matrix crosses initial regularity duals `0.01` and `0.05` with
regret limits `0.00025`, `0.0005`, and `0.001`, for six candidates. Controls are
historical main, `noguard`, matched `avlcompact`, confirmed main, the V11
same-entropy action-dual baseline, and the best-CV V12 absolute-target
candidate. Training seeds are `16013,16031,16053,16077`; evaluation seeds are
`49017,49041,49059,49083`. At 40 training episodes this is 48 independent
training jobs and 192 frozen common-random-number rollouts.

The locked V13 screen requires complete clean-source manifests, exact configs
and seeds, causal evidence coverage of at least `0.50`, zero execution
adjustment, the exact regret mode and dimensionless scale, every rollout's mean
regret at or below its configured limit, and at least 20 percent mean regret
reduction relative to the V11 same-entropy control. Outcome gates preserve the
V12 resource and `noguard` journey/CV requirements, require noninferiority to
the V11 same-entropy control, and require at least `0.25 min` journey recovery
from V12 without more than `0.02` CV reversal. Selection priority is fixed from
the weakest intervention (initial dual `0.01`, largest limit) to the strongest.
Any selected result remains exploratory and ineligible for a paper claim until
an independent long-run confirmation passes.

A two-episode engineering smoke of the middle `initial=0.01, limit=0.0005`
candidate completed training and frozen evaluation without runtime failure.
The episode-zero policy penalty was `0.04047`, below the `0.13--1.01` initial
range observed in V12, and the dual moved gradually from `0.01` to `0.01055`.
The frozen rollout had causal evidence coverage `0.78231` and mean executed
regret `0.000090`, while absolute target cost remained `0.00569`. This confirms
that the new objective and diagnostics execute as designed without forcing
absolute target tracking. The smoke is not an efficacy result and does not
change the locked 40-episode screen.

## Engineering-v13 zero-hold regret outcome (2026-08-30)

All 48 scheduler shards completed across `node001` through `node006`. The
strict aggregate verifies the exact 12-config matrix, four training seeds,
four common-random-number evaluation seeds, 192 unique checkpoint-39
rollouts, complete run manifests, and clean source commit
`0d08c2e2449bdf9c94d8c178a54ca999dc6b2bc6`. The locked V13 gate returns
`no_pass`; no candidate is eligible for confirmation or a paper claim.

The failure is an outcome tradeoff, not a broken regret constraint. Every V13
candidate satisfies its registered positive-regret limit, reduces regret from
the same-entropy V11 control, recovers journey time from V12, retains zero
execution adjustment, and exceeds the causal-evidence threshold. None reaches
the preregistered valid-state entropy target within 40 episodes. More
importantly, no row satisfies all journey, CV, holding, denied-dispatch, and
matched-control gates.

The weakest-dual, tightest-limit row (`l001/e25/r00025`) has mean action regret
`7.25e-6`; it improves journey by `0.5160 min` and CV by `0.02436` versus
`noguard`, with both confidence-interval upper bounds below zero. It nevertheless
worsens CV by `0.01659` versus confirmed main and by `0.01180` versus the
same-entropy V11 control. The looser `l001/e25/r0010` row is the closest V13
frontier point to V11: it improves journey by `0.3802 min` versus V11 while
worsening CV by only `0.00265`, but it adds `4655.9` holding-vehicle seconds
versus `noguard` and misses the journey and CV confidence gates against that
reference. Stronger initial duals reduce holding and journey further but lose
CV. Mean valid-state entropy spans approximately `0.896--1.191`, rather than
the target `0.4865`.

Synchronized episode traces show that the conditional-temperature optimizer is
active: regularity alpha falls from about `0.0496` to `0.038--0.040`, entropy
falls, and action regret responds monotonically to the configured limit and
dual. The remaining frontier is driven chiefly by average holding allocation.
A further global lambda or regret-limit sweep would move along the same
journey--CV curve rather than resolve it. V14 must therefore use the existing
deployable APC/AVL load, capacity, queue, and causal balancing target to allocate
regularity-improving holding selectively across states while retaining the
zero-hold regret constraint. Replay-level target-by-load allocation is audited
before that objective is implemented.

## Engineering-v14 capacity-gated gain preregistration (2026-08-30)

The V13 replay allocation audit reads the exact episode-39 training replay from
V11, V12, and all six V13 candidates: 32 checkpoints, 6,656,000 transitions,
and approximately 78--81 percent valid compact causal target observations. It
recomputes the bounded absolute target cost, zero-hold baseline, positive
regularity gain, and positive zero-hold regret using the checkpoint's own
feature indices and scales. At matched target-severity bands, V13 already holds
high-load buses approximately 3--5 seconds less than low-load buses; the learned
critic is therefore load-sensitive and the V13 failure is not attributable to
an absent load feature. High-load states nevertheless constitute approximately
43 percent of valid replay events and 34 percent of holding and positive-gain
mass. The tight `l001/r00025` anchor has the strongest journey/resource result,
but lacks enough selective regularity gain to recover CV.

V14 starts only from that tight V13 anchor. It retains the positive zero-hold
regret constraint and adds an actor-only benefit
`w * E[c^p * max(C(0)-C(a), 0)] / 0.002`, where `C(a)` is the same bounded
two-sided action cost, `c` is the current deployable remaining-capacity feature,
and causal-invalid states receive zero weight. The benefit can reward only an
action that improves regularity relative to zero holding; it cannot reward
overshoot, cannot force an execution-time adjustment, and uses no future or
latent state. Its fixed normalization is close to the observed valid-replay
regularity-gain scale and makes `w` interpretable in actor-loss units.

The exploratory grid crosses gain weights `0.005, 0.01, 0.02` with capacity
exponents `1, 2`. All six candidates preserve initial regularity dual `0.01`,
regret limit `0.00025`, dimensionless regret scaling, conditional-entropy
target `0.25`, seven discrete holding bins, compact APC/AVL context, and zero
execution guard. Controls are historical main, `noguard`, compact context-only,
confirmed main, V11 same-entropy absolute target, and the V13 tight anchor.
Fresh training seeds are `17013,17031,17053,17077`; frozen evaluation seeds are
`50017,50041,50059,50083`. Exact experiment-field and result-filename search
found no prior use of these seeds. At 40 episodes the matrix contains 48
independent training jobs and 192 frozen common-random-number rollouts, with the
V13 tight anchor as the paired aggregate reference.

The locked V14 mechanism gate requires complete clean-source manifests; exact
configs, seeds, checkpoint 39, and common random numbers; zero execution
adjustment; the exact V3 capacity-gain contract; initial dual `0.01`; raw regret
limit `0.00025` and unit scaled limit; every rollout's mean regret at or below
`0.00025`; causal evidence coverage of at least `0.50`; positive realized
capacity-gated gain and capacity-gate coverage in every rollout; and exact
agreement, up to the registered CSV rounding tolerance, among realized gain,
its `0.002` normalization, the configured weight, and the reported bonus. The
outcome gate requires CV improvement of at least `0.010` versus the V13 anchor with its
paired confidence-interval upper bound below zero, no more than `0.20 min`
journey worsening versus that anchor, at least `0.25 min` journey and `0.030`
CV improvement versus `noguard`, at least `0.50 min` journey improvement and no
more than `0.005` CV reversal versus confirmed main, and at least `0.25 min`
journey improvement and no more than `0.005` CV reversal versus V11. Holding
and denied-dispatch limits remain `noguard + 10%` of the adverse historical-main
gap. Selection prefers the smallest effective intervention: lower weight first
and exponent two before exponent one at equal weight. A selected row remains
exploratory and requires disjoint 200-episode confirmation; `no_pass` rules out
this fixed capacity-gain family rather than licensing a post-hoc threshold.

### V14 implementation smoke (2026-08-30)

A local two-episode smoke used one disjoint engineering seed and one frozen
evaluation seed for the V13 anchor, the weakest linear-gate V14 candidate, and
the strongest squared-gate V14 candidate. All three jobs completed checkpoint
1, strict aggregation, and checkpoint-restored frozen evaluation. The V13
control reports exactly zero realized capacity gain, scaled gain, bonus, and
capacity gate. The `w=0.005,p=1` candidate reports frozen realized gain
`0.00170836`, mean capacity gate `0.43032956`, and bonus `0.00427090`; the
`w=0.02,p=2` candidate reports `0.00125604`, `0.30484511`, and `0.01256042`.
Both rows satisfy the registered normalization and bonus identities after
eight-decimal CSV rounding. Targeted mechanism, optimizer, checkpoint, and
fail-closed gate tests pass.

The two-episode frozen mean action regret is `0.00096267` for all three rows,
above the formal `0.00025` limit. This is an early-training constraint transient,
not a passing result. The smoke establishes only that V14 is active, causal,
serialized, restorable, and auditable. The preregistered 40-episode screen still
rejects any candidate whose regret limit is missed in even one frozen rollout.

### V14 scheduler dispatch (2026-08-30)

The formal exploratory screen is frozen at clean detached source commit
`4246c7f92aec9f109424b4f7fec3d284bf7b8198` in
`FreqDuet-v6-engineering-v14-snapshot`. Scheduler tasks `t84745--t84792`
cover the exact 48 single-training-seed shards under run name
`protocol_v6_capacity_gain_ep40_s4_e4_v14`. The tasks are distributed evenly
across `node001--node006` at eight shards per node, use the isolated
`freqduet-cpu-py310` interpreter, declare four CPU cores and 1270 MB RAM each,
and retain full training artifacts remotely for one strict node-side aggregate.

Five initial SSH handshakes entered scheduler retry and then recovered without
changing task identity or node assignment. The early health audit found all 48
tasks running with nonzero CPU and RAM samples. This dispatch record is not an
outcome: aggregation, the locked capacity-gain gate, and any candidate decision
remain pending until every shard completes.

### V14 capacity-gain screen outcome (2026-08-30)

All 48 scheduler shards completed. The node-side strict aggregate verified the
exact clean source commit, 12 configurations, four training seeds, four common
evaluation seeds, checkpoint 39, complete run manifests, and 192 unique frozen
rollouts. Only `w=0.02,p=1` passes the preregistered mechanism, outcome, and
resource gates. It improves headway CV by `0.02625` relative to the V13 tight
anchor, with paired confidence-interval upper bound `-0.01321`, while increasing
restricted journey by `0.19028 min`. Relative to confirmed main it improves
journey by `0.62128 min` with CV delta `+0.00194`; relative to V11 it improves
journey by `0.71567 min` with CV delta `+0.00449`; and relative to `noguard` it
improves journey by `1.95547 min` and CV by `0.07047`.

Every frozen rollout satisfies the `0.00025` mean zero-hold-regret limit and
the causal-evidence threshold. Realized capacity gain is positive in every
rollout, its normalization and bonus arithmetic are exact within CSV rounding,
and execution adjustment remains zero. The selected row averages
`45138.75` holding-vehicle seconds and `34623.69` denied-dispatch events, both
inside the locked resource limits. The exploratory gate therefore returns
`exploratory_candidate_selected`, not a paper-eligible confirmation.

### V14 200-episode independent confirmation preregistration (2026-08-30)

The confirmation freezes the exact selected configuration and the same V14
behavior source; it performs no further candidate search. Controls are
historical main, `noguard`, compact context-only, confirmed main, V11
same-entropy, and the V13 tight anchor. Eight fresh training seeds are
`18013,18031,18053,18077,18097,18109,18143,18161`; eight fresh frozen evaluation
seeds are `51017,51041,51059,51083,51101,51119,51143,51167`. Exact scheduler
command-field search found no prior use. At 200 episodes this design contains
56 independent training shards and 448 frozen common-random-number rollouts.

The confirmation requires the exact screen-selected candidate, unchanged clean
behavior source and scenario contract, disjoint seeds, checkpoint 199, complete
manifests, and the same causal mechanism, regret, realized-gain, and resource
checks. It additionally requires CV improvement of at least `0.010` versus V13
with paired CI upper bound below zero, negative CV delta on at least 75 percent
of training seeds, and journey noninferiority versus V13 with paired CI upper
bound at or below `+0.20 min`. The original noguard/current/V11 effect margins
remain locked. Failure of any condition yields `capacity_gain_not_confirmed`;
there is no fallback candidate or post-hoc threshold change.

### V14 200-episode independent confirmation dispatch (2026-08-30)

The locked confirmation was submitted through scheduler as tasks `t85021` to
`t85076` under run name
`protocol_v6_capacity_gain_confirm_ep200_s8_e8_v14`. All 56 one-job shards use
the unchanged clean behavior snapshot at commit
`4246c7f92aec9f109424b4f7fec3d284bf7b8198`, stage `confirmation`, 200 training
episodes, and the preregistered eight training and eight evaluation seeds.
Tasks are pinned across `node001` through `node006` with distribution
`10/10/9/9/9/9`, four declared CPU cores and 1270 MB RAM per shard. Results
remain on the HPC nodes; no checkpoint or shard CSV is synchronized locally.

The post-dispatch health probe found all 56 tasks running with nonzero CPU and
RAM samples. This record establishes execution identity only. Confirmation,
claim eligibility, aggregation, and effect estimates remain pending until every
shard completes and the locked confirmation gate passes.

### V14 200-episode independent confirmation outcome (2026-08-31)

All 56 shards completed without retry or failure. Scheduler task `t85973`
performed the strict node-side aggregate and synchronized only the four small
combined-summary artifacts. The aggregate verifies the exact seven-config
matrix, eight fresh training seeds, eight fresh common-random-number evaluation
seeds, 448 unique checkpoint-199 rollouts, clean unchanged source commit
`4246c7f92aec9f109424b4f7fec3d284bf7b8198`, and the unchanged scenario
contract.

The locked gate returns `capacity_gain_not_confirmed`. Every mechanism check
passes: positive capacity-gated gain is realized in every rollout, causal
evidence coverage and zero execution adjustment hold, the gain arithmetic is
exact, and action regret remains within the registered limit. The candidate
improves headway CV relative to V13 by `-0.00872`, with paired 95% confidence
interval upper bound `-0.00142`, but misses the preregistered minimum effect of
`-0.010`. Restricted journey worsens by `+0.25003 min`, with confidence interval
upper bound `+0.42874 min`, and therefore fails the `+0.20 min` noninferiority
limit. All eight training-seed mean CV deltas are negative, so the failure is
an effect-size and journey tradeoff rather than inconsistent direction.

The allocation evidence identifies the remaining defect. Relative to V13, the
candidate raises mean lower action from `11.88 s` to `13.41 s`, holding from
`61767.73` to `69753.83 vehicle-s`, and denied dispatch events from `59983.42`
to `94101.03`, while reducing mean distance to the analytic target from
`11.56 s` to `9.91 s`. The actor-only benefit therefore buys local regularity
by moving farther toward the full balancing action, but remaining capacity does
not capture the fleet-cycle opportunity cost of those extra holding seconds.
V15 must change the objective's allocation efficiency; changing the V14 gate,
reusing its confirmation seeds, or extending the same global weight sweep is
ruled out.

## Engineering-v15 holding-efficiency gain preregistration (2026-08-31)

V15 preserves the V14/V13 compact causal state, seven discrete holding bins,
zero-hold regret constraint, conditional entropy, remaining-capacity gate, and
zero execution adjustment. It changes only the actor-only regularity benefit.
For discrete action `a`, action range `a_max`, and V14 positive capacity-gated
gain `G(a)`, the new benefit is
`G_eff(a) = G(a) / (1 + beta * a / a_max)`. The multiplier is one at zero
holding, decreases smoothly with the holding time consumed, uses only the
current candidate action, and cannot make a non-improving or overshooting
action positive. It therefore represents holding opportunity cost without
adding a latent fleet variable or changing the deployment state.

The locked grid crosses gain weights `0.025`, `0.030`, and `0.035` with action
efficiency penalties `beta=0.5`, `1.0`, and `2.0`; capacity exponent remains
one. Controls are historical hard main, `noguard`, compact context-only,
confirmed main, V11 same-entropy, the V13 tight anchor, and the same-source V14
selected configuration. Fresh training seeds are `19013,19031,19053,19077`;
fresh evaluation seeds are `52017,52041,52059,52083`. The 40-episode screen
therefore contains 64 independent training jobs and 256 frozen
common-random-number rollouts.

The locked mechanism gate requires the exact V4 policy and V2 efficiency-gain
contracts, unchanged V13 regret settings, positive causal executed gain in
every rollout, an executed efficiency gate strictly below one and no smaller
than `1/(1+beta)`, exact gain/scale/bonus arithmetic, causal evidence coverage,
and zero execution adjustment. A candidate must improve journey by at least
`0.05 min` and CV by at least `0.001` relative to same-source V14 while not
increasing mean action, holding, or denied dispatch. It must also retain the
V14 absolute gates versus V13, `noguard`, confirmed main, V11, and the
historical resource limits. Priority is fixed by lower gain weight and then
lower efficiency penalty. A selected row remains exploratory and requires a
fresh 200-episode confirmation; no pass rules out this objective family.

### V15 implementation smoke (2026-08-31)

A local two-episode smoke used one engineering training seed and one frozen
evaluation seed for V13, V14, and the middle V15 row (`w=0.03,beta=1`). All
three completed training, checkpoint restore, frozen evaluation, and strict
aggregation. V13 reports zero gain. V14 retains efficiency penalty zero and
executed efficiency gate exactly one. V15 reports executed gate `0.81818`,
positive gain `0.00091908`, scaled gain `0.45954`, and bonus `0.0137862`, with
the registered arithmetic. Training diagnostics independently show positive
actor gain and efficiency gates `0.76465` and `0.81592` in the two episodes.
Frozen evaluation correctly reports actor-update metrics as zero because no
optimizer step occurs; the locked gate therefore uses serialized contract,
unit-tested actor mathematics, and executed frozen metrics rather than treating
those zero fields as evidence. The smoke is an implementation result only.

### V15 holding-efficiency screen dispatch (2026-08-31)

The preregistered screen was submitted through scheduler as tasks `t85985` to
`t86048` under run name
`protocol_v6_efficiency_gain_ep40_s4_e4_v15`. All 64 one-job shards use the
clean detached snapshot at commit
`ce472066eb7b7525f5367f81e7259395b932d40c`, stage `exploratory`, 40 training
episodes, and the registered four training and four frozen evaluation seeds.
Tasks are pinned across `node001` through `node006` with distribution
`11/11/11/11/10/10`, four declared CPU cores and 1270 MB RAM per shard. Heavy
artifacts are suppressed and result synchronization is disabled.

Seven first launch attempts encountered transient jump-host SSH connection
closures; scheduler retried the existing task records, and the post-dispatch
probe found all 64 tasks running. Representative node probes showed nonzero CPU
and RAM use, and logs contained no Python exception or configuration failure.
This record establishes execution identity and initial health only. Candidate
selection and every effect claim remain pending complete aggregation and the
locked efficiency-gain screen gate.

### V15 holding-efficiency screen outcome (2026-08-31)

All 64 shards completed successfully. Scheduler task `t86745` performed the
strict node-side aggregate and synchronized only the four combined-summary
artifacts. The manifest verifies the exact 16-config matrix, four fresh
training seeds, four fresh common-random-number evaluation seeds, 256 unique
checkpoint-39 rollouts, clean source commit
`ce472066eb7b7525f5367f81e7259395b932d40c`, and the registered exploratory
protocol.

The locked gate returns `no_pass`; no V15 row is eligible for confirmation.
Every candidate passes the mechanism contract, including positive realized
gain, active capacity and efficiency gates, exact arithmetic, causal evidence,
the zero-hold regret limit, and zero execution adjustment. The failure is an
outcome tradeoff. Relative to same-source V14, every efficiency candidate
reduces or preserves action and holding, but every row has worse mean headway
CV. The closest row (`w=0.025,beta=1`) changes journey by `-0.04948 min` and CV
by `+0.00050`; it narrowly misses the registered `-0.05 min` journey gain and
fails the required `-0.001` CV gain. The strongest journey row
(`w=0.025,beta=2`) improves journey by `-0.14348 min` but worsens CV by
`+0.01769`.

Per-training-seed comparison confirms a state-dependent Pareto tradeoff rather
than one anomalous seed: the near row sometimes increases holding and improves
CV, while in other seeds it suppresses holding and improves journey. A fixed
action-only efficiency penalty cannot distinguish useful anti-bunching holds
from holds made while the fleet is saturated. V16 must condition the soft
opportunity cost on causal fleet utilization; changing V15 thresholds or
continuing a global `weight,beta` sweep is ruled out.

## Engineering-v16 fleet-conditioned efficiency gain preregistration (2026-08-31)

V16 preserves the V13/V14 compact causal target, seven discrete actions,
conditional entropy, zero-hold regret constraint, remaining-seat capacity gate,
and noguard execution semantics. It adds one deployable context feature:
`fleet_utilization`, the fraction of the executable fleet budget currently
reported in service by the AVL roster. It changes only the actor-side positive
regularity gain. For utilization `u`, pressure exponent `p`, and V14 gain
`G(a)`, the registered objective is
`q(u)=clip((u-0.9)/(1.0-0.9),0,1)^p` and
`G_fleet(a,u)=G(a)/(1+beta*(a/a_max)*q(u))`. Below 90 percent utilization it is
exactly V14; at full utilization it becomes the V15 action-efficiency cost.
No action is clipped, replaced, or adjusted after policy inference.

The locked grid crosses gain weights `0.020`, `0.025`, and `0.030`, high-pressure
penalties `beta=0.5` and `1.0`, and pressure exponents `p=1` and `2`, for 12
candidates. Controls are historical hard main, `noguard`, compact context-only,
confirmed main, V11 same-entropy, V13, same-source V14, and the nearest V15 row.
Fresh training seeds are `20013,20031,20053,20077`; fresh evaluation seeds are
`53017,53041,53059,53083`. Exact scheduler command-field search found no prior
use. The 40-episode screen therefore contains 80 independent training jobs and
320 frozen common-random-number rollouts.

The locked mechanism gate requires the exact V5/V3 objective contracts, fleet
utilization and pressure strictly inside their valid causal ranges in every
rollout, nonconstant pressure below saturation, positive realized gain, exact
gain arithmetic, the V13 regret and evidence limits, and zero execution
adjustment. Outcome gates retain all V15 comparisons and resource limits. In
particular, a candidate must improve V14 journey by at least `0.05 min` and CV
by at least `0.001`, with lower action and no increase in holding or denied
dispatch. It must also improve CV over the nearest V15 row by at least `0.001`
while keeping journey within `+0.02 min`. Priority is lower gain weight, lower
penalty, then linear before quadratic pressure. A selected row remains
exploratory and requires fresh 200-episode confirmation; no pass rejects this
objective family without changing thresholds.

### V16 implementation smoke (2026-08-31)

The first two-episode smoke verified V14/V15 backward compatibility and showed
that action-time utilization was high enough for fleet pressure to matter but
that pressure starts `0.75` and `0.90` were too similar. Before any formal seed
was used, the grid was therefore frozen to start `0.90` with exponents one and
two. A second two-episode checkpoint/restore smoke used only the final p1/p2
configuration names. Frozen evaluation reported the same utilization
`0.92418`, pressures `0.43353` and `0.37246`, efficiency gates `0.94534` and
`0.95360`, positive realized gains, and zero execution adjustment. Training
diagnostics independently reported positive actor gain and distinct nonzero
pressure for both exponents. These are implementation checks only and are not
used as effect evidence.

### V16 fleet-conditioned screen dispatch (2026-08-31)

The preregistered 20-config screen was submitted through scheduler as tasks
`t86787` to `t86866` under run name
`protocol_v6_fleet_efficiency_gain_ep40_s4_e4_v16`. All 80 one-job shards use
the clean detached snapshot at commit
`ba19ad20b0ea4de68af1c3933051cc483d23e92b`, stage `exploratory`, 40 training
episodes, and the registered four training and four frozen evaluation seeds.
Tasks are pinned across `node001` through `node006` with distribution
`14/14/13/13/13/13`, four declared CPU cores and 1270 MB RAM per shard. Heavy
artifacts are suppressed and result synchronization is disabled.

The initial bulk launch started 60 shards while 20 launch attempts encountered
transient jump-host SSH resets or remote working-directory probe timeouts.
Scheduler retried the existing task records; a final targeted dispatch reused
the last three queued records without creating duplicate shards. The verified
post-dispatch state is 80 of 80 tasks running. A representative record reports
the exact source commit and protocol arguments, nonzero CPU and RAM use, and
the isolated `freqduet-cpu-py310` interpreter. This record establishes source,
protocol, and initial execution health only. Candidate selection and every
effect claim remain pending complete aggregation and the locked V16 gate.

### V16 fleet-conditioned screen outcome (2026-08-31)

All 80 shards completed successfully. Scheduler task `t86939` performed the
strict node-side aggregate; only the four combined-summary artifacts were
synchronized locally. The manifest verifies the exact 20-config matrix, four
fresh training seeds, four fresh common-random-number evaluation seeds, 320
unique checkpoint-39 rollouts, clean source commit
`ba19ad20b0ea4de68af1c3933051cc483d23e92b`, and the registered exploratory
protocol.

The locked gate returns `no_pass`; no V16 row is eligible for confirmation.
Every candidate passes the registered mechanism, causal-evidence, regret,
arithmetic, and zero-execution-adjustment checks. The nearest row
(`w=0.030,beta=0.5,p=1`) improves journey relative to same-source V14 by
`-0.16169 min`, but worsens headway CV by `+0.00528`. Relative to the nearest
V15 row it improves journey by `-0.03583 min` and worsens CV by `+0.00029`.
Its mean action is `7.94 s`, holding is `41293.8 vehicle-s`, and denied
dispatch is `31047.6`; V14 is `9.18 s`, `47729.7 vehicle-s`, and `37776.7`.
Thus the failure is not missing activation or excess resource use. The
action-dependent multiplier systematically suppresses larger useful holds and
moves the within-state regularity-gain optimum away from the causal target.

The V15 and V16 screens jointly rule out another global action-dependent
`weight,beta,pressure` sweep. The next objective must retain noguard execution
semantics and preserve the V14 per-action gain ordering inside each state.

## Engineering-v17 target-preserving fleet gain preregistration (2026-08-31)

V17 preserves the V14 compact causal target, seven discrete actions,
conditional entropy, zero-hold regret constraint, remaining-capacity gate, and
noguard execution semantics. It also retains the V16 causal fleet-utilization
feature. It removes the action-dependent efficiency multiplier. For V14 gain
`G(a,s)`, fleet pressure
`q(u)=clip((u-0.9)/(1.0-0.9),0,1)`, normalized causal target `t(s)`, target
exponent `r`, and opportunity-cost penalty `beta`, the new actor-side gain is

`G_tp(a,s) = G(a,s) / (1 + beta * q(u) * t(s)^r)`.

For `r=0`, target pressure is defined as one, including a zero target; this is
the target-agnostic state-scalar branch. For `r=1`, high target magnitude
increases the opportunity-cost discount. In either branch every action bin in
the same state receives exactly the same positive multiplier, so the V14 gain
ordering and argmax are unchanged. No action is clipped, replaced, or adjusted
after policy inference.

The locked grid crosses gain weights `0.020`, `0.025`, and `0.030`,
opportunity-cost penalties `beta=0.5` and `1.0`, and target exponents `r=0`
and `1`, for 12 candidates. Fleet-pressure start/full/exponent are fixed to
`0.9/1.0/1.0`; action-efficiency penalty is exactly zero. Controls are
historical hard main, `noguard`, compact context-only, confirmed main, V11,
V13, same-source V14, nearest V15, and nearest V16. Fresh training seeds are
`21013,21031,21053,21077`; fresh evaluation seeds are
`54017,54041,54059,54083`. Exact scheduler command-field search found no prior
use. The 40-episode screen contains 84 independent training jobs and 336
frozen common-random-number rollouts.

The locked mechanism gate requires the exact V6/V4 target-preserving
contracts, action-dependent penalty zero, the registered state-scalar
arithmetic, active causal fleet and target signals, positive realized gain,
the V13 regret and evidence limits, and zero execution adjustment. A candidate
must improve V14 journey by at least `0.05 min` and CV by at least `0.001`,
without increasing mean action, holding, or denied dispatch. It must improve
CV over both nearest V15 and V16 by at least `0.001` while keeping journey
within `+0.02 min`, improve target absolute error over V16, and retain the
historical absolute gates against V13, `noguard`, confirmed main, and V11.
Priority is lower gain weight, lower opportunity penalty, then `r=1` before
`r=0`. A selected row remains exploratory and requires fresh long-training
confirmation; no pass rejects this objective family without threshold changes.

### V17 implementation smoke (2026-08-31)

A three-config two-episode checkpoint/restore smoke verified V13 plus the
`w=0.025,beta=1` V17 rows for `r=0` and `r=1`. Both V17 rows report positive
executed gain, zero action-dependent penalty, zero execution adjustment, and
the same fleet utilization (`0.97030`) and pressure (`0.81955`). The `r=0`
row reports target pressure `1.0` and state-scalar gate `0.58136`; the `r=1`
row reports target pressure `0.42491` and gate `0.82061`, matching the locked
formula.

A separate warmup-free actor smoke forced optimizer updates. Its final episode
reports actor gain `0.00057661`, fleet pressure `0.87154`, target pressure
`0.40377`, and state-scalar gate `0.81443`, with nonzero policy and critic
gradients. Unit tests independently compare V17 against V14 on the same state
and verify that the complete per-action gain vector differs only by one scalar
and has the same argmax. These are implementation checks only; no outcome claim
is made before the locked 40-episode matrix finishes.

### V17 target-preserving gain screen dispatch (2026-08-31)

The preregistered 21-config screen was submitted through scheduler as tasks
`t86967` to `t87050` under run name
`protocol_v6_target_preserving_gain_ep40_s4_e4_v17`. All 84 one-job shards use
the clean detached snapshot at commit
`c6b94265975870990f4a4427dc3b4182d32d13cb`, stage `exploratory`, 40 training
episodes, and the registered four training and four frozen evaluation seeds.
Tasks are pinned evenly across `node001` through `node006`, 14 per node, with
four declared CPU cores and 1270 MB RAM per shard. Heavy artifacts are
suppressed and result synchronization is disabled.

The initial dispatch launched 66 shards and left 18 in scheduler-managed retry
after transient jump-host SSH resets or remote working-directory probe
timeouts. Scheduler reused the original records; no duplicate shard was
created. The verified final state is 84 of 84 tasks running, with exactly 14 on
each node. Representative target-preserving candidate `t87003` is the first
candidate/seed shard (`job-start=36`), reports the exact source commit, clean
tracked state, and isolated `freqduet-cpu-py310` interpreter, and completed
with 685 MB peak RAM. This record establishes source, protocol, and execution
health only. Every effect and selection claim remains pending strict
aggregation and the locked V17 gate.

### V17 frozen-evaluation audit correction (2026-08-31)

After all shards finished and before interpreting any outcome comparison, the
first gate execution exposed one internally inconsistent mechanism check. It
required `lower_regularity_policy_actor_target_pressure_mean` to be nonzero in
the frozen evaluation table. Frozen evaluation deliberately performs no actor
optimizer update, so this and the other actor-update diagnostics are exactly
zero; the V15 audit contract already records that behavior. Execution-side
target pressure, gain, and gate metrics were nonzero and matched the registered
formula in all V17 rollouts.

Gate version `freqduet-v17-target-preserving-gain-screen-v2` corrects only this
diagnostic mismatch. It validates target pressure from the executed frozen
policy and separately requires the frozen actor-update metric to remain zero.
The config list, seeds, source, pairing, mechanism parameters, outcome
thresholds, priority order, and all effect values are unchanged. A regression
test now rejects nonzero actor-update telemetry in frozen evaluation. This is
an audit-contract correction, not a candidate or threshold change.

### V17 target-preserving gain screen outcome (2026-08-31)

All 84 shards completed successfully, with no failed or cancelled task.
Scheduler task `t87122` performed the strict node-side aggregate and
synchronized only four combined-summary files (approximately 1.4 MB). The
manifest verifies the exact 21-config matrix, four fresh training seeds, four
fresh common-random-number evaluation seeds, 336 unique checkpoint-39
rollouts, clean source commit
`c6b94265975870990f4a4427dc3b4182d32d13cb`, and the registered exploratory
protocol.

The corrected locked gate returns `no_pass`; no V17 row is eligible for
confirmation. All strict and mechanism checks pass for all 12 candidates:
executed target and fleet pressure are active, the state-scalar arithmetic is
correct, realized gain is positive, frozen evaluation performs no actor
update, causal evidence and zero-hold regret satisfy their limits, and
execution adjustment remains zero. The failure is entirely in the registered
outcome checks.

The closest CV row (`w=0.030,beta=0.5,r=0`) changes restricted journey by
`+0.01295 min` and headway CV by `+0.00213` relative to same-source V14. It
also raises mean action from `7.5975 s` to `7.6519 s`, holding from
`39501.9` to `39797.2 vehicle-s`, and denied dispatch from `26780.0` to
`27426.8`. Its four training-seed mean CV deltas versus V14 have two positive
and two negative signs, so the small aggregate gap is not a stable hidden
improvement. The strongest journey row (`w=0.030,beta=1,r=1`) improves journey
by `-0.09912 min` but worsens CV by `+0.02029` while reducing mean action to
`6.3950 s`.

Across the grid, stronger state-wide discount generally removes holding and
improves journey at the cost of headway stability. Preserving the V14 gain
ordering inside a state is therefore insufficient: a global state scalar still
changes the gain's strength relative to the passenger critic and produces the
same journey--CV frontier seen in V15 and V16. Another
`weight,beta,target-pressure` sweep is ruled out. The next intervention must
allocate regularity credit to causally attributable high-frequency demand
states, as required by the lower-controller responsibility in `dev_manual.md`,
rather than discount all actions in a state uniformly.

### V17 replay allocation evidence (2026-08-31)

Before designing V18, a node-side read-only audit examined the episode-39
lower replay buffers for same-source V14, nearest V16, the V17 CV-closest row,
and the V17 journey-best row. Scheduler task `t87275` audited all 16 requested
checkpoints, four training seeds per configuration, without copying a
checkpoint locally. Each replay contains 208,000 transitions; approximately
165,000--169,000 per checkpoint have valid compact two-sided evidence. Only a
1.1 MB JSON and 38 KB CSV were synchronized.

The local positive-HF-energy distribution is stable across policies and seeds:
its one-third and two-third quantiles are approximately `0.0209` and `0.0403`,
with the latter varying by less than `0.0004` within every policy. HF-active
states constitute about 30 percent of valid replay. In target-matched bands,
holding falls as HF energy rises. Relative to V14, the V17 journey-best row
reduces mean holding action by about `0.34 s` in HF-inactive states and
`0.64 s` in HF-active states. The V17 CV-closest row instead increases action
mainly in HF-inactive states. Thus the prior global state scalar mixes two
roles: the learned passenger critic already responds to informative local HF
demand, while the analytic regularity prior remains useful when that signal is
weak. This evidence rejects a hard HF threshold and supports a smooth
HF-conditioned opportunity cost on the auxiliary regularity gain.

### Engineering-v18 HF-opportunity gain preregistration (2026-08-31)

V18 starts from same-source V14 and preserves its compact causal target, seven
discrete actions, conditional entropy, zero-hold regret constraint,
remaining-seat capacity gate, and noguard execution semantics. It changes only
the actor-side positive regularity gain. Let `e_H(s)` be the already-observed,
causal local HF-energy feature, `s_H=0.04`, and `beta_H>0`. The registered soft
pressure and state-scalar multiplier are

`p_H(s) = (e_H(s)/s_H) / (1 + e_H(s)/s_H)` and
`g_H(s) = 1 / (1 + beta_H p_H(s))`.

For V14 gain `G(a,s)`, V18 uses `G_H(a,s)=g_H(s)G(a,s)`. At zero HF energy it
is exactly V14. For every state, all action bins receive the same multiplier,
so V14 gain ordering and its analytic target are unchanged. The scale is fixed
from the stable replay two-third quantile before V18 training; there is no hard
activation threshold, post-policy clipping, or execution adjustment.

The locked grid crosses gain weights `0.020`, `0.025`, and `0.030` with HF
opportunity penalties `0.5`, `1.0`, and `2.0`, for nine candidates. Energy
scale/exponent are fixed to `0.04/1.0`. Controls will include historical hard
main, `noguard`, compact context-only, confirmed main, V11, V13, same-source
V14, nearest V15/V16, and both reported V17 frontier rows. Fresh formal
training seeds are `22013,22031,22053,22077`; frozen common-random-number
evaluation seeds are `55017,55041,55059,55083`.

The mechanism gate requires exact V7/V5 contracts, positive and nonconstant HF
energy pressure, the registered pressure/gate arithmetic in actor and frozen
execution telemetry, positive realized gain, V13 causal-evidence and regret
limits, and zero execution adjustment. An outcome pass must improve same-source
V14 restricted journey by at least `0.05 min` and headway CV by at least
`0.001`, without increasing mean action, holding vehicle-seconds, or denied
dispatch. Candidate priority is lower gain weight, then lower HF penalty. A
screen pass remains exploratory and requires fresh 200-episode confirmation;
no pass rejects this objective family without changing thresholds.

### V18 implementation smoke (2026-09-01)

A two-config, two-episode smoke compared same-source V14 with the registered
`w=0.025,beta_H=1.0,s_H=0.04` V18 row using non-formal seeds `21901/54901`.
Training, exact checkpoint restore, and frozen evaluation completed. V18
training telemetry reported HF energy pressure `0.13262`, state-scalar gate
`0.91348`, and positive actor gain; frozen execution reported pressure
`0.16201`, gate `0.89119`, positive realized gain, zero actor-update telemetry,
and zero execution adjustment. The V14 control retained zero HF-opportunity
parameters and an execution gate of exactly one. Unit tests independently
verify the analytic pressure, zero-energy V14 equivalence, per-state action-bin
ordering, runner execution arithmetic, configuration grid, and feature index.
These are implementation checks only. The 23 MB temporary smoke directory was
deleted after recording these values and is not used as effect evidence.

### V18 implementation-failure audit and corrected rerun (2026-09-01)

The first formal V18 execution completed all 80 training shards and 320 frozen
rollouts, but it is not effect evidence for the registered HF-opportunity
objective. All nine candidates improved headway CV relative to V14 by roughly
`0.0056--0.0132` while worsening restricted journey by `0.79--1.09 min` and
increasing mean action, holding, and denied dispatch. This direction contradicted
the registered intervention and triggered a loss-path audit rather than a
threshold change.

The audit found a concrete implementation omission: V7 was registered as a
zero-hold-regret objective, but `_regularity_policy_cost()` did not include V7
in the modes using `(absolute target cost - zero-hold cost).clamp_min(0)`.
Consequently, the first execution used absolute target tracking for its dual
constraint and directly encouraged larger holding. The HF gain gate itself and
its telemetry were correct, but the run did not implement the registered V14
constraint semantics. It is therefore labeled an invalid engineering run, not
a negative algorithm result.

The correction makes constraint-cost semantics an explicit checkpoint and
diagnostic contract (`zero_hold_regret_v2`) instead of deriving it again in the
loss function. The V18 gate now rejects any candidate lacking that telemetry,
and the regression test requires V7 at zero HF energy to equal V14 in both
constraint cost and capacity gain. The nine configurations and all outcome
thresholds remain unchanged. To avoid reusing observed outcomes, the corrected
formal rerun uses fresh training seeds `23013,23031,23053,23077` and frozen
evaluation seeds `56017,56041,56059,56083`; it remains exploratory and requires
fresh 200-episode confirmation if any candidate passes.

### V18 corrected outcome (2026-09-01)

The corrected V18r1 screen completed 80/80 scheduler shards and all 320 frozen
common-random-number rollouts with no failed or retried task. The strict
aggregator verified the clean source commit, exact 20-configuration matrix,
fresh seeds, 40-episode exploratory protocol, V13 reference, unique rollouts,
and run manifests. Every candidate also passed the mechanism gate, including
the explicit `zero_hold_regret_v2` constraint-cost contract, frozen actor,
causal HF evidence, positive realized gain, regret limit, and zero execution
adjustment. The result is therefore valid effect evidence for the registered
objective.

The locked gate returns `no_pass`. The closest resource-preserving candidate,
`w=0.025,beta_H=0.5`, improves restricted journey relative to same-source V14
by `0.13559 min` and does not increase mean action, holding vehicle-seconds, or
denied dispatch, but worsens headway CV by `0.00731` and is not CV-noninferior
to current main or V11. At the other side of the frontier,
`w=0.020,beta_H=0.5` improves V14 headway CV by `0.00309` but worsens journey
by `0.04334 min` and increases action, holding, and denied dispatch. No row
satisfies the preregistered joint outcome gate; no candidate is eligible for
200-episode confirmation or a paper claim.

The corrected result confirms that smoothly scaling the positive analytic gain
by local HF energy still changes its strength against the passenger critic and
does not remove the journey--regularity tradeoff. V18 is rejected as a mainline
objective family. Further work must change how the lower policy represents or
predicts action value, rather than sweep additional gain weights, HF penalties,
or state-wide scalar gates.
