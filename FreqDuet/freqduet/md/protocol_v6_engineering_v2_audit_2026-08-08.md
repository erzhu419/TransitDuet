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
