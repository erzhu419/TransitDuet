# Stage287: frozen selectors on new prefix scenes

Stage249-286 improved diagnosis, not independent evidence that the new selector
beats Stage247. Stop changing heads on those development caches.

## Frozen comparison

- Five candidates: raw, unchanged Stage247 `state_gls_pooled_safe`, uniform
  training-mean service offset (`linear_offset_clip`), action-mean level,
  observed-state level. The last three use raw risk, raw feasibility gate,
  original action anchoring and the Stage282 clip. No contrast additions.
- Preserve Stage247's original two-head/scalar-path scoring and selected
  strength. Forcing raw risk into that historical model would change the baseline.
- Six probes per scene: 8, 9, 22, 23, 36, 37. Equal average of 32 frozen
  historical mixed-view cell/bank profiles; these are not 32 independent seeds.
- Exact replay of 19,200 cached choices without labels precedes any new native
  rollout. No fitting, model selection or lower-policy updates.

## New evidence

1. Software pilot: 10 old replica1 bridge scenes and 10 separate new scenes,
   one probe and fresh draw0 each. Check causal replay, physical accounting,
   frozen weights, identity and resource usage, not performance selection.
2. Bridge: those 10 old scenes, 32 new futures (draw1-32), excluding pilot draw0.
3. Main: 80 genuinely new scenes, eight in each of 2 roots x 5 regimes,
   64 new futures each. The 10 new pilot scenes are excluded.

One native source episode captures all six preaction contexts and reference
snapshots per scene. Each future uses the existing exact prefix/state/RNG
checks. Only within-profile policy disagreements are replayed, plus native
identity. Zero-disagreement states remain in the six-state scene denominator.
Continuation jobs use four futures per chunk. Full artifacts stay on servers.

## Decision

Primary contrasts are observed-state level minus uniform and minus Stage247.
Require both total cost and wait upper 95% prefix-bootstrap bounds below zero
for both comparisons. Stratify resampling by root/regime; bootstrap 80 scenes,
not probes, futures or frozen profiles. Report paired conditional MC separately.
Bridge evidence has conditional MC uncertainty only, with one scene per stratum.

- Bridge reproduces but new scenes fail: next test independent training-scene
  coverage while retaining a simple structure.
- Bridge fails against uniform: stop the complex level branch; retain uniform
  and Stage247 as controls.
- New scenes pass joint primary comparisons: next test repeated closed-loop
  interventions, without declaring end-to-end Freq-HRL validated here.

## Limits

New prefixes remain within the two historical frozen lower roots and five
regimes. This is a single-intervention test, not new-domain, training-seed or
full learned-HRL validation. Primary contrasts measure improvements relative
to uniform and Stage247; absolute benefit against raw is reported separately.
Stage248/253 confirmation labels remain closed.

Run: `native_transit_frozen_prefix_stage287_20261017_r1`. Register frozen code
before dispatch; qualify pilot before main prefix/continuation collection.

## Execution

Seven focused unit tests passed. Freeze `t166452` on node004 reproduced all
19,200 cached choices exactly, with zero fits or native episodes. Initial
`t166434` stopped after action replay because the server lacked the old
reserved-scene registration; only that 12,497-byte metadata file was transferred
before retry. Frozen models remain server-only; the pulled receipt is 2,659 bytes.

Pilot `t166461`-`t166480`: all 20 tasks completed with exit0, 99 native episodes,
maximum RSS 680.64 MiB, mean 27.53 seconds/episode. `t166495` qualified the
software-only pilot without reading performance labels. Only a 2,668-byte
pilot summary was pulled. Main prefix/continuation tasks reserve 1.5 GiB/CPU
(over twice the pilot maximum), with no node pin or scientific contract change.

All 80 new-prefix capture tasks `t166511`-`t166590` completed with exit0.
Seal `t166591` fixed 435 disagreement states out of 540 states in 90 scenes;
the 1,360 continuation tasks `t166592`-`t167951` planned 104,448 native episodes.
The main plan pull is 159,674 bytes; no snapshots, trajectories or checkpoints
were pulled.

The initial main batch failed the strict snapshot check across worker processes
and was stopped: 126 failed and 1,360 cancelled, including automatic retries.
It supplies no performance evidence. The original pilot captured and replayed
in one process and did not cover this boundary. Native construction forks and
restores the caller's Torch RNG; deterministic evaluation leaves it unchanged.
Replay now restores the archived caller stream before starting each episode,
retaining the original full state/RNG comparison. Models, choices and future
seeds are unchanged. Main restart requires separate-worker checks on all six
probes in all 20 archived pilot prefixes (240 identity/observer episodes).
