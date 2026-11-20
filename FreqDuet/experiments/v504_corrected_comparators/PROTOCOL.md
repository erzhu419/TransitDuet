# V504: Corrected APC, matched non-harmonic policy and dynamic rule

The V503 align-versus-swap criterion was not met. This study tests two
different questions: whether harmonic policy inputs improve on the same
learners with raw inputs, and whether the learned controller improves on a
dynamic rule with the same control permissions. It is not a retest of V503.

- **Aligned:** reuse each of the four frozen V503 episode-200 policy pairs.
- **NoFreq:** retrain only this arm on the exact V503 seeds, training days,
  action set, rewards, update counts, and model dimensions. Replace the
  demand blocks with the same summaries of the completed one-minute APC
  input and its change before harmonic filtering. The initialization uses the
  same static prior. The shared ledger still uses harmonic LF; this removes
  harmonic actor inputs, not all harmonic information from the system.
- **Manual:** choose from the same 15 upper actions, at the same 900 s
  clock, by nearest directional square-root dispatch target from raw
  completed APC bins. At stops choose the best predicted holding reward
  among the same six bins remaining legal, using causal raw downstream
  boarding rates in place of later training labels. Ties favor smaller
  holding. No extra plan constraints or synthetic slack.

All arms use the same corrected simulator, inventory guard, schedule
projection, 14 physical vehicles, 262 trips, six holding bins, 0.50 retention
and budget accounting. Different trajectories can lead to different
realized holding, even with a matched budget rule.

## Gates and accounting

1. Zero-traffic preflight checks finished-bin APC semantics, action
   equality, public schedule metrics, the new day split and availability
   of the frozen actors. Run the three-arm engineering smoke on the old
   V503 smoke day. The aligned row must reproduce all original metrics
   exactly (excluding wall time), and all arms must match the exogenous
   tape and passenger count, with the full roster and correct scheduled
   learning updates. Smoke is not used to choose policies.
2. After engineering acceptance, evaluate 16 new common MBTA-informed
   simulation days. Train four NoFreq seeds for 200 episodes before their
   evaluation; evaluation results do not enter training and no checkpoint
   selection is performed. There are 6 smoke paths and 944 formal paths
   (800 training, 64 NoFreq evaluation, 64 frozen aligned evaluation,
   16 deterministic manual evaluation). Scheduler high, manual dispatch,
   Linux CPU nodes, no automatic retry, at most nine formal tasks.
3. Paired contrasts are Aligned minus NoFreq and Aligned minus Manual.
   Resample paired training seed IDs and days independently, repeating the
   deterministic rule value across seeds without treating it as trained
   replication. Compare restricted journey, arrival CV2, readiness and
   absolute actual dispatch deviation from the same original timetable
   (decomposed into earliness and lateness). Report each seed, wait, onboard
   exposure, holding, and unserved rate. A joint pass for each contrast
   requires a negative upper 95% endpoint for journey and nonpositive
   upper endpoints for CV2, readiness and absolute schedule deviation,
   under complete service. An apparent benefit with a failed joint gate
   remains an operating trade-off, not a confirmed win. Timetable deviation
   is an operational measure against a common nominal schedule, not an
   objective that makes all retiming intrinsically undesirable.

No main policy changes. Preserve all V500--V503 records. Return only
compact per-day metric summaries; leave weights and training records on
the servers.
