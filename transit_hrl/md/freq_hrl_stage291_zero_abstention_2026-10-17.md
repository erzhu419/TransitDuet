# Stage291: zero-event abstention versus matched thinning

Reuse Stage290 frozen proposals and Stage288 training futures only. Primary
proposal/screen/evaluation banks remain 0-7 / 8-15 / 16-23; all six bank-role
permutations are fixed. No fits, simulations, models or held rows are loaded.

- Reject Stage290-accepted proposals if all eight paired screening fleet
  differences are zero. Keep other nonpositive-mean proposals, including
  nonzero signed cancellation. No threshold grid or action reselection.
- Match uniform random thinning exactly in kept count per profile/root/regime/
  probe. Report its exact expected value, not a selected random seed. Freeze
  choices and inclusion probabilities before evaluation. In each prefix
  bootstrap, recompute matching counts using screening membership only.
- Use 60-prefix stratified bootstrap5000, seed291, with shared-future MC SE.
  Joint retention requires cost/wait CI95 upper <0 versus raw and negative
  points in both evaluation halves. Cost/fleet information versus matched
  thinning is reported separately. Profiles/permutations are not new seeds.
- One scheduler CPU/3 GiB task, node001-node006 allowed, unpinned. Ledgers stay
  remote; pull only the scalar report. No raw-reference inference or retraining.

Limitations: this rule was motivated by Stage290 development outcomes, so its
reuse is exploratory, not fresh confirmation. Random thinning targets expected
value, not assignment spread or a deployable policy; count-matching conditions
on this cohort, not unseen prefixes. Restricted single-intervention metrics
and the need for future new-prefix confirmation are unchanged.

Run: `native_transit_zero_abstention_stage291_20261017_r1`.

Validation: seven focused tests passed (including cached end-to-end replay,
exact thinning expectation and bootstrap rematching); both CLI entrypoints
started successfully.
