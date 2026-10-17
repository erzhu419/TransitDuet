# Stage290: independent fleet screening

Reuse Stage288 training caches only: 60 prefixes, 360 states, 24 futures and
16 fixed profiles. No fits, native episodes, actor updates or held-prefix reads.

- Primary: propose with futures 0-7, freeze; screen with 8-15, freeze; evaluate
  with 16-23. Reuse Stage289's wait-constrained minimum-cost proposal. Accept
  only if screening mean fleet difference versus own raw <=0; otherwise raw.
  No alternative-action search, margin or gate tuning on the screen bank.
- Matched controls: same8 unscreened proposal, shared-bank fleet screen, and
  pooled16 proposal using the same total information. Pooled16/raw/Stage247
  choices must replay Stage289 exactly. All six bank-role permutations are
  fixed; none replaces the primary or supplies independent replications.
- Use the shared prefix statistics (60-prefix stratified bootstrap5000,
  seed290; profiles/probes collapsed before CI; paired conditional MC SE).
  Joint retention requires independent-screen versus raw cost/wait CI95
  upper bounds <0 and negative points in both four-future halves. Fleet
  reduction versus same8 proposal is a separate conclusion.
- Report accepted/rejected proposal counts and evaluation contributions,
  separating accepted proposals with zero observed screening fleet difference.
  One scheduler CPU/3 GiB task, any node001-node006, no pin. Ledgers remain
  remote; only scalar reports are pulled.

Limitations: these are reused training-development futures and finite-MC
information probes, not a deployable selector, risk certificate or new-prefix
confirmation. A zero observed fleet difference is not evidence of zero risk.

Run: `native_transit_fleet_screen_stage290_20261017_r1`.

Validation: seven Stage290 tests and six Stage289 regression tests pass;
both Stage290 CLI entry points start. Shared statistics retain Stage289's
defaults; Stage290 supplies its six arms and seed explicitly.
