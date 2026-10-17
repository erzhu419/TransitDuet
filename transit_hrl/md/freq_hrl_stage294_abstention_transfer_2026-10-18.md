# Stage294: Frozen Abstention Rule On New Prefixes

Run: `native_transit_abstention_transfer_stage294_20261018_r1`.

Stage292 retained waiting benefits but failed the joint cost/wait criterion.
Stage293 ruled out the bootstrap convention as the explanation and identified
substantial prefix heterogeneity. The next budget buys new scene coverage.

## Frozen Experiment

- 240 genuinely new prefixes: 24 per root/regime, roots 397/401, all five regimes,
  all six probes. Ten separate software-pilot prefixes do not enter inference.
- Four controls: raw, unchanged Stage247, Stage291 zero-event abstention, exact
  expected count-matched random thinning. Reuse frozen references/models.
- Independently sample P8 proposal, S8 screen, E32 evaluation futures per state.
  Save P8 proposals before S8 is read and seal all actions/weights before E32.
- Match thinning per profile/root/regime/probe across the 24 new prefixes;
  never use evaluation outcomes for matching or repeat a random assignment.
- Average 16 historical profiles and six probes before prefix inference.
  Fixed-weight stratified bootstrap: 5,000 resamples, seed294. Report shared-future
  conditional MC uncertainty separately. Neither profiles nor futures are seeds.
- Primary success: cost and wait versus raw have both 95% interval upper bounds
  below zero, and both independent E16 halves improve both outcomes. Cost/fleet
  information versus thinning are separate secondary tests. No regime pruning.

## Execution

Cached bridge must exactly reproduce old raw/247 actions, P8 proposals,
abstention choices and expected matching. Then run native P2/S2/E2 preflight
on all ten root/regime groups before the main 240 captures and 2,880 menu tasks.
Main menu budget: 161,280 native episodes. E32 replays only frozen disagreements;
its exact budget is sealed before evaluation. No fits or actor updates.

Scheduler places one-CPU tasks freely across node001-node006. Stage only code
and the registration input directory, not remote result directories. Pull only
small scalar receipts/reports; models, snapshots and rollout panels stay remote.

## Decision

If joint transfer passes, test repeated decisions with the frozen lower next.
If waiting transfers but cost does not, retain the limited service claim and
stop presenting this rule as joint improvement. If transfer fails, do not tune
on this confirmation cohort: return to independent training-scene coverage.

## Current Status

Cached bridge t175907 completed on node006, exit0: all 360 old states and 16
profiles exactly reproduce raw/247 controls, P8 proposals, S8 abstention and
expected matching. Twenty tests and both CLI startup checks passed. Pilots
t175910-t175919 are running across node002/004/005/006; sampled node002 Python
workers each use about one CPU core. Main sampling has not been dispatched.
Only the 7,634-byte scalar freeze report was pulled. Runtime code: `38de1865f5`.

## Limitations

This is finite-MC planning-rule transfer after one intervention, not a deployable
learned policy or complete end-to-end Freq-HRL validation. Source-root lower
weights remain fixed. No performance conclusion is available at registration.
