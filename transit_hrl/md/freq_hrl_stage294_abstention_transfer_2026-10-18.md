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
expected matching. Twenty tests and both CLI startup checks passed.

Pilots t175910-t175919 completed done/exit0. Aggregate t175936 also completed
done/exit0: 430 native episodes, 60 exact identities, all ten root/regime groups
and all six probe positions qualified. P2/S2/E2 draws were disjoint (0-1, 8-9,
16-17); each group retained all 16 frozen profiles. No outcomes selected a rule.
Peak RSS was 676.46 MiB. Fourteen-episode banks took 351.59-386.28 seconds;
the fixed 1,536 MiB, one-CPU allocation is sufficient. An eight-future main menu
task has 56 episodes, roughly 24 minutes at pilot throughput.

Formal captures t175937-t176176 completed done/exit0: 240 qualified prefixes,
1,440 states, 24 prefixes in each of ten root/regime groups, 16 profiles each.
Existing scheduler targeted probes refreshed delayed terminal accounting;
no capture was rerun and no scheduler source was modified.

Menu tasks t176191-t179070 completed done/exit0. Sealing task t179254 completed
done/exit0 on node006: all 23,040 state-futures, 161,280 native episodes and
23,040 exact identities passed; values were finite and cost decomposition error
was zero. P8/S8/E32 banks are disjoint. Twenty-two focused/regression tests pass.

The frozen screen accepted 6,259 changed profile-state proposals; zero-event
abstention retains 864. Expected matched thinning also retains 864 across 960
strata (maximum count error 1.78e-15). These are decisions, not independent seeds.
No evaluation labels were read. Audit code revision: `e41234b617`; selectors,
thresholds and inference are unchanged.

Independent E32 tasks t179258-t183453 (4,196 tasks) are submitted to the unpinned
node001-node006 pool: 1,049 active states, 33,568 state-futures, 141,376 native
episodes. The other 391 states have exact zero paired effects and remain in the
1,440-state/240-prefix denominator. Draw16-47 uses frozen choices and matching
weights, with no reselection or fits. Performance conclusions await this bank.
At 06:58 UTC, 275 tasks were running and 3,921 queued, with no failures;
placement was dynamic across node002/004/005/006. A node006 process sample
confirmed 76 active E32 Python workers, with median CPU usage 98.9%.
Only the 14,184-byte scalar seal report was pulled; the action/weight ledger,
models, snapshots and rollout panels remain server-only.

## Limitations

This is finite-MC planning-rule transfer after one intervention, not a deployable
learned policy or complete end-to-end Freq-HRL validation. Source-root lower
weights remain fixed. No performance conclusion is available at registration.
