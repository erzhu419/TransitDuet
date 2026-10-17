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

Run: `native_transit_zero_abstention_stage291_20261017_r1`.

Validation: seven focused tests passed (including cached end-to-end replay,
exact thinning expectation and bootstrap rematching); both CLI entrypoints
started successfully.

## Result and next decision

Task `t173114`, node006. Pulled only 87,122 bytes of scalar JSON. All unchanged
control means reproduce Stage290 exactly across six permutations.

Primary kept 153/1,369 profile-state changes; matched thinning also expects
153, with 103 retained changes in genuinely randomizable matching strata.
Below, cost is multiplied by 1e6; intervals are paired prefix-bootstrap CI95.

| Contrast | Cost [CI95] | Wait/min [CI95] |
| --- | --- | --- |
| Abstain - raw | -269.42 [-692.33, +46.60] | -0.002141 [-0.004534, -0.000490] |
| Matched thinning - raw | -76.33 [-569.03, +169.16] | -0.001647 [-0.004056, -0.000523] |
| Abstain - matched thinning | -193.09 [-234.85, -18.01] | -0.000494 [-0.000681, +0.000089] |

Both evaluation halves improve cost/wait versus raw, but the primary joint
criterion fails. Cost information versus matched thinning passes its prefix
criterion; fleet information does not. Four/six overlapping permutations pass
the cost-information criterion and two/six pass joint retention, not six new
replications. Abstention retains only 29.7% of Stage290's waiting gain.

Next: freeze the primary proposal, screening membership, choices and thinning
weights; confirm with new futures, replaying only frozen action disagreements.
No further threshold/head search on these caches. Do not promote this controller.

Limitations: the rule was motivated by Stage290 outcomes on this development
cache, so this is exploratory. The cost difference versus thinning has conditional
future MC SE 110.96 in the same cost units, separately from the prefix interval;
it is not fresh-future confirmation. Thinning targets expected value, not assignment
spread, and matches this cohort rather than unseen prefixes. These are restricted
single interventions with 60 prefixes, not independent profiles or end-to-end HRL.
