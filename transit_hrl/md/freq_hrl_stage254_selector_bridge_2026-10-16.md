# Stage254: Cached OOF-to-Full Bridge

Stage253 confirms held-prefix waiting benefit, not the preregistered joint goal.
Bank0 fleet cost worsens; seen-prefix waiting remains a positive point mean.
Freeze all six Stage252 protocols and both filters. Do not choose an arm or
change the critic/loss from these outcomes.

For each existing prefix/half/bank, read cached inner actions and selected keys,
full-model actions and cached raw references. Use the same opposite16 Stage249
futures for inner/full comparisons. Stage253 has sparse action coverage, so it
cannot supply unobserved inner-action labels without additional simulation.

Separate validation optimism (OOF independent minus validation), correction
transfer (full independent minus OOF independent), and raw-reference action
shift. Selected-action transfer equals correction transfer plus reference shift.
Report raw-choice and feasible-mask agreement strata, both banks, four views,
prefixes and scenarios. Reproduce Stage252 full train60 outcomes and selected
OOF validation outcomes before accepting a cell.

One single-core scheduler analysis over 32 cached cells, unbound on node001-node006.
Zero new episodes, fits, inner reconstruction solves, actor updates or prefixes.
Only the scalar report is pulled; block ledgers and source caches stay remote.

Limitations: post-confirmation development diagnosis, not a new confirmation or
CI. Prefix-excluded inner references intentionally differ from full references;
disagreement is not presumed to be an implementation bug. Full versus inner also
changes training prefixes and future count (8 to 16); this is not their separate
causal effect. Stage248 outcomes and Stage169 reserves stay closed. No default
or deployment change.

## Results And Next

Task t165394 completed exit 0 on node006. All 32 cells reused 92,160 paired
action-state exposures; maximum prior mean-outcome difference 3.47e-18. Five tests
passed. Only the 3,688,548-byte scalar report was pulled; zero new fits/episodes.

Expanded8 pooled-safe wait correction, mixed seen prefixes / common new futures:

| Model | Selected Validation | OOF Independent | Full Independent |
|---|---:|---:|---:|
| free_old | -0.000538053 | -0.000135661 | +0.000140332 |
| shrunk | -0.000635937 | -0.000245540 | +0.000199967 |

OOF-to-independent optimism explains 59.3% / 46.7% of the validation-to-full wait
gap along this decomposition; correction transfer explains the rest. About 80%
of cost optimism is fleet cost. Raw choices differ on 45.0% of exposures, predicted
wait-gate masks on 72.66%. Full raw improves cost by -0.002167852 and wait by
-0.007231022 min versus inner raw; full selected actions also improve absolute
cost/wait versus inner selected actions. Reduced incremental utility is not an
absolute full-policy regression. Remaining transfer loss also occurs when raw
actions agree, so reference changes are not a complete causal explanation.

Next: test alignment to the fixed-past deployment bundle (reference, feature
coordinates, prior and RMS convention), with unchanged selector/critic rules and
frozen full models. Use new2/new8 validation only: the past bundle was fitted on
old A labels, so old_A2 cannot provide clean validation for that contrast. This
would assess incremental correction conditional on a pretrained bundle, not OOF
validation of the entire learner. Test validation uncertainty separately afterward.
No alignment-fit task has been submitted yet. Feasibility here means the cached
predicted-wait gate, not physical action legality; all results remain descriptive.
