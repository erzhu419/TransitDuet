# Stage-28 Genuine Plan-Hold Supervision

Reuse Stage-26 remote controller checkpoints for roots 209011/209061;
preflight uses 208001. Load weights without training and first reproduce
one frozen Stage-12 factual episode (return and ISE tolerance 1e-8).
New seed bases are 3279000/3280000/3281000, fit offsets 1-16 and evaluation
offsets 101-108 (preflight two paths per role). Exclude inherited paths,
Stage-26 paths and actual iteration-derived controller training seeds.

Collect 20 checks per full path, offsets 0/5/10/15/20 four times each.
Keep balanced-jitter calls strictly before each check. Capture the causal
64-frame prefix before intervention; preserve lower feedback and history.
Renew calls the upper at the check, then holds that plan for 150 steps.
Keep executes the old waypoint unchanged for 100 steps, then renews once
and executes its new waypoint for 50 more steps. No other upper calls.

ISE keep-minus-renew at 10/25/50/100 steps is a gross plan-renewal curve:
renew has one post-check planner call, keep has zero. These are not
equal-call comparisons. At the primary 150-step settlement endpoint,
both arms have one executed post-check call and identical prefix calls.
The late call is followed by real execution, not dummy budget padding.
This block-level budget is distinct from the original per-period budget.

Retain Stage-27's 31 causal features, methods history/current-repeat/shuffled,
train-only standardization and alpha-one ridge on summed squared error,
unpenalized intercept. Five independent heads: three fits/15 scalar solves
per root, no controller or gradient updates. Prediction gate: history beats
zero/current/shuffled on both gross horizon-averaged rate MSE and settled
rate MSE. Decision gate: strictly positive settled ISE benefit versus
current, shuffled, always-renew and always-keep. Both roots must pass;
ties fail. No coefficient, horizon, seed or threshold selection afterward.

Full replay charge: 657600 pair steps plus 1200 factual steps per root,
1317600 total new steps. Preflight: eight pairs, 3760 pair steps plus 300
factual steps. Raw sequences and step costs stay on the server; sync only
result JSON. Scheduler pool node001-node006, no fixed node; full 16 workers
plus coordinator (17 CPU/24 GB), preflight one worker (2 CPU/3 GB).
Twenty focused tests passed: actual old-waypoint holding, executed late
renewal, prefix/future isolation, gross versus settled gates, fit isolation,
cached factual matching, exact step accounting and remote-only cache staging.

Implementation frozen at `f455dda8d0`. Preflight `t101665` completed on
node004: factual return/ISE errors zero, eight pairs, 4060 new steps,
three fits/15 scalar solves. Independent row/call/metric recomputation
matched. Retrieved 48727 bytes of JSON, no raw cache or checkpoint.
Both scientific gates failed; retain the result with full settings unchanged.
Full tasks `t101666/101667` register roots 209011/209061 at the same frozen
revision, 480 pairs per root, with no independent-confirmation claim.

## Limitations

This fresh-path development screen reuses controller roots and changes the
continuation budget. It is not policy deployment, independent confirmation,
plan-lifetime identification or a domain-general performance claim. Gross
hold curves cannot establish performance under an equal planner budget.
