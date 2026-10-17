# Stage290: independent fleet screening

Run: `native_transit_fleet_screen_stage290_20261017_r1`.
Reuse Stage288 train-only caches: 60 prefixes, 360 states, 24 futures, 16 fixed
profiles. Primary proposal/screen/evaluation banks are 0-7 / 8-15 / 16-23;
freeze before each next bank. Screen the frozen proposal's fleet mean <=raw,
otherwise raw; no alternative search. Controls: same8 unscreened/shared-screen,
pooled16, raw and Stage247. Six fixed permutations; no winner selection.
Inference: 60-prefix stratified bootstrap5000, seed290, shared-future MC SE.
Joint retention requires cost/wait CI95 upper <0 and both halves' points <0.

## Result and next decision

`t173096` completed on node006, exit0, 1 CPU/3 GiB, unpinned. Zero fits,
episodes or held reads. Thirteen tests/two CLI startups pass; all six queries
replay Stage289 controls exactly. Only 108,000 scalar-report bytes pulled.
Primary independent screen minus raw: cost +33.84 [-850.57, 1039.91] x1e6,
wait -0.0072132 [-0.0102648, -0.0045582] min. The joint gate fails; cost
halves are -712.47 and +780.15 x1e6. Fleet reduction versus the same8 proposal
is supported: -605.83 [-1040.08, -198.93] x1e6. Total cost difference remains
inconclusive (-321.79 [-738.50, 67.23] x1e6), while wait worsens versus that
proposal by +0.0028637 [0.0016179, 0.0042439] min.

Of 1369 accepted-change exposures, 1216 (88.8%) have zero observed screening
fleet difference. They contribute +811.99 fleet/+303.26 cost x1e6, exceeding
the overall +770.40/+33.84; other accepted actions offset part of this loss.
Across six permutations, wait improves in six, joint CIs pass in none, and
fleet reduction passes in four.

Retain the independent-screening mechanism evidence, not the selector. Next:
test abstention on zero-observed-event proposals against a blinded matched-count
thinning control, separating risk information from merely taking fewer actions.
No further fits, simulations or tasks were dispatched.

Limitations: reused development data; permutations/profile exposures are not
independent replications. Finite-MC information probes are not deployable
selectors, risk certificates or new-prefix confirmation. Zero observed events
do not establish zero risk; subgroup contributions are descriptive.
