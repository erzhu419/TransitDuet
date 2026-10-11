# Stage162: Frozen Allocation-Strength Frontier

Stage161 reduces waiting but increases fleet cost on both roots. Before another
learned-upper run, test whether the wider representation has a useful region
under the SAME physical objective, fleet target and native lower policy.

At every existing rolling decision, interpolate preferred intervals as
`(1-alpha)*current_tentative_intervals + alpha*causal_forecast_intervals`, then
use the unchanged bounded-simplex projection and irrevocable six-trip commitment.
Alpha is constant within an episode: 0, .25, .5, .75, 1. Zero reproduces rolling
nominal; one reproduces the complete Stage161 rolling forecast. Intermediate
strengths temper each proposal; they are not linear interpolations of complete
episode schedules or guarantees of monotone fleet costs. Window forecast is
the common performance control. No changes to dispatch physics or cost weights.

Two roots, five regimes, four registered scenes, six conditions: 240 frozen
episodes, 14,731,200 native ticks, zero training updates. Each of ten cells first
reproduces its window forecast and both rolling endpoints before testing the
interior strengths. All 120 source controls must match physical outcomes AND
complete planned/actual schedules. All scenes, including OOD fleet jumps, stay.

Report all five strengths versus window forecast, both per root and per regime:
cost, restricted waiting, native reward, fleet/nonfleet cost, headway, unserved
and completion. No winner is selected as a deployed policy from these scenes.
If an interior strength improves cost without sacrificing service, the next
experiment can learn a causal strength action and compare its own constant
control. If not, add explicit supply coupling before training, not more seeds.

Scheduler node001-006 unpinned, 1 CPU/3 GB per task; native checkpoints/CSV stay
remote and only compact result JSONs sync. Qualification and performance pending.
117 focused tests pass, including endpoint reproduction, interpolation before
projection, changed-observation commitment/tail behavior, complete budgets,
source schedule checks, component attribution and unpinned scheduler placement.
Local execution tests use the native dispatcher with a stub environment, not
full native simulation or training.

Run `native_transit_allocation_frontier_stage162_frozen_20261011_r1`:
t142039-t142048 DONE on node004/005/006. All 240 frozen episodes qualify,
including 120 exact physical AND schedule reproductions of the source controls.
All 14,731,200 native ticks completed, zero training updates.

| Strength | Cost delta 397 / 401 | Wait delta, min 397 / 401 |
| --- | --- | --- |
| 0 | +0.000767 / +0.002681 | +0.00720 / +0.02870 |
| .25 | +0.246695 / +0.185123 | -0.31450 / -0.09305 |
| .5 | +0.221667 / +0.165346 | -0.56805 / -0.31125 |
| .75 | +0.213043 / +0.127256 | -0.68525 / -0.49170 |
| 1 | +0.179531 / +0.107845 | -0.82495 / -0.68220 |

No positive-strength candidate improves mean cost on either root. Alpha .25/.5/
.75 raise peak fleet on 13/20 scenes for root397 and 9/20 for root401; alpha1
does so on 12/20 and 9/20. Fleet costs are discontinuous, not proportional to
alpha. Small per-update strengths also accumulate through tentative schedules;
they are not equally small episode-level interventions. Root401 high-noise
alpha .75/1 is the sole regime/root panel with mean cost improvement, not a
global success. Do not train an alpha policy from this frontier or change cost
weights to declare success. Next isolate explicit causal supply/occupancy costs
against the same candidate selector without its supply term.

## Limitations

This is a frozen development frontier, not learned adaptation, independent
confirmation, a hard fleet constraint or end-to-end Freq-HRL validation.
