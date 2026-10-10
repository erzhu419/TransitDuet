# Stage157: Learned Residual Service Plans

Stage156 qualifies executable allocation and shows small forecast gains, not
learned planning. Freeze the same two native lower policies (roots347/359),
train fresh upper SAC seeds397/401 with the existing shared actor-critic core.

At each committed six-departure block, the upper adds two bounded residual
coefficients (linear/curvature, each +/-20 s) to the causal forecast interval
preferences. Endpoint/time/event budgets and 240-480 s interval bounds remain.
34 causal inputs: original LF/HF-summary native state, five forecast preferences,
compressed queues/load, continuous time, fleet peak, previous action and other
direction's committed plan. No stress labels or future realized demand inputs.

Reward is `100*(prefix_cost_before-prefix_cost_after)`, using the SAME physical
restricted-wait/fleet/headway/unserved/completion cost, censoring passengers at
the current clock. Gamma=1 in a finite episode: the sum telescopes to initial
minus final service cost, including the final recovery tail. One transition per
actual plan decision; lower steps do not repeat upper actions/log probabilities.

120 balanced-regime training episodes/root, 10 random-action warmup, 5,500 SAC
updates; native weights never update. Last actor only. Evaluate twenty paired
scenes with learned, zero residual (forecast), nominal, and constant residual
fixed from the final actor's mean on TRAINING states before evaluation.
All forty forecast/nominal controls per root must reproduce Stage156 exactly.
Qualification: three full source reproductions plus independent short learning,
then reset the upper model/RNG for full training. 24,552,000 main native ticks
plus 389,880 qualification ticks; code-only scheduler node001-006.
86 focused tests pass, including exact zero-residual forecast execution,
current-clock censoring, terminal-tail credit, SAC learning, actor-noise RNG
preservation and rejection of unpaired controls or changed credit budgets.

Run `native_transit_learned_residual_plan_stage157_development_20261010_r1`:
t141463/root397 RUNNING on node005; t141464/root401 RUNNING on node006.
Both reproduced the original learned-dispatch source episode. Remaining worker
qualification and final learned-plan performance are pending.

## Limitations

Two-root learned-upper development, not joint two-level training or confirmation.
No promotion mechanism is tested. A small forecast gain does not imply residual
learning will improve it; learned versus constant residual is necessary to
separate state adaptation from a fixed bias. Goal-dependent lower reward stays
diagnostic. No convergence guarantee is asserted for gamma=1 neural soft SAC.
SAC retains entropy regularization; the telescoping equality concerns physical
reward, not the entropy-augmented optimization objective.
Fixed window endpoints prevent reallocating service between windows, not just
changing total daily frequency; this experiment tests within-window planning.
