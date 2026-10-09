# Stage147: Frozen Native Mechanism Diagnosis

Stage146 has no supported service-cost advantage despite a smaller upper
HF-action proxy. Before another training change, diagnose its 24 deployments.
One registered scene per regime measures actor input scale, action saturation,
and same-state action changes when the routed band or all upper frequency
features are zeroed. Probes are passive during simulation; zero-input queries
run afterward. Baseline metrics must exactly reproduce Stage146.

Correct-routing actors also execute two paired physical interventions: zero
upper target-headway delta, and zero learned holding (native boarding dwell is
unchanged). Every episode reloads the final deployment, keeps fleet 12 and the
registered 61,380-second clock, and forbids network updates during diagnosis.
One-episode preflight precedes the 24-task, 200-episode diagnostic matrix.
There is no seed expansion, checkpoint download or checkpoint selection.

V1 preflight `t137327` failed before evaluation. Its incorrectly declared
results-parent staging used scheduler `rsync --delete` and removed the
server-only Stage146 checkpoint siblings. Compact source results are intact.
V2 stages code only and first recovers checkpoints by exact recipe/seed replay,
requiring matching recorded demand, training curves, update counts and actor
changes. Root101 recovery precedes the full matrix; other roots recover in
their diagnostic jobs. This is restoration work, not independent evidence.
Recovery preflight `t137332` completed on node005. All 300 episode demand counts,
32 recorded training rows, 2,700/9,000 upper/lower updates and actor-change
maxima matched. Its 61,380-tick baseline exactly reproduced Stage146.

Root101/low-noise had no 1%-edge action saturation. Lower holding was
4.36 +/- 5.18 seconds, but zeroing its band changed the same-state action by
only 0.00593 seconds on average. Raw headway inputs averaged 354-360 seconds
against 0.011-0.037 for the frequency tail. Upper target delta was
0.290 +/- 0.148 seconds; zeroing dynamic/all frequency slots changed it by
0.208/0.981 seconds. These are single-root diagnostics, not performance claims.

The native HIRO upper changes target headway, not launch time, whereas its
default hindsight gap credit uses actual launches. Any effect is indirect
through the lower and fleet loop; neutral-upper interventions will measure
whether that path has meaningful control authority. Preserve the registered
recipe until the cross-root diagnosis is complete.

Eighteen focused tests passed. Full diagnosis uses 24 tasks: 23 additional
checkpoint replays (6,900 training episodes), then 200 frozen episodes. Root101
reuses its recovered checkpoint. Replay work and diagnostic ticks are reported
separately; there is no new optimizer root or independent performance result.
Full matrix `native_transit_diagnostics_stage147_full_20261009_r1` is submitted
as `t137622` through `t137645`, dynamically eligible for node001-006, one CPU
and 3 GB per task, without node binding. Only compact JSON is synchronized.

If the lower is saturated and barely reacts to its band, qualify the existing
dimensionless physical encoder before repeating matched routing training. If
neutral upper has negligible effect, fix goal execution/credit before changing
the frequency filter. If actors respond but service does not, investigate
control authority and objective alignment rather than adding seeds.

## Completed Result

All 24 tasks completed. All 24 baseline deployments reproduced Stage146;
23 full-matrix recovery receipts plus the earlier root101 receipt matched their
recorded training observables. Restoration consumed 423,522,000 ticks in this
matrix (441,936,000 including preflight); frozen diagnosis consumed 12,276,000.

Across eight correct-routing roots, zeroing upper commands changed restricted
wait by -0.000625 minutes and service cost by -0.0000634 on average, with no
peak-fleet or completion change. Upper commands were not universally zero:
root means ranged from -12.48 to +7.81 seconds. Nevertheless their physical
service effect was negligible. Neutralizing upper increased native episode
reward by 8.73 while leaving service essentially unchanged, consistent with
target-dependent reward rather than useful physical upper control.

Zero holding increased restricted wait by 0.1298 minutes and headway CV by
0.12691, and reduced reward by 551.81. It also reduced peak fleet by 0.5 and
composite service cost by 0.05101: the learned lower has control authority,
but its wait/headway benefits carry a fleet-cost tradeoff.

No material action-edge saturation appeared. Correct-routing lower band
zeroing changed its same-state action by only 0.00861 seconds on average;
raw-history/swapped effects were 0.01422/0.05812 seconds. The scale imbalance
is a conditioning concern, not proof by itself. Stage148 will cross the
existing explicit-target physical lower encoder with target-independent
upper wait/fleet interval credit, holding routing and the native trainer fixed.

## Limitations

These are post-result development diagnostics on reused scenes, not independent
performance confirmation. Feature-zeroing can be off-distribution and cannot
by itself establish causal frequency attribution. Physical interventions test
the existing learned controllers, not a retrained no-upper/no-lower baseline.
The legacy lower LF ratio acts on cumulative holding and is not an action-band
share; neither that ratio nor the uncentered upper proxy diagnoses saturation.
Recovered weights cannot be compared to lost original arrays. Matching recorded
training observables and baseline outcomes is the available recovery evidence.
