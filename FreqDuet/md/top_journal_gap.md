# FreqDuet Top-Journal Gap Backlog

Last updated: 2026-06-10 CST

This file records the remaining gap between the current FreqDuet implementation
and a top-journal-ready paper package. It should be used as the execution
backlog after `dev_manual.md` and `GPT.md`: every item below needs either a
completed implementation/experiment, a documented negative result, or an
explicit paper-scope decision.

## Current Status

FreqDuet is now a working main-path prototype rather than a design sketch.
The current promoted line includes:

- causal harmonic demand decomposition with historical priors;
- LF/HF state separation;
- upper target-headway / timetable policy;
- lower discrete holding policy;
- lower high-frequency credit;
- leakage penalty;
- DriftFB feedback;
- guarded promotion with active and persistent gates;
- trace logger, audit tooling, MI / shock-response style diagnostics;
- scheduler/HPC-ready experiment scripts;
- a 40-episode, 20-seed, four-domain final matrix under the current naming.

Latest current-name 40ep final matrix:

```text
main        1.5446
rawhistory  1.5505
nopromotion 1.5599
allfreq     1.5681
nofreq      1.6281
noleakage   1.9113
```

Per-domain main result:

```text
terminal  1.463
highnoise 1.874
odshift   1.531
rushshift 1.311
```

Interpretation: the main method is best on average and best in terminal/rush,
but it is not yet best in every domain. `rawhistory` is still slightly better in
highnoise, and `rawhistory` / `nopromotion` are slightly better in odshift. This
means the method direction is effective, but the paper evidence is not closed.

2026-06-03 update: the current-name 200ep matrix has been synced and
aggregated. It confirms that `noleakage` is clearly bad, but it also exposed a
long-training weakness in the previous main line: lower action/drift rose after
roughly 80 episodes while `lower_lambda` decayed. The drift-cost repair
candidate closed this gap and has been promoted into `F_freqduet_*_main_hiro`.
Historical pre-repair aliases are kept as `*_main_predriftcost_hiro`.

## Top-Journal Gap Summary

Status legend:

- `[x]` completed enough for the current prototype;
- `[~]` partially done, needs paper-grade completion;
- `[ ]` not yet complete.

### 1. Current-Version Long Training

Status: `[x]` promoted 200ep matrix complete

The 200ep current-name matrix is now available, but it does not close the paper
claim by itself because the current main loses its 40ep advantage in some
domains during longer training.

2026-06-02 start: added the current paper-longtrain runner and paired-delta
summary tooling. The first six 80-run shards were auto-routed by scheduler to
Slurm because each requested 80 CPU workers; they were cancelled after staying
`PENDING` under `QOSMaxCpuPerUserLimit`.

The active launch is the direct-node retry below. It uses 30-core shards to stay
below scheduler's large-CPU Slurm auto-route threshold and to make the tasks
visible as normal scheduler direct-node jobs in `tui-top`.

```text
matrix: 4 domains x 6 methods x 20 paired seeds = 480 runs
episodes: 200
last_k: 100
effective active ranges:
  t5907  node001   0-30
  t5926  node005   30-60
  t5909  node003   60-90
  t5910  node004   90-120
  t5911  node005   120-150
  t5912  node006   150-180
  t5913  node001   180-210
  t5927  node006   210-240
  t5915  node003   240-270
  t5916  node004   270-300
  t5917  node005   300-330
  t5918  node006   330-360
  t5919  node001   360-390
  t5928  node001   390-420
  t5921  node003   420-450
  t5922  node004   450-480
state at launch check: all 16 effective ranges running direct on node001/node003/node004/node005/node006
```

2026-06-03 result: the pre-repair current 200ep summary is under
`FreqDuet/freqduet/results_freqduet/paper_longtrain_current_ep200_wu10`.
Overall paired deltas show that main is decisively better than `noleakage`, but
not decisively better than `nofreq`, `rawhistory`, `allfreq`, or `nopromotion`.
The diagnosis is that long-horizon lower drift was only a reward shaping term,
not a Lagrangian cost. The active repair adds `lower_drift_cost_*` leakage keys
and injects rolling drift excess into lower constrained cost.

The promoted 200ep matrix is under
`FreqDuet/freqduet/results_freqduet/paper_longtrain_promoted_ep200_wu10`.
Overall paired composite deltas for promoted main are:

```text
vs nofreq      -0.0719  CI [-0.1322, -0.0266]
vs rawhistory  -0.0528  CI [-0.0724, -0.0333]
vs allfreq     -0.0629  CI [-0.0913, -0.0346]
vs nopromotion -0.0542  CI [-0.0720, -0.0354]
vs noleakage   -0.3052  CI [-0.3908, -0.2286]
```

Done means:

- run current `main` against `nofreq`, `rawhistory`, `allfreq`,
  `nopromotion`, and `noleakage`;
- use 100 or 200 episodes;
- cover terminal, highnoise, odshift, and rushshift;
- report seed-level paired deltas, mean, std, confidence interval, and best/worst
  seed behavior;
- record whether the 40ep advantage survives long training.

### 2. Systematic Generalization Matrix

Status: `[~]` three-shift 100ep matrix complete; broader scenario families open

Highnoise, odshift, and rushshift are present, but this is not yet a complete
held-out generalization package.

2026-06-03 update: `scripts/run_freqduet_generalization_matrix.sh` now defaults
to highnoise / odshift / rushshift x six methods x 20 paired seeds at 100
episodes. The current promoted 100ep three-shift matrix has been executed under
`results_freqduet/generalization_promoted_ep100_wu10`; broader scenario
families beyond the three current held-out shifts are still open.

2026-06-08 update: added
`scripts/submit_freqduet_broad_generalization_scheduleurm.py`, a direct-node
scheduleurm submitter for a broader 8-scenario x 6-method x 20-seed matrix:
`noise10`, `noise20`, `noise40`, `od20`, `od50`, `rush_early`, `rush_late`, and
`rush_extreme`. It generates configs under
`configs_freqduet/paper_generalization`, shards the matrix into 30-job CPU
tasks, and hard-pins shards to `node001-node006` to keep them visible as
scheduleurm direct-node jobs.

2026-06-08 run: submitted all 32 shards as scheduleurm direct-node tasks
`t7829-t7860` on `node001-node006` with no Slurm backend jobs. The first status
check found 18 running and 14 queued; the queued shards are still visible in the
scheduler queue, with `node001` cooling down after high-priority preemption and
the rest waiting for 30 free CPUs on their pinned nodes. Remote logs have started
under
`/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet/results_freqduet/broad_generalization_ep100_wu10/logs_shards`.
The experiment remains open until all shards finish, sync back, aggregate, and
produce paired deltas / CI.

2026-06-08 completion: the broad matrix finished and was synced/aggregated:
`960/960` expected rows, `missing=0`. Overall composite paired deltas for main
are:

```text
vs nofreq      -0.0012  CI [-0.0436, +0.0294]
vs rawhistory  +0.0047  CI [-0.0084, +0.0169]
vs allfreq     +0.0028  CI [-0.0157, +0.0200]
vs nopromotion +0.0035  CI [-0.0190, +0.0248]
vs noleakage   -0.3411  CI [-0.4622, -0.2328]
```

Interpretation: the frequency-separated main remains clearly superior to
`noleakage`, but the broader held-out scenarios do not prove a significant
overall advantage over `nofreq`, `rawhistory`, `allfreq`, or `nopromotion`.
Family-level results are also mostly tied except the no-leakage failure. This
closes the broad matrix as evidence, but it keeps the top-journal claim at a
robustness/tradeoff level rather than a universal dominance claim.

2026-06-08 repair follow-up: diagnosis showed the broad weakness is concentrated
in mild/medium held-out scenarios where promotion absorption can raise lower
action and fleet overshoot, while `noise40` still benefits from promotion. Added
a causal promotion absorption energy gate (`promotion.adapt_high_energy_min`)
and submitted the `promenergy08` candidate (`t7925-t7930`) for the 8-scenario x
20-seed broad matrix. This is a candidate only; it is not promoted unless it
improves paired broad deltas without losing the `noise40` gain.

2026-06-08 `promenergy08` result: completed `160/160` rows. It improves overall
composite versus main weakly (`-0.0083`, CI `[-0.0224,+0.0054]`) and helps
OD/rush, but it hurts `noise40` versus main (`+0.0505`, CI
`[-0.0165,+0.1217]`). Do not promote it directly. Submitted a narrower
threshold sweep, `promenergy06/promenergy07`, to test whether a lower energy
gate preserves the `noise40` gain while keeping the OD/rush repair.

2026-06-09 `promenergy06/promenergy07` result: completed `320/320` rows. Neither
threshold beats current main overall:

```text
promenergy06 vs main  +0.0019  CI [-0.0118, +0.0173]
promenergy07 vs main  +0.0043  CI [-0.0112, +0.0202]
```

Both keep the strong `noleakage` advantage but do not solve the broad internal
baseline tie. This closes the promotion-energy gate sweep as an ambiguous or
negative repair. The remaining weakness is not a simple promotion threshold; it
likely needs either a better value model for when HF holding is worth fleet/CV
cost, or a stronger external baseline/framing decision that treats broad
internal ties as robustness rather than dominance.

2026-06-09 history-prior auxiliary follow-up: added a causal harmonic
history-aux path that appends short realized-demand residual bins to LF/HF
features without future leakage. Ungated residual history did not repair the
broad tie:

```text
histaux3 vs main  +0.0181  CI [-0.0142, +0.0641]
histaux6 vs main  +0.0088  CI [-0.0069, +0.0280]
```

The failure mode matches the earlier concern that raw single-day history has
too much variance: `histaux6` significantly hurt the demand-noise family
(`+0.0386`, CI `[+0.0074,+0.0744]`) and `noise40` (`+0.1107`, CI
`[+0.0354,+0.1826]`). A high-frequency-energy gate was then added so residual
history is down-weighted when the harmonic decomposer sees unstable HF energy.
The gated sweep completed `320/320` rows:

```text
histaux6eg05 vs main  +0.0027  CI [-0.0166, +0.0219]
histaux6eg06 vs main  -0.0046  CI [-0.0250, +0.0181]
```

`histaux6eg06` is the only useful candidate from this branch: it keeps the
harmonic historical prior, no longer significantly harms `noise40`
(`+0.0049`, CI `[-0.0729,+0.0803]`), and weakly improves the broad mean, but the
effect is not statistically closed. Do not promote it yet. The next step is a
200ep main-vs-`histaux6eg06` longtrain check before any alias change or push as
a good result.

2026-06-09/10 longtrain result: the clean current-code 4-domain x 2-method x
20-seed x 200ep matrix for `main` versus `histaux6eg06` completed as scheduler
direct-node tasks `t8769-t8774`. It was not routed through Slurm. The run is
under `results_freqduet/histaux6eg06_longtrain_ep200_wu10` and aggregated
`160/160` rows.

```text
terminal   main - histaux6eg06  +0.0216  CI [-0.0062,+0.0496]
highnoise  main - histaux6eg06  -0.0044  CI [-0.0555,+0.0468]
odshift    main - histaux6eg06  -0.0111  CI [-0.0579,+0.0372]
rushshift  main - histaux6eg06  -0.0004  CI [-0.0316,+0.0308]
overall    main - histaux6eg06  +0.0014  CI [-0.0215,+0.0248]
```

Positive `main - histaux6eg06` means the candidate is better. The 200ep result
is therefore a statistical tie, not a promotion result. Mechanism check shows
`histaux6eg06` reduces upper HF power, but it raises lower action in highnoise,
OD shift, and rush shift, increasing wait enough to cancel the terminal gain.
Decision: do not change main aliases or push this branch as a good algorithmic
result.

2026-06-10 targeted follow-up: launched `histaux6eg06upper`, which keeps the
same gated harmonic residual memory in the upper timetable state but sets
`lower_bins: 0`, so the lower holding policy sees the same frequency feature
dimension as current main. Local smoke passed, with feature dimensions:
`main upper/lower = 6/4`, full `histaux6eg06 = 12/10`, and
`histaux6eg06upper = 12/4`. The 4-domain x 20-seed x 200ep candidate-only run
is `histaux6eg06upper_longtrain_ep200_wu10`, submitted as direct-node tasks
`t8877-t8882` on `node001-node006`.

2026-06-10 `histaux6eg06upper` result: completed `80/80` candidate rows and
compared against the same-code 200ep main rows from
`histaux6eg06_longtrain_ep200_wu10`. It is worse in all four domains:

```text
terminal   candidate - main  +0.0253  CI [-0.0263,+0.0893]
highnoise  candidate - main  +0.0393  CI [-0.0242,+0.1129]
odshift    candidate - main  +0.0288  CI [-0.0475,+0.1350]
rushshift  candidate - main  +0.0368  CI [-0.0328,+0.1203]
overall    candidate - main  +0.0325  CI [-0.0217,+0.1087]
```

Mechanism check: `histaux6eg06upper` raises wait by `+0.287min` overall and
raises lower action by about `+0.97s`, despite not directly feeding history
residuals to the lower state. This closes the current history-aux branch:
residual memory is useful as a diagnostic, but not a validated main-module
repair. Do not promote `histaux3`, `histaux6`, `histaux6eg05`,
`histaux6eg06`, or `histaux6eg06upper`.

Done means:

- evaluate multiple demand-noise levels instead of one highnoise setting;
- evaluate multiple OD profile shifts;
- evaluate multiple rush-pattern shifts;
- include service stochasticity / route perturbation if feasible;
- separate in-distribution seeds from held-out scenario families;
- show whether frequency separation is robust or only tuned to one service day.

### 3. Per-Domain Weakness Repair

Status: `[~]` lower drift repaired; fixed-headway gap still open

The current main wins on average but loses narrowly in highnoise and odshift.
This must either be repaired or explained with statistically defensible
tradeoffs.

2026-06-03 update: 200ep diagnosis pointed to lower-drift/Lagrangian mismatch
rather than only decomposer smoothing. `main_driftcost` beat previous main in
all four domains at 20 seeds / 200 episodes and has been promoted into the main
aliases. Mechanism summary shows lower action dropping from roughly 8-9s to
4.3-4.7s and lower drift penalty dropping from roughly 0.33-0.39 to 0.13-0.15.

2026-06-03 late update: the 100ep promoted generalization matrix plus external
fixed-headway baseline exposed a different weakness: promoted main is robust
against internal ablations, but fixed-headway remains very competitive because
FreqDuet can pay extra composite cost through fleet overshoot/CV. A fleet
no-harm guard was tested without changing the promoted main alias:

```text
candidate           overall vs promoted main      overall vs fixed-headway
noharm full         -0.0450 CI [-0.0792,-0.0106]  -0.0142 CI [-0.0524,+0.0233]
upper-only          -0.0460 CI [-0.0775,-0.0123]  -0.0138 CI [-0.0528,+0.0228]
lower-only          -0.0456 CI [-0.0793,-0.0120]  -0.0156 CI [-0.0481,+0.0147]
adaptive gate       -0.0457 CI [-0.0790,-0.0131]  -0.0133 CI [-0.0404,+0.0156]
adaptive2 gate      -0.0311 CI [-0.0680,+0.0042]  -0.0002 CI [-0.0330,+0.0351]
```

Interpretation: lower-only no-harm is the safest current candidate because it
keeps the rushshift gap to fixed statistically ambiguous and avoids the
upper-only rush degradation. Adaptive gates are documented negative results:
they still activate upper projection in terminal/rush and do not beat
lower-only. The fixed-headway gap is not closed enough for a top-journal claim;
do not promote an adaptive gate without new evidence.

2026-06-03 fixed-gap follow-up: several 100/200ep attempts were run and should
be treated as negative evidence rather than promotion candidates.

100ep fleet-safe scalarization:

```text
fleetstable / fleetfloor30 changed w_fleet as intended but did not close the
fixed-headway gap. Overall vs fixed-headway stayed statistically ambiguous, and
terminal/rush still carried the gap.
```

100ep proactive lower no-harm:

```text
proactive1 overall vs fixed-headway  +0.0295 CI [-0.0043,+0.0633]
proactive2 overall vs fixed-headway  +0.0306 CI [+0.0016,+0.0606]
hfgate     overall vs fixed-headway  +0.0349 CI [+0.0058,+0.0614]
hfgate2    overall vs fixed-headway  +0.0550 CI [+0.0239,+0.0891]
```

Interpretation: proactive shrink can help highnoise overshoot, but hand gates
trigger in OD/terminal/rush trajectories and damage wait/composite.

200ep lower-only / stronger drift follow-up:

```text
loweronly    overall vs fixed-headway +0.0252 CI [-0.0109,+0.0641]
             terminal +0.0691 and rush +0.0782 were significant regressions
driftcost50  overall vs fixed-headway +0.0524 CI [+0.0164,+0.0918]
adaptdrift   overall vs fixed-headway +0.0528 CI [+0.0202,+0.0860]
```

Interpretation: lower-only no-harm does not survive 200ep as a main promotion.
Stronger/adaptive drift cost improves terminal/rush in isolation but destroys
OD/highnoise tradeoff, so it is not a universal repair.

Current decision: do not promote fleetstable, fleetfloor30, proactive1,
proactive2, hfgate, hfgate2, driftcost50, or adaptdrift. The fixed-headway gap
needs a mechanism change rather than another hand threshold: likely a
holding-value-per-fleet-second guard, or Phase-4 terminal/first-stop execution
that moves useful delay off the on-route fleet.

2026-06-03 value-guard candidate: implemented a causal lower holding guard that
clips an on-route holding action only when its estimated passenger/headway value
is below its fleet-pressure cost. The first local 2-seed / 10ep screen is
promising but not sufficient for promotion:

```text
soft valueguard vs main, composite delta:
terminal   -0.1723 CI [-0.4381,+0.0935]
highnoise  -0.2125 CI [-0.3920,-0.0330]
odshift    +0.2342 CI [+0.2329,+0.2355]
rushshift  +0.0519 CI [-0.3089,+0.4127]
overall    -0.0247 CI [-0.0455,-0.0039]
```

Interpretation: adding headway-correction value fixes the earlier OD/rush damage
from a pure passenger-HF guard, but OD overshoot remains unstable. A valueguard
+ fleet-floor combination was also screened locally and rejected because it
removed the highnoise benefit. Next step is a scheduler-visible 100ep/20seed
valueguard screen against main and fixed-headway before any alias promotion.

2026-06-07 scheduler-visible valueguard screen: the 100ep / 20-seed run on
`node001-node006` rejects the soft valueguard candidate. Against promoted main,
paired composite deltas were:

```text
terminal   +0.0332 CI [+0.0021,+0.0635]
highnoise  +0.0262 CI [-0.0102,+0.0625]
odshift    +0.0276 CI [-0.0078,+0.0592]
rushshift  +0.0844 CI [+0.0531,+0.1153]
overall    +0.0428 CI [+0.0266,+0.0589]
```

The regression is mainly passenger wait: overall wait increased by +0.282s and
rushshift by +0.728s. The guard also drove larger raw lower actions during
training, which indicates policy compensation around the hard action clip.
Decision: keep valueguard as a documented negative candidate, do not promote it.
The next fixed-headway-gap attempt should move useful delay to terminal
execution or first-stop control rather than adding another lower hard gate.

2026-06-08 CST terminal-execution follow-up: three Phase-4-style attempts were
screened on scheduler direct CPU nodes and should not be promoted.

`termhold45` was a control rather than a true new mechanism: promoted main
already uses a 45s terminal shift cap, so the config mostly repeated the same
terminal-dispatch surface. Its 40ep / 20-seed paired composite delta versus
main was overall `+0.0064 CI [-0.0157,+0.0288]`, with no stable gain.

`termfb30` converted completed-trip lower holding history into a terminal
no-early-launch bias, but in the last-k training window the bias collapsed to
near zero (`terminal_feedback_bias_mean` overall `+0.0089s` versus main). The
40ep / 20-seed paired composite deltas were:

```text
terminal   +0.0464 CI [+0.0063,+0.0922]
highnoise  +0.0607 CI [-0.0008,+0.1209]
odshift    -0.0013 CI [-0.0453,+0.0470]
rushshift  -0.0248 CI [-0.0688,+0.0196]
overall    +0.0202 CI [-0.0035,+0.0461]
```

Interpretation: historical lower holding is a weak trigger after the promoted
main has already driven lower holding down; it does not reliably move enough
delay to the terminal.

`termrelief20` used causal on-route fleet pressure to add terminal launch
relief near the fleet budget. It did create real terminal action
(`terminal_feedback_bias_mean` overall `+5.86s`, launch shift `+6.62s`), but
the tradeoff was not universal:

```text
terminal   +0.0376 CI [-0.0238,+0.0966]
highnoise  +0.0691 CI [-0.0387,+0.1664]
odshift    +0.0236 CI [-0.0272,+0.0780]
rushshift  -0.0428 CI [-0.1075,+0.0206]
overall    +0.0219 CI [-0.0147,+0.0577]
```

Mechanism diagnosis: terminal relief improves the rushshift fleet tradeoff, but
it increases terminal/highnoise passenger wait and significantly raises CV
overall (`+0.0057 CI [+0.0012,+0.0101]`). Fixed-headway-gap repair therefore
should not be another unconditional terminal delay rule. The next credible
direction is a value-aware first-stop/terminal action that only converts delay
when predicted fleet relief exceeds the wait/CV cost, or a scoped paper
decision that full terminal dispatch remains future work.

2026-06-08 value-aware terminal screen: `termvalue20` added a causal headway
value gate on top of fleet pressure. The final screened version only delays
when the same-direction dispatch gap is not beyond the target headway plus a
30s tolerance, and it discounts terminal delay under high low-frequency demand.
It produced moderate real terminal action (`terminal_feedback_bias_mean`
overall `+2.44s`, launch shift `+2.22s`) but still did not beat main:

```text
terminal   -0.0117 CI [-0.0825,+0.0563]
highnoise  +0.0232 CI [-0.0737,+0.1119]
odshift    +0.0247 CI [-0.0483,+0.1014]
rushshift  +0.0278 CI [-0.0317,+0.0872]
overall    +0.0160 CI [-0.0178,+0.0486]
```

Mechanism diagnosis: the gate can recover some terminal overshoot
(`-0.0625`) and highnoise CV (`-0.0119`), but it raises lower action and does
not generalize to OD/rush. This closes the current terminal-delay branch as a
main-line repair. Future Phase-4 work should be either a learned first-stop /
terminal action with explicit wait-CV-fleet value estimation, or a paper-scope
decision that target-headway executable timetable is the main contribution.

2026-06-08 lower soft-value-cost follow-up: after hard valueguard compensation
was rejected, `valuesoft35` moved the same HF/passenger/headway value signal
from action clipping into the lower Lagrangian cost. This avoided direct action
pollution but still did not produce a universal gain:

```text
terminal   -0.0252 CI [-0.0778,+0.0299]
highnoise  -0.0119 CI [-0.1046,+0.0784]
odshift    +0.0006 CI [-0.0840,+0.0943]
rushshift  +0.1304 CI [+0.0608,+0.1984]
overall    +0.0235 CI [-0.0111,+0.0552]
```

The failure mode is clear: the added cost is numerically small, but in
rushshift it moves upper launch timing later (`terminal_launch_shift_mean`
`+1.38s`) and raises wait by about `+1.35min`. An LF-safe gate using
`max_low_forecast: 0.48` was then tested to avoid penalizing lower holding
during clear low-frequency/rush shifts, but it worsened the tradeoff further:

```text
terminal   +0.0536 CI [-0.0119,+0.1242]
highnoise  +0.0758 CI [-0.0003,+0.1468]
odshift    +0.0052 CI [-0.0474,+0.0555]
rushshift  +0.1960 CI [+0.1422,+0.2503]
overall    +0.0826 CI [+0.0605,+0.1044]
```

Decision: keep soft lower value-cost as a documented negative branch, not a
main candidate. The fixed-headway gap should not be attacked by more lower
penalties or heuristic terminal-delay rules. The credible remaining options are
either a learned first-stop/terminal value model with explicit wait-CV-fleet
counterfactuals, or a paper-scope decision that the current executable
target-headway timetable is the validated contribution while full terminal
dispatch is Phase 4.

2026-06-10 terminal headway-floor follow-up: tested executable terminal
dispatch floors at `1.00x` and `0.95x` of the base headway to prevent
near-terminal bunching. Both variants are negative and should not be promoted:

```text
headfloor100 vs main  overall +0.0574 CI [+0.0301,+0.0867]
headfloor095 vs main  overall +0.0637 CI [+0.0257,+0.1009]

headfloor100 vs fixed overall +0.0681 CI [+0.0327,+0.1072]
headfloor095 vs fixed overall +0.0744 CI [+0.0377,+0.1138]
```

Mechanism check: `headfloor100` increased overall wait by `+0.530min` versus
main while changing overshoot by only `+0.006`; `headfloor095` increased wait by
`+0.452min` and overshoot by `+0.046`. This rules out a simple terminal
minimum-gap floor as the fixed-headway-gap repair. The next active attempt is an
episode-level causal fixed-expert selector that compares learned FreqDuet
against a no-holding fixed-headway expert through historical composite EMAs;
it must be accepted only if it improves the gap without degenerating into
always selecting fixed-headway.

2026-06-10 selector follow-up: the first `fixedselector` diagnostic run was
submitted to node001-node004, but one shard (`t9384`) was preempted by a
high-priority task and the zhengliang-hpc jump path became temporarily
unreachable, leaving the matrix open. During the wait, a stricter
`fixedselector_balanced` variant was added. It starts selector EMAs only after
`start_ep`, collects both expert observations, and alternates periodic probes so
a positive result cannot be explained by stale learned estimates or a one-way
collapse to fixed-headway. Local smoke passed:

```text
2-seed / 60ep local screen, last-k fixed-expert active rate
terminal   0.650
highnoise  0.300
odshift    0.300
rushshift  0.825
```

This is not performance evidence, only a sanity check that the selector is not
trivially always-fixed. The next valid result must be a scheduler-visible
20-seed / 100ep matrix on node001-node006, compared against both current main
and external fixed-headway.

2026-06-11 diagnostic `fixedselector` result: the original one-way EMA selector
completed `80/80` rows after the node001-node006 entrance recovered. It is a
negative result:

```text
fixedselector vs main   overall +0.0472 CI [+0.0221,+0.0717]
fixedselector vs fixed  overall +0.0579 CI [+0.0254,+0.0915]
```

Its last-k fixed-expert active rate was about `0.64` overall, but it still lost
to current main and fixed-headway, especially in OD/rush. This confirms that a
simple historical EMA selector is not enough; the only remaining selector
candidate is the stricter `fixedselector_balanced` matrix now running as
`fixedselector_balanced_screen_ep100_wu10`.

2026-06-11 `fixedselector_balanced` result: completed `80/80` rows on
node001-node006. It also fails and should not be promoted:

```text
fixedselector_balanced vs main   overall +0.0485 CI [+0.0251,+0.0721]
fixedselector_balanced vs fixed  overall +0.0592 CI [+0.0300,+0.0898]
```

The failure is concentrated in rushshift:

```text
rushshift vs main  +0.1804 CI [+0.1355,+0.2253]
rushshift vs fixed +0.2003 CI [+0.1614,+0.2350]
```

Mechanism check: balanced selection reduced average lower action but also cut
terminal launch shift from the current main's `11.5s` overall to `3.9s`, and
rushshift wait rose by `+1.55min` versus main. Fixed-active and learned-active
episodes were both worse than the current uninterrupted main, so interleaving a
fixed expert is not a safe no-regret wrapper for this training loop.

2026-06-11 evaluation correction: `compare_freqduet_external_baseline.py` now
filters candidate CSVs by inferred `--candidate-method` and errors on duplicate
domain/seed rows after filtering. This fixed a methodological issue where a
multi-method CSV could be treated as one candidate. With the corrected filter,
the clean current main vs fixed-headway 100ep comparison is:

```text
terminal        +0.0039 CI [-0.0417,+0.0495]
highnoise       -0.0027 CI [-0.0610,+0.0576]
odshift         +0.0215 CI [-0.0252,+0.0669]
rushshift       +0.0200 CI [-0.0205,+0.0609]
overall_shared  +0.0107 CI [-0.0244,+0.0472]
```

Interpretation: fixed-headway remains a very strong baseline, but the current
main is no longer significantly worse under the clean 20-seed / 100ep
comparison. The paper claim should be statistical tie/robustness versus
fixed-headway plus clear wins over weaker rule/MPC baselines, not universal
dominance over fixed-headway.

Done means:

- inspect highnoise and odshift seed traces;
- identify whether the gap comes from decomposer smoothing, promotion timing,
  raw-history fallback, lower credit scale, DriftFB, or upper replan cadence;
- test targeted fixes without degrading terminal/rush;
- if no universal fix exists, document the tradeoff and report domain-specific
  confidence intervals.

### 4. Paper-Grade Decomposer Evidence

Status: `[x]` fixed validation package generated; final manuscript selection pending

The harmonic causal decomposer and synthetic evaluator exist, but the paper still
needs direct evidence on real/sim demand traces.

2026-06-03 update: `scripts/make_freqduet_decomposer_figures.py` now writes a
fixed decomposer validation package to
`results_freqduet/decomposer_validation/current_trace`, including synthetic
LF/HF/burst truth, harmonic prior sensitivity, and trace LF/HF alignment CSVs
and figures.

2026-06-08 update: the decomposer package is included in the fixed paper bundle
under `results_freqduet/paper_package/current/figures/decomposer_validation_current_trace`
with source CSVs and PNG/PDF exports. The code path is causal: traces are built
from the online `DemandFrequencyTracker.update(...)` stream and do not use
future bins.

Done means:

- plot `lambda_L` against daily peak structure;
- plot `lambda_H` against local burst events;
- show HF energy versus wait/holding spike alignment;
- audit that the decomposer uses only causal history and no future leakage;
- run cutoff/window sensitivity;
- compare harmonic-prior decomposition against raw history, EMA, and non-prior
  variants.

### 5. FreqDuet-Specific Mechanism Figures

Status: `[x]` fixed FreqDuet figure package generated; final panel selection pending

The existing figure scripts are not yet a complete FreqDuet mechanism package.

2026-06-03 update: `scripts/make_freqduet_mechanism_figures.py` writes
FreqDuet-specific mechanism CSVs and figures to
`results_freqduet/mechanism_figures/current_ep200` and
`results_freqduet/mechanism_figures/promoted_ep200`, including HF-to-holding,
lower drift, promotion active/inactive, method/domain bars, action/state
spectrum, and longtrain drift curves.

2026-06-08 update: these mechanism figures and source tables are included in
`results_freqduet/paper_package/current/figures/mechanism_current_ep200` and
`results_freqduet/paper_package/current/figures/mechanism_promoted_ep200`.

Done means:

- generate HF residual to holding lag plots;
- generate lower LF drift distributions;
- show promotion before/after wait, overshoot, and replan behavior;
- show why `allfreq` pushes lower actions too large or too noisy;
- show leakage and DriftFB effects;
- add action-spectrum / state-spectrum diagnostics;
- save all figures through a fixed script with stable output paths.

### 6. Phase 4 Terminal Dispatch Scope

Status: `[x]` scoped as current-paper target-headway timetable; full terminal dispatch future work

The current promoted path is closer to a target-headway / executable timetable
MVP. `dev_manual.md` also describes a stronger Phase 4 direction involving real
terminal dispatch, actual launch time, and first-stop holding.

Done means one of the following:

- implement and validate the stronger real terminal timetable path as the paper
  main line; or
- explicitly scope the paper method as a target-headway timetable policy and
  move full terminal dispatch to future work / appendix.

If targeting a top transport journal, implementing the stronger Phase 4 path is
preferred unless it destabilizes the method.

2026-06-08 update: the current fixed-headway-gap repair attempts provide
evidence against heuristic terminal delay and lower value-cost penalties as the
Phase 4 shortcut. The remaining decision is now explicit: either implement a
learned first-stop / actual-terminal-launch value model with counterfactual
wait-CV-fleet estimates, or scope full terminal dispatch as future work while
paper-main remains target-headway executable timetable control.

2026-06-10 decision: the current paper scope is now explicitly documented in
`FreqDuet/md/phase4_scope_decision.md`. The promoted contribution remains a
frequency-separated target-headway / executable timetable controller. Full
actual terminal launch rescheduling and first-stop holding are kept as Phase 4
future work unless a learned value model is later implemented and validated.
This decision is based on negative or neutral heuristic evidence: `termhold45`
neutral, `termfb30` near-zero bias and terminal regression, `termrelief20`
real terminal action but wait/CV tradeoff, `termvalue20` weak/negative, and
`valuesoft35` / LF-safe lower value-cost negative.

### 7. External Baselines

Status: `[~]` classical, closest-preserved TransitDuet-family, and SUMO-RL-style baselines complete; original TransitDuet restore optional

Current evidence is mostly FreqDuet internal ablations and TransitDuet-family
comparisons. Top-journal claims need stronger outside baselines.

2026-06-03 update: `scripts/run_freqduet_external_baselines.py` now runs
FreqDuet-format classical baselines using the same config/env perturbations and
paper composite output. It currently covers fixed-headway without holding,
rule-based holding, and a simple MPC/forecast triple controller. A 1-episode
smoke passed, but the full 20-seed external baseline matrix has not been run.
The closest preserved TransitDuet-family baseline was completed on 2026-06-10;
the SUMO-RL-style online RL baseline is now an active screen and remains to be
judged after aggregation.

2026-06-03 late update: the full 100ep / 20-seed external classical matrix was
run under `results_freqduet/external_baselines_promoted_ep100`. Fixed-headway is
the strongest classical baseline; rule-holding and the simple MPC/forecast
controller are much weaker. Promoted main and no-harm candidates beat fixed on
OD shift and are competitive overall, but they do not yet decisively dominate
fixed-headway across terminal/highnoise/rushshift. Added
`scripts/compare_freqduet_external_baseline.py` for paired candidate-vs-external
comparisons with bootstrap CIs.

2026-06-08 update: the remote external classical results have been synced into
the local FreqDuet tree and included in
`results_freqduet/paper_package/current/tables/external_baselines_summary.csv`.

2026-06-10 update: added
`scripts/build_freqduet_transitduet_baseline.py` to extract the closest
preserved TransitDuet-family baseline from the promoted 200ep matrix. This row
is the no-frequency HIRO controller (`nofreq`) evaluated inside the same
FreqDuet runner, domains, seeds, and horizon; it is not claimed to be the
unmodified original TransitDuet repository. The generated package is under
`results_freqduet/transitduet_like_baseline_ep200` and is included in the paper
package tables.

```text
promoted main vs transitduet_like_nofreq_hiro, 200ep composite
terminal   -0.0251  CI [-0.0708,+0.0160]
highnoise  -0.0714  CI [-0.1484,-0.0018]
odshift    -0.0881  CI [-0.1639,-0.0237]
rushshift  -0.1028  CI [-0.1617,-0.0536]
overall    -0.0719  CI [-0.1322,-0.0266]
```

Interpretation: promoted FreqDuet decisively beats the closest preserved
TransitDuet-family no-frequency baseline overall, and significantly in
highnoise, odshift, and rushshift. Terminal remains a statistical tie. A true
original TransitDuet restore can still be added as a stronger appendix baseline
if required, but the current manuscript must label this row as
`transitduet_like_nofreq_hiro`, not original TransitDuet.

2026-06-10 update 2: added a SUMO-RL-style single-level online holding RL
baseline, `sumorl_holdrl`, using the same simulator, lower RE-SAC Lagrangian
trainer, and discrete holding bins, but with upper learning, LF/HF frequency
features, DriftFB, promotion, and frequency-attributed reward disabled. The
100ep / 20-seed / four-domain screen completed under
`results_freqduet/sumorl_holdrl_ep100_wu10_r1`.

```text
sumorl_holdrl vs current main 100ep reference, candidate - main
terminal   +0.0288  CI [-0.0244,+0.0813]
highnoise  +0.0374  CI [-0.0505,+0.1336]
odshift    +0.0554  CI [-0.0085,+0.1260]
rushshift  +0.0571  CI [-0.0047,+0.1466]
overall    +0.0447  CI [-0.0074,+0.1066]

sumorl_holdrl vs fixed-headway, candidate - fixed
terminal   +0.0326  CI [-0.0022,+0.0659]
highnoise  +0.0347  CI [-0.0354,+0.1097]
odshift    +0.0769  CI [+0.0148,+0.1492]
rushshift  +0.0771  CI [+0.0055,+0.1700]
overall    +0.0553  CI [+0.0065,+0.1143]
```

Interpretation: the plain online holding RL baseline is credible enough to
report as external RL evidence, but it is not a strong tuned baseline: it is a
weak statistical tie against current main and significantly worse than
fixed-headway overall, while still beating rule-holding and simple MPC. To avoid
strawman criticism, a stronger richer-state variant,
`sumorl_rawhist_holdrl`, has been launched. It keeps the single-level discrete
holding setup but exposes six trailing realized-demand bins to the lower policy,
matching the SUMO-RL lesson that online control needed larger state plus a
smaller/discrete action space. The active scheduler tasks are `t9027`, `t9036`,
`t9037`, and `t9038`.

2026-06-10 update 3: the richer-state `sumorl_rawhist_holdrl` screen completed
under `results_freqduet/sumorl_rawhist_holdrl_ep100_wu10` with `80/80` expected
rows. It improves over plain `sumorl_holdrl`, especially highnoise/odshift/rush,
but it still does not beat fixed-headway.

```text
sumorl_rawhist_holdrl vs plain sumorl_holdrl, candidate - plain
terminal   +0.0200  CI [-0.0139,+0.0536]
highnoise  -0.0368  CI [-0.1156,+0.0412]
odshift    -0.0450  CI [-0.1172,+0.0172]
rushshift  -0.0359  CI [-0.1270,+0.0257]
overall    -0.0244  CI [-0.0793,+0.0147]

sumorl_rawhist_holdrl vs current main 100ep reference, candidate - main
terminal   +0.0488  CI [+0.0097,+0.0920]
highnoise  +0.0006  CI [-0.0633,+0.0668]
odshift    +0.0104  CI [-0.0407,+0.0603]
rushshift  +0.0213  CI [-0.0172,+0.0613]
overall    +0.0202  CI [-0.0089,+0.0498]

sumorl_rawhist_holdrl vs fixed-headway, candidate - fixed
terminal   +0.0526  CI [+0.0205,+0.0815]
highnoise  -0.0021  CI [-0.0680,+0.0674]
odshift    +0.0319  CI [-0.0061,+0.0682]
rushshift  +0.0412  CI [+0.0063,+0.0777]
overall    +0.0309  CI [+0.0043,+0.0592]
```

Interpretation: the "larger state + discrete holding" online RL baseline has
now been tested and is the fairer external RL row to report. It closes much of
the gap to current main, but fixed-headway remains significantly better overall.
This should be framed as negative external RL evidence: in this simulator,
single-level online holding is not enough; the frequency-separated hierarchy is
needed for the learned controller to be competitive, and fixed-headway remains a
hard classical baseline.

Done means:

- include fixed-headway and rule-based holding baselines;
- include original TransitDuet or closest preserved version;
- include a tuned SUMO-RL-style online RL baseline if feasible;
- include an MPC / forecast-control / classical heuristic baseline if feasible;
- tune baselines enough that they are credible and not strawman comparisons.

### 8. Statistical Rigor

Status: `[~]` paired statistics tooling complete; final paper tables pending

Mean scores are not enough for top-journal claims, especially when domain-level
gaps are close.

2026-06-03 update: current longtrain summarization, candidate comparison, and
promoted-main table generation now write seed-level paired deltas, bootstrap
CIs, and win rates via `scripts/summarize_freqduet_paper_matrix.py`,
`scripts/compare_freqduet_candidate.py`, and
`scripts/build_freqduet_promoted_matrix.py`.

2026-06-03 late update: `scripts/compare_freqduet_external_baseline.py` adds the
same paired-delta/bootstrap-CI workflow for external baselines. Split no-harm,
adaptive, and adaptive2 screens all have seed-level CSVs and paired CI outputs
under `results_freqduet/noharm_*_screen_ep100_wu10`.

Done means:

- preserve seed-level outputs for every table;
- run paired tests against the same seeds;
- report confidence intervals or bootstrap intervals;
- report effect sizes, not only p-values;
- flag negative or ambiguous domains instead of hiding them in averages.

### 9. Reproducibility And Config Hygiene

Status: `[~]` paper manifest started; aliases locked, historical cleanup pending

There are many historical configs, failed ablations, and renamed aliases. This
is acceptable during research but risky for paper reproduction.

2026-06-03 update: `FreqDuet/freqduet/paper_manifest.yaml` now maps current
longtrain, driftcost candidate, held-out generalization, decomposer validation,
and mechanism figures to configs, seeds, scripts, logs, and result paths. The
promoted main aliases now include drift-cost. The remaining work is to move
historical failed configs into a clearly marked layer and keep final table/figure
manifests current as new generalization and baseline results land.

2026-06-08 update: the broad generalization matrix now has a deterministic
config generator / scheduler submitter and a manifest entry
(`broad_generalization_ep100_wu10`). This reduces the risk of manually mixing
old highnoise/odshift/rush aliases with current promoted configs.

2026-06-08 follow-up: added
`scripts/sync_freqduet_broad_generalization.py` and
`scripts/summarize_freqduet_broad_generalization.py`. The first script syncs the
direct-node broad run, calls the existing `run_freqduet_ablation.py
--aggregate-only`, and then generates broad-specific summaries. The second
script recognizes `F_freqduet_broad_<scenario>_<method>_hiro` configs and writes
scenario, family, and overall paired deltas with bootstrap confidence intervals.

Done means:

- create a canonical paper config layer;
- separate `paper_main`, `paper_ablation`, `paper_generalization`, and
  `historical_failed` configs;
- add a result manifest mapping each table/figure to configs, seeds, episodes,
  and scripts;
- provide one-command scripts for final matrix, long training, generalization,
  mechanism figures, and statistical tables;
- avoid ambiguity between old main aliases and current promotion main.

### 10. Theoretical / Method Framing

Status: `[~]` manuscript framing note drafted; final paper text pending

The method framing is strong, but a top-journal paper needs a tighter
assumption-and-mechanism explanation.

2026-06-08 update: added `FreqDuet/md/method_framing.md`, which defines the
paper argument as frequency-separated HRL for exogenous time-series control
rather than generic feature engineering. The note now covers causal filtering,
LF/HF layer allocation, promotion, leakage/DriftFB, evidence hooks, rival
explanations, and current limits. It is included in the paper package manifest;
the remaining work is to turn it into polished manuscript sections.

Done means:

- formalize frequency-separated HRL as exogenous-state decomposition, not just
  feature engineering;
- state the causal filtering assumption and no-future-leakage property;
- explain why LF demand belongs in upper timetable planning and HF residual
  belongs in lower holding;
- define leakage and promotion as cross-frequency correction mechanisms;
- state limits: non-stationary OD shifts, extreme noise, and promotion false
  positives.

### 11. Realism / Data Evidence

Status: `[~]` evidence audit complete; external real calibration still open

For a top transport / ITS journal, simulation-only evidence may be insufficient
unless the simulator is strongly calibrated.

2026-06-08 update: added `FreqDuet/md/realism_data_evidence.md`. Current
evidence supports an OD-driven simulator claim: `env/sim.py` reads
`data/passenger_OD.xlsx`, station/route/timetable spreadsheets are used by the
environment, the harmonic prior is fit from the historical OD table, and the
paper package includes highnoise, odshift, rushshift, trace, decomposer,
mechanism, and external classical baseline artifacts. No local AFC/APC
calibration dataset or multi-day real profile split was found, so this remains a
realism gap rather than a closed paper claim.

Done means:

- document demand calibration source and route assumptions;
- add real or semi-real service-day demand profiles if available;
- hold out route/day/profile families;
- report robustness under fleet-size, dwell-time, and passenger-arrival
  stochasticity changes.

### 12. Paper Tables, Figures, And Negative Results

Status: `[x]` current package assembled; manuscript curation pending

The code has many raw results; the current paper-facing package is now
assembled, with final manuscript curation still pending.

2026-06-08 update: `scripts/build_freqduet_paper_package.py` now assembles the
current paper bundle at `results_freqduet/paper_package/current`. The package
contains 8 core table CSVs, 43 figure/source-data artifacts, the manifest
snapshot, and a generated negative-results appendix. `package_manifest.json`
currently reports `missing=0`.

2026-06-08 follow-up: the package builder also copies manuscript notes under
`manuscript_notes/`, currently including method framing and realism/data evidence
audit notes.

2026-06-08 broad-generalization follow-up: generated paper-generalization config
snapshots are copied under `configs/paper_generalization/` so the broader matrix
can be reproduced from the paper package without regenerating config overlays
first.

2026-06-08 reproduce-script follow-up: the package now also copies 12
paper-facing scripts under `scripts/`, including run, sync, summarize, compare,
figure, and package-builder entry points. The latest package build reports
`copied=131 missing=0`.

2026-06-08 broad-result follow-up: after broad matrix completion, the package
also includes broad generalization per-seed, summary, completion-audit,
method-summary, and paired-delta tables. The latest package build reports
`copied=136 missing=0`.

2026-06-09 config-snapshot follow-up: the package builder now copies explicit
`config_set` snapshots, not only generated `config_dir` folders. This includes
the current longtrain, failed candidates, and the queued `histaux6eg06`
longtrain configs. The latest package build reports `copied=791 missing=0`.

Done means:

- produce final main table;
- produce ablation table;
- produce generalization table;
- produce long-training table;
- produce mechanism figures;
- produce decomposer validation figures;
- include a concise negative-results appendix explaining failed variants.

## Recommended Execution Order

1. Lock the current paper-main config and clean aliases.
2. Run current-version 100/200ep long training across the key baselines.
3. Run systematic held-out generalization.
4. Repair or explain highnoise and odshift weaknesses.
5. Build decomposer and mechanism figure scripts.
6. Add paired statistics / bootstrap reporting.
7. Decide and document the Phase 4 terminal-dispatch scope.
8. Add external baselines.
9. Organize paper configs, manifests, and one-command reproduction scripts.
10. Assemble paper tables, figures, and negative-results appendix.

## Working Rule For Future Development

Do not mark a gap closed just because code exists. Mark it closed only when:

- the implementation is in the promoted FreqDuet copy, not only the old
  TransitDuet tree;
- there is a reproducible config or script;
- there are seed-level results;
- the result is either effective or the failure is documented;
- the conclusion is consistent with `dev_manual.md` and `GPT.md`;
- the current branch has been pushed after a good result.
