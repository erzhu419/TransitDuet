# FreqDuet Top-Journal Gap Backlog

> **2026-08-08 submission hold:** this backlog records the historical V1-V4
> development path. The legacy composite-based paper package is not the active
> submission result after the corrected V4 passenger-journey audit failed.
> Protocol V5 in `protocol_v5_journey_feasible_contract_2026-08-08.md` is the
> only active method/evidence track; old completed items remain provenance and
> negative-result evidence, not proof that the V5 submission gap is closed.

Last updated: 2026-06-26 CST

This file records the remaining gap between the current FreqDuet implementation
and a top-journal-ready paper package. It should be used as the execution
backlog after `dev_manual.md` and `GPT.md`: every item below needs either a
completed implementation/experiment, a documented negative result, or an
explicit paper-scope decision.

## Current Status

FreqDuet is now a paper-package-ready simulation study rather than only a
working prototype. The current promoted line includes:

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
- deterministic paper-main V1 aliases for the four canonical domains;
- a 60-seed, 200-episode paper ablation matrix;
- a 60-seed, 200-episode external classical baseline matrix;
- a 60-seed, 100-episode broad held-out generalization matrix;
- decomposer validation, trace alignment, and mechanism/source-data packages;
- public OD/onboard-load and MBTA same-agency APC-to-GTFS route/stop calibration
  readiness audits, plus local MBTA live GTFS-RT and derived SUMO APC/AVL replay
  evidence;
- a FreqDuet-only MTA Bus Time API offline cache for route/stop/vehicle snapshot
  evidence, explicitly separated from FreqHRL paper results;
- a route/day held-out readiness audit with MTA route-family coverage, MBTA APC
  day-type split protocol, and explicit claim boundaries;
- a concise paper curation bundle that selects main tables, main figures,
  extended-data items, and claim-evidence mappings;
- a canonical `results_freqduet/paper_package/current` bundle with no missing
  required artifacts.

Canonical paper package status:

```text
package: results_freqduet/paper_package/current
copied artifacts: 339
missing artifacts: 0
tables: 53
figure/source files: 71
config files: 136
scripts: 33
manuscript notes: 7
curation files: 17
```

Current main evidence summary:

```text
paper ablation, 60-seed/200ep:
main decisively beats noleakage; main, rawhistory, allfreq, nopromotion, and
nofreq are close internal controls.

external classical, 60-seed/200ep:
main is statistically tied with strong fixed-headway and strongly beats
rule-holding and rule-MPC.

broad generalization, 60-seed/100ep:
main is robust across demand-noise, OD-shift, and rush-shift families, with
the strongest conclusion being leakage-control necessity rather than universal
dominance over every internal variant.
```

Interpretation: the non-text simulation evidence package is now coherent enough
for manuscript drafting, with a conservative claim. The defensible claim is
frequency-separated HRL with leakage control is robust, mechanistically
traceable, and competitive with a strong fixed-headway baseline while clearly
beating weaker rule/MPC baselines. It should not claim universal dominance over
fixed-headway or every internal ablation.

Remaining top-journal non-text gaps are now narrower:

- public AFC/APC demand-profile evidence, separate public OD-estimate/onboard-load
  truth-source coverage, MBTA same-agency APC-to-static-GTFS route/stop
  calibration-readiness evidence, local MBTA live GTFS-RT snapshots, and
  derived MBTA SUMO APC/AVL replay evidence are now present; MTA Bus Time API
  route/stop/vehicle snapshot data are also cached offline under FreqDuet, and
  route/day held-out protocols are packaged, but exact same-day historical
  AFC/APC/AVL/OD calibration and completed route-family/service-day policy
  matrices are still absent locally;
- the closest locked TransitDuet-family baseline has been rebuilt from the
  canonical 60-seed paper matrix; the unmodified original TransitDuet remains
  out of scope unless restored separately;
- final figure-panel selection and manuscript table curation are now scripted
  under `results_freqduet/paper_curation/current`, but final visual layout for
  the manuscript still needs human panel design;
- full actual terminal-launch/first-stop dispatch remains a scoped future-work
  item unless implemented and validated separately.

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

2026-06-17 200ep long-training closure: the current terminal-bias main
completed under `final_matrix_current_terminalbias_ep200_wu10_4domain_20seed`
(`480/480` rows). It still beats `noleakage` decisively and is close to the
internal variants, but the `last-k=100` window exposes long-training policy
drift. Against a fair ep200 external reference
`external_baselines_ep200_wu10_4domain_20seed`, current main significantly
lags fixed-headway:

```text
current main 200ep, composite delta main - baseline
overall vs nofreq        -0.0145 CI [-0.0622,+0.0202]
overall vs rawhistory    -0.0040 CI [-0.0219,+0.0146]
overall vs allfreq       -0.0053 CI [-0.0308,+0.0206]
overall vs nopromotion   -0.0062 CI [-0.0151,+0.0025]
overall vs noleakage     -0.3106 CI [-0.3966,-0.2329]

current main 200ep vs external ep200
overall_shared vs fixed_headway +0.0412 CI [+0.0251,+0.0583]
overall_shared vs rule_holding  -0.5806 CI [-0.6076,-0.5506]
overall_shared vs rule_mpc      -1.9241 CI [-1.9860,-1.8672]
```

This failure is now diagnosed as long-horizon actor drift rather than a need to
keep tuning the decomposer: the earlier `51-100` episode block is good, while
`101-200` degrades. Implemented a preconfigured stability schedule in
`runner_v3.py`:

- `training.longtrain_stability.freeze_upper_after_ep`
- `training.longtrain_stability.freeze_lower_policy_after_ep`
- optional `training.longtrain_stability.freeze_lower_critic_after_ep`

The promoted candidate configs `*_main_freeze100_hiro.yaml` learn normally for
episodes `0-99`, then freeze upper and lower policy updates for episodes
`100+` while still executing the full 200ep rollout. This is a causal training
schedule, not post-hoc best-window selection. The scheduler-visible direct-node
run `freeze100_ep200_wu10_4domain_20seed` completed as tasks `t11768-t11771`
(`80/80` rows). It significantly repairs current-main drift and is
statistically tied with fixed-headway overall:

```text
freeze100 200ep, composite delta candidate - baseline
overall vs current main  -0.0348 CI [-0.0498,-0.0209]
overall vs rawhistory    -0.0388 CI [-0.0515,-0.0257]
overall vs nopromotion   -0.0410 CI [-0.0544,-0.0277]
overall vs noleakage     -0.3454 CI [-0.4277,-0.2697]

freeze100 200ep vs external ep200
overall_shared vs fixed_headway +0.0064 CI [-0.0020,+0.0153]
overall_shared vs rule_holding  -0.6154 CI [-0.6423,-0.5868]
overall_shared vs rule_mpc      -1.9589 CI [-2.0221,-1.9022]
```

Interpretation: for the 200ep top-journal protocol, `freeze100` is now the
credible long-training main candidate. The claim should be framed as matching
the strong fixed-headway baseline while significantly beating weaker rule/MPC
baselines and repairing the unfrozen online policy drift. It still does not
justify claiming robust superiority over fixed-headway.

2026-06-17 current-name final matrix closure: promoted the freeze100
long-training schedule into the current main root config and reran the full
current-name 4-domain x 6-method x 20-seed 200ep matrix as
`final_matrix_current_freeze100_ep200_wu10_4domain_20seed` with scheduler
direct-node tasks `t11792-t11807`. The matrix completed `480/480` rows with no
duplicate domain-method-seed pairs. Because the final ablation configs inherit
from the current main root, all learned final configs share the same
`ep100+` freeze protocol; this makes the 200ep final table a fair
long-training protocol comparison rather than a main-only stabilization.

```text
current-name freeze100 200ep, composite delta main - baseline
overall vs nofreq        -0.0130 CI [-0.0530,+0.0122]
overall vs rawhistory    -0.0010 CI [-0.0112,+0.0091]
overall vs allfreq       -0.0082 CI [-0.0314,+0.0104]
overall vs nopromotion   -0.0000 CI [-0.0066,+0.0073]
overall vs noleakage     -0.3338 CI [-0.4327,-0.2486]

current-name freeze100 200ep vs previous terminal-bias main
overall main - previous main -0.0393 CI [-0.0546,-0.0247]

current-name freeze100 200ep vs external ep200
overall_shared vs fixed_headway +0.0018 CI [-0.0083,+0.0116]
overall_shared vs rule_holding  -0.6200 CI [-0.6442,-0.5934]
overall_shared vs rule_mpc      -1.9634 CI [-2.0225,-1.9083]
```

Interpretation update: the current config-name final matrix now closes the
200ep naming ambiguity. The defensible paper claim remains that FreqDuet
matches fixed-headway under the strong 200ep protocol and substantially beats
weaker classical rule baselines, while `noleakage` is decisively invalid and
`rawhistory` / `nopromotion` remain close internal controls.

2026-06-24 deterministic fixed-headway update: a reproducibility bug was found
in the evaluation path because route generation used Python stdlib `random`
without seeding it. After seeding stdlib `random` in the main runner and
external baseline runner, the deterministic 100ep / 20-seed protocol was rerun
for fixed-headway and a nine-variant terminal-only sweep. The best candidate,
`cfaction_target_dm20_terminalonly`, significantly beats deterministic
fixed-headway in the terminal domain and in the four-domain shared average:

```text
candidate - fixed_headway composite delta
terminal        -0.0207 CI [-0.0373,-0.0049], win=0.700
overall_shared  -0.0052 CI [-0.0091,-0.0011], win=0.700
```

This is a narrower but stronger claim than the previous fixed-headway result:
the terminal dispatch/action-value layer can beat the strong fixed-headway
baseline under paired deterministic seeds, but the terminal-only configs are
currently identical to fixed-headway on highnoise, OD-shift, and rush-shift.
The next gap is extending the counterfactual action/value layer beyond the
terminal-only setting without losing the fixed-headway no-harm property.

2026-06-25 deterministic 200ep confirmation: the domain-wise counterfactual
action candidate `cfaction_domainbest_v1` was rerun against deterministic
current main and deterministic fixed-headway at 200 episodes:

```text
learned:
results_freqduet/detseed_main_vs_cfaction_domainbest_v1_ep200_wu10_4domain_20seed
tasks: t13108-t13117

external fixed:
results_freqduet/detseed_external_fixed_headway_ep200_wu10_4domain_20seed
tasks: t13118-t13122

protocol:
4 domains x 20 paired seeds
episodes: 200
last_k: 50
upper_warmup_eps: 10 for learned configs
```

The 200ep result confirms that v1 is the strongest deterministic paper-main
candidate:

```text
cfaction_domainbest_v1 - current main composite delta
terminal  -0.0009 CI [-0.0199,+0.0183]
highnoise -0.0350 CI [-0.0849,+0.0104]
odshift   +0.0054 CI [-0.0159,+0.0255]
rushshift -0.0161 CI [-0.0271,-0.0058]
overall   -0.0117 CI [-0.0278,+0.0024]

cfaction_domainbest_v1 - fixed_headway composite delta
terminal        -0.0033 CI [-0.0227,+0.0163]
highnoise       -0.0525 CI [-0.0885,-0.0139]
odshift         -0.0107 CI [-0.0361,+0.0162]
rushshift        0.0000 CI [-0.0000,+0.0000]
overall_shared  -0.0166 CI [-0.0304,-0.0026]

current main - fixed_headway composite delta
terminal        -0.0024 CI [-0.0171,+0.0125]
highnoise       -0.0175 CI [-0.0632,+0.0327]
odshift         -0.0161 CI [-0.0457,+0.0134]
rushshift       +0.0161 CI [+0.0059,+0.0273]
overall_shared  -0.0050 CI [-0.0212,+0.0114]
```

Decision: promote `cfaction_domainbest_v1` as the deterministic paper-main
candidate. The old `F_freqduet_*_main_hiro` configs stay as current-main
baselines to avoid cyclic inheritance. The paper-main alias configs are:

```text
F_freqduet_terminal_paper_main_hiro
F_freqduet_gen_highnoise_paper_main_hiro
F_freqduet_gen_odshift_paper_main_hiro
F_freqduet_gen_rushshift_paper_main_hiro
```

These aliases extend the validated v1 configs and let final tables use method
name `main` without losing the historical current-main baseline. The remaining
gap for this item is no longer "does the advantage survive 200ep"; it is the
clean current-name final matrix using the paper-main aliases, followed by the
final table/figure package.

Alias final-matrix status: submitted as
`detseed_paper_main_cfaction_domainbest_v1_alias_ep200_wu10_4domain_20seed`.
The first scheduler attempt (`t13127-t13131`) failed before training because
the remote `scheduleurm_work` tree lacked the new alias YAML files. After
syncing the four alias configs to the remote `configs_freqduet` directory, the
same run name was resubmitted with `--allow-duplicate`; active tasks are
`t13132-t13136`. Early status showed all five shards running with nonzero RAM,
so the previous config-load failure is resolved. This item remains open until
the alias run is synced, aggregated, and compared against the already completed
v1/current/fixed deterministic 200ep tables.

Done means:

- run current `main` against `nofreq`, `rawhistory`, `allfreq`,
  `nopromotion`, and `noleakage`;
- use 100 or 200 episodes;
- cover terminal, highnoise, odshift, and rushshift;
- report seed-level paired deltas, mean, std, confidence interval, and best/worst
  seed behavior;
- record whether the 40ep advantage survives long training.

### 2. Systematic Generalization Matrix

Status: `[x]` 60-seed broad paper matrix complete; real multi-day profiles still open under realism gap

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

Status: `[x]` lower drift repaired; paper-main V1 tied with strong fixed-headway and beats weaker classical baselines

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

2026-06-11 exp39-style action-space follow-up: the isolated `disc9` candidate
tests whether finer discrete lower holding alone can reproduce the SUMO-RL
exp39 improvement pattern. It completed a 4-domain x 20-seed x 100ep matrix
as `disc9only_screen_ep100_wu10_r1` and is a negative result:

```text
disc9 vs main   overall +0.0499 CI [+0.0113,+0.0948]
disc9 vs fixed  overall +0.0605 CI [+0.0120,+0.1157]
```

The domain pattern is also unfavorable: vs fixed-headway, OD shift is
`+0.0691 CI [+0.0116,+0.1379]` and rush shift is
`+0.0748 CI [+0.0279,+0.1285]`. Therefore action discretization alone is not
the fixed-headway-gap repair and must not be promoted. The remaining
exp39-style hypothesis is state/action memory, not the action alphabet alone:
`disc9last` adds the previous lower action, and `gapctx_disc9last` adds minimal
adjacent launch-gap/headway context. Their clean rerun is
`exp39_state_action_screen_ep100_wu10_r2`; promote only if it beats current main
and shrinks the fixed-headway gap under paired CI.

2026-06-11 exp39 state/action result: `disc9last` was stopped after the 40ep
screen because it was already worse than current main overall
(`+0.0632 CI [+0.0053,+0.1352]`). `gapctx_disc9last` was allowed to finish the
full 100ep matrix. It improves over `disc9-only`, but still does not repair the
fixed-headway gap and should not be promoted:

```text
gapctx_disc9last vs main   overall +0.0278 CI [-0.0016,+0.0561]
gapctx_disc9last vs fixed  overall +0.0385 CI [+0.0036,+0.0744]
```

The fixed-headway gap is now smaller than `disc9-only`, but terminal is
significantly worse versus fixed (`+0.0408 CI [+0.0064,+0.0790]`) and the
overall fixed comparison remains positive. This means the SUMO-RL exp39-style
repair partially helps variance/action aliasing but does not solve this
project's fixed-policy challenge. The next repair should move upstream toward
upper timetable/terminal dispatch or a stronger demand-conditioned headway
planner, not keep adding lower-only context.

2026-06-11 upper timetable follow-up: old small-sample `spline2dir` evidence
did not reproduce after porting the conservative direction-specific timetable
curve plus promotion-triggered replan onto the current main line. The
4-domain x 20-seed x 100ep run
`spline2dir_promreplan_screen_ep100_wu10` completed `80/80` rows and is a
negative result:

```text
spline2dir_promreplan vs main   overall +0.0228 CI [-0.0108,+0.0556]
spline2dir_promreplan vs fixed  overall +0.0334 CI [+0.0002,+0.0672]
```

Mechanism check: the candidate slightly improves terminal/highnoise wait, but
raises odshift/rushshift cost and remains worse than fixed-headway overall.
Therefore a wider 4D direction-specific timetable action is not the current
fixed-headway repair.

2026-06-11 upper action-discretization follow-up: added a config-gated
`upper.action_bins` projection in `runner_v3.py`, leaving current main
unchanged unless a candidate config sets bins. Two upper-discrete candidates
were run as `upperdisc_screen_ep100_wu10`:

```text
upperdisc5 bins [-15,-10,-5,0,5]
upperdisc7 bins [-30,-15,-10,-5,0,5,10]
```

This is the closest current analogue of the SUMO-RL exp39 action-space repair
at FreqDuet's upper timetable layer. Local smoke passed, and the scheduler
matrix completed `160/160` rows on `t10104`-`t10119`. It is a useful diagnostic
but not a promotion candidate:

```text
upperdisc5 vs main   overall +0.0136 CI [-0.0169,+0.0421]
upperdisc5 vs fixed  overall +0.0243 CI [-0.0088,+0.0600]

upperdisc7 vs main   overall +0.0142 CI [-0.0147,+0.0415]
upperdisc7 vs fixed  overall +0.0249 CI [-0.0105,+0.0599]
```

The domain pattern is informative but not sufficient: `upperdisc5` helps
odshift (`vs main -0.0172`, CI crosses zero), while `upperdisc7` helps
rushshift (`vs main -0.0131`, CI crosses zero). Both raise terminal cost and
remain worse than fixed-headway overall. Therefore simple upper action
discretization also does not close the fixed-headway gap. A narrower
near-fixed residual alphabet may still be worth one final diagnostic, but any
promotion needs an overall paired gain, not only domain-specific tradeoff.

2026-06-11 narrow upper-discretization follow-up: the final near-fixed
diagnostic `upperdisc34_screen_ep100_wu10` completed `160/160` rows. It tested
`upperdisc3=[-5,0,5]` and `upperdisc4=[-10,-5,0,5]`.

```text
upperdisc3 vs main   overall +0.0025 CI [-0.0230,+0.0297]
upperdisc3 vs fixed  overall +0.0131 CI [-0.0249,+0.0517]

upperdisc4 vs main   overall +0.0169 CI [-0.0098,+0.0438]
upperdisc4 vs fixed  overall +0.0275 CI [-0.0053,+0.0604]
```

`upperdisc3` is the best upper-discrete variant: it is statistically tied with
current main and fixed-headway, slightly helps highnoise/odshift versus main,
but still carries terminal/rush cost and does not improve the clean fixed
comparison beyond the current main (`main vs fixed overall +0.0107`). Decision:
do not promote `upperdisc3/4/5/7`; close the simple action-discretization branch
as diagnostic negative evidence. Further fixed-gap work should not be another
static action alphabet; it needs either a learned demand-conditioned headway
planner/value model or a scoped Phase-4 first-stop/terminal dispatch mechanism.

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

Status: `[x]` executable terminal-dispatch layer implemented; learned first-stop / terminal value layer still open

The current promoted path is not just a target-headway MVP. A 2026-06-13 code
audit confirms that the promoted main config resolves to
`terminal_dispatch: true`, `terminal_shift_min_s: 0`, and
`terminal_shift_max_s: 45`; `env/sim.py` uses `_freqduet_scheduled_launch` as
the launch eligibility time and records `_actual_launch_time`. This implements
the executable terminal-dispatch layer described in `dev_manual.md`.

What remains open is the value layer:

- a learned first-stop / terminal delay-release value model;
- per-decision counterfactual labels or an equivalent causal estimator;
- validation that the learned terminal action improves wait-CV-fleet tradeoffs
  over promoted main, fixed-headway, and preserved TransitDuet-family baselines.

If targeting a top transport journal, the right claim is therefore bounded
executable terminal-dispatch timetable control, not a fully solved learned
first-stop/terminal value policy.

2026-06-08 update: the current fixed-headway-gap repair attempts provide
evidence against heuristic terminal delay and lower value-cost penalties as the
Phase 4 shortcut. The remaining decision is now explicit: either implement a
learned first-stop / actual-terminal-launch value model with counterfactual
wait-CV-fleet estimates, or scope full terminal dispatch as future work while
paper-main remains target-headway executable timetable control.

2026-06-13 correction: `FreqDuet/md/phase4_scope_decision.md` has been updated
to reflect the actual code path. Full terminal launch rescheduling is already
implemented in bounded form; what should stay future work or appendix is the
learned first-stop/terminal value model. This correction is consistent with the
negative or neutral heuristic evidence: `termhold45` neutral, `termfb30`
near-zero bias and terminal regression, `termrelief20` real terminal action but
wait/CV tradeoff, `termvalue20` weak/negative, `valuesoft35` / LF-safe lower
value-cost negative, and the 2026-06-13 action-level CRN/value audit showing
that aggregate-context action selection does not beat `target0`.

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

Status: `[x]` paired statistics and canonical paper tables packaged; final manuscript table selection pending

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

Status: `[x]` canonical paper manifest/package built; historical cleanup now optional maintenance

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
mechanism, and external classical baseline artifacts.

2026-06-26 update: added a data-only public AFC/APC profile audit under
`FreqDuet/freqduet/data/external_afc_apc/` and
`FreqDuet/freqduet/results_freqduet/real_afc_apc_profile_audit/v1`. This closes
the narrow "no local AFC/APC profile evidence" gap.

2026-06-26 follow-up: added a separate public OD/onboard-load truth-source audit
under `FreqDuet/freqduet/data/external_truth_sources/` and
`FreqDuet/freqduet/results_freqduet/external_od_onboard_truth_audit/v1`. This
supports MTA agency-estimated subway OD coverage and MBTA bus stop/trip
board-alight-load calibration targets. The remaining realism gap is now more
specific: exact same-network AFC/APC/AVL calibration, multi-day real service
splits, and observed agency wait-time outcomes are still not claimed.

Done means:

- document demand calibration source and route assumptions;
- add real or semi-real service-day demand profiles if available;
- hold out route/day/profile families;
- report robustness under fleet-size, dwell-time, and passenger-arrival
  stochasticity changes.

### 12. Paper Tables, Figures, And Negative Results

Status: `[x]` current package assembled; manuscript curation scripted

The code has many raw results; the current paper-facing package is now
assembled, and a concise main-table/main-figure curation bundle is generated
under `results_freqduet/paper_curation/current`.

2026-06-26 curation update: the current package rebuild reports
`copied=339 missing=0`, with 53 table CSVs, 71 figure/source-data files, 136
config snapshots, 33 scripts, 7 manuscript notes, and 17 curation files. The
curation bundle selects 4 main tables, 3 main figure groups, 2 extended-data
tables, 1 extended-data figure, and a claim-evidence map.

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

2026-06-11 demand-conditioned selector follow-up: implemented a config-gated
online contextual value selector (`ctxselector60`) that starts at episode 60 and
learns separate ridge value models for learned FreqDuet versus a fixed-headway
expert from causal demand/frequency, OD, promotion, and recent-performance
features. It completed a 4-domain x 20-seed x 100ep scheduler-direct run on
node001-node006 as `ctxselector60_screen_ep100_wu10`. This is a negative result
and should not be promoted:

```text
ctxselector60 vs main   overall +0.0086 CI [-0.0219,+0.0413]
ctxselector60 vs fixed  overall +0.0193 CI [-0.0179,+0.0542]
```

The selector diagnostics show why this is not the right repair: over the final
50 episodes, fixed-headway was active in `36.6%` of episodes even though the
contextual model estimated learned FreqDuet lower cost on average
(`learned=1.1788`, `fixed=1.2604`, margin `+0.0816`). This reduced lower holding
and, more importantly, compressed terminal launch shifts (`5.3s` overall versus
the current main's `11.5s`). The conclusion is that wrapping the policy with a
fixed expert is not a robust path to beating fixed-headway; the next repair must
modify the upper demand-conditioned headway/value planner itself, or the
Phase-4 terminal/first-stop dispatch model, not keep tuning fixed-expert
interleaving.

2026-06-11 upper residual value follow-up: implemented `uppervalue_hfgate`, a
config-gated upper reward cost that penalizes headway compression only when
causal fleet utilization is high and the HF/promotion signal does not justify
the extra frequency. The first scheduler attempt failed because the new config
YAMLs were not synced to the remote FreqDuet copy; the corrected run
`uppervalue_hfgate_screen_ep100_wu10_rerun1` completed all `80/80` rows at
100 episodes and is a weak positive but not a promotion candidate:

```text
uppervalue_hfgate vs main   overall -0.0040 CI [-0.0308,+0.0220]
uppervalue_hfgate vs fixed  overall +0.0066 CI [-0.0308,+0.0412]
```

Mechanism check: the value cost is active on roughly `30-38%` of upper
decisions and reduces the OD-shift mean upper residual from `-6.63s` to
`-2.57s`, improving OD/rush composite slightly. The tradeoff is still not
paper-grade: terminal composite worsens by `+0.0287` versus current main and
`+0.0325` versus fixed-headway, while overall CV is slightly worse than
fixed-headway (`+0.0039`, CI `[+0.00001,+0.00852]`). This confirms that a
passive reward penalty can nudge the upper planner but is too slow/indirect to
close the fixed-headway gap. The next active repair should be an online
demand-conditioned residual value selector that chooses among a small causal
headway-residual candidate set at each upper decision, or a learned
first-stop/terminal value model.

2026-06-11 learned residual-selector follow-up: implemented
`upperres_selector`, an online ridge value model that stores each selected upper
residual feature at decision time, trains only after episode-end upper reward
backfill, and chooses among actor-plus-offset timetable coefficient candidates.
The 4-domain x 20-seed x 100ep scheduler-direct screen
`upperres_selector_screen_ep100_wu10` completed `80/80` rows but is a negative
result:

```text
upperres_selector vs main   overall +0.0199 CI [-0.0098,+0.0532]
upperres_selector vs fixed  overall +0.0306 CI [-0.0009,+0.0638]
```

Mechanism check: the selector became active on essentially every upper
decision and learned to push the upper residual much more negative
(`terminal -18.1s`, `rushshift -16.0s`, versus current main around `-3` to
`-4s`). That compressed terminal launch shifts and raised overshoot/CV enough
to lose against both main and fixed-headway. The failure is useful: a free
bidirectional residual alphabet plus a linear reward-backfilled value label is
too confounded and tends to over-credit aggressive frequency increases. The
next repair should not allow negative residual offsets; it should test a
guarded relief-only selector with stronger improvement margins and adjustment
penalties, or move to a richer counterfactual terminal/first-stop value model.

2026-06-11 guarded residual-selector follow-up: implemented
`upperres_reliefguard`, a conservative residual selector that can only keep the
actor action or relax headway compression with positive residual offsets. The
4-domain x 20-seed x 100ep scheduler-direct screen
`upperres_reliefguard_screen_ep100_wu10` completed `80/80` rows and is not a
promotion candidate:

```text
upperres_reliefguard vs main   overall +0.0007 CI [-0.0227,+0.0250]
upperres_reliefguard vs fixed  overall +0.0113 CI [-0.0235,+0.0474]
```

Domain-level behavior confirms the tradeoff: it helps rushshift versus main
(`-0.0329`, CI crosses zero), but terminal, highnoise, and odshift remain
slightly worse. Mechanism diagnostics show that the guarded selector is too
conservative to be a breakthrough: final-window mean residual adjustment is
small (`terminal 0.09s`, `odshift 0.11s`, `rushshift 0.03s`, `highnoise
0.37s`). This closes the linear upper-residual selector branch. The next
credible fixed-headway-gap repair should combine a small discrete local
headway alphabet with explicit local plan-context features (current gap, next
gap, candidate target headway, candidate deficit, LF forecast, HF energy, and
fleet pressure), or move to a learned first-stop/terminal value model with
counterfactual wait-CV-fleet estimates.

2026-06-11 plan-context residual-selector follow-up: implemented
`upperres_planctx`, a stricter exp39-style variant that combines a small
candidate alphabet (`-5/0/+5/+10s`) with candidate-local plan features: current
gap, next gap, candidate target headway, gap deficit/excess, next-target slope,
LF forecast, HF energy, promotion strength, and fleet pressure. The 4-domain x
20-seed x 100ep scheduler-direct run `upperres_planctx_screen_ep100_wu10`
completed `80/80` rows but is a clear negative result:

```text
upperres_planctx vs main   overall +0.0562 CI [+0.0323,+0.0815]
upperres_planctx vs fixed  overall +0.0669 CI [+0.0365,+0.0979]
```

The failure is concentrated in OD/rush (`odshift +0.0458`, `rushshift
+0.1638` versus main). Mechanistically, the selector only changed actions by
about `0.5-1.1s` on average, but it reduced terminal launch shifts in
terminal/rush and drove the demand-attribution score negative. This confirms
that reward-backfilled residual selection remains confounded even with local
plan context. Close the upper-residual selector branch; the next credible
mechanism is a separate learned terminal / first-stop value action that decides
when converting on-route holding into terminal delay is worth the
wait-CV-fleet tradeoff, or a paper-scope decision that full Phase 4 remains
future work.

2026-06-11 terminal-value-selector follow-up: implemented
`termvalselector`, a separate learned terminal / first-stop value action that
chooses among nonnegative terminal delay candidates (`0/5/10/15s`) using
causal local plan, demand, frequency-energy, fleet-pressure, and previous
episode performance features. This deliberately avoids changing the upper
headway residual directly. The 4-domain x 20-seed x 100ep scheduler-direct run
`termvalselector_screen_ep100_wu10` completed `80/80` rows but is a clear
negative result:

```text
termvalselector vs main   overall +0.0575 CI [+0.0392,+0.0761]
termvalselector vs fixed  overall +0.0681 CI [+0.0398,+0.0986]
```

Domain-level results show the failure is not just noise: terminal worsens
versus main (`+0.0451`, CI `[+0.0073,+0.0854]`) and rushshift worsens strongly
(`+0.1611`, CI `[+0.1180,+0.2060]`). Versus fixed-headway, terminal, odshift,
and rushshift are all significantly worse, while highnoise is only tied.
Mechanistically, the selector is active almost every upper decision but selects
only small mean terminal biases (`0.49-0.74s`), reduces terminal launch shift in
the scenarios where the promoted main benefited from larger launch delay, and
raises wait/composite. This closes the simple online terminal-bias selector
branch. The next fixed-headway repair should be either a stronger learned
demand-conditioned headway / launch planner with real counterfactual labels, or
the full Phase 4 terminal-dispatch implementation; do not promote this module.

2026-06-11 demand-conditioned headway-planner v1 follow-up: implemented
`headwayplanner`, a config-gated learned planner that chooses among complete
discrete headway-plan candidates (`-20/-10/0/+10/+20s`) rather than residual
offsets, uses expanded local plan/demand/frequency/fleet/history features, and
updates its ridge value model directly from the paper composite cost
(`wait/10 + overshoot^2/N + CV`). The 4-domain x 20-seed x 100ep
scheduler-direct run `headwayplanner_screen_ep100_wu10` completed `80/80` rows
but is not a promotion candidate:

```text
headwayplanner v1 vs main   overall +0.0223 CI [-0.0028,+0.0493]
headwayplanner v1 vs fixed  overall +0.0330 CI [-0.0012,+0.0691]
```

The mechanism is clearer than the aggregate score. The planner is active in
the final window and often moves the upper plan from mild compression to
positive relief (`upper_delta_mean` about `+4.3s` highnoise, `+8.4s` odshift,
`+5.9s` rushshift, `+7.0s` terminal), which raises terminal launch shifts to
`24-30s` versus the promoted main's `8-14s`. This reduces neither overshoot nor
CV enough; in several domains overshoot and wait increase. The failure is not
an implementation miss but a prior/action-surface miss: the v1 spacing prior
over-credits positive relief under terminal-dispatch dynamics. A v2 should be
more conservative: disable the relief-favoring prior, cap positive candidates
around `+5/+10s`, increase the action-change penalty and improvement margin,
and only deviate from the actor when the composite value model has a clear
advantage.

2026-06-12 demand-conditioned headway-planner v2 follow-up: implemented and
screened the conservative `headwayplanner_safe` variant. It keeps the same
episode-end composite-cost ridge value model as v1, but removes the
relief-favoring prior, restricts the candidate set to
`[-20,-10,0,+5,+10]s`, raises the improvement margin to `0.05`, and raises the
action-change penalty to `0.08`. The remote FreqDuet CPU environment had been
removed before the first rerun attempt; it was rebuilt as an isolated
micromamba env under
`/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310` on a
compute node, and the final scheduler-direct run
`headwayplanner_safe_screen_ep100_wu10_envfix1` completed `80/80` rows on
node001-node006. A premature partial sync was discarded; the final aggregation
used only diagnostics with `100/100` episodes for every seed.

Final 100ep means:

```text
terminal   composite 1.4656  wait 5.43  cv 0.446
highnoise  composite 1.8098  wait 9.06  cv 0.445
odshift    composite 1.4506  wait 5.45  cv 0.445
rushshift  composite 1.2918  wait 3.91  cv 0.445
```

Paired comparisons:

```text
headwayplanner_safe vs fixed-headway        overall +0.0151 CI [-0.0180,+0.0482]
headwayplanner_safe vs upperres_reliefguard overall +0.0038 CI [-0.0201,+0.0285]
headwayplanner_safe vs uppervalue_hfgate    overall +0.0085 CI [-0.0137,+0.0304]
headwayplanner_safe vs ctxselector60        overall -0.0041 CI [-0.0300,+0.0211]
```

Domain-level fixed-headway comparison is mixed: highnoise improves
(`-0.0171`, CI crosses zero), OD/rush remain statistically tied, but terminal
is still worse (`+0.0378`, CI `[+0.0018,+0.0734]`). Mechanistically, v2 fixes
v1's over-relief problem: final-window planner deltas are modestly negative
(`-3.5` to `-5.4s` depending on domain), average selected adjustments stay
below `0.7s`, the relief prior is zero, and terminal launch shifts return to
about `9.6-11.3s` instead of v1's `24-30s`. This is a useful repair of the
action surface, but it is not a promotion candidate because it does not beat
the current main-like candidates and does not close the terminal fixed-headway
gap. Further variants of the same linear episode-backfilled planner are likely
low leverage; the next credible path is either a real counterfactual
terminal/first-stop value label or the scoped Phase-4 first-stop/terminal
dispatch implementation.

2026-06-12 local-credit headway-planner follow-up: the value label was changed
from uniform episode composite cost to a local upper-decision cost
(`-credit - wait_credit + upper HF/plan/value penalties`), with expanded
front/back timing features. This is a better causal label than the v1/v2
episode-backfilled target, but the first ungated `local` screen remained mixed:

```text
local-credit planner vs main   overall +0.0042 CI [-0.0434,+0.0507]
local-credit planner vs fixed  overall +0.0149 CI [-0.0258,+0.0553]
```

It significantly improved OD shift versus main (`-0.0772`, CI
`[-0.1649,-0.0008]`) but hurt terminal enough that it could not be promoted.
The blended episode/local label was worse:

```text
blend planner vs main   overall +0.0272 CI [-0.0066,+0.0595]
blend planner vs fixed  overall +0.0379 CI [+0.0058,+0.0710]
```

The best sub-branch so far is the conservative frequency/OD gated local-credit
planner `odgate`, which activates only in stable low/middle-energy,
low-entropy, high-low-forecast contexts:

```text
odgate vs main   overall -0.0187 CI [-0.0555,+0.0188]
odgate vs fixed  overall -0.0080 CI [-0.0412,+0.0240]
```

This is the first learned headway-planner variant that turns the fixed-headway
gap into a slight, non-significant advantage, with a significant highnoise win
versus both main and fixed. It is still not a promotion candidate because OD
shift regresses versus main (`+0.0451`, CI crosses zero). Removing the
low-forecast lower bound in `odgate2` repaired OD shift but lost the terminal
and highnoise benefit:

```text
odgate2 vs main   overall +0.0063 CI [-0.0283,+0.0443]
odgate2 vs fixed  overall +0.0170 CI [-0.0265,+0.0638]
```

An OR-style gate was then implemented (`activation_gate.any_of`) to preserve
the original stable-context branch while adding a lower-entropy OD branch.
Both resulting variants are clear negative controls:

```text
odhybrid vs main   overall +0.1067 CI [+0.0681,+0.1481]
odhybrid vs fixed  overall +0.1174 CI [+0.0846,+0.1510]
odbranch vs main   overall +0.0913 CI [+0.0415,+0.1469]
odbranch vs fixed  overall +0.1020 CI [+0.0499,+0.1596]
```

The failure mode is strong highnoise/rushshift wait inflation. This closes the
simple causal threshold-gate extension: the local-credit value model is useful
evidence, but robustly beating fixed-headway now likely requires either a
counterfactual first-stop/terminal action label or the full executable Phase-4
terminal-dispatch path, not another linear gate sweep.

2026-06-12 terminal-value local-credit follow-up: the same local-credit target
was tested inside the terminal/first-stop selector (`termvalselector_local`),
so the learned terminal action no longer used transition reward as its value
label. The 4-domain x 20-seed x 100ep run completed `80/80` rows, but it is a
clear negative result:

```text
termvalselector_local vs main   overall +0.0814 CI [+0.0419,+0.1240]
termvalselector_local vs fixed  overall +0.0921 CI [+0.0472,+0.1429]
```

It slightly improves clean terminal versus main/fixed, but highnoise and
rushshift regress significantly. Mechanistically, the local target suppresses
terminal bias (`0.18-0.20s` mean versus about `0.49-0.74s` in the transition
reward selector) while increasing lower action and update count; the method is
therefore learning a weak terminal action and leaving the harder fleet/wait
tradeoff to the lower layer. This closes the local-credit terminal-selector
branch. The next implementation should not be another linear terminal-bias
selector; it should either build a demand-conditioned counterfactual value
model over executable terminal/headway actions or explicitly scope Phase-4
terminal dispatch outside the current paper.

2026-06-12 discrete-action context follow-up: following the SUMO-RL lesson that
continuous holding actions and too-narrow state can keep online learning below
fixed policy, the headway value planner was extended with explicit discrete
candidate-delta basis features (`-20/-10/0/+5/+10s`) and interactions with
low-frequency forecast, HF/MF energy, OD entropy, fleet pressure, and previous
wait/CV/overshoot diagnostics. The 4-domain x 20-seed x 100ep run
`headwayplanner_discctx_ep100_wu10` completed `80/80` rows:

```text
discctx vs main   overall +0.0106 CI [-0.0383,+0.0519]
discctx vs fixed  overall +0.0213 CI [-0.0250,+0.0626]
```

It did improve highnoise (`vs main -0.0489`, CI crosses zero), but it regressed
OD shift and rushshift; versus fixed-headway, OD shift and rushshift were
significantly worse. Mechanistically, the discrete-context planner stayed
active almost every decision, compressed the executable terminal launch shift
to roughly `2-5s`, and raised upper HF power. This is useful evidence that
discrete action representation helps high-frequency noise, but it is not a
promotion candidate.

A high-frequency activation gate was then tested to restrict the discrete
planner to high-HF regimes:

```text
discctx_hfgate vs main   overall +0.0354 CI [-0.0082,+0.0767]
discctx_hfgate vs fixed  overall +0.0461 CI [+0.0026,+0.0858]
```

This is a clear negative result. The gate reduced planner activation to
`0-2%`, restored larger terminal launch shifts, but did not preserve the
highnoise gain and remained significantly worse than fixed-headway overall.
This closes the current action-basis/gate branch. A publishable fixed-headway
repair now needs a stronger counterfactual value model over executable
terminal/headway actions, or the paper should explicitly frame fixed-headway as
a strong matched baseline that the method ties rather than universally beats.

2026-06-12 deterministic fixed-fallback follow-up: an offline feasibility check
showed that mixing the best learned branch (`odgate`) with fixed-headway can
beat either alone if the selector can identify OD/rush regimes. Implemented a
default-off deterministic fixed expert rule inside `fixed_expert_selector`,
then tested `odgate_fixedrule`: use odgate only when the demand profile has no
OD-profile shift prior and no rush-shift prior, otherwise fall back to the
fixed-headway expert. The 4-domain x 20-seed x 100ep run completed `80/80`
rows:

```text
odgate_fixedrule vs main   overall -0.0025 CI [-0.0516,+0.0490]
odgate_fixedrule vs fixed  overall +0.0082 CI [-0.0455,+0.0718]
```

The rule behaved as intended (`fixed_active=0` for terminal/highnoise and
`fixed_active=1` for OD/rush), but it did not reproduce the offline oracle:
terminal lost odgate's gain, highnoise was only weakly better, and OD fallback
did not match the external fixed baseline. This is not a promotion candidate.
It is useful as evidence that simple expert switching is insufficient under
online training dynamics. The next real repair must estimate counterfactual
costs for executable terminal/headway choices under matched demand seeds, not
infer them from same-run aggregates.

2026-06-13 counterfactual value audit v1: added
`scripts/build_freqduet_counterfactual_value_dataset.py`, which aligns completed
100ep rollouts by `(domain, seed)` and builds real matched-seed labels for
main, fixed-headway, rule baselines, `odgate`, `safe`, `discctx`,
`discctx_hfgate`, and `odgate_fixedrule`. Outputs are under
`results_freqduet/counterfactual_value_ep100/current/`.

The audit confirms that the repair target is real: the per-seed oracle over
executable candidates has overall composite `1.3212`, much lower than main
`1.5000`, fixed-headway `1.4893`, and the best single learned candidate
`odgate` `1.4813`. Best choices are not a trivial one-method rule; every domain
has mixed winners across `odgate`, `discctx`, `discctx_hfgate`,
`odgate_fixedrule`, fixed, main, and safe.

```text
ridge diagnostic selector vs main  overall +0.0134 CI [-0.0270,+0.0578]
ridge diagnostic selector vs fixed overall +0.0241 CI [-0.0167,+0.0679]

domain_mean selector vs main       overall -0.0244 CI [-0.0604,+0.0115]
domain_mean selector vs fixed      overall -0.0137 CI [-0.0480,+0.0192]
```

Interpretation: naive ridge over current run-level diagnostics is not enough;
it still selects poor candidates on OD/rush. A cross-validated
domain-conditioned empirical selector does move in the right direction and
beats main/fixed in mean, but the CI still crosses zero. This is a useful
positive signal, not a promotion result. The next implementation step should
be a true multi-candidate value selector with counterfactual candidate costs,
or a stricter action-level common-random-number rollout dataset. A binary
learned-vs-fixed wrapper cannot express the observed oracle because the best
choice is sometimes `odgate`, sometimes fixed, sometimes `odgate_fixedrule`,
and sometimes other planner variants.

Follow-up implementation: added a reproducible `cfvalue_domainmean` config
family and manifest entry. It is a first executable approximation of the
counterfactual selector rather than the final value model:

- terminal/highnoise: use `headwayplanner_odgate`;
- OD shift: use fixed-headway expert from episode 0;
- rush shift: use `headwayplanner_odgate_fixedrule`.

Local 4-config x 1-seed x 2ep smoke passed and confirmed method inference in
the comparison scripts. The first formal submit (`t11243`, `t11244`, `t11245`,
`t11246`, `t11248`, `t11250`) failed fast because the new `cfvalue_domainmean`
YAMLs had not yet been synced to the remote scheduler worktree. After syncing
the configs/scripts and verifying Python/import/config smoke on `node001` to
`node006`, those stale scheduler escalations were marked resolved.

The active 4-domain x 20-seed x 100ep formal matrix then ran as
scheduler-direct CPU shards on `node001-node006`:

```text
t11262 node006 shard_0000_0014
t11263 node002 shard_0014_0028
t11264 node001 shard_0028_0042
t11265 node003 shard_0042_0056
t11266 node005 shard_0056_0070
t11267 node004 shard_0070_0080
```

2026-06-13 result: `cfvalue_domainmean_ep100_wu10_r2` completed and synced
`80/80` rows, but it is not a promotion candidate:

```text
cfvalue_domainmean vs current main overall +0.0134 CI [-0.0172,+0.0454]
cfvalue_domainmean vs fixed-headway overall +0.0240 CI [-0.0078,+0.0533]
```

It only weakly helped highnoise; terminal, OD shift, and rush shift regressed.
The OD/rush diagnosis showed that the internal fixed fallback was not identical
to the external `fixed_headway` executor: it used the trip's existing
`target_headway` and shared the global environment RNG with runner/training
initialization. This broke the intended common-random-number comparison and made
fixed fallback worse than the external fixed baseline.

Implementation follow-up: added a config-gated
`fixed_expert_selector.strict_headway_s` and
`fixed_expert_selector.reset_env_rng`, plus an independent per-run fleet RNG in
`runner_v3.py`. A local OD/rush smoke confirmed the N-fleet sequence now aligns
with the external fixed baseline. Added `cfvalue_noisegate`, a stricter
counterfactual-value selector:

- terminal: exact fixed-headway;
- highnoise: `headwayplanner_odgate`;
- OD shift: exact fixed-headway;
- rush shift: exact fixed-headway.

Local 4-config x 1-seed x 2ep smoke passed and method inference recognizes
`cfvalue_noisegate`. The active formal matrix is:

```text
run: cfvalue_noisegate_ep100_wu10
t11331 node006 shard_0000_0014
t11332 node002 shard_0014_0028
t11335 node001 shard_0028_0042
t11336 node003 shard_0042_0056
t11338 node005 shard_0056_0070
t11339 node004 shard_0070_0080
```

2026-06-13 result: `cfvalue_noisegate_ep100_wu10` completed on
`node001-node006`, synced `80/80` diagnostics, and was aggregated under
`results_freqduet/cfvalue_noisegate_ep100_wu10/combined_summary`.

```text
cfvalue_noisegate summary, composite mean
terminal   1.4322
highnoise  1.7870
odshift    1.4445
rushshift  1.2699

cfvalue_noisegate vs current main, candidate - main
terminal   +0.0005 CI [-0.0500,+0.0457]
highnoise  -0.0372 CI [-0.0998,+0.0261]
odshift    -0.0066 CI [-0.0524,+0.0389]
rushshift  -0.0230 CI [-0.0663,+0.0203]
overall    -0.0166 CI [-0.0542,+0.0191]

cfvalue_noisegate vs fixed-headway, candidate - fixed
terminal   +0.0044 CI [-0.0151,+0.0245]
highnoise  -0.0399 CI [-0.0722,-0.0049]
odshift    +0.0149 CI [-0.0038,+0.0323]
rushshift  -0.0030 CI [-0.0129,+0.0069]
overall    -0.0059 CI [-0.0186,+0.0073]
```

Interpretation: this is a useful positive result and closes the immediate
fixed-headway gap from "significantly behind" to "statistically tied overall",
with a significant highnoise win against fixed-headway. It still should not be
written as universal dominance over fixed-headway because terminal and OD shift
are only tied/weakly worse and the overall CI crosses zero. The paper claim can
now say that the demand-conditioned selector preserves fixed-headway-level
robustness while improving the noisy-demand regime; a stronger top-journal claim
still needs a true multi-candidate counterfactual value model or Phase-4
terminal/first-stop dispatch validation.

2026-06-13 multi-candidate counterfactual follow-up: tested
`cfvalue_multicand`, a stricter executable approximation of the offline
multi-candidate evidence that uses `headwayplanner_odgate` for
terminal/highnoise and exact fixed-headway for OD/rush. It completed as
scheduler-visible direct-node tasks on `node001-node006`:

```text
run: cfvalue_multicand_ep100_wu10
t11369 node006 shard_0000_0014
t11370 node002 shard_0014_0028
t11371 node001 shard_0028_0042
t11372 node003 shard_0042_0056
t11373 node005 shard_0056_0070
t11374 node004 shard_0070_0080
```

The run synced and aggregated `80/80` diagnostics under
`results_freqduet/cfvalue_multicand_ep100_wu10/combined_summary`.

```text
cfvalue_multicand summary, composite mean
terminal   1.4296
highnoise  1.8199
odshift    1.4414
rushshift  1.2686

cfvalue_multicand vs current main, candidate - main
overall    -0.0101 CI [-0.0469,+0.0249]

cfvalue_multicand vs fixed-headway, candidate - fixed
overall    +0.0005 CI [-0.0119,+0.0135]

cfvalue_multicand vs cfvalue_noisegate, candidate - noisegate
terminal   -0.0026 CI [-0.0233,+0.0170]
highnoise  +0.0329 CI [-0.0003,+0.0649]
odshift    -0.0032 CI [-0.0204,+0.0141]
rushshift  -0.0014 CI [-0.0087,+0.0060]
overall    +0.0064 CI [-0.0026,+0.0155]
```

Decision: do not promote `cfvalue_multicand`. It ties current main and
fixed-headway, but it is worse than `cfvalue_noisegate`, mainly because the
highnoise odgate branch regressed on this rerun even though its YAML differs
from `cfvalue_noisegate` only by `_name` / comments. This confirms that
run-level candidate switching is too noisy to justify a stronger claim without
common-random-number action labels or a genuinely learned value model.

Tooling follow-up: added
`scripts/export_freqduet_counterfactual_selector_configs.py`. Given
`counterfactual_candidate_rows.csv`, it exports a selector audit and optional
executable configs. On the current audit:

```text
conservative_vs_fixed selector
terminal   fixed
highnoise  odgate   delta vs fixed -0.1103 CI [-0.2068,-0.0138]
odshift    fixed
rushshift  fixed

domain_mean exploratory selector
terminal   odgate            delta vs fixed -0.0198 CI [-0.0887,+0.0598]
highnoise  odgate            delta vs fixed -0.1103 CI [-0.1968,-0.0140]
odshift    fixed
rushshift  odgate_fixedrule  delta vs fixed -0.0238 CI [-0.0866,+0.0458]
```

Interpretation: the conservative bootstrap selector reconstructs
`cfvalue_noisegate` from matched rollout labels, so `noisegate` is now
documented as a data-driven conservative selector rather than a hand-tuned rule.
The exploratory selector still has terminal/rush CIs crossing zero, so the
remaining top-journal gap is not another domain-level router. It requires either
action-level common-random-number counterfactual labels for terminal/headway
choices, or a Phase-4 first-stop / actual terminal dispatch implementation.

2026-06-13 action-level CRN follow-up: implemented a config-gated
`upper.action_override` in `runner_v3.py`, generated the `cfaction_v1` matrix
with `scripts/build_freqduet_action_counterfactual_configs.py`, and ran
`cfaction_crn_screen_ep60_wu10` as 24 configs x 20 seeds x 60 episodes. The
matrix covers four domains, two execution modes (`target`, `terminalhold45`),
and three fixed upper deltas (`-20/0/+20s`). It completed as scheduler tasks
`t11376-t11381` on node001-node006, with 480/480 diagnostics aggregated.

The paired action analysis is in
`results_freqduet/cfaction_crn_screen_ep60_wu10/action_counterfactual_analysis`
from `scripts/summarize_freqduet_action_counterfactual.py`. Key composite
deltas are:

```text
candidate - same-mode delta0, negative is better
overall target -20s          +0.0063 CI [-0.0094,+0.0238]
overall target +20s          +0.0257 CI [+0.0094,+0.0421]
overall terminalhold45 -20s  -0.0097 CI [-0.0280,+0.0079]
overall terminalhold45 +20s  -0.0106 CI [-0.0262,+0.0055]

terminalhold45 - target, negative is better
overall delta0               +0.0196 CI [+0.0034,+0.0360]
overall +20s                 -0.0167 CI [-0.0313,-0.0014]

best fixed action by composite
terminal   target/delta0
highnoise  target/delta0
odshift    terminalhold45/+20s, delta vs target0 -0.0065 CI [-0.0281,+0.0132]
rushshift  target/+20s,        delta vs target0 -0.0044 CI [-0.0154,+0.0062]
overall    target/delta0
```

Decision: do not promote fixed action overrides. The action mechanism is wired
correctly (`upper_delta_mean` tracks `-19/0/+19s`, and `+20s` produces about
`43.9s` terminal launch shift under terminal execution), but no fixed
headway/terminal delta robustly improves the overall objective. This closes the
simple fixed-action CRN branch and strengthens the next-step requirement: a
real counterfactual value model must be context conditioned at the
decision/action level, or the work should move to the full Phase-4 terminal /
first-stop dispatch implementation rather than another fixed-delta sweep.

Action-conditioned value-model audit: using the same CRN labels, added
`scripts/fit_freqduet_action_value_model.py` to fit a single offline
`V(context, action)` ridge model over action features (`delta`, sign,
terminalhold45 flag), domain features, current-main diagnostic context, and
context-action interactions. This is stronger than the earlier per-candidate
ridge audit because candidate effects share parameters through explicit action
features. Seed-held-out CV still does not beat the baseline action:

```text
overall selected composite, lower is better
global_action_mean / target0      1.4979
action_domain                     1.5037, delta vs target0 +0.0058 CI [-0.0007,+0.0126]
context_action                    1.5037, delta vs target0 +0.0058 CI [-0.0010,+0.0128]
context_action_interact           1.5001, delta vs target0 +0.0022 CI [-0.0096,+0.0134]
oracle best fixed action          1.4521
```

Interpretation: there is real oracle headroom inside the fixed-action candidate
set, but aggregate episode/context diagnostics are not enough to select the
right action on held-out seeds. The missing unit is per-decision
counterfactual value, not another episode-level router. This makes the next
credible implementation either (i) a true per-dispatch/first-stop
counterfactual label mechanism, or (ii) the full Phase-4 actual terminal
dispatch path.

2026-06-13 trip-level value follow-up: synced all `480/480` `trip_details.csv`
files from the same action CRN run and added
`scripts/build_freqduet_trip_counterfactual_value_dataset.py`. The script
matches rows by `(domain, seed, episode, trip id)` across the six action
candidates and builds a trip-level action-value table. This is closer to the
needed decision unit than episode summaries, but it is still based on realized
rollout traces rather than exact simulator snapshot replay.

For `gap_dev` over the last 30 episodes, the matched trip-level audit produced
`41920` aligned trip contexts:

```text
selected trip gap cost, lower is better
global_action_mean / target0      0.1606
context_action                    0.1615, delta vs target0 +0.0009 CI [-0.0019,+0.0036]
context_action_interact           0.1696, delta vs target0 +0.0091 CI [+0.0026,+0.0158]
oracle best trip action           0.0378
```

Interpretation: the oracle gap is much larger at trip level than at episode
level, but the old trip log features (`hour`, `trip id`, direction, period,
headway, realized holding moments) are too shallow to learn a robust selector.
`runner_v3.py` now logs richer per-dispatch context in `trip_details.csv`:
upper-decision flags, promotion replan, effective launch shift, terminal gap
pressure, fleet pressure, waiting total, and harmonic LF/HF/promotion context.
The next experiment should rerun the action CRN matrix with these richer logs,
then refit the trip-level value selector before touching the online policy.

2026-06-13 rich-context CRN result: generated `cfaction_v2` configs with the
same four domains, two execution modes, and three fixed deltas, but with
`training.trip_dump_freq: 1` so every BiLevel episode dumps the enriched
`trip_details.csv`. Synced the updated runner/scripts/configs to the isolated
remote FreqDuet worktree and submitted
`cfaction_v2_tripctx_ep60_wu10` as scheduler-visible direct-node CPU tasks:

```text
t11390 shard_0000_0080 done node006
t11391 shard_0080_0160 done node001
t11392 shard_0160_0240 done node002
t11393 shard_0240_0320 done node003
t11394 shard_0320_0400 done node005
t11395 shard_0400_0480 done node004
```

The run synced and aggregated `480/480` diagnostics. Action-level paired
summary is under
`results_freqduet/cfaction_v2_tripctx_ep60_wu10/action_counterfactual_analysis`.
The overall best fixed action is `terminalhold45/delta0`, but its improvement
over `target0` is not significant:

```text
overall terminalhold45/delta0 - target0  -0.0054 CI [-0.0207,+0.0101]
odshift target/+20s - target0             -0.0272 CI [-0.0516,-0.0017]
rushshift terminalhold45/-20s - target0   -0.0106 CI [-0.0205,-0.0013]
```

The dense trip-level value audit used `480` `trip_details.csv` files and
`628800` matched trip contexts (`3772800` long action rows). The script was
optimized during this run to use vectorized trip selection, seed-level
bootstrap, vectorized domain mean prediction, and summary-only exports for
dense logs.

```text
trip gap cost, lower is better
target0/global baseline          0.1098
action_domain selected           0.1093, delta -0.0005 CI [-0.0021,+0.0009]
context_action selected          0.1093, delta -0.0005 CI [-0.0020,+0.0008]
context_action_interact selected 0.1172, delta +0.0074 CI [+0.0047,+0.0102]
oracle best trip action          0.0259
```

Decision: do not promote the fixed-action value selector. Rich per-dispatch
context confirms large trip-level oracle headroom, but it still cannot produce
a seed-held-out selector with robust improvement over `target0`; adding
context-action interactions overfits and significantly worsens gap cost. This
closes the shallow fixed-delta/value branch as diagnostic evidence. The next
credible repair is not another fixed action sweep; it is either simulator
snapshot / replay counterfactual labels at the dispatch decision, or a Phase-4
learned terminal / first-stop dispatch value layer.

2026-06-14 snapshot-replay counterfactual value follow-up: added true
per-dispatch snapshot audit tooling:

- `scripts/audit_freqduet_snapshot_counterfactual.py`
- `scripts/run_freqduet_snapshot_counterfactual_matrix.py`
- `scripts/submit_freqduet_snapshot_cf_scheduleurm.py`
- `scripts/fit_freqduet_snapshot_value_model.py`

The first snapshot matrix (`snapshot_cf_v1_4domain_20seed_snap20_h600`) is now
marked invalid for evidence because candidate replays did not restore common
random numbers. Its apparent oracle gap should not be used in the paper. The
audit script now captures/restores NumPy, Python, and Torch RNG states around
every candidate replay so candidate actions are evaluated under matched
passenger-arrival randomness and the audit does not perturb the live trajectory.

The CRN ep0/toolchain run
`snapshot_cf_v2_crn_earlyterm_4domain_20seed_snap20_h1800` showed the expected
pattern: there is real terminal-action oracle headroom, but fixed early/late
launch rules are unsafe on average. A residual seed-held-out random-forest
selector produced only a small offline gain and was treated as tooling evidence,
not a trained-policy result, because snapshots came from the initial/random
policy state.

The trained-state pilot
`snapshot_cf_v3_burn40_earlyterm_4domain_10seed_snap12_h1800` samples after 40
burn-in episodes, uses five terminal candidates
`term45_m60`, `term45_m30`, `term45_0`, `term45_p30`, and `term45_p60`, and
replays each candidate for a 1800s horizon from identical simulator snapshots.
It produced `2400` label rows (`4` domains x `10` seeds x `12` snapshots x `5`
candidates). Key proxy-cost deltas versus `term45_0` are:

```text
oracle best overall        -0.0478 CI [-0.0569,-0.0374]
RF residual selector       -0.0085 CI [-0.0152,-0.0021]
ExtraTrees residual        -0.0044 CI [-0.0093,+0.0003]
HistGB residual            -0.0033 CI [-0.0128,+0.0068]
```

Per-domain RF residual selector deltas:

```text
terminal   +0.0019 CI [-0.0063,+0.0106]
highnoise  -0.0166 CI [-0.0236,-0.0092]
odshift    -0.0187 CI [-0.0380,-0.0041]
rushshift  -0.0006 CI [-0.0125,+0.0107]
```

Decision: this is the first positive trained-state counterfactual-value signal,
but it is still an offline pilot, not an online-policy result. Do not promote it
into the main runner yet. The next credible step is to scale the burn-in
snapshot matrix to the full 20 paired seeds or switch to checkpoint-based
snapshot sampling to avoid retraining burn-in episodes for every audit job, then
wire the residual value selector into an online evaluation wrapper and compare
against current main and fixed-headway under paired 100/200ep validation.

2026-06-14 full 20-seed trained-state snapshot result: submitted
`snapshot_cf_v4_burn40_earlyterm_4domain_20seed_snap12_h1800` as
scheduler-visible direct-node tasks `t11426-t11431` on `node001-node006`.
The run completed with `80/80` audit jobs and produced `4800` label rows
(`4` domains x `20` seeds x `12` snapshots x `5` candidates). All labels have
`ep=40`, confirming post-burn-in sampling.

Candidate/oracle deltas versus `term45_0`:

```text
oracle best overall  -0.0495 CI [-0.0573,-0.0431]
term45_m30 overall   +0.0494 CI [+0.0072,+0.0990]
term45_m60 overall   +0.0573 CI [+0.0131,+0.1083]
term45_p30 overall   +0.0034 CI [-0.0097,+0.0145]
term45_p60 overall   +0.0016 CI [-0.0121,+0.0132]
```

Seed-held-out residual selector results:

```text
RF residual selector       -0.0164 CI [-0.0254,-0.0091]
ExtraTrees residual        -0.0158 CI [-0.0248,-0.0084]
HistGB residual            -0.0144 CI [-0.0238,-0.0067]
linear context interaction -0.0080 CI [-0.0193,+0.0015]
```

Per-domain RF residual selector deltas:

```text
terminal   -0.0085 CI [-0.0144,-0.0028]
highnoise  -0.0113 CI [-0.0207,-0.0020]
odshift    -0.0120 CI [-0.0246,-0.0012]
rushshift  -0.0339 CI [-0.0645,-0.0111]
overall    -0.0164 CI [-0.0254,-0.0091]
```

Decision update: the 10-seed pilot was not luck. The per-dispatch
counterfactual value signal is now stable enough to justify an online guarded
evaluation wrapper. It is still not promoted into paper-main until an executable
online run shows paired improvement versus current main and fixed-headway under
the normal 100/200ep evaluation protocol.

2026-06-14 online wrapper result: the executable `snapshotRF` wrapper was
implemented with a pure-NumPy random-forest artifact fallback so it runs in the
isolated CPU conda env on `node001-node006` without installing sklearn. The
first online matrix,
`snapshotrf_online_ep100_wu10_4domain_20seed`, ran `8` configs x `20` seeds x
`100` episodes as scheduler-visible direct-node tasks `t11437-t11453`
(`16/16` done, `0` failed). It compared current main and online `snapshotRF`
under the same seeds.

```text
snapshotRF m00 vs current main, composite delta candidate - main
terminal   -0.0024 CI [-0.0213,+0.0171]
highnoise  +0.0166 CI [-0.0172,+0.0545]
odshift    +0.0034 CI [-0.0167,+0.0242]
rushshift  +0.0035 CI [-0.0059,+0.0124]
overall    +0.0053 CI [-0.0049,+0.0152]

snapshotRF m00 vs fixed-headway, composite delta candidate - fixed
terminal        -0.0029 CI [-0.0229,+0.0168]
highnoise       -0.0380 CI [-0.0743,+0.0006]
odshift         +0.0050 CI [-0.0136,+0.0236]
rushshift       +0.0060 CI [-0.0049,+0.0172]
overall_shared  -0.0075 CI [-0.0204,+0.0047]

current main vs fixed-headway in the same run
overall_shared  -0.0128 CI [-0.0240,-0.0014]
```

Interpretation: the current main now beats fixed-headway overall in this clean
100ep paired run, while the online `snapshotRF` wrapper weakens that result and
does not improve current main. Therefore `snapshotRF m00` is not promoted.

2026-06-14 stricter guard check: added episode diagnostics
`snapshot_value_*` and ran
`snapshotrf_m02_online_ep100_wu10_4domain_20seed`, using the same RF model but
requiring `improve_margin >= 0.02`. This matrix ran `4` candidate configs x
`20` seeds x `100` episodes as tasks `t11499-t11506` (`8/8` done, `0` failed).

```text
snapshotRF m02 vs current main, composite delta candidate - main
terminal   -0.0084 CI [-0.0240,+0.0088]
highnoise  +0.0151 CI [-0.0122,+0.0428]
odshift    +0.0091 CI [-0.0072,+0.0258]
rushshift  -0.0040 CI [-0.0140,+0.0053]
overall    +0.0029 CI [-0.0059,+0.0121]

snapshotRF m02 vs snapshotRF m00
terminal   -0.0061 CI [-0.0222,+0.0088]
highnoise  -0.0015 CI [-0.0416,+0.0380]
odshift    +0.0057 CI [-0.0102,+0.0198]
rushshift  -0.0075 CI [-0.0179,+0.0026]
overall    -0.0023 CI [-0.0119,+0.0071]

snapshotRF m02 vs fixed-headway
overall_shared  -0.0098 CI [-0.0218,+0.0026]
```

The stricter guard improves m00 slightly but still does not beat current main.
The m02 diagnostics show the selector is evaluated on about `0.336-0.342` of
trips and still changes about `0.129-0.158` of trips, with mean predicted margin
`0.012-0.025`. This is not a robust online repair. Keep the snapshot replay and
RF selector as mechanism/negative evidence. The next fixed-headway-gap work
should pivot to either Phase-4 terminal/first-stop dispatch or a stronger
counterfactual value design that trains from true online rollouts/checkpoints,
not this direct RF wrapper.

2026-06-16 Phase-4 terminal-bias repair: implemented a new
`snapshot_value_selector.apply_mode: terminal_bias` path in `runner_v3.py`.
Unlike the earlier online RF wrappers, this mode does not overwrite the actor's
low-frequency headway/timetable action. It maps only positive selected
counterfactual offsets into a nonnegative first-stop / terminal dispatch bias,
and leaves the actor action unchanged (`snapshot_value_override_mean = 0`).

The 40ep pilot
`snapshottermbias_pilot_ep40_wu10_4domain_20seed` compared current main,
`snapshottermbias` m00, and `snapshottermbias_m01` across the four held-out
domains. m00 was rejected because rushshift worsened significantly. m01 was
promising: terminal improved significantly and the fixed-headway gap collapsed
to statistical parity.

```text
40ep m01 vs current main, composite delta candidate - main
terminal   -0.0297 CI [-0.0552,-0.0048]
highnoise  -0.0217 CI [-0.1002,+0.0576]
odshift    -0.0184 CI [-0.0418,+0.0065]
rushshift  +0.0083 CI [-0.0053,+0.0215]
overall    -0.0154 CI [-0.0359,+0.0049]

40ep m01 vs fixed-headway
overall_shared  +0.0047 CI [-0.0452,+0.0551]
```

The decisive 100ep validation
`snapshottermbias_m01_ep100_wu10_4domain_20seed` ran `8` configs x `20` seeds x
`100` episodes as scheduler-visible direct-node tasks `t11689-t11720`
(`32/32` done, `0` failed). It used `last-k=50` and compared old main against
the m01 terminal-bias candidate under the same paired seeds.

```text
100ep m01 vs old main, composite delta candidate - main
terminal   -0.0161 CI [-0.0336,+0.0022]
highnoise  -0.0200 CI [-0.0618,+0.0206]
odshift    -0.0149 CI [-0.0285,-0.0011]
rushshift  +0.0023 CI [-0.0066,+0.0116]
overall    -0.0122 CI [-0.0241,-0.0001]

100ep old main vs fixed-headway
overall_shared  +0.0077 CI [-0.0084,+0.0245]

100ep m01 vs fixed-headway
terminal        -0.0038 CI [-0.0194,+0.0118]
highnoise       -0.0249 CI [-0.0626,+0.0141]
odshift         +0.0089 CI [-0.0076,+0.0253]
rushshift       +0.0020 CI [-0.0072,+0.0108]
overall_shared  -0.0045 CI [-0.0153,+0.0077]
```

Decision: promote the m01 terminal-bias selector into
`F_freqduet_terminal_main_hiro.yaml`. This closes the immediate fixed-headway
gap: new main is significantly better than old main overall at 100ep and is
statistically tied with fixed-headway while still strongly beating rule holding
and rule MPC. Remaining Phase-4 work is no longer "prove first-stop/terminal
value action can help"; it is to polish mechanism figures, rerun final current
name matrices after the alias change, and decide whether to extend beyond
bounded terminal bias into richer actual terminal launch / first-stop holding
policy learning.

2026-06-17 current-name final matrix follow-up: after promoting terminal-bias
into the main aliases, reran the current config names under
`final_matrix_current_terminalbias_ep100_wu10_4domain_20seed` (`4` domains x
`6` methods x `20` paired seeds x `100` episodes, `last-k=50`). All `480/480`
rows synced and aggregated. This confirms the paper table will not have an
ambiguous "main" row.

```text
current-name 100ep main vs internal baselines, composite delta main - baseline
overall vs nofreq      -0.0233 CI [-0.0764,+0.0102]
overall vs rawhistory  +0.0106 CI [-0.0034,+0.0245]
overall vs allfreq     -0.0050 CI [-0.0294,+0.0172]
overall vs nopromotion +0.0024 CI [-0.0114,+0.0148]
overall vs noleakage   -0.3251 CI [-0.4213,-0.2368]

current-name 100ep main vs external baselines, composite delta main - baseline
overall_shared vs fixed_headway +0.0041 CI [-0.0093,+0.0178]
overall_shared vs rule_holding  -0.5858 CI [-0.6120,-0.5594]
overall_shared vs rule_mpc      -2.0281 CI [-2.1370,-1.9175]
```

Interpretation: the promoted main is statistically tied with fixed-headway and
strongly better than weaker classical baselines. Internally, `noleakage` is
decisively bad and `nofreq` is weaker on average, but `rawhistory` and
`nopromotion` remain close enough that the method claim should be framed as
robust frequency-separated control with essential leakage/terminal-bias
mechanisms, not universal dominance over every internal variant. The main
selector is active in all four domains with nonzero terminal-bias events and
`snapshot_value_override_mean = 0`, so the result is aligned with the Phase-4
bounded terminal-dispatch design rather than an actor-action override.

The matching 200ep current-name matrix has been launched as
`final_matrix_current_terminalbias_ep200_wu10_4domain_20seed` with scheduler
direct-node tasks `t11752-t11767` on `node001-node006` (`480` jobs,
`last-k=100`). It remains open until sync, aggregation, paired CI, and external
fixed-headway comparison are complete.

2026-06-17 domain-conditioned selector alignment audit: after the 200ep
freeze100 closure, a config audit found that the promoted current-name
generalization aliases (`F_freqduet_gen_highnoise_main_hiro`,
`F_freqduet_gen_odshift_main_hiro`, and `F_freqduet_gen_rushshift_main_hiro`)
were inheriting the terminal root's
`upper.snapshot_value_selector.domain: terminal`. The validated m01
terminal-bias matrix used domain-specific selector one-hots for highnoise,
odshift, and rushshift, so the previous current-name final matrix did not fully
match the tested m01 protocol outside the terminal domain.

Fixed the alias layer by adding explicit domain overrides to all current-name
held-out configs, including the learned ablation variants:

```text
highnoise configs -> snapshot_value_selector.domain: highnoise
odshift configs   -> snapshot_value_selector.domain: odshift
rushshift configs -> snapshot_value_selector.domain: rushshift
```

Local and remote config-load smoke tests confirmed the resolved domains. The
corrected 24-config x 20-seed x 200ep current-name matrix has been launched as
`final_matrix_current_domainfix_ep200_wu10_4domain_20seed` with scheduler
direct-node CPU tasks `t11844-t11859` on `node001-node006`. This result should
replace the previous current-name table if it improves or materially changes
the paired conclusions. Do not push/promote the alias fix as a paper result
until the matrix is aggregated and compared against the external fixed-headway,
rule-holding, and rule-MPC baselines.

2026-06-18 result: the domain-conditioned current-name matrix completed all
`16/16` scheduler tasks (`t11844-t11859`). Because node-local result paths were
not mirrored back into the local workspace automatically, the lightweight
artifacts from `node001-node006` were manually consolidated on the HPC login
side without checkpoints. Aggregation found the expected `480/480`
`diagnostics.csv` files and wrote:

```text
results_freqduet/final_matrix_current_domainfix_ep200_wu10_4domain_20seed/combined_summary
results_freqduet/final_matrix_current_domainfix_ep200_wu10_4domain_20seed/paper_matrix_summary
results_freqduet/final_matrix_current_domainfix_ep200_wu10_4domain_20seed/external_fixed_comparison
```

The corrected current-name main is now statistically tied with the strong
fixed-headway baseline overall, and the previous significant gap is largely
removed. Composite delta is `main - fixed`, so positive is worse:

```text
terminal        +0.0102 CI [-0.0051,+0.0260]
highnoise       +0.0031 CI [-0.0238,+0.0285]
odshift         +0.0119 CI [-0.0015,+0.0262]
rushshift       +0.0089 CI [-0.0005,+0.0203]
overall_shared  +0.0085 CI [-0.0015,+0.0189]
```

Internal ablation conclusions are unchanged in direction. `noleakage` is
strongly worse in every domain, confirming the leakage guard is necessary.
`nofreq`, `allfreq`, `rawhistory`, and `nopromotion` remain close to main under
the 200ep protocol; therefore this result supports the conservative paper claim
that FreqDuet matches fixed-headway while improving over weaker external
baselines and demonstrating necessary frequency/no-leakage structure, not a
claim of robust universal superiority over fixed-headway.

2026-06-18 fixed-gap optimization follow-up: the only promising earlier release
branch was the minimal terminal early-release cap (`release5`), but the old
generalization configs inherited the terminal snapshot-selector domain. To
avoid repeating that alias bug, added current-domainfix variants that inherit
the corrected per-domain current main roots and only change
`upper.timetable_planner.terminal_shift_min_s` from `0s` to `-5s`:

```text
F_freqduet_terminal_main_release5_current_hiro.yaml
F_freqduet_gen_highnoise_main_release5_current_hiro.yaml
F_freqduet_gen_odshift_main_release5_current_hiro.yaml
F_freqduet_gen_rushshift_main_release5_current_hiro.yaml
```

The 4-domain x 20-seed x 200ep run completed as scheduler tasks
`t11873-t11878` on `node001-node006` and wrote:

```text
results_freqduet/release5_current_domainfix_ep200_wu10_4domain_20seed/combined_summary
results_freqduet/release5_current_domainfix_ep200_wu10_4domain_20seed/compare_release5_current_vs_main_domainfix_ep200
results_freqduet/release5_current_domainfix_ep200_wu10_4domain_20seed/compare_release5_current_vs_external_fixed_ep200
```

Composite paired delta is `release5_current - baseline`, so negative is better.
Against current main:

```text
terminal  +0.0029 CI [-0.0163,+0.0262]
highnoise -0.0118 CI [-0.0334,+0.0104]
odshift   -0.0154 CI [-0.0313,+0.0004]
rushshift +0.0042 CI [-0.0047,+0.0139]
overall   -0.0050 CI [-0.0158,+0.0087]
```

Against external fixed-headway:

```text
terminal        +0.0130 CI [-0.0053,+0.0353]
highnoise       -0.0087 CI [-0.0380,+0.0184]
odshift         -0.0035 CI [-0.0145,+0.0078]
rushshift       +0.0131 CI [+0.0033,+0.0237]
overall_shared  +0.0035 CI [-0.0080,+0.0161]
```

Interpretation: `release5_current` is a weak local wait-reduction signal, not a
promotion candidate. It improves wait overall versus fixed (`-0.0690`) and
slightly improves highnoise/odshift composite versus main, but the benefit is
offset by higher CV/overshoot and a significant rushshift loss to fixed-headway.
Mechanistically, release5 increases terminal launch-shift variability versus
main (`terminal_launch_shift_std` overall `+2.0934`, CI `[+0.7025,+3.5819]`),
which is exactly the failure mode a terminal early-release gate must control.

A no-rerun diagnostic domain gate that uses release5 only on highnoise+odshift
and current main on terminal+rush is still not statistically decisive:

```text
vs main          overall -0.0068 CI [-0.0142,+0.0013]
vs fixed-headway overall +0.0017 CI [-0.0090,+0.0124]
```

Decision: do not promote or push `release5_current` as paper main. Keep it as
negative/mechanism evidence. The next credible fixed-headway repair should be a
state-dependent terminal-release value/guard that explicitly penalizes launch
shift variance and rush-pattern overshoot, or a first-stop/terminal dispatch
value layer trained from matched counterfactual rollouts. More unconditional
release-cap sweeps are unlikely to close the gap.

2026-06-18 active follow-up: implemented the state-dependent conservative
terminal release guard as `releaseguard`. It reuses the existing causal
`terminal_early_release_adaptive` hook but adds previous-episode risk gates:

```text
min_prev_wait_min: 5.6
max_prev_overshoot_norm: 0.22
max_prev_headway_cv: 0.455
max_prev_terminal_shift_mean_s: 23.0
max_prev_terminal_shift_std_s: 11.0
max_peak_shift_abs: 0.0
base_min_s: 0.0
relaxed_min_s: -5.0
```

This is deliberately not another unconditional release-cap sweep: the default
path remains no early release, rush-shift scenarios are blocked by
`max_peak_shift_abs`, and high launch-shift variance shuts the release gate in
the next episode. Added current-domainfix configs:

```text
F_freqduet_terminal_main_releaseguard_hiro.yaml
F_freqduet_gen_highnoise_main_releaseguard_hiro.yaml
F_freqduet_gen_odshift_main_releaseguard_hiro.yaml
F_freqduet_gen_rushshift_main_releaseguard_hiro.yaml
```

Local compile/config-load checks passed, and a 3-episode highnoise smoke ran
without breaking runner execution. The full 4-domain x 20-seed x 200ep
validation is now running as scheduler tasks `t11906-t11911`:

```text
results_freqduet/releaseguard_ep200_wu10_4domain_20seed
```

Do not promote or push this branch until it is synced, aggregated, and compared
against both `final_matrix_current_domainfix_ep200_wu10_4domain_20seed` and
`external_baselines_ep200_wu10_4domain_20seed`.

2026-06-19 result: synced and aggregated the full `releaseguard` matrix
(`4` domains x `20` seeds x `200` episodes). `releaseguard` is a real risk
guard but not a paper-main improvement:

```text
vs current main, composite delta = releaseguard - main
terminal  -0.0127 CI [-0.0245,-0.0021]
highnoise +0.0111 CI [-0.0219,+0.0423]
odshift   -0.0075 CI [-0.0229,+0.0082]
rushshift +0.0045 CI [-0.0048,+0.0149]
overall   -0.0011 CI [-0.0106,+0.0083]

vs fixed-headway, composite delta = releaseguard - fixed
terminal  -0.0026 CI [-0.0157,+0.0090]
highnoise +0.0142 CI [-0.0190,+0.0479]
odshift   +0.0044 CI [-0.0113,+0.0219]
rushshift +0.0134 CI [+0.0033,+0.0250]
overall   +0.0074 CI [-0.0042,+0.0204]
```

Mechanism check: compared with unconditional `release5`, the guard sharply
reduces launch-shift variance (`terminal_launch_shift_std` terminal `-3.51s`,
rushshift `-2.31s`) and blocks rushshift early release completely, but the main
objective does not improve (`overall +0.0039` vs `release5`, CI crosses zero).
Compared with fixed-headway, wait is lower overall (`-0.028 min`) but CV
(`+0.0053`) and overshoot (`+0.0185`) increase, cancelling the wait gain.

Decision: keep `releaseguard` as negative/mechanism evidence, do not promote it
or push it as the new main. The next repair should stop tuning scalar release
caps and move to matched counterfactual terminal/first-stop value selection:
for each demand seed and state bucket, label candidate executable headway or
first-stop release actions by rollout cost, then deploy only actions with a
positive paired value margin and explicit CV/overshoot risk terms.

2026-06-19 follow-up: after decomposing the `releaseguard` result, the 200ep
fixed-headway gap is not a wait problem. Current main has slightly lower wait
than fixed overall, but worse CV and fleet overshoot. Rather than expanding
early release, added a smaller risk-screened terminal-bias variant
`snapshottermbias_cap15` that keeps the promoted snapshot value selector and
removes the larger `p30` terminal-bias candidates:

```text
allowed_methods:
  - actor_term45_0
  - actor_target_p15
  - actor_term45_p15
```

This tests whether capping first-stop/terminal bias at the smaller +15s action
keeps the m01 wait benefit while reducing CV/overshoot variance. Added current
domainfix configs:

```text
F_freqduet_terminal_main_snapshottermbias_cap15_hiro.yaml
F_freqduet_gen_highnoise_main_snapshottermbias_cap15_hiro.yaml
F_freqduet_gen_odshift_main_snapshottermbias_cap15_hiro.yaml
F_freqduet_gen_rushshift_main_snapshottermbias_cap15_hiro.yaml
```

Local and remote config-load checks passed, with selector domains resolving to
`terminal/highnoise/odshift/rushshift` respectively and `improve_margin=0.01`.
The 4-domain x 20-seed x 200ep validation is running as scheduler-visible
direct-node tasks `t11924-t11929`:

```text
results_freqduet/snapshottermbias_cap15_ep200_wu10_4domain_20seed
```

Decision rule: only promote or push if paired 200ep results improve current
main overall and do not worsen the fixed-headway comparison; otherwise keep it
as a documented negative cap experiment.

2026-06-19 result: synced and aggregated the full
`snapshottermbias_cap15` matrix (`80/80` runs). The smaller cap does reduce
terminal-bias magnitude and launch-shift variance, but it does not improve the
objective enough to promote:

```text
vs current main, composite delta = cap15 - main
terminal  -0.0116 CI [-0.0253,+0.0010]
highnoise +0.0220 CI [-0.0153,+0.0607]
odshift   -0.0081 CI [-0.0234,+0.0076]
rushshift -0.0043 CI [-0.0124,+0.0032]
overall   -0.0005 CI [-0.0110,+0.0105]

vs fixed-headway, composite delta = cap15 - fixed
terminal  -0.0014 CI [-0.0150,+0.0116]
highnoise +0.0251 CI [-0.0154,+0.0660]
odshift   +0.0039 CI [-0.0126,+0.0221]
rushshift +0.0045 CI [-0.0047,+0.0145]
overall   +0.0080 CI [-0.0055,+0.0222]
```

Mechanism check: cap15 cuts `snapshot_value_terminal_bias_mean` from roughly
`2.49s` to `1.90s` and `terminal_launch_shift_std` from roughly `10.26s` to
`8.71s`, but highnoise loses the wait benefit from larger value-selected
terminal bias. A no-rerun domain/prior gate that keeps current main in
highnoise and uses cap15 only in terminal/odshift/rushshift would improve
current main overall (`-0.0060`, CI `[-0.0116,-0.0007]`) while staying tied
with fixed-headway (`+0.0025`, CI `[-0.0063,+0.0112]`). The next validation is
therefore a causal snapshot candidate gate: retain `+30s` only when demand
noise is high enough to justify it, otherwise cap positive terminal-bias
candidates at `+15s`. Do not promote `snapshottermbias_cap15` itself.

2026-06-19 active follow-up: implemented the causal candidate gate as
`snapshotnoisegate`. The runner now supports
`upper.snapshot_value_selector.candidate_gate` with a default positive-offset
cap and an optional high-demand-noise cap. The validation candidate keeps the
main snapshot value model and full candidate set, but applies:

```text
default_max_positive_offset_s: 15.0
high_noise_min_demand_noise: 0.25
high_noise_max_positive_offset_s: 30.0
```

Thus highnoise (`demand_noise=0.30`) keeps the `+30s` value-selected
wait-relief path, while terminal/odshift/rushshift are capped at `+15s` to
reduce CV/overshoot and launch-shift variance. Local compile/config checks and
a 3-episode smoke passed; the smoke confirmed terminal max terminal-bias `15s`
and highnoise max `30s`. Added current-domainfix configs:

```text
F_freqduet_terminal_main_snapshotnoisegate_hiro.yaml
F_freqduet_gen_highnoise_main_snapshotnoisegate_hiro.yaml
F_freqduet_gen_odshift_main_snapshotnoisegate_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotnoisegate_hiro.yaml
```

Decision rule: promote only if the full 200ep paired matrix reproduces the
no-rerun hybrid signal: improvement over current main and no significant loss
to fixed-headway. Otherwise keep it as a documented negative candidate.

2026-06-19 result: the full `snapshotnoisegate` matrix completed (`80/80`) and
was not promoted. It successfully expressed the intended mechanism in smoke,
but the 200ep paired result did not improve the paper objective:

```text
vs current main, composite delta = snapshotnoisegate - main
terminal  -0.0091 CI [-0.0222,+0.0033]
highnoise +0.0131 CI [-0.0090,+0.0355]
odshift   -0.0014 CI [-0.0135,+0.0105]
rushshift -0.0004 CI [-0.0084,+0.0081]
overall   +0.0005 CI [-0.0075,+0.0087]

vs fixed-headway, composite delta = snapshotnoisegate - fixed
terminal  +0.0011 CI [-0.0077,+0.0107]
highnoise +0.0162 CI [-0.0085,+0.0430]
odshift   +0.0105 CI [-0.0026,+0.0252]
rushshift +0.0085 CI [-0.0024,+0.0207]
overall   +0.0091 CI [-0.0004,+0.0202]
```

Decision: do not promote or push `snapshotnoisegate`. The useful signal remains
the domain/prior split seen in the no-rerun hybrid: highnoise should keep the
current main candidate set, while terminal/odshift/rushshift should use the
smaller cap15 candidate set. Added `snapshotdomaincap` configs to validate that
as a reproducible 4-domain method rather than a post-hoc table splice.

2026-06-19 result: the full `snapshotdomaincap` matrix completed (`80/80`).
It improved current main directionally, especially odshift, but it is not
paper-main safe because rushshift becomes significantly worse than the strong
fixed-headway baseline:

```text
vs current main, composite delta = snapshotdomaincap - main
terminal  -0.0090 CI [-0.0197,+0.0013]
highnoise -0.0078 CI [-0.0368,+0.0226]
odshift   -0.0146 CI [-0.0281,-0.0011]
rushshift +0.0008 CI [-0.0059,+0.0074]
overall   -0.0077 CI [-0.0174,+0.0023]

vs fixed-headway, composite delta = snapshotdomaincap - fixed
terminal  +0.0012 CI [-0.0080,+0.0099]
highnoise -0.0048 CI [-0.0294,+0.0223]
odshift   -0.0027 CI [-0.0182,+0.0129]
rushshift +0.0097 CI [+0.0006,+0.0198]
overall   +0.0009 CI [-0.0083,+0.0110]
```

Decision: do not promote. The next targeted candidate is
`snapshotodtermcap`: cap terminal and odshift only, while keeping highnoise and
rushshift as current main. A no-rerun seed-level sanity check predicts
`-0.0059` overall vs current main (CI `[-0.0105,-0.0014]`) and no overall loss
to fixed-headway (`+0.0026`, CI `[-0.0060,+0.0110]`).

2026-06-19 result: the full `snapshotodtermcap` matrix completed (`80/80`).
It is a cleaner domain-prior version of the cap idea, but still does not meet
the promotion rule:

```text
vs current main, composite delta = snapshotodtermcap - main
terminal  -0.0014 CI [-0.0136,+0.0098]
highnoise -0.0008 CI [-0.0322,+0.0297]
odshift   -0.0060 CI [-0.0216,+0.0085]
rushshift +0.0020 CI [-0.0045,+0.0090]
overall   -0.0016 CI [-0.0108,+0.0076]

vs fixed-headway, composite delta = snapshotodtermcap - fixed
terminal  +0.0088 CI [+0.0001,+0.0178]
highnoise +0.0023 CI [-0.0236,+0.0271]
odshift   +0.0059 CI [-0.0127,+0.0250]
rushshift +0.0109 CI [+0.0039,+0.0187]
overall   +0.0070 CI [-0.0027,+0.0166]
```

Decision: do not promote. The targeted cap is not enough; it still loses to
fixed-headway in terminal/rushshift. The next fixed-headway-gap attempt should
move away from terminal-bias cap tuning and test the SUMO-RL-style repair:
smaller/discrete lower holding actions plus explicit previous/following vehicle
and time-phase state.

2026-06-19 result: the `spacectx_disc9` matrix completed (`80/80`). This is
the direct SUMO-RL-style candidate: keep the validated FreqDuet hierarchy, add
lower spatiotemporal context (previous/following headways, launch gaps, station
phase, route progress, time sin/cos, upstream/downstream queues), and replace
continuous lower holding with 9 bins `[0,3,6,9,12,15,20,30,45]`.

```text
vs current main, composite delta = spacectx_disc9 - main
terminal  +0.0142 CI [-0.0092,+0.0451]
highnoise -0.0052 CI [-0.0324,+0.0226]
odshift   -0.0098 CI [-0.0234,+0.0028]
rushshift +0.0044 CI [-0.0046,+0.0126]
overall   +0.0009 CI [-0.0095,+0.0129]

vs fixed-headway, composite delta = spacectx_disc9 - fixed
terminal  +0.0244 CI [+0.0009,+0.0609]
highnoise -0.0021 CI [-0.0284,+0.0225]
odshift   +0.0021 CI [-0.0126,+0.0176]
rushshift +0.0132 CI [+0.0038,+0.0230]
overall   +0.0094 CI [-0.0002,+0.0200]
```

Decision: do not promote. The state/action repair gives the right directional
signal in highnoise and odshift, but it hurts terminal/rushshift and makes the
fixed-headway gap domain-significant there. This rules out a global
spacectx-disc9 main. If reused, it needs a causal gate that enables the richer
lower context only in noisy/OD-shift regimes, not a universal replacement.

2026-06-19 follow-up: implemented and validated `spacectx_causalgate`, a
history-prior causal gate for the SUMO-RL-style lower context/action repair. The
first two attempts were cancelled during early sanity checks because an
episode-instantaneous gate opened too often in terminal seeds. The completed
version uses a delayed historical EMA gate (`min_episode=40`,
`min_history_episodes=30`) and highnoise evidence from
`freq_promotion_absorbed >= 0.12`; OD-shift still uses
`freq_od_entropy <= 0.9535`. This made the gate domain-specific in diagnostics:
last-100 gate means were terminal `0.079`, highnoise `1.000`, odshift `0.821`,
and rushshift `0.0005`.

```text
vs current main, composite delta = spacectx_causalgate - main
terminal  +0.0003 CI [-0.0152,+0.0153]
highnoise -0.0005 CI [-0.0268,+0.0237]
odshift   -0.0003 CI [-0.0163,+0.0155]
rushshift +0.0117 CI [-0.0007,+0.0235]
overall   +0.0028 CI [-0.0078,+0.0127]

vs fixed-headway, composite delta = spacectx_causalgate - fixed
terminal  +0.0105 CI [-0.0012,+0.0219]
highnoise +0.0026 CI [-0.0254,+0.0301]
odshift   +0.0116 CI [-0.0030,+0.0267]
rushshift +0.0206 CI [+0.0119,+0.0326]
overall   +0.0113 CI [+0.0030,+0.0197]
```

Decision: do not promote and stop tuning this lower-spacectx/discrete-action
branch globally. The causal gate fixed the specificity problem, but performance
still only ties current main and remains significantly worse than fixed-headway
overall, with a rushshift regression. The next credible fixed-headway-gap work
should return to counterfactual value/Phase-4 dispatch, especially mechanisms
that directly trade CV and fleet overshoot against wait, rather than expanding
lower local state again.

2026-06-19 active follow-up: implemented `snapshotriskpenalty`, a deploy-time
risk penalty around the rollout-trained snapshot value selector. The value
model and main candidate set are unchanged; only positive terminal-bias
candidates above `+15s` receive an additional score penalty when previous
episode or current dispatch risk is high. The risk terms are explicitly tied to
the fixed-headway gap decomposition: previous `headway_cv`, normalized
`fleet_overshoot`, `terminal_launch_shift_std`, current active headway CV, and
current fleet pressure. Default main behavior is unchanged unless
`upper.snapshot_value_selector.risk_penalty.enable` is set.

Added configs:

```text
F_freqduet_terminal_main_snapshotriskpenalty_hiro.yaml
F_freqduet_gen_highnoise_main_snapshotriskpenalty_hiro.yaml
F_freqduet_gen_odshift_main_snapshotriskpenalty_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotriskpenalty_hiro.yaml
```

Local compile/config checks passed, and a forced early-selector smoke
(`upper_warmup_eps=1`) confirmed the new diagnostics are written:
`snapshot_value_risk_score_mean`, `snapshot_value_risk_penalty_mean`, and
`snapshot_value_risk_penalty_max_mean`. The formal `4` domain x `20` seed x
`200` episode validation is running through scheduler-visible direct-node tasks
`t12022-t12027` on `node001-node006`:

```text
results_freqduet/snapshotriskpenalty_ep200_wu10_4domain_20seed
```

Decision rule: promote only if paired 200ep results improve or match current
main while closing the fixed-headway CV/overshoot gap without a significant
wait regression. If it simply reduces bias but loses wait, keep it as negative
evidence and move to retraining the counterfactual value model with explicit
CV/overshoot labels.

2026-06-20 result: the full `snapshotriskpenalty` matrix completed (`80/80`).
The deploy-time risk penalty is directionally useful, especially in highnoise,
but the global risk-penalty branch is not itself safe enough to promote because
rushshift remains significantly worse than fixed-headway:

```text
vs current main, composite delta = snapshotriskpenalty - main
terminal  -0.0075 CI [-0.0192,+0.0033]
highnoise -0.0204 CI [-0.0536,+0.0124]
odshift   -0.0002 CI [-0.0138,+0.0120]
rushshift +0.0035 CI [-0.0035,+0.0111]
overall   -0.0062 CI [-0.0157,+0.0039]

vs fixed-headway, composite delta = snapshotriskpenalty - fixed
terminal  +0.0026 CI [-0.0104,+0.0149]
highnoise -0.0173 CI [-0.0537,+0.0151]
odshift   +0.0117 CI [-0.0009,+0.0247]
rushshift +0.0124 CI [+0.0017,+0.0237]
overall   +0.0023 CI [-0.0086,+0.0137]
```

Mechanism check: the risk-penalty diagnostics were active and small
(`snapshot_value_risk_score_mean` around `0.33-0.36`; selected penalty near
`1e-4-3e-4`; max candidate penalty around `0.001`). It reduced highnoise wait
strongly, but the penalty is too weak or too global to fix rushshift. Decision:
do not promote `snapshotriskpenalty` alone.

The useful result is a domain-prior mix over completed 200ep matrices:

```text
terminal  -> snapshottermbias_cap15
highnoise -> snapshotriskpenalty
odshift   -> snapshotdomaincap
rushshift -> snapshottermbias_cap15
```

This is now codified as `snapshotriskmix` configs and summarized as
`snapshotriskmix_norerun_ep200_wu10_4domain_20seed`. Seed-level no-rerun
paired CIs, using the already completed runs with the alias config names, are:

```text
vs current main, composite delta = snapshotriskmix - main
terminal  -0.0116 CI [-0.0249,+0.0010]
highnoise -0.0204 CI [-0.0525,+0.0112]
odshift   -0.0146 CI [-0.0283,-0.0008]
rushshift -0.0043 CI [-0.0123,+0.0029]
overall   -0.0127 CI [-0.0228,-0.0026]

vs fixed-headway, composite delta = snapshotriskmix - fixed
terminal  -0.0014 CI [-0.0155,+0.0114]
highnoise -0.0173 CI [-0.0541,+0.0151]
odshift   -0.0027 CI [-0.0184,+0.0136]
rushshift +0.0045 CI [-0.0045,+0.0143]
overall   -0.0042 CI [-0.0159,+0.0078]
```

This is the first 200ep candidate in this branch that both improves current
main significantly and removes the significant fixed-headway loss in every
domain. The remaining caveat is that the mix was selected after seeing the
completed domain matrices. Before calling it final paper main, either rerun the
alias configs under the current method name or run a held-out-seed matrix with
current main and fixed-headway.

2026-06-20 validation update: the same-name `snapshotriskmix` alias rerun
completed as `snapshotriskmix_ep200_wu10_4domain_20seed` (`80/80`). It
preserves the positive direction but does not reproduce the post-hoc no-rerun
significance, and rushshift still has a borderline fixed-headway gap:

```text
vs current main, composite delta = snapshotriskmix - main
terminal  -0.0097 CI [-0.0238,+0.0031]
highnoise -0.0107 CI [-0.0362,+0.0148]
odshift   -0.0142 CI [-0.0287,+0.0002]
rushshift -0.0007 CI [-0.0074,+0.0060]
overall   -0.0088 CI [-0.0185,+0.0003]

vs fixed-headway, composite delta = snapshotriskmix - fixed
terminal  +0.0004 CI [-0.0108,+0.0118]
highnoise -0.0076 CI [-0.0394,+0.0256]
odshift   -0.0023 CI [-0.0179,+0.0139]
rushshift +0.0082 CI [+0.0000,+0.0171]
overall   -0.0003 CI [-0.0118,+0.0125]
```

Decision: `snapshotriskmix` remains useful evidence for domain-prior repair,
but it should not be promoted as final main from this rerun alone.

2026-06-20 rushshift zero-bias diagnostic: `snapshotzerobias` disables all
positive terminal-bias candidates in rushshift and keeps only the actor
fallback/zero-offset candidate. The 20-seed rush-only 200ep run completed under
`snapshotzerobias_rush_ep200_wu10_20seed`. It did not improve rushshift:

```text
vs current main, composite delta = snapshotzerobias - main
rushshift +0.0004 CI [-0.0074,+0.0071]

vs fixed-headway, composite delta = snapshotzerobias - fixed
rushshift +0.0093 CI [+0.0008,+0.0172]
```

Interpretation: the rush fixed-headway gap is not solved by removing positive
terminal bias globally. Fully zeroing bias increases overshoot, while cap15
still has the best rush mean among completed runs. The next repair is therefore
a causal hard gate, not a static zero-bias policy.

2026-06-20 active follow-up: added `snapshotrushriskgate`, a rush-only
candidate gate that keeps the cap15 candidate set in normal states but hard
caps positive terminal-bias offsets to `0s` after a high-risk previous episode
(`headway_cv > 0.49`, normalized `fleet_overshoot > 0.30`, or
`terminal_launch_shift_std > 14s`). This tests whether the useful cap15 branch
can be retained while preventing launch-shift variance from compounding.

```text
config:
F_freqduet_gen_rushshift_main_snapshotrushriskgate_hiro.yaml

run:
results_freqduet/snapshotrushriskgate_rush_ep200_wu10_20seed

scheduler tasks:
t12077-t12081 on node004/node001/node006/node002/node003
```

Promotion rule for this narrow repair: it must improve rushshift versus current
main and remove the significant rushshift loss to fixed-headway; otherwise keep
the hard gate as negative mechanism evidence and stop tuning static terminal
bias caps.

Parallel threshold sweep launched to avoid serially waiting on a single gate:

```text
configs:
F_freqduet_gen_rushshift_main_snapshotrushriskgate_loose0_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotrushriskgate_soft7_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotrushriskgate_tight7_hiro.yaml

run:
results_freqduet/snapshotrushriskgate_variants_rush_ep200_wu10_3x20seed

scheduler tasks:
t12082-t12091 on node001-node006 direct CPU scheduler
```

Interpretation rule: if none of the hard/soft risk gates beats cap15/current
main and removes the rush fixed-headway gap, this closes the static terminal
bias gate branch. The next credible fixed-headway-gap repair would need
matched counterfactual value labels with explicit CV/overshoot targets, not
more hand-tuned caps.

2026-06-20 result: both rush risk-gate runs completed. None satisfied the
promotion rule. The best variant, `snapshotrushriskgate` (`mid0`), weakly
improves rushshift versus current main but the effect is negligible and the CI
crosses zero. It also only turns the fixed-headway comparison into a statistical
tie; the mean remains worse than fixed-headway.

```text
rushshift composite, candidate delta; lower is better
variant  comp    vs main                 vs fixed-headway
mid0     1.3187  -0.0008 [-0.0067,+0.0057]  +0.0081 [-0.0013,+0.0182]
tight7   1.3202  +0.0007 [-0.0065,+0.0084]  +0.0096 [-0.0005,+0.0206]
soft7    1.3222  +0.0027 [-0.0037,+0.0091]  +0.0116 [+0.0012,+0.0232]
loose0   1.3234  +0.0039 [-0.0031,+0.0114]  +0.0128 [+0.0041,+0.0216]
```

Mechanism diagnosis: the gates lowered selected terminal-bias magnitude, but
wait/CV/overshoot did not jointly improve. This closes the static
terminal-bias cap / risk-gate branch as negative evidence. The next repair
should train or fit a value selector on matched counterfactual labels that
include explicit CV/overshoot/launch-shift terms, rather than continuing to
hand tune positive-bias thresholds.

Immediate next repair launched: the existing v6 actor-relative snapshot labels
show rushshift's integrated replay cost is lowest for target/headway actions
(`actor_target_0` and nearby target offsets), not terminal-hold bias. Added two
target-only action-override configs and launched them together with the existing
full `snapshotactorrel` action-override config.

2026-06-20 diagnostic correction: the first
`snapshotactorrel_action_rush_ep200_wu10_3x20seed` screen completed, but it was
invalid for the intended test because `snapshotactorrel_hiro` inherited
`apply_mode: terminal_bias` from the current main root. Diagnostics confirmed
`snapshot_value_override_mean = 0`, so no upper action override was actually
executed. The result is kept only as an inheritance/config diagnostic, not as
evidence against action override. The config now explicitly sets:

```yaml
upper:
  snapshot_value_selector:
    apply_mode: action_override
```

The corrected run is:

```text
configs:
F_freqduet_gen_rushshift_main_snapshotactorrel_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotactorrel_targetonly_hiro.yaml
F_freqduet_gen_rushshift_main_snapshotactorrel_targetonly_m02_hiro.yaml

run:
results_freqduet/snapshotactorrel_actionfix_rush_ep200_wu10_3x20seed

scheduler tasks:
t12151-t12160 on node001-node006 direct CPU scheduler
```

Decision rule: promote or fold into a new domain mix only if target/action
override improves rushshift versus current main and removes the fixed-headway
mean gap without creating a significant CV/overshoot regression. If it fails,
the next step is to regenerate matched snapshot labels with a stronger
CV/overshoot/launch-shift objective and train a new selector artifact.

2026-06-21 result: the corrected action-override screen completed and the
override path was active (`snapshot_value_override_mean > 0`,
`snapshot_value_terminal_bias_mean = 0`). It still failed the promotion rule:

```text
rushshift composite, candidate delta; lower is better
variant            comp    override_mean  vs main                 vs fixed-headway
full actor-rel     1.3383  0.1546         +0.0189 [+0.0021,+0.0426]  +0.0277 [+0.0126,+0.0485]
target-only        1.3213  0.3100         +0.0019 [-0.0027,+0.0067]  +0.0107 [+0.0014,+0.0222]
target-only m02    1.3221  0.0836         +0.0027 [-0.0035,+0.0092]  +0.0115 [+0.0021,+0.0218]
```

Interpretation: the full actor-relative action override over-corrects and is
significantly worse than current main. The target-only variants are statistically
tied with current main, but both remain significantly worse than fixed-headway.
This closes the direct reuse of the v6 snapshot actor-relative artifact as
negative evidence. The active next step is a fresh matched common-random-number
action-counterfactual label matrix with richer trip context, explicit terminal
dispatch candidates, and trip-level value fitting:

```text
run:
results_freqduet/cfaction_v2_ep100_wu10_4domain_20seed

matrix:
4 domains x 6 fixed actions x 20 paired seeds = 480 runs
episodes: 100
last_k: 50
trip_dump_freq: 1

scheduler tasks:
t12171-t12180 on node001-node006 direct CPU scheduler
```

This run should be used to fit and audit a stronger counterfactual value model.
If the seed-held-out value model still cannot beat `target0`/fixed-action
baselines, the fixed-delta action-value branch should be treated as diagnostic
only and the next real repair should move to simulator-snapshot replay labels or
Phase-4 first-stop/actual terminal dispatch.

2026-06-21 result: the `cfaction_v2_ep100_wu10_4domain_20seed` matrix completed
with `480/480` diagnostics and `480/480` `trip_details.csv` files. Episode-level
paired fixed-action analysis again shows direction but not a promotable fixed
action:

```text
best fixed action by episode composite, best - target0; lower is better
terminal   target -20s          -0.0091 CI [-0.0272,+0.0082]
highnoise  terminalhold45 -20s  -0.0180 CI [-0.0654,+0.0295]
odshift    target -20s          -0.0087 CI [-0.0294,+0.0134]
rushshift  terminalhold45 -20s  -0.0026 CI [-0.0132,+0.0084]
overall    terminalhold45 -20s  -0.0081 CI [-0.0200,+0.0041]
```

Trip-level value fitting used `1,048,000` aligned trip contexts and `6,288,000`
action rows over last-50 episodes. It exposes large oracle headroom but no
seed-held-out deployable selector:

```text
metric: gap_dev, baseline target0
oracle_best cost                 0.0238
target0 cost                     0.1041
action_domain selector delta     +0.0001 CI [-0.0008,+0.0010]
context_action selector delta    +0.0001 CI [-0.0008,+0.0011]
domain_action_mean selector      +0.0001 CI [-0.0008,+0.0011]
context_action_interact delta    +0.0086 CI [+0.0052,+0.0134]
```

Interpretation: richer trip context and 100ep labels confirm the oracle gap, but
the current linear/ridge action-value model cannot recover it; adding dense
context-action interactions overfits and significantly worsens. The fixed-delta
counterfactual branch should remain diagnostic evidence. The next credible
repair is either (i) a non-linear/causal selector audit that does not overfit
seed-held-out folds, or (ii) the larger Phase-4 move to simulator-snapshot
replay labels / actual terminal and first-stop dispatch value.

2026-06-21 nonlinear selector audit: added
`scripts/fit_freqduet_trip_oracle_classifier.py`, which changes the diagnostic
from row-wise linear cost regression to a seed-held-out context -> oracle-action
classification problem. This keeps the same last-50 CRN labels but asks whether
expanded state can select among the six discrete fixed actions. A local
`HistGradientBoostingClassifier` audit (`max_iter=60`) produced the first
positive fixed-action value signal:

```text
metric: gap_dev, classifier selected action - target0; lower is better
terminal   -0.0106 CI [-0.0124,-0.0087]
highnoise  -0.0098 CI [-0.0121,-0.0073]
odshift    -0.0092 CI [-0.0121,-0.0068]
rushshift  -0.0092 CI [-0.0118,-0.0063]
overall    -0.0097 CI [-0.0110,-0.0084]
```

The selected cost drops from `target0 = 0.1041` to `0.0944`, while the oracle
best remains `0.0238`, so the classifier captures a real but still partial
piece of the available trip-level headroom. This does not yet promote a runtime
policy: it is an offline seed-held-out diagnostic using realized trip context
and local sklearn. The next implementable repair is to export this selector
logic into a deployable, dependency-controlled runtime module or distill it into
a small table/tree, then run an online 4-domain validation against current main
and fixed-headway.

2026-06-21 deployable action-tree distillation: added
`upper/counterfactual_action_selector.py` and
`scripts/fit_freqduet_trip_action_tree_selector.py`. The new script fits a
shallow decision tree from the same matched trip-level labels, exports a JSON
artifact, and the runtime scorer loads that artifact without sklearn. The
seed-held-out offline tree (`max_depth=8`, `min_samples_leaf=1000`) retained a
large fraction of the HGB signal:

```text
metric: gap_dev, tree selected action - target0; lower is better
terminal   -0.0089 CI [-0.0108,-0.0070]
highnoise  -0.0072 CI [-0.0097,-0.0041]
odshift    -0.0096 CI [-0.0128,-0.0070]
rushshift  -0.0068 CI [-0.0095,-0.0039]
overall    -0.0081 CI [-0.0095,-0.0065]
```

Smoke test with `--upper-warmup-eps 0` confirmed the selector is active only at
upper replan events, writes trip-level diagnostics, and can switch the
executable action class. A 24-config online validation is now running on
node001-node006 via scheduler as
`cfactiontree_ep100_wu10_4domain_20seed` (`t12244`-`t12249`): per domain it
compares current main, action-tree, fixedselector_balanced, target0,
target_m20, and terminalhold45_m20. Do not promote until this online matrix is
synced, aggregated, and paired against both current main and the fixed-headway
wrapper.

2026-06-21 online action-tree result: `cfactiontree` is **not promoted**. It
slightly improves the current main on mean but not significantly:

```text
cfactiontree - main, composite; lower is better
terminal   -0.0088 CI [-0.0267,+0.0097]
highnoise  -0.0096 CI [-0.0622,+0.0451]
odshift    -0.0020 CI [-0.0262,+0.0213]
rushshift  +0.0011 CI [-0.0079,+0.0105]
overall    -0.0048 CI [-0.0228,+0.0129]
```

Mechanistically it mostly collapsed to `target_m20`: selected delta averaged
about `-18s`, terminal-dispatch was active only `4-6%`, and it did not beat the
simple fixed-action candidates. However, the same online matrix shows a
disturbance-conditioned simple rule has real headroom if it is validated without
post-hoc leakage: terminal -> target0, highnoise -> terminalhold45_m20,
odshift -> target_m20, rushshift -> target0 gives an already-runnable composite
of `1.4774` versus current main `1.4939`, delta `-0.0165` CI
`[-0.0312,-0.0009]`. Because the original generated target configs inherited
terminal-dispatch plumbing from main, new explicit rule configs were added with
true target-only branches where intended:

```text
F_freqduet_terminal_main_cfactionrule_v1_hiro        target0, terminal_dispatch=false
F_freqduet_gen_highnoise_main_cfactionrule_v1_hiro  terminalhold45_m20
F_freqduet_gen_odshift_main_cfactionrule_v1_hiro    target_m20, terminal_dispatch=false
F_freqduet_gen_rushshift_main_cfactionrule_v1_hiro  target0, terminal_dispatch=false
```

The confirmatory 4-config x 20-seed x 100ep matrix completed as
`cfactionrule_v1_ep100_wu10_4domain_20seed`. It is not promoted:

```text
cfactionrule_v1 - current main, composite; lower is better
overall -0.0023 CI [-0.0214,+0.0127]
terminal -0.0045 CI [-0.0230,+0.0123]
highnoise -0.0198 CI [-0.0625,+0.0239]
odshift +0.0097 CI [-0.0115,+0.0321]
rushshift +0.0054 CI [-0.0014,+0.0122]
```

This also corrected an earlier interpretation trap: the generated
`terminalhold45_*` action configs allow terminal shift, but under fixed
`action_override` they do not create a terminal feedback/value bias, so measured
`terminal_launch_shift_mean` is near zero. Treat them as discrete headway-action
counterfactuals, not as validated Phase-4 terminal/first-stop dispatch evidence.
The exp39-style state/action repair matrix
`exp39_state_action_ep100_wu10_4domain_20seed` completed. No single global
module is yet promoted, but the signal is much stronger than the value-selector
detours:

```text
candidate - same-run main, composite; lower is better
upperdisc4 overall -0.0092 CI [-0.0302,+0.0054]
upperdisc5 overall -0.0067 CI [-0.0220,+0.0105]
upperdisc3 overall -0.0055 CI [-0.0220,+0.0118]
spacectx_disc9 overall -0.0033 CI [-0.0172,+0.0093]
histdisc overall -0.0002 CI [-0.0148,+0.0193]

post-hoc domain-best:
terminal -> upperdisc5
highnoise -> upperdisc4
odshift -> upperdisc5
rushshift -> histdisc
overall -0.0168 CI [-0.0352,-0.0017]
```

The confirmatory rerun completed as
`exp39_domainbest_confirm_ep100_wu10_4domain_20seed`, scheduler tasks
`t12318`-`t12329`. It failed to reproduce the post-hoc gain and is not
promoted:

```text
domainbest_confirm - same-run main, composite; lower is better
terminal -0.0002 CI [-0.0175,+0.0169]
highnoise +0.0385 CI [-0.0102,+0.0888]
odshift +0.0101 CI [-0.0092,+0.0278]
rushshift +0.0054 CI [-0.0070,+0.0176]
overall +0.0135 CI [-0.0024,+0.0377]
```

Across the original exp39 run and confirm rerun, the selected domain-best
modules average only `-0.0017`, with highnoise flipping from helpful to harmful.
Conclusion: the current exp39-style configs are useful negative/diagnostic
evidence, but the fixed-headway gap is not solved by a simple domain-level
choice of upper action alphabet/history.

Follow-up correction: the first exp39 matrix tested upper discretization and
upper history mostly as separate configs. The `upperhist` matrix
(`upperhist_ep100_wu10_4domain_20seed`, scheduler tasks `t12359`-`t12381`)
tested the actual combined repair: preserve each domain's main config, then add
short upper state history plus near-fixed upper action bins (`upperhist3/4/5`).
It also failed to close the gap:

```text
candidate - same-run main, composite; lower is better
upperhist4 overall -0.0020 CI [-0.0136,+0.0130]
upperhist5 overall +0.0013 CI [-0.0132,+0.0149]
upperhist3 overall +0.0039 CI [-0.0104,+0.0230]
post-hoc domain-best overall -0.0046 CI [-0.0166,+0.0063]
```

Conclusion: the straightforward exp39-style repair path is not robust in this
FreqDuet setting. Future optimization should move to a stronger learned
rollout/value planner or a more structural Phase-4 dispatch layer, not more
manual domain-level action alphabet tuning.

2026-06-22 active follow-up: after `upperhist` closed the simple
exp39-style state/action route, the next repair moved back to
counterfactual Phase-4 value learning. Added a new snapshot replay label,
`risk_proxy_cost`, in `scripts/audit_freqduet_snapshot_counterfactual.py`.
The old `proxy_cost` / `integrated_proxy_cost` remains unchanged for historical
reproducibility; `risk_proxy_cost` adds explicit CV, fleet-overshoot,
terminal-delay, and positive-offset risk terms inside the matched replay label
instead of adding a hand-tuned deploy-time penalty after training.

Local and remote compile checks passed, and a one-snapshot smoke verified that
`risk_proxy_cost` is emitted and aggregate-able. The formal v7 trained-state
snapshot matrix finished as scheduler-visible direct-node CPU tasks:

```text
run:
results_freqduet/snapshot_cf_v7_riskproxy_burn40_actorrel_4domain_20seed_snap12_h1800

tasks:
t12401 node001
t12402 node002
t12403 node003
t12404 node004
t12405 node005
t12406 node006

matrix:
4 domains x 20 seeds
burn-in episodes: 40
snapshots per run: 12
horizon: 1800s
candidates: actor-relative target/terminalhold45 offsets [-30,-15,0,+15,+30]
baseline: actor_term45_0
metric: risk_proxy_cost
```

Promotion rule: only proceed to an online wrapper if the seed-held-out value
model trained on `risk_proxy_cost` improves the actor baseline with a
domain-safe CI and shows it is reducing CV/overshoot risk rather than merely
collapsing to a single fixed target action. If it fails, this becomes negative
evidence that the current first-stop/terminal value layer needs richer
state/replay or a fuller terminal dispatch formulation.

Status after aggregation/model fit:

```text
labels: 80/80 paired domain-seed labels aggregated
best online-candidate model: random_forest_context_action_interact
offline selected delta vs actor_term45_0: -0.0148
95% CI: [-0.0274, -0.0043]

per-domain selected deltas:
terminal  -0.0168  CI [-0.0305, -0.0048]
highnoise -0.0169  CI [-0.0355, -0.0031]
odshift   -0.0135  CI [-0.0250, -0.0038]
rushshift -0.0121  CI [-0.0232, -0.0030]
```

Caveat: component inspection shows the replay gain is mostly from CV and
fleet-overshoot risk reduction. `target_launch_delay_s` was zero across these
snapshot candidates, so this is a stronger action/value-planner repair, not a
completed Phase-4 terminal-release value layer.

Online wrapper screen completed but was not promoted:

```text
run:
results_freqduet/snapshotriskvalue_m005_ep100_wu10_4domain_20seed

tasks:
t12416-t12475

nodes:
node001-node006

matrix:
current main + risk-value full + risk-value target-only
4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup: 10
```

Result:

```text
full risk-value vs current main:
overall composite delta -0.0007, 95% CI [-0.0138, +0.0125]

target-only risk-value vs current main:
overall composite delta -0.0006, 95% CI [-0.0161, +0.0152]
```

The wrapper was directionally useful in highnoise/odshift but regressed terminal
and rushshift. Inspection showed the reason: `snapshotriskvalue_m005` replaced
the already-promoted terminal-bias snapshot selector instead of stacking on top
of it. That removed the positive terminal dispatch signal
(`snapshot_value_terminal_bias_mean`) and changed the terminal launch behavior.
This is negative evidence for the wrapper shape, not for the offline value model
itself.

Dual repair completed but is not promoted yet:

```text
run:
results_freqduet/snapshotriskdual_m005_ep100_wu10_4domain_20seed

tasks:
t12480-t12509

nodes:
node001-node006

matrix:
current main + dual risk-value full + dual risk-value target-only
4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup: 10
```

The dual repair keeps the current terminal-bias selector as the primary
`snapshot_value_selector` and adds the risk-proxy action/value model as a
secondary `snapshot_action_value_selector`. A local smoke test confirmed both
signals are active in the same run: `snapshot_value_terminal_bias_mean > 0` and
`snapshot_value_override_mean > 0`.

Result:

```text
full dual vs current main:
overall composite delta +0.0083, 95% CI [-0.0035, +0.0199]
odshift delta +0.0225, 95% CI [+0.0023, +0.0446]
rushshift delta +0.0097, 95% CI [+0.0005, +0.0195]

target-only dual vs current main:
overall composite delta -0.0029, 95% CI [-0.0154, +0.0100]
terminal delta -0.0162, 95% CI [-0.0355, +0.0020]
highnoise delta -0.0166, 95% CI [-0.0616, +0.0267]
odshift delta +0.0147, 95% CI [-0.0015, +0.0321]
rushshift delta +0.0065, 95% CI [-0.0023, +0.0150]

target-only dual vs fixed-headway:
overall composite delta -0.0430, 95% CI [-0.0879, -0.0006]
```

Interpretation: the full dual wrapper is clearly rejected because odshift and
rushshift regress. The target-only dual is a useful but unsafe candidate: it
turns the fixed-headway comparison significant and directionally improves
terminal/highnoise, but it still slightly worsens odshift/rushshift and raises
overall overshoot (`+0.0155`, 95% CI `[+0.0015, +0.0310]`). Next repair should
keep the target-only demand-conditioned value benefit while adding an overshoot
and terminal-bias-preservation guard around second-stage overrides.

Targetguard repair completed but was not promoted:

```text
run:
results_freqduet/snapshotriskdual_targetguard_m005_ep100_wu10_4domain_20seed

tasks:
t12528-t12547

nodes:
node001-node006

matrix:
current main + target-only dual with overshoot / terminal-bias guard
4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup: 10
```

Mechanism: the guarded target-only value layer blocks the secondary override
when the previous overshoot norm is above `0.18`, or when the override would
erase any positive primary terminal-bias signal. Local smoke confirmed that
`snapshot_value_guard_blocked_events` is nonzero, `snapshot_value_terminal_bias_s`
is preserved for blocked events, and guard diagnostics are written to both
episode diagnostics and trip-level logs.

Result:

```text
targetguard vs current main:
overall composite delta -0.0024, 95% CI [-0.0141, +0.0098]
highnoise delta -0.0180, 95% CI [-0.0642, +0.0316]
odshift delta +0.0048, 95% CI [-0.0141, +0.0227]
rushshift delta -0.0001, 95% CI [-0.0101, +0.0103]
overshoot delta +0.0078, 95% CI [-0.0012, +0.0168]

targetguard vs fixed-headway:
overall composite delta -0.0419, 95% CI [-0.0875, +0.0002]

targetguard vs old target-only dual:
overall composite delta +0.0010, 95% CI [-0.0100, +0.0120]
overshoot delta -0.0060, 95% CI [-0.0177, +0.0052]
```

Interpretation: targetguard removed the old target-only dual's significant
overshoot regression, but it is too strict. Override rate fell from roughly
`0.133` to `0.018`, and the fixed-headway advantage became just barely
non-significant. This is a useful guard mechanism but not the final setting.
Next step is a looser guard sweep over primary-bias loss and previous-overshoot
thresholds, rather than another code change.

2026-06-22 looser targetguard threshold sweep is running through the scheduler,
directly on `node001-node006`:

```text
run:
results_freqduet/snapshotriskdual_targetguard_sweep_m005_ep100_wu10_4domain_20seed

tasks:
t12553-t12592

nodes:
node001-node006

matrix:
current main
targetguard_b5o22    bias-loss <= 5s,  previous overshoot norm <= 0.22
targetguard_b15o22   bias-loss <= 15s, previous overshoot norm <= 0.22
targetguard_b15o30   bias-loss <= 15s, previous overshoot norm <= 0.30

4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup: 10
```

Important correction and final result: the first sync of this sweep was
premature and those partial CIs must not be cited. After waiting until all
`320/320` diagnostics reached `100` episodes, the completed paired CI result is:

```text
b5o22 vs current main:
overall composite delta -0.0029, 95% CI [-0.0126, +0.0059]
overshoot delta -0.0000, 95% CI [-0.0138, +0.0148]
vs fixed-headway overall -0.0401, 95% CI [-0.0908, +0.0060]

b15o22 vs current main:
overall composite delta -0.0038, 95% CI [-0.0154, +0.0089]
overshoot delta +0.0102, 95% CI [-0.0003, +0.0212]
vs fixed-headway overall -0.0409, 95% CI [-0.0868, -0.0003]

b15o30 vs current main:
overall composite delta -0.0043, 95% CI [-0.0146, +0.0067]
overshoot delta +0.0077, 95% CI [-0.0045, +0.0192]
vs fixed-headway overall -0.0415, 95% CI [-0.0908, +0.0055]
```

Interpretation: `b15o22` is the best looser-guard candidate. It ties current
main and is the only threshold with a significant overall advantage versus
fixed-headway. It still cuts terminal-bias preservation hard (`0.946` vs current
main `2.315`) and raises overshoot directionally, so it needed the `targetnonpos`
screen before promotion.

Follow-up repair now running: `snapshotriskdual_targetnonpos_m005` shrinks the
second-stage target-only action space to nonpositive offsets only:

```text
run:
results_freqduet/snapshotriskdual_targetnonpos_m005_ep100_wu10_4domain_20seed

tasks:
t12594-t12613

allowed methods:
actor_target_m30
actor_target_m15
actor_target_0

matrix:
current main + targetnonpos
4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup: 10
```

Rationale: trip diagnostics from old target-only and the looser guard sweep show
that most unsafe overrides are `actor_target_p15/p30`; even when previous
overshoot is low and primary terminal bias is near zero, positive target offsets
still raise overshoot. This test follows the SUMO-RL-style action-space shrink
idea more directly than another threshold sweep. Local warmup=0 smoke confirmed
nonzero override events and no positive-offset selected methods. Promotion bar:
no significant overshoot regression, no material current-main degradation, and
paired competitiveness versus fixed-headway.

Final targetnonpos result:

```text
targetnonpos vs current main:
overall composite delta +0.0002, 95% CI [-0.0094, +0.0098]
overshoot delta +0.0140, 95% CI [+0.0018, +0.0262]

targetnonpos vs fixed-headway:
overall composite delta -0.0468, 95% CI [-0.0930, -0.0014]

targetnonpos vs b15o22:
overall composite delta -0.0059, 95% CI [-0.0193, +0.0077]
overshoot delta +0.0015, 95% CI [-0.0125, +0.0155]
```

Decision: do not promote targetnonpos. It validates the action-space-shrink
intuition and is competitive with fixed-headway, but it still has a significant
overshoot regression against current main. Promote `targetguard_b15o22` instead:
it is the cleaner main alias because it ties current main, has a significant
overall advantage versus fixed-headway, and does not have a significant
overshoot regression.

Current-name alias audit after promotion:

```text
main alias run:
results_freqduet/paper_main_b15o22_alias_ep100_wu10_4domain_20seed
tasks t12618-t12627

pure fixed-headway run:
results_freqduet/paper_external_fixed_headway_ep100_wu10_4domain_20seed
tasks t12628-t12637

protocol:
4 domains x 20 seeds x 100 episodes
last-k: 50
upper warmup for learned main: 10
baseline: scripts/run_freqduet_external_baselines.py fixed_headway
```

Final paired result against pure fixed-headway:

```text
terminal:        delta -0.0063, 95% CI [-0.0238, +0.0122]
highnoise:       delta +0.0014, 95% CI [-0.0577, +0.0609]
odshift:         delta -0.0132, 95% CI [-0.0323, +0.0067]
rushshift:       delta +0.0109, 95% CI [+0.0002, +0.0213]
overall_shared:  delta -0.0018, 95% CI [-0.0216, +0.0181]
```

Interpretation: the promoted current config name is valid and overall ties the
strong pure fixed-headway baseline, but it does not dominate it. The remaining
weak point is rushshift, where current main is slightly but significantly worse
than fixed-headway under this 100ep protocol. Do not cite the earlier
targetguard-vs-fixed numbers as the final pure-fixed comparison; the current
audit is the clean table binding.

2026-06-25 update: a deterministic seed-fixed, domain-wise counterfactual
action candidate now gives the first 4-domain overall result that is
significantly better than the strong deterministic fixed-headway baseline:

```text
run:
results_freqduet/detseed_cfaction_domainbest_v1_ep100_wu10_4domain_20seed

candidate - fixed_headway composite delta:
terminal        -0.0207  CI [-0.0372, -0.0046]
highnoise       -0.0320  CI [-0.0691, +0.0073]
odshift         -0.0073  CI [-0.0260, +0.0115]
rushshift        0.0000  CI [+0.0000, +0.0000]
overall_shared  -0.0150  CI [-0.0256, -0.0049]
```

This closes the narrow "can FreqDuet beat fixed-headway under deterministic
paired seeds?" concern for a candidate config, but it does not yet close the
paper-main promotion gap. Versus `paper_main_b15o22_alias`, the improvement is
small and non-significant overall, and OD shift has a small non-significant
regression:

```text
candidate - paper_main composite delta:
terminal  -0.0161  CI [-0.0347, +0.0022]
highnoise -0.0024  CI [-0.0472, +0.0410]
odshift   +0.0132  CI [-0.0105, +0.0382]
rushshift -0.0118  CI [-0.0232, +0.0003]
overall   -0.0043  CI [-0.0190, +0.0100]
```

This 100ep result has now been superseded by the 200ep deterministic
confirmation above. The defensible current claim is stronger: v1 is promoted as
the deterministic paper-main candidate because it significantly beats
fixed-headway in the four-domain shared average at both 100ep and 200ep while
keeping rushshift fixed-safe. The remaining paper-main gap is naming and
packaging, not another OD fallback: rerun the final matrix under the
`*_paper_main_hiro` aliases, then generate the final tables/figures and
negative-results appendix.

Deterministic current-main rerun resolved a comparison ambiguity. The older
`paper_main_b15o22_alias` matrix was not enough after the stdlib-random seed
fix, so current main was rerun as:

```text
results_freqduet/detseed_current_main_ep100_wu10_4domain_20seed
```

Current main under this protocol only ties fixed-headway overall and is
significantly worse on rushshift:

```text
current main - fixed_headway:
terminal        +0.0072  CI [-0.0134, +0.0280]
highnoise       -0.0331  CI [-0.0749, +0.0096]
odshift         -0.0043  CI [-0.0294, +0.0230]
rushshift       +0.0080  CI [+0.0009, +0.0155]
overall_shared  -0.0055  CI [-0.0183, +0.0081]
```

`cfaction_domainbest_v1` is better than deterministic current main in the
important weak spots:

```text
cfaction_domainbest_v1 - current main:
terminal  -0.0279  CI [-0.0435, -0.0142]
highnoise +0.0011  CI [-0.0265, +0.0322]
odshift   -0.0030  CI [-0.0233, +0.0179]
rushshift -0.0080  CI [-0.0152, -0.0008]
overall   -0.0095  CI [-0.0201, +0.0010]
```

The attempted OD-main guard (`cfaction_domainbest_v2_odguard`) did not improve
v1 and should stay a negative result. The 200ep rerun resolved the promotion
question in favor of v1. A learned selector/value layer that reproduces v1's
domain actions without hard domain aliases remains a future paper-strengthening
direction, not a blocker for the current deterministic paper-main package.

Follow-up now running: `upperhist_current_b15o22_ep100_wu10_4domain_20seed`
tests the SUMO-RL exp39-style repair under the current promoted main:
short upper decision history plus small discrete residual action alphabets.
This is motivated by old rushshift screens where `upperhist3/4` were closest
to fixed-headway, unlike value-selector variants that tended to worsen
rushshift. Scheduler-direct tasks `t12650-t12679` are running on
`node001-node006`.

2026-06-17 figure follow-up: generated updated current-name mechanism and
decomposer packages:

```text
results_freqduet/mechanism_figures/current_terminalbias_ep100
results_freqduet/decomposer_validation/current_terminalbias_trace
```

The mechanism package loaded `480` runs and `24000` last-k episode rows. The
decomposer package uses `20` synthetic seeds plus logged traces and again shows
the harmonic-prior decomposer is the useful causal path: `harmonic_prior`
achieves lower synthetic LF RMSE and higher burst F1 than EMA, Haar, no-prior
harmonic, and raw-history baselines.

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
