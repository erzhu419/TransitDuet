# Unified Prospective Validation Protocol

Prepared: 2026-10-22. Status: **draft, not recipe-locked; pilot complete, recipe on hold**.
This document integrates the four preparations and takes precedence over their
earlier seed-count/statistical suggestions. The machine-readable reservation is
[`protocols/plan_validity_preregistration_2026-10-22.json`](../protocols/plan_validity_preregistration_2026-10-22.json).
Unknown costs and operational distance thresholds remain explicit pending fields.
Preparation verification:78 focused tests passed, including development/test separation,
shared-cost allocation, seed reservation, global alpha and separate C1/C2 logic.
No task was submitted and no performance evidence was generated at preparation time.
Execution update2026-10-28: the
[native preflight/pilot scheduler scripts](freq_hrl_native_development_scheduler_2026-10-28.md)
were subsequently authorized and run. Preflight passed; the
[pilot readout](freq_hrl_native_development_pilot_2026-10-28.md) has9 completed
measurements and3 interrupted cells. No confirmation training/evaluation was run.
Update2026-10-29: the3 recovery attempts completed, yielding12 logical pilot
measurements. The [complete readout](freq_hrl_native_development_complete_2026-10-29.md)
retains PointMaze positives but finds no stable Reacher advantage; one Reacher
stopping head fails its registered best-clock competence test. Interrupted costs
remain unknown. Recipe lock and confirmation remain blocked.
Update2026-11-10: the bounded [frozen diagnostic](freq_hrl_local_episode_result_2026-11-10.md)
completed on reserved pilot evaluation paths without fitting. PointMaze positives
remain; Reacher has opposing root effects and both prediction/credit limitations.
This diagnostic does not clear the failed competence or lifecycle-accounting gates.

## Questions And Claims

Freq-HRL addresses multiscale control with useful persistent plans and fast
feedback. PointMaze tracking and Reacher tracking test the same renewal objective,
fitting recipe and timing rule on different physical plants, with independently
trained weights. The study does not rank general-purpose RL optimizers.

The primary candidate is the original history renewal-value recipe. Two required
comparators are the development-selected strongest simple threshold and the
history learned-termination policy. Both comparisons must pass on both tasks;
choosing whichever comparator loses is not an admissible conclusion.

**C1, bounded main result:** in-distribution full-episode ISE improvement with
tail/risk noninferiority, eligible comparators and matched lifecycle-budget caps
on both tasks. **C2, stronger robustness result:** the same criteria also hold
under both registered stress profiles. A failed stress boundary blocks C2,
not automatically C1 or the entire project. If only tracking improves while
risk precision is insufficient, report that partial result rather than C1.

Fixed timing, jitter, best clock, current-only predictors, DF-matched current/
history ridge and the current learned head remain secondary mechanism controls.
Use their shared-controller comparison to interpret timing/representation, not
as a cold-build cost comparison or an extra confirmatory significance family.
Their analyses are descriptive; additional superiority claims require a new
prospective allocation. Custom stopping heads do not establish a named-method
reproduction result. Historical positives and Stage304's negative Transit result
remain separate evidence.

## Frozen Algorithm And Artifacts

Formal episodes use1,200 steps at10ms, first upper call0, one call per50-step bin,
checks0/5/10/15/20 and deadline25:24 calls, durations25-75. The local target remains
50-step `ISE(keep)-ISE(renew)` without later upper calls. Keep the original
feature construction, grouped ridge grid, within-fit OOF q75 threshold, controller
PPO backend and actual-duration credit. The nominal common recipe is384 controller
iterations, four renewal events/class,12 termination pairs/path, two termination
rounds,200 epochs, controller width128, termination width64 and matched slope DF8.
The JSON records the remaining numerical clocks, optimizer and driver defaults.
Task-specific geometry/units are the previously prepared physical adapters.

There are two locks. The **recipe lock**, before confirmation-root training,
fixes the complete seed/trial roster, recipe, costs/caps, risk definitions and
all comparisons. Each confirmation root may train, select checkpoints and fit
its timing rule only on its declared development roles using the frozen recipe.
The **global artifact lock**, before ANY confirmation-path access, binds all48
primary pipeline artifacts:2 tasks x8 roots x3 methods, including source report,
run ID, checkpoint, frozen rule/fitting cache and budget admission. Freeze all
tasks/comparators together; early confirmation outcomes cannot affect later fits.

Controller selection and timing selection are different roles. Lowest-ISE learned
stopping checkpoints remain selected even if collapsed. Development eligibility
requires multiple offsets, state-conditional stopping at a common clock and
selection improvement over the best constant clock. Pilot failure stops recipe
confirmation. An ineligible confirmation-root comparator is retained with its
outcomes and blocks the strong comparative claim; it is never replaced by a
more favorable root or artificially forced to make early calls.

## Seed Reservation

Three disjoint phases have1 preflight,2 cost-pilot and8 confirmation optimizer
roots per task. The arithmetic namespace is reserved in the JSON, not inferred
from filenames: base1,970,000,000 plus phase*1,000,000, task*100,000 and root*10,000.
Offsets1,000/2,000/3,000/4,000/5,000 identify these roles:

- Controller training:8 path roots, passed through the existing iteration seed generator.
- Controller checkpoint selection:8 paths.
- Timing supervision/fitting:8 paths, with grouped CV restricted to this role.
- Timing/threshold/checkpoint selection:8 paths.
- Reserved final evaluation:512 complete independent paths, split in listed order256/256.

Preflight instead uses2/2/3/2/16 paths; pilot uses8/8/8/8/16. These phase-specific
evaluation paths are development-only. The final diagnostic subset contains
the first four paths from each evaluation half:8 paths,3 probes/path and4
conditional future replicas, split2/2. Counts and generator tags are frozen.
Paths are paired across methods but distinct across optimizer roots and phases.
Only the role inputs, not primitive steps/probes/futures, define independent N.

The generated reservation has22 root/task cells and8,968 distinct role/root IDs.
Before recipe lock, audit historical exposure using compact server metadata and
the resolved effective training seeds. The draft namespace is not a claim that
this historical audit has already happened. Revealed historical confirmation
paths cannot be relabelled as fresh paths.

## Total Budget

Primary lifecycle accounting uses N=100 deployments, one CPU thread per worker
and matched CPU/library profiles. One numeric CPU cap and environment-step cap
per task apply to all methods. Calibration chooses caps from measured pilot
costs, before confirmation; it does not choose favorable test effects.

```text
root_search = all declared local trials, including failures and losing attempts
shared_development = measured method-specific preflight/pilot/search costs
root_lifecycle_CPU = root_search_CPU + shared_development_CPU / 8
                     + 100 * validation_control_CPU_per_episode
charged_steps = root_search_steps + shared_development_steps / 8
```

The budget selector accepts separate measured/planned shared reports using
`--shared-inputs`/`--shared-plans`. Pilot models cannot become confirmation
winners. Divide by the complete declared cohort, never only successful roots;
missing/unknown costs block admission. Common infrastructure/secondary mechanism
experiments and confirmation/branch diagnostics have separate campaign ledgers.
Report their actual totals once; they are not free deployment supervision.
Any research work used to tune a primary method belongs in its charged search,
not in a diagnostic ledger. Historical unmeasured R&D costs are outside this
prospective campaign's accounting scope.

Include setup, controller rollouts/optimization/selection, replay prefixes and
branches, fitting/OOF, timing search, scoring and artifact export. Control latency
spans history processing through action output, excludes simulator advancement,
and reports p50/p95/p99/max, first-step latency and10ms deadline misses. The
primary p99 cap is10ms. Simulator CPU is included in offline costs. Equal calls
are not equal CPU. N=1/1000 and other resource ladders are secondary cost curves,
not extra primary tests. This measures warm-worker CPU, not cold-process or GPU
efficiency. Budget admission is measured feasibility, not a preemptive limiter.

## Inference And Reliability

Use the same frozen pair under unchanged dynamics, half regime dwell, and
triple force amplitude/double duration. Operational failure is a predeclared
per-task distance exceedance lasting25 steps. Risk-rate margin is0.02; the
empirical upper-decile ISE mean has margin0. These margins are fixed design
choices, not zero-risk or population-CVaR guarantees. Distance cutoffs remain
pending development-only operational calibration.

Each task/comparator family receives alpha0.0125:2 tasks x2 comparators exhaust
global alpha0.05. Within a family, half covers mean ISE, tail ISE and failure-
rate root intervals across3 profiles; half covers conditional harmful-discordance
bounds across8 roots and3 profiles. Thus endpoint alpha is0.0125/(3*6), and each
one-sided harmful-event bound uses0.0125/(3*8*2). Analyze each task/comparator
separately with the registered comparator field; never concatenate them as roots.

Paired root means, with root-level t intervals, are the primary unit. Conditional
path intervals, local MSE, occupancy and probe intervals remain descriptive.
Both evaluation halves must have negative ISE point differences; neither half
may select a model. Report absolute/relative effects in task units; do not pool
raw PointMaze and normalized Reacher ISE. For zero harmful discordances409 paths
are needed even to reach the registered2pp precision at the assigned alpha;
512 is the fixed prospective count, not a promise of adequate power for mean
improvement. No post-result seed expansion or favorable interim stopping.

The reliability code checks shared causal prefixes and factual replay, then
separates local value, within-window opportunity gap and episode tail using
`G = L + (W-L) + T`. Independent conditional futures preserve past/current latent
state. Fixed-clock jitter/deployed occupancy and first/second-half costs expose
distribution changes. No new fitting or policy repair follows these diagnostics.

## Resume Order

1. **Recover runtime:** use scheduler workers on node001-node006, not Slurm or
   login-node installs. Use one thread/core, work-conserving placement and no hard
   node pins. Retrieve only logs/compact JSONs; keep caches/checkpoints server-side.
2. **Native plumbing preflight:** both plants, reduced300-step/2-iteration builds,
   pairs/path2, unchanged10ms/50/25/5 timing and50-step target. Test real model
   loading, duration/credit, identical-prefix/factual replay, accounting and risk
   fields. These reduced episodes are not performance evidence.
3. **Development cost/competence pilot:** separate pilot roots; cold-build the
   three primary methods with the nominal recipe. Check learning/plan authority,
   query coverage, learned stopping eligibility and cost/latency. Resolve unknown
   thresholds/caps and affordable precision using development data only. Any
   recipe repair requires another explicit draft before confirmation.
4. **Recipe lock and confirmation-root builds:** audit seed exposure; approve
   numeric budgets/risks and the complete roster; train all new roots using only
   their four development roles. Charge failed/retried builds and pilot overhead.
5. **Global artifact lock:** bind all48 primary sources and budget admissions,
   retain ineligible/missing roots, then review the generated registry. Unknown
   costs or missing bindings prevent the strong joint claim.
6. **Formal evaluation:** frozen artifacts only; both comparator cases on all
   declared paths/profiles, then the smaller predeclared probe subset. Reusing
   weights does not retrain them. Existing separate cases may repeat candidate
   simulation/probes; charge actual repeated evaluation cost and never count
   those repetitions as extra independent evidence.
7. **Fixed analysis and decision:** verify all four task/comparator summaries,
   all roots, path rosters, family alpha and margins. Report C1/C2 separately,
   effect sizes, budgets, comparator failures and unresolved precision. Poor
   outcomes receive no root replacement, retuning or confirmation-data repair.
   Outcome-independent infrastructure retries use the same inputs and retain
   costs. An algorithmic repair after revealed tests starts a new registration
   and population while preserving the original results.

## Registry Entry And Limitations

`scripts/prepare_plan_validity_preregistration.py --output PATH` writes a draft
only; `--input EXISTING_JSON --output REVIEW_JSON` validates its edited `plan`
section. The command has no execution or scheduler path. Its readiness is a
structural plan review, not verification of asserted artifact evidence.
After evaluation, `--summaries FILES...` checks the four task/comparator summaries
against the registered alpha, margins, root/episode counts and retained profiles,
and reports C1/C2 separately. Incomplete/unlocked evidence has no claim verdict.

Current state is `draft_not_locked`: all12 pilot measurements are available,
but full lifecycle accounting, numeric caps, distance cutoffs, competent stopping
baselines on every Reacher root, historical exposure audit and48 artifact
bindings remain pending. Root intervals with8 roots are approximate. Conditional-future
MSE is not observable-state Bayes error; the local identity is not an episode
regret theorem. Own-schedule full-algorithm ranking, named timing reproductions
and universal/real-world deployment claims remain outside this registration.

The 2026-11-10 [matched-target repair](freq_hrl_target_credit_repair_2026-11-10.md)
compares L/W/G on common training prefixes, capacity and conditional futures.
Four native prechecks passed before main submission. New development prefixes
are reserved separately from the revealed pilot evaluation paths; no confirmation
roots are accessed. This repair does not change the original gates or clear HOLD.
All four main cells subsequently completed: W improves matched L on average but
not the frozen history reference; G is not a reliable repair. The
[readout](freq_hrl_target_credit_result_2026-11-10.md) retains root/half differences
and separates independent-future prediction from new-prefix transfer.
The subsequent [W factorial](freq_hrl_window_coverage_2026-11-10.md) freezes
8-prefix/4-future versus 32-prefix/1-future training at DF2/DF8, with new held
paths and unchanged controls. Its four native prechecks passed; parent HOLD and
confirmation restrictions remain unchanged.
All four factorial cells completed: wider fitting coverage improves held W
prediction and Reacher ISE point estimates, but no arm beats pooled history or
threshold. The [result](freq_hrl_window_coverage_result_2026-11-10.md) identifies
bin-start-only supervision versus later deployed checks as the next support gap;
it does not relax confirmation gates or alter the frozen models.
The next [phase-support factorial](freq_hrl_phase_support_2026-11-10.md) separates
bin-start/reachable-check supervision from explicit clock information at DF8 and
96 fitting futures/arm, using another held cohort. Four native prechecks passed,
including real nonzero phases and unchanged 24-call budgets. It remains a
development repair and does not release confirmation.
The four phase-support cells completed. All optional phases and causal clock
inputs are active, but the joint repair worsens PointMaze and gives only a fragile
Reacher point gain, without surpassing the old controls. The
[readout](freq_hrl_phase_support_result_2026-11-10.md) retains the whole cohort and
prioritizes prefix/continuation consistency diagnostics; confirmation stays held.

The [frozen prefix/continuation diagnostic](freq_hrl_continuation_diagnostic_2026-11-10.md)
crosses original history and checks/clock without further fitting. Four native
prechecks gate eight new paths per existing root, with matched conditional
futures at common optional clocks. Prefix effects, continuation effects and
their interaction are read separately; exclusive clocks and fixed-grid occupancy
remain visible. Conditional-path uncertainty does not add trained roots or
release the original confirmation gates.
All eight crossed-continuation shards completed. The
[readout](freq_hrl_continuation_result_2026-11-10.md) finds no complete-episode
advantage over history. Prefix-sensitive error, continuation-dependent labels,
and local/long-horizon credit differ by root; cache-only future-half and zero-cut
controls do not identify a universal repair. They do not select a new candidate,
add trained roots, authorize new jobs or release HOLD.

The next [paired-label experiment](freq_hrl_label_continuation_2026-11-10.md)
fixes C0-visited training states and changes only label continuation. It retains
DF8, training-A OOF calibration and equal fitting information, and reserves new
prefixes for a fixed-continuation one-step readout. Native prechecks precede
sampling; all roots' fits precede new-prefix access. No new-head episode loop
or confirmation is automatically scheduled.
All eight training-sampling shards subsequently completed. The four cache-only
fit jobs froze both DF8 heads/root without new simulation, then the eight
reserved-prefix probe shards started. This preserves the frozen one-step scope;
training-fit statistics do not establish fresh-prefix or closed-loop improvement.
All eight probes completed. The [result](freq_hrl_label_continuation_result_2026-11-10.md)
finds fresh-prefix W error worse on three roots and no stable W/G gain. PointMaze
actions scarcely change; Reacher changes actions without reliable transfer.
Continuation-only relabeling is not sufficient. No closed-loop follow-up was
submitted and the original confirmation HOLD remains.

The [A-only decision calibration](freq_hrl_decision_calibration_2026-11-10.md)
subsequently completed four cache-only jobs, reconstructing 256 OOF ridge folds
without changing full heads or taking native steps. The
[readout](freq_hrl_decision_calibration_result_2026-11-10.md) finds no stable joint
W/G gain across B halves. Selected A gains can fail on independent B futures even
with OOF scores; relative constraints also fail to transfer to full-model scores.
Recovered scores are saved for a bounded score-source diagnostic, not B-based
retuning. No further native evaluation or confirmation is released; HOLD remains.

The next [saved-score source control](freq_hrl_score_source_2026-11-10.md)
freezes OOF-selected and full-score-selected A-only cuts, crosses both score views,
and compares them directly on identical full-model scores. It reuses saved OOF
predictions with zero new fits or native steps. Full-score A calibration is
in-sample; previously revealed B is diagnostic only. Original gates stay held.
All four controls completed with zero fits/native steps. The
[readout](freq_hrl_score_source_result_2026-11-11.md) finds that score alignment
repairs the empirical full-score W floor and improves B W point behavior, but G
is still root-dependent and no G prefix interval excludes zero. Separate
calibration information is the next design need, not another B-based cut search.
No further jobs or confirmation are released.

The [independent-prefix calibration protocol](freq_hrl_independent_calibration_2026-11-11.md)
now separates new precheck, 16-path calibration and 8-path readout cohorts/root.
All eight heads stay fixed, A4 alone selects cuts, and all four roots must freeze
before new readout access. Only four native prechecks are released (48,000-step
cap); 6,336,000 formal sampling/readout steps remain unsubmitted pending a separate
decision. This is development work, not release of the original confirmation cohort.
All four native prechecks subsequently passed on node004/node006, using 33,600
steps with no fitting or controller updates. The
[execution receipt](../results/plan_validity_independent_calibration_20261111/precheck_summary.json)
retains the zero formal-task count at precheck completion, when both formal cohorts
were unused. The next bounded release is calibration sampling only, not confirmation or episodes.
The subsequent user-authorized release submits eight calibration shards only
(t214476-t214483; 4,224,000-step budget), preserving all frozen cohorts and rules.
The [sampling receipt](../results/plan_validity_independent_calibration_20261111/sampling_release.json)
keeps this stage distinct from later cut freezing and fresh-prefix readout; neither
later phase nor the original confirmation cohort is released.

All eight samples completed within the exact budget, followed by four A4-only
zero-simulation cut-freeze jobs. The
[calibration result](freq_hrl_independent_calibration_result_2026-11-11.md) preserves
positive W behavior and local C0/old-cut gains, but G remains unstable; the only
both-half G gains versus quantile occur at a root with zero immediate renewals
on the sampled states. All eight cuts are now frozen. Recommend the already
declared whole-cohort E8 prefix-transfer test before any further calibration
changes; no readout jobs or original confirmation are released by this analysis.
The subsequent user-authorized step releases the four frozen whole-cohort E8
readouts (t214504-t214507; 2,112,000 steps), with all four roots and both heads.
The [E8 release receipt](../results/plan_validity_independent_calibration_20261111/readout_release.json)
keeps launch separate from outcome. No cut/model revision, extra roots, original
confirmation or complete-episode follow-up is released.

The four E8 readouts completed at exactly 2,112,000 steps. The
[frozen result](freq_hrl_independent_calibration_result_2026-11-11.md#frozen-e8-result)
does not meet both-root joint W/G stability. Preserve PointMaze root2's positive
half-group points and Reacher's short-window losses; do not generalize either
beyond the sampled interventions. Stop scalar-cut repair and keep E8 as exposed
evidence. No new tasks or original confirmation are released.

The subsequent [nested boundary-credit precheck](freq_hrl_boundary_credit_precheck_2026-11-11.md)
changes label acquisition rather than reopening the frozen cuts. Independent
realized windows and paired conditional suffix redraws separate two sources of
credit noise. Only four mechanical prechecks (52,800 steps, zero fits) are in
scope. E8 stays exposed evidence; formal variance sampling and policy fitting
are not released.
All four prechecks passed at the exact replay-only budget. This establishes the
boundary-label acquisition path; it does not qualify a continuation critic.
The next decision is a cost-accounted variance diagnostic on separate
development prefixes, not a new threshold or deployment trial.
The next [frozen variance diagnostic](freq_hrl_boundary_variance_2026-11-11.md)
uses 32 separate development prefixes, 96 states and 4x4 nested draws/state.
All complete replays count toward 3,955,200 steps across eight single-core jobs.
The release covers variance acquisition/readout only; no fitting, cut search,
E8 reuse or policy validation is admitted.
The user-authorized release submits t214559-t214566 after 26 focused tests.
The [variance launch receipt](../results/plan_validity_boundary_variance_20261111/release.json)
preserves exact budget, rosters and live status. No training or validation stage
follows automatically from this descriptive acquisition.
All eight variance shards completed with exact roster and native-step closure.
The linked readout separates window and suffix uncertainty; G suffix variance
shares are 71.7%-87.0%, with no cross-root stable sign claim. Only compact
moments were retrieved. The next proposed cache-only boundary-tail probe is
not released by these results; original confirmation remains on HOLD.
The user's next-step authorization now releases only the
[cache-only boundary-tail probe](freq_hrl_boundary_tail_probe_2026-11-11.md):
four frozen root jobs, total DF 3, eight whole-prefix folds, suffix 0/1 fitting
and 2/3 readout. Budget: 72 ridge fits, 36 constant fits, zero native steps.
No policy, threshold, upstream distillation or original confirmation is released.
All four cache workers completed the exact 72/36-fit, zero-step budget.
Only Reacher root1 meets the frozen boundary-tail rule. PointMaze exposes
near-zero contrast-scale extrapolation; Reacher root2 remains half-group mixed.
Preserve v1 losses and fixed-model attribution. A parent-feature scaling repair
is the next design, not an automatic refit or evidence of deployable timing gain.
The next user authorization releases the single
[parent-scale repair](freq_hrl_boundary_parent_scale_2026-11-11.md): 36 potential
ridge fits only, frozen baseline predictions reused, zero new native steps.
Keep DF 3, suffix A/B, whole-prefix folds and the original qualification rule.
No clipping, additional geometry search or policy confirmation is released.
The 36-fit repair completed with exact baseline reproduction and zero native
steps. PointMaze scale amplification is removed; three roots pass the declared
B rule, not the all-root rule. A/B disagreement and Reacher root2 remain.
Keep v1 and repaired results; no root-specific version choice or further fitting.
The next [fixed-prediction noise attribution](freq_hrl_boundary_noise_attribution_2026-11-11.md)
is a single read-only scheduler job: paired-loss suffix-B MC SE, prefix readout
deletion and noise-corrected fixed-prefix dispersion, no fit or native step.
No A/B-gap test, confidence interval, additional sampling or policy release.
The fixed-prediction readout completed with zero counted fit/native work.
PointMaze root1 is not single-prefix driven; Reacher root1's pooled improvement
is smaller than its suffix SE, while root2 also has persistent prefix/known-state
problems. Preserve these distinctions; the next data design must separate
training-future information from prefix coverage, without retuning this bank.
The next [boundary information design](freq_hrl_boundary_information_2026-11-11.md)
freezes dense/wide at 1,411,200 incremental native steps/root each, including
wide factual overhead; labels are 780/768, CPU is measured rather than assumed
equal. Only four 22,800-step mechanical prechecks are released now. Formal
sampling, eight potential fits and reserved new-prefix E readout remain gated.
All four information prechecks passed at exactly 91,200 steps and zero fits.
Cached/fresh invariants hold; no reserved training/evaluation role was accessed.
Next phase is 32 training-acquisition shards, not model fitting or E readout;
that formal release has not occurred.
The subsequent user authorization now releases the linked 32-shard training
acquisition manifest only: 11,289,600 native steps, zero fits. Workers verify
all four prechecks, preserve partial paid rows and never access readout seeds.
No automatic fitting, E readout or policy promotion follows acquisition.
The 32 acquisition tasks have launched with actual native progress; the linked
snapshot retains task identities, CPU permissions, costs and partial completion.
No model fitting or E-prefix access is released while acquisition is pending.
All 32 tasks have now completed at the exact 11,289,600-step acquisition budget.
The next [audit-and-fit release](freq_hrl_boundary_information_fit_2026-11-11.md)
audits every private bank before fitting eight DF-3 potentials, zero native
steps and zero baseline refits. Full A means include Dense's eight/nine labels.
All models freeze before training metrics; E readout is still not released.
