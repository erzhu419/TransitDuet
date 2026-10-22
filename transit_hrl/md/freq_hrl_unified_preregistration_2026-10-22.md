# Unified Prospective Validation Protocol

Date: 2026-10-22. Status: **draft, not locked and not submitted for execution**.
This document integrates the four preparations and takes precedence over their
earlier seed-count/statistical suggestions. The machine-readable reservation is
[`protocols/plan_validity_preregistration_2026-10-22.json`](../protocols/plan_validity_preregistration_2026-10-22.json).
Unknown costs and operational distance thresholds remain explicit pending fields.
Verification:78 focused tests passed, including development/test separation,
shared-cost allocation, seed reservation, global alpha and separate C1/C2 logic.
No task was submitted and no new performance evidence was generated.

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

Current state is `draft_not_locked`: actual pilot costs, distance cutoffs,
competent baseline evidence, historical exposure audit and48 artifact bindings
do not exist yet. Root intervals with8 roots are approximate. Conditional-future
MSE is not observable-state Bayes error; the local identity is not an episode
regret theorem. Own-schedule full-algorithm ranking, named timing reproductions
and universal/real-world deployment claims remain outside this registration.
