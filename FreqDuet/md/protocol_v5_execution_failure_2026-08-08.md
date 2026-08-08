# FreqDuet Protocol V5 Execution Failure

Date: 2026-08-08

## Decision

Protocol V5 is **implementation-invalid**. Its development screen is not valid
method evidence, and no V5 metric may be promoted, relabeled as V6, or used to
support an effectiveness or baseline-superiority claim.

The failure was identified before the V5 development matrix completed. All 88
running scheduler tasks, `t71918` through `t72005`, were canceled. Their logs
may be retained only as implementation-failure diagnostics.

## Root cause

The exact timetable planner materialized a plan when the upper policy made a
new decision. On later callbacks that were supposed to reuse that cached plan,
the runner called the mutating planner path again. The cached action was
therefore reapplied to future trips instead of being read without side effects.

A deterministic local reproduction showed the downstream endpoint phase
changing under repeated callbacks even though no new upper decision was made:

```text
callback 0: endpoint phase shift =   0 s
callback 1: endpoint phase shift = -48 s
callback 2: endpoint phase shift = -88 s
```

The same action semantics also admitted receding-horizon phase drift at genuine
replan boundaries: a zero-sum projection over each changing finite horizon did
not guarantee zero accumulated phase over a stable rolling block. Consequently,
the reported per-plan projected delta sum could be zero while the executed
timetable still drifted across callbacks and replans.

This violates the executable-upper-action and low-frequency ownership
requirements inherited from the V4 method contract. It also breaks the central
interpretation in `md/GPT.md` and `md/dev_manual.md`: the upper controller must
produce the slow timetable plan once, while the lower controller supplies local
high-frequency corrections without an accidental second upper action.

## Why cancellation was mandatory

The defect changes the treatment rather than merely changing a diagnostic. A
policy trained under repeated schedule mutation is not executing the action
declared by the method, so more seeds, longer training, or confidence intervals
cannot repair the evidence. The complete V5 matrix was therefore stopped
instead of being aggregated with a caveat.

The cancellation applies to the entire `t71918`-`t72005` range. Partial V5
checkpoints, evaluation CSVs, and warm-up observations are ineligible for:

- candidate selection;
- comparison with fixed-headway, rule holding, rule MPC, or TransitDuet;
- ablation or mechanism tables;
- paper figures, claims, or supplementary statistics.

## V6 repair boundary

Protocol V6 is a new engineering protocol, not a corrected label for V5. Its
implementation repair set is:

1. rolling zero-sum headway blocks that remain phase-conserving across replans;
2. a read-only cached-plan summary path, with mutation only on a new decision;
3. within-bin causal passenger release and completed-bin APC aggregation;
4. lower holding guard evidence from matched pre-action departure gaps;
5. separate commanded and realized holding accounting;
6. episode-global additive interval credit, with clipping after aggregation so
   reward does not depend on interval partitioning;
7. fail-closed decision, matrix, and paper-artifact provenance.

These are implementation and evidence-integrity repairs. At this date V6 has
not passed its development screen or independent confirmation. Therefore this
document does **not** establish that V6 is effective, that frequency separation
is supported, or that V6 matches or outperforms any baseline.

## Required historical wording

Permitted wording:

> Protocol V5 was invalidated before completion because cached timetable reuse
> mutated future departures and finite-horizon replanning accumulated phase
> drift; all 88 development tasks were canceled and excluded from analysis.

Forbidden wording includes any statement that V5 was a negative-result
ablation, an underperforming but valid model, or an empirical predecessor whose
numbers can be compared with V6. It was an invalid execution protocol.
