# Phase 4 Terminal Dispatch Scope Decision

Last updated: 2026-06-10 CST

## Decision

The current paper main line is scoped as a frequency-separated
target-headway / executable timetable controller. Full real terminal dispatch
with actual launch-time rescheduling and first-stop holding is kept as Phase 4
future work, not as the current promoted main path.

This is a scope decision, not a simplification of the method design. The
validated implementation follows the `dev_manual.md` staged path:

- Phase 3: upper policy produces a smooth low-frequency target-headway
  timetable, evaluated through executable target headways in the existing HIRO
  runner.
- Phase 4: upper policy directly changes actual terminal launch times and
  treats first-stop holding as a separate physical control surface.

The paper should claim the Phase 3 contribution unless a learned Phase 4 value
model is implemented and validated later.

## Why Not Promote A Heuristic Phase 4 Patch

The terminal-dispatch branch was tested through several causal, no-future-leak
heuristics. None produced a stable improvement over the promoted main line.

`termhold45` mostly repeated the existing terminal shift-cap surface. Its
40-episode, 20-seed result was neutral:

```text
overall candidate - main composite: +0.0064 CI [-0.0157,+0.0288]
```

`termfb30` converted completed-trip lower holding history into a no-early-launch
terminal bias. The bias collapsed to near zero in the last training window, and
the overall result was weakly negative:

```text
overall candidate - main composite: +0.0202 CI [-0.0035,+0.0461]
terminal candidate - main composite: +0.0464 CI [+0.0063,+0.0922]
terminal_feedback_bias_mean overall: +0.0089s versus main
```

`termrelief20` created real terminal action from on-route fleet pressure, but
the wait/CV tradeoff was not universal:

```text
overall candidate - main composite: +0.0219 CI [-0.0147,+0.0577]
terminal_feedback_bias_mean overall: +5.86s
terminal_launch_shift_mean overall: +6.62s
overall CV delta: +0.0057 CI [+0.0012,+0.0101]
```

`termvalue20` added a causal headway-value gate to avoid delaying departures
when passenger/CV cost was too high. It created moderate terminal action, but
still did not beat main:

```text
overall candidate - main composite: +0.0160 CI [-0.0178,+0.0486]
terminal_feedback_bias_mean overall: +2.44s
terminal_launch_shift_mean overall: +2.22s
```

The lower value-cost branch also does not justify substituting lower penalties
for true terminal dispatch:

```text
valuesoft35 overall candidate - main composite: +0.0235 CI [-0.0111,+0.0552]
valuesoft35_lfsafe overall candidate - main composite: +0.0826 CI [+0.0605,+0.1044]
```

The common failure mode is that hand-written terminal delay or lower value-cost
rules can reduce one component, but they raise passenger wait, CV, or
domain-specific fleet overshoot elsewhere. This is not strong enough for a
paper-main promotion.

## What A Valid Phase 4 Upgrade Requires

A credible Phase 4 implementation should not be another threshold rule. It
needs a learned or explicitly estimated value model for first-stop / terminal
delay:

- state: low-frequency demand trend, high-frequency energy, terminal queue,
  same-direction dispatch gap, fleet pressure, recent lower drift, and
  direction-specific headway state;
- action: actual terminal launch delay or release decision, separate from
  middle-stop lower holding;
- value: counterfactual wait, CV, and fleet-relief estimates for delaying now
  versus releasing now;
- constraints: no future demand leakage, hard fleet-budget accounting, and
  paired-seed evaluation against promoted main, fixed-headway, and the
  closest preserved TransitDuet-family baseline.

Promotion criteria should be at least:

- significant or practically meaningful 100/200-episode improvement over the
  promoted main overall;
- no significant regression in terminal, highnoise, odshift, or rushshift;
- improved or non-regressed comparison against fixed-headway;
- interpretable evidence that terminal delay is replacing harmful on-route
  holding rather than merely shifting cost between wait, CV, and overshoot.

Until that evidence exists, the manuscript should present terminal dispatch as
future work and keep the validated target-headway executable timetable as the
main contribution.
