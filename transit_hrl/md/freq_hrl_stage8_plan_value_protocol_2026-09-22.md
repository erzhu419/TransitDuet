# Stage-8 Plan-Value Qualification Protocol

Date: 2026-09-22

Evidence role: task-qualification development

Algorithm revision: `92735a4050eb175cc76d9a2418401e805fc8ca1b`

Experiment protocol: `pointmaze_plan_value_stage8_v1_development`

## Decision Being Tested

Stage 7 did not confirm a benefit from complete Haar coordinates, selective
frequency routing, or a hierarchy-by-multiscale interaction. Stage 8 does not
try another filter. It asks whether the next proposed learning problem exists:

> Given the same upper-planning budget, is control improved by knowing the
> current persistent regime and placing replans at task-relevant times?

This stage can authorize development of a learned belief and trigger. It
cannot validate such an algorithm because neither is trained here.

## Hidden-Regime Task

Each episode lasts 12 physical seconds (`1200 x 0.01 s`). The target moves on
the PointMaze U route under a hidden signed-speed regime selected from
`{-1.25, -0.55, 0.55, 1.25}` world units per second. Regimes persist for a
random 0.80-1.60 seconds and then change at non-fixed times. The actor sees the
current target but not the regime label, switch time, or future target.

The observable task stream also contains measured 0.04-0.10-second force
pulses and a piecewise-changing reward-irrelevant distractor. Both are seeded
independently of actions. This separates persistent plan-relevant change,
short disturbances that the lower controller can handle, and large irrelevant
observation changes.

The history window is 0.64 seconds. The fixed upper period is 0.50 seconds, or
2 upper calls per second. The lower controller acts every 0.01 seconds.

## Methods

| Method | Upper input | Lower input |
|---|---|---|
| `hrl_regime_history` | physical state, target error, causal raw task history | physical state, waypoint error, same raw history |
| `hrl_regime_oracle_context` | same input plus one-hot current true regime | identical to history lower input; no regime label |

The oracle policy receives the current regime only. It receives no future
regime or switch time. Parameter budgets are matched within optimizer root.
Both methods use the same goal-conditioned SMDP PPO core and task reward.

## Frozen-Policy Interventions

Every trained policy is evaluated on paired held-out paths under:

- fixed 0.50-second replanning;
- one initial plan held stale for the episode;
- fixed timing with a bounded 0.25-world-unit waypoint perturbation;
- privileged event timing with delays of 0, 0.10, 0.25, and 0.50 seconds.

For each privileged event schedule, the nearest periodic calls are relocated.
The total upper-call count is unchanged. These schedules are constructed from
the full hidden event table and are therefore oracle references, not deployable
candidates. Continuing a plan always retains closed-loop lower control.

Variable option durations are recorded explicitly. Upper returns use
discounted task reward with `gamma^duration` bootstrap; lower intrinsic GAE
ends at the actual waypoint change. No trigger receives lower intrinsic reward.

## Endpoints And Gate

The primary endpoint is physical-time integrated squared tracking error. The
optimizer root is the statistical unit; held-out paths are averaged within
root. Root-paired two-sided 95% Student-t intervals are used.

Stage 9 is authorized only if all conditions hold:

1. The history controller improves integrated tracking loss over its paired
   untrained policy.
2. Fixed refresh improves over a stale initial plan.
3. The unperturbed plan improves over the waypoint-perturbed control.
4. Current-regime oracle context improves fixed-period control.
5. Zero-delay oracle event timing improves over fixed timing at the same call
   count.
6. Delaying oracle timing by 0.25 seconds has a positive loss cost.
7. The upper 95% interval for the event-conditioned causal observation delay
   is below 0.25 seconds.

The observation-delay measure uses true events only to select diagnostic
windows. Within each window it uses target velocities only after their target
endpoint has been observed. It is an identifiability witness, not a deployable
change detector.

## Matrix And Stop Rule

The software preflight contains two cells: both methods at root `204901`, two
training iterations, one seed per role, and a 240-step episode containing
hidden-regime events.

The unchanged development matrix contains two methods at eight fresh optimizer
roots, for 16 independent cells. Each root uses eight training paths, eight
checkpoint-selection paths, and sixteen held-out paths. All role seeds are
fresh relative to Stage 7. The scheduler may place every single-core cell on
any of `node001`-`node006`; no cell is bound to a node.

Only compact `result.json` artifacts are synchronized. Checkpoints and raw
trajectories remain disabled. Sequential root extension is forbidden. If any
gate component fails, Stage 9 is not launched; the failed prerequisite is
reported and the task or control definition must be revised before a new
protocol.

## Commands

```bash
python3 scripts/submit_pointmaze_plan_value_stage8_scheduleurm.py \
  --run-name pointmaze_plan_value_stage8_v1_preflight_20260922_r1 \
  --preflight
```

After preflight audit, the development launch omits `--preflight`. Analysis is
performed with `scripts/analyze_pointmaze_plan_value_stage8.py` only after all
registered cells complete.
