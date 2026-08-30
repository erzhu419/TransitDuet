# MuJoCo v17.2 Smooth Macro Gauge Preflight

## Motivation

v17.1 established a negative result: exact zero-DC lower control over every
16-step macro interval removes too much slow actuation and fails the reward and
upper-frequency gates. More gate tuning cannot repair that structural conflict.

v17.2 instead treats the additive upper/lower split as an identifiable causal
coordinate choice. A prior-step low-pass estimate of the total action is sampled
at each upper boundary. One frozen endpoint-exact smoothstep curve supplies the
canonical upper responsibility over the next macro interval, while the lower
responsibility is the exact additive complement. Per-step projection keeps both
components feasible. Gauge strength interpolates the reported split but never
changes the executed total action.

## Frozen Identity

- Algorithm revision: `22b77b5ed820276644e6d512ebff9dd8a6b3fe77`
- Source manifest: `c7f6a535ff4bd1db0fdf77019882d8d96435e8cead08dc12617a5634fe072b63`
- Core protocol: `freq_hrl_mujoco_shared_core_v17_2_smooth_macro_gauge`
- Development protocol: `mujoco_v17_2_smooth_macro_gauge_preflight_v1`
- Evidence role: development only, not confirmatory

## Paired Design

The matrix contains nine training cells: three environments crossed with
`alpha` values 0.05, 0.10, and 0.20, using one fresh optimizer seed. Each cell
trains one reward-only strength-zero policy on four fresh training roots and
selects its checkpoint on four disjoint selection roots.

The frozen checkpoint is then evaluated twice on the same 40 development
held-out paths: five disturbance regimes crossed with eight fresh seeds. The
first pass uses gauge strength zero and the second uses strength one. This
paired intervention removes optimizer variation from the gauge comparison and
allows exact path-level checks.

## Gates

Every eligible alpha must pass every gate in all three environments:

1. Reward, executed-action, and latent-policy trace hashes match on all 40
   paired paths.
2. Episode return and latent frequency metrics match numerically within the
   frozen tolerances.
3. Router and responsibility reconstruction RMS are at most `1e-7`.
4. Mean component-feasibility projection rate is at most 0.25.
5. Mean upper HPF8 power, lower LPF32 drift, and their normalized joint merit
   each improve by at least 10%.

If multiple alphas pass, selection maximizes the worst-environment joint-merit
reduction, then the median reduction. If none pass, no multiseed expansion is
allowed; the negative result must be registered before another mechanism is
designed.

## Scheduler And Artifacts

The launcher uses scheduleurm with dynamic placement over `node001-node006`:
`require_node=None`, one physical CPU core and 1 GB RAM per task. Only
`cell_summary.json`, `evaluation_rows.csv`, and the server artifact locator are
synchronized locally. Checkpoints and training histories remain on the worker
nodes.

```bash
cd /home/erzhu419/mine_code/TransitDuet_freqhrl_v3/transit_hrl
python3 scripts/submit_mujoco_v17_2_smooth_macro_gauge_preflight_scheduleurm.py \
  --run-name mujoco_v17_2_smooth_macro_gauge_preflight_20260831_r1 \
  --dispatch
```

## Claim Boundary

A supported result would show that one causal, identifiable responsibility
gauge improves the registered frequency attribution without changing the
frozen policy or its environment path. It would not show reward improvement,
learned constraint improvement, robustness across optimizer seeds, or
confirmatory algorithm performance. Those require a fresh leakage-active
multiseed campaign after one alpha is frozen.
