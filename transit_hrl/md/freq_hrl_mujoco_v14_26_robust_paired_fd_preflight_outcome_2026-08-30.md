# MuJoCo v14.26 robust paired finite-difference outcome

## Frozen execution

- Source revision: `4a517c45ca`
- Run: `mujoco_v14_26_robust_paired_fd_preflight_20260830_r1`
- Scheduler tasks: `t84819`-`t84821`
- Placement: dynamic scheduler placement on `node002`; no required node and no Slurm
- Completion: all three tasks finished without retry or runtime error
- Evidence role: adaptive post-v14.25 preflight, not confirmatory evidence

The synchronized payloads passed the frozen contract checks. Each cell contains
160 critic-train intervention paths, 160 critic-holdout intervention paths, 64
design paths, and, when selected, 64 untouched validation paths.

## Results

| Environment | Upper R2 | Lower R2 | Upper overall cosine | Lower overall cosine | Upper minimum mode cosine | Lower minimum mode cosine | Gate | Validation |
|---|---:|---:|---:|---:|---:|---:|---|---|
| HalfCheetah-v5 | 0.6652 | 0.9339 | 0.7841 | 0.8827 | -0.2929 | -0.0041 | fail | not run |
| Hopper-v5 | 0.6029 | 0.9135 | 0.7596 | 0.5918 | 0.3070 | -0.2954 | fail | not run |
| Walker2d-v5 | 0.2600 | 0.9523 | 0.8471 | 0.4771 | 0.4299 | 0.0703 | pass | supported |

All upper and lower critics passed both positive holdout R2 and positive action
permutation gain. All six aggregate paired directions also generalized with
positive train-versus-holdout cosine. The frozen every-mode agreement gate
rejected HalfCheetah because the upper mixed-mode cosine was `-0.2929` and the
lower mixed-mode cosine was `-0.0041`; it rejected Hopper because the lower
standard-mode cosine was `-0.2954`.

Walker selected output-bias RMS `1e-4`. On its untouched validation paths,
frequency-violation merit fell from `0.0554017` to `0.0523669`, a `5.48%`
relative reduction, with zero reward violations. The validation worst
frequency violation changed from `0.05263` to `0.05338`, remaining within the
frozen three-times-baseline funnel.

## Decision

The all-environment preflight is **not supported** (`1/3` cells supported).
The result does support the central diagnosis behind v14.26: direct paired
directions are much more stable than the v14.25 fitted-critic derivative, and
they produce an independently validated Walker improvement. It does not justify
relaxing the frozen gate after outcome access.

The remaining estimator error is localized to within-mode SPSA estimates. Each
path supplies one random Rademacher direction, so the coordinate-median estimate
still contains cross-coordinate terms even with eight roots per mode. The next
development protocol will replace random directions with a balanced orthogonal
Hadamard design and recover each mode gradient from its directional-derivative
system. Fresh role roots and untouched validation remain mandatory.
