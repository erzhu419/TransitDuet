"""Fresh-root upper-only sample-consistency development, fixed before execution."""

PROTOCOL = "mujoco_v25_sample_consistent_upper_development_v1"
CORE_PROTOCOL = "freq_hrl_mujoco_shared_core_v25_sample_consistent_upper_training"
ALGORITHM_REVISION = "pending_source_commit"
ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAIN_MODES = ("standard", "low_frequency", "high_frequency", "mixed")
EVAL_MODES = (*TRAIN_MODES, "ood_chirp")

# Drawn once with NumPy Generator(250092); no overlap with prior MuJoCo specs.
OPTIMIZER_SEEDS = (4218603422, 1046322914, 3702952031, 374876713)
TRAIN_SEEDS = (1085748914, 2284490494, 1653160040, 208812861)
SELECTION_SEEDS = (3299287598, 649595808, 2020009690, 3815444903)
EVAL_SEEDS = (16055742, 2272836666, 2028026568, 2459608210, 2605446795, 3184544235, 2161880964, 2310225186)
PREFLIGHT_SEEDS = dict(optimizer=(25009401,), train=(25009402,), selection=(25009403,), evaluation=(25009404,))
ZERO = "zero_consistency"
CONTROL = "raw_mean"
DIAGNOSTIC = "raw_sample"
CANDIDATE = "action_sample"
ARMS = (ZERO, CONTROL, DIAGNOSTIC, CANDIDATE)


def options(arm, *, preflight=False):
    active = arm != ZERO
    return {
        "method": "freq_hrl", "disturbance-mode": "standard",
        "training-disturbance-modes": TRAIN_MODES, "evaluation-disturbance-modes": EVAL_MODES,
        "train-seeds": PREFLIGHT_SEEDS["train"] if preflight else TRAIN_SEEDS,
        "selection-seeds": PREFLIGHT_SEEDS["selection"] if preflight else SELECTION_SEEDS,
        "eval-seeds": PREFLIGHT_SEEDS["evaluation"] if preflight else EVAL_SEEDS,
        "steps": 64 if preflight else 512, "episode-horizon": 128 if preflight else 1000,
        "iterations": 8 if preflight else 512, "upper-period": 16, "hidden-dim": 64,
        "learning-rate": 3e-4, "ppo-clip-ratio": 0.1,
        "upper-projection-consistency-coef": 0.1 if active else 0.0,
        "lower-projection-consistency-coef": 0.1 if active else 0.0,
        "upper-projection-target-aggregation": "decision_time",
        "upper-projection-consistency-objective": arm if active else "raw_mean",
        "projection-consistency-update-mode": "scalarized",
        "projection-consistency-weighting": "uniform",
        "projection-consistency-training-schedule": "delayed_linear" if active else "constant",
        "projection-consistency-warmup-fraction": 0.5 if active else 0.0,
        "projection-consistency-ramp-fraction": 0.25 if active else 0.0,
        "terminal-reserve-context": True, "terminal-reserve-projection": True,
        "terminal-reserve-upper-window": 8, "terminal-reserve-lower-window": 32,
        "lower-lf-rms-budget": 0.0475, "upper-hf-rms-budget": 0.075,
        "upper-action-scale": 1.0, "lower-action-scale": 1.0,
        "upper-action-decoder-mode": "hold", "responsibility-mode": "additive",
        "lower-action-router-mode": "direct", "leakage-constraint-scope": "responsibility",
        "leakage-cost-mode": "power_excess", "upper-constraint-mode": "static_reward_penalty",
        "upper-hf-penalty-coef": 0.0, "upper-dual-lr": 0.0, "lower-dual-lr": 0.0,
        "upper-constraint-update-mode": "scalarized", "lower-constraint-update-mode": "scalarized",
        "checkpoint-selection-mode": "crossed_conditions", "checkpoint-score-mode": "mean_reward",
        "checkpoint-smoothing-window": 1, "checkpoint-min-delta": 0.0,
        "checkpoint-minimum-iteration": 3 if preflight else 383,
        "checkpoint-evaluation-interval": 2 if preflight else 16,
        "control-protocol-version": CORE_PROTOCOL, "code-revision": ALGORITHM_REVISION,
    }


CONTRACT = {
    "evidence_role": "development_only_not_confirmatory",
    "primary_candidate": CANDIDATE,
    "primary_control": CONTROL,
    "diagnostic_only": DIAGNOSTIC,
    "isolated_change": "upper fixed-residual consistency; lower remains raw-mean; no variance-head update",
    "selection": "retain shared per-step checkpoint scoring to isolate loss; evaluate episode return",
    "sample_offset": "rollout action minus pre-update mean, detached once for all PPO epochs",
    "reward_gate": "action_sample wins >=8/12 roots and >=2/4 per environment vs raw_mean; mean improves in >=2 environments; no >5% regression vs raw_mean or zero in any environment",
    "correction_gate": "component and total correction each improve >=5% vs zero in >=2 environments; neither regresses >5% vs raw_mean or zero in any environment; Hopper total <=0.25",
    "validity_gate": "all arms: finite metrics, zero certificate violations, fallback <=0.05 and prefix power within budgets+1e-8; paired capacity and path identity",
    "loss_units": "raw and bounded-action MSE are not comparable and are not adoption gates",
    "decision": "advance action_sample only if every gate passes; raw_sample cannot be selected post hoc",
    "preflight": "12 short cells use separate roots; tests execution and finite active updates, never reward selection",
    "limitation": "tanh derivatives rescale gradients; controlled independent Gaussian diagnostic is not an on-policy performance result",
}
