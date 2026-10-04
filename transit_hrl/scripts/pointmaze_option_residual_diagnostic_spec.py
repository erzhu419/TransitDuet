"""Frozen diagnostic for the failed Stage112 learned-versus-forecast gate."""
from scripts import pointmaze_option_residual_stage111_spec as source

ROOT = source.ROOT
PERIODS = source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_option_residual_diagnostic_stage112_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_residual_diagnostic.py"
SOURCE_RUN = "pointmaze_option_residual_stage111_full_20261004_r1"
FULL_RUN = "pointmaze_option_residual_train_stage112_full_20261004_r2"
WORKERS = 4
EPISODES_PER_PERIOD = 32
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = 112300


def arguments(root):
    return source.arguments(root, preflight=False)


def roots():
    return (410011,)


def diagnostic_seeds(root):
    roots().index(root)
    base = 112300000 + roots().index(root) * 100000
    return list(range(base + 96001, base + 96001 + EPISODES_PER_PERIOD))


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def checkpoint(root, period, arm):
    return (ROOT / "results" / FULL_RUN / "cells" / f"replicate_{root}" / "final_weights"
        / f"period_{period}_{arm}.pt")


def budget():
    horizon = arguments(roots()[0]).horizon
    episodes = len(PERIODS) * EPISODES_PER_PERIOD * 2
    return {"periods": len(PERIODS), "episodes_per_arm_period": EPISODES_PER_PERIOD,
        "native_episodes": episodes, "native_steps": episodes * horizon,
        "native_lower_calls": episodes * horizon, "checkpoint_loads": len(PERIODS) * 2,
        "return_pairs": len(PERIODS) * EPISODES_PER_PERIOD,
        "bootstrap_draws": len(PERIODS) * 6 * BOOTSTRAP_DRAWS}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "full_run": FULL_RUN,
        "comparison": "learned_final_branch_vs_forecast_final_branch_same_seed_and_noise",
        "frozen": "Stage112_final_weights_and_Stage111_source_no_training_or_checkpoint_write",
        "metrics": ["return_difference", "advice_norm", "residual_correction_norm", "advice_cosine"],
        "selection": "single_root_410011_fixed_before_reading_diagnostic_results",
        "statistics": "32_fresh_pairs_per_period_percentile_bootstrap65536_seed112300",
        "raw_artifacts": "server_only_compact_scalar_pull"}
