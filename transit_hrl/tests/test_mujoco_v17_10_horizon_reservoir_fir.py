from types import SimpleNamespace

import numpy as np

from scripts import mujoco_v17_10_horizon_reservoir_fir_spec as spec
from scripts.mujoco_v17_8_causal_fir import (
    apply_causal_fir_with_prefix_high_frequency_budget,
)
from scripts.submit_mujoco_v17_10_selection_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v17_10_horizon_reservoir_fir import (
    candidate_configs,
)


def _oscillatory_case(length: int = 40):
    time = np.arange(length, dtype=np.float64)
    total = np.stack(
        (0.4 * np.sin(1.7 * time), 0.4 * np.cos(1.3 * time)), axis=1
    )
    model = {
        "window": 1,
        "action_dimension": 2,
        "coefficients": np.array([1.8 * np.eye(2)]),
    }
    return total, model


def test_horizon_reservoir_keeps_endpoint_budget_and_reduces_projection():
    total, model = _oscillatory_case()
    kwargs = {
        "output_gain": 1.0,
        "upper_action_limit": 1.0,
        "lower_action_limit": 1.0,
        "upper_window": 8,
        "upper_rms_budget": 0.075,
        "power_tolerance": 1e-10,
    }
    strict = apply_causal_fir_with_prefix_high_frequency_budget(
        total, model, **kwargs
    )
    reservoir = apply_causal_fir_with_prefix_high_frequency_budget(
        total, model, energy_reserve_steps=24, **kwargs
    )
    assert reservoir["minimum_horizon_certified"]
    assert reservoir["prefix_upper_power_max"] >= 0.075 ** 2
    endpoint_power = np.mean(np.square(
        reservoir["upper_high_frequency_residual"]
    ))
    assert endpoint_power <= 0.075 ** 2 + 1e-10
    strict_early_projection = np.mean(np.square(
        strict["upper"][:2] - strict["raw_upper"][:2]
    ))
    reservoir_early_projection = np.mean(np.square(
        reservoir["upper"][:2] - reservoir["raw_upper"][:2]
    ))
    assert reservoir_early_projection < strict_early_projection


def test_horizon_reservoir_fails_certification_on_short_trajectory():
    total, model = _oscillatory_case(length=12)
    result = apply_causal_fir_with_prefix_high_frequency_budget(
        total,
        model,
        output_gain=1.0,
        upper_action_limit=1.0,
        lower_action_limit=1.0,
        upper_window=8,
        upper_rms_budget=0.075,
        power_tolerance=1e-10,
        energy_reserve_steps=16,
    )
    assert not result["minimum_horizon_certified"]


def test_v17_10_grid_and_scheduler_contract_are_frozen():
    assert len(candidate_configs()) == 32
    args = SimpleNamespace(
        dataset_run_name="v17_8_dataset_test",
        run_name="v17_10_selection_test",
        python_executable="python3",
        cpu=8,
        ram_mb=8192,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["cpu"] == 8
    assert task["allow_cpu_training"]
    assert ".server_artifacts" in task["stage_excludes"]
