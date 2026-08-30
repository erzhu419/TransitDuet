from types import SimpleNamespace

import numpy as np

from scripts import mujoco_v17_11_fractional_reservoir_fir_spec as spec
from scripts.mujoco_v17_8_causal_fir import (
    apply_causal_fir_with_prefix_high_frequency_budget,
)
from scripts.submit_mujoco_v17_11_selection_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v17_11_fractional_reservoir_fir import (
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


def _projection_kwargs() -> dict[str, float | int]:
    return {
        "output_gain": 1.0,
        "upper_action_limit": 1.0,
        "lower_action_limit": 1.0,
        "upper_window": 8,
        "upper_rms_budget": 0.075,
        "power_tolerance": 1e-10,
    }


def test_fraction_zero_is_strict_prefix_projection():
    total, model = _oscillatory_case()
    strict = apply_causal_fir_with_prefix_high_frequency_budget(
        total, model, **_projection_kwargs()
    )
    fractional = apply_causal_fir_with_prefix_high_frequency_budget(
        total,
        model,
        energy_reserve_steps=24,
        energy_borrow_fraction=0.0,
        **_projection_kwargs(),
    )
    np.testing.assert_allclose(fractional["upper"], strict["upper"])
    np.testing.assert_allclose(fractional["lower"], strict["lower"])
    np.testing.assert_allclose(
        fractional["upper_high_frequency_residual"],
        strict["upper_high_frequency_residual"],
    )


def test_fraction_one_is_full_v17_10_reservoir():
    total, model = _oscillatory_case()
    inherited = apply_causal_fir_with_prefix_high_frequency_budget(
        total,
        model,
        energy_reserve_steps=24,
        **_projection_kwargs(),
    )
    explicit = apply_causal_fir_with_prefix_high_frequency_budget(
        total,
        model,
        energy_reserve_steps=24,
        energy_borrow_fraction=1.0,
        **_projection_kwargs(),
    )
    np.testing.assert_allclose(explicit["upper"], inherited["upper"])
    np.testing.assert_allclose(explicit["lower"], inherited["lower"])


def test_fractional_credit_is_repaid_by_certified_horizon():
    total, model = _oscillatory_case()
    kwargs = _projection_kwargs()
    strict = apply_causal_fir_with_prefix_high_frequency_budget(
        total, model, **kwargs
    )
    fractional = apply_causal_fir_with_prefix_high_frequency_budget(
        total,
        model,
        energy_reserve_steps=24,
        energy_borrow_fraction=0.5,
        **kwargs,
    )
    assert fractional["minimum_horizon_certified"]
    strict_early_projection = np.mean(np.square(
        strict["upper"][:2] - strict["raw_upper"][:2]
    ))
    fractional_early_projection = np.mean(np.square(
        fractional["upper"][:2] - fractional["raw_upper"][:2]
    ))
    assert fractional_early_projection < strict_early_projection
    residual = fractional["upper_high_frequency_residual"][:24]
    horizon_power = float(np.mean(np.square(residual)))
    assert horizon_power <= kwargs["upper_rms_budget"] ** 2 + 1e-10


def test_v17_11_grid_and_scheduler_contract_are_frozen():
    configs = candidate_configs()
    assert len(configs) == 40
    assert {row["energy_borrow_fraction"] for row in configs} == set(
        spec.ENERGY_BORROW_FRACTIONS
    )
    args = SimpleNamespace(
        dataset_run_name="v17_8_dataset_test",
        run_name="v17_11_selection_test",
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
