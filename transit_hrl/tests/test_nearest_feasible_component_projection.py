import numpy as np

from freq_hrl.experiments.mujoco.nearest_feasible_component_projection import (
    project_nearest_feasible_components,
)


def test_feasible_reference_is_an_exact_fixed_point():
    upper = np.zeros((24, 2), dtype=np.float64)
    lower = np.tile(np.array([[0.02, -0.01]]), (24, 1))
    result = project_nearest_feasible_components(
        upper,
        lower,
        upper_rms_budget=0.075,
        lower_rms_budget=0.0475,
    )
    assert result.feasible
    np.testing.assert_allclose(result.upper, upper, atol=1e-12)
    np.testing.assert_allclose(result.lower, lower, atol=1e-12)
    assert result.component_correction_rms == 0.0
    assert result.total_action_correction_rms == 0.0


def test_infeasible_constant_lower_is_projected_to_lpf_ball():
    upper = np.zeros((1, 1), dtype=np.float64)
    lower = np.full((1, 1), 0.20, dtype=np.float64)
    result = project_nearest_feasible_components(
        upper,
        lower,
        upper_rms_budget=0.075,
        lower_rms_budget=0.05,
    )
    assert result.feasible
    assert result.upper_power <= 0.075 ** 2 + 1e-9
    assert result.lower_power <= 0.05 ** 2 + 1e-9
    np.testing.assert_allclose(result.upper, 0.0, atol=1e-9)
    np.testing.assert_allclose(result.lower, 0.05, atol=2e-8)
    np.testing.assert_allclose(
        result.total_action_correction_rms, 0.15, atol=2e-8
    )


def test_optional_total_box_is_enforced_without_breaking_frequency_budgets():
    upper = np.full((24, 2), 0.70, dtype=np.float64)
    lower = np.full((24, 2), 0.55, dtype=np.float64)
    result = project_nearest_feasible_components(
        upper,
        lower,
        upper_rms_budget=0.075,
        lower_rms_budget=0.40,
        include_total_action_box=True,
        total_action_limit=1.0,
    )
    assert result.feasible
    assert np.max(np.abs(result.upper + result.lower)) <= 1.0 + 1e-9
    assert result.total_action_bound_violation_max <= 1e-9
    assert result.upper_power <= 0.075 ** 2 + 1e-9
    assert result.lower_power <= 0.40 ** 2 + 1e-9


def test_projection_is_deterministic():
    rng = np.random.default_rng(17)
    upper = rng.normal(0.0, 0.25, size=(40, 3))
    lower = rng.normal(0.08, 0.20, size=(40, 3))
    kwargs = {
        "upper_rms_budget": 0.075,
        "lower_rms_budget": 0.0475,
        "include_total_action_box": True,
    }
    first = project_nearest_feasible_components(upper, lower, **kwargs)
    second = project_nearest_feasible_components(upper, lower, **kwargs)
    np.testing.assert_array_equal(first.upper, second.upper)
    np.testing.assert_array_equal(first.lower, second.lower)
    assert first.summary() == second.summary()
