import json

import numpy as np

from freq_hrl.core import LeakageRegularizer
from freq_hrl.core.leakage import causal_rolling_mean
from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (
    causal_rolling_operator,
    responsibility_frequency_powers,
    solve_full_horizon_responsibility_oracle,
)


def test_rolling_operator_matches_registered_causal_filter():
    rng = np.random.default_rng(17601)
    values = rng.normal(size=(47, 3))
    operator = causal_rolling_operator(values.shape[0], 8)
    np.testing.assert_allclose(
        operator @ values,
        causal_rolling_mean(values, 8),
        rtol=0.0,
        atol=1e-14,
    )


def test_oracle_matches_leakage_regularizer_metrics_and_reconstructs():
    rng = np.random.default_rng(17603)
    total = rng.uniform(-0.8, 0.8, size=(48, 2))
    result = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=0.25,
        lower_rms_budget=0.25,
        multiplier_bisection_steps=10,
    )
    metrics = LeakageRegularizer(
        upper_hf_window=8, lower_lf_window=32
    ).compute(result.upper, result.lower)
    assert abs(result.upper_power - metrics["UpperHFPowerAbs"]) <= 1e-14
    assert abs(result.lower_power - metrics["LowerLFDriftAbs"]) <= 1e-14
    np.testing.assert_allclose(result.upper + result.lower, total, atol=1e-14)
    assert result.bound_violation_max <= 1e-12
    assert result.reconstruction_error_max <= 1e-14
    assert result.solver_optimality_max <= 1e-7
    assert result.kkt_residual_inf <= 1e-6


def test_zero_trace_is_jointly_feasible():
    result = solve_full_horizon_responsibility_oracle(
        np.zeros((32, 2)),
        upper_rms_budget=0.01,
        lower_rms_budget=0.01,
    )
    assert result.status == "joint_frequency_budgets_feasible"
    assert result.joint_feasible
    assert result.upper_power <= 1e-30
    assert result.lower_power == 0.0


def test_fixed_component_box_certifies_lower_floor():
    total = np.full((40, 1), 2.0)
    result = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=0.01,
        lower_rms_budget=0.5,
    )
    assert result.status == "lower_budget_infeasible_at_upper_constrained_floor"
    assert result.upper_constraint_feasible
    assert not result.joint_feasible
    np.testing.assert_allclose(result.upper, 1.0)
    np.testing.assert_allclose(result.lower, 1.0)
    assert result.upper_power <= 1e-30
    assert result.lower_power == 1.0
    assert result.kkt_residual_inf == 0.0


def test_fixed_alternating_components_certify_upper_infeasibility():
    total = np.where(np.arange(40)[:, None] % 2 == 0, 2.0, -2.0)
    result = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=0.05,
        lower_rms_budget=1.0,
    )
    assert result.status == "upper_budget_physically_infeasible"
    assert not result.upper_constraint_feasible
    assert not result.joint_feasible
    assert result.minimum_upper_power > result.upper_power_budget
    json.dumps(result.summary(), allow_nan=False)


def test_oracle_dominates_any_upper_feasible_reference_on_lower_power():
    rng = np.random.default_rng(17605)
    total = rng.uniform(-0.7, 0.7, size=(56, 2))
    reference_upper = 0.5 * total
    reference_upper_power, reference_lower_power = (
        responsibility_frequency_powers(total, reference_upper)
    )
    result = solve_full_horizon_responsibility_oracle(
        total,
        upper_rms_budget=np.sqrt(reference_upper_power) + 1e-5,
        lower_rms_budget=1.0,
        multiplier_bisection_steps=12,
    )
    assert result.upper_power <= result.upper_power_budget + 1e-8
    assert result.lower_power <= reference_lower_power + 1e-8
