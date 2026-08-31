import numpy as np
import pytest

from freq_hrl.core import (
    CausalRecedingHorizonJointProjector,
    CausalTerminalReserveProjector,
)


UPPER_BUDGET = 0.12
LOWER_BUDGET = 0.08


def _projector():
    return CausalTerminalReserveProjector(
        upper_window=4,
        lower_window=6,
        upper_rms_budget=UPPER_BUDGET,
        lower_rms_budget=LOWER_BUDGET,
    )


def _run(upper, lower):
    projector = _projector()
    projector.reset(upper.shape[1])
    return [
        projector.project(u, l)
        for u, l in zip(upper, lower, strict=True)
    ]


def _prefix_powers(upper, lower, *, upper_window=4, lower_window=6):
    upper_energy = 0.0
    lower_energy = 0.0
    dimension = upper.shape[1]
    powers = []
    for index in range(upper.shape[0]):
        upper_start = max(0, index - upper_window + 1)
        lower_start = max(0, index - lower_window + 1)
        upper_mean = np.mean(upper[upper_start:index + 1], axis=0)
        lower_mean = np.mean(lower[lower_start:index + 1], axis=0)
        upper_energy += float(np.sum(np.square(upper[index] - upper_mean)))
        lower_energy += float(np.sum(np.square(lower_mean)))
        denominator = float((index + 1) * dimension)
        powers.append((
            upper_energy / denominator,
            lower_energy / denominator,
        ))
    return np.asarray(powers)


def test_smooth_certified_trace_preserves_total_action():
    time = np.arange(40, dtype=np.float64)
    upper = (0.1 * np.sin(0.03 * time))[:, None]
    lower = (0.01 * np.sin(0.8 * time))[:, None]
    rows = _run(upper, lower)
    assert all(row["fixed_total_feasible"] for row in rows)
    assert all(row["terminal_certificate_feasible"] for row in rows)
    assert all(not row["total_action_changed"] for row in rows)
    np.testing.assert_allclose(
        np.stack([row["total"] for row in rows]),
        upper + lower,
        atol=1e-9,
    )


def test_every_realized_prefix_respects_both_frequency_budgets():
    rng = np.random.default_rng(123)
    upper = rng.uniform(-0.9, 0.9, size=(80, 3))
    lower = rng.uniform(-0.9, 0.9, size=(80, 3))
    rows = _run(upper, lower)
    projected_upper = np.stack([row["upper"] for row in rows])
    projected_lower = np.stack([row["lower"] for row in rows])
    powers = _prefix_powers(projected_upper, projected_lower)
    assert np.max(powers[:, 0]) <= UPPER_BUDGET ** 2 + 2e-8
    assert np.max(powers[:, 1]) <= LOWER_BUDGET ** 2 + 2e-8
    np.testing.assert_allclose(
        powers[:, 0],
        [row["upper_prefix_power"] for row in rows],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        powers[:, 1],
        [row["lower_prefix_power"] for row in rows],
        atol=1e-12,
    )
    assert all(row["upper_terminal_reserve_min_margin"] >= -2e-8 for row in rows)
    assert all(row["lower_terminal_reserve_min_margin"] >= -2e-8 for row in rows)


def test_fixed_total_infeasibility_changes_physical_action_but_stays_certified():
    upper = np.full((20, 1), 0.9, dtype=np.float64)
    lower = np.full((20, 1), 0.9, dtype=np.float64)
    rows = _run(upper, lower)
    assert any(not row["fixed_total_feasible"] for row in rows)
    assert any(row["total_action_changed"] for row in rows)
    assert all(row["terminal_certificate_feasible"] for row in rows)
    assert max(row["reconstruction_error_max"] for row in rows) <= 1e-12


def test_shifted_backup_is_feasible_at_the_next_replan():
    projector = _projector()
    projector.reset(2)
    first = projector.project(np.array([0.8, -0.7]), np.array([0.8, 0.7]))
    second = projector.project(first["upper"], np.zeros(2, dtype=np.float64))
    assert second["recursive_backup_feasible_at_entry"]
    assert second["fixed_total_feasible"]
    assert second["terminal_certificate_feasible"]
    assert not second["total_action_changed"]
    np.testing.assert_allclose(second["total"], first["upper"], atol=1e-10)


def test_policy_context_is_fixed_size_and_contains_only_realized_history():
    projector = _projector()
    projector.reset(2)
    initial_actions, initial_scalars = projector.policy_context
    assert len(initial_actions) == (4 - 1) + (6 - 1)
    assert len(initial_scalars) == 5
    assert all(np.array_equal(value, np.zeros(2)) for value in initial_actions)
    row = projector.project(np.array([0.3, -0.2]), np.zeros(2))
    actions, scalars = projector.policy_context
    np.testing.assert_allclose(actions[2], row["upper"], atol=1e-12)
    np.testing.assert_allclose(actions[-1], row["lower"], atol=1e-12)
    assert scalars[0] == pytest.approx(np.log(2.0))
    assert scalars[3] == pytest.approx(1.0 / 3.0)
    assert scalars[4] == pytest.approx(1.0 / 5.0)


def test_observe_executed_matches_raw_prefix_audit_without_projection():
    projector = _projector()
    projector.reset(2)
    upper = np.asarray([[0.0, 0.0], [0.8, -0.8], [-0.8, 0.8]])
    lower = np.asarray([[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]])
    rows = [
        projector.observe_executed(u, l)
        for u, l in zip(upper, lower, strict=True)
    ]
    powers = _prefix_powers(upper, lower)
    np.testing.assert_allclose(
        [row["upper_prefix_power"] for row in rows],
        powers[:, 0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [row["lower_prefix_power"] for row in rows],
        powers[:, 1],
        atol=1e-12,
    )
    assert not rows[-1]["prefix_budget_feasible"]
    action_contexts, scalar_contexts = projector.policy_context
    np.testing.assert_allclose(action_contexts[2], upper[-1], atol=1e-12)
    np.testing.assert_allclose(action_contexts[-1], lower[-1], atol=1e-12)
    assert scalar_contexts[1] > 1.0
    assert scalar_contexts[2] > 1.0


def test_backup_tail_flushes_both_finite_memory_filters():
    rng = np.random.default_rng(411)
    upper = rng.uniform(-0.7, 0.7, size=(12, 2))
    lower = rng.uniform(-0.7, 0.7, size=(12, 2))
    rows = _run(upper, lower)
    realized_upper = np.stack([row["upper"] for row in rows])
    realized_lower = np.stack([row["lower"] for row in rows])
    upper_tail = np.repeat(realized_upper[-1][None, :], 5, axis=0)
    lower_tail = np.zeros((7, 2), dtype=np.float64)
    full_upper = np.concatenate((realized_upper, upper_tail), axis=0)
    full_lower = np.concatenate((realized_lower, lower_tail), axis=0)
    upper_index = full_upper.shape[0] - 1
    lower_index = full_lower.shape[0] - 1
    upper_residual = full_upper[upper_index] - np.mean(
        full_upper[upper_index - 3:upper_index + 1], axis=0
    )
    lower_residual = np.mean(
        full_lower[lower_index - 5:lower_index + 1], axis=0
    )
    np.testing.assert_allclose(upper_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(lower_residual, 0.0, atol=1e-12)


def test_future_proposals_cannot_change_the_realized_prefix():
    rng = np.random.default_rng(722)
    upper = rng.normal(0.0, 0.15, size=(35, 2))
    lower = rng.normal(0.0, 0.15, size=(35, 2))
    changed_upper = upper.copy()
    changed_lower = lower.copy()
    changed_upper[18:] += 0.6
    changed_lower[18:] -= 0.3
    first = _run(upper, lower)
    second = _run(changed_upper, changed_lower)
    np.testing.assert_allclose(
        np.stack([row["upper"] for row in first[:18]]),
        np.stack([row["upper"] for row in second[:18]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.stack([row["total"] for row in first[:18]]),
        np.stack([row["total"] for row in second[:18]]),
        atol=1e-12,
    )


def test_terminal_reserve_closes_the_v18_moving_horizon_debt_failure():
    rng = np.random.default_rng(8401)
    upper = rng.uniform(-0.9, 0.9, size=(8, 1))
    lower = rng.uniform(-0.9, 0.9, size=(8, 1))
    old = CausalRecedingHorizonJointProjector(
        upper_window=4,
        lower_window=6,
        upper_rms_budget=UPPER_BUDGET,
        lower_rms_budget=LOWER_BUDGET,
        planning_horizon=8,
        forecast_mode="hold",
    )
    old.reset(1)
    old_rows = [
        old.project(u, l)
        for u, l in zip(upper, lower, strict=True)
    ]
    new_rows = _run(upper, lower)
    assert max(row["upper_prefix_power"] for row in old_rows) > UPPER_BUDGET ** 2
    assert max(row["upper_prefix_power"] for row in new_rows) <= (
        UPPER_BUDGET ** 2 + 2e-8
    )
    assert all(row["terminal_certificate_feasible"] for row in new_rows)


def test_configuration_and_reset_contracts_are_fail_closed():
    with pytest.raises(ValueError, match="windows"):
        CausalTerminalReserveProjector(upper_window=1)
    with pytest.raises(ValueError, match="positive"):
        CausalTerminalReserveProjector(lower_rms_budget=0.0)
    projector = _projector()
    with pytest.raises(RuntimeError, match="reset"):
        projector.project(np.zeros(1), np.zeros(1))
    with pytest.raises(RuntimeError, match="reset"):
        projector.observe_executed(np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="positive"):
        projector.reset(0)
