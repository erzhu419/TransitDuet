import numpy as np

from freq_hrl.core.causal_joint_frequency_projector import (
    CausalJointFrequencyProjector,
)
from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (
    responsibility_frequency_powers,
)


def _run(upper, lower, *, mode="prefix_ledger"):
    projector = CausalJointFrequencyProjector(
        upper_window=4,
        lower_window=6,
        upper_rms_budget=0.12,
        lower_rms_budget=0.08,
        budget_mode=mode,
    )
    projector.reset(upper.shape[1])
    rows = [projector.project(u, l) for u, l in zip(upper, lower, strict=True)]
    return rows


def test_fixed_total_is_preserved_when_a_causal_split_exists():
    upper = np.zeros((20, 2), dtype=np.float64)
    lower = np.zeros((20, 2), dtype=np.float64)
    lower[:, 0] = 0.03 * np.sin(np.arange(20))
    rows = _run(upper, lower)
    assert all(row["fixed_total_feasible"] for row in rows)
    assert all(not row["total_action_changed"] for row in rows)
    np.testing.assert_allclose(
        np.stack([row["total"] for row in rows]), upper + lower, atol=1e-10
    )


def test_actor_floor_changes_total_and_satisfies_prefix_budgets():
    upper = np.full((40, 1), 0.9, dtype=np.float64)
    lower = np.full((40, 1), 0.9, dtype=np.float64)
    rows = _run(upper, lower, mode="instantaneous")
    assert any(row["total_action_changed"] for row in rows)
    assert all(row["component_feasible"] for row in rows)
    projected_upper = np.stack([row["upper"] for row in rows])
    projected_lower = np.stack([row["lower"] for row in rows])
    power = responsibility_frequency_powers(
        projected_upper + projected_lower,
        projected_upper,
        upper_window=4,
        lower_window=6,
    )
    assert power[0] <= 0.12**2 + 1e-9
    assert power[1] <= 0.08**2 + 1e-9


def test_future_changes_do_not_change_the_projected_prefix():
    rng = np.random.default_rng(91)
    upper = rng.normal(0.0, 0.2, size=(30, 3))
    lower = rng.normal(0.0, 0.2, size=(30, 3))
    changed_upper = upper.copy()
    changed_lower = lower.copy()
    changed_upper[15:] += 0.7
    changed_lower[15:] -= 0.4
    first = _run(upper, lower)
    second = _run(changed_upper, changed_lower)
    np.testing.assert_allclose(
        np.stack([row["total"] for row in first[:15]]),
        np.stack([row["total"] for row in second[:15]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.stack([row["upper"] for row in first[:15]]),
        np.stack([row["upper"] for row in second[:15]]),
        atol=1e-12,
    )


def test_prefix_ledger_matches_exact_endpoint_frequency_power():
    rng = np.random.default_rng(311)
    upper = rng.uniform(-0.8, 0.8, size=(80, 4))
    lower = rng.uniform(-0.8, 0.8, size=(80, 4))
    rows = _run(upper, lower, mode="prefix_ledger")
    assert all(row["component_feasible"] for row in rows)
    projected_upper = np.stack([row["upper"] for row in rows])
    projected_lower = np.stack([row["lower"] for row in rows])
    upper_power, lower_power = responsibility_frequency_powers(
        projected_upper + projected_lower,
        projected_upper,
        upper_window=4,
        lower_window=6,
    )
    assert upper_power <= 0.12**2 + 1e-8
    assert lower_power <= 0.08**2 + 1e-8
    assert np.isclose(rows[-1]["upper_prefix_power"], upper_power)
    assert np.isclose(rows[-1]["lower_prefix_power"], lower_power)
    assert max(row["reconstruction_error_max"] for row in rows) <= 1e-12
