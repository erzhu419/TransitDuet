import numpy as np

from freq_hrl.core import (
    AffineQuadraticBallProjector,
    CausalJointFrequencyProjector,
    CausalRecedingHorizonJointProjector,
)


def _run(upper, lower, *, horizon=16, forecast_mode="hold"):
    projector = CausalRecedingHorizonJointProjector(
        upper_window=4,
        lower_window=6,
        upper_rms_budget=0.12,
        lower_rms_budget=0.08,
        planning_horizon=horizon,
        forecast_mode=forecast_mode,
    )
    projector.reset(upper.shape[1])
    return [
        projector.project(u, l)
        for u, l in zip(upper, lower, strict=True)
    ]


def test_affine_quadratic_projection_reaches_offset_frequency_ball():
    operator = np.array([[0.5, 0.0], [-0.5, 0.5]])
    offset = np.array([[0.1, -0.2], [0.0, 0.1]])
    projector = AffineQuadraticBallProjector(
        operator, offset, radius_squared=0.01
    )
    values = np.array([[0.8, -0.7], [0.9, 0.6]])
    projected = projector.project(values)
    assert projector.feasible
    assert projector.energy(projected) <= 0.01 + 1e-9
    assert np.linalg.norm(projected - values) > 0.0


def test_smooth_feasible_trace_preserves_current_total():
    time = np.arange(40, dtype=np.float64)
    upper = (0.1 * np.sin(0.03 * time))[:, None]
    lower = (0.02 * np.sin(0.8 * time))[:, None]
    rows = _run(upper, lower)
    assert all(row["fixed_total_forecast_feasible"] for row in rows)
    assert all(not row["total_action_changed"] for row in rows)
    np.testing.assert_allclose(
        np.stack([row["total"] for row in rows]), upper + lower, atol=1e-8
    )


def test_receding_horizon_changes_less_than_instantaneous_projection():
    upper = np.concatenate((
        np.zeros((10, 1), dtype=np.float64),
        np.full((30, 1), 0.5, dtype=np.float64),
    ))
    lower = np.concatenate((
        np.zeros((10, 1), dtype=np.float64),
        np.full((30, 1), 0.05, dtype=np.float64),
    ))
    horizon_rows = _run(upper, lower, horizon=16)
    instantaneous = CausalJointFrequencyProjector(
        upper_window=4,
        lower_window=6,
        upper_rms_budget=0.12,
        lower_rms_budget=0.08,
        budget_mode="instantaneous",
    )
    instantaneous.reset(1)
    instant_rows = [
        instantaneous.project(u, l)
        for u, l in zip(upper, lower, strict=True)
    ]
    horizon_rms = np.sqrt(np.mean(np.square(np.stack([
        row["total_correction"] for row in horizon_rows
    ]))))
    instant_rms = np.sqrt(np.mean(np.square(np.stack([
        row["total_correction"] for row in instant_rows
    ]))))
    assert horizon_rms < instant_rms
    assert horizon_rms <= 1e-10
    assert all(not row["total_action_changed"] for row in horizon_rows)


def test_receding_horizon_changes_persistently_infeasible_total():
    upper = np.full((30, 1), 0.9, dtype=np.float64)
    lower = np.full((30, 1), 0.9, dtype=np.float64)
    rows = _run(upper, lower, horizon=16)
    correction = np.stack([row["total_correction"] for row in rows])
    assert np.sqrt(np.mean(np.square(correction))) > 0.1
    assert any(row["total_action_changed"] for row in rows)


def test_future_proposal_changes_do_not_change_realized_prefix():
    rng = np.random.default_rng(722)
    upper = rng.normal(0.0, 0.15, size=(35, 2))
    lower = rng.normal(0.0, 0.15, size=(35, 2))
    changed_upper = upper.copy()
    changed_lower = lower.copy()
    changed_upper[18:] += 0.6
    changed_lower[18:] -= 0.3
    first = _run(upper, lower, forecast_mode="damped_velocity")
    second = _run(
        changed_upper, changed_lower, forecast_mode="damped_velocity"
    )
    np.testing.assert_allclose(
        np.stack([row["total"] for row in first[:18]]),
        np.stack([row["total"] for row in second[:18]]),
        atol=1e-10,
    )
