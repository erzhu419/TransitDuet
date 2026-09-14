from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from freq_hrl.core import CausalTerminalReserveProjector


class _ReferenceProjector(CausalTerminalReserveProjector):
    """Pre-optimization implementation retained only for differential tests."""

    def _project_ball_intersection(
        self,
        proposed: np.ndarray,
        *,
        balls: list[tuple[np.ndarray, float]],
        low: np.ndarray,
        high: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        projectors = [
            self._ball_projector_reference(center, radius)
            for center, radius in balls
        ]
        projectors.append(lambda values: np.clip(values, low, high))
        return self._dykstra_reference(
            np.clip(proposed, low, high),
            projectors,
        )

    def _dykstra_reference(
        self,
        start: np.ndarray,
        projectors: list[Callable[[np.ndarray], np.ndarray]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        values = np.asarray(start, dtype=np.float64).copy()
        residuals = [np.zeros_like(values) for _ in projectors]
        converged = False
        iteration = 0
        for iteration in range(1, self.maximum_projection_iterations + 1):
            previous = values.copy()
            for index, projector in enumerate(projectors):
                shifted = values + residuals[index]
                projected = np.asarray(projector(shifted), dtype=np.float64)
                residuals[index] = shifted - projected
                values = projected
            if float(np.max(np.abs(values - previous))) <= self.projection_tolerance:
                converged = True
                break
        return values, {"converged": converged, "iterations": iteration}

    @staticmethod
    def _ball_projector_reference(
        center: np.ndarray,
        radius: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        origin = np.asarray(center, dtype=np.float64)
        bound = float(radius)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("ball radius must be finite and non-negative")

        def project(values: np.ndarray) -> np.ndarray:
            point = np.asarray(values, dtype=np.float64)
            delta = point - origin
            norm = float(np.linalg.norm(delta))
            if norm <= bound or norm <= 1e-30:
                return point.copy()
            return origin + (bound / norm) * delta

        return project


def _projector(projector_type, dimension: int):
    projector = projector_type(
        upper_window=8,
        lower_window=32,
        upper_rms_budget=0.075,
        lower_rms_budget=0.0475,
    )
    projector.reset(dimension)
    return projector


def _assert_equivalent(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if key == "projection_iterations":
            continue
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=1e-9,
                atol=2e-8,
                err_msg=key,
            )
        elif isinstance(expected_value, (bool, np.bool_)):
            assert actual_value == expected_value, key
        elif isinstance(expected_value, (float, np.floating)):
            assert np.isclose(
                actual_value,
                expected_value,
                rtol=1e-9,
                atol=2e-8,
            ), key
        else:
            assert actual_value == expected_value, key


@pytest.mark.parametrize("dimension, seed", [(1, 9101), (3, 9103), (6, 9106)])
def test_optimized_projection_matches_reference_trajectory(dimension, seed):
    rng = np.random.default_rng(seed)
    steps = 48
    phase = np.arange(steps, dtype=np.float64)[:, None]
    offsets = np.arange(dimension, dtype=np.float64)[None, :]
    upper = 0.55 * np.sin(0.17 * phase + 0.3 * offsets)
    upper += rng.normal(0.0, 0.24, size=(steps, dimension))
    lower = 0.45 * np.sin(0.83 * phase + 0.2 * offsets)
    lower += rng.normal(0.0, 0.27, size=(steps, dimension))
    upper = np.clip(upper, -0.95, 0.95)
    lower = np.clip(lower, -0.95, 0.95)

    optimized = _projector(CausalTerminalReserveProjector, dimension)
    reference = _projector(_ReferenceProjector, dimension)
    for index, (upper_action, lower_action) in enumerate(
        zip(upper, lower, strict=True)
    ):
        if index % 11 == 0:
            _assert_equivalent(
                optimized.preview(upper_action, lower_action),
                reference.preview(upper_action, lower_action),
            )
        _assert_equivalent(
            optimized.project(upper_action, lower_action),
            reference.project(upper_action, lower_action),
        )


def test_containing_balls_are_removed_without_reordering_active_constraints():
    balls = [
        (np.asarray([0.0, 0.0]), 0.25),
        (np.asarray([0.0, 0.0]), 0.50),
        (np.asarray([0.1, 0.0]), 0.40),
        (np.asarray([1.0, 0.0]), 0.20),
    ]
    retained = CausalTerminalReserveProjector._nonredundant_balls(balls)
    assert len(retained) == 2
    np.testing.assert_array_equal(retained[0][0], balls[0][0])
    np.testing.assert_array_equal(retained[1][0], balls[3][0])
    assert [radius for _, radius in retained] == [0.25, 0.20]


def test_single_active_ball_solution_skips_iterative_projection():
    class FailIfDykstraRuns(CausalTerminalReserveProjector):
        def _dykstra_ball_box(self, *args, **kwargs):
            raise AssertionError("analytic single-ball solution was missed")

    projector = FailIfDykstraRuns()
    projected, metadata = projector._project_ball_intersection(
        np.asarray([0.9, 0.0]),
        balls=[
            (np.asarray([0.0, 0.0]), 0.5),
            (np.asarray([0.1, 0.0]), 0.5),
        ],
        low=np.asarray([-1.0, -1.0]),
        high=np.asarray([1.0, 1.0]),
    )
    np.testing.assert_allclose(projected, [0.5, 0.0], atol=1e-14)
    assert metadata == {"converged": True, "iterations": 1}


def test_analytic_path_resolves_a_reference_iteration_limit_case():
    dimension = 3
    seed = 9332
    rng = np.random.default_rng(seed)
    steps = 30
    phase = np.arange(48, dtype=np.float64)[:, None]
    offsets = np.arange(dimension, dtype=np.float64)[None, :]
    upper = 0.55 * np.sin(0.17 * phase + 0.3 * offsets)
    upper += rng.normal(0.0, 0.24, size=(48, dimension))
    lower = 0.45 * np.sin(0.83 * phase + 0.2 * offsets)
    lower += rng.normal(0.0, 0.27, size=(48, dimension))
    upper = np.clip(upper, -0.95, 0.95)
    lower = np.clip(lower, -0.95, 0.95)

    optimized = _projector(CausalTerminalReserveProjector, dimension)
    reference = _projector(_ReferenceProjector, dimension)
    for index in range(steps - 1):
        actual = optimized.project(upper[index], lower[index])
        expected = reference.project(upper[index], lower[index])
        _assert_equivalent(actual, expected)

    upper_action = upper[steps - 1]
    lower_action = lower[steps - 1]
    actual = optimized.preview(upper_action, lower_action)
    expected = reference.preview(upper_action, lower_action)
    assert actual["projection_converged"]
    assert not expected["projection_converged"]
    assert expected["projection_iterations"] == 512
    assert actual["terminal_certificate_feasible"]
    actual_distance = float(
        np.sum(np.square(actual["upper"] - upper_action))
        + np.sum(np.square(actual["lower"] - lower_action))
    )
    expected_distance = float(
        np.sum(np.square(expected["upper"] - upper_action))
        + np.sum(np.square(expected["lower"] - lower_action))
    )
    assert actual_distance <= expected_distance + 1e-10
