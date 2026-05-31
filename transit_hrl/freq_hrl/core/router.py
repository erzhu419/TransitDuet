"""Frequency responsibility router.

The router is the key algorithmic boundary in Freq-HRL: low-frequency and
compressed high-frequency summaries go to the upper planner; local high and
mid-frequency residual context goes to the lower controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    return value


@dataclass
class FrequencyRouter:
    """Apply upper/lower information masks for Freq-HRL."""

    upper_forbidden_keys: tuple[str, ...] = (
        "x_high_sequence",
        "x_high_raw_sequence",
        "x_high_local_station_vector",
        "raw_high_frequency",
    )
    lower_forbidden_keys: tuple[str, ...] = (
        "x_low_forecast_full",
        "x_low_forecast_horizon",
        "future_low",
        "high_level_value",
    )
    upper_extra_keys: tuple[str, ...] = field(default_factory=tuple)
    lower_extra_keys: tuple[str, ...] = field(default_factory=tuple)

    def upper_view(
        self,
        freq_features: Mapping[str, Any],
        z_upper: Mapping[str, Any] | np.ndarray | None = None,
        promotion: Mapping[str, Any] | None = None,
        leakage_feedback: Any | None = None,
    ) -> dict[str, Any]:
        state = {
            "z_upper": _copy_value(z_upper) if z_upper is not None else {},
            "x_low": _copy_value(freq_features.get("x_low", 0.0)),
            "x_low_slope": _copy_value(freq_features.get("x_low_slope", 0.0)),
            "x_low_forecast": _copy_value(freq_features.get("x_low_forecast", 0.0)),
            "x_low_uncertainty": _copy_value(freq_features.get("x_low_uncertainty", 0.0)),
            "x_mid_energy": _copy_value(freq_features.get("x_mid_energy", 0.0)),
            "x_high_energy": _copy_value(freq_features.get("x_high_energy", 0.0)),
            "x_high_persistence": _copy_value(freq_features.get("x_high_persistence", 0.0)),
            "promotion": dict(promotion or freq_features.get("promotion", {}) or {}),
            "leakage_feedback": _copy_value(
                leakage_feedback if leakage_feedback is not None else freq_features.get("leakage_feedback", 0.0)
            ),
        }
        for key in self.upper_extra_keys:
            if key in freq_features and key not in self.upper_forbidden_keys:
                state[key] = _copy_value(freq_features[key])
        self.assert_upper_contract(state)
        return state

    def lower_view(
        self,
        freq_features: Mapping[str, Any],
        z_lower: Mapping[str, Any] | np.ndarray | None = None,
        current_plan: Mapping[str, Any] | np.ndarray | None = None,
        target_error: Any | None = None,
    ) -> dict[str, Any]:
        state = {
            "z_lower": _copy_value(z_lower) if z_lower is not None else {},
            "current_plan": _copy_value(current_plan) if current_plan is not None else {},
            "target_error": _copy_value(target_error if target_error is not None else freq_features.get("target_error", 0.0)),
            "x_high": _copy_value(freq_features.get("x_high", 0.0)),
            "x_high_delta": _copy_value(freq_features.get("x_high_delta", 0.0)),
            "x_high_energy": _copy_value(freq_features.get("x_high_energy", 0.0)),
            "x_high_persistence": _copy_value(freq_features.get("x_high_persistence", 0.0)),
            "x_mid": _copy_value(freq_features.get("x_mid", 0.0)),
            "shock_age": _copy_value(freq_features.get("shock_age", 0.0)),
        }
        for key in self.lower_extra_keys:
            if key in freq_features and key not in self.lower_forbidden_keys:
                state[key] = _copy_value(freq_features[key])
        self.assert_lower_contract(state)
        return state

    def assert_upper_contract(self, state: Mapping[str, Any]) -> None:
        for key in self.upper_forbidden_keys:
            assert key not in state, f"upper state leaked forbidden key {key!r}"
        assert "x_high" not in state, "upper state must not contain raw high residual"

    def assert_lower_contract(self, state: Mapping[str, Any]) -> None:
        for key in self.lower_forbidden_keys:
            assert key not in state, f"lower state leaked forbidden key {key!r}"
        assert "x_low_forecast" not in state, "lower state must not contain full low forecast"
