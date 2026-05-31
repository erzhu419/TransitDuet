"""Learned plan-curve action mapping for dual-level RL policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..policies import BernsteinPlanCurve


@dataclass
class PlanActionResult:
    target: np.ndarray
    coefficients: np.ndarray
    smoothness_penalty: float


@dataclass
class LearnedPlanActionMapper:
    """Convert upper actor latent actions into executable plan-curve targets."""

    curve: BernsteinPlanCurve
    coefficient_scale: float = 1.0
    eval_offset_s: float = 300.0

    @property
    def action_dim(self) -> int:
        return int(self.curve.action_dim)

    def coefficients(self, latent_action: Sequence[float]) -> np.ndarray:
        latent = np.asarray(latent_action, dtype=np.float64).reshape(-1)
        if latent.size != self.curve.action_dim:
            raise ValueError(f"expected latent action dim {self.curve.action_dim}, got {latent.size}")
        scale = max(float(self.coefficient_scale), 1e-9)
        return np.tanh(latent) * scale

    def target(self, current_value: Sequence[float], latent_action: Sequence[float]) -> PlanActionResult:
        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current.size != self.curve.n_entities:
            current = np.resize(current, self.curve.n_entities)
        coeffs = self.coefficients(latent_action)
        values = np.asarray([
            self.curve.value_at(
                float(current[i]),
                coeffs,
                offset_s=float(self.eval_offset_s),
                entity_index=i,
            )
            for i in range(self.curve.n_entities)
        ], dtype=np.float64)
        return PlanActionResult(
            target=values,
            coefficients=coeffs,
            smoothness_penalty=float(self.curve.smoothness_penalty(coeffs)),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "plan_basis_dim": int(self.curve.basis_dim),
            "plan_horizon_s": float(self.curve.horizon_s),
            "plan_eval_offset_s": float(self.eval_offset_s),
            "plan_coefficient_scale": float(self.coefficient_scale),
            "plan_action_dim": int(self.action_dim),
        }
