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
    anchor_first_coefficient: bool = False

    def __post_init__(self) -> None:
        if self.anchor_first_coefficient and int(self.curve.basis_dim) < 2:
            raise ValueError(
                "an anchored plan curve requires at least two Bernstein coefficients"
            )

    @property
    def action_dim(self) -> int:
        if not self.anchor_first_coefficient:
            return int(self.curve.action_dim)
        per_entity = int(self.curve.basis_dim) - 1
        return int(
            per_entity
            if self.curve.shared_entities
            else per_entity * int(self.curve.n_entities)
        )

    def coefficients(self, latent_action: Sequence[float]) -> np.ndarray:
        latent = np.asarray(latent_action, dtype=np.float64).reshape(-1)
        if latent.size != self.action_dim:
            raise ValueError(f"expected latent action dim {self.action_dim}, got {latent.size}")
        scale = max(float(self.coefficient_scale), 1e-9)
        bounded = np.tanh(latent) * scale
        if not self.anchor_first_coefficient:
            return bounded
        if self.curve.shared_entities:
            return np.concatenate([np.zeros(1, dtype=np.float64), bounded])
        per_entity = int(self.curve.basis_dim) - 1
        blocks = [
            np.concatenate([
                np.zeros(1, dtype=np.float64),
                bounded[i * per_entity:(i + 1) * per_entity],
            ])
            for i in range(int(self.curve.n_entities))
        ]
        return np.concatenate(blocks)

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
            "plan_anchor_first_coefficient": bool(
                self.anchor_first_coefficient
            ),
        }


@dataclass
class LearnedPlanCurveState:
    """Execute one learned Bernstein action over primitive control steps.

    A new macro action is rebased on the currently executing plan value. With an
    anchored mapper, the first Bernstein coefficient is exactly zero, so both a
    scheduled replan and an early promotion replan are continuous at activation.
    """

    mapper: LearnedPlanActionMapper
    gross_cap: float | None = 1.0

    def __post_init__(self) -> None:
        if not self.mapper.anchor_first_coefficient:
            raise ValueError(
                "LearnedPlanCurveState requires anchor_first_coefficient=True"
            )
        self.reset()

    def reset(self) -> None:
        self.origin_s: float | None = None
        self.base_value: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None
        self.activation_count = 0

    @property
    def active(self) -> bool:
        return (
            self.origin_s is not None
            and self.base_value is not None
            and self.coefficients is not None
        )

    def _cap(self, value: np.ndarray) -> np.ndarray:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.gross_cap is None:
            return out
        cap = max(float(self.gross_cap), 0.0)
        gross = float(np.sum(np.abs(out)))
        if gross > cap and gross > 1e-12:
            out = out * (cap / gross)
        return out

    def value_at(self, now_s: float) -> np.ndarray:
        if not self.active:
            raise RuntimeError("activate must be called before value_at")
        offset_s = max(float(now_s) - float(self.origin_s), 0.0)
        values = np.asarray([
            self.mapper.curve.value_at(
                float(self.base_value[i]),
                self.coefficients,
                offset_s=offset_s,
                entity_index=i,
            )
            for i in range(int(self.mapper.curve.n_entities))
        ], dtype=np.float64)
        return self._cap(values)

    def activate(
        self,
        *,
        now_s: float,
        current_value: Sequence[float],
        latent_action: Sequence[float],
    ) -> PlanActionResult:
        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current.size != int(self.mapper.curve.n_entities):
            current = np.resize(current, int(self.mapper.curve.n_entities))
        base = self.value_at(now_s) if self.active else self._cap(current)
        coefficients = self.mapper.coefficients(latent_action)
        self.origin_s = float(now_s)
        self.base_value = self._cap(base)
        self.coefficients = coefficients.copy()
        self.activation_count += 1
        target = self.value_at(now_s)
        return PlanActionResult(
            target=target,
            coefficients=coefficients.copy(),
            smoothness_penalty=float(
                self.mapper.curve.smoothness_penalty(coefficients)
            ),
        )
