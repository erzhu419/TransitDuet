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
    reference_coefficients: np.ndarray | None = None
    residual_coefficients: np.ndarray | None = None


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

    def residual_coefficients(
        self, latent_action: Sequence[float]
    ) -> np.ndarray:
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

    def reference_coefficients(
        self,
        current_value: Sequence[float],
        desired_value: Sequence[float],
    ) -> np.ndarray:
        """Fit a causal low-frequency reference curve from current to desired."""

        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        desired = np.asarray(desired_value, dtype=np.float64).reshape(-1)
        if current.size != int(self.curve.n_entities):
            current = np.resize(current, int(self.curve.n_entities))
        if desired.size != int(self.curve.n_entities):
            desired = np.resize(desired, int(self.curve.n_entities))
        delta = desired - current
        if self.curve.shared_entities:
            coefficients = np.linspace(
                0.0,
                float(np.mean(delta)) if delta.size else 0.0,
                int(self.curve.basis_dim),
                dtype=np.float64,
            )
        else:
            coefficients = np.concatenate([
                np.linspace(
                    0.0,
                    float(value),
                    int(self.curve.basis_dim),
                    dtype=np.float64,
                )
                for value in delta
            ])
        return np.clip(
            coefficients,
            float(self.curve.delta_min),
            float(self.curve.delta_max),
        )

    def coefficients(
        self,
        latent_action: Sequence[float],
        *,
        reference_coefficients: Sequence[float] | None = None,
    ) -> np.ndarray:
        residual = self.residual_coefficients(latent_action)
        if reference_coefficients is None:
            return residual
        reference = np.asarray(
            reference_coefficients, dtype=np.float64
        ).reshape(-1)
        if reference.size != int(self.curve.action_dim):
            raise ValueError(
                "reference coefficient dimension must match the plan curve"
            )
        return np.clip(
            reference + residual,
            float(self.curve.delta_min),
            float(self.curve.delta_max),
        )

    def target(
        self,
        current_value: Sequence[float],
        latent_action: Sequence[float],
        *,
        reference_target: Sequence[float] | None = None,
    ) -> PlanActionResult:
        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current.size != self.curve.n_entities:
            current = np.resize(current, self.curve.n_entities)
        reference = (
            None
            if reference_target is None
            else self.reference_coefficients(current, reference_target)
        )
        residual = self.residual_coefficients(latent_action)
        coeffs = self.coefficients(
            latent_action, reference_coefficients=reference
        )
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
            reference_coefficients=(
                None if reference is None else reference.copy()
            ),
            residual_coefficients=residual.copy(),
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
            "plan_reference_residual_composition": True,
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

    def snapshot(self) -> "LearnedPlanCurveState":
        """Return an independent causal copy of the currently active curve."""

        if not self.active:
            raise RuntimeError("activate must be called before snapshot")
        copied = LearnedPlanCurveState(
            mapper=self.mapper,
            gross_cap=self.gross_cap,
        )
        copied.origin_s = float(self.origin_s)
        copied.base_value = np.asarray(self.base_value, dtype=np.float64).copy()
        copied.coefficients = np.asarray(
            self.coefficients, dtype=np.float64
        ).copy()
        copied.activation_count = int(self.activation_count)
        return copied

    def activate(
        self,
        *,
        now_s: float,
        current_value: Sequence[float],
        latent_action: Sequence[float],
        reference_target: Sequence[float] | None = None,
    ) -> PlanActionResult:
        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current.size != int(self.mapper.curve.n_entities):
            current = np.resize(current, int(self.mapper.curve.n_entities))
        base = self.value_at(now_s) if self.active else self._cap(current)
        reference = (
            None
            if reference_target is None
            else self.mapper.reference_coefficients(base, reference_target)
        )
        residual = self.mapper.residual_coefficients(latent_action)
        coefficients = self.mapper.coefficients(
            latent_action, reference_coefficients=reference
        )
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
            reference_coefficients=(
                None if reference is None else reference.copy()
            ),
            residual_coefficients=residual.copy(),
        )
