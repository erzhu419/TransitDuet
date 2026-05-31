"""Low-frequency plan curve parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence

import numpy as np


@dataclass
class BernsteinPlanCurve:
    """Map upper action coefficients to a smooth future plan curve."""

    horizon_s: float = 3600.0
    basis_dim: int = 4
    min_value: float = -np.inf
    max_value: float = np.inf
    delta_min: float = -1.0
    delta_max: float = 1.0
    n_entities: int = 1
    shared_entities: bool = False

    @property
    def action_dim(self) -> int:
        return self.basis_dim if self.shared_entities else self.basis_dim * self.n_entities

    @property
    def action_low(self) -> list[float]:
        return [float(self.delta_min)] * self.action_dim

    @property
    def action_high(self) -> list[float]:
        return [float(self.delta_max)] * self.action_dim

    def basis(self, offset_s: float) -> np.ndarray:
        n = max(0, int(self.basis_dim) - 1)
        if n == 0:
            return np.ones(1, dtype=np.float64)
        x = float(np.clip(float(offset_s) / max(float(self.horizon_s), 1e-9), 0.0, 1.0))
        return np.asarray(
            [comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i)) for i in range(n + 1)],
            dtype=np.float64,
        )

    def coefficients(self, action: Iterable[float], entity_index: int = 0) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {a.size}")
        if self.shared_entities:
            return a
        start = int(entity_index) * self.basis_dim
        return a[start:start + self.basis_dim]

    def delta_at(self, action: Iterable[float], offset_s: float, entity_index: int = 0) -> float:
        return float(np.dot(self.coefficients(action, entity_index), self.basis(offset_s)))

    def value_at(self, base_value: float, action: Iterable[float], offset_s: float, entity_index: int = 0) -> float:
        value = float(base_value) + self.delta_at(action, offset_s, entity_index)
        return float(np.clip(value, self.min_value, self.max_value))

    def smoothness_penalty(self, action: Iterable[float]) -> float:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size != self.action_dim:
            raise ValueError(f"expected action dim {self.action_dim}, got {a.size}")
        blocks = [a] if self.shared_entities else [
            a[i * self.basis_dim:(i + 1) * self.basis_dim]
            for i in range(self.n_entities)
        ]
        denom = max(max(abs(float(self.delta_min)), abs(float(self.delta_max))) ** 2, 1e-9)
        penalties = []
        for block in blocks:
            if block.size >= 3:
                curv = np.diff(block, n=2)
                penalties.append(float(np.mean(curv * curv) / denom))
            elif block.size == 2:
                slope = np.diff(block)
                penalties.append(float(0.25 * np.mean(slope * slope) / denom))
        return float(np.mean(penalties)) if penalties else 0.0


@dataclass
class CausalPlanCurveState:
    """Causal rolling plan state for vector-valued upper plans.

    A replan converts the current target/value and the desired future target
    into Bernstein coefficients.  Reuse evaluates the active curve without
    peeking at the new desired target, so lower control receives a smooth
    executable plan rather than a one-step target jump.
    """

    curve: BernsteinPlanCurve
    replan_interval_s: float = 300.0
    desired_change_threshold: float = 0.05
    gross_cap: float | None = 1.0

    def __post_init__(self) -> None:
        self.origin_s: float | None = None
        self.base_value: np.ndarray | None = None
        self.desired_value: np.ndarray | None = None
        self.action: np.ndarray | None = None
        self.decisions = 0
        self.reuses = 0
        self.smoothness_penalties: list[float] = []

    def _fit_action(self, base_value: np.ndarray, desired_value: np.ndarray) -> np.ndarray:
        base = np.asarray(base_value, dtype=np.float64).reshape(-1)
        desired = np.asarray(desired_value, dtype=np.float64).reshape(-1)
        if desired.size != self.curve.n_entities:
            desired = np.resize(desired, self.curve.n_entities)
        if base.size != self.curve.n_entities:
            base = np.resize(base, self.curve.n_entities)
        delta = desired - base
        if self.curve.shared_entities:
            avg_delta = float(np.mean(delta)) if delta.size else 0.0
            return np.linspace(0.0, avg_delta, self.curve.basis_dim, dtype=np.float64)
        blocks = [
            np.linspace(0.0, float(delta_i), self.curve.basis_dim, dtype=np.float64)
            for delta_i in delta
        ]
        return np.concatenate(blocks) if blocks else np.zeros(self.curve.action_dim, dtype=np.float64)

    def _value(self, now_s: float) -> np.ndarray:
        if self.origin_s is None or self.base_value is None or self.action is None:
            return np.zeros(self.curve.n_entities, dtype=np.float64)
        offset = float(now_s) - float(self.origin_s)
        return np.asarray([
            self.curve.value_at(
                float(self.base_value[i]),
                self.action,
                offset,
                entity_index=i,
            )
            for i in range(self.curve.n_entities)
        ], dtype=np.float64)

    def _gross_cap(self, value: np.ndarray) -> np.ndarray:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.gross_cap is None:
            return out
        gross = float(np.sum(np.abs(out)))
        cap = max(float(self.gross_cap), 0.0)
        if gross > cap and gross > 1e-12:
            out = out * (cap / gross)
        return out

    def should_replan(self, now_s: float, desired_value: Sequence[float], force: bool = False) -> bool:
        if force or self.origin_s is None or self.action is None or self.desired_value is None:
            return True
        elapsed = float(now_s) - float(self.origin_s)
        if elapsed < 0.0 or elapsed >= max(float(self.replan_interval_s), 1e-9):
            return True
        desired = np.asarray(desired_value, dtype=np.float64).reshape(-1)
        if desired.size != self.curve.n_entities:
            desired = np.resize(desired, self.curve.n_entities)
        diff = float(np.max(np.abs(desired - self.desired_value))) if desired.size else 0.0
        return diff >= max(float(self.desired_change_threshold), 0.0)

    def target_toward(
        self,
        now_s: float,
        current_value: Sequence[float],
        desired_value: Sequence[float],
        force_replan: bool = False,
    ) -> dict[str, object]:
        desired = np.asarray(desired_value, dtype=np.float64).reshape(-1)
        if desired.size != self.curve.n_entities:
            desired = np.resize(desired, self.curve.n_entities)
        current = np.asarray(current_value, dtype=np.float64).reshape(-1)
        if current.size != self.curve.n_entities:
            current = np.resize(current, self.curve.n_entities)
        replan = self.should_replan(now_s, desired, force=force_replan)
        if replan:
            base = self._value(now_s) if self.action is not None else current
            self.origin_s = float(now_s)
            self.base_value = self._gross_cap(base)
            self.desired_value = self._gross_cap(desired)
            self.action = self._fit_action(self.base_value, self.desired_value)
            self.decisions += 1
            penalty = self.curve.smoothness_penalty(self.action)
            self.smoothness_penalties.append(penalty)
            offset = 0.0
        else:
            self.reuses += 1
            penalty = self.smoothness_penalties[-1] if self.smoothness_penalties else 0.0
            offset = float(now_s) - float(self.origin_s or now_s)
        target = self._gross_cap(self._value(now_s))
        return {
            "target": target,
            "action": np.asarray(self.action, dtype=np.float64).copy() if self.action is not None else np.zeros(self.curve.action_dim),
            "base": np.asarray(self.base_value, dtype=np.float64).copy() if self.base_value is not None else current.copy(),
            "desired": np.asarray(self.desired_value, dtype=np.float64).copy() if self.desired_value is not None else desired.copy(),
            "offset_s": float(offset),
            "replan": bool(replan),
            "smoothness_penalty": float(penalty),
            "reuse_ratio": float(self.reuses / max(self.reuses + self.decisions, 1)),
        }
