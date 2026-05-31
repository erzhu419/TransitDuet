"""Low-frequency plan curve parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable

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
