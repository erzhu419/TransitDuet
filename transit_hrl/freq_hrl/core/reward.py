"""Reward and credit attribution helpers for Freq-HRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass
class RewardAttributionAccumulator:
    """Accumulate frequency-aware reward attribution records.

    Domains provide the four required attribution channels from their own
    semantics.  This class deliberately stays domain-agnostic and only handles
    accounting and summary statistics.
    """

    records: list[dict[str, float]] = field(default_factory=list)

    def reset(self) -> None:
        self.records.clear()

    def log_step(
        self,
        task_reward: float,
        low_frequency_cost: float,
        high_frequency_cost: float,
        leakage_cost: float,
        promotion_adaptation_cost: float,
        upper_credit: float | None = None,
        lower_credit: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        low = float(low_frequency_cost)
        high = float(high_frequency_cost)
        leakage = float(leakage_cost)
        promotion = float(promotion_adaptation_cost)
        if upper_credit is None:
            upper_credit = -low - 0.5 * leakage - promotion
        if lower_credit is None:
            lower_credit = -high - 0.5 * leakage
        record = {
            "task_reward": float(task_reward),
            "low_frequency_cost": low,
            "high_frequency_cost": high,
            "leakage_cost": leakage,
            "promotion_adaptation_cost": promotion,
            "upper_credit": float(upper_credit),
            "lower_credit": float(lower_credit),
        }
        if metadata:
            for key, value in metadata.items():
                try:
                    record[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        self.records.append(record)
        return record.copy()

    def summary(self, prefix: str = "freq_attr") -> dict[str, float]:
        if not self.records:
            keys = [
                "task_reward",
                "low_frequency_cost",
                "high_frequency_cost",
                "leakage_cost",
                "promotion_adaptation_cost",
                "upper_credit",
                "lower_credit",
            ]
            out = {f"{prefix}_{key}_sum": 0.0 for key in keys}
            out.update({f"{prefix}_{key}_mean": 0.0 for key in keys})
            out[f"{prefix}_n"] = 0
            return out
        keys = sorted({key for row in self.records for key in row.keys()})
        out: dict[str, float] = {f"{prefix}_n": float(len(self.records))}
        for key in keys:
            vals = np.asarray([row.get(key, 0.0) for row in self.records], dtype=np.float64)
            out[f"{prefix}_{key}_sum"] = float(vals.sum())
            out[f"{prefix}_{key}_mean"] = float(vals.mean())
        return out

    def episode_metrics(self, prefix: str = "freq_attr") -> dict[str, float]:
        """Return per-episode metrics with aggregation-friendly names."""
        if not self.records:
            keys = [
                "task_reward",
                "low_frequency_cost",
                "high_frequency_cost",
                "leakage_cost",
                "promotion_adaptation_cost",
                "upper_credit",
                "lower_credit",
            ]
            out = {f"{prefix}_{key}": 0.0 for key in keys}
            out.update({f"{prefix}_{key}_total": 0.0 for key in keys})
            out[f"{prefix}_n"] = 0.0
            return out
        keys = sorted({key for row in self.records for key in row.keys()})
        out: dict[str, float] = {f"{prefix}_n": float(len(self.records))}
        for key in keys:
            vals = np.asarray([row.get(key, 0.0) for row in self.records], dtype=np.float64)
            out[f"{prefix}_{key}"] = float(vals.mean())
            out[f"{prefix}_{key}_total"] = float(vals.sum())
        return out
