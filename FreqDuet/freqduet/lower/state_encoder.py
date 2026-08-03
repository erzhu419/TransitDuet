"""Physical, dimensionless state encoding for station-level holding control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PhysicalLowerStateEncoder:
    """Replace legacy identifiers and mixed units with physical features.

    The encoded vector keeps the legacy dimensionality so frequency and local
    context tails remain compatible. The first legacy slot (``bus_id``) is
    replaced by the upper planner's target headway, recovered from the forward
    headway and its already-normalized deviation.
    """

    base_state_dim: int
    max_station_id: int
    service_duration_h: float
    action_range_s: float
    headway_norm_s: float = 600.0
    dwell_norm_s: float = 300.0
    speed_norm_mps: float = 15.0
    target_min_s: float = 120.0
    target_max_s: float = 900.0
    headway_ratio_clip: float = 3.0
    dwell_clip: float = 3.0
    speed_clip: float = 2.0
    deviation_clip: float = 3.0
    tail_clip: float = 5.0

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        *,
        base_state_dim: int,
        max_station_id: int,
        service_duration_h: float,
        action_range_s: float,
    ) -> "PhysicalLowerStateEncoder":
        values = dict(cfg or {})
        target_min_s = max(
            float(values.get("target_min_s", 120.0)), 1.0)
        target_max_s = max(
            float(values.get("target_max_s", 900.0)), target_min_s)
        return cls(
            base_state_dim=int(base_state_dim),
            max_station_id=max(int(max_station_id), 1),
            service_duration_h=max(float(service_duration_h), 1.0),
            action_range_s=max(float(action_range_s), 1e-6),
            headway_norm_s=max(
                float(values.get("headway_norm_s", 600.0)), 1e-6),
            dwell_norm_s=max(
                float(values.get("dwell_norm_s", 300.0)), 1e-6),
            speed_norm_mps=max(
                float(values.get("speed_norm_mps", 15.0)), 1e-6),
            target_min_s=target_min_s,
            target_max_s=target_max_s,
            headway_ratio_clip=max(
                float(values.get("headway_ratio_clip", 3.0)), 1.0),
            dwell_clip=max(float(values.get("dwell_clip", 3.0)), 1.0),
            speed_clip=max(float(values.get("speed_clip", 2.0)), 1.0),
            deviation_clip=max(
                float(values.get("deviation_clip", 3.0)), 1.0),
            tail_clip=max(float(values.get("tail_clip", 5.0)), 1.0),
        )

    def _target_headway(self, forward_s: float, deviation: float) -> float:
        denominator = 1.0 + float(deviation)
        target = self.headway_norm_s
        if float(forward_s) > 0.0 and abs(denominator) > 0.05:
            candidate = float(forward_s) / denominator
            if np.isfinite(candidate):
                target = candidate
        return float(np.clip(target, self.target_min_s, self.target_max_s))

    def encode(self, observation) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.size < self.base_state_dim or self.base_state_dim < 8:
            raise ValueError(
                "lower observation is shorter than the configured base schema")
        values = np.nan_to_num(
            raw.astype(np.float64, copy=True),
            nan=0.0,
            posinf=self.tail_clip,
            neginf=-self.tail_clip,
        )

        direction_up = bool(values[3] >= 0.5)
        station = float(np.clip(values[1], 0.0, self.max_station_id))
        station_progress = (
            station / self.max_station_id
            if direction_up
            else (self.max_station_id - station) / self.max_station_id
        )
        deviation = float(values[7])
        target_s = self._target_headway(values[4], deviation)

        encoded = values.copy()
        encoded[0] = np.clip(target_s / self.headway_norm_s, 0.0, 2.0)
        encoded[1] = np.clip(station_progress, 0.0, 1.0)
        encoded[2] = np.clip(
            values[2] / self.service_duration_h, 0.0, 1.0)
        encoded[3] = 1.0 if direction_up else -1.0
        encoded[4] = np.clip(
            values[4] / target_s, 0.0, self.headway_ratio_clip)
        encoded[5] = np.clip(
            values[5] / target_s, 0.0, self.headway_ratio_clip)
        encoded[6] = np.clip(
            values[6] / self.dwell_norm_s, 0.0, self.dwell_clip)
        encoded[7] = np.clip(
            deviation, -self.deviation_clip, self.deviation_clip)
        encoded[8:self.base_state_dim] = np.clip(
            values[8:self.base_state_dim] / self.speed_norm_mps,
            0.0,
            self.speed_clip,
        )
        encoded[self.base_state_dim:] = np.clip(
            values[self.base_state_dim:], -self.tail_clip, self.tail_clip)
        return encoded.astype(np.float32)

    def encode_action(self, action_s: float) -> float:
        return float(np.clip(
            float(action_s) / self.action_range_s, 0.0, 1.0))
