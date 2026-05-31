"""Frequency responsibility diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .leakage import LeakageRegularizer, as_2d


def _float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray | None:
    values = _float_array(values)
    if values.size == 0 or float(np.std(values)) < 1e-12:
        return None
    edges = np.quantile(values, np.linspace(0.0, 1.0, max(2, int(bins)) + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        lo = float(values.min())
        hi = float(values.max())
        if abs(hi - lo) < 1e-12:
            return None
        edges = np.linspace(lo, hi, max(2, int(bins)) + 1)
    return edges


def binned_mutual_information(xs: Any, ys: Any, bins: int = 8, min_n: int = 10, normalized: bool = True) -> float:
    x = _float_array(xs)
    y = _float_array(ys)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < int(min_n):
        return 0.0
    x_edges = _quantile_bins(x, bins)
    y_edges = _quantile_bins(y, bins)
    if x_edges is None or y_edges is None:
        return 0.0

    xb = np.digitize(x, x_edges[1:-1], right=False)
    yb = np.digitize(y, y_edges[1:-1], right=False)
    joint = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64)
    for i, j in zip(xb, yb):
        joint[int(i), int(j)] += 1.0
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i, j in zip(*np.nonzero(pxy > 0.0)):
        mi += pxy[i, j] * math.log(pxy[i, j] / max(px[i] * py[j], 1e-12))
    if not normalized:
        return float(max(mi, 0.0))
    hx = -float(np.sum(px[px > 0.0] * np.log(px[px > 0.0])))
    hy = -float(np.sum(py[py > 0.0] * np.log(py[py > 0.0])))
    return float(np.clip(mi / math.sqrt(max(hx * hy, 1e-12)), 0.0, 1.0))


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


class FrequencyDiagnostics:
    """Collect episode traces and summarize Freq-HRL responsibility metrics."""

    def __init__(self, leakage: LeakageRegularizer | None = None, mi_bins: int = 8) -> None:
        self.leakage = leakage or LeakageRegularizer()
        self.mi_bins = int(mi_bins)
        self.reset()

    def reset(self) -> None:
        self.times: list[float] = []
        self.upper_samples: list[tuple[float, float, float]] = []
        self.lower_samples: list[tuple[float, float, float]] = []
        self.upper_effects: list[Any] = []
        self.lower_effects: list[Any] = []
        self.promotion_times: list[float] = []
        self.regime_shift_times: list[float] = []
        self.shock_times: list[float] = []
        self.lower_response_times: list[float] = []

    def log_step(
        self,
        t: float,
        states: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        freq_features: Mapping[str, Any] | None = None,
        effects: Mapping[str, Any] | None = None,
    ) -> None:
        states = states or {}
        actions = actions or {}
        freq_features = freq_features or {}
        effects = effects or {}
        self.times.append(float(t))

        low = _scalar(freq_features.get("x_low", 0.0))
        high = _scalar(freq_features.get("x_high", freq_features.get("x_high_energy", 0.0)))
        upper_action = actions.get("upper", None)
        lower_action = actions.get("lower", None)
        if upper_action is not None:
            self.upper_samples.append((low, abs(high), _scalar(upper_action)))
        if lower_action is not None:
            self.lower_samples.append((low, high, _scalar(lower_action)))

        if "upper" in effects:
            self.upper_effects.append(effects["upper"])
        elif upper_action is not None:
            self.upper_effects.append(upper_action)
        if "lower" in effects:
            self.lower_effects.append(effects["lower"])
        elif lower_action is not None:
            self.lower_effects.append(lower_action)

        promotion = freq_features.get("promotion", states.get("promotion", {}))
        if isinstance(promotion, Mapping) and promotion.get("promote", False):
            self.promotion_times.append(float(t))
        if bool(states.get("regime_shift", False)):
            self.regime_shift_times.append(float(t))
        if bool(states.get("shock", False)):
            self.shock_times.append(float(t))
        if bool(states.get("lower_responded", False)):
            self.lower_response_times.append(float(t))

    def _focus_metrics(self) -> dict[str, float]:
        metrics = {
            "focus_score": 0.0,
            "upper_low_mi": 0.0,
            "upper_high_mi": 0.0,
            "lower_high_mi": 0.0,
            "lower_low_mi": 0.0,
        }
        if self.upper_samples:
            arr = np.asarray(self.upper_samples, dtype=np.float64)
            metrics["upper_low_mi"] = binned_mutual_information(arr[:, 0], arr[:, 2], bins=self.mi_bins)
            metrics["upper_high_mi"] = binned_mutual_information(arr[:, 1], arr[:, 2], bins=self.mi_bins)
        if self.lower_samples:
            arr = np.asarray(self.lower_samples, dtype=np.float64)
            metrics["lower_high_mi"] = binned_mutual_information(arr[:, 1], arr[:, 2], bins=self.mi_bins)
            metrics["lower_low_mi"] = binned_mutual_information(arr[:, 0], arr[:, 2], bins=self.mi_bins)
        metrics["focus_score"] = (
            metrics["upper_low_mi"]
            - metrics["upper_high_mi"]
            + metrics["lower_high_mi"]
            - metrics["lower_low_mi"]
        )
        return metrics

    def _delay(self, triggers: list[float], references: list[float]) -> float:
        if not triggers or not references:
            return 0.0
        delays = []
        for ref in references:
            later = [t for t in triggers if t >= ref]
            if later:
                delays.append(min(later) - ref)
        return float(np.mean(delays)) if delays else 0.0

    def summarize_episode(self) -> dict[str, float]:
        upper_effect = as_2d(self.upper_effects) if self.upper_effects else np.zeros((0, 1))
        lower_effect = as_2d(self.lower_effects) if self.lower_effects else np.zeros((0, 1))
        leakage = self.leakage.compute(upper_effect, lower_effect)
        focus = self._focus_metrics()
        return {
            "UpperHFPower": float(leakage["UpperHFPower"]),
            "LowerLFDrift": float(leakage["LowerLFDrift"]),
            "FocusScore": float(focus["focus_score"]),
            "PromotionDelay": self._delay(self.promotion_times, self.regime_shift_times),
            "ShockResponseTime": self._delay(self.lower_response_times, self.shock_times),
            **{k: float(v) for k, v in focus.items()},
        }
