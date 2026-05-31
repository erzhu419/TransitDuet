"""Transit frequency tracker backed by the generic Freq-HRL core."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping
import math

import numpy as np

from ...core import CausalPromotionGate, MultiEntityBinnedStream
from ...encoders import (
    CausalAdaptiveWaveletEncoder,
    CausalEMAEncoder,
    CausalFourierEncoder,
    CausalPoissonHarmonicEncoder,
)


def _as_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _as_energy(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(max(float(np.mean(arr * arr)), 0.0)))


class TransitFrequencyTracker:
    """Compatibility layer for FreqTransitDuet state features.

    The public methods mirror the useful parts of FreqDuet's
    ``DemandFrequencyTracker`` so a copied transit runner can be adapted with
    minimal changes, while the actual decomposition and promotion logic live in
    the shared `freq_hrl` core.
    """

    def __init__(
        self,
        update_interval_s: float = 1.0,
        bin_sec: float = 60.0,
        method: str = "ema",
        low_period_s: float = 1800.0,
        fast_period_s: float = 300.0,
        mid_period_s: float = 900.0,
        energy_period_s: float = 600.0,
        persistence_period_s: float = 900.0,
        persistence_threshold: float = 1.0,
        forecast_horizon_s: float = 1800.0,
        harmonic_period_s: float = 14 * 3600.0,
        fourier_k: int = 4,
        harmonic_forgetting: float = 0.995,
        harmonic_learning_rate: float = 0.4,
        harmonic_ridge: float = 1.0,
        harmonic_observation_model: str = "poisson",
        harmonic_nb_dispersion: float = 20.0,
        adaptive_learn_rate: float = 0.02,
        adaptive_ridge: float = 1e-6,
        adaptive_max_predictor: float = 3.0,
        global_demand_norm: float = 50.0,
        local_demand_norm: float = 10.0,
        slope_norm: float = 5.0,
        upper_mode: str = "low",
        lower_mode: str = "high",
        promotion_enable: bool = False,
        promotion_window_s: float = 900.0,
        promotion_residual_threshold: float = 2.0,
        promotion_persistence_ratio: float = 0.4,
        promotion_cooldown_s: float = 1800.0,
    ) -> None:
        self.update_interval_s = max(float(update_interval_s), 1e-9)
        self.bin_sec = max(float(bin_sec), self.update_interval_s)
        self.bin_steps = max(1, int(round(self.bin_sec / self.update_interval_s)))
        self.bin_interval_s = self.bin_steps * self.update_interval_s
        self.method = str(method or "ema").lower()
        self.upper_mode = str(upper_mode or "low").lower()
        self.lower_mode = str(lower_mode or "high").lower()
        self.global_demand_norm = max(float(global_demand_norm), 1e-9)
        self.local_demand_norm = max(float(local_demand_norm), 1e-9)
        self.slope_norm = max(float(slope_norm), 1e-9)
        self.promotion_enabled = bool(promotion_enable)
        self.od_features_enabled = False
        self.promotion_absorptions = 0
        self.promotion_absorbed_rate = 0.0
        self.total_updates = 0
        self._pending_steps = 0
        self._pending_station: dict[Any, float] = defaultdict(float)
        self._known_local_keys: set[Any] = set()
        self._stream = MultiEntityBinnedStream(bin_sec=self.bin_interval_s)

        if self.method in {
            "poisson_harmonic",
            "dynamic_harmonic_poisson",
            "dynamic_harmonic_nb",
            "negative_binomial_harmonic",
            "nb_harmonic",
        }:
            self.method = "poisson_harmonic"
            obs_model = (
                "negative_binomial"
                if str(method or "").lower() in {
                    "dynamic_harmonic_nb",
                    "negative_binomial_harmonic",
                    "nb_harmonic",
                }
                else harmonic_observation_model
            )
            self.encoder = CausalPoissonHarmonicEncoder(
                update_interval_s=self.bin_interval_s,
                period_s=harmonic_period_s,
                fourier_k=fourier_k,
                learning_rate=harmonic_learning_rate,
                ridge=harmonic_ridge,
                observation_model=obs_model,
                nb_dispersion=harmonic_nb_dispersion,
                residual_period_s=fast_period_s,
                mid_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        elif self.method in {"fourier", "harmonic", "dynamic_harmonic", "harmonic_rls"}:
            self.method = "fourier"
            self.encoder = CausalFourierEncoder(
                update_interval_s=self.bin_interval_s,
                period_s=harmonic_period_s,
                fourier_k=fourier_k,
                forgetting_factor=harmonic_forgetting,
                residual_period_s=fast_period_s,
                mid_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        elif self.method in {"adaptive_wavelet", "learnable_wavelet", "lifting_wavelet"}:
            self.method = "adaptive_wavelet"
            self.encoder = CausalAdaptiveWaveletEncoder(
                update_interval_s=self.bin_interval_s,
                low_period_s=low_period_s,
                residual_period_s=fast_period_s,
                mid_period_s=mid_period_s,
                slope_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
                learn_rate=adaptive_learn_rate,
                ridge=adaptive_ridge,
                max_predictor=adaptive_max_predictor,
            )
        elif self.method in {"ema", "causal_ema", "raw_history", "history", "raw"}:
            raw_history_mode = self.method in {"raw_history", "history", "raw"}
            self.method = "raw_history" if raw_history_mode else "ema"
            self.encoder = CausalEMAEncoder(
                update_interval_s=self.bin_interval_s,
                low_period_s=self.bin_interval_s if raw_history_mode else low_period_s,
                fast_period_s=self.bin_interval_s if raw_history_mode else fast_period_s,
                mid_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        else:
            raise ValueError(f"unknown transit frequency method: {method}")

        self._promotion_cfg = {
            "update_interval_s": self.bin_interval_s,
            "window_s": promotion_window_s,
            "residual_threshold": promotion_residual_threshold,
            "persistence_ratio": promotion_persistence_ratio,
            "cooldown_s": promotion_cooldown_s,
        }
        self.global_promotion_gate = (
            CausalPromotionGate(**self._promotion_cfg)
            if self.promotion_enabled else None
        )
        self.local_promotion_gates: dict[Any, CausalPromotionGate] = {}
        self.upper_feature_dim = self._upper_dim_for_mode(self.upper_mode)
        self.lower_feature_dim = self._lower_dim_for_mode(self.lower_mode)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None, update_interval_s: float = 1.0) -> "TransitFrequencyTracker":
        cfg = dict(cfg or {})
        promotion_cfg = dict(cfg.get("promotion", {}) or {})
        window_s = promotion_cfg.get("window_s", float(promotion_cfg.get("window_min", 15.0)) * 60.0)
        cooldown_s = promotion_cfg.get("cooldown_s", float(promotion_cfg.get("cooldown_min", 30.0)) * 60.0)
        low_period_s = cfg.get("low_period_s", float(cfg.get("low_cut_min", 30.0)) * 60.0)
        fast_period_s = cfg.get("fast_period_s", float(cfg.get("high_cut_min", 5.0)) * 60.0)
        mid_period_s = cfg.get("middle_period_s", float(cfg.get("mid_cut_min", 15.0)) * 60.0)
        return cls(
            update_interval_s=update_interval_s,
            bin_sec=cfg.get("bin_sec", 60.0),
            method=cfg.get("method", "ema"),
            low_period_s=low_period_s,
            fast_period_s=fast_period_s,
            mid_period_s=mid_period_s,
            energy_period_s=cfg.get("energy_period_s", 600.0),
            persistence_period_s=cfg.get("persistence_period_s", window_s),
            persistence_threshold=cfg.get("persistence_threshold", 1.0),
            forecast_horizon_s=cfg.get("forecast_horizon_s", 1800.0),
            harmonic_period_s=cfg.get("harmonic_period_s", 14 * 3600.0),
            fourier_k=cfg.get("fourier_k", cfg.get("fourier_K", 4)),
            harmonic_forgetting=cfg.get("harmonic_forgetting", 0.995),
            harmonic_learning_rate=cfg.get("harmonic_learning_rate", 0.4),
            harmonic_ridge=cfg.get("harmonic_ridge", 1.0),
            harmonic_observation_model=cfg.get("harmonic_observation_model", "poisson"),
            harmonic_nb_dispersion=cfg.get("harmonic_nb_dispersion", 20.0),
            adaptive_learn_rate=cfg.get("adaptive_learn_rate", 0.02),
            adaptive_ridge=cfg.get("adaptive_ridge", 1e-6),
            adaptive_max_predictor=cfg.get("adaptive_max_predictor", 3.0),
            global_demand_norm=cfg.get("global_demand_norm", 50.0),
            local_demand_norm=cfg.get("local_demand_norm", 10.0),
            slope_norm=cfg.get("slope_norm", 5.0),
            upper_mode=cfg.get("upper_mode", "low"),
            lower_mode=cfg.get("lower_mode", "high"),
            promotion_enable=promotion_cfg.get("enable", cfg.get("promotion_enable", False)),
            promotion_window_s=window_s,
            promotion_residual_threshold=promotion_cfg.get("residual_threshold", 2.0),
            promotion_persistence_ratio=promotion_cfg.get("persistence_ratio", 0.4),
            promotion_cooldown_s=cooldown_s,
        )

    def reset(self, episode_id: int | None = None) -> None:
        self.encoder.reset(episode_id)
        self.total_updates = 0
        self._pending_steps = 0
        self._pending_station.clear()
        self._known_local_keys.clear()
        self._stream.reset()
        if self.global_promotion_gate is not None:
            self.global_promotion_gate.reset()
        self.local_promotion_gates.clear()

    def update(self, arrivals_by_station: Mapping[Any, float] | None, arrivals_by_od: Mapping[Any, float] | None = None) -> None:
        arrivals_by_station = arrivals_by_station or {}
        for key, count in arrivals_by_station.items():
            local_key = self._normalize_local_key(key)
            self._pending_station[local_key] += float(count)
            self._known_local_keys.add(local_key)
        self._pending_steps += 1
        if self._pending_steps < self.bin_steps:
            return
        self._flush_pending()

    def _flush_pending(self) -> None:
        scale = 60.0 / max(self.bin_interval_s, 1e-9)
        timestamp = self.total_updates * self.bin_interval_s
        total = float(sum(self._pending_station.values())) * scale
        self.encoder.update({
            "timestamp": timestamp,
            "entity_id": "global",
            "x_raw": [total],
        })
        global_features = self.encoder.features(timestamp, "global")
        if self.global_promotion_gate is not None:
            global_signal = self.global_promotion_gate.update(global_features, timestamp)
            global_features["promotion"] = global_signal

        for key in sorted(self._known_local_keys, key=lambda x: repr(x)):
            count = float(self._pending_station.get(key, 0.0)) * scale
            self.encoder.update({
                "timestamp": timestamp,
                "entity_id": key,
                "x_raw": [count],
            })
            if self.promotion_enabled:
                gate = self._get_local_promotion_gate(key)
                gate.update(self.encoder.features(timestamp, key), timestamp)

        self._pending_station.clear()
        self._pending_steps = 0
        self.total_updates += 1

    def _normalize_local_key(self, key: Any) -> tuple[int, bool]:
        if isinstance(key, tuple) and len(key) >= 2:
            return (int(key[0]), bool(key[1]))
        if isinstance(key, Mapping):
            return (int(key.get("station_id", 0)), bool(key.get("direction", True)))
        return (int(key), True)

    def _get_local_promotion_gate(self, key: Any) -> CausalPromotionGate:
        key = self._normalize_local_key(key)
        if key not in self.local_promotion_gates:
            self.local_promotion_gates[key] = CausalPromotionGate(**self._promotion_cfg)
        return self.local_promotion_gates[key]

    def _features_or_zero(self, entity_id: Any, norm: float) -> dict[str, Any]:
        try:
            return self.encoder.features(entity_id=entity_id)
        except KeyError:
            z = np.asarray([0.0], dtype=np.float64)
            return {
                "x_raw": z,
                "x_low": z,
                "x_low_slope": z,
                "x_low_forecast": z.reshape(1, 1),
                "x_low_uncertainty": z,
                "x_mid": z,
                "x_high": z,
                "x_high_delta": z,
                "x_high_energy": z,
                "x_high_persistence": z,
                "shock_age": z,
            }

    def _forecast_terminal(self, features: Mapping[str, Any]) -> float:
        forecast = np.asarray(features.get("x_low_forecast", 0.0), dtype=np.float64)
        if forecast.ndim == 0:
            return float(forecast)
        return float(forecast.reshape(-1)[-1]) if forecast.size else 0.0

    def _upper_low_features(self, features: Mapping[str, Any]) -> list[float]:
        return [
            _as_scalar(features["x_low"]) / self.global_demand_norm,
            _as_scalar(features["x_low_slope"]) / self.slope_norm,
            self._forecast_terminal(features) / self.global_demand_norm,
            _as_energy(features["x_high_energy"]) / self.global_demand_norm,
        ]

    def _upper_high_features(self, features: Mapping[str, Any]) -> list[float]:
        return [
            _as_scalar(features["x_high"]) / self.global_demand_norm,
            _as_scalar(features["x_high_delta"]) / self.slope_norm,
            _as_energy(features["x_high_energy"]) / self.global_demand_norm,
        ]

    def _upper_middle_features(self, features: Mapping[str, Any]) -> list[float]:
        return [
            _as_scalar(features["x_mid"]) / self.global_demand_norm,
            _as_energy(features["x_mid"]) / self.global_demand_norm,
            _as_scalar(features["x_high_persistence"]),
        ]

    def _promotion_features(self, gate: CausalPromotionGate | None) -> list[float]:
        if gate is None:
            return [0.0, 0.0, 0.0] if self.promotion_enabled else []
        sig = gate.signal()
        return [
            1.0 if sig.promote else 0.0,
            float(sig.promotion_strength),
            float(sig.shock_age),
        ]

    def upper_features(self, mode: str | None = None) -> np.ndarray:
        features = self._features_or_zero("global", self.global_demand_norm)
        mode = str(mode or self.upper_mode).lower()
        if mode in {"low", "lf", "split"}:
            vals = self._upper_low_features(features)
        elif mode in {"high", "hf"}:
            vals = self._upper_high_features(features)
        elif mode in {"low_mid", "low_middle", "regime"}:
            vals = self._upper_low_features(features) + self._upper_middle_features(features)
        elif mode in {"all", "allfreq", "all_freq"}:
            vals = self._upper_low_features(features)[:3] + self._upper_high_features(features) + self._upper_middle_features(features)
        else:
            raise ValueError(f"unknown upper frequency mode: {mode}")
        vals += self._promotion_features(self.global_promotion_gate)
        return np.asarray(vals, dtype=np.float32)

    def lower_features(self, station_id: int, direction: bool, mode: str | None = None) -> np.ndarray:
        key = (int(station_id), bool(direction))
        features = self._features_or_zero(key, self.local_demand_norm)
        global_features = self._features_or_zero("global", self.global_demand_norm)
        mode = str(mode or self.lower_mode).lower()
        high_feats = [
            _as_scalar(features["x_high"]) / self.local_demand_norm,
            _as_scalar(features["x_high_delta"]) / self.local_demand_norm,
            _as_energy(features["x_high_energy"]) / self.local_demand_norm,
            _as_energy(global_features["x_high_energy"]) / self.global_demand_norm,
        ]
        low_feats = [
            _as_scalar(features["x_low"]) / self.local_demand_norm,
            _as_scalar(features["x_low_slope"]) / self.local_demand_norm,
            self._forecast_terminal(features) / self.local_demand_norm,
            _as_scalar(global_features["x_low"]) / self.global_demand_norm,
        ]
        mid_feats = [
            _as_scalar(features["x_mid"]) / self.local_demand_norm,
            _as_energy(features["x_mid"]) / self.local_demand_norm,
            _as_scalar(features["shock_age"]),
        ]
        if mode in {"high", "hf", "split"}:
            vals = high_feats
        elif mode in {"low", "lf"}:
            vals = low_feats
        elif mode in {"high_mid", "high_middle", "regime"}:
            vals = high_feats[:3] + mid_feats + [_as_scalar(global_features["x_high_persistence"])]
        elif mode in {"all", "allfreq", "all_freq"}:
            vals = low_feats[:3] + high_feats
        else:
            raise ValueError(f"unknown lower frequency mode: {mode}")
        vals += self._promotion_features(self.local_promotion_gates.get(key))
        return np.asarray(vals, dtype=np.float32)

    def local_high_value(self, station_id: int, direction: bool) -> float:
        features = self._features_or_zero((int(station_id), bool(direction)), self.local_demand_norm)
        return _as_scalar(features["x_high"]) / self.local_demand_norm

    def local_low_value(self, station_id: int, direction: bool) -> float:
        features = self._features_or_zero((int(station_id), bool(direction)), self.local_demand_norm)
        return _as_scalar(features["x_low"]) / self.local_demand_norm

    def local_promotion_summary(self, station_id: int, direction: bool) -> dict[str, Any]:
        gate = self.local_promotion_gates.get((int(station_id), bool(direction)))
        if gate is None:
            return {
                "promote": False,
                "promotion_strength": 0.0,
                "reason": "inactive",
                "cooldown_remaining": 0.0,
                "shock_age": 0.0,
                "score": 0.0,
                "direction": 0.0,
            }
        return gate.signal().to_mapping()

    def summary(self) -> dict[str, Any]:
        features = self._features_or_zero("global", self.global_demand_norm)
        promotion = (
            self.global_promotion_gate.signal().to_mapping()
            if self.global_promotion_gate is not None
            else {
                "promote": False,
                "promotion_strength": 0.0,
                "shock_age": 0.0,
                "score": 0.0,
            }
        )
        return {
            "freq_low_demand": _as_scalar(features["x_low"]) / self.global_demand_norm,
            "freq_low_slope": _as_scalar(features["x_low_slope"]) / self.slope_norm,
            "freq_low_forecast": self._forecast_terminal(features) / self.global_demand_norm,
            "freq_high_energy": _as_energy(features["x_high_energy"]) / self.global_demand_norm,
            "freq_middle": _as_scalar(features["x_mid"]) / self.global_demand_norm,
            "freq_middle_energy": _as_energy(features["x_mid"]) / self.global_demand_norm,
            "freq_updates": int(self.total_updates),
            "freq_method": self.method,
            "freq_promotion_flag": 1.0 if promotion.get("promote", False) else 0.0,
            "freq_promotion_strength": float(promotion.get("promotion_strength", 0.0)),
            "freq_promotion_age": float(promotion.get("shock_age", 0.0)),
            "freq_promotion_score": float(promotion.get("score", 0.0)),
            "freq_promotion_absorptions": int(self.promotion_absorptions),
            "freq_promotion_absorbed": float(self.promotion_absorbed_rate),
        }

    def _upper_dim_for_mode(self, mode: str) -> int:
        mode = str(mode or "low").lower()
        base = {
            "low": 4,
            "lf": 4,
            "split": 4,
            "high": 3,
            "hf": 3,
            "low_mid": 7,
            "low_middle": 7,
            "regime": 7,
            "all": 9,
            "allfreq": 9,
            "all_freq": 9,
        }.get(mode)
        if base is None:
            raise ValueError(f"unknown upper frequency mode: {mode}")
        return base + (3 if self.promotion_enabled else 0)

    def _lower_dim_for_mode(self, mode: str) -> int:
        mode = str(mode or "high").lower()
        base = {
            "high": 4,
            "hf": 4,
            "split": 4,
            "low": 4,
            "lf": 4,
            "high_mid": 7,
            "high_middle": 7,
            "regime": 7,
            "all": 7,
            "allfreq": 7,
            "all_freq": 7,
        }.get(mode)
        if base is None:
            raise ValueError(f"unknown lower frequency mode: {mode}")
        return base + (3 if self.promotion_enabled else 0)
