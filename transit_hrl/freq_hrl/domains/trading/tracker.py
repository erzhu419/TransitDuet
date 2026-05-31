"""Trading frequency tracker backed by the generic Freq-HRL core."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import math

import numpy as np

from ...core import CausalPromotionGate
from ...encoders import (
    CausalEMAEncoder,
    CausalFourierEncoder,
    CausalHaarWaveletEncoder,
    CausalStateSpaceEncoder,
)


def _arr(value: Any) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim == 0:
        out = out.reshape(1)
    return out.reshape(-1)


def _energy(value: Any) -> float:
    arr = _arr(value)
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(max(float(np.mean(arr * arr)), 0.0)))


class TradingFrequencyTracker:
    """Causal frequency features for market bars.

    The tracker accepts generic bar features such as returns, volume shock,
    spread proxy, realized volatility, or order-flow imbalance.  It exposes a
    compact upper view and a residual lower view, matching the Freq-HRL protocol.
    """

    def __init__(
        self,
        bar_sec: float = 60.0,
        method: str = "ema",
        low_period_s: float = 7200.0,
        fast_period_s: float = 300.0,
        mid_period_s: float = 1800.0,
        energy_period_s: float = 600.0,
        persistence_period_s: float = 1800.0,
        persistence_threshold: float = 1.0,
        forecast_horizon_s: float = 14400.0,
        harmonic_period_s: float = 24 * 3600.0,
        fourier_k: int = 4,
        harmonic_forgetting: float = 0.995,
        state_process_var: float = 1e-7,
        state_measurement_var: float = 1e-5,
        state_slope_period_s: float = 1800.0,
        feature_norm: Sequence[float] | float = 1.0,
        upper_mode: str = "low",
        lower_mode: str = "high",
        promotion_enable: bool = True,
        promotion_window_s: float = 1800.0,
        promotion_residual_threshold: float = 2.0,
        promotion_persistence_ratio: float = 0.4,
        promotion_cooldown_s: float = 3600.0,
        promotion_regime_threshold: float | None = None,
        promotion_min_age_s: float = 0.0,
        promotion_activation_strength_threshold: float = 0.0,
        promotion_startup_strength_age_s: float = 0.0,
        promotion_startup_strength_threshold: float = 0.0,
        promotion_adapt_low: bool = True,
        promotion_adapt_gain: float = 0.25,
    ) -> None:
        self.bar_sec = max(float(bar_sec), 1e-9)
        self.method = str(method or "ema").lower()
        self.upper_mode = str(upper_mode or "low").lower()
        self.lower_mode = str(lower_mode or "high").lower()
        self.feature_norm = _arr(feature_norm)
        if self.feature_norm.size == 1:
            self.feature_norm = np.asarray([max(float(self.feature_norm[0]), 1e-9)])
        self.promotion_enabled = bool(promotion_enable)
        self.promotion_adapt_low = bool(promotion_adapt_low)
        self.promotion_adapt_gain = max(float(promotion_adapt_gain), 0.0)
        self.promotion_absorptions = 0
        self.promotion_absorbed_norm = 0.0
        self.total_updates = 0

        if self.method in {"fourier", "harmonic", "dynamic_harmonic", "harmonic_rls"}:
            self.method = "fourier"
            self.encoder = CausalFourierEncoder(
                update_interval_s=self.bar_sec,
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
        elif self.method in {"state_space", "statespace", "kalman", "causal_state_space"}:
            self.method = "state_space"
            self.encoder = CausalStateSpaceEncoder(
                update_interval_s=self.bar_sec,
                process_var=state_process_var,
                measurement_var=state_measurement_var,
                slope_period_s=state_slope_period_s,
                residual_period_s=fast_period_s,
                mid_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        elif self.method in {"wavelet", "haar", "haar_wavelet", "causal_wavelet"}:
            self.method = "haar_wavelet"
            self.encoder = CausalHaarWaveletEncoder(
                update_interval_s=self.bar_sec,
                low_window_s=low_period_s,
                short_window_s=fast_period_s,
                slope_period_s=state_slope_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        elif self.method in {"ema", "causal_ema"}:
            self.method = "ema"
            self.encoder = CausalEMAEncoder(
                update_interval_s=self.bar_sec,
                low_period_s=low_period_s,
                fast_period_s=fast_period_s,
                mid_period_s=mid_period_s,
                energy_period_s=energy_period_s,
                persistence_period_s=persistence_period_s,
                persistence_threshold=persistence_threshold,
                forecast_horizon_s=forecast_horizon_s,
            )
        else:
            raise ValueError(f"unknown trading frequency method: {method}")

        self.promotion_gate = (
            CausalPromotionGate(
                update_interval_s=self.bar_sec,
                window_s=promotion_window_s,
                residual_threshold=promotion_residual_threshold,
                persistence_ratio=promotion_persistence_ratio,
                cooldown_s=promotion_cooldown_s,
                regime_threshold=promotion_regime_threshold,
                min_age_s=promotion_min_age_s,
                activation_strength_threshold=promotion_activation_strength_threshold,
                startup_strength_age_s=promotion_startup_strength_age_s,
                startup_strength_threshold=promotion_startup_strength_threshold,
            )
            if self.promotion_enabled else None
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "TradingFrequencyTracker":
        cfg = dict(cfg or {})
        promotion_cfg = dict(cfg.get("promotion", {}) or {})
        return cls(
            bar_sec=cfg.get("bar_sec", 60.0),
            method=cfg.get("method", "ema"),
            low_period_s=cfg.get("low_period_s", float(cfg.get("low_cut_min", 120.0)) * 60.0),
            fast_period_s=cfg.get("fast_period_s", float(cfg.get("high_cut_min", 5.0)) * 60.0),
            mid_period_s=cfg.get("mid_period_s", float(cfg.get("mid_cut_min", 30.0)) * 60.0),
            energy_period_s=cfg.get("energy_period_s", 600.0),
            persistence_period_s=cfg.get("persistence_period_s", promotion_cfg.get("window_s", 1800.0)),
            persistence_threshold=cfg.get("persistence_threshold", 1.0),
            forecast_horizon_s=cfg.get("forecast_horizon_s", 14400.0),
            harmonic_period_s=cfg.get("harmonic_period_s", 24 * 3600.0),
            fourier_k=cfg.get("fourier_k", cfg.get("fourier_K", 4)),
            harmonic_forgetting=cfg.get("harmonic_forgetting", 0.995),
            state_process_var=cfg.get("state_process_var", 1e-7),
            state_measurement_var=cfg.get("state_measurement_var", 1e-5),
            state_slope_period_s=cfg.get("state_slope_period_s", cfg.get("slope_period_s", 1800.0)),
            feature_norm=cfg.get("feature_norm", 1.0),
            upper_mode=cfg.get("upper_mode", "low"),
            lower_mode=cfg.get("lower_mode", "high"),
            promotion_enable=promotion_cfg.get("enable", cfg.get("promotion_enable", True)),
            promotion_window_s=promotion_cfg.get("window_s", float(promotion_cfg.get("window_min", 30.0)) * 60.0),
            promotion_residual_threshold=promotion_cfg.get("residual_threshold", 2.0),
            promotion_persistence_ratio=promotion_cfg.get("persistence_ratio", 0.4),
            promotion_cooldown_s=promotion_cfg.get("cooldown_s", float(promotion_cfg.get("cooldown_min", 60.0)) * 60.0),
            promotion_regime_threshold=promotion_cfg.get("regime_threshold", cfg.get("promotion_regime_threshold", None)),
            promotion_min_age_s=promotion_cfg.get("min_age_s", float(promotion_cfg.get("min_age_min", 0.0)) * 60.0),
            promotion_activation_strength_threshold=promotion_cfg.get(
                "activation_strength_threshold",
                cfg.get("promotion_activation_strength_threshold", 0.0),
            ),
            promotion_startup_strength_age_s=promotion_cfg.get(
                "startup_strength_age_s",
                float(promotion_cfg.get("startup_strength_age_min", 0.0)) * 60.0,
            ),
            promotion_startup_strength_threshold=promotion_cfg.get(
                "startup_strength_threshold",
                cfg.get("promotion_startup_strength_threshold", 0.0),
            ),
            promotion_adapt_low=promotion_cfg.get("adapt_low", cfg.get("promotion_adapt_low", True)),
            promotion_adapt_gain=promotion_cfg.get("adapt_gain", cfg.get("promotion_adapt_gain", 0.25)),
        )

    def reset(self, episode_id: int | None = None) -> None:
        self.encoder.reset(episode_id)
        self.total_updates = 0
        if self.promotion_gate is not None:
            self.promotion_gate.reset()
        self.promotion_absorptions = 0
        self.promotion_absorbed_norm = 0.0

    def _norm(self, dim: int) -> np.ndarray:
        if self.feature_norm.size == dim:
            return np.maximum(self.feature_norm, 1e-9)
        return np.ones(dim, dtype=np.float64) * max(float(self.feature_norm.reshape(-1)[0]), 1e-9)

    def update_bar(self, features: Mapping[str, Any] | Sequence[float], t: float | None = None) -> dict[str, Any]:
        if isinstance(features, Mapping):
            value = features.get("x_raw", None)
            if value is None:
                ordered = []
                for key in ("return", "volume", "spread", "volatility", "imbalance"):
                    if key in features:
                        ordered.append(features[key])
                value = ordered if ordered else [0.0]
            timestamp = float(features.get("timestamp", self.total_updates * self.bar_sec if t is None else t))
        else:
            value = features
            timestamp = self.total_updates * self.bar_sec if t is None else float(t)
        self.encoder.update({
            "timestamp": timestamp,
            "entity_id": "market",
            "x_raw": _arr(value),
        })
        feats = self.encoder.features(timestamp, "market")
        if self.promotion_gate is not None:
            signal = self.promotion_gate.update(feats, timestamp)
            if (
                self.promotion_adapt_low
                and signal.get("promote", False)
                and signal.get("reason") in {"persistent_high_residual", "hysteresis"}
                and hasattr(self.encoder, "promote_residual")
            ):
                absorbed = float(self.encoder.promote_residual(
                    "market",
                    strength=signal.get("promotion_strength", 0.0),
                    gain=self.promotion_adapt_gain,
                ))
                if absorbed > 0.0:
                    self.promotion_absorptions += 1
                    self.promotion_absorbed_norm += absorbed
                    feats = self.encoder.features(timestamp, "market")
            feats["promotion"] = signal
        self.total_updates += 1
        return feats

    def features(self) -> dict[str, Any]:
        try:
            return self.encoder.features(entity_id="market")
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

    def upper_features(self, mode: str | None = None) -> np.ndarray:
        feats = self.features()
        norm = self._norm(_arr(feats["x_low"]).size)
        mode = str(mode or self.upper_mode).lower()
        low = _arr(feats["x_low"]) / norm
        slope = _arr(feats["x_low_slope"]) / norm
        uncertainty = _arr(feats["x_low_uncertainty"]) / norm
        high_energy = _arr(feats["x_high_energy"]) / (norm * norm)
        high_persistence = _arr(feats["x_high_persistence"])
        if mode in {"low", "lf", "split"}:
            vals = np.concatenate([low, slope, uncertainty, high_energy, high_persistence])
        elif mode in {"high", "hf"}:
            vals = np.concatenate([_arr(feats["x_high"]) / norm, high_energy, high_persistence])
        elif mode in {"all", "allfreq", "all_freq"}:
            vals = np.concatenate([
                low,
                slope,
                _arr(feats["x_mid"]) / norm,
                _arr(feats["x_high"]) / norm,
                uncertainty,
                high_energy,
                high_persistence,
            ])
        else:
            raise ValueError(f"unknown upper mode: {mode}")
        if self.promotion_gate is not None:
            sig = self.promotion_gate.signal()
            vals = np.concatenate([
                vals,
                np.asarray([1.0 if sig.promote else 0.0, sig.promotion_strength, sig.shock_age], dtype=np.float64),
            ])
        return vals.astype(np.float32)

    def lower_features(
        self,
        current_target: Sequence[float] | None = None,
        current_position: Sequence[float] | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        feats = self.features()
        dim = _arr(feats["x_high"]).size
        norm = self._norm(dim)
        mode = str(mode or self.lower_mode).lower()
        target = np.zeros(dim, dtype=np.float64) if current_target is None else _arr(current_target)
        position = np.zeros(dim, dtype=np.float64) if current_position is None else _arr(current_position)
        if target.size != dim:
            target = np.resize(target, dim)
        if position.size != dim:
            position = np.resize(position, dim)
        gap = target - position
        high = _arr(feats["x_high"]) / norm
        high_delta = _arr(feats["x_high_delta"]) / norm
        high_energy = _arr(feats["x_high_energy"]) / (norm * norm)
        mid = _arr(feats["x_mid"]) / norm
        shock_age = _arr(feats["shock_age"])
        if mode in {"high", "hf", "split"}:
            vals = np.concatenate([gap, high, high_delta, high_energy, shock_age])
        elif mode in {"high_mid", "regime"}:
            vals = np.concatenate([gap, high, mid, high_energy, shock_age])
        elif mode in {"all", "allfreq", "all_freq"}:
            vals = np.concatenate([gap, _arr(feats["x_low"]) / norm, high, mid, shock_age])
        else:
            raise ValueError(f"unknown lower mode: {mode}")
        if self.promotion_gate is not None:
            sig = self.promotion_gate.signal()
            vals = np.concatenate([
                vals,
                np.asarray([1.0 if sig.promote else 0.0, sig.promotion_strength, sig.shock_age], dtype=np.float64),
            ])
        return vals.astype(np.float32)

    def summary(self) -> dict[str, Any]:
        feats = self.features()
        sig = (
            self.promotion_gate.signal()
            if self.promotion_gate is not None
            else None
        )
        return {
            "freq_low_norm": _energy(feats["x_low"]),
            "freq_high_energy": _energy(feats["x_high_energy"]),
            "freq_middle_norm": _energy(feats["x_mid"]),
            "freq_persistence": _energy(feats["x_high_persistence"]),
            "freq_updates": int(self.total_updates),
            "freq_method": self.method,
            "freq_promotion_flag": 1.0 if sig is not None and sig.promote else 0.0,
            "freq_promotion_strength": 0.0 if sig is None else float(sig.promotion_strength),
            "freq_promotion_age": 0.0 if sig is None else float(sig.shock_age),
            "freq_promotion_score": 0.0 if sig is None else float(sig.score),
            "freq_promotion_absorptions": int(self.promotion_absorptions),
            "freq_promotion_absorbed_norm": float(self.promotion_absorbed_norm),
        }
