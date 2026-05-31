"""On-policy Gaussian trading policy for Freq-HRL interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .interfaces import HighLevelDecision, HighLevelPlanner, LowLevelController, LowLevelDecision


def _arr(value: Any, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


def _gross_cap(target: np.ndarray, max_gross: float = 1.0) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    gross = float(np.sum(np.abs(target)))
    if gross > max_gross and gross > 1e-12:
        target = target * (max_gross / gross)
    return target


@dataclass
class PolicyGradientTradingParams:
    """Shared linear actor weights trained by episodic policy gradient."""

    upper_low: float = 1.0
    upper_mid: float = 0.20
    upper_high: float = 0.0
    upper_promotion: float = 0.40
    upper_position: float = -0.05
    upper_bias: float = 0.0
    lower_base_logit: float = 0.20
    lower_align: float = 0.20
    lower_energy: float = 0.0
    upper_std: float = 0.25
    lower_std: float = 0.20

    @classmethod
    def trainable_names(cls) -> tuple[str, ...]:
        return (
            "upper_low",
            "upper_mid",
            "upper_high",
            "upper_promotion",
            "upper_position",
            "upper_bias",
            "lower_base_logit",
            "lower_align",
            "lower_energy",
        )

    @classmethod
    def from_vector(cls, value: np.ndarray, template: "PolicyGradientTradingParams | None" = None) -> "PolicyGradientTradingParams":
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        names = cls.trainable_names()
        if arr.size != len(names):
            raise ValueError(f"expected {len(names)} policy-gradient parameters, got {arr.size}")
        payload = (template or cls()).to_mapping()
        payload.update({name: float(arr[i]) for i, name in enumerate(names)})
        return cls.from_mapping(payload)

    def to_vector(self) -> np.ndarray:
        mapping = self.to_mapping()
        return np.asarray([mapping[name] for name in self.trainable_names()], dtype=np.float64)

    def to_mapping(self) -> dict[str, float]:
        return {
            "upper_low": float(self.upper_low),
            "upper_mid": float(self.upper_mid),
            "upper_high": float(self.upper_high),
            "upper_promotion": float(self.upper_promotion),
            "upper_position": float(self.upper_position),
            "upper_bias": float(self.upper_bias),
            "lower_base_logit": float(self.lower_base_logit),
            "lower_align": float(self.lower_align),
            "lower_energy": float(self.lower_energy),
            "upper_std": float(self.upper_std),
            "lower_std": float(self.lower_std),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PolicyGradientTradingParams":
        payload = cls().to_mapping()
        payload.update({key: float(val) for key, val in dict(value).items()})
        payload["upper_std"] = max(payload["upper_std"], 1e-3)
        payload["lower_std"] = max(payload["lower_std"], 1e-3)
        return cls(**payload)


class PolicyGradientTradingPlanner(HighLevelPlanner):
    """Low-frequency Gaussian actor; deterministic mode uses the actor mean."""

    def __init__(
        self,
        params: PolicyGradientTradingParams | None = None,
        scale: float = 0.0014,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.params = params or PolicyGradientTradingParams()
        self.scale = float(scale)
        self.sample = bool(sample)
        self.rng = rng or np.random.default_rng(0)

    def _features(
        self,
        observation: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> np.ndarray:
        freq = dict(context.get("frequency", {}) or {})
        raw = _arr(observation.get("raw_signal", []))
        dim = raw.size if raw.size else int(context.get("n_assets", 1))
        x_low = _arr(freq.get("x_low", np.zeros(dim)), dim) / max(self.scale, 1e-9)
        x_mid = _arr(freq.get("x_mid", np.zeros(dim)), dim) / max(self.scale, 1e-9)
        x_high = _arr(freq.get("x_high", np.zeros(dim)), dim) / max(self.scale, 1e-9)
        position = _arr(observation.get("position", np.zeros(dim)), dim)
        promotion = dict(freq.get("promotion", {}) or {})
        promoted = bool(promotion.get("promote", False))
        strength = float(promotion.get("promotion_strength", 0.0)) if promoted else 0.0
        return np.stack([
            x_low,
            x_mid,
            x_high,
            strength * x_mid,
            position,
            np.ones(dim, dtype=np.float64),
        ], axis=1)

    def plan(
        self,
        observation: Mapping[str, Any],
        upper_features: np.ndarray,
        context: Mapping[str, Any] | None = None,
    ) -> HighLevelDecision:
        context = context or {}
        features = self._features(observation, context)
        p = self.params
        weights = np.asarray([
            p.upper_low,
            p.upper_mid,
            p.upper_high,
            p.upper_promotion,
            p.upper_position,
            p.upper_bias,
        ], dtype=np.float64)
        mean = np.clip(features @ weights, -4.0, 4.0)
        if self.sample:
            latent = mean + p.upper_std * self.rng.normal(size=mean.shape)
            action = _gross_cap(np.tanh(latent))
            coeff = (latent - mean) / max(p.upper_std * p.upper_std, 1e-9)
            grad = np.zeros(len(PolicyGradientTradingParams.trainable_names()), dtype=np.float64)
            grad[:6] = np.sum(coeff[:, None] * features, axis=0)
        else:
            latent = mean
            action = _gross_cap(np.tanh(mean))
            grad = np.zeros(len(PolicyGradientTradingParams.trainable_names()), dtype=np.float64)
        return HighLevelDecision(
            action=action,
            plan={"type": "pg_linear_target", "latent": latent.copy()},
            metadata={"policy_grad_logp": grad, "sampled": self.sample},
        )


class PolicyGradientTradingController(LowLevelController):
    """High-frequency Gaussian execution-speed actor."""

    def __init__(
        self,
        params: PolicyGradientTradingParams | None = None,
        scale: float = 0.0014,
        sample: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.params = params or PolicyGradientTradingParams()
        self.scale = float(scale)
        self.sample = bool(sample)
        self.rng = rng or np.random.default_rng(0)

    def act(
        self,
        observation: Mapping[str, Any],
        lower_features: np.ndarray,
        high_level_decision: HighLevelDecision,
        context: Mapping[str, Any] | None = None,
    ) -> LowLevelDecision:
        context = context or {}
        freq = dict(context.get("frequency", {}) or {})
        target = _arr(high_level_decision.action)
        position = _arr(observation.get("position", np.zeros_like(target)), target.size)
        x_high = _arr(freq.get("x_high", np.zeros_like(target)), target.size)
        energy = np.sqrt(np.maximum(_arr(freq.get("x_high_energy", np.zeros_like(target)), target.size), 0.0))
        gap = target - position
        align = np.tanh(np.sign(gap) * x_high / max(self.scale, 1e-9))
        energy_feature = np.tanh(energy / max(self.scale, 1e-9))
        p = self.params
        features = np.stack([
            np.ones(target.size, dtype=np.float64),
            align,
            energy_feature,
        ], axis=1)
        weights = np.asarray([p.lower_base_logit, p.lower_align, p.lower_energy], dtype=np.float64)
        mean = np.clip(features @ weights, -4.0, 4.0)
        if self.sample:
            latent = mean + p.lower_std * self.rng.normal(size=mean.shape)
            coeff = (latent - mean) / max(p.lower_std * p.lower_std, 1e-9)
            grad = np.zeros(len(PolicyGradientTradingParams.trainable_names()), dtype=np.float64)
            grad[6:9] = np.sum(coeff[:, None] * features, axis=0)
        else:
            latent = mean
            grad = np.zeros(len(PolicyGradientTradingParams.trainable_names()), dtype=np.float64)
        speed = 0.05 + 0.95 / (1.0 + np.exp(-latent))
        return LowLevelDecision(
            action={
                "execution_speed": np.clip(speed, 0.05, 1.0),
                "residual_order": np.zeros_like(target),
            },
            metadata={"policy_grad_logp": grad, "sampled": self.sample},
        )
