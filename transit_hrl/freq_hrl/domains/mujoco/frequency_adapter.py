"""Causal observation routing and actuation disturbances for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.responsibility_gauge import (
    CausalAuditAlignedGaugeFixer,
    CausalGaugeFixer,
)


DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
    "ood_chirp",
)

RESPONSIBILITY_MODES = (
    "additive",
    "causal_lf_transfer",
)

LOWER_ACTION_ROUTER_MODES = (
    "direct",
    "causal_ema_high_pass",
    "causal_ema_conservative_transfer",
    "causal_joint_band_projection",
    "causal_total_action_gauge",
    "causal_audit_aligned_gauge",
)

LOWER_ACTION_ROUTER_CONTRACTS = {
    "direct": "direct_latent_to_effective_lower_action_v1",
    "causal_ema_high_pass": (
        "latent_proposal_minus_scaled_prior_only_ema_baseline_with_observed_"
        "router_state_strength_and_effective_action_clipping_v3"
    ),
    "causal_ema_conservative_transfer": (
        "causal_prior_only_ema_split_with_removed_component_transferred_to_"
        "upper_and_exact_pre_split_action_execution_v4"
    ),
    "causal_joint_band_projection": (
        "causal_lower_lpf32_minus_upper_hpf8_transfer_and_exact_complement_"
        "with_pre_split_action_execution_v1"
    ),
    "causal_total_action_gauge": (
        "causal_total_action_ema_gauge_fixed_responsibility_with_exact_"
        "pre_split_action_execution_v1"
    ),
    "causal_audit_aligned_gauge": (
        "causal_total_action_gauge_fixed_adaptive_lpf32_hpf8_feedback_with_"
        "exact_pre_split_action_execution_v1"
    ),
}


def lower_action_router_contract(mode: str) -> str:
    """Return the versioned runtime contract for a supported router."""

    try:
        return LOWER_ACTION_ROUTER_CONTRACTS[str(mode)]
    except KeyError as exc:
        raise ValueError(f"unknown lower-action router mode: {mode}") from exc


@dataclass
class CausalBandDecomposer:
    """Two-rate causal EMA decomposition with no future observations."""

    slow_alpha: float = 0.04
    fast_alpha: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < float(self.slow_alpha) < float(self.fast_alpha) <= 1.0:
            raise ValueError("band alphas must satisfy 0 < slow < fast <= 1")
        self.slow_alpha = float(self.slow_alpha)
        self.fast_alpha = float(self.fast_alpha)
        self._slow: np.ndarray | None = None
        self._fast: np.ndarray | None = None
        self._previous: np.ndarray | None = None

    def reset(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        current = self._finite_vector(observation)
        self._slow = current.copy()
        self._fast = current.copy()
        self._previous = current.copy()
        return self._bands(current, np.zeros_like(current))

    def update(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        current = self._finite_vector(observation)
        if self._slow is None or self._fast is None or self._previous is None:
            return self.reset(current)
        if current.shape != self._slow.shape:
            raise ValueError("observation dimension changed within an episode")
        delta = current - self._previous
        self._slow += self.slow_alpha * (current - self._slow)
        self._fast += self.fast_alpha * (current - self._fast)
        self._previous = current.copy()
        return self._bands(current, delta)

    @staticmethod
    def _finite_vector(observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float64).reshape(-1)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("MuJoCo observation must be a finite vector")
        return value

    def _bands(
        self,
        current: np.ndarray,
        delta: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self._slow is None or self._fast is None:
            raise RuntimeError("decomposer must be reset before use")
        return {
            "raw": current.astype(np.float32, copy=True),
            "slow": self._slow.astype(np.float32, copy=True),
            "mid": (self._fast - self._slow).astype(np.float32, copy=False),
            "high": (current - self._fast).astype(np.float32, copy=False),
            "delta": np.asarray(delta, dtype=np.float32).copy(),
        }


@dataclass
class CausalResponsibilityTransfer:
    """Move prior lower LF bias to the next upper macro action causally."""

    mode: str = "additive"
    alpha: float = 0.04

    def __post_init__(self) -> None:
        if str(self.mode) not in RESPONSIBILITY_MODES:
            raise ValueError(f"unknown responsibility mode: {self.mode}")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("responsibility-transfer alpha must be in (0, 1]")
        self.mode = str(self.mode)
        self.alpha = float(self.alpha)
        self._raw_lower_lf: np.ndarray | None = None
        self._effective_transfer: np.ndarray | None = None
        self._upper_policy: np.ndarray | None = None
        self._upper_responsibility: np.ndarray | None = None

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("responsibility-transfer action_dim must be positive")
        zeros = np.zeros(int(action_dim), dtype=np.float64)
        self._raw_lower_lf = zeros.copy()
        self._effective_transfer = zeros.copy()
        self._upper_policy = zeros.copy()
        self._upper_responsibility = zeros.copy()

    @property
    def raw_lower_lf(self) -> np.ndarray:
        self._require_reset()
        return self._raw_lower_lf.astype(np.float32, copy=True)

    @property
    def effective_transfer(self) -> np.ndarray:
        self._require_reset()
        return self._effective_transfer.astype(np.float32, copy=True)

    def preview_upper(self, upper_policy: np.ndarray) -> dict[str, np.ndarray | float]:
        """Preview a boundary assignment without mutating filter state."""

        self._require_reset()
        policy = self._aligned_action(upper_policy)
        requested = (
            self._raw_lower_lf.copy()
            if self.mode == "causal_lf_transfer"
            else np.zeros_like(policy)
        )
        lower_headroom = -1.0 - policy
        upper_headroom = 1.0 - policy
        effective = np.clip(requested, lower_headroom, upper_headroom)
        responsibility = policy + effective
        saturated = np.abs(effective - requested) > 1e-12
        return {
            "upper_policy": policy.astype(np.float32, copy=True),
            "requested_transfer": requested.astype(np.float32, copy=True),
            "effective_transfer": effective.astype(np.float32, copy=True),
            "upper_responsibility": responsibility.astype(np.float32, copy=True),
            "headroom_saturation_rate": float(np.mean(saturated)),
        }

    def begin_macro(self, upper_policy: np.ndarray) -> dict[str, np.ndarray | float]:
        assignment = self.preview_upper(upper_policy)
        self._upper_policy = np.asarray(
            assignment["upper_policy"], dtype=np.float64
        ).copy()
        self._effective_transfer = np.asarray(
            assignment["effective_transfer"], dtype=np.float64
        ).copy()
        self._upper_responsibility = np.asarray(
            assignment["upper_responsibility"], dtype=np.float64
        ).copy()
        return assignment

    def split_lower(self, raw_lower: np.ndarray) -> dict[str, np.ndarray]:
        """Return the responsibility split and update LF state for the future."""

        self._require_reset()
        raw = self._aligned_action(raw_lower)
        lower_responsibility = raw - self._effective_transfer
        original_total = self._upper_policy + raw
        reassigned_total = self._upper_responsibility + lower_responsibility
        reconstruction_error = reassigned_total - original_total
        before = self._raw_lower_lf.copy()
        self._raw_lower_lf += self.alpha * (raw - self._raw_lower_lf)
        return {
            "raw_lower": raw.astype(np.float32, copy=True),
            "lower_responsibility": lower_responsibility.astype(
                np.float32, copy=True
            ),
            "effective_transfer": self._effective_transfer.astype(
                np.float32, copy=True
            ),
            "raw_lower_lf_before": before.astype(np.float32, copy=True),
            "raw_lower_lf_after": self._raw_lower_lf.astype(
                np.float32, copy=True
            ),
            "reconstruction_error": reconstruction_error.astype(
                np.float64, copy=True
            ),
        }

    def _require_reset(self) -> None:
        if any(value is None for value in (
            self._raw_lower_lf,
            self._effective_transfer,
            self._upper_policy,
            self._upper_responsibility,
        )):
            raise RuntimeError("responsibility transfer must be reset before use")

    def _aligned_action(self, action: np.ndarray) -> np.ndarray:
        self._require_reset()
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        if (
            value.shape != self._raw_lower_lf.shape
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("responsibility action must be finite and aligned")
        return value


@dataclass
class CausalLowerActionRouter:
    """Turn a latent lower proposal into a causally high-passed action effect.

    The EMA baseline is computed only from proposals available before the
    current action.  Its current value is exposed through ``context`` so the
    policy observes all router state.  ``direct`` exactly preserves the legacy
    action path and exposes the preceding effective lower action.  The
    conservative-transfer mode returns the removed component as a causal upper
    transfer, making transfer plus effective action reconstruct the latent
    proposal even when the effective branch clips.  Joint-band projection uses
    the same causal windows as the audit: it transfers lower LPF32 minus upper
    HPF8, then assigns the exact complement to the lower responsibility.
    Total-action gauge mode instead computes a causal low-pass of the additive
    total and is invariant to the raw upper/lower factorization at full strength.
    Audit-aligned gauge mode adapts that low-pass cutoff from the registered
    upper-HPF and lower-LPF budget imbalance.
    """

    mode: str = "direct"
    alpha: float = 0.10
    strength: float = 1.0
    upper_rms_budget: float = 0.075
    lower_rms_budget: float = 0.0475
    audit_adaptation_rate: float = 0.03

    def __post_init__(self) -> None:
        if str(self.mode) not in LOWER_ACTION_ROUTER_MODES:
            raise ValueError(f"unknown lower-action router mode: {self.mode}")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("lower-action router alpha must be in (0, 1]")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("lower-action router strength must be in [0, 1]")
        if (
            not np.isfinite(float(self.upper_rms_budget))
            or float(self.upper_rms_budget) <= 0.0
            or not np.isfinite(float(self.lower_rms_budget))
            or float(self.lower_rms_budget) <= 0.0
        ):
            raise ValueError("lower-action router RMS budgets must be positive")
        if (
            not np.isfinite(float(self.audit_adaptation_rate))
            or float(self.audit_adaptation_rate) < 0.0
        ):
            raise ValueError(
                "lower-action router audit adaptation rate must be non-negative"
            )
        self.mode = str(self.mode)
        self.alpha = float(self.alpha)
        self.strength = float(self.strength)
        self.upper_rms_budget = float(self.upper_rms_budget)
        self.lower_rms_budget = float(self.lower_rms_budget)
        self.audit_adaptation_rate = float(self.audit_adaptation_rate)
        self._baseline: np.ndarray | None = None
        self._previous_effective: np.ndarray | None = None
        self._joint_upper_history: list[np.ndarray] = []
        self._joint_lower_history: list[np.ndarray] = []
        self._gauge_fixer: (
            CausalGaugeFixer | CausalAuditAlignedGaugeFixer | None
        ) = None

    def reset(self, action_dim: int) -> None:
        if int(action_dim) < 1:
            raise ValueError("lower-action router action_dim must be positive")
        zeros = np.zeros(int(action_dim), dtype=np.float64)
        self._baseline = zeros.copy()
        self._previous_effective = zeros.copy()
        self._joint_upper_history = []
        self._joint_lower_history = []
        if self.mode == "causal_total_action_gauge":
            self._gauge_fixer = CausalGaugeFixer(
                alpha=self.alpha, strength=self.strength
            )
        elif self.mode == "causal_audit_aligned_gauge":
            self._gauge_fixer = CausalAuditAlignedGaugeFixer(
                upper_window=8,
                lower_window=32,
                upper_rms_budget=self.upper_rms_budget,
                lower_rms_budget=self.lower_rms_budget,
                initial_alpha=self.alpha,
                adaptation_rate=self.audit_adaptation_rate,
                strength=self.strength,
            )
        else:
            self._gauge_fixer = None
        if self._gauge_fixer is not None:
            self._gauge_fixer.reset(int(action_dim))

    @property
    def context(self) -> np.ndarray:
        self._require_reset()
        if self._gauge_fixer is not None:
            return self._gauge_fixer.context
        value = (
            self._previous_effective
            if self.mode == "direct"
            else self._baseline
        )
        return value.astype(np.float32, copy=True)

    def route(
        self,
        latent_action: np.ndarray,
        *,
        upper_action: np.ndarray | None = None,
        action_limit: float = 1.0,
    ) -> dict[str, np.ndarray | float]:
        self._require_reset()
        latent = np.asarray(latent_action, dtype=np.float64).reshape(-1)
        limit = float(action_limit)
        if (
            latent.shape != self._baseline.shape
            or not np.all(np.isfinite(latent))
        ):
            raise ValueError("latent lower action must be finite and aligned")
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError("lower-action router limit must be positive")

        baseline_before = self._baseline.copy()
        audit_alpha_before = 0.0
        audit_alpha_after = 0.0
        audit_band_imbalance = 0.0
        if self.mode in {
            "causal_joint_band_projection",
            "causal_total_action_gauge",
            "causal_audit_aligned_gauge",
        }:
            if upper_action is None:
                raise ValueError(
                    "selected responsibility router requires the current upper action"
                )
            upper = np.asarray(upper_action, dtype=np.float64).reshape(-1)
            if upper.shape != latent.shape or not np.all(np.isfinite(upper)):
                raise ValueError(
                    "joint-band upper action must be finite and aligned"
                )
            if self.mode in {
                "causal_total_action_gauge",
                "causal_audit_aligned_gauge",
            }:
                if self._gauge_fixer is None:
                    raise RuntimeError("total-action gauge router is not initialized")
                fixed = self._gauge_fixer.split(
                    upper,
                    latent,
                    lower_limit=limit,
                )
                audit_alpha_before = float(fixed.get("alpha_before", 0.0))
                audit_alpha_after = float(fixed.get("alpha_after", 0.0))
                audit_band_imbalance = float(fixed.get(
                    "normalized_band_imbalance", 0.0
                ))
                transfer_target = np.asarray(
                    fixed["canonical_upper"], dtype=np.float64
                ) - upper
                requested_transfer = np.asarray(
                    fixed["transfer"], dtype=np.float64
                )
                requested = np.asarray(fixed["lower"], dtype=np.float64)
            else:
                self._joint_upper_history.append(upper.copy())
                self._joint_lower_history.append(latent.copy())
                self._joint_upper_history = self._joint_upper_history[-8:]
                self._joint_lower_history = self._joint_lower_history[-32:]
                upper_low = np.mean(self._joint_upper_history, axis=0)
                upper_high = upper - upper_low
                lower_low = np.mean(self._joint_lower_history, axis=0)
                transfer_target = lower_low - upper_high
                requested_transfer = self.strength * transfer_target
                requested = latent - requested_transfer
        else:
            if upper_action is not None:
                raise ValueError(
                    "upper_action is only valid for joint-band projection"
                )
            upper = np.zeros_like(latent)
            requested = (
                latent
                if self.mode == "direct"
                else latent - self.strength * baseline_before
            )
        effective = np.clip(requested, -limit, limit)
        clipped = np.abs(effective - requested) > 1e-12
        if self.mode == "causal_joint_band_projection":
            self._baseline = transfer_target.copy()
        elif self.mode in {
            "causal_total_action_gauge",
            "causal_audit_aligned_gauge",
        }:
            if self._gauge_fixer is None:
                raise RuntimeError("total-action gauge router is not initialized")
            self._baseline = np.asarray(
                self._gauge_fixer.context, dtype=np.float64
            )
        elif self.mode != "direct":
            self._baseline += self.alpha * (latent - self._baseline)
        removed = latent - effective
        upper_transfer = (
            removed
            if self.mode in {
                "causal_ema_conservative_transfer",
                "causal_joint_band_projection",
                "causal_total_action_gauge",
                "causal_audit_aligned_gauge",
            }
            else np.zeros_like(removed)
        )
        self._previous_effective = effective.copy()
        return {
            "latent": latent.astype(np.float32, copy=True),
            "baseline_before": baseline_before.astype(np.float32, copy=True),
            "baseline_after": self._baseline.astype(np.float32, copy=True),
            "requested_effective": requested.astype(np.float32, copy=True),
            "effective": effective.astype(np.float32, copy=True),
            "removed_low_frequency": removed.astype(np.float32, copy=True),
            "upper_transfer": upper_transfer.astype(np.float32, copy=True),
            "transfer_reconstruction_error": (
                upper_transfer + effective - latent
            ).astype(np.float64, copy=True),
            "clip_rate": float(np.mean(clipped)),
            "audit_alpha_before": audit_alpha_before,
            "audit_alpha_after": audit_alpha_after,
            "audit_normalized_band_imbalance": audit_band_imbalance,
        }

    def _require_reset(self) -> None:
        if self._baseline is None or self._previous_effective is None:
            raise RuntimeError("lower-action router must be reset before use")


def action_from_unit_box(
    normalized_action: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map a clipped action in [-1, 1] to a finite Box action space."""

    unit = np.clip(
        np.asarray(normalized_action, dtype=np.float64).reshape(-1),
        -1.0,
        1.0,
    )
    lower = np.asarray(low, dtype=np.float64).reshape(-1)
    upper = np.asarray(high, dtype=np.float64).reshape(-1)
    if (
        unit.shape != lower.shape
        or lower.shape != upper.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(upper <= lower)
    ):
        raise ValueError("action bounds must be aligned, finite, and ordered")
    return (lower + 0.5 * (unit + 1.0) * (upper - lower)).astype(
        np.float32
    )


def deterministic_actuation_disturbance(
    *,
    mode: str,
    step: int,
    action_dim: int,
    seed: int,
    horizon: int,
) -> np.ndarray:
    """Return a deterministic normalized-action disturbance for one step."""

    name = str(mode)
    if name not in DISTURBANCE_MODES:
        raise ValueError(f"unknown MuJoCo disturbance mode: {name}")
    if int(step) < 0 or int(action_dim) < 1 or int(horizon) < 1:
        raise ValueError("step, action_dim, and horizon must be valid")
    if name == "standard":
        return np.zeros(int(action_dim), dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=int(action_dim))
    directions = rng.choice((-1.0, 1.0), size=int(action_dim))
    t = float(step)
    low = 0.16 * np.sin(2.0 * np.pi * t / 160.0 + phases)
    high = 0.08 * np.sin(2.0 * np.pi * t / 5.0 + phases) * directions
    if name == "low_frequency":
        value = low
    elif name == "high_frequency":
        value = high
    elif name == "mixed":
        shift = 0.10 * directions if int(step) >= int(horizon) // 2 else 0.0
        value = low + high + shift
    else:
        progress = min(max(t / max(float(horizon - 1), 1.0), 0.0), 1.0)
        cycles = 0.5 * t / 160.0 + 8.0 * progress * progress
        value = 0.14 * np.sin(2.0 * np.pi * cycles + phases)
    return np.asarray(value, dtype=np.float32)
