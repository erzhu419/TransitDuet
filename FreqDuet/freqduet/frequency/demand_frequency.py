"""
Causal demand frequency features for FreqDuet.

Four causal decomposers are available:

* ``ema``: low/fast exponential moving averages, useful as a stable baseline.
* ``haar``: trailing-window Haar-style multiscale averages. It is still fully
  causal: the low component is a long trailing boxcar average, the fast
  component is a short trailing boxcar average, and the high component is their
  residual.
* ``harmonic``: recursive least-squares dynamic harmonic smoother on
  ``log1p(arrival_rate)``. This is the current main non-wavelet intensity path:
  low frequency is the causal harmonic forecast, high frequency is innovation.
* ``harmonic_nb``: count-aware dynamic harmonic state-space model with a
  negative-binomial extended-Kalman update and robust innovations.
* ``raw_history``: no frequency split; exposes trailing realized-demand bins as
  the RawHistory ablation required by `GPT.md`.

For the harmonic path, an optional history-auxiliary feature block can append
short causal residual bins relative to the harmonic low-frequency estimate.
This keeps the historical-prior LF/HF split as the main signal while giving
the policy local memory in scenarios where raw history is competitive.

This is the first code path aligned with the wavelet direction in `GPT.md`;
MODWT/learnable wavelets can replace these classes behind the same API.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import math

import numpy as np

from .intensity_estimator import (
    CausalHarmonicBandState,
    CausalNegativeBinomialHarmonicBandState,
)
from .promotion_gate import CausalPromotionGate


def _alpha_from_period(update_interval_s, period_s):
    """Convert an EMA time constant to an update coefficient."""
    period_s = max(float(period_s), 1e-6)
    update_interval_s = max(float(update_interval_s), 1e-6)
    return float(1.0 - math.exp(-update_interval_s / period_s))


@dataclass
class _EMABandState:
    low_alpha: float
    fast_alpha: float
    energy_alpha: float

    def __post_init__(self):
        self.low = 0.0
        self.fast = 0.0
        self.prev_low = 0.0
        self.high = 0.0
        self.prev_high = 0.0
        self.high_energy = 0.0
        self.n = 0

    def reset(self):
        self.low = 0.0
        self.fast = 0.0
        self.prev_low = 0.0
        self.high = 0.0
        self.prev_high = 0.0
        self.high_energy = 0.0
        self.n = 0

    def update(self, value):
        value = float(value)
        if self.n == 0:
            self.low = value
            self.fast = value
            self.prev_low = value
            self.high = 0.0
            self.prev_high = 0.0
            self.high_energy = 0.0
            self.n = 1
            return

        self.prev_low = self.low
        self.prev_high = self.high
        self.low = (1.0 - self.low_alpha) * self.low + self.low_alpha * value
        self.fast = (1.0 - self.fast_alpha) * self.fast + self.fast_alpha * value
        self.high = self.fast - self.low
        self.high_energy = (
            (1.0 - self.energy_alpha) * self.high_energy
            + self.energy_alpha * (self.high ** 2)
        )
        self.n += 1

    @property
    def low_slope(self):
        return self.low - self.prev_low if self.n > 1 else 0.0

    @property
    def high_slope(self):
        return self.high - self.prev_high if self.n > 1 else 0.0


class _CausalHaarBandState:
    """Trailing-window two-scale Haar-like splitter."""

    def __init__(self, update_interval_s, low_period_s, fast_period_s,
                 energy_alpha):
        self.low_window = max(1, int(round(float(low_period_s) / update_interval_s)))
        self.fast_window = max(1, int(round(float(fast_period_s) / update_interval_s)))
        self.energy_alpha = float(energy_alpha)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.low_window)
        self.fast_samples = deque(maxlen=self.fast_window)
        self.low = 0.0
        self.fast = 0.0
        self.prev_low = 0.0
        self.high = 0.0
        self.prev_high = 0.0
        self.high_energy = 0.0
        self.n = 0

    def update(self, value):
        value = float(value)
        self.prev_low = self.low
        self.prev_high = self.high
        self.samples.append(value)
        self.fast_samples.append(value)
        self.low = float(np.mean(self.samples)) if self.samples else 0.0
        self.fast = float(np.mean(self.fast_samples)) if self.fast_samples else 0.0
        self.high = self.fast - self.low
        if self.n == 0:
            self.high_energy = 0.0
        else:
            self.high_energy = (
                (1.0 - self.energy_alpha) * self.high_energy
                + self.energy_alpha * (self.high ** 2)
            )
        self.n += 1

    @property
    def low_slope(self):
        return self.low - self.prev_low if self.n > 1 else 0.0

    @property
    def high_slope(self):
        return self.high - self.prev_high if self.n > 1 else 0.0


class _RawHistoryBandState:
    """Trailing realized-demand bins for the RawHistory baseline."""

    def __init__(self, history_bins):
        self.history_bins = max(1, int(history_bins))
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=self.history_bins)
        self.low = 0.0
        self.prev_low = 0.0
        self.high = 0.0
        self.prev_high = 0.0
        self.high_energy = 0.0
        self.n = 0

    def update(self, value):
        value = float(value)
        self.prev_low = self.low
        self.prev_high = self.high
        self.samples.append(value)
        arr = np.asarray(self.samples, dtype=np.float64)
        self.low = value
        self.high = value - float(arr.mean()) if arr.size else 0.0
        self.high_energy = float(arr.var()) if arr.size > 1 else 0.0
        self.n += 1

    @property
    def low_slope(self):
        return self.low - self.prev_low if self.n > 1 else 0.0

    @property
    def high_slope(self):
        return self.high - self.prev_high if self.n > 1 else 0.0

    def history(self, bins):
        values = list(self.samples)[-bins:]
        if len(values) < bins:
            values = [0.0] * (bins - len(values)) + values
        return values


class DemandFrequencyTracker:
    """Online frequency splitter for realized passenger-arrival counts.

    Values are converted to passengers per minute before filtering, making the
    features independent of the simulator's passenger update interval.
    """

    upper_feature_dim = 4
    lower_feature_dim = 4

    def __init__(
        self,
        update_interval_s=1.0,
        low_period_s=3600.0,
        fast_period_s=300.0,
        energy_period_s=600.0,
        middle_period_s=1800.0,
        global_demand_norm=50.0,
        local_demand_norm=10.0,
        slope_norm=5.0,
        method="haar",
        forecast_horizon_s=1800.0,
        forecast_mode="causal",
        upper_history_bins=6,
        lower_history_bins=6,
        od_features=False,
        upper_mode="low",
        lower_mode="high",
        bin_sec=None,
        harmonic_period_s=50400.0,
        fourier_k=4,
        harmonic_forgetting=0.995,
        harmonic_prior_var=100.0,
        harmonic_residual_period_s=None,
        harmonic_prior=None,
        harmonic_nb_dispersion=20.0,
        harmonic_process_var=1e-5,
        harmonic_innovation_clip=3.0,
        harmonic_observation_var_floor=1e-3,
        lower_high_noise_z=0.0,
        lower_high_noise_energy_min=0.0,
        promotion_enable=False,
        promotion_window_s=900.0,
        promotion_residual_threshold=1.0,
        promotion_persistence_ratio=0.35,
        promotion_cooldown_s=600.0,
        promotion_state_features=True,
        promotion_adapt_low=False,
        promotion_adapt_gain=0.10,
        promotion_adapt_strength_min=0.15,
        promotion_adapt_local=False,
        promotion_adapt_active_only=False,
        promotion_adapt_requires_persistent=False,
        promotion_adapt_high_energy_min=0.0,
        promotion_residual_mode="abs",
        history_aux_enable=False,
        history_aux_upper_bins=0,
        history_aux_lower_bins=0,
        history_aux_mode="residual",
        history_aux_clip=3.0,
        history_aux_high_energy_max=0.0,
        history_aux_high_energy_width=0.0,
        history_aux_min_weight=0.0,
    ):
        self.update_interval_s = float(update_interval_s)
        self.method = str(method).lower()
        self.upper_mode = str(upper_mode or "low").lower()
        self.lower_mode = str(lower_mode or "high").lower()
        self.od_features_enabled = bool(od_features)
        self.upper_history_bins = max(1, int(upper_history_bins))
        self.lower_history_bins = max(1, int(lower_history_bins))
        self.history_aux_enabled = bool(history_aux_enable)
        self.history_aux_upper_bins = max(0, int(history_aux_upper_bins))
        self.history_aux_lower_bins = max(0, int(history_aux_lower_bins))
        self.history_aux_mode = str(history_aux_mode or "residual").lower()
        if self.history_aux_mode not in {"residual", "innovation", "raw", "rate"}:
            raise ValueError(
                f"Unknown history auxiliary mode: {history_aux_mode}")
        self.history_aux_clip = max(float(history_aux_clip), 0.0)
        self.history_aux_high_energy_max = max(
            float(history_aux_high_energy_max), 0.0)
        self.history_aux_high_energy_width = max(
            float(history_aux_high_energy_width), 0.0)
        self.history_aux_min_weight = float(np.clip(
            history_aux_min_weight, 0.0, 1.0))
        self.bin_sec = float(bin_sec) if bin_sec else self.update_interval_s
        self.bin_steps = max(1, int(round(self.bin_sec / self.update_interval_s)))
        self.bin_interval_s = self.bin_steps * self.update_interval_s
        self.forecast_mode = str(forecast_mode or "causal").strip().lower()
        if self.forecast_mode not in {
                "causal", "linear_bins", "legacy_seconds_linear"}:
            raise ValueError(
                "forecast_mode must be causal, linear_bins, or "
                "legacy_seconds_linear")
        forecast_interval_s = (
            self.update_interval_s
            if self.forecast_mode == "legacy_seconds_linear"
            else self.bin_interval_s
        )
        self.forecast_steps = max(
            float(forecast_horizon_s) / max(forecast_interval_s, 1e-6), 1.0)
        energy_alpha = _alpha_from_period(self.bin_interval_s, energy_period_s)
        middle_alpha = _alpha_from_period(self.bin_interval_s, middle_period_s)
        residual_alpha = _alpha_from_period(
            self.bin_interval_s, harmonic_residual_period_s or fast_period_s)
        self._pending_station = defaultdict(float)
        self._pending_od = defaultdict(float)
        self._pending_steps = 0
        self.harmonic_prior = harmonic_prior or {}
        self.lower_high_noise_z = max(float(lower_high_noise_z), 0.0)
        self.lower_high_noise_energy_min = max(
            float(lower_high_noise_energy_min), 0.0)
        self.promotion_enabled = bool(promotion_enable)
        self.promotion_state_features = (
            self.promotion_enabled and bool(promotion_state_features))
        self.promotion_feature_dim = 3 if self.promotion_state_features else 0
        self.promotion_adapt_low = bool(promotion_adapt_low)
        self.promotion_adapt_gain = max(float(promotion_adapt_gain), 0.0)
        self.promotion_adapt_strength_min = max(
            float(promotion_adapt_strength_min), 0.0)
        self.promotion_adapt_local = bool(promotion_adapt_local)
        self.promotion_adapt_active_only = bool(promotion_adapt_active_only)
        self.promotion_adapt_requires_persistent = bool(
            promotion_adapt_requires_persistent)
        self.promotion_adapt_high_energy_min = max(
            float(promotion_adapt_high_energy_min), 0.0)
        self.promotion_residual_mode = str(
            promotion_residual_mode or "abs").lower()
        self.promotion_absorptions = 0
        self.promotion_absorbed_rate = 0.0

        if self.method == "ema":
            low_alpha = _alpha_from_period(update_interval_s, low_period_s)
            fast_alpha = _alpha_from_period(update_interval_s, fast_period_s)
            self._state_factory = lambda: _EMABandState(
                low_alpha=low_alpha,
                fast_alpha=fast_alpha,
                energy_alpha=energy_alpha)
        elif self.method in {"haar", "causal_haar"}:
            self.method = "haar"
            self._state_factory = lambda: _CausalHaarBandState(
                update_interval_s=self.bin_interval_s,
                low_period_s=low_period_s,
                fast_period_s=fast_period_s,
                energy_alpha=energy_alpha)
        elif self.method in {
            "harmonic", "dynamic_harmonic", "harmonic_rls",
        }:
            self.method = "harmonic"
            self._state_factory = lambda prior_theta=None: CausalHarmonicBandState(
                update_interval_s=self.bin_interval_s,
                period_s=harmonic_period_s,
                fourier_k=fourier_k,
                forgetting_factor=harmonic_forgetting,
                prior_var=harmonic_prior_var,
                energy_alpha=energy_alpha,
                residual_alpha=residual_alpha,
                middle_alpha=middle_alpha,
                prior_theta=prior_theta)
        elif self.method in {
            "harmonic_nb", "dynamic_harmonic_nb", "count_harmonic",
        }:
            self.method = "harmonic_nb"
            self._state_factory = (
                lambda prior_theta=None:
                CausalNegativeBinomialHarmonicBandState(
                    update_interval_s=self.bin_interval_s,
                    period_s=harmonic_period_s,
                    fourier_k=fourier_k,
                    forgetting_factor=harmonic_forgetting,
                    prior_var=harmonic_prior_var,
                    energy_alpha=energy_alpha,
                    residual_alpha=residual_alpha,
                    middle_alpha=middle_alpha,
                    prior_theta=prior_theta,
                    nb_dispersion=harmonic_nb_dispersion,
                    process_var=harmonic_process_var,
                    innovation_clip=harmonic_innovation_clip,
                    observation_var_floor=(
                        harmonic_observation_var_floor),
                )
            )
        elif self.method in {"raw_history", "history"}:
            self.method = "raw_history"
            self.od_features_enabled = False
            self.history_aux_enabled = False
            max_bins = max(self.upper_history_bins, self.lower_history_bins)
            self._state_factory = lambda: _RawHistoryBandState(max_bins)
        else:
            raise ValueError(f"Unknown demand frequency method: {method}")

        self._history_aux_max_bins = max(
            1, self.history_aux_upper_bins, self.history_aux_lower_bins)
        self._global_history_aux = deque(maxlen=self._history_aux_max_bins)
        self._local_history_aux = defaultdict(
            lambda: deque(maxlen=self._history_aux_max_bins))
        self.upper_feature_dim = self._upper_dim_for_mode(self.upper_mode)
        self.lower_feature_dim = self._lower_dim_for_mode(self.lower_mode)

        self.global_state = self._new_state(self._prior_for("global", None))
        self.local_states = {}
        self.od_states = {}
        self._od_summary_cache = (0.0, 0.0, 0)
        self._od_summary_cache_updates = -1
        self.global_promotion_gate = (
            CausalPromotionGate(
                update_interval_s=self.bin_interval_s,
                window_s=promotion_window_s,
                residual_threshold=promotion_residual_threshold,
                persistence_ratio=promotion_persistence_ratio,
                cooldown_s=promotion_cooldown_s,
                residual_mode=self.promotion_residual_mode,
            )
            if self.promotion_enabled else None
        )
        self.local_promotion_gates = {}

        self.global_demand_norm = max(float(global_demand_norm), 1e-6)
        self.local_demand_norm = max(float(local_demand_norm), 1e-6)
        self.slope_norm = max(float(slope_norm), 1e-6)
        self.total_updates = 0

    @classmethod
    def from_config(cls, cfg, update_interval_s=1.0):
        cfg = cfg or {}
        promotion_cfg = cfg.get("promotion", {}) or {}
        window_s = promotion_cfg.get("window_s", None)
        if window_s is None:
            window_s = float(promotion_cfg.get("window_min", 15.0)) * 60.0
        cooldown_s = promotion_cfg.get("cooldown_s", None)
        if cooldown_s is None:
            cooldown_s = float(promotion_cfg.get("cooldown_min", 10.0)) * 60.0
        return cls(
            update_interval_s=update_interval_s,
            low_period_s=cfg.get("low_period_s", 3600.0),
            fast_period_s=cfg.get("fast_period_s", 300.0),
            energy_period_s=cfg.get("energy_period_s", 600.0),
            middle_period_s=cfg.get("middle_period_s", 1800.0),
            global_demand_norm=cfg.get("global_demand_norm", 50.0),
            local_demand_norm=cfg.get("local_demand_norm", 10.0),
            slope_norm=cfg.get("slope_norm", 5.0),
            method=cfg.get("method", "haar"),
            forecast_horizon_s=cfg.get("forecast_horizon_s", 1800.0),
            forecast_mode=cfg.get("forecast_mode", "causal"),
            upper_history_bins=cfg.get("upper_history_bins", 6),
            lower_history_bins=cfg.get("lower_history_bins", 6),
            od_features=cfg.get("od_features", False),
            upper_mode=cfg.get("upper_mode", "low"),
            lower_mode=cfg.get("lower_mode", "high"),
            bin_sec=cfg.get("bin_sec", None),
            harmonic_period_s=cfg.get("harmonic_period_s", 50400.0),
            fourier_k=cfg.get("fourier_K", cfg.get("fourier_k", 4)),
            harmonic_forgetting=cfg.get("harmonic_forgetting", 0.995),
            harmonic_prior_var=cfg.get("harmonic_prior_var", 100.0),
            harmonic_residual_period_s=cfg.get(
                "harmonic_residual_period_s",
                cfg.get("fast_period_s", 300.0)),
            harmonic_prior=cfg.get("harmonic_prior", None),
            harmonic_nb_dispersion=cfg.get(
                "harmonic_nb_dispersion", 20.0),
            harmonic_process_var=cfg.get(
                "harmonic_process_var", 1e-5),
            harmonic_innovation_clip=cfg.get(
                "harmonic_innovation_clip", 3.0),
            harmonic_observation_var_floor=cfg.get(
                "harmonic_observation_var_floor", 1e-3),
            lower_high_noise_z=cfg.get("lower_high_noise_z", 0.0),
            lower_high_noise_energy_min=cfg.get(
                "lower_high_noise_energy_min", 0.0),
            promotion_enable=promotion_cfg.get(
                "enable", cfg.get("promotion_enable", False)),
            promotion_window_s=window_s,
            promotion_residual_threshold=promotion_cfg.get(
                "residual_threshold", cfg.get("promotion_residual_threshold", 1.0)),
            promotion_persistence_ratio=promotion_cfg.get(
                "persistence_ratio", cfg.get("promotion_persistence_ratio", 0.35)),
            promotion_cooldown_s=cooldown_s,
            promotion_state_features=promotion_cfg.get(
                "state_features", cfg.get("promotion_state_features", True)),
            promotion_adapt_low=promotion_cfg.get(
                "adapt_low", cfg.get("promotion_adapt_low", False)),
            promotion_adapt_gain=promotion_cfg.get(
                "adapt_gain", cfg.get("promotion_adapt_gain", 0.10)),
            promotion_adapt_strength_min=promotion_cfg.get(
                "adapt_strength_min",
                cfg.get("promotion_adapt_strength_min", 0.15)),
            promotion_adapt_local=promotion_cfg.get(
                "adapt_local", cfg.get("promotion_adapt_local", False)),
            promotion_adapt_active_only=promotion_cfg.get(
                "adapt_active_only",
                cfg.get("promotion_adapt_active_only", False)),
            promotion_adapt_requires_persistent=promotion_cfg.get(
                "adapt_requires_persistent",
                cfg.get("promotion_adapt_requires_persistent", False)),
            promotion_adapt_high_energy_min=promotion_cfg.get(
                "adapt_high_energy_min",
                cfg.get("promotion_adapt_high_energy_min", 0.0)),
            promotion_residual_mode=promotion_cfg.get(
                "residual_mode", cfg.get("promotion_residual_mode", "abs")),
            history_aux_enable=(cfg.get("history_aux", {}) or {}).get(
                "enable", cfg.get("history_aux_enable", False)),
            history_aux_upper_bins=(cfg.get("history_aux", {}) or {}).get(
                "upper_bins", cfg.get("history_aux_upper_bins", 0)),
            history_aux_lower_bins=(cfg.get("history_aux", {}) or {}).get(
                "lower_bins", cfg.get("history_aux_lower_bins", 0)),
            history_aux_mode=(cfg.get("history_aux", {}) or {}).get(
                "mode", cfg.get("history_aux_mode", "residual")),
            history_aux_clip=(cfg.get("history_aux", {}) or {}).get(
                "clip", cfg.get("history_aux_clip", 3.0)),
            history_aux_high_energy_max=(cfg.get("history_aux", {}) or {}).get(
                "high_energy_max",
                cfg.get("history_aux_high_energy_max", 0.0)),
            history_aux_high_energy_width=(cfg.get("history_aux", {}) or {}).get(
                "high_energy_width",
                cfg.get("history_aux_high_energy_width", 0.0)),
            history_aux_min_weight=(cfg.get("history_aux", {}) or {}).get(
                "min_weight", cfg.get("history_aux_min_weight", 0.0)),
        )

    def _new_state(self, prior_theta=None):
        if self.method in {"harmonic", "harmonic_nb"}:
            return self._state_factory(prior_theta)
        return self._state_factory()

    def _prior_for(self, kind, key):
        if not self.harmonic_prior:
            return None
        if kind == "global":
            return self.harmonic_prior.get("global")
        priors = self.harmonic_prior.get(kind, {})
        return priors.get(key)

    def _get_local_state(self, key):
        if key not in self.local_states:
            self.local_states[key] = self._new_state(self._prior_for("local", key))
        return self.local_states[key]

    def _get_od_state(self, key):
        if key not in self.od_states:
            self.od_states[key] = self._new_state(self._prior_for("od", key))
        return self.od_states[key]

    def _get_local_promotion_gate(self, key):
        if key not in self.local_promotion_gates:
            self.local_promotion_gates[key] = CausalPromotionGate(
                update_interval_s=self.bin_interval_s,
                window_s=self.global_promotion_gate.window_bins * self.bin_interval_s,
                residual_threshold=self.global_promotion_gate.residual_threshold,
                persistence_ratio=self.global_promotion_gate.persistence_ratio,
                cooldown_s=self.global_promotion_gate.cooldown_bins * self.bin_interval_s,
                residual_mode=self.global_promotion_gate.residual_mode,
            )
        return self.local_promotion_gates[key]

    def _update_state(self, state, value, step):
        if self.method in {"harmonic", "harmonic_nb"}:
            state.update(value, step=step)
        else:
            state.update(value)

    def reset(self):
        self.global_state.reset()
        self.local_states.clear()
        self.od_states.clear()
        if self.global_promotion_gate is not None:
            self.global_promotion_gate.reset()
        self.local_promotion_gates.clear()
        self.promotion_absorptions = 0
        self.promotion_absorbed_rate = 0.0
        self._pending_station.clear()
        self._pending_od.clear()
        self._global_history_aux.clear()
        self._local_history_aux.clear()
        self._pending_steps = 0
        self.total_updates = 0
        self._od_summary_cache = (0.0, 0.0, 0)
        self._od_summary_cache_updates = -1

    def update(self, arrivals_by_station, arrivals_by_od=None):
        """Update filters from {(station_id, direction): count} arrivals."""
        arrivals_by_station = arrivals_by_station or {}
        arrivals_by_od = arrivals_by_od or {}
        for key, count in arrivals_by_station.items():
            self._pending_station[key] += float(count)
        if self.od_features_enabled:
            for key, count in arrivals_by_od.items():
                self._pending_od[key] += float(count)
        self._pending_steps += 1
        if self._pending_steps < self.bin_steps:
            return

        station_counts = dict(self._pending_station)
        od_counts = dict(self._pending_od) if self.od_features_enabled else {}
        self._pending_station.clear()
        self._pending_od.clear()
        self._pending_steps = 0
        self._apply_update(station_counts, od_counts)

    def _apply_update(self, arrivals_by_station, arrivals_by_od):
        total_count = float(sum(arrivals_by_station.values()))
        scale = 60.0 / max(self.bin_interval_s, 1e-6)
        bin_step = int(self.total_updates)
        global_rate = total_count * scale
        if self.history_aux_enabled:
            self._global_history_aux.append(global_rate)
        self._update_state(self.global_state, global_rate, bin_step)
        if self.global_promotion_gate is not None:
            gate = self.global_promotion_gate
            gate.update(self.global_state.high, self.global_state.low)
            self._maybe_promote_state(
                self.global_state, gate, self.global_demand_norm)

        local_keys = set(self.local_states.keys()) | set(arrivals_by_station.keys())
        for key in local_keys:
            state = self._get_local_state(key)
            local_rate = float(arrivals_by_station.get(key, 0.0)) * scale
            if self.history_aux_enabled:
                self._local_history_aux[key].append(local_rate)
            self._update_state(
                state, local_rate, bin_step)
            if self.promotion_enabled:
                gate = self._get_local_promotion_gate(key)
                gate.update(state.high, state.low)
                if self.promotion_adapt_local:
                    self._maybe_promote_state(
                        state, gate, self.local_demand_norm)

        if self.od_features_enabled:
            od_keys = set(self.od_states.keys()) | set(arrivals_by_od.keys())
            for key in od_keys:
                self._update_state(
                    self._get_od_state(key),
                    float(arrivals_by_od.get(key, 0.0)) * scale,
                    bin_step)
        self.total_updates += 1
        self._od_summary_cache_updates = -1

    def _maybe_promote_state(self, state, gate, demand_norm):
        if (not self.promotion_adapt_low
                or self.method not in {"harmonic", "harmonic_nb"}
                or gate is None
                or not getattr(gate, "flag", 0.0)
                or (self.promotion_adapt_active_only
                    and not getattr(gate, "active", 0.0))
                or (self.promotion_adapt_requires_persistent
                    and not getattr(gate, "persistent", 0.0))
                or float(getattr(gate, "strength", 0.0))
                < self.promotion_adapt_strength_min
                or not hasattr(state, "promote_residual")):
            return 0.0
        if self.promotion_adapt_high_energy_min > 0.0:
            high_energy = math.sqrt(max(getattr(state, "high_energy", 0.0), 0.0))
            high_norm = high_energy / max(float(demand_norm), 1e-6)
            if high_norm < self.promotion_adapt_high_energy_min:
                return 0.0
        absorbed = float(state.promote_residual(
            strength=gate.strength,
            gain=self.promotion_adapt_gain))
        if abs(absorbed) > 1e-9:
            self.promotion_absorptions += 1
            self.promotion_absorbed_rate += absorbed
        return absorbed

    def _forecast_value(self, state):
        if self.forecast_mode == "causal" and hasattr(state, "forecast"):
            return max(0.0, float(state.forecast(self.forecast_steps)))
        return max(0.0, state.low + state.low_slope * self.forecast_steps)

    def _high_noise_floor(self, state):
        if self.lower_high_noise_z <= 0.0 or not self._lower_high_noise_active():
            return 0.0
        return self._high_noise_floor_for_low(state.low)

    def _high_noise_floor_for_low(self, low):
        if self.lower_high_noise_z <= 0.0 or not self._lower_high_noise_active():
            return 0.0
        return self.lower_high_noise_z * math.sqrt(max(float(low), 0.0) + 1.0)

    def _lower_high_noise_active(self):
        if self.lower_high_noise_energy_min <= 0.0:
            return True
        high_energy = math.sqrt(max(self.global_state.high_energy, 0.0))
        high_norm = high_energy / max(self.global_demand_norm, 1e-6)
        return high_norm >= self.lower_high_noise_energy_min

    def _denoise_high_value(self, high, state):
        return self._denoise_high_for_low(high, state.low)

    def _denoise_high_for_low(self, high, low):
        if self.lower_high_noise_z <= 0.0:
            return float(high)
        high = float(high)
        floor = self._high_noise_floor_for_low(low)
        return math.copysign(max(abs(high) - floor, 0.0), high)

    def _history_aux_dim(self, level):
        if not self.history_aux_enabled or self.method == "raw_history":
            return 0
        if str(level).lower() == "upper":
            return self.history_aux_upper_bins
        return self.history_aux_lower_bins

    def _history_aux_weight(self):
        if self.history_aux_high_energy_max <= 0.0:
            return 1.0
        high_energy = math.sqrt(max(self.global_state.high_energy, 0.0))
        high_norm = high_energy / max(self.global_demand_norm, 1e-6)
        if high_norm <= self.history_aux_high_energy_max:
            return 1.0
        if self.history_aux_high_energy_width <= 0.0:
            return self.history_aux_min_weight
        over = (
            high_norm - self.history_aux_high_energy_max
        ) / self.history_aux_high_energy_width
        over = float(np.clip(over, 0.0, 1.0))
        return (1.0 - over) + over * self.history_aux_min_weight

    def _history_aux_features(self, samples, state, bins, norm):
        bins = int(bins)
        if bins <= 0:
            return []
        values = list(samples)[-bins:]
        if len(values) < bins:
            values = [0.0] * (bins - len(values)) + values
        arr = np.asarray(values, dtype=np.float64)
        if self.history_aux_mode in {"residual", "innovation"}:
            arr = arr - float(getattr(state, "low", 0.0))
        elif self.history_aux_mode not in {"raw", "rate"}:
            raise ValueError(
                f"Unknown history auxiliary mode: {self.history_aux_mode}")
        arr = arr / max(float(norm), 1e-6)
        if self.history_aux_clip > 0.0:
            arr = np.clip(arr, -self.history_aux_clip, self.history_aux_clip)
        arr = arr * self._history_aux_weight()
        return arr.astype(np.float32).tolist()

    def _upper_history_aux_features(self, state):
        if not self.history_aux_enabled:
            return []
        return self._history_aux_features(
            self._global_history_aux, state,
            self.history_aux_upper_bins, self.global_demand_norm)

    def _lower_history_aux_features(self, key, state):
        if not self.history_aux_enabled:
            return []
        return self._history_aux_features(
            self._local_history_aux.get(key, []), state,
            self.history_aux_lower_bins, self.local_demand_norm)

    def _upper_dim_for_mode(self, mode):
        if self.method == "raw_history":
            return self.upper_history_bins + self.promotion_feature_dim
        mode = str(mode or "low").lower()
        od_dim = 2 if self.od_features_enabled else 0
        if mode in {"low", "lf", "split"}:
            base_dim = 4 + od_dim + self.promotion_feature_dim
        elif mode in {"low_mid", "low_middle", "regime"}:
            base_dim = 7 + od_dim + self.promotion_feature_dim
        elif mode in {"high", "hf"}:
            base_dim = 3 + od_dim + self.promotion_feature_dim
        elif mode in {"all", "allfreq", "all_freq"}:
            base_dim = 6 + od_dim + self.promotion_feature_dim
        else:
            raise ValueError(f"Unknown upper frequency mode: {mode}")
        return base_dim + self._history_aux_dim("upper")

    def _lower_dim_for_mode(self, mode):
        if self.method == "raw_history":
            return self.lower_history_bins + self.promotion_feature_dim
        mode = str(mode or "high").lower()
        if mode in {"high", "hf", "split"}:
            base_dim = 4 + self.promotion_feature_dim
        elif mode in {"high_prior", "hf_prior", "high_context", "hf_context"}:
            base_dim = 6 + self.promotion_feature_dim
        elif mode in {"high_mid", "high_middle", "regime"}:
            base_dim = 7 + self.promotion_feature_dim
        elif mode in {"low", "lf"}:
            base_dim = 4 + self.promotion_feature_dim
        elif mode in {"all", "allfreq", "all_freq"}:
            base_dim = 7 + self.promotion_feature_dim
        else:
            raise ValueError(f"Unknown lower frequency mode: {mode}")
        return base_dim + self._history_aux_dim("lower")

    def _od_summary_features(self):
        if self._od_summary_cache_updates == self.total_updates:
            return self._od_summary_cache
        states = [s for s in self.od_states.values() if s.n > 0]
        if not states:
            self._od_summary_cache = (0.0, 0.0, 0)
            self._od_summary_cache_updates = self.total_updates
            return self._od_summary_cache
        lows = np.asarray([max(s.low, 0.0) for s in states], dtype=np.float64)
        total = float(lows.sum())
        if total <= 1e-9 or lows.size <= 1:
            entropy = 0.0
        else:
            p = lows / total
            entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(lows.size))
        od_high_energy = math.sqrt(
            max(sum(max(s.high_energy, 0.0) for s in states), 0.0))
        self._od_summary_cache = (
            entropy, od_high_energy / self.global_demand_norm, len(states))
        self._od_summary_cache_updates = self.total_updates
        return self._od_summary_cache

    def _upper_od_features(self):
        if not self.od_features_enabled:
            return []
        return list(self._od_summary_features()[:2])

    def _upper_low_features(self, state):
        high_energy = math.sqrt(max(state.high_energy, 0.0))
        return [
            state.low / self.global_demand_norm,
            state.low_slope / self.slope_norm,
            self._forecast_value(state) / self.global_demand_norm,
            high_energy / self.global_demand_norm,
        ]

    def _upper_high_features(self, state):
        high_energy = math.sqrt(max(state.high_energy, 0.0))
        return [
            state.high / self.global_demand_norm,
            state.high_slope / self.slope_norm,
            high_energy / self.global_demand_norm,
        ]

    def _upper_middle_features(self, state):
        middle_energy = math.sqrt(max(getattr(state, "middle_energy", 0.0), 0.0))
        return [
            getattr(state, "middle", 0.0) / self.global_demand_norm,
            getattr(state, "middle_slope", 0.0) / self.slope_norm,
            middle_energy / self.global_demand_norm,
        ]

    def upper_features(self, mode=None):
        """Return upper demand features normalized for policy input.

        Default split mode returns [low, low_slope, low_forecast, high_energy].
        RawHistory mode returns trailing realized-demand bins.
        """
        s = self.global_state
        if self.method == "raw_history":
            feats = (
                np.array(s.history(self.upper_history_bins), dtype=np.float32)
                / self.global_demand_norm
            )
            if self.promotion_state_features and self.global_promotion_gate is not None:
                feats = np.concatenate(
                    [feats, self.global_promotion_gate.features()])
            return feats.astype(np.float32)

        mode = str(mode or self.upper_mode).lower()
        if mode in {"low", "lf", "split"}:
            feats = self._upper_low_features(s)
        elif mode in {"low_mid", "low_middle", "regime"}:
            feats = self._upper_low_features(s) + self._upper_middle_features(s)
        elif mode in {"high", "hf"}:
            feats = self._upper_high_features(s)
        elif mode in {"all", "allfreq", "all_freq"}:
            feats = self._upper_low_features(s)[:3] + self._upper_high_features(s)
        else:
            raise ValueError(f"Unknown upper frequency mode: {mode}")
        feats = feats + self._upper_od_features()
        feats = feats + self._upper_history_aux_features(s)
        if self.promotion_state_features and self.global_promotion_gate is not None:
            feats.extend(self.global_promotion_gate.features().tolist())
        return np.array(feats, dtype=np.float32)

    def lower_features(self, station_id, direction, mode=None):
        """Return station-local residual/history features for lower control."""
        key = (int(station_id), bool(direction))
        s = self.local_states.get(key)
        if s is None:
            return np.zeros(self.lower_feature_dim, dtype=np.float32)
        if self.method == "raw_history":
            feats = (
                np.array(s.history(self.lower_history_bins), dtype=np.float32)
                / self.local_demand_norm
            )
            gate = self.local_promotion_gates.get(key)
            if self.promotion_state_features and gate is not None:
                feats = np.concatenate([feats, gate.features()])
            elif self.promotion_state_features:
                feats = np.concatenate(
                    [feats, np.zeros(self.promotion_feature_dim, dtype=np.float32)])
            return feats.astype(np.float32)

        local_energy = math.sqrt(max(s.high_energy, 0.0))
        global_energy = math.sqrt(max(self.global_state.high_energy, 0.0))
        local_high = self._denoise_high_value(s.high, s)
        prev_local_high = self._denoise_high_for_low(
            s.prev_high, getattr(s, "prev_low", s.low))
        local_energy = max(local_energy - self._high_noise_floor(s), 0.0)
        high_feats = [
            local_high / self.local_demand_norm,
            (local_high - prev_local_high) / self.local_demand_norm,
            local_energy / self.local_demand_norm,
            global_energy / self.global_demand_norm,
        ]
        low_feats = [
            s.low / self.local_demand_norm,
            s.low_slope / self.local_demand_norm,
            self._forecast_value(s) / self.local_demand_norm,
            self.global_state.low / self.global_demand_norm,
        ]
        pos_high = max(float(local_high), 0.0)
        low_prior = max(float(s.low), 1.0)
        high_prior_share = pos_high / (low_prior + pos_high + 1e-6)
        high_prior_feats = high_feats + [
            s.low / self.local_demand_norm,
            high_prior_share,
        ]
        local_middle_energy = math.sqrt(max(getattr(s, "middle_energy", 0.0), 0.0))
        global_middle_energy = math.sqrt(
            max(getattr(self.global_state, "middle_energy", 0.0), 0.0))
        middle_feats = [
            getattr(s, "middle", 0.0) / self.local_demand_norm,
            getattr(s, "middle_slope", 0.0) / self.local_demand_norm,
            local_middle_energy / self.local_demand_norm,
        ]
        mode = str(mode or self.lower_mode).lower()
        if mode in {"high", "hf", "split"}:
            feats = high_feats
        elif mode in {"high_prior", "hf_prior", "high_context", "hf_context"}:
            feats = high_prior_feats
        elif mode in {"high_mid", "high_middle", "regime"}:
            feats = high_feats[:3] + middle_feats + [
                global_middle_energy / self.global_demand_norm]
        elif mode in {"low", "lf"}:
            feats = low_feats
        elif mode in {"all", "allfreq", "all_freq"}:
            feats = low_feats[:3] + high_feats
        else:
            raise ValueError(f"Unknown lower frequency mode: {mode}")
        feats = feats + self._lower_history_aux_features(key, s)
        if self.promotion_state_features:
            gate = self.local_promotion_gates.get(key)
            gate_feats = (
                gate.features().tolist()
                if gate is not None
                else [0.0] * self.promotion_feature_dim
            )
            feats = feats + gate_feats
        return np.array(feats, dtype=np.float32)

    def local_high_value(self, station_id, direction):
        key = (int(station_id), bool(direction))
        s = self.local_states.get(key)
        if s is None:
            return 0.0
        return float(self._denoise_high_value(s.high, s) / self.local_demand_norm)

    def local_high_raw_value(self, station_id, direction):
        key = (int(station_id), bool(direction))
        s = self.local_states.get(key)
        if s is None:
            return 0.0
        return float(s.high / self.local_demand_norm)

    def local_trace_summary(self, station_id, direction):
        key = (int(station_id), bool(direction))
        s = self.local_states.get(key)
        if s is None:
            return {
                "local_low": 0.0,
                "local_high": 0.0,
                "local_high_raw": 0.0,
                "local_high_feature": 0.0,
                "local_high_noise_floor": 0.0,
                "local_high_energy": 0.0,
                "local_high_energy_feature": 0.0,
                "local_high_prior_share": 0.0,
                "local_middle": 0.0,
                "local_middle_energy": 0.0,
            }
        raw_high = float(s.high)
        feature_high = float(self._denoise_high_value(raw_high, s))
        noise_floor = float(self._high_noise_floor(s))
        high_energy = float(math.sqrt(max(s.high_energy, 0.0)))
        high_energy_feature = max(high_energy - noise_floor, 0.0)
        pos_high = max(feature_high, 0.0)
        low_prior = max(float(s.low), 1.0)
        high_prior_share = pos_high / (low_prior + pos_high + 1e-6)
        return {
            "local_low": float(s.low),
            "local_high": raw_high,
            "local_high_raw": raw_high,
            "local_high_feature": feature_high,
            "local_high_noise_floor": noise_floor,
            "local_high_energy": high_energy,
            "local_high_energy_feature": high_energy_feature,
            "local_high_prior_share": float(high_prior_share),
            "local_middle": float(getattr(s, "middle", 0.0)),
            "local_middle_energy": float(math.sqrt(max(
                getattr(s, "middle_energy", 0.0), 0.0))),
        }

    def local_low_value(self, station_id, direction):
        key = (int(station_id), bool(direction))
        s = self.local_states.get(key)
        if s is None:
            return 0.0
        return float(s.low / self.local_demand_norm)

    def local_promotion_summary(self, station_id, direction):
        key = (int(station_id), bool(direction))
        gate = self.local_promotion_gates.get(key)
        if gate is None:
            return {"flag": 0.0, "strength": 0.0, "age": 0.0, "score": 0.0}
        return gate.summary()

    def summary(self):
        s = self.global_state
        high_energy = math.sqrt(max(s.high_energy, 0.0))
        middle_energy = math.sqrt(max(getattr(s, "middle_energy", 0.0), 0.0))
        od_entropy, od_high_energy, od_active = self._od_summary_features()
        promotion = (
            self.global_promotion_gate.summary()
            if self.global_promotion_gate is not None
            else {"flag": 0.0, "strength": 0.0, "age": 0.0, "score": 0.0}
        )
        return {
            "freq_low_demand": float(s.low / self.global_demand_norm),
            "freq_low_slope": float(s.low_slope / self.slope_norm),
            "freq_low_forecast": float(
                self._forecast_value(s) / self.global_demand_norm),
            "freq_high_energy": float(high_energy / self.global_demand_norm),
            "freq_middle": float(
                getattr(s, "middle", 0.0) / self.global_demand_norm),
            "freq_middle_energy": float(middle_energy / self.global_demand_norm),
            "freq_od_entropy": float(od_entropy),
            "freq_od_high_energy": float(od_high_energy),
            "freq_od_active": int(od_active),
            "freq_updates": int(self.total_updates),
            "freq_method": self.method,
            "freq_promotion_flag": float(promotion["flag"]),
            "freq_promotion_strength": float(promotion["strength"]),
            "freq_promotion_age": float(promotion["age"]),
            "freq_promotion_score": float(promotion["score"]),
            "freq_promotion_direction": float(
                promotion.get("direction", 0.0)),
            "freq_promotion_active": float(promotion.get("active", 0.0)),
            "freq_promotion_persistent": float(
                promotion.get("persistent", 0.0)),
            "freq_promotion_ratio": float(promotion.get("ratio", 0.0)),
            "freq_promotion_absorptions": int(self.promotion_absorptions),
            "freq_promotion_absorbed": float(
                self.promotion_absorbed_rate / self.global_demand_norm),
        }
