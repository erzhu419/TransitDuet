"""
runner_v3.py
============
TransitDuet v3: bi-level bus control with switchable cross-level coupling
(``coupling_mode``: ``hiro`` | ``haar`` | ``channels``). Used by every paper
result in the current paper pipeline (Tables I/II + every figure); the legacy
``runner_v2.py`` is retained only as a frozen reference of the channels-mode
v2 baseline and is not used by any active script (see ``scripts/README.md``).

Coupling modes (all share the same lower-level pessimistic ensemble SAC
Lagrangian holding
controller; they differ only in how the upper output δ_t is consumed):
  hiro      The upper output is a per-dispatch target-headway shift; the
            lower's Lagrangian cost penalises deviation from
            (h_target + δ_t). Launch time is unchanged. This is the main
            paper result (H_hiro).
  channels  v2 channels-mode: δ_t directly perturbs launch time; the upper
            still gets holding-feedback in its state, so behaves like v2.
  haar      v2 channels-mode launch shift PLUS a clipped upper advantage
            injected into the lower's reward as a HAAR-style cross-advantage
            bonus, gated by a PIPER reachability classifier.

Mechanism (HIRO mode):
  Upper outputs δ_t for the next dispatch event (one decision per dispatch,
  ~264 events per simulated service day in our calibrated corridor; not a
  fixed-period 300 s timer). The lower then tracks the resulting target
  headway via Lagrangian holding control; CS-BAPR + HoldFB close the
  upper--lower loop and θ-OGD adaptively penalises fleet overshoot.

Usage:
    python -u runner_v3.py --config configs_ablation/H_hiro.yaml \
        [--episodes 300] [--seed 42] [--gpu]
"""

import copy
import os
import sys
import argparse
import csv
import hashlib
import json
import pickle
import random
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, deque

sys.stdout.reconfigure(line_buffering=True)

_torch_threads = os.environ.get("FREQDUET_TORCH_THREADS") or os.environ.get("TORCH_NUM_THREADS")
if _torch_threads:
    try:
        _torch_threads = max(1, int(_torch_threads))
        torch.set_num_threads(_torch_threads)
        torch.set_num_interop_threads(_torch_threads)
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from env.evaluation import normalize_wait_metric, service_cost_views
from frequency.diagnostics import (
    demand_attribution_mi,
    shock_response_metrics,
)
from upper.resac_upper import RESACUpperTrainer
from upper.credit_assignment import UpperCreditAssignment
from upper.interval_credit import UpperIntervalOutcomeTracker
from upper.measurement_proj import MeasurementProjection
from upper.plan_execution import UpperPlanExecutionContract
from upper.timetable_planner import TimetableCurvePlanner
from upper.counterfactual_action_selector import (
    ACTION_SPECS,
    CounterfactualActionTreeSelector,
)
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from lower.lifecycle import LowerEpisodeLifecycle
from lower.observation_contract import LowerObservationContract
from lower.state_encoder import PhysicalLowerStateEncoder
from lower.holding_externality import LoadWeightedHoldingPenalty
from lower.causal_holding_guard import CausalHoldingActionGuard
from coupling.holding_feedback import HoldingFeedback
from coupling.belief_tracker import BeliefTracker, SurpriseComputer
from randomness import RandomnessContract


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """Load YAML config, supporting _extends: <parent_file>."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    if '_extends' in cfg:
        parent_path = cfg.pop('_extends')
        base_dir = os.path.dirname(os.path.abspath(path))
        if os.path.isabs(parent_path):
            parent_full = parent_path
        else:
            candidates = [
                os.path.join(base_dir, parent_path),
                os.path.join(base_dir, '..', parent_path),
                os.path.join(str(SCRIPT_DIR), parent_path),
            ]
            parent_full = next((p for p in candidates if os.path.exists(p)), candidates[-1])
        parent = load_config(parent_full)
        cfg = _deep_merge(parent, cfg)
    return cfg


def config_fingerprint(config):
    """Hash the resolved training semantics while ignoring artifact location."""
    payload = copy.deepcopy(dict(config))
    payload.pop('logging', None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


# ═══════════════════════════════════════════════════════════════
#  Diagnostic helpers
# ═══════════════════════════════════════════════════════════════

def _stat(arr):
    """μ / σ / min / max for a list of floats."""
    if not arr:
        return {'mean': 0., 'std': 0., 'min': 0., 'max': 0., 'n': 0}
    a = np.asarray(arr, dtype=np.float64)
    return {'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'max': float(a.max()), 'n': len(a)}


def _causal_moving_average(values, window):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    window = max(1, int(window))
    out = np.empty_like(a)
    for i in range(a.size):
        j0 = max(0, i - window + 1)
        out[i] = a[j0:i + 1].mean()
    return out


def _upper_hf_power_ratio(delta_by_dir, window):
    """Approximate |HPF(delta_t)|^2 / |delta_t|^2 for upper actions."""
    high_power = 0.0
    total_power = 0.0
    for seq in delta_by_dir.values():
        if len(seq) < 2:
            continue
        a = np.asarray(seq, dtype=np.float64)
        low = _causal_moving_average(a, window)
        high = a - low
        high_power += float(np.dot(high, high))
        total_power += float(np.dot(a, a))
    return high_power / max(total_power, 1e-9) if total_power > 0 else 0.0


def _lower_lf_drift_ratio(actions_by_dir, window):
    """Approximate |LPF(cumsum(a_L))|^2 / |cumsum(a_L)|^2."""
    low_power = 0.0
    total_power = 0.0
    for seq in actions_by_dir.values():
        if len(seq) < 2:
            continue
        drift = np.cumsum(np.asarray(seq, dtype=np.float64))
        low = _causal_moving_average(drift, window)
        low_power += float(np.dot(low, low))
        total_power += float(np.dot(drift, drift))
    return low_power / max(total_power, 1e-9) if total_power > 0 else 0.0


def _abs_corr(xs, ys):
    if len(xs) < 3 or len(ys) < 3:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _demand_attribution_score(upper_samples, lower_samples):
    """Correlation proxy for the demand attribution diagnostic in GPT.md."""
    score = 0.0
    if upper_samples:
        low_u, high_u, act_u = zip(*upper_samples)
        score += _abs_corr(low_u, act_u) - _abs_corr(high_u, act_u)
    if lower_samples:
        low_l, high_l, act_l = zip(*lower_samples)
        score += _abs_corr(high_l, act_l) - _abs_corr(low_l, act_l)
    return float(score)


class DiagnosticLog:
    """Collects per-episode diagnostics and writes them as CSV + JSON."""

    HEADER = [
        'ep', 'stage', 'wall_env_s', 'wall_train_s',
        # env
        'protocol_version', 'config_fingerprint_sha256',
        'randomness_contract',
        'randomness_fingerprint_sha256',
        'avg_wait_min', 'avg_wait_observed_min',
        'restricted_wait_horizon_min',
        'avg_in_vehicle_observed_min',
        'restricted_in_vehicle_horizon_min',
        'avg_total_journey_observed_min',
        'restricted_total_journey_horizon_min',
        'passengers_generated', 'passengers_unserved',
        'passenger_unserved_rate',
        'headway_sample_count', 'trips_unlaunched', 'trip_launch_rate',
        'headway_state_arrival_event_count',
        'headway_state_spatial_fallback_count',
        'headway_state_target_default_count',
        'headway_state_arrival_event_rate',
        'trips_completed', 'trips_incomplete', 'trip_completion_rate',
        'simulation_end_time_s', 'done_reason', 'scenario_tape_id',
        'peak_fleet', 'fleet_inventory_mode', 'physical_vehicle_count',
        'fleet_capacity', 'fleet_ready_up', 'fleet_ready_down',
        'fleet_denied_dispatch_events', 'fleet_denied_retry_trip_seconds',
        'fleet_denied_trips',
        'fleet_readiness_delay_mean_s', 'fleet_readiness_delay_max_s',
        'fleet_denied_trip_rate',
        'holding_vehicle_seconds',
        'holding_vehicle_seconds_per_launched_trip',
        'holding_passenger_seconds',
        'holding_passenger_min_per_generated',
        'commanded_holding_vehicle_seconds',
        'commanded_holding_passenger_seconds',
        'commanded_holding_passenger_min_per_generated',
        'terminal_actual_dispatch_gap_mean_s',
        'terminal_dispatch_execution_error_mean_s',
        'terminal_dispatch_execution_error_abs_mean_s',
        'invalid_headway_decisions_masked',
        'lower_observation_contract', 'headway_reward_mode',
        'frequency_observation_source', 'lower_observation_ledger_hash',
        'headway_cv',
        'service_cost', 'service_cost_wait_metric',
        'service_cost_observed', 'service_cost_restricted',
        'ep_reward', 'ep_cost',
        'ep_steps', 'n_dispatches',
        # lower policy
        'lower_action_mean', 'lower_action_std', 'lower_action_min', 'lower_action_max',
        'lower_headway_state_mode', 'lower_state_input_schema',
        'lower_context_gate_enabled', 'lower_context_gate_active_mean',
        'lower_action_bins_gate_enabled', 'lower_action_bins_gate_active_mean',
        'lower_reward_mean', 'lower_reward_std',
        'lower_load_hold_penalty_mean', 'lower_load_hold_penalty_max',
        'lower_load_ratio_mean', 'lower_normalized_person_delay_mean',
        'lower_causal_guard_enabled', 'lower_causal_guard_evidence_mode',
        'lower_causal_guard_active_mean',
        'lower_causal_guard_limit_mean_s',
        'lower_causal_guard_adjustment_mean_s',
        # lower training
        'lower_q_mean', 'lower_q_std', 'lower_q_loss', 'lower_q_mse',
        'lower_ood_loss', 'lower_q_l1', 'lower_q_l1_penalty',
        'lower_cost_q_mean', 'lower_cost_q_loss',
        'lower_policy_loss', 'lower_pi_grad_norm', 'lower_q_grad_norm',
        'lower_alpha', 'lower_lambda',
        'lower_replay_size',
        'lower_trip_boundary_resets',
        'lower_pending_states_dropped',
        'lower_pending_actions_dropped',
        'lower_pending_states_consumed',
        'lower_pending_actions_consumed',
        'lower_terminal_action_masks', 'lower_terminal_transitions',
        'lower_terminal_outcomes_missing',
        'lower_policy_frozen', 'lower_critic_frozen',
        # upper policy (only after warmup)
        'upper_delta_mean', 'upper_delta_std', 'upper_delta_min', 'upper_delta_max',
        'upper_reward_mean', 'upper_reward_std',
        'upper_system_reward_mean', 'upper_system_reward_sum',
        'upper_reliability_reward_sum',
        'upper_gap_credit_mean', 'upper_gap_credit_std',
        'upper_interval_reward_mean', 'upper_interval_reward_sum',
        'upper_interval_wait_cost_sum',
        'upper_interval_onboard_cost_sum',
        'upper_interval_dispatch_backlog_cost_sum',
        'upper_interval_headway_cost_sum',
        'upper_interval_fleet_cost_sum',
        'upper_interval_coverage_mean',
        # upper training
        'upper_q_mean', 'upper_q_std', 'upper_q_loss', 'upper_q_mse',
        'upper_ood_loss', 'upper_q_l1', 'upper_q_l1_penalty',
        'upper_duration_steps_mean', 'upper_transition_duration_steps_mean',
        'upper_transition_stream_count', 'upper_transition_short_ratio',
        'upper_policy_loss',
        'upper_pi_grad_norm', 'upper_q_grad_norm',
        'upper_alpha', 'upper_replay_size',
        'upper_policy_frozen',
        # coupling
        'hold_fb_mean', 'hold_fb_std', 'hold_fb_n_trips',
        'hold_fb_trip_finalizations',
        'hold_fb_dir0_mean', 'hold_fb_dir1_mean',
        'hold_penalty_mean',
        'freq_holdfb_same_hold', 'freq_holdfb_same_wait',
        'freq_holdfb_other_hold', 'freq_holdfb_other_wait',
        'freq_holdfb_decisions',
        'freq_driftfb_same_drift', 'freq_driftfb_same_excess',
        'freq_driftfb_other_drift', 'freq_driftfb_other_excess',
        'freq_driftfb_decisions',
        'theta_wait', 'theta_fleet', 'theta_cv',
        # CS-BAPR belief
        'surprise', 'belief_window', 'belief_cp_prob', 'belief_entropy',
        # v2j belief-weighted MORL
        'w_wait', 'w_fleet', 'w_cv',
        # v2k elastic fleet
        'N_fleet', 'fleet_overshoot',
        # FreqDuet causal demand-frequency diagnostics
        'freq_low_demand', 'freq_low_slope', 'freq_low_forecast',
        'freq_high_energy', 'freq_middle', 'freq_middle_energy',
        'freq_od_entropy', 'freq_od_high_energy',
        'freq_od_active', 'freq_updates',
        'freq_promotion_flag', 'freq_promotion_strength',
        'freq_promotion_age', 'freq_promotion_score',
        'freq_promotion_active', 'freq_promotion_persistent',
        'freq_promotion_ratio',
        'freq_promotion_absorptions', 'freq_promotion_absorbed',
        # FreqDuet frequency-leakage regularization
        'lower_drift_signal_mode',
        'lower_drift_load_mean', 'lower_drift_load_max',
        'lower_drift_penalty_mean', 'lower_drift_penalty_max',
        'lower_drift_cost_mean', 'lower_drift_cost_max',
        'lower_drift_cost_adaptive_gate_mean',
        'lower_trip_hold_total_mean', 'lower_trip_hold_total_std',
        'lower_trip_hold_total_max',
        'upper_hf_penalty_mean', 'upper_hf_penalty_max',
        'upper_residual_value_cost_mean', 'upper_residual_value_cost_max',
        'upper_residual_value_cost_active_mean',
        'upper_residual_selector_enabled',
        'upper_residual_selector_active_mean',
        'upper_residual_selector_adjust_mean',
        'upper_residual_selector_adjust_max',
        'upper_residual_selector_margin_mean',
        'upper_residual_selector_actor_pred_mean',
        'upper_residual_selector_selected_pred_mean',
        'upper_residual_selector_feature_norm_mean',
        'upper_residual_selector_updates',
        'headway_value_planner_enabled',
        'headway_value_planner_active_mean',
        'headway_value_planner_adjust_mean',
        'headway_value_planner_adjust_max',
        'headway_value_planner_delta_mean',
        'headway_value_planner_delta_max',
        'headway_value_planner_margin_mean',
        'headway_value_planner_actor_pred_mean',
        'headway_value_planner_selected_pred_mean',
        'headway_value_planner_prior_mean',
        'headway_value_planner_target_cost_mean',
        'headway_value_planner_target_cost_max',
        'headway_value_planner_feature_norm_mean',
        'headway_value_planner_updates',
        # FreqDuet layer-frequency allocation diagnostics
        'upper_hf_power_ratio', 'lower_lf_drift_ratio',
        'demand_attr_score',
        'demand_attr_mi_score',
        'demand_attr_mi_upper_low', 'demand_attr_mi_upper_high',
        'demand_attr_mi_lower_high', 'demand_attr_mi_lower_low',
        'shock_response_time_mean_s', 'shock_response_time_std_s',
        'shock_response_hit_rate', 'shock_events', 'shock_action_mean_s',
        # FreqDuet frequency-attributed passenger-wait reward diagnostics
        'freq_wait_lower_penalty_mean', 'freq_wait_lower_penalty_max',
        'freq_wait_lower_board_credit_mean',
        'freq_wait_lower_board_credit_max',
        'freq_wait_lower_board_credit_gate_mean',
        'freq_wait_lower_hold_penalty_mean',
        'freq_wait_lower_hold_penalty_max',
        'freq_wait_lower_net_mean',
        'freq_wait_upper_credit_mean', 'freq_wait_upper_credit_std',
        'freq_wait_low_share_mean', 'freq_wait_lower_high_share_mean',
        'freq_wait_lower_raw_credit_weight_mean',
        'freq_wait_boarded_pax',
        # FreqDuet timetable-curve upper diagnostics
        'upper_plan_penalty_mean', 'upper_plan_penalty_max',
        'upper_plan_target_mean', 'upper_plan_target_std',
        'upper_plan_decisions', 'upper_plan_reuse_ratio',
        'upper_plan_projection_mode', 'upper_interval_wait_ownership',
        'upper_plan_headway_budget_mode',
        'upper_plan_raw_delta_mean_s',
        'upper_plan_projected_delta_mean_s',
        'upper_plan_projected_delta_sum_abs_mean_s',
        'terminal_launch_shift_mean', 'terminal_launch_shift_std',
        'terminal_shift_cap_mean', 'terminal_shift_cap_max',
        'terminal_shift_min_mean', 'terminal_shift_min_min',
        'terminal_feedback_bias_mean', 'terminal_feedback_bias_max',
        'terminal_feedback_events',
        'terminal_value_selector_enabled',
        'terminal_value_selector_active_mean',
        'terminal_value_selector_bias_mean',
        'terminal_value_selector_bias_max',
        'terminal_value_selector_margin_mean',
        'terminal_value_selector_actor_pred_mean',
        'terminal_value_selector_selected_pred_mean',
        'terminal_value_selector_feature_norm_mean',
        'terminal_value_selector_target_cost_mean',
        'terminal_value_selector_target_cost_max',
        'terminal_value_selector_updates',
        'snapshot_value_selector_enabled',
        'snapshot_value_active_mean',
        'snapshot_value_events',
        'snapshot_value_changed_mean',
        'snapshot_value_changed_events',
        'snapshot_value_override_mean',
        'snapshot_value_override_events',
        'snapshot_value_terminal_dispatch_mean',
        'snapshot_value_terminal_dispatch_events',
        'snapshot_value_terminal_bias_mean',
        'snapshot_value_terminal_bias_max',
        'snapshot_value_terminal_bias_events',
        'snapshot_value_margin_mean',
        'snapshot_value_margin_max',
        'snapshot_value_pred_mean',
        'snapshot_value_baseline_pred_mean',
        'snapshot_value_candidate_gate_cap_mean',
        'snapshot_value_candidate_gate_filtered_mean',
        'snapshot_value_risk_score_mean',
        'snapshot_value_risk_penalty_mean',
        'snapshot_value_risk_penalty_max_mean',
        'snapshot_value_guard_blocked_mean',
        'snapshot_value_guard_blocked_events',
        'snapshot_value_guard_negative_target_mean',
        'snapshot_value_guard_negative_target_events',
        'snapshot_value_guard_negative_target_blocked_mean',
        'snapshot_value_guard_negative_target_blocked_events',
        'snapshot_value_guard_prev_overshoot_norm_mean',
        'snapshot_value_guard_fleet_pressure_norm_mean',
        'snapshot_value_guard_primary_bias_mean',
        'cf_action_selector_enabled',
        'cf_action_selector_active_mean',
        'cf_action_selector_events',
        'cf_action_selector_changed_mean',
        'cf_action_selector_terminal_dispatch_mean',
        'cf_action_selector_delta_mean',
        'cf_action_selector_delta_std',
        'cf_action_selector_confidence_mean',
        'terminal_headway_floor_mean', 'terminal_headway_floor_events',
        'fleet_noharm_upper_pressure_mean',
        'fleet_noharm_upper_adjust_mean',
        'fleet_noharm_upper_events',
        'fleet_noharm_upper_gate_active_mean',
        'fleet_noharm_lower_pressure_mean',
        'fleet_noharm_lower_adjust_mean',
        'fleet_noharm_lower_events',
        'fleet_noharm_lower_gate_active_mean',
        'fleet_noharm_lower_proactive_adjust_mean',
        'fleet_noharm_lower_proactive_events',
        'fleet_noharm_lower_proactive_gate_active_mean',
        'fleet_noharm_lower_value_guard_adjust_mean',
        'fleet_noharm_lower_value_guard_events',
        'fleet_noharm_lower_value_guard_active_mean',
        'fleet_noharm_lower_value_guard_value_mean',
        'fleet_noharm_lower_value_guard_headway_mean',
        'fleet_noharm_lower_value_guard_cost_mean',
        'fleet_noharm_lower_value_soft_cost_mean',
        'fleet_noharm_lower_value_soft_cost_max',
        'fleet_noharm_lower_value_soft_events',
        'fleet_noharm_lower_value_soft_active_mean',
        'fleet_noharm_lower_value_soft_value_mean',
        'fleet_noharm_lower_value_soft_headway_mean',
        'fleet_noharm_lower_value_soft_risk_mean',
        'fleet_noharm_lower_value_soft_violation_mean',
        'fixed_selector_fixed_active',
        'fixed_selector_learned_cost_ema',
        'fixed_selector_fixed_cost_ema',
        'fixed_selector_learned_count',
        'fixed_selector_fixed_count',
        'fixed_selector_context_enabled',
        'fixed_selector_context_learned_value',
        'fixed_selector_context_fixed_value',
        'fixed_selector_context_margin',
        'fixed_selector_context_feature_norm',
    ]

    def __init__(self, log_dir, resume=False):
        self.csv_path = os.path.join(log_dir, 'diagnostics.csv')
        self.json_path = os.path.join(log_dir, 'diagnostics.json')
        self._rows = []
        # Write CSV header only if not resuming or CSV missing
        if resume and os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline='') as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != self.HEADER:
                raise RuntimeError(
                    "diagnostics schema changed; start a new protocol-v2 run "
                    "instead of resuming this legacy directory")
        else:
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.HEADER)

    def append(self, row_dict):
        """Append one episode row. Missing keys default to 0."""
        self._rows.append(row_dict)
        row = [row_dict.get(k, 0.) for k in self.HEADER]
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self._rows, f, indent=1, default=str)


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  v3 helpers: PIPER-style reachability classifier
# ═══════════════════════════════════════════════════════════════

class ReachabilityMLP(torch.nn.Module):
    """Small MLP that maps (s_upper, δ_t, hold_summary) → P(plan reachable) ∈ [0,1].

    Used by HAAR-mode coupling as a gate on the per-trip advantage signal
    injected into lower-level rewards. Trained with binary cross-entropy
    against post-hoc labels: 1[|gap_dev_i| < threshold].
    """

    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        # input = state_dim (upper state) + 1 (δ_t) + 1 (avg hold) + 1 (hold std) = state_dim + 3
        in_dim = state_dim + 3
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(x)).squeeze(-1)


def _reach_features(s_upper, delta_t, hold_mean, hold_std):
    """Concat upper state with the action and lower-feedback summary."""
    feats = list(s_upper)
    feats.append(delta_t / 120.0)        # normalised δ
    feats.append(hold_mean / 60.0)
    feats.append(hold_std / 60.0)
    return np.array(feats, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════

class TransitDuetV2Runner:
    """v2 training loop: per-trip upper decisions + holding feedback coupling."""

    def __init__(self, config, device='cpu'):
        self.cfg = config
        self.device = device
        self.exp_name = str(config.get('_name', 'v2'))
        self.protocol_version = str(
            (config.get('protocol', {}) or {}).get(
                'version', 'freqduet-eval-v2'))
        self.base_seed = int(config.get('seed', 0))
        self.config_fingerprint_sha256 = config_fingerprint(config)
        training_cfg = config.get('training', {}) or {}
        randomness_cfg = config.get('randomness', {}) or {}
        self.randomness = RandomnessContract(
            self.base_seed,
            randomness_cfg.get('mode', 'global_legacy'))
        self._random_stream_names = (
            'fleet', 'upper_init', 'upper_policy', 'upper_replay',
            'lower_init', 'lower_policy', 'lower_replay',
            'upper_residual_selector', 'headway_value_selector',
            'terminal_value_selector', 'fixed_expert_selector',
            'tpc_mixture', 'reachability_init', 'reachability_replay')
        self.randomness_manifest = self.randomness.manifest(
            self._random_stream_names)
        self.fleet_rng = self.randomness.numpy('fleet')
        self._upper_residual_selector_rng = self.randomness.numpy(
            'upper_residual_selector')
        self._headway_value_selector_rng = self.randomness.numpy(
            'headway_value_selector')
        self._terminal_value_selector_rng = self.randomness.numpy(
            'terminal_value_selector')
        self._fixed_expert_selector_rng = self.randomness.numpy(
            'fixed_expert_selector')
        self._tpc_rng = self.randomness.numpy('tpc_mixture')
        self._reachability_rng = self.randomness.numpy('reachability_replay')
        self.decouple_init_seeds = bool(
            training_cfg.get('decouple_init_seeds', False))
        self.checkpoint_contract = str(training_cfg.get(
            'checkpoint_contract', 'deployment_only_legacy')).strip().lower()
        if self.checkpoint_contract not in {
                'deployment_only_legacy', 'exact_training_state_v4'}:
            raise ValueError(
                'training.checkpoint_contract must be '
                'deployment_only_legacy or exact_training_state_v4')
        if (self.checkpoint_contract == 'exact_training_state_v4'
                and not self.randomness.isolated):
            raise ValueError(
                'exact_training_state_v4 requires isolated_streams_v4')
        stability_cfg = training_cfg.get('longtrain_stability', {}) or {}

        def _optional_freeze_ep(name):
            value = stability_cfg.get(name, training_cfg.get(name, None))
            if value is None:
                return None
            value = int(value)
            return value if value >= 0 else None

        self.freeze_lower_policy_after_ep = _optional_freeze_ep(
            'freeze_lower_policy_after_ep')
        self.freeze_lower_critic_after_ep = _optional_freeze_ep(
            'freeze_lower_critic_after_ep')
        self.freeze_upper_after_ep = _optional_freeze_ep(
            'freeze_upper_after_ep')
        objective_cfg = config.get('objective', {}) or {}
        self.objective_weights = dict(
            objective_cfg.get('weights', {}) or {})
        self.objective_wait_metric = normalize_wait_metric(
            objective_cfg.get('wait_metric', 'observed'))

        # Environment
        env_cfg = config.get('env', {})
        env_path = os.path.join(str(SCRIPT_DIR), config['env']['path'])
        self.env = env_bus(
            env_path,
            route_sigma=config['env']['route_sigma'],
            env_config=env_cfg,
        )
        self.env.scenario_seed = int(
            env_cfg.get('scenario_seed', self.base_seed))
        self.env.configure_frequency_features(config.get('frequency', {}))
        self.env.enable_plot = False
        self.env._n_fleet_target = config['upper']['N_fleet']
        self.env.demand_noise = env_cfg.get('demand_noise', 0.0)
        self.env.demand_scale = env_cfg.get('demand_scale', 1.0)
        self.env.demand_hourly_multipliers = env_cfg.get(
            'demand_hourly_multipliers', None)
        self.env.service_start_hour = env_cfg.get('service_start_hour', 6)
        self.env.service_end_hour = env_cfg.get('service_end_hour', 19)
        self.env.od_noise = env_cfg.get('od_noise', 0.0)
        self.env.od_noise_clip = env_cfg.get('od_noise_clip', [0.3, 2.0])
        self.env.peak_shift_choices = env_cfg.get('peak_shift_choices', None)
        self.env.peak_shift_probs = env_cfg.get('peak_shift_probs', None)

        state_dim = self.env.state_dim

        # ── Upper policy ──
        upper_cfg = config['upper']
        self.upper_credit_assignment = UpperCreditAssignment.from_config(
            upper_cfg.get('credit_assignment', {}))
        self.upper_interval_credit = UpperIntervalOutcomeTracker.from_config(
            upper_cfg.get('interval_credit', {}))
        self.env._upper_interval_outcome_tracker = (
            self.upper_interval_credit
            if self.upper_interval_credit.enabled else None
        )
        self.upper_transition_stream_mode = str(upper_cfg.get(
            'transition_stream_mode', 'legacy_global')).strip().lower()
        if self.upper_transition_stream_mode not in {
                'legacy_global', 'planner_key'}:
            raise ValueError(
                "upper.transition_stream_mode must be legacy_global or "
                "planner_key")
        holding_state_cfg = upper_cfg.get('holding_state', {}) or {}
        self.upper_holding_state_source = str(
            holding_state_cfg.get('source', 'env_legacy')).strip().lower()
        if self.upper_holding_state_source not in {
                'env_legacy', 'trip_lifecycle'}:
            raise ValueError(
                "upper.holding_state.source must be env_legacy or "
                "trip_lifecycle")
        self.upper_holding_state_episode_local = bool(
            holding_state_cfg.get('episode_local', True))
        self.delta_max = upper_cfg.get('delta_max', 120.0)
        planner_cfg = upper_cfg.get('timetable_planner', {})
        self.upper_plan_execution = UpperPlanExecutionContract.from_config(
            planner_cfg)
        self.timetable_planner = None
        self.upper_plan_penalty_weight = 0.0
        self.timetable_replan_interval_s = 0.0
        self.timetable_action_ema_alpha = 1.0
        self.timetable_terminal_dispatch = False
        self.timetable_promotion_replan = False
        self.timetable_promotion_replan_strength_min = 0.0
        self.timetable_promotion_replan_cooldown_s = 0.0
        self.timetable_plan_all_directions = False
        self.timetable_terminal_hf_shift_max_s = None
        self.timetable_terminal_hf_energy_min = 0.0
        self.timetable_terminal_early_release_enable = False
        self.timetable_terminal_early_release_base_min_s = 0.0
        self.timetable_terminal_early_release_relaxed_min_s = 0.0
        self.timetable_terminal_early_release_max_high_energy = None
        self.timetable_terminal_early_release_max_middle_energy = None
        self.timetable_terminal_early_release_min_od_entropy = None
        self.timetable_terminal_early_release_max_od_high_energy = None
        self.timetable_terminal_early_release_max_low_forecast = None
        self.timetable_terminal_early_release_min_action_mean_s = None
        self.timetable_terminal_early_release_min_current_delta_s = None
        self.timetable_terminal_early_release_min_prev_wait_min = None
        self.timetable_terminal_early_release_max_prev_overshoot_norm = None
        self.timetable_terminal_early_release_max_prev_headway_cv = None
        self.timetable_terminal_early_release_max_prev_terminal_shift_mean_s = None
        self.timetable_terminal_early_release_max_prev_terminal_shift_std_s = None
        self.timetable_terminal_early_release_max_peak_shift_abs = None
        self.timetable_terminal_early_release_min_updates = 0
        self.timetable_terminal_feedback_enable = False
        self.timetable_terminal_feedback_gain = 0.0
        self.timetable_terminal_feedback_max_s = 0.0
        self.timetable_terminal_feedback_min_s = 0.0
        self.timetable_terminal_feedback_deadband_s = 0.0
        self.timetable_terminal_feedback_min_trips = 1
        self.timetable_terminal_feedback_ema_weight = 0.0
        self.timetable_terminal_fleet_relief_enable = False
        self.timetable_terminal_fleet_relief_max_s = 0.0
        self.timetable_terminal_fleet_relief_min_s = 0.0
        self.timetable_terminal_fleet_relief_pressure_start = 0.0
        self.timetable_terminal_fleet_relief_pressure_full = 1.0
        self.timetable_terminal_value_relief_enable = False
        self.timetable_terminal_value_relief_max_s = 0.0
        self.timetable_terminal_value_relief_pressure_start = 0.0
        self.timetable_terminal_value_relief_pressure_full = 1.0
        self.timetable_terminal_value_relief_gap_norm_s = 60.0
        self.timetable_terminal_value_relief_min_gap_s = 0.0
        self.timetable_terminal_value_relief_gap_tolerance_s = 0.0
        self.timetable_terminal_value_relief_gap_gain = 1.0
        self.timetable_terminal_value_relief_demand_weight = 0.0
        self.timetable_terminal_value_selector_enable = False
        self.timetable_terminal_value_selector_start_configured = False
        self.timetable_terminal_value_selector_learn_start_configured = False
        self.timetable_terminal_value_selector_start_ep = 30
        self.timetable_terminal_value_selector_learn_start_ep = 10
        self.timetable_terminal_value_selector_min_observations = 32
        self.timetable_terminal_value_selector_ridge = 0.25
        self.timetable_terminal_value_selector_feature_clip = 8.0
        self.timetable_terminal_value_selector_cost_clip = 8.0
        self.timetable_terminal_value_selector_epsilon = 0.0
        self.timetable_terminal_value_selector_improve_margin = 0.0
        self.timetable_terminal_value_selector_delay_penalty = 0.0
        self.timetable_terminal_value_selector_bias_norm_s = 5.0
        self.timetable_terminal_value_selector_target = 'transition_reward'
        self.timetable_terminal_value_selector_episode_weight = 1.0
        self.timetable_terminal_value_selector_local_weight = 1.0
        self.timetable_terminal_value_selector_reward_weight = 1.0
        self.timetable_terminal_value_selector_candidates = np.asarray(
            [0.0], dtype=np.float32)
        self.timetable_terminal_value_selector_A = None
        self.timetable_terminal_value_selector_b = None
        self.timetable_terminal_value_selector_updates = 0
        self.timetable_headway_value_planner_enable = False
        self.timetable_headway_value_planner_start_configured = False
        self.timetable_headway_value_planner_learn_start_configured = False
        self.timetable_headway_value_planner_start_ep = 50
        self.timetable_headway_value_planner_learn_start_ep = 20
        self.timetable_headway_value_planner_min_observations = 256
        self.timetable_headway_value_planner_ridge = 0.75
        self.timetable_headway_value_planner_feature_clip = 8.0
        self.timetable_headway_value_planner_cost_clip = 8.0
        self.timetable_headway_value_planner_epsilon = 0.0
        self.timetable_headway_value_planner_improve_margin = 0.0
        self.timetable_headway_value_planner_adjust_penalty = 0.0
        self.timetable_headway_value_planner_adjust_norm_s = 15.0
        self.timetable_headway_value_planner_delta_norm_s = 20.0
        self.timetable_headway_value_planner_candidate_deltas = np.asarray(
            [0.0], dtype=np.float32)
        self.timetable_headway_value_planner_candidate_offsets = np.zeros(
            0, dtype=np.float32)
        self.timetable_headway_value_planner_action_basis_enable = False
        self.timetable_headway_value_planner_action_basis_mode = 'onehot_rbf'
        self.timetable_headway_value_planner_action_basis_width_s = 10.0
        self.timetable_headway_value_planner_action_basis_centers = np.asarray(
            [-20.0, -10.0, 0.0, 5.0, 10.0], dtype=np.float32)
        self.timetable_headway_value_planner_action_basis_interactions = True
        self.timetable_headway_value_planner_prior_weight = 0.0
        self.timetable_headway_value_planner_spacing_weight = 1.0
        self.timetable_headway_value_planner_wait_weight = 0.5
        self.timetable_headway_value_planner_fleet_weight = 0.5
        self.timetable_headway_value_planner_cv_target = 0.438
        self.timetable_headway_value_planner_overshoot_target = 0.13
        self.timetable_headway_value_planner_wait_target_min = 5.5
        self.timetable_headway_value_planner_terminal_shift_target_s = 12.0
        self.timetable_headway_value_planner_target = 'episode_composite'
        self.timetable_headway_value_planner_episode_weight = 1.0
        self.timetable_headway_value_planner_local_weight = 1.0
        self.timetable_headway_value_planner_reward_weight = 1.0
        self.timetable_headway_value_planner_gate_enable = False
        self.timetable_headway_value_planner_gate_min_low_forecast = None
        self.timetable_headway_value_planner_gate_max_low_forecast = None
        self.timetable_headway_value_planner_gate_min_high_energy = None
        self.timetable_headway_value_planner_gate_max_high_energy = None
        self.timetable_headway_value_planner_gate_min_middle_energy = None
        self.timetable_headway_value_planner_gate_max_middle_energy = None
        self.timetable_headway_value_planner_gate_min_od_entropy = None
        self.timetable_headway_value_planner_gate_max_od_entropy = None
        self.timetable_headway_value_planner_gate_max_promotion_strength = None
        self.timetable_headway_value_planner_gate_any_of = []
        self.timetable_headway_value_planner_A = None
        self.timetable_headway_value_planner_b = None
        self.timetable_headway_value_planner_updates = 0
        self.timetable_terminal_headway_floor_enable = False
        self.timetable_terminal_headway_floor_ratio = 0.0
        self.timetable_terminal_headway_floor_min_s = 0.0
        self.upper_residual_value_cost_enable = False
        self.upper_residual_value_cost_weight = 0.0
        self.upper_residual_value_cost_action_norm_s = 15.0
        self.upper_residual_value_cost_fleet_util_start = 0.75
        self.upper_residual_value_cost_fleet_util_full = 1.05
        self.upper_residual_value_cost_high_start = 0.045
        self.upper_residual_value_cost_high_full = 0.085
        self.upper_residual_value_cost_promotion_relief = 0.5
        self._active_timetable_plans = {}
        self._last_promotion_replan_launch = {}
        if bool(planner_cfg.get('enable', False)):
            self.timetable_planner = TimetableCurvePlanner.from_config(
                planner_cfg, delta_max_s=self.delta_max)
            self.upper_plan_penalty_weight = float(
                planner_cfg.get('smooth_penalty', 0.0))
            self.timetable_replan_interval_s = float(
                planner_cfg.get('replan_interval_s', 900.0))
            self.timetable_action_ema_alpha = float(
                np.clip(planner_cfg.get('action_ema_alpha', 1.0), 0.0, 1.0))
            self.timetable_terminal_dispatch = bool(
                planner_cfg.get('terminal_dispatch', False))
            self.timetable_promotion_replan = bool(
                planner_cfg.get('promotion_replan', False))
            self.timetable_promotion_replan_strength_min = float(
                planner_cfg.get('promotion_replan_strength_min', 0.0))
            self.timetable_promotion_replan_cooldown_s = max(float(
                planner_cfg.get('promotion_replan_cooldown_s', 0.0)), 0.0)
            self.timetable_plan_all_directions = bool(
                planner_cfg.get('plan_all_directions', False))
            hf_shift_max = planner_cfg.get(
                'terminal_shift_high_energy_max_s',
                planner_cfg.get('terminal_high_energy_shift_max_s', None))
            if hf_shift_max is not None:
                self.timetable_terminal_hf_shift_max_s = float(hf_shift_max)
            self.timetable_terminal_hf_energy_min = max(float(
                planner_cfg.get('terminal_shift_high_energy_min',
                                planner_cfg.get('terminal_high_energy_min',
                                                0.0))), 0.0)
            early_release_cfg = (
                planner_cfg.get('terminal_early_release_adaptive', {}) or {})
            self.timetable_terminal_early_release_enable = bool(
                early_release_cfg.get('enable', False))
            self.timetable_terminal_early_release_base_min_s = float(
                early_release_cfg.get(
                    'base_min_s',
                    self.timetable_planner.terminal_shift_min_s))
            self.timetable_terminal_early_release_relaxed_min_s = float(
                early_release_cfg.get(
                    'relaxed_min_s',
                    self.timetable_planner.terminal_shift_min_s))
            self.timetable_terminal_early_release_min_updates = max(
                0, int(early_release_cfg.get('min_updates_required', 0)))

            def _optional_float(name):
                value = early_release_cfg.get(name)
                return None if value is None else float(value)

            self.timetable_terminal_early_release_max_high_energy = (
                _optional_float('max_high_energy'))
            self.timetable_terminal_early_release_max_middle_energy = (
                _optional_float('max_middle_energy'))
            self.timetable_terminal_early_release_min_od_entropy = (
                _optional_float('min_od_entropy'))
            self.timetable_terminal_early_release_max_od_high_energy = (
                _optional_float('max_od_high_energy'))
            self.timetable_terminal_early_release_max_low_forecast = (
                _optional_float('max_low_forecast'))
            self.timetable_terminal_early_release_min_action_mean_s = (
                _optional_float('min_action_mean_s'))
            self.timetable_terminal_early_release_min_current_delta_s = (
                _optional_float('min_current_delta_s'))
            self.timetable_terminal_early_release_min_prev_wait_min = (
                _optional_float('min_prev_wait_min'))
            self.timetable_terminal_early_release_max_prev_overshoot_norm = (
                _optional_float('max_prev_overshoot_norm'))
            self.timetable_terminal_early_release_max_prev_headway_cv = (
                _optional_float('max_prev_headway_cv'))
            self.timetable_terminal_early_release_max_prev_terminal_shift_mean_s = (
                _optional_float('max_prev_terminal_shift_mean_s'))
            self.timetable_terminal_early_release_max_prev_terminal_shift_std_s = (
                _optional_float('max_prev_terminal_shift_std_s'))
            self.timetable_terminal_early_release_max_peak_shift_abs = (
                _optional_float('max_peak_shift_abs'))
            terminal_fb_cfg = planner_cfg.get('terminal_feedback', {}) or {}
            self.timetable_terminal_feedback_enable = bool(
                terminal_fb_cfg.get('enable', False))
            self.timetable_terminal_feedback_gain = max(float(
                terminal_fb_cfg.get('gain', 0.0)), 0.0)
            self.timetable_terminal_feedback_max_s = max(float(
                terminal_fb_cfg.get('max_s', 0.0)), 0.0)
            self.timetable_terminal_feedback_min_s = max(float(
                terminal_fb_cfg.get('min_s', 0.0)), 0.0)
            self.timetable_terminal_feedback_deadband_s = max(float(
                terminal_fb_cfg.get('deadband_s', 0.0)), 0.0)
            self.timetable_terminal_feedback_min_trips = max(
                1, int(terminal_fb_cfg.get('min_trips', 1)))
            self.timetable_terminal_feedback_ema_weight = float(np.clip(
                terminal_fb_cfg.get('ema_weight', 0.0), 0.0, 1.0))
            terminal_relief_cfg = (
                planner_cfg.get('terminal_fleet_relief', {}) or {})
            self.timetable_terminal_fleet_relief_enable = bool(
                terminal_relief_cfg.get('enable', False))
            self.timetable_terminal_fleet_relief_max_s = max(float(
                terminal_relief_cfg.get('max_s', 0.0)), 0.0)
            self.timetable_terminal_fleet_relief_min_s = max(float(
                terminal_relief_cfg.get('min_s', 0.0)), 0.0)
            self.timetable_terminal_fleet_relief_pressure_start = float(
                terminal_relief_cfg.get('pressure_start', 0.0))
            self.timetable_terminal_fleet_relief_pressure_full = float(
                terminal_relief_cfg.get('pressure_full', 1.0))
            terminal_value_cfg = (
                planner_cfg.get('terminal_value_relief', {}) or {})
            self.timetable_terminal_value_relief_enable = bool(
                terminal_value_cfg.get('enable', False))
            self.timetable_terminal_value_relief_max_s = max(float(
                terminal_value_cfg.get('max_s', 0.0)), 0.0)
            self.timetable_terminal_value_relief_pressure_start = float(
                terminal_value_cfg.get('pressure_start', 0.0))
            self.timetable_terminal_value_relief_pressure_full = float(
                terminal_value_cfg.get('pressure_full', 1.0))
            self.timetable_terminal_value_relief_gap_norm_s = max(float(
                terminal_value_cfg.get('gap_norm_s', 60.0)), 1e-6)
            self.timetable_terminal_value_relief_min_gap_s = max(float(
                terminal_value_cfg.get('min_gap_deficit_s', 0.0)), 0.0)
            self.timetable_terminal_value_relief_gap_tolerance_s = max(float(
                terminal_value_cfg.get('gap_tolerance_s', 0.0)), 0.0)
            self.timetable_terminal_value_relief_gap_gain = max(float(
                terminal_value_cfg.get('gap_gain', 1.0)), 0.0)
            self.timetable_terminal_value_relief_demand_weight = max(float(
                terminal_value_cfg.get('demand_weight', 0.0)), 0.0)
            terminal_selector_cfg = (
                planner_cfg.get('terminal_value_selector', {}) or {})
            self.timetable_terminal_value_selector_enable = bool(
                terminal_selector_cfg.get('enable', False))
            self.timetable_terminal_value_selector_start_configured = (
                'start_ep' in terminal_selector_cfg)
            self.timetable_terminal_value_selector_learn_start_configured = (
                'learn_start_ep' in terminal_selector_cfg)
            self.timetable_terminal_value_selector_start_ep = int(
                terminal_selector_cfg.get('start_ep', 30))
            self.timetable_terminal_value_selector_learn_start_ep = int(
                terminal_selector_cfg.get(
                    'learn_start_ep',
                    max(0, self.timetable_terminal_value_selector_start_ep - 20)))
            self.timetable_terminal_value_selector_min_observations = max(
                1, int(terminal_selector_cfg.get('min_observations', 32)))
            self.timetable_terminal_value_selector_ridge = max(float(
                terminal_selector_cfg.get('ridge', 0.25)), 1e-6)
            self.timetable_terminal_value_selector_feature_clip = max(float(
                terminal_selector_cfg.get('feature_clip', 8.0)), 0.0)
            self.timetable_terminal_value_selector_cost_clip = max(float(
                terminal_selector_cfg.get('cost_clip', 8.0)), 0.0)
            self.timetable_terminal_value_selector_epsilon = float(np.clip(
                terminal_selector_cfg.get('epsilon', 0.0), 0.0, 1.0))
            self.timetable_terminal_value_selector_improve_margin = max(float(
                terminal_selector_cfg.get('improve_margin', 0.0)), 0.0)
            self.timetable_terminal_value_selector_delay_penalty = max(float(
                terminal_selector_cfg.get('delay_penalty', 0.0)), 0.0)
            self.timetable_terminal_value_selector_bias_norm_s = max(float(
                terminal_selector_cfg.get('bias_norm_s', 5.0)), 1e-6)
            self.timetable_terminal_value_selector_target = str(
                terminal_selector_cfg.get(
                    'target', 'transition_reward')).strip().lower()
            self.timetable_terminal_value_selector_episode_weight = float(
                terminal_selector_cfg.get('episode_weight', 1.0))
            self.timetable_terminal_value_selector_local_weight = float(
                terminal_selector_cfg.get('local_weight', 1.0))
            self.timetable_terminal_value_selector_reward_weight = float(
                terminal_selector_cfg.get('reward_weight', 1.0))
            bias_candidates = terminal_selector_cfg.get(
                'candidate_bias_s', [0.0, 5.0, 10.0, 15.0])
            self.timetable_terminal_value_selector_candidates = np.asarray(
                sorted({max(0.0, float(x)) for x in bias_candidates}),
                dtype=np.float32)
            if not np.any(np.isclose(
                    self.timetable_terminal_value_selector_candidates, 0.0)):
                self.timetable_terminal_value_selector_candidates = np.concatenate([
                    np.asarray([0.0], dtype=np.float32),
                    self.timetable_terminal_value_selector_candidates,
                ])
            headway_value_cfg = (
                planner_cfg.get('headway_value_planner', {}) or {})
            self.timetable_headway_value_planner_enable = bool(
                headway_value_cfg.get('enable', False))
            self.timetable_headway_value_planner_start_configured = (
                'start_ep' in headway_value_cfg)
            self.timetable_headway_value_planner_learn_start_configured = (
                'learn_start_ep' in headway_value_cfg)
            self.timetable_headway_value_planner_start_ep = int(
                headway_value_cfg.get('start_ep', 50))
            self.timetable_headway_value_planner_learn_start_ep = int(
                headway_value_cfg.get(
                    'learn_start_ep',
                    max(0, self.timetable_headway_value_planner_start_ep - 30)))
            self.timetable_headway_value_planner_min_observations = max(
                1, int(headway_value_cfg.get('min_observations', 256)))
            self.timetable_headway_value_planner_ridge = max(float(
                headway_value_cfg.get('ridge', 0.75)), 1e-6)
            self.timetable_headway_value_planner_feature_clip = max(float(
                headway_value_cfg.get('feature_clip', 8.0)), 0.0)
            self.timetable_headway_value_planner_cost_clip = max(float(
                headway_value_cfg.get('cost_clip', 8.0)), 0.0)
            self.timetable_headway_value_planner_epsilon = float(np.clip(
                headway_value_cfg.get('epsilon', 0.0), 0.0, 1.0))
            self.timetable_headway_value_planner_improve_margin = max(float(
                headway_value_cfg.get('improve_margin', 0.0)), 0.0)
            self.timetable_headway_value_planner_adjust_penalty = max(float(
                headway_value_cfg.get('adjust_penalty', 0.0)), 0.0)
            self.timetable_headway_value_planner_adjust_norm_s = max(float(
                headway_value_cfg.get('adjust_norm_s', 15.0)), 1e-6)
            self.timetable_headway_value_planner_delta_norm_s = max(float(
                headway_value_cfg.get('delta_norm_s', 20.0)), 1e-6)
            candidate_deltas = headway_value_cfg.get(
                'candidate_deltas_s', [-20.0, -10.0, 0.0, 10.0, 20.0])
            self.timetable_headway_value_planner_candidate_deltas = np.asarray(
                sorted({float(x) for x in candidate_deltas}),
                dtype=np.float32)
            if not np.any(np.isclose(
                    self.timetable_headway_value_planner_candidate_deltas,
                    0.0)):
                self.timetable_headway_value_planner_candidate_deltas = (
                    np.concatenate([
                        self.timetable_headway_value_planner_candidate_deltas,
                        np.asarray([0.0], dtype=np.float32),
                    ]))
            candidate_offsets = headway_value_cfg.get(
                'candidate_offsets_s', [])
            self.timetable_headway_value_planner_candidate_offsets = (
                np.asarray([float(x) for x in candidate_offsets],
                           dtype=np.float32).reshape(-1))
            action_basis_cfg = (
                headway_value_cfg.get('action_basis', {}) or {})
            self.timetable_headway_value_planner_action_basis_enable = bool(
                action_basis_cfg.get('enable', False))
            self.timetable_headway_value_planner_action_basis_mode = str(
                action_basis_cfg.get('mode', 'onehot_rbf')).strip().lower()
            self.timetable_headway_value_planner_action_basis_width_s = max(
                float(action_basis_cfg.get('width_s', 10.0)), 1e-6)
            basis_centers = action_basis_cfg.get(
                'centers_s',
                headway_value_cfg.get('candidate_deltas_s', candidate_deltas))
            self.timetable_headway_value_planner_action_basis_centers = (
                np.asarray(sorted({float(x) for x in basis_centers}),
                           dtype=np.float32).reshape(-1))
            if self.timetable_headway_value_planner_action_basis_centers.size == 0:
                self.timetable_headway_value_planner_action_basis_centers = (
                    np.asarray([0.0], dtype=np.float32))
            self.timetable_headway_value_planner_action_basis_interactions = bool(
                action_basis_cfg.get('interactions', True))
            self.timetable_headway_value_planner_prior_weight = max(float(
                headway_value_cfg.get('prior_weight', 0.0)), 0.0)
            self.timetable_headway_value_planner_spacing_weight = max(float(
                headway_value_cfg.get('spacing_weight', 1.0)), 0.0)
            self.timetable_headway_value_planner_wait_weight = max(float(
                headway_value_cfg.get('wait_weight', 0.5)), 0.0)
            self.timetable_headway_value_planner_fleet_weight = max(float(
                headway_value_cfg.get('fleet_weight', 0.5)), 0.0)
            self.timetable_headway_value_planner_cv_target = float(
                headway_value_cfg.get('cv_target', 0.438))
            self.timetable_headway_value_planner_overshoot_target = max(float(
                headway_value_cfg.get('overshoot_target', 0.13)), 0.0)
            self.timetable_headway_value_planner_wait_target_min = max(float(
                headway_value_cfg.get('wait_target_min', 5.5)), 0.0)
            self.timetable_headway_value_planner_terminal_shift_target_s = max(
                float(headway_value_cfg.get(
                    'terminal_shift_target_s', 12.0)), 0.0)
            self.timetable_headway_value_planner_target = str(
                headway_value_cfg.get(
                    'target', 'episode_composite')).strip().lower()
            self.timetable_headway_value_planner_episode_weight = float(
                headway_value_cfg.get('episode_weight', 1.0))
            self.timetable_headway_value_planner_local_weight = float(
                headway_value_cfg.get('local_weight', 1.0))
            self.timetable_headway_value_planner_reward_weight = float(
                headway_value_cfg.get('reward_weight', 1.0))
            headway_gate_cfg = (
                headway_value_cfg.get('activation_gate', {}) or {})
            self.timetable_headway_value_planner_gate_enable = bool(
                headway_gate_cfg.get('enable', False))

            def _gate_optional_float(name):
                value = headway_gate_cfg.get(name)
                return None if value is None else float(value)

            self.timetable_headway_value_planner_gate_min_low_forecast = (
                _gate_optional_float('min_low_forecast'))
            self.timetable_headway_value_planner_gate_max_low_forecast = (
                _gate_optional_float('max_low_forecast'))
            self.timetable_headway_value_planner_gate_min_high_energy = (
                _gate_optional_float('min_high_energy'))
            self.timetable_headway_value_planner_gate_max_high_energy = (
                _gate_optional_float('max_high_energy'))
            self.timetable_headway_value_planner_gate_min_middle_energy = (
                _gate_optional_float('min_middle_energy'))
            self.timetable_headway_value_planner_gate_max_middle_energy = (
                _gate_optional_float('max_middle_energy'))
            self.timetable_headway_value_planner_gate_min_od_entropy = (
                _gate_optional_float('min_od_entropy'))
            self.timetable_headway_value_planner_gate_max_od_entropy = (
                _gate_optional_float('max_od_entropy'))
            self.timetable_headway_value_planner_gate_max_promotion_strength = (
                _gate_optional_float('max_promotion_strength'))
            gate_any_of = headway_gate_cfg.get('any_of', []) or []
            if isinstance(gate_any_of, dict):
                gate_any_of = [gate_any_of]
            self.timetable_headway_value_planner_gate_any_of = []
            gate_keys = (
                'min_low_forecast', 'max_low_forecast',
                'min_high_energy', 'max_high_energy',
                'min_middle_energy', 'max_middle_energy',
                'min_od_entropy', 'max_od_entropy',
                'max_promotion_strength',
            )
            if isinstance(gate_any_of, (list, tuple)):
                for group in gate_any_of:
                    if not isinstance(group, dict):
                        continue
                    parsed = {}
                    for key in gate_keys:
                        value = group.get(key)
                        if value is not None:
                            parsed[key] = float(value)
                    if parsed:
                        self.timetable_headway_value_planner_gate_any_of.append(
                            parsed)
            terminal_floor_cfg = (
                planner_cfg.get('terminal_headway_floor', {}) or {})
            self.timetable_terminal_headway_floor_enable = bool(
                terminal_floor_cfg.get('enable', False))
            default_ratio = (
                1.0 if self.timetable_terminal_headway_floor_enable else 0.0)
            self.timetable_terminal_headway_floor_ratio = max(float(
                terminal_floor_cfg.get('ratio', default_ratio)), 0.0)
            self.timetable_terminal_headway_floor_min_s = max(float(
                terminal_floor_cfg.get('min_s', 0.0)), 0.0)
            upper_value_cfg = (
                planner_cfg.get('upper_residual_value_cost', {}) or {})
            self.upper_residual_value_cost_enable = bool(
                upper_value_cfg.get('enable', False))
            self.upper_residual_value_cost_weight = max(float(
                upper_value_cfg.get('weight', 0.0)), 0.0)
            self.upper_residual_value_cost_action_norm_s = max(float(
                upper_value_cfg.get('action_norm_s', 15.0)), 1e-6)
            self.upper_residual_value_cost_fleet_util_start = float(
                upper_value_cfg.get('fleet_util_start', 0.75))
            self.upper_residual_value_cost_fleet_util_full = float(
                upper_value_cfg.get('fleet_util_full', 1.05))
            self.upper_residual_value_cost_high_start = max(float(
                upper_value_cfg.get('high_energy_start', 0.045)), 0.0)
            self.upper_residual_value_cost_high_full = max(float(
                upper_value_cfg.get('high_energy_full', 0.085)), 1e-6)
            self.upper_residual_value_cost_promotion_relief = max(float(
                upper_value_cfg.get('promotion_relief', 0.5)), 0.0)
        # v2k: elastic fleet — sample N_fleet per episode
        self.fleet_mode = upper_cfg.get('fleet_mode', 'fixed')
        self.fleet_min = upper_cfg.get('fleet_min', 8)
        self.fleet_max = upper_cfg.get('fleet_max', 16)
        self.N_fleet_default = upper_cfg['N_fleet']
        self._current_N_fleet = self.N_fleet_default  # set per-episode in elastic mode
        freq_cfg = config.get('frequency', {})
        self.upper_state_dim = upper_cfg.get('state_dim', 10)
        if (freq_cfg.get('enable', False)
                or self.protocol_version == 'freqduet-eval-v5'):
            self.upper_state_dim = self.env.upper_state_dim
        freq_holdfb_cfg = freq_cfg.get('hold_feedback', {}) or {}
        self.freq_holdfb_enable = bool(freq_holdfb_cfg.get('enable', False))
        self.freq_holdfb_window = max(
            1, int(freq_holdfb_cfg.get('window', 512)))
        self.freq_holdfb_wait_norm_s = max(
            float(freq_holdfb_cfg.get('wait_norm_s', 600.0)), 1e-6)
        self.freq_holdfb_wait_clip = max(
            float(freq_holdfb_cfg.get('wait_clip', 2.0)), 0.0)
        self.freq_holdfb_board_norm = max(
            float(freq_holdfb_cfg.get('board_norm', 8.0)), 1e-6)
        self.freq_holdfb_high_threshold = max(
            float(freq_holdfb_cfg.get('high_threshold', 0.0)), 0.0)
        # Features appended to upper state:
        # [same-dir HF-hold, same-dir HF-wait, other-dir HF-hold, other-dir HF-wait].
        self.freq_holdfb_dim = 4 if self.freq_holdfb_enable else 0
        freq_driftfb_cfg = freq_cfg.get('drift_feedback', {}) or {}
        self.freq_driftfb_enable = bool(freq_driftfb_cfg.get('enable', False))
        self.freq_driftfb_norm_s = max(
            float(freq_driftfb_cfg.get(
                'norm_s',
                config.get('leakage', {}).get('lower_drift_budget_s', 180.0))),
            1e-6)
        self.freq_driftfb_clip = max(
            float(freq_driftfb_cfg.get('clip', 2.0)), 0.0)
        # Features appended to upper state:
        # [same-dir drift load, same-dir excess, other-dir drift load, other-dir excess].
        self.freq_driftfb_dim = 4 if self.freq_driftfb_enable else 0
        state_history_cfg = upper_cfg.get('state_history', {}) or {}
        self.upper_state_history_enable = bool(
            state_history_cfg.get('enable', False))
        self.upper_state_history_len = (
            max(0, int(state_history_cfg.get('length', 0)))
            if self.upper_state_history_enable else 0)
        self.upper_state_history_action_norm_s = max(float(
            state_history_cfg.get('action_norm_s', 60.0)), 1e-6)
        self.upper_state_history_shift_norm_s = max(float(
            state_history_cfg.get('shift_norm_s', 60.0)), 1e-6)
        self.upper_state_history_wait_norm_min = max(float(
            state_history_cfg.get('wait_norm_min', 10.0)), 1e-6)
        self.upper_state_history_waiting_norm = max(float(
            state_history_cfg.get('waiting_norm', 500.0)), 1e-6)
        self.upper_state_history_plan_penalty_norm = max(float(
            state_history_cfg.get('plan_penalty_norm', 5.0)), 1e-6)
        self.upper_state_history_step_dim = (
            23 if self.upper_state_history_len > 0 else 0)
        self.upper_state_history_dim = (
            self.upper_state_history_len
            * self.upper_state_history_step_dim)
        self._upper_state_history = deque(
            maxlen=max(1, self.upper_state_history_len))
        self.upper_state_dim += (
            self.freq_holdfb_dim + self.freq_driftfb_dim
            + self.upper_state_history_dim)
        self.upper_action_dim = int(upper_cfg.get('action_dim', 1))
        if self.timetable_planner is not None:
            self.upper_action_dim = self.timetable_planner.action_dim
        self.upper_plan_context_dim = self.upper_plan_execution.context_dim(
            self.upper_action_dim)
        self.upper_state_dim += self.upper_plan_context_dim
        action_low = upper_cfg.get('action_low', None)
        action_high = upper_cfg.get('action_high', None)
        if self.timetable_planner is not None:
            action_low = self.timetable_planner.action_low
            action_high = self.timetable_planner.action_high
        if action_low is None:
            action_low = [-self.delta_max] * self.upper_action_dim
        if action_high is None:
            action_high = [self.delta_max] * self.upper_action_dim
        self.upper_action_low = np.asarray(action_low, dtype=np.float32)
        self.upper_action_high = np.asarray(action_high, dtype=np.float32)
        self.upper_action_bins = None
        self.upper_action_candidates = None
        self.upper_action_override_enable = False
        self.upper_action_override_values = None
        self.upper_action_override_disable_value_selectors = True
        upper_bins = upper_cfg.get('action_bins', None)
        if upper_bins:
            action_bins = np.asarray(
                [float(x) for x in upper_bins], dtype=np.float32)
            action_bins = np.unique(np.clip(
                action_bins,
                float(self.upper_action_low.min()),
                float(self.upper_action_high.max())))
            if action_bins.size < 2:
                raise ValueError("upper.action_bins must contain at least two values")
            self.upper_action_bins = action_bins
        raw_candidates = upper_cfg.get('action_candidates', None)
        if raw_candidates:
            candidates = np.asarray(raw_candidates, dtype=np.float32)
            if (candidates.ndim != 2
                    or candidates.shape[1] != self.upper_action_dim
                    or candidates.shape[0] < 2):
                raise ValueError(
                    "upper.action_candidates must have shape "
                    f"[n_candidates, {self.upper_action_dim}] with at least "
                    "two candidates")
            if self.upper_action_bins is not None:
                raise ValueError(
                    "upper.action_bins and upper.action_candidates are mutually "
                    "exclusive")
            if (np.any(candidates < self.upper_action_low.reshape(1, -1))
                    or np.any(candidates > self.upper_action_high.reshape(1, -1))):
                raise ValueError(
                    "upper.action_candidates must lie within planner bounds")
            if np.unique(candidates, axis=0).shape[0] != candidates.shape[0]:
                raise ValueError(
                    "upper.action_candidates contains duplicate rows")
            self.upper_action_candidates = candidates

        action_override_cfg = upper_cfg.get('action_override', {}) or {}
        self.upper_action_override_enable = bool(
            action_override_cfg.get('enable', False))
        if self.upper_action_override_enable:
            raw_values = action_override_cfg.get(
                'values_s',
                action_override_cfg.get(
                    'values',
                    action_override_cfg.get('delta_s', 0.0)))
            if isinstance(raw_values, (int, float)):
                override_values = np.full(
                    self.upper_action_dim, float(raw_values),
                    dtype=np.float32)
            else:
                override_values = np.asarray(
                    [float(x) for x in raw_values], dtype=np.float32).reshape(-1)
                if override_values.size == 1 and self.upper_action_dim > 1:
                    override_values = np.full(
                        self.upper_action_dim, float(override_values[0]),
                        dtype=np.float32)
            if override_values.size != self.upper_action_dim:
                raise ValueError(
                    "upper.action_override values size must be 1 or match "
                    f"upper action_dim={self.upper_action_dim}")
            self.upper_action_override_values = np.clip(
                override_values,
                self.upper_action_low,
                self.upper_action_high,
            ).astype(np.float32)
        self.upper_action_override_disable_value_selectors = bool(
            action_override_cfg.get('disable_value_selectors', True))

        cf_action_cfg = (
            upper_cfg.get('counterfactual_action_selector', {}) or {})
        self.cf_action_selector_enable = bool(
            cf_action_cfg.get('enable', False))
        self.cf_action_selector_start_ep = int(
            cf_action_cfg.get('start_ep', 0))
        self.cf_action_selector_artifact = str(
            cf_action_cfg.get('artifact', '')).strip()
        self.cf_action_selector_default_method = str(
            cf_action_cfg.get('default_method', 'target0')).strip()
        self.cf_action_selector_disable_value_selectors = bool(
            cf_action_cfg.get('disable_value_selectors', True))
        self.cf_action_selector_terminal_shift_min_s = float(
            cf_action_cfg.get('terminal_shift_min_s', 0.0))
        self.cf_action_selector_terminal_shift_max_s = float(
            cf_action_cfg.get('terminal_shift_max_s', 45.0))
        self.cf_action_selector_ep_norm_denominator = max(float(
            cf_action_cfg.get('ep_norm_denominator', 99.0)), 1.0)
        allowed_cf_methods = cf_action_cfg.get('allowed_methods', [])
        self.cf_action_selector_allowed_methods = {
            str(method).strip() for method in allowed_cf_methods
            if str(method).strip()
        }
        self.cf_action_selector_model = None
        if self.cf_action_selector_enable:
            if not self.cf_action_selector_artifact:
                raise ValueError(
                    "upper.counterfactual_action_selector.artifact is required")
            artifact_path = Path(self.cf_action_selector_artifact)
            if not artifact_path.is_absolute():
                artifact_path = SCRIPT_DIR / artifact_path
            self.cf_action_selector_model = (
                CounterfactualActionTreeSelector.load(artifact_path))

        snapshot_selector_cfg = (
            upper_cfg.get('snapshot_value_selector', {}) or {})
        self.snapshot_value_selector_enable = bool(
            snapshot_selector_cfg.get('enable', False))
        self.snapshot_value_selector_start_configured = (
            'start_ep' in snapshot_selector_cfg)
        self.snapshot_value_selector_start_ep = int(
            snapshot_selector_cfg.get('start_ep', 30))
        self.snapshot_value_selector_artifact = str(
            snapshot_selector_cfg.get('artifact', '')).strip()
        self.snapshot_value_selector_domain = str(
            snapshot_selector_cfg.get('domain', '')).strip().lower()
        self.snapshot_value_selector_improve_margin = max(float(
            snapshot_selector_cfg.get('improve_margin', 0.0)), 0.0)
        self.snapshot_value_selector_fallback_method = str(
            snapshot_selector_cfg.get('fallback_method', 'term45_0')).strip()
        self.snapshot_value_selector_fallback_action = str(
            snapshot_selector_cfg.get('fallback_action', 'candidate')
        ).strip().lower()
        self.snapshot_value_selector_apply_mode = str(
            snapshot_selector_cfg.get('apply_mode', 'action_override')
        ).strip().lower()
        if self.snapshot_value_selector_apply_mode in {
                'terminal', 'terminal_bias_only', 'bias'}:
            self.snapshot_value_selector_apply_mode = 'terminal_bias'
        if self.snapshot_value_selector_apply_mode not in {
                'action_override', 'terminal_bias'}:
            raise ValueError(
                "upper.snapshot_value_selector.apply_mode must be "
                "'action_override' or 'terminal_bias'")
        allowed_methods = snapshot_selector_cfg.get('allowed_methods', [])
        blocked_methods = snapshot_selector_cfg.get('blocked_methods', [])
        self.snapshot_value_selector_allowed_methods = {
            str(method).strip() for method in allowed_methods
            if str(method).strip()
        }
        self.snapshot_value_selector_blocked_methods = {
            str(method).strip() for method in blocked_methods
            if str(method).strip()
        }
        snapshot_candidate_gate_cfg = (
            snapshot_selector_cfg.get('candidate_gate', {}) or {})
        self.snapshot_value_candidate_gate_enable = bool(
            snapshot_candidate_gate_cfg.get('enable', False))

        def _candidate_gate_float(name):
            value = snapshot_candidate_gate_cfg.get(name)
            return None if value is None else float(value)

        self.snapshot_value_candidate_gate_default_max_positive_offset_s = (
            _candidate_gate_float('default_max_positive_offset_s'))
        self.snapshot_value_candidate_gate_high_noise_min_demand_noise = (
            _candidate_gate_float('high_noise_min_demand_noise'))
        self.snapshot_value_candidate_gate_high_noise_max_positive_offset_s = (
            _candidate_gate_float('high_noise_max_positive_offset_s'))
        self.snapshot_value_candidate_gate_risk_max_positive_offset_s = (
            _candidate_gate_float('risk_max_positive_offset_s'))
        self.snapshot_value_candidate_gate_max_prev_headway_cv = (
            _candidate_gate_float('max_prev_headway_cv'))
        self.snapshot_value_candidate_gate_max_prev_overshoot_norm = (
            _candidate_gate_float('max_prev_overshoot_norm'))
        self.snapshot_value_candidate_gate_max_prev_terminal_shift_std_s = (
            _candidate_gate_float('max_prev_terminal_shift_std_s'))
        snapshot_risk_penalty_cfg = (
            snapshot_selector_cfg.get('risk_penalty', {}) or {})
        self.snapshot_value_risk_penalty_enable = bool(
            snapshot_risk_penalty_cfg.get('enable', False))

        def _risk_penalty_float(name, default=None):
            value = snapshot_risk_penalty_cfg.get(name, default)
            return None if value is None else float(value)

        self.snapshot_value_risk_penalty_weight = max(float(
            snapshot_risk_penalty_cfg.get('weight', 0.0)), 0.0)
        self.snapshot_value_risk_penalty_positive_offset_start_s = max(
            _risk_penalty_float('positive_offset_start_s', 15.0) or 0.0,
            0.0)
        self.snapshot_value_risk_penalty_positive_offset_scale_s = max(
            _risk_penalty_float('positive_offset_scale_s', 15.0) or 15.0,
            1e-6)
        self.snapshot_value_risk_penalty_max_score = max(
            _risk_penalty_float('max_risk_score', 3.0) or 0.0,
            0.0)
        self.snapshot_value_risk_penalty_max_penalty = max(
            _risk_penalty_float('max_penalty', 0.05) or 0.0,
            0.0)
        self.snapshot_value_risk_penalty_prev_headway_cv_target = (
            _risk_penalty_float('prev_headway_cv_target'))
        self.snapshot_value_risk_penalty_prev_headway_cv_width = max(
            _risk_penalty_float('prev_headway_cv_width', 0.05) or 0.05,
            1e-6)
        self.snapshot_value_risk_penalty_prev_overshoot_norm_target = (
            _risk_penalty_float('prev_overshoot_norm_target'))
        self.snapshot_value_risk_penalty_prev_overshoot_norm_width = max(
            _risk_penalty_float('prev_overshoot_norm_width', 0.075) or 0.075,
            1e-6)
        self.snapshot_value_risk_penalty_prev_terminal_shift_std_target_s = (
            _risk_penalty_float('prev_terminal_shift_std_target_s'))
        self.snapshot_value_risk_penalty_prev_terminal_shift_std_width_s = max(
            _risk_penalty_float('prev_terminal_shift_std_width_s', 4.0) or 4.0,
            1e-6)
        self.snapshot_value_risk_penalty_context_headway_cv_target = (
            _risk_penalty_float('context_headway_cv_target'))
        self.snapshot_value_risk_penalty_context_headway_cv_width = max(
            _risk_penalty_float('context_headway_cv_width', 0.05) or 0.05,
            1e-6)
        self.snapshot_value_risk_penalty_context_fleet_pressure_target = (
            _risk_penalty_float('context_fleet_pressure_target'))
        self.snapshot_value_risk_penalty_context_fleet_pressure_width = max(
            _risk_penalty_float('context_fleet_pressure_width', 0.25) or 0.25,
            1e-6)
        self.snapshot_value_selector_probe_only = bool(
            snapshot_selector_cfg.get('probe_only', False))
        self.snapshot_value_selector_model = None
        self.snapshot_value_selector_forest = None
        self.snapshot_value_selector_meta = {}
        self.snapshot_value_selector_feature_cols = []
        self.snapshot_value_selector_feature_medians = {}
        self.snapshot_value_selector_candidate_methods = []
        if self.snapshot_value_selector_enable:
            self._load_snapshot_value_selector()

        snapshot_action_selector_cfg = (
            upper_cfg.get('snapshot_action_value_selector', {}) or {})
        self.snapshot_action_value_selector_enable = bool(
            snapshot_action_selector_cfg.get('enable', False))
        self.snapshot_action_value_selector_start_configured = (
            'start_ep' in snapshot_action_selector_cfg)
        self.snapshot_action_value_selector_start_ep = int(
            snapshot_action_selector_cfg.get('start_ep', 30))
        self.snapshot_action_value_selector_artifact = str(
            snapshot_action_selector_cfg.get('artifact', '')).strip()
        self.snapshot_action_value_selector_domain = str(
            snapshot_action_selector_cfg.get('domain', '')).strip().lower()
        self.snapshot_action_value_selector_improve_margin = max(float(
            snapshot_action_selector_cfg.get('improve_margin', 0.0)), 0.0)
        self.snapshot_action_value_selector_fallback_method = str(
            snapshot_action_selector_cfg.get(
                'fallback_method', 'actor_term45_0')).strip()
        self.snapshot_action_value_selector_fallback_action = str(
            snapshot_action_selector_cfg.get('fallback_action', 'actor')
        ).strip().lower()
        allowed_action_methods = snapshot_action_selector_cfg.get(
            'allowed_methods', [])
        blocked_action_methods = snapshot_action_selector_cfg.get(
            'blocked_methods', [])
        self.snapshot_action_value_selector_allowed_methods = {
            str(method).strip() for method in allowed_action_methods
            if str(method).strip()
        }
        self.snapshot_action_value_selector_blocked_methods = {
            str(method).strip() for method in blocked_action_methods
            if str(method).strip()
        }
        action_guard_cfg = (
            snapshot_action_selector_cfg.get('guard', {}) or {})
        self.snapshot_action_value_guard_enable = bool(
            action_guard_cfg.get('enable', False))

        def _action_guard_float(name, default=None):
            value = action_guard_cfg.get(name, default)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        self.snapshot_action_value_guard_max_prev_overshoot_norm = (
            _action_guard_float('max_prev_overshoot_norm'))
        self.snapshot_action_value_guard_max_prev_headway_cv = (
            _action_guard_float('max_prev_headway_cv'))
        self.snapshot_action_value_guard_max_context_headway_cv = (
            _action_guard_float('max_context_headway_cv'))
        self.snapshot_action_value_guard_max_fleet_pressure_norm = (
            _action_guard_float('max_fleet_pressure_norm'))
        self.snapshot_action_value_guard_max_abs_offset_s = (
            _action_guard_float('max_abs_offset_s'))
        self.snapshot_action_value_guard_min_margin = (
            _action_guard_float('min_margin'))
        self.snapshot_action_value_guard_min_margin_per_abs_offset_norm = (
            _action_guard_float('min_margin_per_abs_offset_norm', 0.0))
        self.snapshot_action_value_guard_max_peak_shift_abs = (
            _action_guard_float('max_peak_shift_abs'))
        self.snapshot_action_value_guard_max_primary_terminal_bias_loss_s = (
            _action_guard_float('max_primary_terminal_bias_loss_s'))
        self.snapshot_action_value_guard_max_negative_target_prev_overshoot_norm = (
            _action_guard_float('max_negative_target_prev_overshoot_norm'))
        self.snapshot_action_value_guard_max_negative_target_fleet_pressure_norm = (
            _action_guard_float('max_negative_target_fleet_pressure_norm'))
        self.snapshot_action_value_guard_min_negative_target_margin = (
            _action_guard_float('min_negative_target_margin'))
        action_risk_margin_cfg = (
            snapshot_action_selector_cfg.get('risk_margin', {}) or {})
        self.snapshot_action_value_risk_margin_enable = bool(
            action_risk_margin_cfg.get('enable', False))

        def _action_risk_margin_float(name, default=None):
            value = action_risk_margin_cfg.get(name, default)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        self.snapshot_action_value_risk_margin_weight = max(float(
            action_risk_margin_cfg.get('weight', 0.0)), 0.0)
        self.snapshot_action_value_risk_margin_max_score = max(
            _action_risk_margin_float('max_risk_score', 3.0) or 0.0,
            0.0)
        self.snapshot_action_value_risk_margin_max_penalty = max(
            _action_risk_margin_float('max_penalty', 0.05) or 0.0,
            0.0)
        self.snapshot_action_value_risk_margin_abs_offset_scale_s = max(
            _action_risk_margin_float('abs_offset_scale_s', 15.0) or 15.0,
            1e-6)
        self.snapshot_action_value_risk_margin_target_base = max(
            _action_risk_margin_float('target_base', 0.0) or 0.0,
            0.0)
        self.snapshot_action_value_risk_margin_negative_multiplier = max(
            _action_risk_margin_float('negative_multiplier', 1.0) or 1.0,
            0.0)
        self.snapshot_action_value_risk_margin_positive_multiplier = max(
            _action_risk_margin_float('positive_multiplier', 1.0) or 1.0,
            0.0)
        self.snapshot_action_value_risk_margin_term45_multiplier = max(
            _action_risk_margin_float('term45_multiplier', 1.0) or 1.0,
            0.0)
        self.snapshot_action_value_risk_margin_prev_headway_cv_target = (
            _action_risk_margin_float('prev_headway_cv_target'))
        self.snapshot_action_value_risk_margin_prev_headway_cv_width = max(
            _action_risk_margin_float('prev_headway_cv_width', 0.05) or 0.05,
            1e-6)
        self.snapshot_action_value_risk_margin_prev_overshoot_norm_target = (
            _action_risk_margin_float('prev_overshoot_norm_target'))
        self.snapshot_action_value_risk_margin_prev_overshoot_norm_width = max(
            _action_risk_margin_float('prev_overshoot_norm_width', 0.075)
            or 0.075,
            1e-6)
        self.snapshot_action_value_risk_margin_context_headway_cv_target = (
            _action_risk_margin_float('context_headway_cv_target'))
        self.snapshot_action_value_risk_margin_context_headway_cv_width = max(
            _action_risk_margin_float('context_headway_cv_width', 0.05)
            or 0.05,
            1e-6)
        self.snapshot_action_value_risk_margin_context_fleet_pressure_target = (
            _action_risk_margin_float('context_fleet_pressure_target'))
        self.snapshot_action_value_risk_margin_context_fleet_pressure_width = max(
            _action_risk_margin_float('context_fleet_pressure_width', 0.25)
            or 0.25,
            1e-6)
        self.snapshot_action_value_risk_margin_primary_bias_target_s = (
            _action_risk_margin_float('primary_terminal_bias_target_s'))
        self.snapshot_action_value_risk_margin_primary_bias_width_s = max(
            _action_risk_margin_float('primary_terminal_bias_width_s', 6.0)
            or 6.0,
            1e-6)
        self.snapshot_action_value_risk_margin_peak_shift_abs_target = (
            _action_risk_margin_float('peak_shift_abs_target'))
        self.snapshot_action_value_risk_margin_peak_shift_abs_width = max(
            _action_risk_margin_float('peak_shift_abs_width', 1.0) or 1.0,
            1e-6)
        self.snapshot_action_value_selector_model = None
        self.snapshot_action_value_selector_forest = None
        self.snapshot_action_value_selector_meta = {}
        self.snapshot_action_value_selector_feature_cols = []
        self.snapshot_action_value_selector_feature_medians = {}
        self.snapshot_action_value_selector_candidate_methods = []
        if self.snapshot_action_value_selector_enable:
            self._load_snapshot_action_value_selector()

        residual_selector_cfg = (
            upper_cfg.get('residual_value_selector', {}) or {})
        self.upper_residual_selector_enable = bool(
            residual_selector_cfg.get('enable', False))
        self.upper_residual_selector_start_configured = (
            'start_ep' in residual_selector_cfg)
        self.upper_residual_selector_learn_start_configured = (
            'learn_start_ep' in residual_selector_cfg)
        self.upper_residual_selector_start_ep = int(
            residual_selector_cfg.get('start_ep', self.upper_warmup
                                      if hasattr(self, 'upper_warmup') else 30))
        self.upper_residual_selector_learn_start_ep = int(
            residual_selector_cfg.get(
                'learn_start_ep', max(0, self.upper_residual_selector_start_ep - 20)))
        self.upper_residual_selector_min_observations = max(
            1, int(residual_selector_cfg.get('min_observations', 32)))
        self.upper_residual_selector_ridge = max(float(
            residual_selector_cfg.get('ridge', 0.25)), 1e-6)
        self.upper_residual_selector_feature_clip = max(float(
            residual_selector_cfg.get('feature_clip', 6.0)), 0.0)
        self.upper_residual_selector_cost_clip = max(float(
            residual_selector_cfg.get('cost_clip', 8.0)), 0.0)
        self.upper_residual_selector_epsilon = float(np.clip(
            residual_selector_cfg.get('epsilon', 0.0), 0.0, 1.0))
        self.upper_residual_selector_improve_margin = max(float(
            residual_selector_cfg.get('improve_margin', 0.0)), 0.0)
        self.upper_residual_selector_adjust_penalty = max(float(
            residual_selector_cfg.get('adjust_penalty', 0.02)), 0.0)
        self.upper_residual_selector_adjust_norm_s = max(float(
            residual_selector_cfg.get('adjust_norm_s', 10.0)), 1e-6)
        self.upper_residual_selector_feature_mode = str(
            residual_selector_cfg.get('feature_mode', 'legacy')).lower()
        self.upper_residual_selector_plan_context = (
            self.upper_residual_selector_feature_mode in {
                'plan_context', 'local_plan', 'planctx', 'contextual_plan'})
        self.upper_residual_selector_compression_safety_weight = max(float(
            residual_selector_cfg.get('compression_safety_weight', 0.0)), 0.0)
        self.upper_residual_selector_compression_norm_s = max(float(
            residual_selector_cfg.get('compression_norm_s', 5.0)), 1e-6)
        self.upper_residual_selector_short_gap_weight = max(float(
            residual_selector_cfg.get('short_gap_weight', 0.0)), 0.0)
        self.upper_residual_selector_fleet_pressure_weight = max(float(
            residual_selector_cfg.get('fleet_pressure_weight', 0.0)), 0.0)
        offsets = residual_selector_cfg.get(
            'candidate_offsets_s', [-10.0, -5.0, 0.0, 5.0, 10.0])
        self.upper_residual_selector_offsets = np.asarray(
            [float(x) for x in offsets], dtype=np.float32).reshape(-1)
        if not np.any(np.isclose(self.upper_residual_selector_offsets, 0.0)):
            self.upper_residual_selector_offsets = np.concatenate([
                self.upper_residual_selector_offsets,
                np.asarray([0.0], dtype=np.float32),
            ])
        self.upper_residual_selector_A = None
        self.upper_residual_selector_b = None
        self.upper_residual_selector_updates = 0

        noharm_cfg = config.get('fleet_noharm', {}) or {}
        upper_noharm_cfg = noharm_cfg.get('upper', {}) or {}
        lower_noharm_cfg = noharm_cfg.get('lower', {}) or {}
        self.fleet_noharm_upper_enable = bool(
            upper_noharm_cfg.get('enable', False))
        self.fleet_noharm_upper_pressure_start = float(
            upper_noharm_cfg.get('pressure_start', 0.0))
        self.fleet_noharm_upper_pressure_full = float(
            upper_noharm_cfg.get('pressure_full', 2.0))
        self.fleet_noharm_upper_shrink_max = float(np.clip(
            upper_noharm_cfg.get('shrink_max', 1.0), 0.0, 1.0))
        self.fleet_noharm_upper_mode = str(
            upper_noharm_cfg.get('mode', 'all')).lower()
        self.fleet_noharm_upper_neutral_s = float(
            upper_noharm_cfg.get('neutral_s', 0.0))
        self.fleet_noharm_upper_gate = self._parse_fleet_noharm_gate(
            upper_noharm_cfg.get('gate', {}))
        self.fleet_noharm_lower_enable = bool(
            lower_noharm_cfg.get('enable', False))
        self.fleet_noharm_lower_pressure_start = float(
            lower_noharm_cfg.get('pressure_start', 0.0))
        self.fleet_noharm_lower_pressure_full = float(
            lower_noharm_cfg.get('pressure_full', 2.0))
        self.fleet_noharm_lower_shrink_max = float(np.clip(
            lower_noharm_cfg.get('shrink_max', 1.0), 0.0, 1.0))
        self.fleet_noharm_lower_min_action_s = float(
            lower_noharm_cfg.get('min_action_s', 0.0))
        self.fleet_noharm_lower_gate = self._parse_fleet_noharm_gate(
            lower_noharm_cfg.get('gate', {}))
        lower_proactive_cfg = lower_noharm_cfg.get('proactive', {}) or {}
        self.fleet_noharm_lower_proactive_enable = bool(
            lower_proactive_cfg.get('enable', False))
        self.fleet_noharm_lower_proactive_pressure_start = float(
            lower_proactive_cfg.get('pressure_start', -2.0))
        self.fleet_noharm_lower_proactive_pressure_full = float(
            lower_proactive_cfg.get('pressure_full', -1.0))
        self.fleet_noharm_lower_proactive_shrink_max = float(np.clip(
            lower_proactive_cfg.get('shrink_max', 1.0), 0.0, 1.0))
        self.fleet_noharm_lower_proactive_gate = self._parse_fleet_noharm_gate(
            lower_proactive_cfg.get('gate', {}))
        lower_value_guard_cfg = lower_noharm_cfg.get('value_guard', {}) or {}
        self.fleet_noharm_lower_value_guard_enable = bool(
            lower_value_guard_cfg.get('enable', False))
        self.fleet_noharm_lower_value_guard_pressure_start = float(
            lower_value_guard_cfg.get('pressure_start', 0.0))
        self.fleet_noharm_lower_value_guard_pressure_full = float(
            lower_value_guard_cfg.get('pressure_full', 2.0))
        self.fleet_noharm_lower_value_guard_min_ratio = max(float(
            lower_value_guard_cfg.get('min_ratio', 1.0)), 0.0)
        self.fleet_noharm_lower_value_guard_cost_weight = max(float(
            lower_value_guard_cfg.get('cost_weight', 1.0)), 1e-6)
        self.fleet_noharm_lower_value_guard_action_norm_s = max(float(
            lower_value_guard_cfg.get('action_norm_s', 45.0)), 1e-6)
        self.fleet_noharm_lower_value_guard_wait_norm_s = max(float(
            lower_value_guard_cfg.get('wait_norm_s', 90.0)), 1e-6)
        self.fleet_noharm_lower_value_guard_wait_clip = max(float(
            lower_value_guard_cfg.get('wait_clip', 2.0)), 0.0)
        self.fleet_noharm_lower_value_guard_board_norm = max(float(
            lower_value_guard_cfg.get('board_norm', 10.0)), 1e-6)
        self.fleet_noharm_lower_value_guard_board_clip = max(float(
            lower_value_guard_cfg.get('board_clip', 2.0)), 0.0)
        self.fleet_noharm_lower_value_guard_board_weight = max(float(
            lower_value_guard_cfg.get('board_weight', 0.25)), 0.0)
        self.fleet_noharm_lower_value_guard_headway_weight = max(float(
            lower_value_guard_cfg.get('headway_weight', 1.0)), 0.0)
        self.fleet_noharm_lower_value_guard_headway_clip = max(float(
            lower_value_guard_cfg.get('headway_clip', 1.0)), 0.0)
        self.fleet_noharm_lower_value_guard_low_floor = max(float(
            lower_value_guard_cfg.get('low_floor', 1e-3)), 1e-9)
        self.fleet_noharm_lower_value_guard_high_share_cap = float(
            lower_value_guard_cfg.get('high_share_cap', 1.0))
        self.fleet_noharm_lower_value_guard_positive_high_only = bool(
            lower_value_guard_cfg.get('positive_high_only', True))
        self.fleet_noharm_lower_value_guard_max_shrink = float(np.clip(
            lower_value_guard_cfg.get('max_shrink', 1.0), 0.0, 1.0))
        self.fleet_noharm_lower_value_guard_min_action_s = max(float(
            lower_value_guard_cfg.get(
                'min_action_s', self.fleet_noharm_lower_min_action_s)), 0.0)
        self.fleet_noharm_lower_value_guard_gate = self._parse_fleet_noharm_gate(
            lower_value_guard_cfg.get('gate', {}))
        lower_value_soft_cfg = (
            lower_noharm_cfg.get('value_soft_cost', {}) or {})
        self.fleet_noharm_lower_value_soft_cost_enable = bool(
            lower_value_soft_cfg.get('enable', False))
        self.fleet_noharm_lower_value_soft_cost_pressure_start = float(
            lower_value_soft_cfg.get(
                'pressure_start',
                self.fleet_noharm_lower_value_guard_pressure_start))
        self.fleet_noharm_lower_value_soft_cost_pressure_full = float(
            lower_value_soft_cfg.get(
                'pressure_full',
                self.fleet_noharm_lower_value_guard_pressure_full))
        self.fleet_noharm_lower_value_soft_cost_min_ratio = max(float(
            lower_value_soft_cfg.get(
                'min_ratio',
                self.fleet_noharm_lower_value_guard_min_ratio)), 0.0)
        self.fleet_noharm_lower_value_soft_cost_weight = max(float(
            lower_value_soft_cfg.get(
                'cost_weight',
                self.fleet_noharm_lower_value_guard_cost_weight)), 0.0)
        self.fleet_noharm_lower_value_soft_cost_action_norm_s = max(float(
            lower_value_soft_cfg.get(
                'action_norm_s',
                self.fleet_noharm_lower_value_guard_action_norm_s)), 1e-6)
        self.fleet_noharm_lower_value_soft_cost_violation_weight = max(float(
            lower_value_soft_cfg.get('violation_weight', 1.0)), 0.0)
        self.fleet_noharm_lower_value_soft_cost_cap = max(float(
            lower_value_soft_cfg.get('cap', 1.0)), 0.0)
        self.fleet_noharm_lower_value_soft_cost_gate = self._parse_fleet_noharm_gate(
            lower_value_soft_cfg.get('gate', {}))

        if self.decouple_init_seeds and not self.randomness.isolated:
            torch.manual_seed(self.base_seed + 1001)
        with self.randomness.torch_initialization('upper_init'):
            self.upper_trainer = RESACUpperTrainer(
                state_dim=self.upper_state_dim, action_dim=self.upper_action_dim,
                hidden_dim=upper_cfg.get('hidden_dim', 64),
                action_low=self.upper_action_low.tolist(),
                action_high=self.upper_action_high.tolist(),
                action_candidates=(
                    self.upper_action_candidates.tolist()
                    if self.upper_action_candidates is not None else None),
                discrete_critic=upper_cfg.get(
                    'discrete_critic', 'continuous_action'),
                ensemble_size=upper_cfg.get('ensemble_size', 10),
                beta=upper_cfg.get('resac_beta', -2.0),
                beta_ood=upper_cfg.get('beta_ood', 0.01),
                weight_reg=upper_cfg.get('weight_reg', 0.01),
                weight_reg_mode=upper_cfg.get('weight_reg_mode', 'sum'),
                lr=upper_cfg.get('lr', 3e-4),
                gamma=upper_cfg.get('gamma', 0.95),
                soft_tau=upper_cfg.get('soft_tau', 5e-3),
                auto_entropy=upper_cfg.get('auto_entropy', True),
                maximum_alpha=upper_cfg.get('maximum_alpha', 0.05),
                initial_alpha=upper_cfg.get('initial_alpha', 0.1),
                minimum_alpha=upper_cfg.get('minimum_alpha', 1e-5),
                temperature_contract=upper_cfg.get(
                    'temperature_contract', 'legacy_capped_scalar'),
                critic_aggregation=upper_cfg.get(
                    'critic_aggregation', 'ensemble_mean_lcb'),
                policy_sample_seed=(
                    self.randomness.seed('upper_policy')
                    if self.randomness.isolated else None),
                replay_seed=(
                    self.randomness.seed('upper_replay')
                    if self.randomness.isolated else None),
                replay_capacity=upper_cfg.get('replay_capacity', 50000),
                device=device)

        # ── Lower policy ──
        lower_cfg = config['lower']
        self.replay_buffer = CostReplayBuffer(
            config['training']['replay_buffer_size'],
            seed=(self.randomness.seed('lower_replay')
                  if self.randomness.isolated else None))
        self.lower_action_bins = None
        bins = lower_cfg.get('action_bins', None)
        if bins:
            action_bins = np.asarray([float(x) for x in bins], dtype=np.float32)
            action_bins = np.unique(np.clip(
                action_bins, 0.0, float(lower_cfg['action_range'])))
            if action_bins.size < 2:
                raise ValueError("lower.action_bins must contain at least two values")
            self.lower_action_bins = action_bins
        lower_bins_gate_cfg = lower_cfg.get('action_bins_gate', {}) or {}
        self.lower_action_bins_gate_enabled = bool(
            lower_bins_gate_cfg.get('enable', False))
        self.lower_action_bins_gate_source = str(
            lower_bins_gate_cfg.get('source', 'lower_context_gate')).lower()
        self.lower_action_bins_gate_threshold = float(
            lower_bins_gate_cfg.get('threshold', 0.5))
        if self.lower_action_bins_gate_enabled and self.lower_action_bins is None:
            raise ValueError(
                "lower.action_bins_gate requires lower.action_bins to be set")
        if (self.lower_action_bins_gate_enabled
                and self.lower_action_bins_gate_source not in {
                    'lower_context_gate', 'context_gate'}):
            raise ValueError(
                "lower.action_bins_gate.source only supports lower_context_gate")
        self.lower_use_last_action_feature = bool(
            lower_cfg.get('use_last_action_feature', False))
        self.lower_terminal_action_mode = str(
            lower_cfg.get('terminal_action_mode', 'legacy')).strip().lower()
        if self.lower_terminal_action_mode not in {
                'legacy', 'mask', 'transition'}:
            raise ValueError(
                "lower.terminal_action_mode must be legacy, mask, or transition")
        self.lower_trip_boundary_mode = str(
            lower_cfg.get('trip_boundary_mode', 'legacy')).strip().lower()
        self.lower_holding_action_trace_mode = str(lower_cfg.get(
            'holding_action_trace_mode', 'positive_only')).strip().lower()
        if self.lower_holding_action_trace_mode not in {
                'positive_only', 'all_decisions'}:
            raise ValueError(
                "lower.holding_action_trace_mode must be positive_only or "
                "all_decisions")
        self.lower_unobserved_action_mode = str(lower_cfg.get(
            'unobserved_action_mode', 'legacy_stale')).strip().lower()
        if self.lower_unobserved_action_mode not in {
                'legacy_stale', 'zero'}:
            raise ValueError(
                "lower.unobserved_action_mode must be legacy_stale or zero")
        if (self.lower_terminal_action_mode == 'transition'
                and self.lower_trip_boundary_mode != 'reset'):
            raise ValueError(
                "lower.terminal_action_mode=transition requires "
                "lower.trip_boundary_mode=reset")
        if (self.lower_terminal_action_mode == 'transition'
                and self.lower_holding_action_trace_mode != 'all_decisions'):
            raise ValueError(
                "lower.terminal_action_mode=transition requires "
                "lower.holding_action_trace_mode=all_decisions")
        if (self.lower_terminal_action_mode == 'transition'
                and self.lower_unobserved_action_mode != 'zero'):
            raise ValueError(
                "lower.terminal_action_mode=transition requires "
                "lower.unobserved_action_mode=zero")
        headway_mode_aliases = {
            'event': 'arrival_event',
            'causal_event': 'arrival_event',
            'spatial': 'spatial_fallback',
            'legacy_spatial': 'spatial_fallback',
        }
        self.lower_headway_state_mode = str(lower_cfg.get(
            'headway_state_mode', 'arrival_event')).strip().lower()
        self.lower_headway_state_mode = headway_mode_aliases.get(
            self.lower_headway_state_mode, self.lower_headway_state_mode)
        if self.lower_headway_state_mode not in {
                'arrival_event', 'spatial_fallback'}:
            raise ValueError(
                "lower.headway_state_mode must be arrival_event or "
                "spatial_fallback")
        self.env.headway_state_mode = self.lower_headway_state_mode
        self.env.holding_action_trace_mode = (
            self.lower_holding_action_trace_mode)
        self.env.unobserved_action_mode = self.lower_unobserved_action_mode
        lower_state_encoder_cfg = lower_cfg.get('state_encoder', {}) or {}
        self.lower_state_input_schema = str(lower_state_encoder_cfg.get(
            'input_schema', 'legacy_headway_deviation')).strip().lower()
        if self.lower_state_input_schema not in {
                'legacy_headway_deviation', 'explicit_target_v2',
                'causal_forward_v4'}:
            raise ValueError(
                "lower.state_encoder.input_schema must be "
                "legacy_headway_deviation, explicit_target_v2, or "
                "causal_forward_v4")
        self.env.lower_state_input_schema = self.lower_state_input_schema
        self.lower_observation_contract = str(lower_cfg.get(
            'observation_contract', self.env.lower_observation_contract
        )).strip().lower()
        self.lower_headway_reward_mode = str(lower_cfg.get(
            'headway_reward_mode', self.env.headway_reward_mode
        )).strip().lower()
        if self.lower_observation_contract not in {
                'latent_oracle_legacy', 'deployable_apc_avl_v4'}:
            raise ValueError('unknown lower.observation_contract')
        if self.lower_headway_reward_mode not in {
                'symmetric_legacy', 'forward_event_only'}:
            raise ValueError('unknown lower.headway_reward_mode')
        self.lower_observation_spec = LowerObservationContract.create(
            mode=self.lower_observation_contract,
            input_schema=self.lower_state_input_schema,
            reward_mode=self.lower_headway_reward_mode,
            unobserved_action_mode=self.lower_unobserved_action_mode,
            frequency_enabled=self.env.frequency_enabled,
            frequency_source=self.env.frequency_observation_source,
            context_features=self.env.lower_context_features,
        )
        self.lower_load_holding_penalty = (
            LoadWeightedHoldingPenalty.from_config(
                lower_cfg.get('load_weighted_holding', {})))
        self.lower_load_holding_penalty.validate_observation_contract(
            observation_mode=self.lower_observation_contract,
            context_features=self.env.lower_context_features,
        )
        self.lower_causal_holding_guard = CausalHoldingActionGuard.from_config(
            lower_cfg.get('causal_holding_guard', {}))
        self.env.lower_observation_contract = self.lower_observation_contract
        self.env.headway_reward_mode = self.lower_headway_reward_mode
        self.lower_state_encoder = None
        if bool(lower_state_encoder_cfg.get('enable', False)):
            encoder_mode = str(lower_state_encoder_cfg.get(
                'mode', 'physical_dimensionless_v1')).lower()
            if encoder_mode not in {
                    'physical_dimensionless_v1', 'physical_v1'}:
                raise ValueError(
                    "lower.state_encoder.mode must be "
                    "physical_dimensionless_v1")
            max_station_id = max(
                (int(station.station_id) for station in self.env.stations),
                default=1,
            )
            service_start = int(env_cfg.get('service_start_hour', 6))
            service_end = int(env_cfg.get('service_end_hour', 19))
            if service_end < service_start:
                service_end += 24
            self.lower_state_encoder = PhysicalLowerStateEncoder.from_config(
                lower_state_encoder_cfg,
                base_state_dim=int(self.env._base_state_dim),
                max_station_id=max_station_id,
                service_duration_h=float(service_end - service_start + 1),
                action_range_s=float(lower_cfg['action_range']),
            )
        lower_state_dim = state_dim + (1 if self.lower_use_last_action_feature else 0)
        lower_trainer_action_bins = (
            None if self.lower_action_bins_gate_enabled else self.lower_action_bins)
        if self.decouple_init_seeds and not self.randomness.isolated:
            torch.manual_seed(self.base_seed + 2001)
        with self.randomness.torch_initialization('lower_init'):
            self.lower_trainer = RESACLagrangianTrainer(
                state_dim=lower_state_dim, action_dim=1,
                hidden_dim=lower_cfg['hidden_dim'],
                action_range=lower_cfg['action_range'],
                cost_limit=lower_cfg['cost_limit'],
                ensemble_size=lower_cfg.get('ensemble_size', 10),
                beta=lower_cfg.get('resac_beta', -2.0),
                beta_ood=lower_cfg.get('beta_ood', 0.01),
                weight_reg=lower_cfg.get('weight_reg', 0.01),
                weight_reg_mode=lower_cfg.get('weight_reg_mode', 'sum'),
                lr=lower_cfg['lr'], lambda_lr=lower_cfg['lambda_lr'],
                gamma=lower_cfg['gamma'], soft_tau=lower_cfg['soft_tau'],
                auto_entropy=lower_cfg['auto_entropy'],
                maximum_alpha=lower_cfg['maximum_alpha'],
                initial_alpha=lower_cfg.get('initial_alpha', 0.1),
                minimum_alpha=lower_cfg.get('minimum_alpha', 1e-5),
                temperature_contract=lower_cfg.get(
                    'temperature_contract', 'legacy_capped_scalar'),
                entropy_action_coordinates=lower_cfg.get(
                    'entropy_action_coordinates', 'physical_legacy'),
                cost_limit_semantics=lower_cfg.get(
                    'cost_limit_semantics', 'per_decision_rate'),
                critic_aggregation=lower_cfg.get(
                    'critic_aggregation', 'ensemble_mean_lcb'),
                policy_sample_seed=(
                    self.randomness.seed('lower_policy')
                    if self.randomness.isolated else None),
                action_bins=lower_trainer_action_bins,
                device=device)
        self.lower_state_dim = lower_state_dim

        # ── Coupling ──
        coupling_cfg = config['coupling']
        # Ablation flags (for paper experiments)
        self.ablate_holding_feedback = coupling_cfg.get('ablate_holding_feedback', False)
        self.ablate_csbapr = coupling_cfg.get('ablate_csbapr', False)
        self.ablate_hindsight_credit = coupling_cfg.get('ablate_hindsight_credit', False)
        self.ablate_morl = coupling_cfg.get('ablate_morl', False)
        self.belief_crisis_cp_threshold = float(
            coupling_cfg.get('belief_crisis_cp_threshold', 0.1))
        self.belief_crisis_cp_width = max(float(
            coupling_cfg.get('belief_crisis_cp_width', 0.2)), 1e-6)
        self.belief_crisis_fleet_boost_max = max(float(
            coupling_cfg.get('belief_crisis_fleet_boost_max', 0.3)), 0.0)
        self.belief_stable_window_threshold = float(
            coupling_cfg.get('belief_stable_window_threshold', 15.0))
        self.belief_stable_window_width = max(float(
            coupling_cfg.get('belief_stable_window_width', 5.0)), 1e-6)
        self.belief_stable_quality_shift_max = max(float(
            coupling_cfg.get('belief_stable_quality_shift_max', 0.15)), 0.0)
        self.belief_fleet_weight_floor = float(np.clip(
            coupling_cfg.get('belief_fleet_weight_floor', 0.0), 0.0, 0.95))
        self.belief_fleet_reward_scale = max(float(
            coupling_cfg.get('belief_fleet_reward_scale', 1.0)), 0.0)

        self.holding_feedback = HoldingFeedback(
            window_size=coupling_cfg.get('feedback_window', 10))
        self.lower_lifecycle = LowerEpisodeLifecycle(
            boundary_mode=self.lower_trip_boundary_mode,
            feedback_mode=coupling_cfg.get(
                'holding_feedback_finalize_mode', 'episode_end'),
        )
        self.measurement_proj = MeasurementProjection(
            N_fleet=upper_cfg['N_fleet'],
            lr=coupling_cfg.get('measurement_lr', 0.01))
        self.alpha_holding = coupling_cfg.get('alpha_holding', 0.5)
        self.upper_warmup = coupling_cfg.get('upper_warmup_eps', 30)
        if (self.snapshot_value_selector_enable
                and not self.snapshot_value_selector_start_configured):
            self.snapshot_value_selector_start_ep = int(self.upper_warmup)
        if (self.snapshot_action_value_selector_enable
                and not self.snapshot_action_value_selector_start_configured):
            self.snapshot_action_value_selector_start_ep = int(self.upper_warmup)
        if (self.upper_residual_selector_enable
                and not self.upper_residual_selector_start_configured):
            self.upper_residual_selector_start_ep = int(self.upper_warmup)
        if (self.upper_residual_selector_enable
                and not self.upper_residual_selector_learn_start_configured):
            self.upper_residual_selector_learn_start_ep = max(
                0, int(self.upper_residual_selector_start_ep) - 20)
        if (self.timetable_terminal_value_selector_enable
                and not self.timetable_terminal_value_selector_start_configured):
            self.timetable_terminal_value_selector_start_ep = int(self.upper_warmup)
        if (self.timetable_terminal_value_selector_enable
                and not self.timetable_terminal_value_selector_learn_start_configured):
            self.timetable_terminal_value_selector_learn_start_ep = max(
                0, int(self.timetable_terminal_value_selector_start_ep) - 20)
        if (self.timetable_headway_value_planner_enable
                and not self.timetable_headway_value_planner_start_configured):
            self.timetable_headway_value_planner_start_ep = int(self.upper_warmup)
        if (self.timetable_headway_value_planner_enable
                and not self.timetable_headway_value_planner_learn_start_configured):
            self.timetable_headway_value_planner_learn_start_ep = max(
                0, int(self.timetable_headway_value_planner_start_ep) - 30)
        selector_cfg = config.get('fixed_expert_selector', {}) or {}
        self.fixed_selector_enable = bool(selector_cfg.get('enable', False))
        self.fixed_selector_start_ep = int(
            selector_cfg.get('start_ep', self.upper_warmup))
        self.fixed_selector_min_observations = max(
            1, int(selector_cfg.get('min_observations', 2)))
        self.fixed_selector_probe_period = int(
            selector_cfg.get('probe_period', 10))
        self.fixed_selector_epsilon = float(np.clip(
            selector_cfg.get('epsilon', 0.0), 0.0, 1.0))
        self.fixed_selector_ema_alpha = float(np.clip(
            selector_cfg.get('ema_alpha', 0.25), 0.0, 1.0))
        self.fixed_selector_margin = float(selector_cfg.get('margin', 0.0))
        self.fixed_selector_probe_mode = str(
            selector_cfg.get('probe_mode', 'fixed')).lower()
        self.fixed_selector_count_start = str(
            selector_cfg.get('count_start', 'upper_warmup')).lower()
        strict_headway = selector_cfg.get('strict_headway_s', None)
        self.fixed_selector_strict_headway_s = (
            None if strict_headway is None else float(strict_headway))
        self.fixed_selector_reset_env_rng = bool(
            selector_cfg.get('reset_env_rng', False))
        self.fixed_selector_cost_ema = {'learned': None, 'fixed': None}
        self.fixed_selector_counts = {'learned': 0, 'fixed': 0}
        self._fixed_expert_active = False
        selector_rule_cfg = selector_cfg.get('deterministic_rule', {}) or {}
        self.fixed_selector_rule_enable = bool(
            selector_rule_cfg.get('enable', False))
        self.fixed_selector_rule_default = str(
            selector_rule_cfg.get('default', 'learned')).strip().lower()

        def _rule_groups(name):
            value = selector_rule_cfg.get(name, [])
            if isinstance(value, dict):
                return [dict(value)]
            if isinstance(value, (list, tuple)):
                return [
                    dict(item) for item in value
                    if isinstance(item, dict)
                ]
            return []

        self.fixed_selector_rule_learned_when = _rule_groups('learned_when')
        self.fixed_selector_rule_fixed_when = _rule_groups('fixed_when')
        selector_context_cfg = selector_cfg.get('contextual_value', {}) or {}
        self.fixed_selector_context_enable = bool(
            selector_context_cfg.get('enable', False))
        self.fixed_selector_context_ridge = max(float(
            selector_context_cfg.get('ridge', 0.25)), 1e-6)
        self.fixed_selector_context_feature_clip = max(float(
            selector_context_cfg.get('feature_clip', 5.0)), 0.0)
        self.fixed_selector_context_use_previous_performance = bool(
            selector_context_cfg.get('use_previous_performance', True))
        self.fixed_selector_context_features = [
            'bias',
            'cfg_demand_noise',
            'cfg_od_noise',
            'cfg_od_clip_width',
            'cfg_peak_shift_abs',
            'prev_freq_low_demand',
            'prev_freq_low_forecast',
            'prev_freq_high_energy',
            'prev_freq_middle_energy',
            'prev_freq_od_entropy',
            'prev_freq_od_high_energy',
            'prev_freq_promotion_strength',
            'prev_freq_promotion_absorbed',
            'prev_upper_hf_power_ratio',
            'prev_lower_lf_drift_ratio',
        ]
        if self.fixed_selector_context_use_previous_performance:
            self.fixed_selector_context_features.extend([
                'prev_wait_norm',
                'prev_overshoot_norm',
                'prev_headway_cv',
                'prev_terminal_shift_norm',
                'prev_lower_drift_cost',
            ])
        context_dim = len(self.fixed_selector_context_features)
        self.fixed_selector_context_A = {
            key: self.fixed_selector_context_ridge
            * np.eye(context_dim, dtype=np.float64)
            for key in ('learned', 'fixed')
        }
        self.fixed_selector_context_b = {
            key: np.zeros(context_dim, dtype=np.float64)
            for key in ('learned', 'fixed')
        }
        self._fixed_selector_prev_diag = None
        self._fixed_selector_current_context = None
        self._fixed_selector_context_learned_value = 0.0
        self._fixed_selector_context_fixed_value = 0.0
        self._fixed_selector_context_margin = 0.0

        # FreqDuet leakage regularization:
        # upper should not emit high-frequency timetable shifts, and lower
        # station holding should not accumulate into a low-frequency timetable
        # drift. These are disabled unless configured.
        leak_cfg = config.get('leakage', {})
        self.leakage_enable = bool(leak_cfg.get('enable', False))
        self.lower_drift_window = int(leak_cfg.get('lower_drift_window', 24))
        self.lower_drift_budget_s = float(leak_cfg.get('lower_drift_budget_s', 180.0))
        self.lower_drift_penalty = float(leak_cfg.get('lower_drift_penalty', 0.0))
        self.lower_drift_cost_weight = float(
            leak_cfg.get('lower_drift_cost_weight', 0.0))
        self.lower_drift_cost_cap = float(
            leak_cfg.get('lower_drift_cost_cap', 1.0))
        self.lower_drift_cost_mode = str(
            leak_cfg.get('lower_drift_cost_mode', 'excess')).lower()
        drift_signal_mode = str(leak_cfg.get(
            'lower_drift_signal_mode', 'rolling_action_window')).lower()
        drift_signal_aliases = {
            'legacy': 'rolling_action_window',
            'rolling': 'rolling_action_window',
            'direction_window': 'rolling_action_window',
            'physical_trip': 'trip_cumulative',
            'trip_total': 'trip_cumulative',
        }
        self.lower_drift_signal_mode = drift_signal_aliases.get(
            drift_signal_mode, drift_signal_mode)
        if self.lower_drift_signal_mode not in {
                'rolling_action_window', 'trip_cumulative'}:
            raise ValueError(
                'leakage.lower_drift_signal_mode must be one of '
                "['rolling_action_window', 'trip_cumulative']")
        drift_cost_adapt_cfg = (
            leak_cfg.get('lower_drift_cost_adaptive', {}) or {})
        self.lower_drift_cost_adaptive_enable = bool(
            drift_cost_adapt_cfg.get('enable', False))
        self.lower_drift_cost_adaptive_extra_weight = max(float(
            drift_cost_adapt_cfg.get('extra_weight', 0.0)), 0.0)
        self.lower_drift_cost_adaptive_gate = self._parse_fleet_noharm_gate(
            drift_cost_adapt_cfg.get('gate', {}))
        self.upper_hf_penalty = float(leak_cfg.get('upper_hf_penalty', 0.0))
        self.upper_lpf_window = int(leak_cfg.get('upper_lpf_window', 6))
        self._lower_drift_by_dir = {
            True: deque(maxlen=max(1, self.lower_drift_window)),
            False: deque(maxlen=max(1, self.lower_drift_window)),
        }
        self._ep_lower_drift_penalties = []
        self._ep_lower_drift_costs = []
        self._ep_lower_drift_loads = []
        self._ep_lower_drift_cost_adaptive_gate = []
        self._ep_upper_hf_penalties = []
        self._ep_upper_residual_value_costs = []
        self._ep_upper_residual_value_cost_active = []
        self._ep_upper_residual_selector_active = []
        self._ep_upper_residual_selector_adjusts = []
        self._ep_upper_residual_selector_margins = []
        self._ep_upper_residual_selector_actor_preds = []
        self._ep_upper_residual_selector_selected_preds = []
        self._ep_upper_residual_selector_feature_norms = []
        self._ep_headway_value_planner_active = []
        self._ep_headway_value_planner_adjusts = []
        self._ep_headway_value_planner_deltas = []
        self._ep_headway_value_planner_margins = []
        self._ep_headway_value_planner_actor_preds = []
        self._ep_headway_value_planner_selected_preds = []
        self._ep_headway_value_planner_priors = []
        self._ep_headway_value_planner_target_costs = []
        self._ep_headway_value_planner_feature_norms = []
        self._ep_upper_plan_penalties = []
        self._ep_upper_plan_targets = []
        self._ep_upper_plan_raw_delta_means = []
        self._ep_upper_plan_projected_delta_means = []
        self._ep_upper_plan_projected_delta_sums = []
        self._ep_upper_plan_decisions = 0
        self._ep_upper_plan_reuses = 0
        self._ep_terminal_launch_shifts = []
        self._ep_terminal_shift_caps = []
        self._ep_terminal_shift_mins = []
        self._ep_terminal_feedback_biases = []
        self._ep_terminal_value_selector_active = []
        self._ep_terminal_value_selector_biases = []
        self._ep_terminal_value_selector_margins = []
        self._ep_terminal_value_selector_actor_preds = []
        self._ep_terminal_value_selector_selected_preds = []
        self._ep_terminal_value_selector_feature_norms = []
        self._ep_terminal_value_selector_target_costs = []
        self._ep_cf_action_selector_active = []
        self._ep_cf_action_selector_changed = []
        self._ep_cf_action_selector_terminal_dispatch = []
        self._ep_cf_action_selector_deltas = []
        self._ep_cf_action_selector_confidences = []
        self._ep_terminal_headway_floors = []
        self._ep_fleet_noharm_upper_pressures = []
        self._ep_fleet_noharm_upper_adjusts = []
        self._ep_fleet_noharm_upper_gate_active = []
        self._ep_fleet_noharm_lower_pressures = []
        self._ep_fleet_noharm_lower_adjusts = []
        self._ep_fleet_noharm_lower_gate_active = []
        self._ep_fleet_noharm_lower_proactive_adjusts = []
        self._ep_fleet_noharm_lower_proactive_gate_active = []
        self._ep_fleet_noharm_lower_value_guard_adjusts = []
        self._ep_fleet_noharm_lower_value_guard_active = []
        self._ep_fleet_noharm_lower_value_guard_values = []
        self._ep_fleet_noharm_lower_value_guard_headway_values = []
        self._ep_fleet_noharm_lower_value_guard_costs = []
        self._ep_fleet_noharm_lower_value_soft_costs = []
        self._ep_fleet_noharm_lower_value_soft_active = []
        self._ep_fleet_noharm_lower_value_soft_values = []
        self._ep_fleet_noharm_lower_value_soft_headway_values = []
        self._ep_fleet_noharm_lower_value_soft_risks = []
        self._ep_fleet_noharm_lower_value_soft_violations = []
        self._active_timetable_plans = {}
        self._last_promotion_replan_launch = {}
        self._ep_lower_actions_by_dir = {True: [], False: []}
        self._ep_upper_deltas_by_dir = {True: [], False: []}
        self._ep_upper_demand_action = []
        self._ep_lower_demand_action = []
        self._freq_holdfb_events = {
            True: deque(maxlen=self.freq_holdfb_window),
            False: deque(maxlen=self.freq_holdfb_window),
        }
        self._ep_freq_holdfb_features = []
        self._ep_freq_driftfb_features = []
        diag_cfg = config.get('diagnostics', {})
        self.freq_diag_mi_bins = int(diag_cfg.get('mi_bins', 8))
        self.freq_diag_shock_threshold = float(
            diag_cfg.get('shock_threshold', 0.10))
        self.freq_diag_shock_action_threshold_s = float(
            diag_cfg.get('shock_action_threshold_s', 10.0))
        self.freq_diag_shock_response_window_s = float(
            diag_cfg.get('shock_response_window_s', 900.0))
        self.freq_diag_shock_same_station = bool(
            diag_cfg.get('shock_same_station', False))
        self._ep_shock_response_events = []

        # Frequency-attributed passenger wait reward. This uses actual boarded
        # passenger waiting time and assigns its low-frequency share to upper
        # planning credit while the high-frequency share shapes lower holding.
        attr_cfg = config.get('reward_attribution', {})
        self.freq_wait_enable = bool(attr_cfg.get('enable', False))
        self.freq_wait_assignment_mode = str(
            attr_cfg.get('assignment_mode', 'snapshot_legacy')
        ).strip().lower()
        if self.freq_wait_assignment_mode not in {
                'snapshot_legacy', 'frozen_passenger'}:
            raise ValueError(
                "reward_attribution.assignment_mode must be "
                "'snapshot_legacy' or 'frozen_passenger'")
        self.freq_wait_upper_weight = float(
            attr_cfg.get('upper_wait_weight', 0.0))
        self.freq_wait_lower_weight = float(
            attr_cfg.get('lower_wait_weight', 0.0))
        self.freq_wait_lower_board_credit_weight = float(
            attr_cfg.get('lower_board_credit_weight', 0.0))
        self.freq_wait_lower_board_credit_adaptive = bool(
            attr_cfg.get('lower_board_credit_adaptive', False))
        self.freq_wait_lower_board_credit_absorbed_min = float(
            attr_cfg.get('lower_board_credit_absorbed_min', 0.0))
        self.freq_wait_lower_board_credit_absorbed_width = max(
            float(attr_cfg.get('lower_board_credit_absorbed_width', 0.05)),
            1e-6)
        self.freq_wait_lower_board_credit_min_gate = float(np.clip(
            attr_cfg.get('lower_board_credit_min_gate', 0.0), 0.0, 1.0))
        self.freq_wait_lower_board_norm = max(
            float(attr_cfg.get('lower_board_norm', 10.0)), 1e-6)
        self.freq_wait_lower_board_clip = max(
            float(attr_cfg.get('lower_board_clip', 2.0)), 0.0)
        self.freq_wait_lower_positive_high_only = bool(
            attr_cfg.get('lower_positive_high_only', False))
        self.freq_wait_lower_share_source = str(
            attr_cfg.get('lower_share_source', 'global')).lower()
        self.freq_wait_lower_high_source = str(
            attr_cfg.get('lower_high_source', 'feature')).lower()
        self.freq_wait_lower_hold_high_source = str(
            attr_cfg.get(
                'lower_hold_high_source',
                attr_cfg.get('lower_high_source', 'feature'))).lower()
        self.freq_wait_lower_high_share_cap = float(
            attr_cfg.get('lower_high_share_cap', 1.0))
        self.freq_wait_lower_hold_penalty_weight = float(
            attr_cfg.get('lower_hold_penalty_weight', 0.0))
        self.freq_wait_lower_hold_norm_s = max(
            float(attr_cfg.get('lower_hold_norm_s', 45.0)), 1e-6)
        self.freq_wait_lower_hold_clip = max(
            float(attr_cfg.get('lower_hold_clip', 2.0)), 0.0)
        self.freq_wait_lower_hold_positive_only = bool(
            attr_cfg.get('lower_hold_positive_high_only', True))
        self.freq_wait_lower_raw_gate_middle_energy_max = float(
            attr_cfg.get('lower_raw_gate_middle_energy_max', 0.04))
        self.freq_wait_lower_raw_gate_middle_energy_min = float(
            attr_cfg.get('lower_raw_gate_middle_energy_min', 0.0))
        self.freq_wait_lower_raw_gate_middle_value_max = attr_cfg.get(
            'lower_raw_gate_middle_value_max', None)
        if self.freq_wait_lower_raw_gate_middle_value_max is not None:
            self.freq_wait_lower_raw_gate_middle_value_max = float(
                self.freq_wait_lower_raw_gate_middle_value_max)
        self.freq_wait_lower_raw_gate_middle_value_width = max(
            float(attr_cfg.get('lower_raw_gate_middle_value_width',
                               attr_cfg.get('lower_raw_gate_width', 0.01))),
            1e-6)
        self.freq_wait_lower_raw_gate_width = max(
            float(attr_cfg.get('lower_raw_gate_width', 0.01)), 1e-6)
        self.freq_wait_lower_raw_gate_high_energy_min = float(
            attr_cfg.get('lower_raw_gate_high_energy_min', 0.0))
        self.freq_wait_lower_raw_gate_high_energy_width = max(
            float(attr_cfg.get(
                'lower_raw_gate_high_energy_width',
                self.freq_wait_lower_raw_gate_width)), 1e-6)
        self.freq_wait_lower_raw_gate_high_energy_max = attr_cfg.get(
            'lower_raw_gate_high_energy_max', None)
        if self.freq_wait_lower_raw_gate_high_energy_max is not None:
            self.freq_wait_lower_raw_gate_high_energy_max = float(
                self.freq_wait_lower_raw_gate_high_energy_max)
        self.freq_wait_lower_raw_gate_absorbed_min = float(
            attr_cfg.get('lower_raw_gate_absorbed_min', 0.0))
        self.freq_wait_lower_raw_gate_absorbed_width = max(
            float(attr_cfg.get(
                'lower_raw_gate_absorbed_width',
                self.freq_wait_lower_raw_gate_width)), 1e-6)
        self.freq_wait_lower_raw_gate_min_weight = float(np.clip(
            attr_cfg.get('lower_raw_gate_min_weight', 0.0), 0.0, 1.0))
        self.freq_wait_norm_s = max(
            float(attr_cfg.get('wait_norm_s', 600.0)), 1e-6)
        self.freq_wait_clip = max(float(attr_cfg.get('wait_clip', 2.0)), 0.0)
        self.freq_wait_low_floor = max(
            float(attr_cfg.get('low_share_floor', 0.05)), 0.0)
        self.freq_wait_normalize_upper = bool(
            attr_cfg.get('normalize_upper_credit', True))
        self._ep_lower_wait_penalties = []
        self._ep_lower_board_credits = []
        self._ep_lower_board_credit_gates = []
        self._ep_lower_high_hold_penalties = []
        self._ep_lower_wait_net = []
        self._ep_upper_wait_credits = []
        self._ep_freq_wait_low_shares = []
        self._ep_freq_wait_lower_high_shares = []
        self._ep_freq_wait_lower_raw_credit_weights = []
        self._ep_freq_wait_boarded_pax = 0
        self._ep_trip_wait_stats = defaultdict(lambda: {
            'pax': 0,
            'wait_s': 0.0,
            'upper_wait_norm_sum': 0.0,
            'low_share_sum': 0.0,
            'events': 0,
        })

        # ─── v3 cross-level coupling mode ───
        # 'channels' (default v2 behaviour: HoldFB + hindsight credit, action = launch shift)
        # 'haar'     (HAAR + PIPER: inject β·clip(A_U,-c,c)·f_k into lower reward via tap_signal)
        # 'hiro'     (HIRO/SHIRO style: δ_t reinterpreted as target-headway shift, lower's
        #             Lagrangian cost becomes goal-conditioned; no upper advantage flow)
        self.coupling_mode = coupling_cfg.get('coupling_mode', 'channels')
        haar_cfg = coupling_cfg.get('haar', {})
        self.haar_beta = float(haar_cfg.get('beta', 0.5))
        self.haar_clip = float(haar_cfg.get('clip', 0.5))
        self.haar_use_reach_gate = bool(haar_cfg.get('use_reach_gate', True))
        self.haar_reach_lr = float(haar_cfg.get('reach_lr', 1e-3))
        self.haar_reach_threshold = float(haar_cfg.get('reach_threshold', 0.5))
        self.reach_net = None        # lazy-init (depends on state_dim)
        self.reach_optimizer = None
        # buffer for reach training: (s_upper, delta, hold_summary, label)
        self._reach_buffer = []

        # ─── TPC-Lower (Target-Policy-Corrected lower SAC) ───
        # Mitigates the "noisy upper contaminates lower" failure mode that loses
        # to the Fixed baseline. Enabled via coupling.tpc_enable in config.
        tpc = coupling_cfg.get('tpc', {})
        self.tpc_enable = bool(tpc.get('enable', False))
        if self.tpc_enable and self.upper_action_candidates is not None:
            raise ValueError(
                "TPC Gaussian behavior mixing is incompatible with a categorical "
                "upper action library")
        self.tpc_eps = float(tpc.get('eps_explore', 0.25))
        self.tpc_sigma_tgt = float(tpc.get('sigma_tgt', 20.0))
        self.tpc_target_distribution = str(tpc.get(
            'target_distribution',
            'legacy_clipped_physical_gaussian')).strip().lower()
        if self.tpc_target_distribution not in {
                'legacy_clipped_physical_gaussian',
                'bounded_logistic_normal_v4'}:
            raise ValueError(
                'coupling.tpc.target_distribution must be '
                'legacy_clipped_physical_gaussian or '
                'bounded_logistic_normal_v4')
        self.tpc_latent_sigma = max(float(
            tpc.get('latent_sigma', 0.25)), 1e-6)
        self.tpc_w_max = float(tpc.get('w_max', 5.0))
        self.tpc_ema_tau = float(tpc.get('ema_tau', 0.005))
        self.tpc_warmstart_lower_from = tpc.get('warmstart_lower_from', None)
        self.target_upper_trainer = None  # initialised at end of upper_warmup
        # global_tid -> {z, delta, log_mu} for IS weight lookup
        self.dispatch_meta = {}
        # bound the metadata dict size (replay capacity / trips_per_episode + buffer)
        self._dispatch_meta_max = int(tpc.get('meta_max_size', 200_000))

        # CS-BAPR belief tracker: detect non-stationarity from upper changes
        self.surprise_computer = SurpriseComputer(
            ema_alpha=coupling_cfg.get('surprise_ema', 0.3))
        self.belief_tracker = BeliefTracker(
            max_run_length=coupling_cfg.get('belief_max_H', 20),
            hazard_rate=coupling_cfg.get('belief_hazard', 0.05))
        self.belief_alpha_boost_max = coupling_cfg.get('belief_alpha_boost', 2.0)

        # Training params
        self.batch_size = lower_cfg.get('batch_size', 512)
        self.updates_per_episode = lower_cfg.get('updates_per_episode', 30)
        self.upper_batch_size = upper_cfg.get('batch_size', 64)
        self.upper_updates = upper_cfg.get('updates_per_episode', 10)

        # Episode bookkeeping
        self._episode_upper_transitions = []
        self._prev_upper_states = {}
        self._ep_lower_actions = []     # all lower actions this episode
        self._ep_lower_context_gate_values = []
        self._ep_lower_action_bins_gate_values = []
        self._ep_lower_rewards = []     # all lower rewards this episode
        self._ep_lower_causal_guard_active = []
        self._ep_lower_causal_guard_limits = []
        self._ep_lower_causal_guard_adjustments = []
        self._ep_upper_deltas = []      # all δ_t this episode
        self._ep_trip_records = []      # per-trip detail for step-level diag
        self._ep_dispatch_times = {'up': [], 'down': []}  # actual launch times per dir
        self._ep_upper_rewards = []     # all upper rewards this episode
        self._ep_upper_system_rewards = []
        self._ep_upper_gap_credits = []
        self._ep_upper_reliability_rewards = []
        self._ep_upper_interval_rewards = []
        self._ep_upper_interval_wait_costs = []
        self._ep_upper_interval_onboard_costs = []
        self._ep_upper_interval_dispatch_backlog_costs = []
        self._ep_upper_interval_headway_costs = []
        self._ep_upper_interval_fleet_costs = []
        self._ep_upper_interval_coverages = []
        self._current_ep = 0

        # Logging
        seed = config.get('seed', 42)
        log_base = config.get('logging', {}).get('logs_dir', 'logs')
        if not os.path.isabs(log_base):
            log_base = os.path.join(str(SCRIPT_DIR), log_base)
        self.log_dir = os.path.join(log_base, f'{self.exp_name}_seed{seed}')
        os.makedirs(self.log_dir, exist_ok=True)
        self.env.configure_frequency_logging(self.log_dir)
        self.history = defaultdict(list)
        self.resume_from_ep = 0  # set by maybe_resume() before train()
        self._deployment_state_loaded = False
        self.diag = None  # created after resume decision in train()

    # ────────────────── Upper callback ──────────────────

    @staticmethod
    def compute_system_reward(z, N_fleet=12):
        """
        Default scalar system reward (fallback, unused when belief-weighted).
        """
        wait_penalty = -z[0] / 10.0
        fleet_penalty = -max(0, z[1] - N_fleet) ** 2 / N_fleet
        cv_penalty = -z[2]
        return wait_penalty + fleet_penalty + cv_penalty

    def compute_belief_weighted_reward(self, z, N_fleet=12):
        """
        BAMOR-style multi-objective scalarization with belief-aware weights.

        Three objective penalties (all negative, higher=better):
          p_wait  = -wait / 10
          p_fleet = -(fleet - N_fleet)² / N_fleet  (only counts overshoot)
          p_cv    = -cv

        Weighting policy:
          Base weights w_base from θ-OGD (long-term adaptation over episodes)
          Belief modulation (short-term shift detection):
            - cp_prob high → changepoint detected → boost fleet weight (safety)
            - window long → stable → shift weight to wait/cv (quality)

        Returns: scalar reward + weight dict for logging
        """
        wait_p = -z[0] / 10.0
        fleet_p = (
            -max(0, z[1] - N_fleet) ** 2 / N_fleet
            * self.belief_fleet_reward_scale)
        cv_p = -z[2]

        # Ablation: fixed equal weights instead of belief-driven
        if self.ablate_morl:
            fixed_w = np.array([0.5, 0.25, 0.25])
            r = fixed_w[0] * wait_p + fixed_w[1] * fleet_p + fixed_w[2] * cv_p
            return float(r * 3.0), fixed_w

        # Base weights from θ-OGD (long-term adaptation, already in [0,1] sum=1)
        base_w = self.measurement_proj.get_reward_weights()  # [w_wait, w_fleet, w_cv]

        # Belief modulation
        cp_prob = self.belief_tracker.changepoint_prob
        window = self.belief_tracker.effective_window

        # Crisis modulation: if changepoint detected, boost fleet safety
        if cp_prob > self.belief_crisis_cp_threshold:
            # Shift mass toward fleet safety during detected non-stationarity.
            crisis_strength = min(
                1.0,
                ((cp_prob - self.belief_crisis_cp_threshold)
                 / self.belief_crisis_cp_width))
            boost = self.belief_crisis_fleet_boost_max * crisis_strength
            adj_w = base_w.copy()
            # Take mass from wait+cv, add to fleet
            adj_w[0] *= (1 - boost)
            adj_w[2] *= (1 - boost)
            adj_w[1] += boost * (base_w[0] + base_w[2])
        # Stable modulation: if very stable, shift toward quality
        elif (self.belief_stable_quality_shift_max > 0.0
              and window > self.belief_stable_window_threshold):
            stability = min(
                1.0,
                ((window - self.belief_stable_window_threshold)
                 / self.belief_stable_window_width))
            shift = self.belief_stable_quality_shift_max * stability
            adj_w = base_w.copy()
            # Take from fleet (already safe), add to quality
            adj_w[1] *= (1 - shift)
            adj_w[0] += shift * base_w[1] * 0.6
            adj_w[2] += shift * base_w[1] * 0.4
        else:
            adj_w = base_w

        # Normalize
        adj_w = adj_w / max(adj_w.sum(), 1e-6)
        if (self.belief_fleet_weight_floor > 0.0
                and adj_w[1] < self.belief_fleet_weight_floor):
            floor = self.belief_fleet_weight_floor
            other_sum = adj_w[0] + adj_w[2]
            if other_sum > 1e-8:
                scale = (1.0 - floor) / other_sum
                adj_w[0] *= scale
                adj_w[2] *= scale
            else:
                adj_w[0] = (1.0 - floor) * 0.6
                adj_w[2] = (1.0 - floor) * 0.4
            adj_w[1] = floor

        # Scalarize with M=3 dimensions
        r = adj_w[0] * wait_p + adj_w[1] * fleet_p + adj_w[2] * cv_p
        # Rescale to match old magnitude (old reward range ≈ [-2, 0])
        r = r * 3.0

        return float(r), adj_w

    def _build_tpc_weight_fn(self):
        """Return a closure that maps batch trip_ids → per-sample IS weights.

        Weight per sample = clip( π_target(δ|z) / μ_behavior(δ|z), 0, w_max ),
        normalised so the batch mean ≈ 1. Samples whose trip_id has no metadata
        in self.dispatch_meta (e.g., evicted by size cap or pre-Phase-1) get 1.
        """
        if self.target_upper_trainer is None or not self.dispatch_meta:
            return None
        target_pi = self.target_upper_trainer.policy_net
        meta = self.dispatch_meta
        w_max = self.tpc_w_max

        def fn(trip_ids):
            # Vectorise where possible: gather z and δ for samples with metadata.
            n = len(trip_ids)
            w = np.ones(n, dtype=np.float32)
            zs, ds, log_mus, idx = [], [], [], []
            for i, tid in enumerate(trip_ids):
                m = meta.get(int(tid))
                if m is not None:
                    zs.append(m['z']); ds.append(m['delta']); log_mus.append(m['log_mu'])
                    idx.append(i)
            if zs:
                zs = np.stack(zs).astype(np.float32)
                ds = np.stack(ds).astype(np.float32)
                log_mus = np.array(log_mus, dtype=np.float32)
                # Batched log_prob under EMA target upper policy
                log_p_target = target_pi.log_prob(
                    zs, ds, coordinates='normalized_unit_interval')
                if np.isscalar(log_p_target):
                    log_p_target = np.array([float(log_p_target)])
                log_w = log_p_target - log_mus
                w_corr = np.clip(np.exp(log_w), 0.0, w_max)
                for j, i in enumerate(idx):
                    w[i] = float(w_corr[j])
            # Normalise to mean ≈ 1
            mean_w = w.mean()
            if mean_w > 1e-6:
                w = w / mean_w
            return w
        return fn

    def _lower_drift_load(
            self, direction, action_s, trip_id, action_already_recorded=False):
        """Return a drift load with explicit legacy or physical-trip semantics."""
        direction = bool(direction)
        action_s = max(float(action_s), 0.0)
        if self.lower_drift_signal_mode == 'trip_cumulative':
            trip_id = int(trip_id)
            cumulative = (
                self.holding_feedback.get_trip_total(trip_id)
                if trip_id >= 0 else 0.0)
            if not action_already_recorded:
                cumulative += action_s
            return float(cumulative)

        hist = self._lower_drift_by_dir[direction]
        hist.append(action_s)
        return float(sum(hist))

    def _lower_drift_penalty(self, drift_load_s):
        """Penalty for lower holding that accumulates as timetable drift."""
        if not self.leakage_enable or self.lower_drift_penalty <= 0:
            return 0.0
        excess = max(0.0, float(drift_load_s) - self.lower_drift_budget_s)
        penalty = self.lower_drift_penalty * (
            excess / max(self.lower_drift_budget_s, 1e-6))
        return float(penalty)

    def _lower_drift_cost(self, drift_load_s):
        """Optional Lagrangian cost for lower holding that becomes LF drift."""
        if (not self.leakage_enable
                or self.lower_drift_cost_weight <= 0.0):
            return 0.0
        adaptive_active = False
        weight = self.lower_drift_cost_weight
        if (self.lower_drift_cost_adaptive_enable
                and self.lower_drift_cost_adaptive_extra_weight > 0.0):
            adaptive_active = self._fleet_noharm_gate_active(
                self.lower_drift_cost_adaptive_gate)
            if adaptive_active:
                weight += self.lower_drift_cost_adaptive_extra_weight
        self._ep_lower_drift_cost_adaptive_gate.append(
            1.0 if adaptive_active else 0.0)
        rolling_hold = max(float(drift_load_s), 0.0)
        budget = max(self.lower_drift_budget_s, 1e-6)
        if self.lower_drift_cost_mode in {'total', 'rolling', 'hold'}:
            signal = rolling_hold / budget
        else:
            signal = max(0.0, rolling_hold - self.lower_drift_budget_s) / budget
        cost = weight * max(signal, 0.0)
        if self.lower_drift_cost_cap >= 0.0:
            cost = min(cost, self.lower_drift_cost_cap)
        return float(cost)

    def _drift_feedback_pair(self, direction):
        if self.lower_drift_signal_mode == 'trip_cumulative':
            stats = self.holding_feedback.get_direction_total_stats(
                bool(direction), budget_s=self.lower_drift_budget_s)
            rolling_hold = float(stats['rolling_mean'])
            excess_s = float(stats['mean_excess'])
        else:
            hist = self._lower_drift_by_dir[bool(direction)]
            rolling_hold = float(sum(hist))
            excess_s = max(0.0, rolling_hold - self.lower_drift_budget_s)
        drift = rolling_hold / self.freq_driftfb_norm_s
        excess = excess_s / self.freq_driftfb_norm_s
        if self.freq_driftfb_clip > 0.0:
            drift = min(drift, self.freq_driftfb_clip)
            excess = min(excess, self.freq_driftfb_clip)
        return float(drift), float(excess)

    def _quantize_lower_action(self, action):
        """Project holding to bins, optionally only under a causal context gate."""
        value = self._lower_action_scalar(action)
        context_gate = float(np.clip(
            getattr(self.env, 'lower_context_gate_value', 1.0), 0.0, 1.0))
        bins_gate = self._lower_action_bins_gate_value(context_gate)
        if hasattr(self, '_ep_lower_context_gate_values'):
            self._ep_lower_context_gate_values.append(context_gate)
        if hasattr(self, '_ep_lower_action_bins_gate_values'):
            self._ep_lower_action_bins_gate_values.append(bins_gate)
        if self.lower_action_bins is not None and bins_gate >= 0.5:
            idx = int(np.argmin(np.abs(self.lower_action_bins - value)))
            value = float(self.lower_action_bins[idx])
        return np.asarray([value], dtype=np.float32)

    def _lower_action_bins_gate_value(self, context_gate):
        if not self.lower_action_bins_gate_enabled:
            return 1.0
        if self.lower_action_bins_gate_source in {
                'lower_context_gate', 'context_gate'}:
            gate_value = float(context_gate)
        else:
            gate_value = 0.0
        return 1.0 if gate_value >= self.lower_action_bins_gate_threshold else 0.0

    @staticmethod
    def _lower_action_scalar(action):
        return float(np.asarray(action, dtype=np.float32).reshape(-1)[0])

    def _fleet_pressure(self):
        n_fleet = max(1, int(getattr(
            self, '_current_N_fleet', getattr(self, 'N_fleet_default', 12))))
        concurrent = sum(1 for bus in getattr(self.env, 'bus_all', [])
                         if getattr(bus, 'on_route', False))
        return float(concurrent), float(n_fleet), float(concurrent - n_fleet)

    def _inject_lifecycle_holding_state(self, state, direction):
        state = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        if self.upper_holding_state_source != 'trip_lifecycle':
            return state
        same = self.holding_feedback.get_direction_stats(bool(direction))
        other = self.holding_feedback.get_direction_stats(not bool(direction))
        state[5:8] = np.asarray([
            float(same['rolling_mean']) / 60.0,
            float(same['rolling_std']) / 60.0,
            float(other['rolling_mean']) / 60.0,
        ], dtype=np.float32)
        return state

    def _upper_plan_context(self, active_plan, decision_time_s):
        return self.upper_plan_execution.plan_context(
            active_plan,
            decision_time_s=decision_time_s,
            action_low=self.upper_action_low,
            action_high=self.upper_action_high,
            replan_interval_s=self.timetable_replan_interval_s,
        )

    def _upper_state_history_vector(self):
        if self.upper_state_history_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        zero = np.zeros(self.upper_state_history_step_dim, dtype=np.float32)
        rows = list(getattr(self, '_upper_state_history', []))
        rows = rows[-self.upper_state_history_len:]
        padded = [zero] * max(0, self.upper_state_history_len - len(rows))
        padded.extend(rows)
        return np.concatenate(padded).astype(np.float32)

    def _augment_upper_state_history(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.upper_state_history_dim <= 0:
            return state
        return np.concatenate([
            state,
            self._upper_state_history_vector(),
        ]).astype(np.float32)

    def _upper_state_history_row(
            self, action_vec, direction, delta_t, launch_shift,
            plan_penalty, upper_decision_taken, promotion_replan):
        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        block = np.asarray(
            action[self._upper_residual_selector_slice(action, direction)],
            dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        action_mean = float(block.mean()) if block.size else 0.0
        action_std = float(block.std()) if block.size else 0.0
        action_slope = float(block[-1] - block[0]) if block.size >= 2 else 0.0
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        prev = self._headway_value_planner_prev_metrics()
        concurrent, n_fleet, pressure = self._fleet_pressure()
        n_fleet = max(float(n_fleet), 1.0)
        try:
            waiting_total = sum(
                len(st.waiting_passengers)
                for st in getattr(self.env, 'stations', []))
        except Exception:
            waiting_total = 0
        action_norm = self.upper_state_history_action_norm_s
        shift_norm = self.upper_state_history_shift_norm_s
        return np.asarray([
            action_mean / action_norm,
            action_std / action_norm,
            action_slope / action_norm,
            float(delta_t) / action_norm,
            float(launch_shift) / shift_norm,
            float(plan_penalty) / self.upper_state_history_plan_penalty_norm,
            1.0 if bool(direction) else -1.0,
            1.0 if bool(upper_decision_taken) else 0.0,
            1.0 if bool(promotion_replan) else 0.0,
            float(prev.get('wait', 0.0)) / self.upper_state_history_wait_norm_min,
            float(prev.get('cv', 0.0)),
            float(prev.get('overshoot_norm', 0.0)),
            float(prev.get('terminal_shift', 0.0)) / shift_norm,
            float(prev.get('lower_action', 0.0)) / 5.0,
            float(prev.get('lower_drift', 0.0)),
            float(freq.get('freq_low_forecast', 0.0)),
            10.0 * float(freq.get('freq_high_energy', 0.0)),
            10.0 * float(freq.get('freq_middle_energy', 0.0)),
            float(freq.get('freq_od_entropy', 0.0)),
            float(freq.get('freq_promotion_strength', 0.0)),
            float(concurrent) / n_fleet,
            float(pressure) / n_fleet,
            float(waiting_total) / self.upper_state_history_waiting_norm,
        ], dtype=np.float32)

    def _update_upper_state_history(
            self, action_vec, direction, delta_t, launch_shift, plan_penalty,
            upper_decision_taken, promotion_replan):
        if self.upper_state_history_dim <= 0:
            return
        row = self._upper_state_history_row(
            action_vec, direction, delta_t, launch_shift, plan_penalty,
            upper_decision_taken, promotion_replan)
        if row.size != self.upper_state_history_step_dim:
            return
        self._upper_state_history.append(row)

    def _upper_transition_stream_key(self, planner_key):
        if self.upper_transition_stream_mode == 'legacy_global':
            return '__legacy_global__'
        return planner_key

    def _close_upper_transition_stream(
            self, stream_key, next_state, done, decision_time_s):
        prev = self._prev_upper_states.get(stream_key)
        if prev is None:
            return
        interval_outcome = self.upper_interval_credit.close(
            stream_key, end_time_s=decision_time_s)
        elapsed_s = max(
            0.0, float(decision_time_s) - float(prev['decision_time_s']))
        self._episode_upper_transitions.append({
            's': prev['s'],
            'a': prev['a'],
            'tid': prev['tid'],
            'dir': prev['dir'],
            'ns': np.asarray(next_state, dtype=np.float32).copy(),
            'done': bool(done),
            'duration_steps': self.upper_plan_execution.duration_steps(
                elapsed_s),
            'duration_s': elapsed_s,
            'a_eff': prev['a_eff'],
            'plan_penalty': prev['plan_penalty'],
            'upper_value_cost': prev['upper_value_cost'],
            'upper_value_active': prev['upper_value_active'],
            'upper_residual_selector_x': prev['upper_residual_selector_x'],
            'terminal_value_selector_x': prev['terminal_value_selector_x'],
            'headway_value_planner_x': prev['headway_value_planner_x'],
            'transition_stream_key': stream_key,
            'interval_outcome': interval_outcome,
        })
        if done:
            del self._prev_upper_states[stream_key]

    def _close_previous_upper_transition(
            self, next_state, done, decision_time_s, planner_key):
        stream_key = self._upper_transition_stream_key(planner_key)
        self._close_upper_transition_stream(
            stream_key,
            next_state=next_state,
            done=done,
            decision_time_s=decision_time_s,
        )

    @staticmethod
    def _pressure_strength(pressure, start, full):
        start = float(start)
        full = float(full)
        pressure = float(pressure)
        if pressure <= start:
            return 0.0
        if full <= start:
            return 1.0
        return float(np.clip((pressure - start) / (full - start), 0.0, 1.0))

    @staticmethod
    def _parse_fleet_noharm_gate(cfg):
        cfg = cfg or {}

        def _optional_float(key):
            value = cfg.get(key, None)
            return None if value is None else float(value)

        return {
            'mode': str(cfg.get('mode', 'always')).lower(),
            'default_active': bool(cfg.get('default_active', True)),
            'min_high_energy': _optional_float('min_high_energy'),
            'max_high_energy': _optional_float('max_high_energy'),
            'min_middle_energy': _optional_float('min_middle_energy'),
            'max_middle_energy': _optional_float('max_middle_energy'),
            'min_od_high_energy': _optional_float('min_od_high_energy'),
            'max_od_high_energy': _optional_float('max_od_high_energy'),
            'min_low_forecast': _optional_float('min_low_forecast'),
            'max_low_forecast': _optional_float('max_low_forecast'),
            'min_low_demand': _optional_float('min_low_demand'),
            'max_low_demand': _optional_float('max_low_demand'),
            'min_updates_required': _optional_float('min_updates_required'),
        }

    def _fleet_noharm_gate_active(self, gate_cfg):
        mode = str(gate_cfg.get('mode', 'always')).lower()
        if mode in {'always', 'on', 'true'}:
            return True
        if mode in {'never', 'off', 'false'}:
            return False
        tracker = getattr(self.env, 'frequency_tracker', None)
        if tracker is None:
            return bool(gate_cfg.get('default_active', True))
        summary = tracker.summary()
        min_updates = gate_cfg.get('min_updates_required')
        if min_updates is not None:
            updates = float(summary.get('freq_updates', 0.0))
            if updates < float(min_updates):
                return bool(gate_cfg.get('default_active', True))
        checks = []

        def _add_min(key, summary_key):
            threshold = gate_cfg.get(key)
            if threshold is not None:
                checks.append(float(summary.get(summary_key, 0.0)) >= float(threshold))

        def _add_max(key, summary_key):
            threshold = gate_cfg.get(key)
            if threshold is not None:
                checks.append(float(summary.get(summary_key, 0.0)) <= float(threshold))

        _add_min('min_high_energy', 'freq_high_energy')
        _add_max('max_high_energy', 'freq_high_energy')
        _add_min('min_middle_energy', 'freq_middle_energy')
        _add_max('max_middle_energy', 'freq_middle_energy')
        _add_min('min_od_high_energy', 'freq_od_high_energy')
        _add_max('max_od_high_energy', 'freq_od_high_energy')
        _add_min('min_low_forecast', 'freq_low_forecast')
        _add_max('max_low_forecast', 'freq_low_forecast')
        _add_min('min_low_demand', 'freq_low_demand')
        _add_max('max_low_demand', 'freq_low_demand')
        if not checks:
            return bool(gate_cfg.get('default_active', True))
        if mode in {'any', 'or'}:
            return any(checks)
        if mode in {'all', 'and'}:
            return all(checks)
        if mode in {'inverse_any', 'not_any'}:
            return not any(checks)
        if mode in {'inverse_all', 'not_all'}:
            return not all(checks)
        return bool(gate_cfg.get('default_active', True))

    def _apply_upper_fleet_noharm(self, action_vec):
        if not self.fleet_noharm_upper_enable:
            return np.asarray(action_vec, dtype=np.float32)
        original = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        _, _, pressure = self._fleet_pressure()
        gate_active = self._fleet_noharm_gate_active(
            self.fleet_noharm_upper_gate)
        self._ep_fleet_noharm_upper_gate_active.append(
            1.0 if gate_active else 0.0)
        strength = self._pressure_strength(
            pressure,
            self.fleet_noharm_upper_pressure_start,
            self.fleet_noharm_upper_pressure_full,
        )
        self._ep_fleet_noharm_upper_pressures.append(max(0.0, pressure))
        if (not gate_active) or strength <= 0.0:
            self._ep_fleet_noharm_upper_adjusts.append(0.0)
            return original

        neutral = self.fleet_noharm_upper_neutral_s
        shrink = strength * self.fleet_noharm_upper_shrink_max
        adjusted = original.copy()
        if self.fleet_noharm_upper_mode in {'positive', 'positive_only'}:
            mask = adjusted > neutral
            adjusted[mask] = neutral + (adjusted[mask] - neutral) * (1.0 - shrink)
        elif self.fleet_noharm_upper_mode in {'negative', 'negative_only'}:
            mask = adjusted < neutral
            adjusted[mask] = neutral + (adjusted[mask] - neutral) * (1.0 - shrink)
        else:
            adjusted = neutral + (adjusted - neutral) * (1.0 - shrink)
        adjusted = np.clip(
            adjusted, self.upper_action_low, self.upper_action_high
        ).astype(np.float32)
        self._ep_fleet_noharm_upper_adjusts.append(
            float(np.mean(np.abs(original - adjusted))))
        return adjusted

    def _quantize_upper_action(self, action_vec):
        action = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if self.upper_action_bins is None:
            return action
        quantized = action.copy()
        for i, value in enumerate(action):
            idx = int(np.argmin(np.abs(self.upper_action_bins - float(value))))
            quantized[i] = float(self.upper_action_bins[idx])
        return np.clip(
            quantized, self.upper_action_low, self.upper_action_high
        ).astype(np.float32)

    def _prepare_upper_action(self, action_vec):
        action = self._apply_upper_fleet_noharm(action_vec)
        return self._quantize_upper_action(action)

    def _resolve_local_path(self, path_text):
        path = Path(str(path_text))
        if path.is_absolute():
            return path
        return SCRIPT_DIR / path

    def _load_snapshot_selector_artifact(self, artifact):
        artifact_dir = self._resolve_local_path(artifact)
        model_path = artifact_dir / 'model.joblib'
        forest_path = artifact_dir / 'forest_model.npz'
        meta_path = artifact_dir / 'model_artifact.json'
        if (not meta_path.exists()
                or (not model_path.exists() and not forest_path.exists())):
            raise FileNotFoundError(
                "snapshot value selector artifact missing model.joblib/"
                f"forest_model.npz or model_artifact.json under {artifact_dir}")
        model = None
        forest = None
        if model_path.exists():
            try:
                import joblib
                model = joblib.load(model_path)
            except ModuleNotFoundError:
                model = None
            except Exception as exc:
                if not forest_path.exists():
                    raise RuntimeError(
                        "failed to load snapshot selector joblib artifact"
                    ) from exc
                model = None
        if model is None:
            if not forest_path.exists():
                raise RuntimeError(
                    "snapshot selector needs sklearn for model.joblib or a "
                    "forest_model.npz fallback artifact")
            with np.load(forest_path) as data:
                forest = {key: data[key] for key in data.files}
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        feature_cols = list(meta.get('feature_cols', []))
        feature_medians = dict(meta.get('feature_medians', {}))
        candidate_methods = list(meta.get('candidate_methods', []))
        if not candidate_methods:
            candidate_methods = [
                'term45_m60', 'term45_m30', 'term45_0',
                'term45_p30', 'term45_p60',
            ]
        return model, forest, meta, feature_cols, feature_medians, candidate_methods

    def _load_snapshot_value_selector(self):
        if not self.snapshot_value_selector_artifact:
            self.snapshot_value_selector_enable = False
            return
        loaded = self._load_snapshot_selector_artifact(
            self.snapshot_value_selector_artifact)
        (
            self.snapshot_value_selector_model,
            self.snapshot_value_selector_forest,
            self.snapshot_value_selector_meta,
            self.snapshot_value_selector_feature_cols,
            self.snapshot_value_selector_feature_medians,
            self.snapshot_value_selector_candidate_methods,
        ) = loaded

    def _load_snapshot_action_value_selector(self):
        if not self.snapshot_action_value_selector_artifact:
            self.snapshot_action_value_selector_enable = False
            return
        loaded = self._load_snapshot_selector_artifact(
            self.snapshot_action_value_selector_artifact)
        (
            self.snapshot_action_value_selector_model,
            self.snapshot_action_value_selector_forest,
            self.snapshot_action_value_selector_meta,
            self.snapshot_action_value_selector_feature_cols,
            self.snapshot_action_value_selector_feature_medians,
            self.snapshot_action_value_selector_candidate_methods,
        ) = loaded

    def _snapshot_value_predict(self, x, model=None, forest=None):
        x = np.asarray(x, dtype=np.float64)
        if model is None and forest is None:
            model = self.snapshot_value_selector_model
            forest = self.snapshot_value_selector_forest
        if model is not None:
            return np.asarray(
                model.predict(x),
                dtype=np.float64)
        if forest is None:
            raise RuntimeError("snapshot value selector model is not loaded")
        ptr = forest['tree_ptr']
        left = forest['children_left']
        right = forest['children_right']
        feature = forest['feature']
        threshold = forest['threshold']
        value = forest['value']
        out = np.zeros(x.shape[0], dtype=np.float64)
        n_trees = max(1, len(ptr) - 1)
        for row_idx, row in enumerate(x):
            total = 0.0
            for tree_idx in range(n_trees):
                start = int(ptr[tree_idx])
                node = 0
                while True:
                    global_node = start + node
                    child_left = int(left[global_node])
                    if child_left < 0:
                        total += float(value[global_node])
                        break
                    feat_idx = int(feature[global_node])
                    if float(row[feat_idx]) <= float(threshold[global_node]):
                        node = child_left
                    else:
                        node = int(right[global_node])
            out[row_idx] = total / float(n_trees)
        return out

    def _snapshot_selector_domain(self):
        if self.snapshot_value_selector_domain:
            return self.snapshot_value_selector_domain
        name = self.exp_name
        if 'gen_highnoise' in name:
            return 'highnoise'
        if 'gen_odshift' in name:
            return 'odshift'
        if 'gen_rushshift' in name:
            return 'rushshift'
        if 'terminal' in name:
            return 'terminal'
        return 'terminal'

    def _snapshot_action_selector_domain(self):
        if self.snapshot_action_value_selector_domain:
            return self.snapshot_action_value_selector_domain
        return self._snapshot_selector_domain()

    @staticmethod
    def _snapshot_selector_action_features(mode, delta_s, actor_delta_s=0.0,
                                           offset_s=None):
        delta_s = float(delta_s)
        actor_delta_s = float(actor_delta_s)
        if offset_s is None:
            offset_s = delta_s - actor_delta_s
        offset_s = float(offset_s)
        scale = 60.0
        delta_minus_actor = delta_s - actor_delta_s
        return {
            'action_mode': mode,
            'action_delta_s': delta_s,
            'action_delta_norm': delta_s / scale,
            'action_abs_delta_norm': abs(delta_s) / scale,
            'action_positive': 1.0 if delta_s > 0 else 0.0,
            'action_negative': 1.0 if delta_s < 0 else 0.0,
            'action_zero': 1.0 if abs(delta_s) < 1e-9 else 0.0,
            'action_term45': 1.0 if mode == 'term45' else 0.0,
            'action_target': 1.0 if mode == 'target' else 0.0,
            'action_term45_x_delta': (
                (1.0 if mode == 'term45' else 0.0) * delta_s / scale),
            'action_term45_x_abs_delta': (
                (1.0 if mode == 'term45' else 0.0) * abs(delta_s) / scale),
            'candidate_offset_norm': offset_s / scale,
            'candidate_abs_offset_norm': abs(offset_s) / scale,
            'candidate_above_actor':
                1.0 if delta_minus_actor > 1e-9 else 0.0,
            'candidate_below_actor':
                1.0 if delta_minus_actor < -1e-9 else 0.0,
            'candidate_same_as_actor':
                1.0 if abs(delta_minus_actor) <= 1e-9 else 0.0,
            'action_delta_minus_actor_norm': delta_minus_actor / scale,
            'action_abs_delta_minus_actor_norm': abs(delta_minus_actor) / scale,
            'action_term45_x_offset': (
                (1.0 if mode == 'term45' else 0.0) * offset_s / scale),
            'action_term45_x_abs_offset': (
                (1.0 if mode == 'term45' else 0.0) * abs(offset_s) / scale),
        }

    def _snapshot_selector_parse_action(self, method, actor_delta_s=0.0):
        text = str(method)
        actor_relative = False
        actor_delta_s = float(actor_delta_s)
        if text.startswith('actor_term45_'):
            mode = 'term45'
            token = text[len('actor_term45_'):]
            actor_relative = True
        elif text.startswith('actor_target_'):
            mode = 'target'
            token = text[len('actor_target_'):]
            actor_relative = True
        elif text.startswith('term45_'):
            mode = 'term45'
            token = text[len('term45_'):]
        elif text.startswith('target_'):
            mode = 'target'
            token = text[len('target_'):]
        elif text == 'target0':
            mode = 'target'
            token = '0'
        else:
            raise ValueError(f"unsupported snapshot selector action {method!r}")
        if token == '0':
            raw_delta_s = 0.0
        elif token.startswith('m'):
            raw_delta_s = -float(token[1:])
        elif token.startswith('p'):
            raw_delta_s = float(token[1:])
        else:
            raw_delta_s = float(token)
        delta_s = actor_delta_s + raw_delta_s if actor_relative else raw_delta_s
        delta_s = float(np.clip(
            delta_s,
            float(self.upper_action_low.min()),
            float(self.upper_action_high.max())))
        offset_s = delta_s - actor_delta_s if actor_relative else raw_delta_s
        return self._snapshot_selector_action_features(
            mode, delta_s, actor_delta_s=actor_delta_s, offset_s=offset_s)

    def _snapshot_selector_headway_cv(self):
        values = []
        for bus in getattr(self.env, 'bus_all', []):
            if not getattr(bus, 'on_route', False):
                continue
            for attr in ('forward_headway', 'backward_headway'):
                value = float(getattr(bus, attr, 0.0) or 0.0)
                if value > 0.0 and np.isfinite(value):
                    values.append(value)
        if len(values) < 2:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.std() / max(arr.mean(), 1.0))

    def _snapshot_selector_context(self, trip, actor_action=None):
        launch = float(getattr(trip, 'launch_time', 0.0))
        hour = 6 + int(launch) // 3600
        period = 'peak' if (7 <= hour <= 9 or 17 <= hour <= 19) else (
            'off' if 9 < hour < 17 else 'trans')
        try:
            waiting = sum(
                len(getattr(station, 'waiting_passengers', []))
                for station in getattr(self.env, 'stations', []))
        except Exception:
            waiting = 0
        concurrent, n_fleet, _ = self._fleet_pressure()
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        base_hw = float(getattr(trip, 'target_headway', 360.0))
        if self.timetable_planner is not None:
            try:
                base_hw = float(self.timetable_planner._base_headway(trip))
            except Exception:
                pass
        actor = np.asarray(
            actor_action if actor_action is not None else [0.0],
            dtype=np.float64).reshape(-1)
        actor_delta_s = float(actor[0]) if actor.size else 0.0
        fleet_target = max(float(n_fleet), 1.0)
        return {
            'dir_signed': 1.0 if bool(getattr(trip, 'direction', True)) else -1.0,
            'dispatch_index_norm':
                float(getattr(trip, 'launch_turn', 0.0)) / 262.0,
            'snapshot_time_norm':
                float(getattr(self.env, 'current_time', 0.0)) / 86400.0,
            'scheduled_launch_norm': launch / 86400.0,
            'hour_norm': float(hour) / 24.0,
            'period_is_peak': 1.0 if period == 'peak' else 0.0,
            'period_is_off': 1.0 if period == 'off' else 0.0,
            'period_is_trans': 1.0 if period == 'trans' else 0.0,
            'base_target_headway_norm': base_hw / 600.0,
            'actor_delta_norm': actor_delta_s / 60.0,
            'actor_abs_delta_norm': abs(actor_delta_s) / 60.0,
            'actor_terminal_dispatch':
                1.0 if self.timetable_terminal_dispatch else 0.0,
            'waiting_total_pre_norm': float(waiting) / 500.0,
            'fleet_concurrent_pre_norm': float(concurrent) / 30.0,
            'fleet_target_pre_norm': fleet_target / 30.0,
            'fleet_pressure_pre': (
                (float(concurrent) - fleet_target) / max(fleet_target, 1.0)),
            'headway_cv_active_pre': self._snapshot_selector_headway_cv(),
            'freq_low_demand': float(freq.get('freq_low_demand', 0.0)),
            'freq_low_forecast': float(freq.get('freq_low_forecast', 0.0)),
            'freq_high_energy': float(freq.get('freq_high_energy', 0.0)),
            'freq_middle_energy': float(freq.get('freq_middle_energy', 0.0)),
            'freq_od_entropy': float(freq.get('freq_od_entropy', 0.0)),
            'freq_promotion_strength':
                float(freq.get('freq_promotion_strength', 0.0)),
            'freq_promotion_active':
                float(freq.get('freq_promotion_active', 0.0)),
            'cfg_demand_noise': float(getattr(self.env, 'demand_noise', 0.0)),
            'cfg_peak_shift_abs': self._fixed_selector_peak_shift_abs(),
        }

    def _snapshot_selector_feature_row(self, method, context, meta=None,
                                       domain=None):
        row = dict(context)
        actor_delta_s = float(row.get('actor_delta_norm', 0.0)) * 60.0
        action = self._snapshot_selector_parse_action(
            method, actor_delta_s=actor_delta_s)
        row.update(action)
        if meta is None:
            meta = self.snapshot_value_selector_meta
        if domain is None:
            domain = self._snapshot_selector_domain()
        domains = ('terminal', 'highnoise', 'odshift', 'rushshift')
        action_cols = [
            'action_delta_norm', 'action_abs_delta_norm',
            'action_positive', 'action_negative', 'action_zero',
            'action_term45', 'action_target',
            'action_term45_x_delta', 'action_term45_x_abs_delta',
        ]
        for dom in domains:
            row[f'domain_is_{dom}'] = 1.0 if domain == dom else 0.0
        for dom in domains:
            domain_col = f'domain_is_{dom}'
            for action_col in action_cols:
                row[f'{domain_col}_x_{action_col}'] = (
                    row[domain_col] * row[action_col])
        for ctx_col in meta.get('context_cols', []):
            for action_col in (
                    'action_delta_norm', 'action_abs_delta_norm',
                    'action_term45'):
                row[f'{ctx_col}_x_{action_col}'] = (
                    float(row.get(ctx_col, 0.0)) * row[action_col])
        return row

    def _snapshot_selector_candidate_gate_cap_s(self):
        if not self.snapshot_value_candidate_gate_enable:
            return None
        cap = self.snapshot_value_candidate_gate_default_max_positive_offset_s
        high_noise_min = (
            self.snapshot_value_candidate_gate_high_noise_min_demand_noise)
        high_noise_cap = (
            self.snapshot_value_candidate_gate_high_noise_max_positive_offset_s)
        demand_noise = float(getattr(self.env, 'demand_noise', 0.0))
        if (high_noise_min is not None
                and high_noise_cap is not None
                and demand_noise >= float(high_noise_min)):
            cap = float(high_noise_cap)
        risk_cap = self.snapshot_value_candidate_gate_risk_max_positive_offset_s
        if risk_cap is not None:
            prev = self._fixed_selector_prev_diag or {}

            def _prev(key, default=0.0):
                try:
                    return float(prev.get(key, default))
                except (TypeError, ValueError):
                    return float(default)

            n_fleet = max(float(_prev('N_fleet', self.N_fleet_default)), 1.0)
            risk = False
            if self.snapshot_value_candidate_gate_max_prev_headway_cv is not None:
                risk = risk or (
                    _prev('headway_cv')
                    > float(self.snapshot_value_candidate_gate_max_prev_headway_cv))
            if self.snapshot_value_candidate_gate_max_prev_overshoot_norm is not None:
                risk = risk or (
                    _prev('fleet_overshoot') / n_fleet
                    > float(self.snapshot_value_candidate_gate_max_prev_overshoot_norm))
            if self.snapshot_value_candidate_gate_max_prev_terminal_shift_std_s is not None:
                risk = risk or (
                    _prev('terminal_launch_shift_std')
                    > float(self.snapshot_value_candidate_gate_max_prev_terminal_shift_std_s))
            if risk:
                cap = (
                    float(risk_cap) if cap is None
                    else min(float(cap), float(risk_cap)))
        return None if cap is None else max(0.0, float(cap))

    def _snapshot_selector_apply_candidate_gate(self, methods, context):
        cap_s = self._snapshot_selector_candidate_gate_cap_s()
        if cap_s is None:
            return list(methods), {'cap_s': 0.0, 'filtered': 0.0}
        actor_delta_s = float(context.get('actor_delta_norm', 0.0)) * 60.0
        fallback = self.snapshot_value_selector_fallback_method
        kept = []
        filtered = 0
        for method in methods:
            if method == fallback:
                kept.append(method)
                continue
            try:
                action = self._snapshot_selector_parse_action(
                    method, actor_delta_s=actor_delta_s)
                offset_s = (
                    float(action.get('candidate_offset_norm', 0.0)) * 60.0)
            except (TypeError, ValueError):
                kept.append(method)
                continue
            if offset_s > cap_s + 1e-9:
                filtered += 1
                continue
            kept.append(method)
        if fallback not in kept:
            kept.append(fallback)
        return kept, {'cap_s': float(cap_s), 'filtered': float(filtered)}

    def _snapshot_selector_risk_penalty_score(self, context):
        if (not self.snapshot_value_risk_penalty_enable
                or self.snapshot_value_risk_penalty_weight <= 0.0):
            return 0.0

        def _excess(value, target, width):
            if target is None:
                return 0.0
            try:
                value = float(value)
                target = float(target)
                width = max(float(width), 1e-6)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value):
                return 0.0
            return max(0.0, (value - target) / width)

        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        n_fleet = max(float(_prev('N_fleet', self.N_fleet_default)), 1.0)
        score = 0.0
        score += _excess(
            _prev('headway_cv'),
            self.snapshot_value_risk_penalty_prev_headway_cv_target,
            self.snapshot_value_risk_penalty_prev_headway_cv_width)
        score += _excess(
            _prev('fleet_overshoot') / n_fleet,
            self.snapshot_value_risk_penalty_prev_overshoot_norm_target,
            self.snapshot_value_risk_penalty_prev_overshoot_norm_width)
        score += _excess(
            _prev('terminal_launch_shift_std'),
            self.snapshot_value_risk_penalty_prev_terminal_shift_std_target_s,
            self.snapshot_value_risk_penalty_prev_terminal_shift_std_width_s)
        score += _excess(
            context.get('headway_cv_active_pre', 0.0),
            self.snapshot_value_risk_penalty_context_headway_cv_target,
            self.snapshot_value_risk_penalty_context_headway_cv_width)
        score += _excess(
            context.get('fleet_pressure_pre', 0.0),
            self.snapshot_value_risk_penalty_context_fleet_pressure_target,
            self.snapshot_value_risk_penalty_context_fleet_pressure_width)
        return float(np.clip(
            score,
            0.0,
            max(self.snapshot_value_risk_penalty_max_score, 0.0)))

    def _snapshot_selector_risk_penalties(self, rows, context):
        if (not self.snapshot_value_risk_penalty_enable
                or self.snapshot_value_risk_penalty_weight <= 0.0):
            return np.zeros(len(rows), dtype=np.float64), 0.0
        risk_score = self._snapshot_selector_risk_penalty_score(context)
        if risk_score <= 0.0:
            return np.zeros(len(rows), dtype=np.float64), 0.0
        penalties = []
        for row in rows:
            offset_s = float(row.get('candidate_offset_norm', 0.0)) * 60.0
            excess_offset = max(
                0.0,
                offset_s
                - self.snapshot_value_risk_penalty_positive_offset_start_s)
            offset_factor = (
                excess_offset
                / self.snapshot_value_risk_penalty_positive_offset_scale_s)
            penalty = (
                self.snapshot_value_risk_penalty_weight
                * offset_factor
                * risk_score)
            penalties.append(min(
                float(penalty),
                self.snapshot_value_risk_penalty_max_penalty))
        return np.asarray(penalties, dtype=np.float64), float(risk_score)

    def _snapshot_action_value_risk_score(self, context, primary_info=None):
        if (not self.snapshot_action_value_risk_margin_enable
                or self.snapshot_action_value_risk_margin_weight <= 0.0):
            return 0.0

        def _excess(value, target, width):
            if target is None:
                return 0.0
            try:
                value = float(value)
                target = float(target)
                width = max(float(width), 1e-6)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value):
                return 0.0
            return max(0.0, (value - target) / width)

        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        primary_info = primary_info or {}
        n_fleet = max(float(_prev('N_fleet', self.N_fleet_default)), 1.0)
        primary_bias_s = max(
            0.0,
            float(primary_info.get('terminal_bias_s', 0.0) or 0.0))
        score = 0.0
        score += _excess(
            _prev('headway_cv'),
            self.snapshot_action_value_risk_margin_prev_headway_cv_target,
            self.snapshot_action_value_risk_margin_prev_headway_cv_width)
        score += _excess(
            _prev('fleet_overshoot') / n_fleet,
            self.snapshot_action_value_risk_margin_prev_overshoot_norm_target,
            self.snapshot_action_value_risk_margin_prev_overshoot_norm_width)
        score += _excess(
            context.get('headway_cv_active_pre', 0.0),
            self.snapshot_action_value_risk_margin_context_headway_cv_target,
            self.snapshot_action_value_risk_margin_context_headway_cv_width)
        score += _excess(
            context.get('fleet_pressure_pre', 0.0),
            self.snapshot_action_value_risk_margin_context_fleet_pressure_target,
            self.snapshot_action_value_risk_margin_context_fleet_pressure_width)
        score += _excess(
            primary_bias_s,
            self.snapshot_action_value_risk_margin_primary_bias_target_s,
            self.snapshot_action_value_risk_margin_primary_bias_width_s)
        score += _excess(
            abs(float(context.get('cfg_peak_shift_abs', 0.0) or 0.0)),
            self.snapshot_action_value_risk_margin_peak_shift_abs_target,
            self.snapshot_action_value_risk_margin_peak_shift_abs_width)
        return float(np.clip(
            score,
            0.0,
            max(self.snapshot_action_value_risk_margin_max_score, 0.0)))

    def _snapshot_action_value_risk_penalties(
            self, rows, context, primary_info=None):
        if (not self.snapshot_action_value_risk_margin_enable
                or self.snapshot_action_value_risk_margin_weight <= 0.0):
            return np.zeros(len(rows), dtype=np.float64), 0.0
        risk_score = self._snapshot_action_value_risk_score(
            context, primary_info=primary_info)
        if risk_score <= 0.0:
            return np.zeros(len(rows), dtype=np.float64), 0.0
        penalties = []
        for row in rows:
            offset_s = float(row.get('candidate_offset_norm', 0.0)) * 60.0
            factor = abs(offset_s) / (
                self.snapshot_action_value_risk_margin_abs_offset_scale_s)
            if float(row.get('action_target', 0.0)) > 0.5:
                factor += self.snapshot_action_value_risk_margin_target_base
            if float(row.get('action_term45', 0.0)) > 0.5:
                factor *= self.snapshot_action_value_risk_margin_term45_multiplier
            if offset_s < -1e-9:
                factor *= self.snapshot_action_value_risk_margin_negative_multiplier
            elif offset_s > 1e-9:
                factor *= self.snapshot_action_value_risk_margin_positive_multiplier
            penalty = (
                self.snapshot_action_value_risk_margin_weight
                * float(risk_score)
                * max(0.0, float(factor)))
            penalties.append(min(
                float(penalty),
                self.snapshot_action_value_risk_margin_max_penalty))
        return np.asarray(penalties, dtype=np.float64), float(risk_score)

    def _select_snapshot_value_action(
            self, action_vec, direction, trip=None, plan_origin_launch=None):
        del direction, plan_origin_launch
        actor_action = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        info = {
            'active': 0.0,
            'selected_method': '',
            'selected_mode': '',
            'terminal_dispatch': 0.0,
            'terminal_bias_s': 0.0,
            'override_action': 0.0,
            'selected_pred': 0.0,
            'baseline_pred': 0.0,
            'margin': 0.0,
            'candidate_gate_cap_s': 0.0,
            'candidate_gate_filtered': 0.0,
            'risk_score': 0.0,
            'risk_penalty': 0.0,
            'risk_penalty_max': 0.0,
        }
        if (not self.snapshot_value_selector_enable
                or (self.snapshot_value_selector_model is None
                    and self.snapshot_value_selector_forest is None)
                or trip is None
                or self._current_ep < self.snapshot_value_selector_start_ep
                or not self.timetable_terminal_dispatch):
            return np.asarray(action_vec, dtype=np.float32), info

        methods = list(self.snapshot_value_selector_candidate_methods)
        if self.snapshot_value_selector_allowed_methods:
            methods = [
                method for method in methods
                if method in self.snapshot_value_selector_allowed_methods
            ]
        if self.snapshot_value_selector_blocked_methods:
            methods = [
                method for method in methods
                if method not in self.snapshot_value_selector_blocked_methods
            ]
        if self.snapshot_value_selector_fallback_method not in methods:
            methods.append(self.snapshot_value_selector_fallback_method)
        context = self._snapshot_selector_context(trip, actor_action=actor_action)
        methods, gate_info = self._snapshot_selector_apply_candidate_gate(
            methods, context)
        info['candidate_gate_cap_s'] = float(gate_info['cap_s'])
        info['candidate_gate_filtered'] = float(gate_info['filtered'])
        rows = [
            self._snapshot_selector_feature_row(method, context)
            for method in methods
        ]
        matrix = []
        for row in rows:
            values = []
            for col in self.snapshot_value_selector_feature_cols:
                value = float(row.get(
                    col,
                    self.snapshot_value_selector_feature_medians.get(
                        col, 0.0)))
                if not np.isfinite(value):
                    value = float(
                        self.snapshot_value_selector_feature_medians.get(
                            col, 0.0))
                values.append(value)
            matrix.append(values)
        x = np.asarray(matrix, dtype=np.float64)
        pred = self._snapshot_value_predict(x)
        risk_penalty, risk_score = self._snapshot_selector_risk_penalties(
            rows, context)
        if risk_penalty.size:
            pred = pred + risk_penalty
        scores = {method: float(value) for method, value in zip(methods, pred)}
        fallback = self.snapshot_value_selector_fallback_method
        baseline_pred = float(scores.get(fallback, np.nan))
        selected_method = min(scores, key=scores.get)
        selected_pred = float(scores[selected_method])
        margin = baseline_pred - selected_pred
        actor_fallback = self.snapshot_value_selector_fallback_action in {
            'actor', 'main', 'policy', 'keep_actor'
        }
        if (not np.isfinite(margin)
                or margin < self.snapshot_value_selector_improve_margin):
            selected_method = fallback
            selected_pred = baseline_pred
            margin = 0.0
        override_action = not (actor_fallback and selected_method == fallback)
        actor_delta_s = float(context.get('actor_delta_norm', 0.0)) * 60.0
        selected_action = self._snapshot_selector_parse_action(
            selected_method, actor_delta_s=actor_delta_s)
        selected_mode = str(selected_action['action_mode'])
        candidate_offset_s = (
            float(selected_action.get('candidate_offset_norm', 0.0)) * 60.0)
        terminal_bias_s = 0.0
        selected_penalty = 0.0
        if risk_penalty.size:
            try:
                selected_index = methods.index(selected_method)
                selected_penalty = float(risk_penalty[selected_index])
            except (ValueError, IndexError):
                selected_penalty = 0.0
        if self.snapshot_value_selector_apply_mode == 'terminal_bias':
            # In terminal-bias mode the counterfactual selector controls only
            # first-stop delay. The actor's headway/timetable coefficients are
            # preserved so the learned low-frequency plan is not overwritten.
            override_action = False
            terminal_bias_s = max(0.0, candidate_offset_s)
        info.update({
            'active': 1.0,
            'selected_method': selected_method,
            'selected_mode': (
                'actor' if not override_action
                else selected_mode),
            'terminal_dispatch': (
                (1.0 if self.timetable_terminal_dispatch else 0.0)
                if not override_action else
                1.0 if selected_mode == 'term45'
                else 0.0),
            'terminal_bias_s': float(terminal_bias_s),
            'override_action': 1.0 if override_action else 0.0,
            'selected_pred': float(selected_pred),
            'baseline_pred': float(baseline_pred),
            'margin': float(margin),
            'risk_score': float(risk_score),
            'risk_penalty': float(selected_penalty),
            'risk_penalty_max': (
                float(np.max(risk_penalty)) if risk_penalty.size else 0.0),
        })
        if self.snapshot_value_selector_probe_only:
            return actor_action, info
        if not override_action:
            return actor_action, info
        selected_vec = np.full(
            self.upper_action_dim,
            float(selected_action['action_delta_s']),
            dtype=np.float32)
        selected_vec = np.clip(
            selected_vec, self.upper_action_low, self.upper_action_high)
        return self._quantize_upper_action(selected_vec), info

    def _select_snapshot_action_value_action(
            self, action_vec, direction, trip=None, plan_origin_launch=None,
            primary_info=None):
        del direction, plan_origin_launch
        actor_action = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        info = {
            'active': 0.0,
            'selected_method': '',
            'selected_mode': '',
            'terminal_dispatch': 0.0,
            'terminal_bias_s': 0.0,
            'override_action': 0.0,
            'selected_pred': 0.0,
            'baseline_pred': 0.0,
            'margin': 0.0,
            'candidate_gate_cap_s': 0.0,
            'candidate_gate_filtered': 0.0,
            'risk_score': 0.0,
            'risk_penalty': 0.0,
            'risk_penalty_max': 0.0,
            'guard_blocked': 0.0,
            'guard_negative_target': 0.0,
            'guard_negative_target_blocked': 0.0,
            'guard_prev_overshoot_norm': 0.0,
            'guard_fleet_pressure_norm': 0.0,
            'guard_primary_terminal_bias_s': 0.0,
        }
        if (not self.snapshot_action_value_selector_enable
                or (self.snapshot_action_value_selector_model is None
                    and self.snapshot_action_value_selector_forest is None)
                or trip is None
                or self._current_ep < self.snapshot_action_value_selector_start_ep
                or not self.timetable_terminal_dispatch):
            return actor_action, info

        methods = list(self.snapshot_action_value_selector_candidate_methods)
        if self.snapshot_action_value_selector_allowed_methods:
            methods = [
                method for method in methods
                if method in self.snapshot_action_value_selector_allowed_methods
            ]
        if self.snapshot_action_value_selector_blocked_methods:
            methods = [
                method for method in methods
                if method not in self.snapshot_action_value_selector_blocked_methods
            ]
        fallback = self.snapshot_action_value_selector_fallback_method
        if fallback not in methods:
            methods.append(fallback)
        context = self._snapshot_selector_context(trip, actor_action=actor_action)
        rows = [
            self._snapshot_selector_feature_row(
                method,
                context,
                meta=self.snapshot_action_value_selector_meta,
                domain=self._snapshot_action_selector_domain())
            for method in methods
        ]
        matrix = []
        for row in rows:
            values = []
            for col in self.snapshot_action_value_selector_feature_cols:
                value = float(row.get(
                    col,
                    self.snapshot_action_value_selector_feature_medians.get(
                        col, 0.0)))
                if not np.isfinite(value):
                    value = float(
                        self.snapshot_action_value_selector_feature_medians.get(
                            col, 0.0))
                values.append(value)
            matrix.append(values)
        x = np.asarray(matrix, dtype=np.float64)
        pred = self._snapshot_value_predict(
            x,
            model=self.snapshot_action_value_selector_model,
            forest=self.snapshot_action_value_selector_forest)
        risk_penalty, risk_score = (
            self._snapshot_action_value_risk_penalties(
                rows, context, primary_info=primary_info))
        if risk_penalty.size:
            pred = pred + risk_penalty
        scores = {method: float(value) for method, value in zip(methods, pred)}
        baseline_pred = float(scores.get(fallback, np.nan))
        selected_method = min(scores, key=scores.get)
        selected_pred = float(scores[selected_method])
        margin = baseline_pred - selected_pred
        actor_fallback = self.snapshot_action_value_selector_fallback_action in {
            'actor', 'main', 'policy', 'keep_actor'
        }
        if (not np.isfinite(margin)
                or margin < self.snapshot_action_value_selector_improve_margin):
            selected_method = fallback
            selected_pred = baseline_pred
            margin = 0.0
        override_action = not (actor_fallback and selected_method == fallback)
        actor_delta_s = float(context.get('actor_delta_norm', 0.0)) * 60.0
        selected_action = self._snapshot_selector_parse_action(
            selected_method, actor_delta_s=actor_delta_s)
        selected_mode = str(selected_action['action_mode'])
        primary_info = primary_info or {}
        primary_terminal_bias_s = max(
            0.0,
            float(primary_info.get('terminal_bias_s', 0.0) or 0.0))
        prev = self._fixed_selector_prev_diag or {}

        def _prev_float(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        prev_fleet = max(_prev_float('N_fleet', self.N_fleet_default), 1.0)
        prev_overshoot_norm = (
            _prev_float('fleet_overshoot', 0.0) / prev_fleet)
        prev_headway_cv = _prev_float('headway_cv', 0.0)
        context_headway_cv = float(
            context.get('headway_cv_active_pre', 0.0) or 0.0)
        fleet_pressure_norm = float(context.get('fleet_pressure_pre', 0.0) or 0.0)
        selected_offset_s = (
            float(selected_action.get('candidate_offset_norm', 0.0)) * 60.0)
        abs_offset_norm = abs(float(
            selected_action.get('candidate_offset_norm', 0.0)))
        negative_target = (
            bool(override_action)
            and selected_mode == 'target'
            and selected_offset_s < -1e-9)
        guard_blocked = False
        negative_target_blocked = False
        if override_action and self.snapshot_action_value_guard_enable:
            max_prev_overshoot = (
                self.snapshot_action_value_guard_max_prev_overshoot_norm)
            if (max_prev_overshoot is not None
                    and prev_overshoot_norm > float(max_prev_overshoot)):
                guard_blocked = True
            max_prev_cv = (
                self.snapshot_action_value_guard_max_prev_headway_cv)
            if (max_prev_cv is not None
                    and prev_headway_cv > float(max_prev_cv)):
                guard_blocked = True
            max_context_cv = (
                self.snapshot_action_value_guard_max_context_headway_cv)
            if (max_context_cv is not None
                    and context_headway_cv > float(max_context_cv)):
                guard_blocked = True
            max_pressure = (
                self.snapshot_action_value_guard_max_fleet_pressure_norm)
            if (max_pressure is not None
                    and fleet_pressure_norm > float(max_pressure)):
                guard_blocked = True
            max_abs_offset = (
                self.snapshot_action_value_guard_max_abs_offset_s)
            if (max_abs_offset is not None
                    and abs(selected_offset_s) > float(max_abs_offset)):
                guard_blocked = True
            max_peak_shift = (
                self.snapshot_action_value_guard_max_peak_shift_abs)
            if (max_peak_shift is not None and float(
                    context.get('cfg_peak_shift_abs', 0.0) or 0.0)
                    > float(max_peak_shift)):
                guard_blocked = True
            min_margin = self.snapshot_action_value_guard_min_margin
            margin_per_offset = (
                self.snapshot_action_value_guard_min_margin_per_abs_offset_norm
                or 0.0)
            if min_margin is not None or margin_per_offset > 0.0:
                required_margin = (
                    (float(min_margin) if min_margin is not None else 0.0)
                    + float(margin_per_offset) * float(abs_offset_norm))
                if margin < required_margin:
                    guard_blocked = True
            max_bias_loss = (
                self.snapshot_action_value_guard_max_primary_terminal_bias_loss_s)
            if (max_bias_loss is not None
                    and primary_terminal_bias_s > float(max_bias_loss)):
                guard_blocked = True
            if negative_target:
                max_neg_prev_overshoot = (
                    self.snapshot_action_value_guard_max_negative_target_prev_overshoot_norm)
                if (max_neg_prev_overshoot is not None
                        and prev_overshoot_norm > float(max_neg_prev_overshoot)):
                    negative_target_blocked = True
                max_neg_pressure = (
                    self.snapshot_action_value_guard_max_negative_target_fleet_pressure_norm)
                if (max_neg_pressure is not None
                        and fleet_pressure_norm > float(max_neg_pressure)):
                    negative_target_blocked = True
                min_neg_margin = (
                    self.snapshot_action_value_guard_min_negative_target_margin)
                if (min_neg_margin is not None
                        and margin < float(min_neg_margin)):
                    negative_target_blocked = True
                guard_blocked = guard_blocked or negative_target_blocked
        if guard_blocked:
            selected_method = fallback
            selected_pred = baseline_pred
            margin = 0.0
            override_action = False
            selected_mode = 'actor'
        selected_penalty = 0.0
        if risk_penalty.size:
            try:
                selected_penalty = float(risk_penalty[methods.index(
                    selected_method)])
            except (ValueError, IndexError):
                selected_penalty = 0.0
        info.update({
            'active': 1.0,
            'selected_method': selected_method,
            'selected_mode': (
                'actor' if not override_action
                else selected_mode),
            'terminal_dispatch': (
                (1.0 if self.timetable_terminal_dispatch else 0.0)
                if not override_action else
                1.0 if selected_mode == 'term45'
                else 0.0),
            'terminal_bias_s': 0.0,
            'override_action': 1.0 if override_action else 0.0,
            'selected_pred': float(selected_pred),
            'baseline_pred': float(baseline_pred),
            'margin': float(margin),
            'risk_score': float(risk_score),
            'risk_penalty': float(selected_penalty),
            'risk_penalty_max': (
                float(np.max(risk_penalty)) if risk_penalty.size else 0.0),
            'guard_blocked': 1.0 if guard_blocked else 0.0,
            'guard_negative_target': 1.0 if negative_target else 0.0,
            'guard_negative_target_blocked': (
                1.0 if negative_target_blocked else 0.0),
            'guard_prev_overshoot_norm': float(prev_overshoot_norm),
            'guard_fleet_pressure_norm': float(fleet_pressure_norm),
            'guard_primary_terminal_bias_s': float(primary_terminal_bias_s),
        })
        if not override_action:
            return actor_action, info
        selected_vec = np.full(
            self.upper_action_dim,
            float(selected_action['action_delta_s']),
            dtype=np.float32)
        selected_vec = np.clip(
            selected_vec, self.upper_action_low, self.upper_action_high)
        return self._quantize_upper_action(selected_vec), info

    def _upper_residual_selector_slice(self, action_vec, direction):
        action = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if self.timetable_planner is not None:
            b = max(1, int(self.timetable_planner.basis_per_direction))
            if action.size == 2 * b:
                return slice(0, b) if bool(direction) else slice(b, 2 * b)
            if action.size == b:
                return slice(0, b)
        return slice(0, action.size)

    def _upper_residual_selector_candidates(self, action_vec, direction):
        base = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        base = np.clip(
            base, self.upper_action_low, self.upper_action_high
        ).astype(np.float32)
        idx = self._upper_residual_selector_slice(base, direction)
        candidates = [self._quantize_upper_action(base)]
        for offset in self.upper_residual_selector_offsets:
            if abs(float(offset)) < 1e-9:
                continue
            cand = base.copy()
            cand[idx] = cand[idx] + float(offset)
            cand = np.clip(
                cand, self.upper_action_low, self.upper_action_high
            ).astype(np.float32)
            candidates.append(self._quantize_upper_action(cand))

        deduped = []
        seen = set()
        for cand in candidates:
            key = tuple(np.round(cand.astype(np.float64), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        return deduped

    def _upper_residual_selector_plan_features(
            self, action_vec, direction, trip=None, plan_origin_launch=None):
        """Candidate-local timetable consequences for contextual residual choice."""
        if (not self.upper_residual_selector_plan_context
                or self.timetable_planner is None
                or trip is None):
            return np.zeros(0, dtype=np.float64)

        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        direction = bool(direction)
        current_launch = float(getattr(trip, 'launch_time', 0.0))
        origin = (
            current_launch if plan_origin_launch is None
            else float(plan_origin_launch))
        offset = current_launch - origin
        base = self.timetable_planner._base_headway(trip)
        target = self.timetable_planner.target_headway(
            base, action, direction, offset)
        delta = target - base

        last_dispatch = float(getattr(
            self.env, '_last_dispatch_time', {}).get(direction, -9999.0))
        now = float(getattr(self.env, 'current_time', current_launch))
        gap_now = now - last_dispatch
        if gap_now > 9000.0 or gap_now < 0.0:
            gap_now = base
        gap_ratio = gap_now / max(target, 1.0)
        gap_deficit = max(0.0, target - gap_now)
        gap_excess = max(0.0, gap_now - target)

        next_trip = None
        for tt in getattr(self.env, 'timetables', []):
            if bool(getattr(tt, 'direction', False)) != direction:
                continue
            if getattr(tt, 'launched', False):
                continue
            if float(getattr(tt, 'launch_time', 0.0)) <= current_launch:
                continue
            if next_trip is None or float(tt.launch_time) < float(next_trip.launch_time):
                next_trip = tt

        if next_trip is None:
            next_base = base
            next_target = target
            next_gap = base
        else:
            next_base = self.timetable_planner._base_headway(next_trip)
            next_offset = float(next_trip.launch_time) - origin
            next_target = self.timetable_planner.target_headway(
                next_base, action, direction, next_offset)
            next_gap = float(next_trip.launch_time) - current_launch

        next_delta = next_target - next_base
        target_slope = next_target - target
        next_gap_ratio = next_gap / max(next_target, 1.0)
        current_compress = max(0.0, -delta)
        next_compress = max(0.0, -next_delta)
        current_relief = max(0.0, delta)
        next_relief = max(0.0, next_delta)

        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        concurrent, n_fleet, pressure = self._fleet_pressure()
        util = float(concurrent) / max(float(n_fleet), 1.0)

        return np.asarray([
            target / 600.0,
            delta / 30.0,
            gap_now / 600.0,
            gap_ratio,
            gap_deficit / 60.0,
            gap_excess / 60.0,
            next_target / 600.0,
            next_delta / 30.0,
            target_slope / 30.0,
            next_gap / 600.0,
            next_gap_ratio,
            current_compress / 15.0,
            next_compress / 15.0,
            current_relief / 15.0,
            next_relief / 15.0,
            util,
            float(pressure) / max(float(n_fleet), 1.0),
            float(freq.get('freq_low_forecast', 0.0)),
            10.0 * float(freq.get('freq_high_energy', 0.0)),
            float(freq.get('freq_promotion_strength', 0.0)),
        ], dtype=np.float64)

    def _upper_residual_selector_safety_penalty(
            self, actor_vec, cand_vec, direction, trip=None,
            plan_origin_launch=None):
        if (self.upper_residual_selector_compression_safety_weight <= 0.0
                or self.timetable_planner is None
                or trip is None):
            return 0.0
        actor_feat = self._upper_residual_selector_plan_features(
            actor_vec, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        cand_feat = self._upper_residual_selector_plan_features(
            cand_vec, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        if actor_feat.size < 17 or cand_feat.size < 17:
            return 0.0
        actor_target = float(actor_feat[0] * 600.0)
        cand_target = float(cand_feat[0] * 600.0)
        extra_compression = max(0.0, actor_target - cand_target)
        if extra_compression <= 0.0:
            return 0.0
        short_gap = max(0.0, 1.0 - float(cand_feat[3]))
        fleet_pressure = max(0.0, float(cand_feat[16]))
        penalty = (
            self.upper_residual_selector_compression_safety_weight
            * extra_compression
            / self.upper_residual_selector_compression_norm_s)
        penalty *= (
            1.0
            + self.upper_residual_selector_short_gap_weight * short_gap
            + self.upper_residual_selector_fleet_pressure_weight * fleet_pressure)
        return float(penalty)

    def _upper_residual_selector_features(
            self, s_upper, action_vec, direction, trip=None,
            plan_origin_launch=None):
        s = np.asarray(s_upper, dtype=np.float64).reshape(-1)
        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        idx = self._upper_residual_selector_slice(action, direction)
        block = np.asarray(action[idx], dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        block_mean = float(block.mean()) if block.size else 0.0
        block_std = float(block.std()) if block.size else 0.0
        block_slope = float(block[-1] - block[0]) if block.size >= 2 else 0.0
        neg = max(0.0, -block_mean) / 15.0
        pos = max(0.0, block_mean) / 15.0
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        concurrent, n_fleet, pressure = self._fleet_pressure()
        util = float(concurrent) / max(float(n_fleet), 1.0)

        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        demand_noise = float(getattr(self.env, 'demand_noise', 0.0))
        od_noise = float(getattr(self.env, 'od_noise', 0.0))
        peak_shift = self._fixed_selector_peak_shift_abs()
        high_energy = float(freq.get('freq_high_energy', 0.0))
        middle_energy = float(freq.get('freq_middle_energy', 0.0))
        promotion = float(freq.get('freq_promotion_strength', 0.0))
        absorbed = float(freq.get('freq_promotion_absorbed', 0.0))
        low_demand = float(freq.get('freq_low_demand', 0.0))
        low_forecast = float(freq.get('freq_low_forecast', 0.0))
        prev_wait = _prev('avg_wait_min') / 10.0
        prev_cv = _prev('headway_cv')
        prev_overshoot = _prev('fleet_overshoot') / max(float(n_fleet), 1.0)

        values = [
            1.0,
            1.0 if bool(direction) else -1.0,
            demand_noise,
            od_noise,
            peak_shift,
            util,
            float(pressure) / max(float(n_fleet), 1.0),
            low_demand,
            low_forecast,
            10.0 * high_energy,
            10.0 * middle_energy,
            promotion,
            absorbed,
            prev_wait,
            prev_cv,
            prev_overshoot,
            block_mean / 30.0,
            block_slope / 30.0,
            block_std / 30.0,
            neg,
            pos,
            float(np.linalg.norm(action)) / (
                max(float(np.sqrt(max(action.size, 1))), 1.0) * 60.0),
            util * neg,
            (10.0 * high_energy) * neg,
            promotion * neg,
            prev_cv * neg,
            demand_noise * neg,
            od_noise * neg,
            peak_shift * neg,
            util * pos,
            (10.0 * high_energy) * pos,
        ]
        clip = self.upper_residual_selector_feature_clip
        state_part = s if clip <= 0.0 else np.clip(s, -clip, clip)
        x = np.concatenate([
            np.asarray(values, dtype=np.float64),
            self._upper_residual_selector_plan_features(
                action, direction, trip=trip,
                plan_origin_launch=plan_origin_launch),
            state_part,
        ])
        if clip > 0.0 and x.size > 1:
            x[1:] = np.clip(x[1:], -clip, clip)
        return x

    def _ensure_upper_residual_selector_model(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.upper_residual_selector_A is None:
            dim = int(x.size)
            self.upper_residual_selector_A = (
                self.upper_residual_selector_ridge
                * np.eye(dim, dtype=np.float64))
            self.upper_residual_selector_b = np.zeros(dim, dtype=np.float64)
            return True
        return int(self.upper_residual_selector_A.shape[0]) == int(x.size)

    def _upper_residual_selector_theta(self, x):
        if not self._ensure_upper_residual_selector_model(x):
            return None
        try:
            return np.linalg.solve(
                self.upper_residual_selector_A,
                self.upper_residual_selector_b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(
                self.upper_residual_selector_A,
                self.upper_residual_selector_b,
                rcond=None)[0]

    def _update_upper_residual_selector(self, x, reward):
        if not self.upper_residual_selector_enable or x is None:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not self._ensure_upper_residual_selector_model(x):
            return
        cost = -float(reward)
        if self.upper_residual_selector_cost_clip > 0.0:
            cost = float(np.clip(
                cost,
                -self.upper_residual_selector_cost_clip,
                self.upper_residual_selector_cost_clip))
        self.upper_residual_selector_A += np.outer(x, x)
        self.upper_residual_selector_b += cost * x
        self.upper_residual_selector_updates += 1

    def _select_upper_residual_value_action(
            self, s_upper, action_vec, direction, trip=None,
            plan_origin_launch=None):
        if not self.upper_residual_selector_enable:
            return np.asarray(action_vec, dtype=np.float32), None
        actor = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        actor_x = self._upper_residual_selector_features(
            s_upper, actor, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        self._ep_upper_residual_selector_feature_norms.append(
            float(np.linalg.norm(actor_x)))
        if (int(self._current_ep) < int(self.upper_residual_selector_start_ep)
                or self.upper_residual_selector_updates
                < self.upper_residual_selector_min_observations):
            self._ep_upper_residual_selector_active.append(0.0)
            self._ep_upper_residual_selector_adjusts.append(0.0)
            self._ep_upper_residual_selector_margins.append(0.0)
            self._ep_upper_residual_selector_actor_preds.append(0.0)
            self._ep_upper_residual_selector_selected_preds.append(0.0)
            return actor, actor_x
        theta = self._upper_residual_selector_theta(actor_x)
        if theta is None:
            self._ep_upper_residual_selector_active.append(0.0)
            self._ep_upper_residual_selector_adjusts.append(0.0)
            self._ep_upper_residual_selector_margins.append(0.0)
            self._ep_upper_residual_selector_actor_preds.append(0.0)
            self._ep_upper_residual_selector_selected_preds.append(0.0)
            return actor, actor_x

        candidates = self._upper_residual_selector_candidates(actor, direction)
        scored = []
        actor_pred = float(np.dot(actor_x, theta))
        for cand in candidates:
            x = self._upper_residual_selector_features(
                s_upper, cand, direction, trip=trip,
                plan_origin_launch=plan_origin_launch)
            pred = float(np.dot(x, theta))
            adjust = float(np.mean(np.abs(cand - actor)))
            safety = self._upper_residual_selector_safety_penalty(
                actor, cand, direction, trip=trip,
                plan_origin_launch=plan_origin_launch)
            score = (
                pred
                + self.upper_residual_selector_adjust_penalty
                * adjust / self.upper_residual_selector_adjust_norm_s
                + safety)
            scored.append((score, pred, adjust, cand, x))

        random_probe = (
            self.upper_residual_selector_epsilon > 0.0
            and self._upper_residual_selector_rng.random()
            < self.upper_residual_selector_epsilon)
        if random_probe and len(scored) > 1:
            chosen = scored[int(
                self._upper_residual_selector_rng.randint(len(scored)))]
        else:
            chosen = min(scored, key=lambda item: item[0])
            actor_score = actor_pred
            if (actor_score - float(chosen[0])
                    < self.upper_residual_selector_improve_margin):
                chosen = (actor_score, actor_pred, 0.0, actor, actor_x)

        score, selected_pred, adjust, selected, selected_x = chosen
        self._ep_upper_residual_selector_active.append(1.0)
        self._ep_upper_residual_selector_adjusts.append(float(adjust))
        self._ep_upper_residual_selector_margins.append(
            float(actor_pred - score))
        self._ep_upper_residual_selector_actor_preds.append(actor_pred)
        self._ep_upper_residual_selector_selected_preds.append(
            float(selected_pred))
        return selected.astype(np.float32), selected_x

    def _headway_value_planner_candidates(self, action_vec, direction):
        base = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        base = np.clip(
            base, self.upper_action_low, self.upper_action_high
        ).astype(np.float32)
        idx = self._upper_residual_selector_slice(base, direction)
        candidates = [('actor', self._quantize_upper_action(base))]

        for delta in self.timetable_headway_value_planner_candidate_deltas:
            cand = base.copy()
            cand[idx] = float(delta)
            cand = np.clip(
                cand, self.upper_action_low, self.upper_action_high
            ).astype(np.float32)
            candidates.append((f'const_{float(delta):+.1f}',
                               self._quantize_upper_action(cand)))

        for offset in self.timetable_headway_value_planner_candidate_offsets:
            if abs(float(offset)) < 1e-9:
                continue
            cand = base.copy()
            cand[idx] = cand[idx] + float(offset)
            cand = np.clip(
                cand, self.upper_action_low, self.upper_action_high
            ).astype(np.float32)
            candidates.append((f'offset_{float(offset):+.1f}',
                               self._quantize_upper_action(cand)))

        deduped = []
        seen = set()
        for name, cand in candidates:
            key = tuple(np.round(cand.astype(np.float64), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            deduped.append((name, cand))
        return deduped

    def _headway_value_planner_plan_features(
            self, action_vec, direction, trip=None, plan_origin_launch=None):
        if self.timetable_planner is None or trip is None:
            return np.zeros(0, dtype=np.float64)

        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        direction = bool(direction)
        current_launch = float(getattr(trip, 'launch_time', 0.0))
        origin = (
            current_launch if plan_origin_launch is None
            else float(plan_origin_launch))
        offset = current_launch - origin
        base = self.timetable_planner._base_headway(trip)
        target = self.timetable_planner.target_headway(
            base, action, direction, offset)
        delta = target - base

        last_dispatch = float(getattr(
            self.env, '_last_dispatch_time', {}).get(direction, -9999.0))
        now = float(getattr(self.env, 'current_time', current_launch))
        gap_now = now - last_dispatch
        if gap_now > 9000.0 or gap_now < 0.0:
            gap_now = base
        gap_ratio = gap_now / max(target, 1.0)
        short_gap = max(0.0, target - gap_now)
        gap_excess = max(0.0, gap_now - target)

        future_targets = []
        next_target = target
        next_delta = delta
        next_gap = base
        next2_target = target
        next2_delta = delta
        next2_gap = base
        future_trips = []
        for tt in getattr(self.env, 'timetables', []):
            if bool(getattr(tt, 'direction', False)) != direction:
                continue
            if getattr(tt, 'launched', False):
                continue
            launch = float(getattr(tt, 'launch_time', 0.0))
            if launch < current_launch - 1e-6:
                continue
            tt_offset = launch - origin
            if tt_offset > self.timetable_planner.horizon_s:
                continue
            future_trips.append(tt)
        future_trips.sort(key=lambda tt: float(tt.launch_time))
        after_current = 0
        for tt in future_trips[:5]:
            tt_base = self.timetable_planner._base_headway(tt)
            tt_offset = float(tt.launch_time) - origin
            tt_target = self.timetable_planner.target_headway(
                tt_base, action, direction, tt_offset)
            future_targets.append(float(tt_target))
            if tt is not trip and float(tt.launch_time) > current_launch:
                after_current += 1
            if after_current == 1:
                next_target = float(tt_target)
                next_delta = float(tt_target - tt_base)
                next_gap = float(tt.launch_time) - current_launch
            elif after_current == 2:
                next2_target = float(tt_target)
                next2_delta = float(tt_target - tt_base)
                next2_gap = float(tt.launch_time) - current_launch
        if future_targets:
            planned_mean = float(np.mean(future_targets))
            planned_std = float(np.std(future_targets))
        else:
            planned_mean = float(target)
            planned_std = 0.0

        block = np.asarray(
            action[self._upper_residual_selector_slice(action, direction)],
            dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        block_mean = float(block.mean()) if block.size else 0.0
        block_std = float(block.std()) if block.size else 0.0
        block_slope = float(block[-1] - block[0]) if block.size >= 2 else 0.0
        compression = max(0.0, -block_mean)
        relief = max(0.0, block_mean)
        prev_gap_error = gap_now - target
        next_gap_error = next_gap - next_target
        next2_gap_error = next2_gap - next2_target
        prev_next_balance = next_gap - gap_now
        target_slope = next_target - target

        return np.asarray([
            target / 600.0,
            delta / 30.0,
            gap_now / 600.0,
            gap_ratio,
            short_gap / 60.0,
            gap_excess / 60.0,
            next_target / 600.0,
            next_delta / 30.0,
            next_gap / 600.0,
            next_gap / max(next_target, 1.0),
            next2_target / 600.0,
            next2_delta / 30.0,
            next2_gap / 600.0,
            next2_gap / max(next2_target, 1.0),
            planned_mean / 600.0,
            planned_std / 60.0,
            block_mean / 30.0,
            block_std / 30.0,
            block_slope / 30.0,
            compression / self.timetable_headway_value_planner_delta_norm_s,
            relief / self.timetable_headway_value_planner_delta_norm_s,
            prev_gap_error / 60.0,
            next_gap_error / 60.0,
            next2_gap_error / 60.0,
            prev_next_balance / 600.0,
            target_slope / 60.0,
        ], dtype=np.float64)

    def _headway_value_planner_prev_metrics(self):
        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        n_fleet = max(float(self._current_N_fleet), 1.0)
        wait = _prev('avg_wait_min')
        cv = _prev('headway_cv')
        overshoot = _prev('fleet_overshoot') / n_fleet
        composite = (
            wait / 10.0
            + (_prev('fleet_overshoot') ** 2) / n_fleet
            + cv)
        return {
            'wait': wait,
            'cv': cv,
            'overshoot_norm': overshoot,
            'composite': composite,
            'terminal_shift': _prev('terminal_launch_shift_mean'),
            'lower_action': _prev('lower_action_mean'),
            'lower_drift': _prev('lower_drift_cost_mean'),
        }

    def _headway_value_planner_action_basis_features(
            self, action_vec, direction, freq, prev, util, pressure_norm):
        """Discrete candidate embedding for demand-conditioned value selection."""
        if not self.timetable_headway_value_planner_action_basis_enable:
            return np.zeros(0, dtype=np.float64)
        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        block = np.asarray(
            action[self._upper_residual_selector_slice(action, direction)],
            dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        mean_delta = float(block.mean()) if block.size else 0.0
        block_std = float(block.std()) if block.size else 0.0
        block_slope = float(block[-1] - block[0]) if block.size >= 2 else 0.0

        centers = np.asarray(
            self.timetable_headway_value_planner_action_basis_centers,
            dtype=np.float64).reshape(-1)
        if centers.size == 0:
            centers = np.asarray([0.0], dtype=np.float64)
        distances = np.abs(centers - mean_delta)
        nearest = int(np.argmin(distances))
        onehot = np.zeros(centers.size, dtype=np.float64)
        onehot[nearest] = 1.0
        width = max(
            float(self.timetable_headway_value_planner_action_basis_width_s),
            1e-6)
        rbf = np.exp(-0.5 * ((mean_delta - centers) / width) ** 2)

        mode = self.timetable_headway_value_planner_action_basis_mode
        action_parts = []
        if 'onehot' in mode or 'one_hot' in mode or mode == 'discrete':
            action_parts.append(onehot)
        if 'rbf' in mode or 'kernel' in mode:
            action_parts.append(rbf)
        if not action_parts:
            action_parts.append(onehot)
        action_key = np.concatenate(action_parts).astype(np.float64)

        context = np.asarray([
            1.0,
            float(freq.get('freq_low_demand', 0.0)),
            float(freq.get('freq_low_forecast', 0.0)),
            10.0 * float(freq.get('freq_high_energy', 0.0)),
            10.0 * float(freq.get('freq_middle_energy', 0.0)),
            float(freq.get('freq_od_entropy', 0.0)),
            float(freq.get('freq_promotion_strength', 0.0)),
            float(freq.get('freq_promotion_absorbed', 0.0)),
            float(prev.get('wait', 0.0)) / 10.0,
            float(prev.get('cv', 0.0)),
            float(prev.get('overshoot_norm', 0.0)),
            float(prev.get('terminal_shift', 0.0)) / 45.0,
            float(prev.get('lower_action', 0.0)) / 5.0,
            float(prev.get('lower_drift', 0.0)),
            float(util),
            float(pressure_norm),
            mean_delta / self.timetable_headway_value_planner_delta_norm_s,
            block_std / self.timetable_headway_value_planner_delta_norm_s,
            block_slope / self.timetable_headway_value_planner_delta_norm_s,
            distances[nearest] / width,
        ], dtype=np.float64)

        feats = [
            np.asarray([
                mean_delta / self.timetable_headway_value_planner_delta_norm_s,
                block_std / self.timetable_headway_value_planner_delta_norm_s,
                block_slope / self.timetable_headway_value_planner_delta_norm_s,
                centers[nearest]
                / self.timetable_headway_value_planner_delta_norm_s,
                distances[nearest] / width,
            ], dtype=np.float64),
            action_key,
        ]
        if self.timetable_headway_value_planner_action_basis_interactions:
            feats.append(np.outer(action_key, context).reshape(-1))
        return np.concatenate(feats).astype(np.float64)

    def _headway_value_planner_features(
            self, s_upper, action_vec, direction, trip=None,
            plan_origin_launch=None):
        direction = bool(direction)
        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        plan = self._headway_value_planner_plan_features(
            action, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        concurrent, n_fleet, pressure = self._fleet_pressure()
        n_fleet = max(float(n_fleet), 1.0)
        util = float(concurrent) / n_fleet
        pressure_norm = float(pressure) / n_fleet
        prev = self._headway_value_planner_prev_metrics()

        waiting_total = 0
        try:
            waiting_total = sum(
                len(st.waiting_passengers)
                for st in getattr(self.env, 'stations', []))
        except Exception:
            waiting_total = 0

        block = np.asarray(
            action[self._upper_residual_selector_slice(action, direction)],
            dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        block_mean = float(block.mean()) if block.size else 0.0
        compression = max(0.0, -block_mean)
        relief = max(0.0, block_mean)
        high_energy = float(freq.get('freq_high_energy', 0.0))
        middle_energy = float(freq.get('freq_middle_energy', 0.0))
        low_forecast = float(freq.get('freq_low_forecast', 0.0))
        promotion = float(freq.get('freq_promotion_strength', 0.0))
        action_basis = self._headway_value_planner_action_basis_features(
            action, direction, freq, prev, util, pressure_norm)

        values = [
            1.0,
            1.0 if direction else -1.0,
            util,
            pressure_norm,
            waiting_total / 500.0,
            float(freq.get('freq_low_demand', 0.0)),
            low_forecast,
            10.0 * high_energy,
            10.0 * middle_energy,
            promotion,
            float(freq.get('freq_promotion_absorbed', 0.0)),
            prev['wait'] / 10.0,
            prev['cv'],
            prev['overshoot_norm'],
            prev['composite'],
            prev['terminal_shift'] / 45.0,
            prev['lower_action'] / 5.0,
            prev['lower_drift'],
            compression / self.timetable_headway_value_planner_delta_norm_s,
            relief / self.timetable_headway_value_planner_delta_norm_s,
            util * compression / self.timetable_headway_value_planner_delta_norm_s,
            util * relief / self.timetable_headway_value_planner_delta_norm_s,
            low_forecast * compression / self.timetable_headway_value_planner_delta_norm_s,
            high_energy * compression / self.timetable_headway_value_planner_delta_norm_s,
            pressure_norm * relief / self.timetable_headway_value_planner_delta_norm_s,
        ]
        clip = self.timetable_headway_value_planner_feature_clip
        state_part = np.asarray(s_upper, dtype=np.float64).reshape(-1)
        if clip > 0.0:
            state_part = np.clip(state_part, -clip, clip)
        x = np.concatenate([
            np.asarray(values, dtype=np.float64),
            action_basis,
            plan,
            state_part,
        ])
        if clip > 0.0 and x.size > 1:
            x[1:] = np.clip(x[1:], -clip, clip)
        return x

    def _headway_value_planner_prior_cost(
            self, action_vec, direction, trip=None, plan_origin_launch=None):
        if self.timetable_headway_value_planner_prior_weight <= 0.0:
            return 0.0
        action = np.asarray(action_vec, dtype=np.float64).reshape(-1)
        block = np.asarray(
            action[self._upper_residual_selector_slice(action, direction)],
            dtype=np.float64).reshape(-1)
        if block.size == 0:
            block = action
        mean_delta = float(block.mean()) if block.size else 0.0
        compression = max(0.0, -mean_delta) / (
            self.timetable_headway_value_planner_delta_norm_s)
        relief = max(0.0, mean_delta) / (
            self.timetable_headway_value_planner_delta_norm_s)

        prev = self._headway_value_planner_prev_metrics()
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        _, n_fleet, pressure = self._fleet_pressure()
        n_fleet = max(float(n_fleet), 1.0)
        pressure_norm = float(pressure) / n_fleet
        spacing_pressure = (
            self.timetable_headway_value_planner_spacing_weight
            * max(0.0, prev['cv']
                  - self.timetable_headway_value_planner_cv_target)
            / 0.02)
        spacing_pressure += (
            self.timetable_headway_value_planner_spacing_weight
            * max(0.0, prev['overshoot_norm']
                  - self.timetable_headway_value_planner_overshoot_target)
            / 0.05)
        spacing_pressure += (
            self.timetable_headway_value_planner_fleet_weight
            * max(0.0, pressure_norm))
        wait_pressure = (
            self.timetable_headway_value_planner_wait_weight
            * max(0.0, prev['wait']
                  - self.timetable_headway_value_planner_wait_target_min)
            / 3.0)
        wait_pressure += (
            0.25 * self.timetable_headway_value_planner_wait_weight
            * max(0.0, float(freq.get('freq_low_forecast', 0.0))))
        shift_deficit = 0.0
        if self.timetable_headway_value_planner_terminal_shift_target_s > 0.0:
            shift_deficit = max(
                0.0,
                self.timetable_headway_value_planner_terminal_shift_target_s
                - prev['terminal_shift'])
            shift_deficit /= max(
                self.timetable_headway_value_planner_terminal_shift_target_s,
                1.0)

        prior = (
            compression * spacing_pressure
            + relief * wait_pressure
            - 0.5 * relief * (spacing_pressure + shift_deficit)
            - 0.25 * compression * wait_pressure)
        return float(np.clip(prior, -4.0, 4.0))

    def _ensure_headway_value_planner_model(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.timetable_headway_value_planner_A is None:
            dim = int(x.size)
            self.timetable_headway_value_planner_A = (
                self.timetable_headway_value_planner_ridge
                * np.eye(dim, dtype=np.float64))
            self.timetable_headway_value_planner_b = np.zeros(
                dim, dtype=np.float64)
            return True
        return int(self.timetable_headway_value_planner_A.shape[0]) == int(x.size)

    def _headway_value_planner_theta(self, x):
        if not self._ensure_headway_value_planner_model(x):
            return None
        try:
            return np.linalg.solve(
                self.timetable_headway_value_planner_A,
                self.timetable_headway_value_planner_b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(
                self.timetable_headway_value_planner_A,
                self.timetable_headway_value_planner_b,
                rcond=None)[0]

    def _headway_value_planner_target_cost(
            self, episode_composite_cost, transition_reward=None,
            local_credit_cost=None):
        mode = str(
            self.timetable_headway_value_planner_target
        ).strip().lower()
        episode_cost = float(episode_composite_cost)
        reward_cost = (
            -float(transition_reward)
            if transition_reward is not None else episode_cost)
        local_cost = (
            float(local_credit_cost)
            if local_credit_cost is not None else reward_cost)

        if mode in {'episode', 'composite', 'episode_composite'}:
            cost = episode_cost
        elif mode in {'reward', 'transition_reward', 'reward_cost'}:
            cost = self.timetable_headway_value_planner_reward_weight * reward_cost
        elif mode in {'local', 'local_credit', 'credit', 'local_credit_cost'}:
            cost = self.timetable_headway_value_planner_local_weight * local_cost
        elif mode in {'blend', 'blended', 'episode_local', 'local_blend'}:
            cost = (
                self.timetable_headway_value_planner_episode_weight
                * episode_cost
                + self.timetable_headway_value_planner_local_weight
                * local_cost)
        elif mode in {'episode_reward', 'reward_blend'}:
            cost = (
                self.timetable_headway_value_planner_episode_weight
                * episode_cost
                + self.timetable_headway_value_planner_reward_weight
                * reward_cost)
        else:
            cost = episode_cost
        return float(cost)

    def _headway_value_planner_gate_pass(self):
        if not self.timetable_headway_value_planner_gate_enable:
            return True
        if getattr(self.env, 'frequency_tracker', None) is None:
            return False
        try:
            freq = self.env.frequency_summary()
        except Exception:
            return False

        def _group_pass(group):
            checks = []

            def _add_min(name, summary_key):
                value = group.get(name)
                if value is not None:
                    checks.append(
                        float(freq.get(summary_key, 0.0)) >= float(value))

            def _add_max(name, summary_key):
                value = group.get(name)
                if value is not None:
                    checks.append(
                        float(freq.get(summary_key, 0.0)) <= float(value))

            _add_min('min_low_forecast', 'freq_low_forecast')
            _add_max('max_low_forecast', 'freq_low_forecast')
            _add_min('min_high_energy', 'freq_high_energy')
            _add_max('max_high_energy', 'freq_high_energy')
            _add_min('min_middle_energy', 'freq_middle_energy')
            _add_max('max_middle_energy', 'freq_middle_energy')
            _add_min('min_od_entropy', 'freq_od_entropy')
            _add_max('max_od_entropy', 'freq_od_entropy')
            _add_max('max_promotion_strength', 'freq_promotion_strength')
            return bool(checks and all(checks))

        base_group = {
            'min_low_forecast':
                self.timetable_headway_value_planner_gate_min_low_forecast,
            'max_low_forecast':
                self.timetable_headway_value_planner_gate_max_low_forecast,
            'min_high_energy':
                self.timetable_headway_value_planner_gate_min_high_energy,
            'max_high_energy':
                self.timetable_headway_value_planner_gate_max_high_energy,
            'min_middle_energy':
                self.timetable_headway_value_planner_gate_min_middle_energy,
            'max_middle_energy':
                self.timetable_headway_value_planner_gate_max_middle_energy,
            'min_od_entropy':
                self.timetable_headway_value_planner_gate_min_od_entropy,
            'max_od_entropy':
                self.timetable_headway_value_planner_gate_max_od_entropy,
            'max_promotion_strength':
                self.timetable_headway_value_planner_gate_max_promotion_strength,
        }
        if _group_pass(base_group):
            return True
        return any(
            _group_pass(group)
            for group in self.timetable_headway_value_planner_gate_any_of)

    def _update_headway_value_planner(self, x, cost):
        if (not self.timetable_headway_value_planner_enable
                or x is None):
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not self._ensure_headway_value_planner_model(x):
            return
        cost = float(cost)
        if self.timetable_headway_value_planner_cost_clip > 0.0:
            cost = float(np.clip(
                cost,
                -self.timetable_headway_value_planner_cost_clip,
                self.timetable_headway_value_planner_cost_clip))
        self._ep_headway_value_planner_target_costs.append(cost)
        self.timetable_headway_value_planner_A += np.outer(x, x)
        self.timetable_headway_value_planner_b += cost * x
        self.timetable_headway_value_planner_updates += 1

    def _select_headway_value_plan_action(
            self, s_upper, action_vec, direction, trip=None,
            plan_origin_launch=None):
        actor = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if not self.timetable_headway_value_planner_enable:
            return actor, None
        actor_x = self._headway_value_planner_features(
            s_upper, actor, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        self._ep_headway_value_planner_feature_norms.append(
            float(np.linalg.norm(actor_x)))
        gate_pass = self._headway_value_planner_gate_pass()
        if (not gate_pass
                or int(self._current_ep) < int(
                self.timetable_headway_value_planner_start_ep)
                or self.timetable_headway_value_planner_updates
                < self.timetable_headway_value_planner_min_observations):
            self._ep_headway_value_planner_active.append(0.0)
            self._ep_headway_value_planner_adjusts.append(0.0)
            self._ep_headway_value_planner_deltas.append(
                float(actor[self._upper_residual_selector_slice(
                    actor, direction)].mean()))
            self._ep_headway_value_planner_margins.append(0.0)
            self._ep_headway_value_planner_actor_preds.append(0.0)
            self._ep_headway_value_planner_selected_preds.append(0.0)
            self._ep_headway_value_planner_priors.append(0.0)
            return actor, actor_x

        theta = self._headway_value_planner_theta(actor_x)
        if theta is None:
            self._ep_headway_value_planner_active.append(0.0)
            self._ep_headway_value_planner_adjusts.append(0.0)
            self._ep_headway_value_planner_deltas.append(
                float(actor[self._upper_residual_selector_slice(
                    actor, direction)].mean()))
            self._ep_headway_value_planner_margins.append(0.0)
            self._ep_headway_value_planner_actor_preds.append(0.0)
            self._ep_headway_value_planner_selected_preds.append(0.0)
            self._ep_headway_value_planner_priors.append(0.0)
            return actor, actor_x

        actor_pred = float(np.dot(actor_x, theta))
        actor_prior = self._headway_value_planner_prior_cost(
            actor, direction, trip=trip,
            plan_origin_launch=plan_origin_launch)
        scored = []
        for _, cand in self._headway_value_planner_candidates(actor, direction):
            x = self._headway_value_planner_features(
                s_upper, cand, direction, trip=trip,
                plan_origin_launch=plan_origin_launch)
            pred = float(np.dot(x, theta))
            adjust = float(np.mean(np.abs(cand - actor)))
            prior = self._headway_value_planner_prior_cost(
                cand, direction, trip=trip,
                plan_origin_launch=plan_origin_launch)
            score = (
                pred
                + self.timetable_headway_value_planner_adjust_penalty
                * adjust
                / self.timetable_headway_value_planner_adjust_norm_s
                + self.timetable_headway_value_planner_prior_weight * prior)
            scored.append((score, pred, adjust, prior, cand, x))

        random_probe = (
            self.timetable_headway_value_planner_epsilon > 0.0
            and self._headway_value_selector_rng.random()
            < self.timetable_headway_value_planner_epsilon)
        if random_probe and len(scored) > 1:
            chosen = scored[int(
                self._headway_value_selector_rng.randint(len(scored)))]
        else:
            chosen = min(scored, key=lambda item: item[0])
            actor_score = (
                actor_pred
                + self.timetable_headway_value_planner_prior_weight
                * actor_prior)
            if (actor_score - float(chosen[0])
                    < self.timetable_headway_value_planner_improve_margin):
                chosen = (actor_score, actor_pred, 0.0, actor_prior,
                          actor, actor_x)

        score, selected_pred, adjust, prior, selected, selected_x = chosen
        selected_block = selected[self._upper_residual_selector_slice(
            selected, direction)]
        self._ep_headway_value_planner_active.append(1.0)
        self._ep_headway_value_planner_adjusts.append(float(adjust))
        self._ep_headway_value_planner_deltas.append(
            float(np.asarray(selected_block, dtype=np.float64).mean()))
        self._ep_headway_value_planner_margins.append(
            float((actor_pred
                   + self.timetable_headway_value_planner_prior_weight
                   * actor_prior) - score))
        self._ep_headway_value_planner_actor_preds.append(actor_pred)
        self._ep_headway_value_planner_selected_preds.append(
            float(selected_pred))
        self._ep_headway_value_planner_priors.append(float(prior))
        return selected.astype(np.float32), selected_x

    def _counterfactual_action_domain_flags(self):
        text = self.exp_name.lower()
        domain = 'terminal'
        if 'highnoise' in text:
            domain = 'highnoise'
        elif 'odshift' in text:
            domain = 'odshift'
        elif 'rushshift' in text:
            domain = 'rushshift'
        return {
            'domain_is_terminal': 1.0 if domain == 'terminal' else 0.0,
            'domain_is_highnoise': 1.0 if domain == 'highnoise' else 0.0,
            'domain_is_odshift': 1.0 if domain == 'odshift' else 0.0,
            'domain_is_rushshift': 1.0 if domain == 'rushshift' else 0.0,
        }

    def _counterfactual_action_feature_values(self, s_upper, trip, direction):
        """Causal feature vector matching the trip-level CRN selector audit."""
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        concurrent, n_fleet, fleet_pressure = self._fleet_pressure()
        try:
            waiting_total = sum(
                len(st.waiting_passengers)
                for st in getattr(self.env, 'stations', []))
        except Exception:
            waiting_total = 0

        base_hw = float(getattr(trip, 'target_headway', 360.0))
        if self.timetable_planner is not None:
            base_hw = float(self.timetable_planner._base_headway(trip))
        current_launch = float(getattr(trip, 'launch_time', 0.0))
        hour = 6 + int(current_launch // 3600)
        period = 'peak' if (7 <= hour <= 9 or 17 <= hour <= 19) else (
            'off' if 9 < hour < 17 else 'trans')
        max_tid = max(
            1.0,
            float(max(
                [getattr(tt, 'launch_turn', 0)
                 for tt in getattr(self.env, 'timetables', [])] or [1])),
        )
        last_dispatch = float(getattr(
            self.env, '_last_dispatch_time', {}).get(bool(direction), -9999.0))
        now = float(getattr(self.env, 'current_time', current_launch))
        terminal_gap_now = now - last_dispatch
        if terminal_gap_now > 9000.0 or terminal_gap_now < 0.0:
            terminal_gap_now = base_hw
        terminal_short_gap = max(0.0, base_hw - terminal_gap_now)
        terminal_over_gap = max(0.0, terminal_gap_now - base_hw)
        s_arr = np.asarray(s_upper, dtype=np.float64).reshape(-1)

        values = {
            'hour_norm': float(hour) / 24.0,
            'tid_norm': float(getattr(trip, 'launch_turn', 0)) / max_tid,
            'ep_norm': float(self._current_ep) / self.cf_action_selector_ep_norm_denominator,
            'dir_signed': 1.0 if bool(direction) else -1.0,
            'period_is_peak': 1.0 if period == 'peak' else 0.0,
            'period_is_off': 1.0 if period == 'off' else 0.0,
            'period_is_trans': 1.0 if period == 'trans' else 0.0,
            'base_hw_norm': base_hw / 600.0,
            'eff_hw_norm': base_hw / 600.0,
            's_hold_mean_norm': float(s_arr[5]) if s_arr.size > 5 else 0.0,
            's_hold_std_norm': float(s_arr[6]) if s_arr.size > 6 else 0.0,
            'terminal_gap_now_norm': terminal_gap_now / 600.0,
            'terminal_short_gap_norm': terminal_short_gap / 600.0,
            'terminal_over_gap_norm': terminal_over_gap / 600.0,
            'fleet_concurrent_norm': float(concurrent) / 30.0,
            'fleet_target_norm': float(n_fleet) / 30.0,
            'fleet_pressure_norm': float(fleet_pressure) / 30.0,
            'waiting_total_norm': float(waiting_total) / 500.0,
            'freq_low_demand_ctx': float(freq.get('freq_low_demand', 0.0)),
            'freq_low_forecast_ctx': float(freq.get('freq_low_forecast', 0.0)),
            'freq_high_energy_ctx': float(freq.get('freq_high_energy', 0.0)),
            'freq_middle_energy_ctx': float(freq.get('freq_middle_energy', 0.0)),
            'freq_od_entropy_ctx': float(freq.get('freq_od_entropy', 0.0)),
            'freq_promotion_strength_ctx': float(freq.get('freq_promotion_strength', 0.0)),
            'freq_promotion_active_ctx': float(freq.get('freq_promotion_active', 0.0)),
        }
        values.update(self._counterfactual_action_domain_flags())
        return values

    def _counterfactual_action_spec(self, method):
        method = str(method).strip()
        if self.cf_action_selector_allowed_methods:
            if method not in self.cf_action_selector_allowed_methods:
                method = self.cf_action_selector_default_method
        if method not in ACTION_SPECS:
            method = 'target0'
        spec = ACTION_SPECS[method]
        return method, float(spec['delta_s']), bool(spec['terminal_dispatch'])

    def _select_counterfactual_action(
            self, s_upper, action_vec, direction, trip=None):
        actor = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if (not self.cf_action_selector_enable
                or self.cf_action_selector_model is None
                or int(self._current_ep) < int(self.cf_action_selector_start_ep)
                or trip is None):
            return actor, None
        values = self._counterfactual_action_feature_values(
            s_upper, trip, direction)
        pred = self.cf_action_selector_model.predict(values)
        method, delta_s, terminal_dispatch = self._counterfactual_action_spec(
            pred.method)
        selected = np.full(
            self.upper_action_dim, float(delta_s), dtype=np.float32)
        selected = np.clip(
            selected,
            self.upper_action_low,
            self.upper_action_high,
        ).astype(np.float32)
        actor_delta = float(np.asarray(actor, dtype=np.float64).mean())
        info = {
            'active': 1.0,
            'selected_method': method,
            'selected_delta_s': float(delta_s),
            'actor_delta_s': actor_delta,
            'changed': 1.0 if abs(actor_delta - float(delta_s)) > 1e-6 else 0.0,
            'terminal_dispatch': 1.0 if terminal_dispatch else 0.0,
            'confidence': float(pred.confidence),
            'node_id': int(pred.node_id),
        }
        self._ep_cf_action_selector_active.append(1.0)
        self._ep_cf_action_selector_changed.append(float(info['changed']))
        self._ep_cf_action_selector_terminal_dispatch.append(
            float(info['terminal_dispatch']))
        self._ep_cf_action_selector_deltas.append(float(delta_s))
        self._ep_cf_action_selector_confidences.append(float(pred.confidence))
        return selected, info

    def _lower_value_guard_signal(self, bus, action_s):
        if bus is None:
            return 0.0, 0.0, 0.0
        board_count = int(getattr(bus, 'last_board_count', 0))
        wait_sum_s = float(getattr(bus, 'last_board_wait_sum_s', 0.0))
        if board_count > 0:
            wait_norm = (wait_sum_s / max(board_count, 1)
                         / self.fleet_noharm_lower_value_guard_wait_norm_s)
            if self.fleet_noharm_lower_value_guard_wait_clip > 0.0:
                wait_norm = min(
                    wait_norm, self.fleet_noharm_lower_value_guard_wait_clip)
            board_norm = (
                board_count / self.fleet_noharm_lower_value_guard_board_norm)
            if self.fleet_noharm_lower_value_guard_board_clip > 0.0:
                board_norm = min(
                    board_norm, self.fleet_noharm_lower_value_guard_board_clip)
        else:
            wait_norm = 0.0
            board_norm = 0.0

        high_share = 0.0
        tracker = getattr(self.env, 'frequency_tracker', None)
        if tracker is not None:
            station_id = int(getattr(
                bus, 'last_board_station_id',
                getattr(getattr(bus, 'last_station', None), 'station_id', 0)))
            direction = bool(getattr(bus, 'direction', True))
            local_high = float(tracker.local_high_value(station_id, direction))
            local_low = 0.0
            if hasattr(tracker, 'local_low_value'):
                local_low = float(tracker.local_low_value(station_id, direction))
            high = (
                max(local_high, 0.0)
                if self.fleet_noharm_lower_value_guard_positive_high_only
                else abs(local_high))
            low = max(
                abs(local_low),
                self.fleet_noharm_lower_value_guard_low_floor)
            high_share = high / (high + low + 1e-9)
            cap = self.fleet_noharm_lower_value_guard_high_share_cap
            if cap >= 0.0:
                high_share = min(high_share, cap)
            high_share = float(np.clip(high_share, 0.0, 1.0))

        value = high_share * (
            wait_norm
            + self.fleet_noharm_lower_value_guard_board_weight * board_norm)
        headway_value = 0.0
        if self.fleet_noharm_lower_value_guard_headway_weight > 0.0:
            target_hw = max(float(getattr(bus, '_target_headway', 360.0)), 1.0)
            fwd = float(getattr(bus, 'forward_headway', target_hw))
            bwd = float(getattr(bus, 'backward_headway', target_hw))
            action_s = max(float(action_s), 0.0)
            before = abs(fwd - target_hw) + abs(bwd - target_hw)
            after = (
                abs((fwd + action_s) - target_hw)
                + abs((bwd - action_s) - target_hw))
            headway_value = max(0.0, (before - after) / target_hw)
            if self.fleet_noharm_lower_value_guard_headway_clip > 0.0:
                headway_value = min(
                    headway_value,
                    self.fleet_noharm_lower_value_guard_headway_clip)
            value += (
                self.fleet_noharm_lower_value_guard_headway_weight
                * headway_value)
        return float(value), float(high_share), float(headway_value)

    def _apply_lower_value_guard(self, action, bus=None):
        if not self.fleet_noharm_lower_value_guard_enable:
            return action
        original = self._lower_action_scalar(action)
        if original <= 0.0:
            self._ep_fleet_noharm_lower_value_guard_adjusts.append(0.0)
            self._ep_fleet_noharm_lower_value_guard_active.append(0.0)
            self._ep_fleet_noharm_lower_value_guard_values.append(0.0)
            self._ep_fleet_noharm_lower_value_guard_headway_values.append(0.0)
            self._ep_fleet_noharm_lower_value_guard_costs.append(0.0)
            return action
        gate_active = self._fleet_noharm_gate_active(
            self.fleet_noharm_lower_value_guard_gate)
        _, _, pressure = self._fleet_pressure()
        strength = self._pressure_strength(
            pressure,
            self.fleet_noharm_lower_value_guard_pressure_start,
            self.fleet_noharm_lower_value_guard_pressure_full,
        )
        value, _, headway_value = self._lower_value_guard_signal(bus, original)
        cost = (
            self.fleet_noharm_lower_value_guard_cost_weight
            * strength
            * original
            / self.fleet_noharm_lower_value_guard_action_norm_s)
        active = bool(gate_active and strength > 0.0 and cost > 0.0)
        self._ep_fleet_noharm_lower_value_guard_active.append(
            1.0 if active else 0.0)
        self._ep_fleet_noharm_lower_value_guard_values.append(float(value))
        self._ep_fleet_noharm_lower_value_guard_headway_values.append(
            float(headway_value))
        self._ep_fleet_noharm_lower_value_guard_costs.append(float(cost))
        if not active:
            self._ep_fleet_noharm_lower_value_guard_adjusts.append(0.0)
            return action

        required = self.fleet_noharm_lower_value_guard_min_ratio * cost
        if value >= required or required <= 1e-9:
            self._ep_fleet_noharm_lower_value_guard_adjusts.append(0.0)
            return action

        denom = (
            self.fleet_noharm_lower_value_guard_min_ratio
            * self.fleet_noharm_lower_value_guard_cost_weight
            * strength)
        allowed = (
            value
            * self.fleet_noharm_lower_value_guard_action_norm_s
            / max(denom, 1e-9))
        allowed = max(0.0, min(float(allowed), original))
        shrink = 1.0 - allowed / max(original, 1e-9)
        shrink = min(
            max(shrink, 0.0),
            self.fleet_noharm_lower_value_guard_max_shrink)
        adjusted = max(
            self.fleet_noharm_lower_value_guard_min_action_s,
            original * (1.0 - shrink))
        adjusted = self._quantize_lower_action(
            np.asarray([adjusted], dtype=np.float32))
        adjust = max(0.0, original - self._lower_action_scalar(adjusted))
        self._ep_fleet_noharm_lower_value_guard_adjusts.append(adjust)
        return adjusted

    def _lower_value_soft_cost(self, bus=None, action_s=0.0):
        if not self.fleet_noharm_lower_value_soft_cost_enable:
            return 0.0
        action_s = max(float(action_s), 0.0)
        if action_s <= 0.0:
            self._ep_fleet_noharm_lower_value_soft_costs.append(0.0)
            self._ep_fleet_noharm_lower_value_soft_active.append(0.0)
            self._ep_fleet_noharm_lower_value_soft_values.append(0.0)
            self._ep_fleet_noharm_lower_value_soft_headway_values.append(0.0)
            self._ep_fleet_noharm_lower_value_soft_risks.append(0.0)
            self._ep_fleet_noharm_lower_value_soft_violations.append(0.0)
            return 0.0

        gate_active = self._fleet_noharm_gate_active(
            self.fleet_noharm_lower_value_soft_cost_gate)
        _, _, pressure = self._fleet_pressure()
        strength = self._pressure_strength(
            pressure,
            self.fleet_noharm_lower_value_soft_cost_pressure_start,
            self.fleet_noharm_lower_value_soft_cost_pressure_full,
        )
        value, _, headway_value = self._lower_value_guard_signal(bus, action_s)
        risk = (
            self.fleet_noharm_lower_value_soft_cost_weight
            * strength
            * action_s
            / self.fleet_noharm_lower_value_soft_cost_action_norm_s)
        active = bool(gate_active and strength > 0.0 and risk > 0.0)
        required = self.fleet_noharm_lower_value_soft_cost_min_ratio * risk
        violation = max(0.0, required - value) if active else 0.0
        soft_cost = (
            self.fleet_noharm_lower_value_soft_cost_violation_weight
            * violation)
        if self.fleet_noharm_lower_value_soft_cost_cap > 0.0:
            soft_cost = min(
                soft_cost,
                self.fleet_noharm_lower_value_soft_cost_cap)

        self._ep_fleet_noharm_lower_value_soft_costs.append(
            float(soft_cost))
        self._ep_fleet_noharm_lower_value_soft_active.append(
            1.0 if active else 0.0)
        self._ep_fleet_noharm_lower_value_soft_values.append(float(value))
        self._ep_fleet_noharm_lower_value_soft_headway_values.append(
            float(headway_value))
        self._ep_fleet_noharm_lower_value_soft_risks.append(float(risk))
        self._ep_fleet_noharm_lower_value_soft_violations.append(
            float(violation))
        return float(soft_cost)

    def _apply_lower_fleet_noharm(self, action, bus=None):
        if not self.fleet_noharm_lower_enable:
            return self._apply_lower_value_guard(action, bus)
        original = self._lower_action_scalar(action)
        _, _, pressure = self._fleet_pressure()
        gate_active = self._fleet_noharm_gate_active(
            self.fleet_noharm_lower_gate)
        self._ep_fleet_noharm_lower_gate_active.append(
            1.0 if gate_active else 0.0)
        strength = self._pressure_strength(
            pressure,
            self.fleet_noharm_lower_pressure_start,
            self.fleet_noharm_lower_pressure_full,
        )
        self._ep_fleet_noharm_lower_pressures.append(max(0.0, pressure))
        base_shrink = (
            strength * self.fleet_noharm_lower_shrink_max
            if gate_active else 0.0)
        proactive_shrink = 0.0
        proactive_gate_active = False
        if self.fleet_noharm_lower_proactive_enable:
            proactive_gate_active = self._fleet_noharm_gate_active(
                self.fleet_noharm_lower_proactive_gate)
            if proactive_gate_active:
                proactive_strength = self._pressure_strength(
                    pressure,
                    self.fleet_noharm_lower_proactive_pressure_start,
                    self.fleet_noharm_lower_proactive_pressure_full,
                )
                proactive_shrink = (
                    proactive_strength
                    * self.fleet_noharm_lower_proactive_shrink_max)
        self._ep_fleet_noharm_lower_proactive_gate_active.append(
            1.0 if proactive_gate_active else 0.0)
        shrink = float(np.clip(max(base_shrink, proactive_shrink), 0.0, 1.0))
        if shrink <= 0.0:
            self._ep_fleet_noharm_lower_adjusts.append(0.0)
            self._ep_fleet_noharm_lower_proactive_adjusts.append(0.0)
            return self._apply_lower_value_guard(action, bus)
        adjusted = max(
            self.fleet_noharm_lower_min_action_s,
            original * (1.0 - shrink),
        )
        adjusted = np.asarray([adjusted], dtype=np.float32)
        adjusted = self._quantize_lower_action(adjusted)
        adjust = max(0.0, original - self._lower_action_scalar(adjusted))
        self._ep_fleet_noharm_lower_adjusts.append(adjust)
        self._ep_fleet_noharm_lower_proactive_adjusts.append(
            adjust if proactive_shrink >= base_shrink and proactive_shrink > 0.0
            else 0.0)
        return self._apply_lower_value_guard(adjusted, bus)

    def _record_frequency_hold_feedback(
            self, direction, local_high, action_s, wait_sum_s, boarded_count):
        """Track lower interventions under positive high-frequency demand."""
        if not self.freq_holdfb_enable:
            return
        pos_high = max(float(local_high), 0.0)
        if pos_high < self.freq_holdfb_high_threshold:
            pos_high = 0.0
        wait_norm = 0.0
        if boarded_count > 0 and wait_sum_s > 0.0:
            wait_norm = (
                float(wait_sum_s)
                / max(int(boarded_count), 1)
                / self.freq_holdfb_wait_norm_s)
            if self.freq_holdfb_wait_clip > 0.0:
                wait_norm = min(wait_norm, self.freq_holdfb_wait_clip)
        board_norm = int(boarded_count) / self.freq_holdfb_board_norm
        board_norm = min(board_norm, 2.0)
        self._freq_holdfb_events[bool(direction)].append((
            pos_high,
            max(float(action_s), 0.0) / 60.0,
            wait_norm,
            board_norm,
        ))

    def _frequency_hold_feedback_stats(self, direction):
        events = list(self._freq_holdfb_events[bool(direction)])
        if not events:
            return (0.0, 0.0)
        arr = np.asarray(events, dtype=np.float64)
        high = arr[:, 0]
        weight = high.sum()
        if weight <= 1e-9:
            return (0.0, 0.0)
        hf_hold = float(np.dot(high, arr[:, 1]) / weight)
        hf_wait = float(np.dot(high, arr[:, 2]) / weight)
        return (hf_hold, hf_wait)

    def _frequency_hold_feedback_features(self, direction):
        if not self.freq_holdfb_enable:
            return np.zeros(0, dtype=np.float32)
        same = self._frequency_hold_feedback_stats(direction)
        other = self._frequency_hold_feedback_stats(not bool(direction))
        feats = np.asarray([same[0], same[1], other[0], other[1]],
                           dtype=np.float32)
        self._ep_freq_holdfb_features.append(feats.copy())
        return feats

    def _frequency_drift_feedback_features(self, direction):
        if not self.freq_driftfb_enable:
            return np.zeros(0, dtype=np.float32)
        same = self._drift_feedback_pair(direction)
        other = self._drift_feedback_pair(not bool(direction))
        feats = np.asarray([same[0], same[1], other[0], other[1]],
                           dtype=np.float32)
        self._ep_freq_driftfb_features.append(feats.copy())
        return feats

    def _augment_lower_state(self, obs, last_action=0.0):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.lower_state_encoder is not None:
            obs = self.lower_state_encoder.encode(obs)
        if not self.lower_use_last_action_feature:
            return obs
        action_feature = float(last_action)
        if self.lower_state_encoder is not None:
            action_feature = self.lower_state_encoder.encode_action(
                action_feature)
        return np.concatenate(
            [obs, np.asarray([action_feature], dtype=np.float32)])

    def _lower_policy_action(self, obs, last_action=0.0, deterministic=False):
        if getattr(self, '_fixed_expert_active', False):
            return np.asarray([0.0], dtype=np.float32)
        state = self._augment_lower_state(obs, last_action)
        action = self.lower_trainer.policy_net.get_action(
            torch.from_numpy(state).float().to(self.device),
            deterministic=deterministic)
        return self._quantize_lower_action(action)

    def _bus_for_agent(self, bus_id):
        bus_id = int(bus_id)
        return next(
            (bus for bus in self.env.bus_all
             if int(getattr(bus, 'bus_id', -1)) == bus_id),
            None,
        )

    def _lower_terminal_action_masked(self, bus):
        if self.lower_terminal_action_mode != 'mask' or bus is None:
            return False
        next_station = getattr(bus, 'next_station', None)
        return int(getattr(next_station, 'station_type', 1)) == 0

    def _lower_action_for_agent(
            self, obs, bus_id, last_action=0.0, deterministic=False):
        bus = self._bus_for_agent(bus_id)
        if self._lower_terminal_action_masked(bus):
            self._ep_lower_terminal_action_masks += 1
            return np.asarray([0.0], dtype=np.float32)
        action = self._lower_policy_action(
            obs, last_action=last_action, deterministic=deterministic)
        action = self._apply_causal_holding_guard(action, bus)
        return self._apply_lower_fleet_noharm(action, bus)

    def _apply_causal_holding_guard(self, action, bus=None):
        requested = self._lower_action_scalar(action)
        departure_mode = (
            self.lower_causal_holding_guard.evidence_mode
            == 'pre_action_departure_v6')
        if departure_mode:
            observed_headway = (
                getattr(bus, 'pre_action_forward_headway', None)
                if bus is not None else None)
            evidence_valid = bool(
                bus is not None
                and getattr(bus, 'pre_action_forward_headway_source', None)
                == 'matched_departure_event')
        else:
            observed_headway = (
                getattr(bus, 'forward_headway', None)
                if bus is not None else None)
            evidence_valid = bool(
                bus is not None
                and getattr(bus, 'forward_headway_source', None)
                == 'arrival_event')
        result = self.lower_causal_holding_guard.evaluate(
            requested,
            forward_headway_s=observed_headway,
            target_headway_s=(
                getattr(bus, '_target_headway', None) if bus is not None else None),
            evidence_valid=evidence_valid,
        )
        allowed = float(result.allowed_s)
        if (self.lower_action_bins is not None
                and not self.lower_action_bins_gate_enabled):
            feasible = self.lower_action_bins[
                self.lower_action_bins <= allowed + 1e-7]
            allowed = float(feasible[-1]) if feasible.size else 0.0
        adjusted = np.asarray([allowed], dtype=np.float32)
        self._ep_lower_causal_guard_active.append(
            1.0 if requested - allowed > 1e-9 else 0.0)
        self._ep_lower_causal_guard_limits.append(float(result.limit_s))
        self._ep_lower_causal_guard_adjustments.append(
            max(0.0, requested - allowed))
        return adjusted

    def _record_lower_transition(
            self, *, key, raw_state, raw_next_state, action, reward, cost,
            previous_action, transition_done, learned_training, bus=None,
            trip_id=None, direction=None, station_id=None,
            board_wait_sum_s=None, board_lf_wait_sum_s=None,
            board_hf_wait_sum_s=None, board_lf_mass=None,
            board_hf_mass=None, board_count=None,
            record_holding_action=True):
        """Shape, diagnose, and optionally replay one physical lower transition."""
        raw_state = np.asarray(raw_state, dtype=np.float32)
        act_val = (
            self._lower_action_scalar(action) if action is not None else 0.0)
        state = self._augment_lower_state(raw_state, previous_action)
        if raw_next_state is None:
            next_state = np.zeros_like(state, dtype=np.float32)
        else:
            next_state = self._augment_lower_state(
                np.asarray(raw_next_state, dtype=np.float32), act_val)

        context_bus = bus
        if context_bus is None:
            context_bus = self._bus_for_agent(int(raw_state[0]))
        cur_tid = int(trip_id) if trip_id is not None else -1
        cur_dir = bool(direction) if direction is not None else True
        if context_bus is not None:
            if trip_id is None:
                cur_tid = int(getattr(context_bus, 'trip_id', -1))
            if direction is None:
                cur_dir = bool(getattr(context_bus, 'direction', True))

        self._ep_lower_terminal_transitions += int(transition_done)
        drift_load = self._lower_drift_load(
            cur_dir,
            act_val,
            cur_tid,
            action_already_recorded=not record_holding_action,
        )
        drift_penalty = self._lower_drift_penalty(drift_load)
        drift_cost = self._lower_drift_cost(drift_load)
        lower_value_soft_cost = self._lower_value_soft_cost(
            context_bus, act_val)
        total_cost = float(cost) + drift_cost + lower_value_soft_cost

        low_demand = 0.0
        local_high = 0.0
        credit_high = 0.0
        hold_credit_high = 0.0
        local_low = None
        station_id_value = -1 if station_id is None else int(station_id)
        freq_summary = None
        tracker = getattr(self.env, 'frequency_tracker', None)
        if tracker is not None:
            freq_summary = tracker.summary()
            low_demand = float(freq_summary.get('freq_low_demand', 0.0))
            if context_bus is not None:
                if station_id is None:
                    station_id_value = int(getattr(
                        context_bus, 'last_board_station_id',
                        getattr(getattr(context_bus, 'last_station', None),
                                'station_id', 0)))
                local_high = tracker.local_high_value(
                    station_id_value, cur_dir)
                raw_high = None
                if hasattr(tracker, 'local_high_raw_value'):
                    raw_high = tracker.local_high_raw_value(
                        station_id_value, cur_dir)
                credit_high, raw_weight = self._select_lower_high_credit(
                    local_high,
                    raw_high,
                    freq_summary,
                    self.freq_wait_lower_high_source,
                )
                hold_credit_high, _ = self._select_lower_high_credit(
                    local_high,
                    raw_high,
                    freq_summary,
                    self.freq_wait_lower_hold_high_source,
                )
                self._ep_freq_wait_lower_raw_credit_weights.append(raw_weight)
                local_low = tracker.local_low_value(
                    station_id_value, cur_dir)

        high_hold_penalty = self._lower_high_hold_penalty(
            hold_credit_high,
            low_demand if local_low is None else local_low,
            act_val,
        )
        load_hold_penalty, load_ratio, normalized_person_delay = (
            self.lower_load_holding_penalty.evaluate(
                raw_state,
                act_val,
                base_state_dim=int(self.env._base_state_dim),
                context_features=self.env.lower_context_features,
            )
        )
        if board_wait_sum_s is None:
            board_wait_sum_s = float(getattr(
                context_bus, 'last_board_wait_sum_s', 0.0)
                if context_bus is not None else 0.0)
        if board_count is None:
            board_count = int(getattr(
                context_bus, 'last_board_count', 0)
                if context_bus is not None else 0)
        if board_lf_wait_sum_s is None:
            board_lf_wait_sum_s = float(getattr(
                context_bus, 'last_board_lf_wait_sum_s', 0.0)
                if context_bus is not None else 0.0)
        if board_hf_wait_sum_s is None:
            board_hf_wait_sum_s = float(getattr(
                context_bus, 'last_board_hf_wait_sum_s', 0.0)
                if context_bus is not None else 0.0)
        if board_lf_mass is None:
            board_lf_mass = float(getattr(
                context_bus, 'last_board_lf_mass', 0.0)
                if context_bus is not None else 0.0)
        if board_hf_mass is None:
            board_hf_mass = float(getattr(
                context_bus, 'last_board_hf_mass', 0.0)
                if context_bus is not None else 0.0)
        wait_penalty = self._record_frequency_wait_credit(
            cur_tid,
            float(board_wait_sum_s),
            int(board_count),
            low_demand,
            credit_high,
            local_low,
            freq_summary,
            lf_wait_sum_s=float(board_lf_wait_sum_s),
            hf_wait_sum_s=float(board_hf_wait_sum_s),
            lf_mass=float(board_lf_mass),
            hf_mass=float(board_hf_mass),
        )
        self._record_frequency_hold_feedback(
            cur_dir,
            credit_high,
            act_val,
            float(board_wait_sum_s),
            int(board_count),
        )
        shaped_reward = (
            float(reward)
            - drift_penalty
            - wait_penalty
            - high_hold_penalty
            - load_hold_penalty
        )

        self._ep_lower_actions.append(act_val)
        self._ep_lower_actions_by_dir[cur_dir].append(act_val)
        self._ep_lower_rewards.append(shaped_reward)
        self._ep_lower_drift_penalties.append(drift_penalty)
        self._ep_lower_drift_costs.append(drift_cost)
        self._ep_lower_drift_loads.append(drift_load)
        self._ep_lower_load_hold_penalties.append(load_hold_penalty)
        self._ep_lower_load_ratios.append(load_ratio)
        self._ep_lower_normalized_person_delays.append(
            normalized_person_delay)
        if cur_tid >= 0 and record_holding_action:
            self.holding_feedback.record_action(cur_tid, act_val)
        if tracker is not None:
            self._ep_lower_demand_action.append((
                low_demand,
                credit_high,
                act_val,
            ))
            self._ep_shock_response_events.append({
                'time_s': float(getattr(self.env, 'current_time', 0.0)),
                'station_id': station_id_value,
                'direction': bool(cur_dir),
                'high': credit_high,
                'action_s': act_val,
            })

        if learned_training:
            global_tid = (
                self._current_ep * 1000 + cur_tid
                if cur_tid >= 0 else int(raw_state[0]))
            self.replay_buffer.push(
                state,
                action,
                shaped_reward,
                total_cost,
                next_state,
                transition_done,
                global_tid,
            )
        return shaped_reward, total_cost, act_val

    def _fixed_headway_callback(self, s_upper_v1, trip):
        if self.fixed_selector_strict_headway_s is not None:
            headway = float(self.fixed_selector_strict_headway_s)
            trip._freqduet_base_target_headway = headway
            return headway
        base_hw = float(getattr(trip, 'target_headway', 360.0))
        if not hasattr(trip, '_freqduet_base_target_headway'):
            trip._freqduet_base_target_headway = base_hw
        return float(getattr(trip, '_freqduet_base_target_headway', base_hw))

    def _select_fixed_expert_for_episode(self, ep, training=True):
        rule_decision = self._fixed_selector_rule_decision(
            ep, training=training)
        if rule_decision is not None:
            return bool(rule_decision)
        if self.fixed_selector_context_enable:
            return self._select_fixed_expert_contextual(ep, training=training)
        if not (self.fixed_selector_enable and training):
            return False
        if int(ep) < int(self.fixed_selector_start_ep):
            return False
        fixed_count = self.fixed_selector_counts['fixed']
        learned_count = self.fixed_selector_counts['learned']
        if fixed_count < self.fixed_selector_min_observations:
            return True
        if learned_count < self.fixed_selector_min_observations:
            return False
        if (self.fixed_selector_probe_period > 0
                and (int(ep) - int(self.fixed_selector_start_ep))
                % self.fixed_selector_probe_period == 0):
            probe_index = (
                (int(ep) - int(self.fixed_selector_start_ep))
                // self.fixed_selector_probe_period)
            if self.fixed_selector_probe_mode in {'alternate', 'balanced'}:
                return bool(probe_index % 2 == 0)
            if self.fixed_selector_probe_mode in {'learned', 'learned_only'}:
                return False
            return True
        if self.fixed_selector_epsilon > 0.0:
            if self._fixed_expert_selector_rng.random() < self.fixed_selector_epsilon:
                return bool(self._fixed_expert_selector_rng.random() < 0.5)
        learned_cost = self.fixed_selector_cost_ema.get('learned')
        fixed_cost = self.fixed_selector_cost_ema.get('fixed')
        if learned_cost is None or fixed_cost is None:
            return False
        return bool(fixed_cost <= learned_cost + self.fixed_selector_margin)

    def _update_fixed_expert_selector(self, fixed_active, composite_cost):
        if not self.fixed_selector_enable:
            return
        key = 'fixed' if fixed_active else 'learned'
        cost = float(composite_cost)
        prev = self.fixed_selector_cost_ema.get(key)
        if prev is None:
            self.fixed_selector_cost_ema[key] = cost
        else:
            alpha = self.fixed_selector_ema_alpha
            self.fixed_selector_cost_ema[key] = (
                alpha * cost + (1.0 - alpha) * float(prev))
        self.fixed_selector_counts[key] += 1
        if self.fixed_selector_context_enable:
            x = self._fixed_selector_current_context
            if x is None:
                x = self._fixed_selector_context_vector()
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            self.fixed_selector_context_A[key] += np.outer(x, x)
            self.fixed_selector_context_b[key] += cost * x

    def _fixed_selector_peak_shift_abs(self):
        choices = getattr(self.env, 'peak_shift_choices', None)
        if not choices:
            return 0.0
        vals = np.asarray(choices, dtype=np.float64).reshape(-1)
        probs = getattr(self.env, 'peak_shift_probs', None)
        if probs is not None:
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
            if probs.size == vals.size and probs.sum() > 0:
                probs = probs / probs.sum()
                return float(np.sum(np.abs(vals) * probs) / 2.0)
        return float(np.mean(np.abs(vals)) / 2.0)

    def _fixed_selector_context_values(self):
        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        od_clip = getattr(self.env, 'od_noise_clip', [1.0, 1.0])
        try:
            od_clip_width = (
                abs(float(od_clip[1]) - float(od_clip[0])) / 2.0
                if len(od_clip) >= 2 else 0.0)
        except (TypeError, ValueError):
            od_clip_width = 0.0
        n_fleet = max(float(_prev('N_fleet', self.N_fleet_default)), 1.0)
        values = {
            'bias': 1.0,
            'cfg_demand_noise': float(getattr(self.env, 'demand_noise', 0.0)),
            'cfg_od_noise': float(getattr(self.env, 'od_noise', 0.0)),
            'cfg_od_clip_width': od_clip_width,
            'cfg_peak_shift_abs': self._fixed_selector_peak_shift_abs(),
            'prev_freq_low_demand': _prev('freq_low_demand'),
            'prev_freq_low_forecast': _prev('freq_low_forecast'),
            'prev_freq_high_energy': 10.0 * _prev('freq_high_energy'),
            'prev_freq_middle_energy': 10.0 * _prev('freq_middle_energy'),
            'prev_freq_od_entropy': _prev('freq_od_entropy'),
            'prev_freq_od_high_energy': 10.0 * _prev('freq_od_high_energy'),
            'prev_freq_promotion_strength':
                _prev('freq_promotion_strength'),
            'prev_freq_promotion_absorbed':
                _prev('freq_promotion_absorbed'),
            'prev_upper_hf_power_ratio': _prev('upper_hf_power_ratio'),
            'prev_lower_lf_drift_ratio': _prev('lower_lf_drift_ratio'),
            'prev_wait_norm': _prev('avg_wait_min') / 10.0,
            'prev_overshoot_norm': _prev('fleet_overshoot') / n_fleet,
            'prev_headway_cv': _prev('headway_cv'),
            'prev_terminal_shift_norm':
                _prev('terminal_launch_shift_mean') / 60.0,
            'prev_lower_drift_cost': _prev('lower_drift_cost_mean'),
        }
        return values

    def _fixed_selector_context_vector(self):
        values = self._fixed_selector_context_values()
        x = np.asarray([
            values[name] for name in self.fixed_selector_context_features
        ], dtype=np.float64)
        clip = self.fixed_selector_context_feature_clip
        if clip > 0.0 and x.size > 1:
            x[1:] = np.clip(x[1:], -clip, clip)
        return x

    def _fixed_selector_rule_group_pass(self, group, values):
        if not group:
            return False
        checks = []
        for key, threshold in group.items():
            if threshold is None:
                continue
            key = str(key)
            if key.startswith('min_'):
                name = key[4:]
                checks.append(
                    float(values.get(name, 0.0)) >= float(threshold))
            elif key.startswith('max_'):
                name = key[4:]
                checks.append(
                    float(values.get(name, 0.0)) <= float(threshold))
            elif key in values:
                checks.append(
                    abs(float(values.get(key, 0.0)) - float(threshold))
                    <= 1e-9)
        return bool(checks and all(checks))

    def _fixed_selector_rule_decision(self, ep, training=True):
        if not (
                self.fixed_selector_rule_enable
                and self.fixed_selector_enable
                and training):
            return None
        if int(ep) < int(self.fixed_selector_start_ep):
            return False
        values = self._fixed_selector_context_values()
        if any(
                self._fixed_selector_rule_group_pass(group, values)
                for group in self.fixed_selector_rule_fixed_when):
            return True
        if any(
                self._fixed_selector_rule_group_pass(group, values)
                for group in self.fixed_selector_rule_learned_when):
            return False
        return self.fixed_selector_rule_default in {
            'fixed', 'fixed_headway', 'expert'}

    def _fixed_selector_predict_context_cost(self, key, x):
        A = self.fixed_selector_context_A[key]
        b = self.fixed_selector_context_b[key]
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, b, rcond=None)[0]
        return float(np.dot(np.asarray(x, dtype=np.float64), theta))

    def _select_fixed_expert_contextual(self, ep, training=True):
        rule_decision = self._fixed_selector_rule_decision(
            ep, training=training)
        if rule_decision is not None:
            self._fixed_selector_current_context = (
                self._fixed_selector_context_vector())
            self._fixed_selector_context_learned_value = 0.0
            self._fixed_selector_context_fixed_value = 0.0
            self._fixed_selector_context_margin = (
                -1.0 if bool(rule_decision) else 1.0)
            return bool(rule_decision)
        self._fixed_selector_current_context = (
            self._fixed_selector_context_vector())
        x = self._fixed_selector_current_context
        learned_pred = self._fixed_selector_predict_context_cost('learned', x)
        fixed_pred = self._fixed_selector_predict_context_cost('fixed', x)
        self._fixed_selector_context_learned_value = learned_pred
        self._fixed_selector_context_fixed_value = fixed_pred
        self._fixed_selector_context_margin = fixed_pred - learned_pred
        if not (self.fixed_selector_enable and training):
            return False
        if int(ep) < int(self.fixed_selector_start_ep):
            return False
        fixed_count = self.fixed_selector_counts['fixed']
        learned_count = self.fixed_selector_counts['learned']
        if fixed_count < self.fixed_selector_min_observations:
            return True
        if learned_count < self.fixed_selector_min_observations:
            return False
        if (self.fixed_selector_probe_period > 0
                and (int(ep) - int(self.fixed_selector_start_ep))
                % self.fixed_selector_probe_period == 0):
            probe_index = (
                (int(ep) - int(self.fixed_selector_start_ep))
                // self.fixed_selector_probe_period)
            if self.fixed_selector_probe_mode in {'alternate', 'balanced'}:
                return bool(probe_index % 2 == 0)
            if self.fixed_selector_probe_mode in {'learned', 'learned_only'}:
                return False
            return True
        if self.fixed_selector_epsilon > 0.0:
            if self._fixed_expert_selector_rng.random() < self.fixed_selector_epsilon:
                return bool(self._fixed_expert_selector_rng.random() < 0.5)
        return bool(fixed_pred <= learned_pred + self.fixed_selector_margin)

    def _fixed_selector_update_start_ep(self):
        if self.fixed_selector_count_start in {'selector', 'selector_start',
                                               'start'}:
            return max(int(self.upper_warmup),
                       int(self.fixed_selector_start_ep))
        return int(self.upper_warmup)

    def _upper_delta_hf_penalty(self, direction, delta_t, prev_delta_by_dir):
        """Penalty for high-frequency upper target-headway oscillation."""
        if not self.leakage_enable or self.upper_hf_penalty <= 0:
            return 0.0
        direction = bool(direction)
        delta_t = float(delta_t)
        if direction not in prev_delta_by_dir:
            prev_delta_by_dir[direction] = delta_t
            return 0.0
        diff = abs(delta_t - prev_delta_by_dir[direction])
        prev_delta_by_dir[direction] = delta_t
        return float(self.upper_hf_penalty * diff / max(self.delta_max, 1e-6))

    def _upper_residual_value_cost(self, effective_delta):
        """Penalty for upper headway compression without enough HF demand value."""
        if (not self.upper_residual_value_cost_enable
                or self.upper_residual_value_cost_weight <= 0.0):
            return 0.0, 0.0
        negative_residual = max(0.0, -float(effective_delta))
        if negative_residual <= 0.0:
            return 0.0, 0.0
        concurrent, n_fleet, _ = self._fleet_pressure()
        fleet_util = concurrent / max(float(n_fleet), 1.0)
        fleet_strength = self._pressure_strength(
            fleet_util,
            self.upper_residual_value_cost_fleet_util_start,
            self.upper_residual_value_cost_fleet_util_full)
        if fleet_strength <= 0.0:
            return 0.0, 0.0
        freq_summary = self.env.frequency_summary()
        high_energy = max(float(
            freq_summary.get('freq_high_energy', 0.0)), 0.0)
        high_value = self._pressure_strength(
            high_energy,
            self.upper_residual_value_cost_high_start,
            self.upper_residual_value_cost_high_full)
        promotion_value = max(float(
            freq_summary.get('freq_promotion_strength', 0.0)), 0.0)
        relief = min(
            1.0,
            high_value
            + self.upper_residual_value_cost_promotion_relief
            * promotion_value)
        risk_gate = max(0.0, 1.0 - relief)
        action_risk = (
            negative_residual
            / self.upper_residual_value_cost_action_norm_s)
        cost = (
            self.upper_residual_value_cost_weight
            * action_risk
            * fleet_strength
            * risk_gate)
        return float(cost), float(1.0 if cost > 1e-9 else 0.0)

    def _terminal_shift_max_for_frequency(self):
        if self.timetable_terminal_hf_shift_max_s is None:
            return None
        if getattr(self.env, 'frequency_tracker', None) is None:
            return None
        freq_summary = self.env.frequency_summary()
        if (float(freq_summary.get('freq_high_energy', 0.0))
                < self.timetable_terminal_hf_energy_min):
            return None
        return float(self.timetable_terminal_hf_shift_max_s)

    def _terminal_shift_min_for_frequency(
            self, action_vec=None, current_delta_s=None):
        """Causal adaptive early-release cap for executable terminal dispatch."""
        if not self.timetable_terminal_early_release_enable:
            return None
        base_min = float(self.timetable_terminal_early_release_base_min_s)
        relaxed_min = float(
            self.timetable_terminal_early_release_relaxed_min_s)
        if getattr(self.env, 'frequency_tracker', None) is None:
            return base_min
        freq_summary = self.env.frequency_summary()
        if (float(freq_summary.get('freq_updates', 0.0))
                < self.timetable_terminal_early_release_min_updates):
            return base_min

        checks = []

        def _add_max(value, summary_key):
            if value is not None:
                checks.append(
                    float(freq_summary.get(summary_key, 0.0)) <= float(value))

        def _add_min(value, summary_key):
            if value is not None:
                checks.append(
                    float(freq_summary.get(summary_key, 0.0)) >= float(value))

        _add_max(
            self.timetable_terminal_early_release_max_high_energy,
            'freq_high_energy')
        _add_max(
            self.timetable_terminal_early_release_max_middle_energy,
            'freq_middle_energy')
        _add_min(
            self.timetable_terminal_early_release_min_od_entropy,
            'freq_od_entropy')
        _add_max(
            self.timetable_terminal_early_release_max_od_high_energy,
            'freq_od_high_energy')
        _add_max(
            self.timetable_terminal_early_release_max_low_forecast,
            'freq_low_forecast')
        if (self.timetable_terminal_early_release_min_action_mean_s
                is not None and action_vec is not None):
            action_mean = float(np.mean(
                np.asarray(action_vec, dtype=np.float32).reshape(-1)))
            checks.append(
                action_mean >= float(
                    self.timetable_terminal_early_release_min_action_mean_s))
        if (self.timetable_terminal_early_release_min_current_delta_s
                is not None and current_delta_s is not None):
            checks.append(
                float(current_delta_s) >= float(
                    self.timetable_terminal_early_release_min_current_delta_s))
        if self.timetable_terminal_early_release_max_peak_shift_abs is not None:
            checks.append(
                self._fixed_selector_peak_shift_abs()
                <= float(self.timetable_terminal_early_release_max_peak_shift_abs))

        prev = self._fixed_selector_prev_diag or {}

        def _prev_float(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        if self.timetable_terminal_early_release_min_prev_wait_min is not None:
            checks.append(
                _prev_float('avg_wait_min')
                >= float(self.timetable_terminal_early_release_min_prev_wait_min))
        prev_fleet = max(
            _prev_float('N_fleet', self.N_fleet_default),
            1.0)
        if self.timetable_terminal_early_release_max_prev_overshoot_norm is not None:
            prev_overshoot_norm = (
                _prev_float('fleet_overshoot') / prev_fleet)
            checks.append(
                prev_overshoot_norm
                <= float(
                    self.timetable_terminal_early_release_max_prev_overshoot_norm))
        if self.timetable_terminal_early_release_max_prev_headway_cv is not None:
            checks.append(
                _prev_float('headway_cv')
                <= float(self.timetable_terminal_early_release_max_prev_headway_cv))
        if self.timetable_terminal_early_release_max_prev_terminal_shift_mean_s is not None:
            checks.append(
                _prev_float('terminal_launch_shift_mean')
                <= float(
                    self.timetable_terminal_early_release_max_prev_terminal_shift_mean_s))
        if self.timetable_terminal_early_release_max_prev_terminal_shift_std_s is not None:
            checks.append(
                _prev_float('terminal_launch_shift_std')
                <= float(
                    self.timetable_terminal_early_release_max_prev_terminal_shift_std_s))
        if checks and all(checks):
            return relaxed_min
        return base_min

    def _terminal_feedback_bias(
            self, direction, trip=None, action_vec=None,
            plan_origin_launch=None):
        """Causal lower-to-terminal shift from completed-trip holding history."""
        if not self.timetable_terminal_dispatch:
            return 0.0
        bias = 0.0
        if (self.timetable_terminal_feedback_enable
                and self.timetable_terminal_feedback_gain > 0.0
                and self.timetable_terminal_feedback_max_s > 0.0):
            stats = self.holding_feedback.get_direction_stats(bool(direction))
            if (int(stats.get('n_trips', 0))
                    >= self.timetable_terminal_feedback_min_trips):
                rolling = max(float(stats.get('rolling_mean', 0.0)), 0.0)
                ema = max(float(stats.get('ema', 0.0)), 0.0)
                ema_w = self.timetable_terminal_feedback_ema_weight
                signal = (1.0 - ema_w) * rolling + ema_w * ema
                signal = max(
                    0.0,
                    signal - self.timetable_terminal_feedback_deadband_s)
                if signal > 0.0:
                    hold_bias = self.timetable_terminal_feedback_gain * signal
                    if self.timetable_terminal_feedback_min_s > 0.0:
                        hold_bias = max(
                            hold_bias,
                            self.timetable_terminal_feedback_min_s)
                    bias = max(bias, float(np.clip(
                        hold_bias,
                        0.0,
                        self.timetable_terminal_feedback_max_s)))

        if (self.timetable_terminal_fleet_relief_enable
                and self.timetable_terminal_fleet_relief_max_s > 0.0):
            _, _, pressure = self._fleet_pressure()
            strength = self._pressure_strength(
                pressure,
                self.timetable_terminal_fleet_relief_pressure_start,
                self.timetable_terminal_fleet_relief_pressure_full)
            if strength > 0.0:
                relief_bias = (
                    self.timetable_terminal_fleet_relief_max_s * strength)
                if self.timetable_terminal_fleet_relief_min_s > 0.0:
                    relief_bias = max(
                        relief_bias,
                        self.timetable_terminal_fleet_relief_min_s)
                bias = max(bias, relief_bias)

        if (self.timetable_terminal_value_relief_enable
                and self.timetable_terminal_value_relief_max_s > 0.0
                and self.timetable_planner is not None
                and trip is not None
                and action_vec is not None):
            _, _, pressure = self._fleet_pressure()
            pressure_strength = self._pressure_strength(
                pressure,
                self.timetable_terminal_value_relief_pressure_start,
                self.timetable_terminal_value_relief_pressure_full)
            if pressure_strength > 0.0:
                direction = bool(direction)
                origin = (
                    float(trip.launch_time) if plan_origin_launch is None
                    else float(plan_origin_launch))
                offset = float(trip.launch_time) - origin
                base = self.timetable_planner._base_headway(trip)
                target = self.timetable_planner.target_headway(
                    base, action_vec, direction, offset)
                last_dispatch = float(getattr(
                    self.env, '_last_dispatch_time', {}).get(
                        direction, -9999.0))
                now = float(getattr(self.env, 'current_time', trip.launch_time))
                gap_now = now - last_dispatch
                gap_room = max(
                    0.0,
                    float(target)
                    + self.timetable_terminal_value_relief_gap_tolerance_s
                    - gap_now
                    - self.timetable_terminal_value_relief_min_gap_s)
                if gap_room > 0.0:
                    gap_strength = min(
                        1.0,
                        gap_room
                        / self.timetable_terminal_value_relief_gap_norm_s)
                    raw_bias = (
                        self.timetable_terminal_value_relief_max_s
                        * pressure_strength
                        * gap_strength
                        * self.timetable_terminal_value_relief_gap_gain)
                    if self.timetable_terminal_value_relief_demand_weight > 0.0:
                        freq_summary = self.env.frequency_summary()
                        demand = max(float(
                            freq_summary.get('freq_low_demand', 0.0)), 0.0)
                        raw_bias /= (
                            1.0
                            + self.timetable_terminal_value_relief_demand_weight
                            * demand)
                    raw_bias = min(
                        raw_bias,
                        self.timetable_terminal_value_relief_max_s,
                        gap_room)
                    bias = max(bias, raw_bias)
        return float(max(bias, 0.0))

    def _terminal_value_selector_features(
            self, direction, trip=None, action_vec=None,
            plan_origin_launch=None, bias_s=0.0):
        direction = bool(direction)
        bias_s = max(float(bias_s), 0.0)
        try:
            freq = self.env.frequency_summary()
        except Exception:
            freq = {}
        concurrent, n_fleet, pressure = self._fleet_pressure()
        util = float(concurrent) / max(float(n_fleet), 1.0)

        current_launch = float(getattr(trip, 'launch_time', 0.0))
        origin = (
            current_launch if plan_origin_launch is None
            else float(plan_origin_launch))
        target = float(getattr(trip, 'target_headway', 360.0))
        delta = 0.0
        if (self.timetable_planner is not None
                and trip is not None
                and action_vec is not None):
            offset = current_launch - origin
            base = self.timetable_planner._base_headway(trip)
            target = self.timetable_planner.target_headway(
                base, action_vec, direction, offset)
            delta = target - base

        last_dispatch = float(getattr(
            self.env, '_last_dispatch_time', {}).get(direction, -9999.0))
        now = float(getattr(self.env, 'current_time', current_launch))
        gap_now = now - last_dispatch
        if gap_now > 9000.0 or gap_now < 0.0:
            gap_now = target
        post_gap = gap_now + bias_s
        gap_ratio = gap_now / max(target, 1.0)
        post_gap_ratio = post_gap / max(target, 1.0)
        short_gap = max(0.0, target - gap_now)
        post_short_gap = max(0.0, target - post_gap)
        over_gap = max(0.0, post_gap - target)

        waiting_total = 0
        try:
            waiting_total = sum(
                len(st.waiting_passengers)
                for st in getattr(self.env, 'stations', []))
        except Exception:
            waiting_total = 0

        prev = self._fixed_selector_prev_diag or {}

        def _prev(key, default=0.0):
            try:
                return float(prev.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        values = [
            1.0,
            1.0 if direction else -1.0,
            bias_s / 15.0,
            target / 600.0,
            delta / 30.0,
            gap_now / 600.0,
            gap_ratio,
            post_gap_ratio,
            short_gap / 60.0,
            post_short_gap / 60.0,
            over_gap / 60.0,
            util,
            float(pressure) / max(float(n_fleet), 1.0),
            waiting_total / 500.0,
            float(freq.get('freq_low_demand', 0.0)),
            float(freq.get('freq_low_forecast', 0.0)),
            10.0 * float(freq.get('freq_high_energy', 0.0)),
            10.0 * float(freq.get('freq_middle_energy', 0.0)),
            float(freq.get('freq_promotion_strength', 0.0)),
            float(freq.get('freq_promotion_absorbed', 0.0)),
            _prev('avg_wait_min') / 10.0,
            _prev('headway_cv'),
            _prev('fleet_overshoot') / max(float(n_fleet), 1.0),
            _prev('terminal_launch_shift_mean') / 45.0,
            _prev('lower_drift_cost_mean'),
            bias_s * util / 15.0,
            bias_s * max(0.0, float(pressure)) / (
                15.0 * max(float(n_fleet), 1.0)),
            bias_s * short_gap / (15.0 * 60.0),
            bias_s * waiting_total / (15.0 * 500.0),
        ]
        x = np.asarray(values, dtype=np.float64)
        clip = self.timetable_terminal_value_selector_feature_clip
        if clip > 0.0 and x.size > 1:
            x[1:] = np.clip(x[1:], -clip, clip)
        return x

    def _ensure_terminal_value_selector_model(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if self.timetable_terminal_value_selector_A is None:
            dim = int(x.size)
            self.timetable_terminal_value_selector_A = (
                self.timetable_terminal_value_selector_ridge
                * np.eye(dim, dtype=np.float64))
            self.timetable_terminal_value_selector_b = np.zeros(
                dim, dtype=np.float64)
            return True
        return int(self.timetable_terminal_value_selector_A.shape[0]) == int(x.size)

    def _terminal_value_selector_theta(self, x):
        if not self._ensure_terminal_value_selector_model(x):
            return None
        try:
            return np.linalg.solve(
                self.timetable_terminal_value_selector_A,
                self.timetable_terminal_value_selector_b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(
                self.timetable_terminal_value_selector_A,
                self.timetable_terminal_value_selector_b,
                rcond=None)[0]

    def _terminal_value_selector_target_cost(
            self, episode_composite_cost, transition_reward=None,
            local_credit_cost=None):
        mode = str(
            self.timetable_terminal_value_selector_target
        ).strip().lower()
        episode_cost = float(episode_composite_cost)
        reward_cost = (
            -float(transition_reward)
            if transition_reward is not None else episode_cost)
        local_cost = (
            float(local_credit_cost)
            if local_credit_cost is not None else reward_cost)

        if mode in {'episode', 'composite', 'episode_composite'}:
            cost = episode_cost
        elif mode in {'reward', 'transition_reward', 'reward_cost'}:
            cost = self.timetable_terminal_value_selector_reward_weight * reward_cost
        elif mode in {'local', 'local_credit', 'credit', 'local_credit_cost'}:
            cost = self.timetable_terminal_value_selector_local_weight * local_cost
        elif mode in {'blend', 'blended', 'episode_local', 'local_blend'}:
            cost = (
                self.timetable_terminal_value_selector_episode_weight
                * episode_cost
                + self.timetable_terminal_value_selector_local_weight
                * local_cost)
        elif mode in {'episode_reward', 'reward_blend'}:
            cost = (
                self.timetable_terminal_value_selector_episode_weight
                * episode_cost
                + self.timetable_terminal_value_selector_reward_weight
                * reward_cost)
        else:
            cost = reward_cost
        return float(cost)

    def _update_terminal_value_selector(self, x, cost):
        if (not self.timetable_terminal_value_selector_enable
                or x is None):
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not self._ensure_terminal_value_selector_model(x):
            return
        cost = float(cost)
        if self.timetable_terminal_value_selector_cost_clip > 0.0:
            cost = float(np.clip(
                cost,
                -self.timetable_terminal_value_selector_cost_clip,
                self.timetable_terminal_value_selector_cost_clip))
        self._ep_terminal_value_selector_target_costs.append(cost)
        self.timetable_terminal_value_selector_A += np.outer(x, x)
        self.timetable_terminal_value_selector_b += cost * x
        self.timetable_terminal_value_selector_updates += 1

    def _select_terminal_value_bias(
            self, direction, trip=None, action_vec=None,
            plan_origin_launch=None, base_bias_s=0.0):
        base_bias = max(float(base_bias_s), 0.0)
        if not self.timetable_terminal_value_selector_enable:
            return base_bias, None
        actor_x = self._terminal_value_selector_features(
            direction, trip=trip, action_vec=action_vec,
            plan_origin_launch=plan_origin_launch, bias_s=base_bias)
        self._ep_terminal_value_selector_feature_norms.append(
            float(np.linalg.norm(actor_x)))
        if (int(self._current_ep) < int(
                self.timetable_terminal_value_selector_start_ep)
                or self.timetable_terminal_value_selector_updates
                < self.timetable_terminal_value_selector_min_observations):
            self._ep_terminal_value_selector_active.append(0.0)
            self._ep_terminal_value_selector_biases.append(base_bias)
            self._ep_terminal_value_selector_margins.append(0.0)
            self._ep_terminal_value_selector_actor_preds.append(0.0)
            self._ep_terminal_value_selector_selected_preds.append(0.0)
            return base_bias, actor_x

        theta = self._terminal_value_selector_theta(actor_x)
        if theta is None:
            self._ep_terminal_value_selector_active.append(0.0)
            self._ep_terminal_value_selector_biases.append(base_bias)
            self._ep_terminal_value_selector_margins.append(0.0)
            self._ep_terminal_value_selector_actor_preds.append(0.0)
            self._ep_terminal_value_selector_selected_preds.append(0.0)
            return base_bias, actor_x

        candidates = list(self.timetable_terminal_value_selector_candidates)
        candidates.append(base_bias)
        candidates = sorted({max(0.0, float(x)) for x in candidates})
        actor_pred = float(np.dot(actor_x, theta))
        scored = []
        for bias in candidates:
            x = self._terminal_value_selector_features(
                direction, trip=trip, action_vec=action_vec,
                plan_origin_launch=plan_origin_launch, bias_s=bias)
            pred = float(np.dot(x, theta))
            delay_cost = (
                self.timetable_terminal_value_selector_delay_penalty
                * bias / self.timetable_terminal_value_selector_bias_norm_s)
            score = pred + delay_cost
            scored.append((score, pred, bias, x))

        random_probe = (
            self.timetable_terminal_value_selector_epsilon > 0.0
            and self._terminal_value_selector_rng.random()
            < self.timetable_terminal_value_selector_epsilon)
        if random_probe and len(scored) > 1:
            chosen = scored[int(
                self._terminal_value_selector_rng.randint(len(scored)))]
        else:
            chosen = min(scored, key=lambda item: item[0])
            actor_score = (
                actor_pred
                + self.timetable_terminal_value_selector_delay_penalty
                * base_bias
                / self.timetable_terminal_value_selector_bias_norm_s)
            if (actor_score - float(chosen[0])
                    < self.timetable_terminal_value_selector_improve_margin):
                chosen = (actor_score, actor_pred, base_bias, actor_x)

        score, selected_pred, selected_bias, selected_x = chosen
        self._ep_terminal_value_selector_active.append(1.0)
        self._ep_terminal_value_selector_biases.append(float(selected_bias))
        self._ep_terminal_value_selector_margins.append(
            float(actor_pred - score))
        self._ep_terminal_value_selector_actor_preds.append(actor_pred)
        self._ep_terminal_value_selector_selected_preds.append(
            float(selected_pred))
        return float(selected_bias), selected_x

    def _freq_wait_high_share(self, low_demand, local_high, positive_only=False):
        """Share of local passenger wait attributed to high-frequency demand."""
        high_raw = float(local_high)
        high = max(high_raw, 0.0) if positive_only else abs(high_raw)
        low = max(abs(float(low_demand)), self.freq_wait_low_floor)
        if high <= 0.0 and low <= 0.0:
            return 0.0
        return float(np.clip(high / (high + low + 1e-9), 0.0, 1.0))

    def _lower_high_hold_penalty(
            self, local_high, lower_low_demand, action_s):
        """Penalty for holding through positive local high-frequency bursts."""
        if (not self.freq_wait_enable
                or self.freq_wait_lower_hold_penalty_weight <= 0.0):
            return 0.0
        action_norm = max(float(action_s), 0.0) / self.freq_wait_lower_hold_norm_s
        if self.freq_wait_lower_hold_clip > 0.0:
            action_norm = min(action_norm, self.freq_wait_lower_hold_clip)
        if action_norm <= 0.0:
            penalty = 0.0
        else:
            high_share = self._freq_wait_high_share(
                lower_low_demand,
                local_high,
                positive_only=self.freq_wait_lower_hold_positive_only)
            if self.freq_wait_lower_high_share_cap >= 0.0:
                high_share = min(
                    high_share, self.freq_wait_lower_high_share_cap)
            penalty = (
                self.freq_wait_lower_hold_penalty_weight
                * high_share
                * action_norm)
        self._ep_lower_high_hold_penalties.append(float(penalty))
        return float(penalty)

    def _adaptive_raw_credit_weight(self, freq_summary):
        """Use raw HF credit only inside a causal high-frequency regime."""
        middle_energy = float(freq_summary.get('freq_middle_energy', 0.0))
        min_cutoff = self.freq_wait_lower_raw_gate_middle_energy_min
        max_cutoff = self.freq_wait_lower_raw_gate_middle_energy_max
        width = self.freq_wait_lower_raw_gate_width
        high_weight = (max_cutoff + width - middle_energy) / width
        if min_cutoff > 0.0:
            low_weight = (middle_energy - min_cutoff) / width
            weight = min(low_weight, high_weight)
        else:
            weight = high_weight
        high_energy_min = self.freq_wait_lower_raw_gate_high_energy_min
        if high_energy_min > 0.0:
            freq_high_energy = float(
                freq_summary.get('freq_high_energy', 0.0))
            high_energy_weight = (
                (freq_high_energy - high_energy_min)
                / self.freq_wait_lower_raw_gate_high_energy_width)
            weight = min(weight, high_energy_weight)
        high_energy_max = self.freq_wait_lower_raw_gate_high_energy_max
        if high_energy_max is not None:
            freq_high_energy = float(
                freq_summary.get('freq_high_energy', 0.0))
            high_energy_cap_weight = (
                (high_energy_max
                 + self.freq_wait_lower_raw_gate_high_energy_width
                 - freq_high_energy)
                / self.freq_wait_lower_raw_gate_high_energy_width)
            weight = min(weight, high_energy_cap_weight)
        absorbed_min = self.freq_wait_lower_raw_gate_absorbed_min
        if absorbed_min > 0.0:
            absorbed = abs(float(
                freq_summary.get('freq_promotion_absorbed', 0.0)))
            absorbed_weight = (
                (absorbed - absorbed_min)
                / self.freq_wait_lower_raw_gate_absorbed_width)
            weight = min(weight, absorbed_weight)
        middle_value_max = self.freq_wait_lower_raw_gate_middle_value_max
        if middle_value_max is not None:
            middle_value = float(freq_summary.get('freq_middle', 0.0))
            middle_value_weight = (
                (middle_value_max
                 + self.freq_wait_lower_raw_gate_middle_value_width
                 - middle_value)
                / self.freq_wait_lower_raw_gate_middle_value_width)
            weight = min(weight, middle_value_weight)
        return float(np.clip(
            weight,
            self.freq_wait_lower_raw_gate_min_weight,
            1.0))

    def _select_lower_high_credit(
            self, feature_high, raw_high, freq_summary, source):
        source = str(source or 'feature').lower()
        feature_high = float(feature_high)
        raw_available = raw_high is not None
        raw_high = feature_high if raw_high is None else float(raw_high)
        if source in {'raw', 'raw_residual'} and raw_available:
            return raw_high, 1.0
        if source in {
                'adaptive_raw', 'raw_adaptive',
                'raw_if_stable', 'raw_when_stable'} and raw_available:
            w = self._adaptive_raw_credit_weight(freq_summary)
            return w * raw_high + (1.0 - w) * feature_high, w
        return feature_high, 0.0

    def _lower_board_credit_gate(self, freq_summary):
        if (not self.freq_wait_lower_board_credit_adaptive
                or self.freq_wait_lower_board_credit_weight <= 0.0):
            return 1.0
        summary = freq_summary or {}
        absorbed = abs(float(summary.get('freq_promotion_absorbed', 0.0)))
        gate = (
            (absorbed - self.freq_wait_lower_board_credit_absorbed_min)
            / self.freq_wait_lower_board_credit_absorbed_width)
        return float(np.clip(
            gate,
            self.freq_wait_lower_board_credit_min_gate,
            1.0))

    def _record_frequency_wait_credit(
            self, trip_id, wait_sum_s, boarded_count, low_demand, local_high,
            lower_low_demand=None, freq_summary=None, lf_wait_sum_s=None,
            hf_wait_sum_s=None, lf_mass=None, hf_mass=None):
        """Return lower net high-frequency wait shaping and store upper credit."""
        if not self.freq_wait_enable or boarded_count <= 0:
            return 0.0

        boarded_count = int(boarded_count)
        wait_sum_s = float(wait_sum_s)
        wait_mean_s = wait_sum_s / max(boarded_count, 1)
        wait_norm = wait_mean_s / self.freq_wait_norm_s
        if self.freq_wait_clip > 0.0:
            wait_norm = min(wait_norm, self.freq_wait_clip)

        if self.freq_wait_assignment_mode == 'frozen_passenger':
            lf_wait_sum_s = float(lf_wait_sum_s or 0.0)
            hf_wait_sum_s = float(hf_wait_sum_s or 0.0)
            lf_mass = float(lf_mass or 0.0)
            hf_mass = float(hf_mass or 0.0)
            wait_error = abs(lf_wait_sum_s + hf_wait_sum_s - wait_sum_s)
            mass_error = abs(lf_mass + hf_mass - boarded_count)
            if wait_error > 1e-6 * max(1.0, abs(wait_sum_s)):
                raise AssertionError('frozen LF/HF wait credit does not conserve')
            if mass_error > 1e-6 * max(1.0, boarded_count):
                raise AssertionError('frozen LF/HF passenger mass does not conserve')
            clip_scale = 1.0
            unscaled_wait_norm = wait_mean_s / self.freq_wait_norm_s
            if self.freq_wait_clip > 0.0 and unscaled_wait_norm > 0.0:
                clip_scale = min(1.0, self.freq_wait_clip / unscaled_wait_norm)
            upper_wait_norm_sum = (
                lf_wait_sum_s / self.freq_wait_norm_s * clip_scale)
            lower_wait_norm = (
                hf_wait_sum_s / max(boarded_count, 1)
                / self.freq_wait_norm_s * clip_scale)
            low_share = lf_mass / max(boarded_count, 1)
            lower_high_share = hf_mass / max(boarded_count, 1)
            lower_penalty = self.freq_wait_lower_weight * lower_wait_norm
            boarded_norm = hf_mass / self.freq_wait_lower_board_norm
        else:
            high_share = self._freq_wait_high_share(low_demand, local_high)
            lower_low_ref = low_demand
            if self.freq_wait_lower_share_source in {'local', 'local_low'}:
                lower_low_ref = (
                    low_demand if lower_low_demand is None else lower_low_demand)
            lower_high_share = self._freq_wait_high_share(
                lower_low_ref, local_high,
                positive_only=self.freq_wait_lower_positive_high_only)
            if self.freq_wait_lower_high_share_cap >= 0.0:
                lower_high_share = min(
                    lower_high_share, self.freq_wait_lower_high_share_cap)
            low_share = 1.0 - high_share
            lower_penalty = (
                self.freq_wait_lower_weight * lower_high_share * wait_norm)
            boarded_norm = boarded_count / self.freq_wait_lower_board_norm
            upper_wait_norm_sum = low_share * wait_norm * boarded_count
        if self.freq_wait_lower_board_clip > 0.0:
            boarded_norm = min(boarded_norm, self.freq_wait_lower_board_clip)
        board_credit_gate = self._lower_board_credit_gate(freq_summary)
        lower_board_credit = (
            self.freq_wait_lower_board_credit_weight
            * board_credit_gate
            * lower_high_share
            * boarded_norm)

        stats = self._ep_trip_wait_stats[int(trip_id)]
        stats['pax'] += boarded_count
        stats['wait_s'] += wait_sum_s
        stats['upper_wait_norm_sum'] += upper_wait_norm_sum
        stats['low_share_sum'] += low_share
        stats['events'] += 1

        self._ep_lower_wait_penalties.append(float(lower_penalty))
        self._ep_lower_board_credits.append(float(lower_board_credit))
        self._ep_lower_board_credit_gates.append(float(board_credit_gate))
        self._ep_lower_wait_net.append(
            float(lower_board_credit - lower_penalty))
        self._ep_freq_wait_low_shares.append(float(low_share))
        self._ep_freq_wait_lower_high_shares.append(float(lower_high_share))
        self._ep_freq_wait_boarded_pax += boarded_count
        return float(lower_penalty - lower_board_credit)

    def _upper_frequency_wait_credits(self, transitions):
        """Per-trip zero-mean upper credit from low-frequency passenger wait."""
        if (not self.freq_wait_enable
                or self.freq_wait_upper_weight <= 0.0
                or not transitions
                or not self._ep_trip_wait_stats):
            return {}

        owner_by_tid = {
            int(tt.launch_turn): int(getattr(
                tt, '_freqduet_planned_by', tt.launch_turn))
            for tt in self.env.timetables
        }
        totals_by_owner = defaultdict(lambda: {
            'pax': 0,
            'upper_wait_norm_sum': 0.0,
        })
        for tid, stats in self._ep_trip_wait_stats.items():
            owner = owner_by_tid.get(int(tid), int(tid))
            totals_by_owner[owner]['pax'] += int(stats.get('pax', 0))
            totals_by_owner[owner]['upper_wait_norm_sum'] += float(
                stats.get('upper_wait_norm_sum', 0.0))

        raw_by_tid = {}
        for owner, stats in totals_by_owner.items():
            pax = max(int(stats['pax']), 1)
            raw_by_tid[int(owner)] = float(
                stats['upper_wait_norm_sum'] / pax)

        values = []
        for trans in transitions:
            tid = int(trans['tid'])
            if tid in raw_by_tid:
                values.append(raw_by_tid[tid])
        if not values:
            return {}

        mean = float(np.mean(values))
        std = float(max(np.std(values), 1e-6))
        credits = {}
        for trans in transitions:
            tid = int(trans['tid'])
            raw = raw_by_tid.get(tid, mean)
            if self.freq_wait_normalize_upper:
                credit = -self.freq_wait_upper_weight * ((raw - mean) / std)
            else:
                credit = -self.freq_wait_upper_weight * raw
            credits[tid] = float(credit)
        return credits

    # ─── v3: HAAR / PIPER reachability + advantage helpers ──────────

    def _ensure_reach_net(self):
        if self.reach_net is None and self.coupling_mode == 'haar' \
                and self.haar_use_reach_gate:
            with self.randomness.torch_initialization('reachability_init'):
                self.reach_net = ReachabilityMLP(
                    self.upper_state_dim).to(self.device)
            self.reach_optimizer = torch.optim.Adam(
                self.reach_net.parameters(), lr=self.haar_reach_lr)

    def _compute_upper_advantage(self, s_upper, delta_t):
        """Estimate A_U(s, a) = Q(s, a) − V(s), with V(s) = Q(s, π(s))."""
        try:
            with torch.no_grad():
                s_t = torch.FloatTensor(s_upper).unsqueeze(0).to(self.device)
                a_t = torch.FloatTensor([[float(delta_t)]]).to(self.device)
                q_sa = self.upper_trainer.q_net(s_t, a_t).mean().item()
                a_pi, _, _, _, _ = self.upper_trainer.policy_net.evaluate(s_t)
                v_s = self.upper_trainer.q_net(s_t, a_pi).mean().item()
            return q_sa - v_s
        except Exception:
            return 0.0

    def _reachability_score(self, s_upper, delta_t, hold_mean, hold_std):
        if not self.haar_use_reach_gate or self.reach_net is None:
            return 1.0
        try:
            x = _reach_features(s_upper, delta_t, hold_mean, hold_std)
            with torch.no_grad():
                t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
                f = float(self.reach_net(t).item())
            return f
        except Exception:
            return 1.0

    def _build_haar_tap_signal(self, trip_gap_devs):
        """HAAR-style per-trip reward bonus: β · clip(A_U, -c, c) · f_k."""
        self._ensure_reach_net()
        if not self._episode_upper_transitions:
            return None
        tap = {}
        for trans in self._episode_upper_transitions:
            tid = int(trans['tid'])
            s_U = np.asarray(trans['s'], dtype=np.float32)
            a_U = float(np.asarray(trans['a']).flatten()[0])
            adv = self._compute_upper_advantage(s_U, a_U)
            adv_clip = float(np.clip(adv, -self.haar_clip, self.haar_clip))
            stats = self.holding_feedback.get_trip_stats(tid)
            hm = float(stats['mean']) if stats else 0.0
            hs = float(stats['std']) if stats else 0.0
            f_k = self._reachability_score(s_U, a_U, hm, hs)
            bonus = self.haar_beta * adv_clip * f_k
            global_tid = self._current_ep * 1000 + tid
            tap[global_tid] = float(bonus)
            # Buffer for reach training (label assigned in _train_reach_classifier)
            if self.haar_use_reach_gate:
                self._reach_buffer.append({
                    'x': _reach_features(s_U, a_U, hm, hs),
                    'tid': tid,
                })
        return tap

    def _train_reach_classifier(self, trip_gap_devs, max_train_size=2048):
        """Binary cross-entropy on (s_U, δ, hold) → 1[gap_dev < threshold]."""
        if not self._reach_buffer:
            return
        # Assign labels to recent buffer entries using current-episode gap_devs
        for entry in self._reach_buffer:
            if 'label' in entry:
                continue
            dev = trip_gap_devs.get(entry['tid'], None)
            if dev is None:
                entry['label'] = 1.0  # treat unobserved as reachable (skip)
            else:
                entry['label'] = 1.0 if dev < self.haar_reach_threshold else 0.0
        # Trim buffer to keep memory bounded
        if len(self._reach_buffer) > max_train_size:
            self._reach_buffer = self._reach_buffer[-max_train_size:]
        # One small gradient step per episode on a sampled batch
        bs = min(len(self._reach_buffer), 128)
        if bs < 8:
            return
        idx = self._reachability_rng.choice(
            len(self._reach_buffer), bs, replace=False)
        xs = np.stack([self._reach_buffer[i]['x'] for i in idx])
        ys = np.array([self._reach_buffer[i]['label'] for i in idx], dtype=np.float32)
        x_t = torch.FloatTensor(xs).to(self.device)
        y_t = torch.FloatTensor(ys).to(self.device)
        self.reach_optimizer.zero_grad()
        p = self.reach_net(x_t)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y_t)
        loss.backward()
        self.reach_optimizer.step()

    def _upper_callback_v2(self, s_upper_v1, trip):
        """Per-dispatch decision: output δ_t, store (s, a, trip_id, s') without reward.
        Reward is backfilled at episode end via hindsight credit assignment."""
        decision_time_s = float(getattr(self.env, 'current_time', trip.launch_time))
        planner_dir = bool(trip.direction)
        planner_key = (
            "__all__" if self.timetable_plan_all_directions else planner_dir)
        state_active_plan = self._active_timetable_plans.get(planner_key)
        s_upper = self._inject_lifecycle_holding_state(
            self.env._build_upper_state_v2(trip), planner_dir)
        if self.freq_holdfb_enable:
            s_upper = np.concatenate([
                np.asarray(s_upper, dtype=np.float32),
                self._frequency_hold_feedback_features(bool(trip.direction)),
            ]).astype(np.float32)
        if self.freq_driftfb_enable:
            s_upper = np.concatenate([
                np.asarray(s_upper, dtype=np.float32),
                self._frequency_drift_feedback_features(bool(trip.direction)),
            ]).astype(np.float32)
        if self.ablate_holding_feedback:
            # Zero out holding feedback state dims [5,6,7]
            s_upper[5:8] = 0.0
            end = len(s_upper)
            if self.freq_driftfb_enable:
                s_upper[end - self.freq_driftfb_dim:end] = 0.0
                end -= self.freq_driftfb_dim
            if self.freq_holdfb_enable:
                s_upper[end - self.freq_holdfb_dim:end] = 0.0

        if self.upper_plan_context_dim > 0:
            s_upper = np.concatenate([
                np.asarray(s_upper, dtype=np.float32),
                self._upper_plan_context(state_active_plan, decision_time_s),
            ]).astype(np.float32)

        s_upper = self._augment_upper_state_history(s_upper)

        # ─── TPC-Lower behaviour-policy sampling ───
        # During Phase 1 (after warmup, target_upper_trainer initialised), sample
        # δ_t from a mixture: ε from current upper (exploratory) + (1−ε) from
        # N(target_mean, σ_tgt) so the lower's training distribution can be
        # importance-corrected back toward the EMA "deployment" upper.
        log_mu = None
        upper_decision_taken = True
        plan_origin_launch = None
        selector_x = None
        headway_selector_x = None
        terminal_selector_x = None
        snapshot_selector_info = None
        cf_action_selector_info = None
        snapshot_write_terminal_dispatch = self.timetable_terminal_dispatch
        plan_id = None
        promotion_replan = False
        policy_command_vec = None
        if self.timetable_planner is not None and self.coupling_mode == 'hiro':
            active_plan = self._active_timetable_plans.get(planner_key)
            if active_plan is not None:
                elapsed = float(trip.launch_time) - float(active_plan['origin'])
                if (self.timetable_promotion_replan
                        and getattr(self.env, 'frequency_tracker', None) is not None):
                    freq_summary = self.env.frequency_summary()
                    promotion_replan = (
                        bool(freq_summary.get('freq_promotion_flag', 0.0))
                        and float(freq_summary.get('freq_promotion_strength', 0.0))
                        >= self.timetable_promotion_replan_strength_min
                    )
                    if promotion_replan and self.timetable_promotion_replan_cooldown_s > 0.0:
                        last_replan = self._last_promotion_replan_launch.get(
                            planner_key)
                        if (last_replan is not None
                                and float(trip.launch_time) - float(last_replan)
                                < self.timetable_promotion_replan_cooldown_s):
                            promotion_replan = False
                if (0.0 <= elapsed < self.timetable_replan_interval_s
                        and elapsed <= self.timetable_planner.horizon_s
                        and not promotion_replan):
                    action_vec = np.asarray(
                        active_plan['action'], dtype=np.float32).reshape(-1)
                    plan_origin_launch = float(active_plan['origin'])
                    plan_id = int(active_plan.get(
                        'plan_id', trip.launch_turn))
                    snapshot_write_terminal_dispatch = bool(
                        active_plan.get(
                            'write_terminal_dispatch',
                            self.timetable_terminal_dispatch))
                    upper_decision_taken = False
                    self._ep_upper_plan_reuses += 1

        if (upper_decision_taken and self._episode_training and self.tpc_enable
                and self.target_upper_trainer is not None):
            target_mean_arr = np.asarray(
                self.target_upper_trainer.policy_net.get_action(
                    s_upper, deterministic=True),
                dtype=np.float32).reshape(-1)
            if self._tpc_rng.random() < self.tpc_eps:
                # exploratory: sample from current π_U
                action_vec = np.asarray(
                    self.upper_trainer.policy_net.get_action(
                        s_upper, deterministic=False),
                    dtype=np.float32).reshape(-1)
            elif self.tpc_target_distribution == 'bounded_logistic_normal_v4':
                target_u = self.upper_trainer.policy_net.normalized_action(
                    target_mean_arr)
                target_z = np.log(target_u / (1.0 - target_u))
                sampled_z = (
                    target_z
                    + self._tpc_rng.randn(self.upper_action_dim)
                    * self.tpc_latent_sigma)
                sampled_u = 1.0 / (1.0 + np.exp(-sampled_z))
                action_vec = (
                    self.upper_action_low
                    + sampled_u * (
                        self.upper_action_high - self.upper_action_low)
                ).astype(np.float32)
            else:
                # target: independent N(target_mean, σ_tgt), clipped to range
                action_vec = np.clip(
                    target_mean_arr
                    + self._tpc_rng.randn(self.upper_action_dim)
                    * self.tpc_sigma_tgt,
                    self.upper_action_low, self.upper_action_high).astype(np.float32)
            # Mixture log-prob log_mu = log( ε π_U + (1-ε) N(target_mean, σ_tgt) )
            log_p_explore = float(self.upper_trainer.policy_net.log_prob(
                s_upper, action_vec,
                coordinates='normalized_unit_interval'))
            if self.tpc_target_distribution == 'bounded_logistic_normal_v4':
                action_u = self.upper_trainer.policy_net.normalized_action(
                    action_vec)
                action_z = np.log(action_u / (1.0 - action_u))
                target_u = self.upper_trainer.policy_net.normalized_action(
                    target_mean_arr)
                target_z = np.log(target_u / (1.0 - target_u))
                standardized = (
                    (action_z - target_z) / self.tpc_latent_sigma)
                log_p_target = float(
                    -0.5 * np.dot(standardized, standardized)
                    - action_vec.size * np.log(
                        self.tpc_latent_sigma * np.sqrt(2 * np.pi))
                    - np.log(action_u * (1.0 - action_u)).sum())
            else:
                z = ((action_vec - target_mean_arr)
                     / max(self.tpc_sigma_tgt, 1e-6))
                log_p_target = float(
                    -0.5 * np.dot(z, z)
                    - action_vec.size * np.log(
                        max(self.tpc_sigma_tgt, 1e-6)
                        * np.sqrt(2 * np.pi)))
            log_mu = float(np.logaddexp(
                np.log(self.tpc_eps + 1e-12) + log_p_explore,
                np.log(1.0 - self.tpc_eps + 1e-12) + log_p_target))
        elif upper_decision_taken:
            action_vec = np.asarray(
                self.upper_trainer.policy_net.get_action(
                    s_upper, deterministic=not self._episode_training),
                dtype=np.float32).reshape(-1)

        if upper_decision_taken and self.upper_action_override_enable:
            action_vec = self.upper_action_override_values.copy()
            log_mu = None

        if upper_decision_taken:
            policy_command_vec = np.asarray(
                action_vec, dtype=np.float32).reshape(-1).copy()

        if (upper_decision_taken and self.timetable_planner is not None
                and self.coupling_mode == 'hiro'):
            plan_origin_launch = float(trip.launch_time)
            plan_id = int(trip.launch_turn)
            prev_plan = self._active_timetable_plans.get(planner_key)
            if prev_plan is not None and self.timetable_action_ema_alpha < 1.0:
                prev_action = np.asarray(
                    prev_plan['action'], dtype=np.float32).reshape(-1)
                action_vec = (
                    self.timetable_action_ema_alpha * action_vec
                    + (1.0 - self.timetable_action_ema_alpha) * prev_action
                ).astype(np.float32)
            elif self.timetable_action_ema_alpha < 1.0:
                action_vec = (
                    self.timetable_action_ema_alpha * action_vec
                ).astype(np.float32)
            action_vec = self._prepare_upper_action(action_vec)
            action_vec, cf_action_selector_info = (
                self._select_counterfactual_action(
                    s_upper, action_vec, planner_dir, trip=trip))
            cf_selector_active = (
                cf_action_selector_info is not None
                and float(cf_action_selector_info.get('active', 0.0)) > 0.5)
            if cf_selector_active:
                snapshot_write_terminal_dispatch = bool(
                    float(cf_action_selector_info.get(
                        'terminal_dispatch', 0.0)) > 0.5)
            selectors_disabled = (
                self.upper_action_override_disable_value_selectors
                or (cf_selector_active
                    and self.cf_action_selector_disable_value_selectors))
            if not selectors_disabled:
                action_vec, selector_x = self._select_upper_residual_value_action(
                    s_upper, action_vec, planner_dir, trip=trip,
                    plan_origin_launch=plan_origin_launch)
                action_vec, headway_selector_x = self._select_headway_value_plan_action(
                    s_upper, action_vec, planner_dir, trip=trip,
                    plan_origin_launch=plan_origin_launch)
                action_vec, snapshot_selector_info = (
                    self._select_snapshot_value_action(
                        action_vec, planner_dir, trip=trip,
                        plan_origin_launch=plan_origin_launch))
                action_vec, snapshot_action_selector_info = (
                    self._select_snapshot_action_value_action(
                        action_vec, planner_dir, trip=trip,
                        plan_origin_launch=plan_origin_launch,
                        primary_info=snapshot_selector_info))
                if (snapshot_action_selector_info is not None
                        and float(snapshot_action_selector_info.get(
                            'override_action', 0.0)) > 0.5):
                    primary_snapshot_info = snapshot_selector_info or {}
                    combined_snapshot_info = dict(primary_snapshot_info)
                    combined_snapshot_info.update(snapshot_action_selector_info)
                    snapshot_selector_info = combined_snapshot_info
                elif (snapshot_action_selector_info is not None
                        and float(snapshot_action_selector_info.get(
                            'guard_blocked', 0.0)) > 0.5):
                    combined_snapshot_info = dict(snapshot_selector_info or {})
                    for key in (
                            'risk_score',
                            'risk_penalty',
                            'risk_penalty_max',
                            'guard_blocked',
                            'guard_negative_target',
                            'guard_negative_target_blocked',
                            'guard_prev_overshoot_norm',
                            'guard_fleet_pressure_norm',
                            'guard_primary_terminal_bias_s'):
                        combined_snapshot_info[key] = (
                            snapshot_action_selector_info.get(key, 0.0))
                    snapshot_selector_info = combined_snapshot_info
                elif snapshot_action_selector_info is not None:
                    combined_snapshot_info = dict(snapshot_selector_info or {})
                    for key in (
                            'risk_score',
                            'risk_penalty',
                            'risk_penalty_max',
                            'guard_prev_overshoot_norm',
                            'guard_fleet_pressure_norm',
                            'guard_primary_terminal_bias_s'):
                        combined_snapshot_info[key] = (
                            snapshot_action_selector_info.get(key, 0.0))
                    snapshot_selector_info = combined_snapshot_info
                if (snapshot_selector_info is not None
                        and float(snapshot_selector_info.get('active', 0.0)) > 0.5):
                    snapshot_write_terminal_dispatch = bool(
                        float(snapshot_selector_info.get(
                            'terminal_dispatch', 0.0)) > 0.5)
            self._active_timetable_plans[planner_key] = {
                'origin': plan_origin_launch,
                'plan_id': int(plan_id),
                'action': action_vec.astype(np.float32).copy(),
                'write_terminal_dispatch': bool(
                    snapshot_write_terminal_dispatch),
            }
            if promotion_replan:
                self._last_promotion_replan_launch[planner_key] = float(
                    trip.launch_time)
            self._ep_upper_plan_decisions += 1
        elif (not upper_decision_taken and self.timetable_planner is not None
                and self.coupling_mode == 'hiro'):
            action_vec = self._prepare_upper_action(action_vec)

        delta_t = float(action_vec[0])
        base_hw = trip.target_headway if hasattr(trip, 'target_headway') else 360.0
        plan_summary = None
        plan_penalty = 0.0
        if self.timetable_planner is not None and self.coupling_mode == 'hiro':
            exact_headway_curve = (
                self.timetable_planner.terminal_schedule_mode
                == 'exact_headway_curve')
            if exact_headway_curve and not snapshot_write_terminal_dispatch:
                raise RuntimeError(
                    'exact_headway_curve cannot disable executable dispatch')
            if plan_id is None:
                plan_id = int(trip.launch_turn)
            current_plan_delta = self.timetable_planner.delta_at(
                action_vec, bool(trip.direction), 0.0)
            terminal_shift_min_s = (
                self._terminal_shift_min_for_frequency(
                    action_vec, current_delta_s=current_plan_delta)
                if snapshot_write_terminal_dispatch else None)
            terminal_shift_max_s = (
                self._terminal_shift_max_for_frequency()
                if snapshot_write_terminal_dispatch else None)
            terminal_shift_bias_s = (
                self._terminal_feedback_bias(
                    bool(trip.direction), trip=trip,
                    action_vec=action_vec,
                    plan_origin_launch=plan_origin_launch)
                if snapshot_write_terminal_dispatch else 0.0)
            snapshot_terminal_bias_s = (
                float((snapshot_selector_info or {}).get(
                    'terminal_bias_s', 0.0))
                if snapshot_write_terminal_dispatch else 0.0)
            if snapshot_terminal_bias_s > 0.0:
                terminal_shift_bias_s = max(
                    float(terminal_shift_bias_s), snapshot_terminal_bias_s)
            if (not exact_headway_curve
                    and snapshot_write_terminal_dispatch
                    and upper_decision_taken
                    and not self.upper_action_override_disable_value_selectors):
                terminal_shift_bias_s, terminal_selector_x = (
                    self._select_terminal_value_bias(
                        bool(trip.direction), trip=trip,
                        action_vec=action_vec,
                        plan_origin_launch=plan_origin_launch,
                        base_bias_s=terminal_shift_bias_s))
            if exact_headway_curve:
                terminal_shift_min_s = None
                terminal_shift_max_s = None
                terminal_shift_bias_s = 0.0
            terminal_floor_ratio = (
                self.timetable_terminal_headway_floor_ratio
                if (not exact_headway_curve
                    and snapshot_write_terminal_dispatch
                    and self.timetable_terminal_headway_floor_enable)
                else 0.0)
            terminal_floor_min_s = (
                self.timetable_terminal_headway_floor_min_s
                if (not exact_headway_curve
                    and snapshot_write_terminal_dispatch
                    and self.timetable_terminal_headway_floor_enable)
                else 0.0)
            if exact_headway_curve and not upper_decision_taken:
                plan_summary = self.timetable_planner.cached_plan_summary(
                    trip, plan_id=int(plan_id))
            else:
                plan_summary = self.timetable_planner.apply(
                    self.env.timetables, trip, action_vec,
                    origin_launch_s=plan_origin_launch,
                    write_scheduled_launch=snapshot_write_terminal_dispatch,
                    terminal_shift_min_s=terminal_shift_min_s,
                    terminal_shift_max_s=terminal_shift_max_s,
                    terminal_shift_bias_s=terminal_shift_bias_s,
                    terminal_headway_floor_ratio=terminal_floor_ratio,
                    terminal_headway_floor_min_s=terminal_floor_min_s,
                    plan_id=plan_id)
            delta_t = float(plan_summary['effective_delta'])
            base_hw = float(plan_summary['base_headway'])
            self._ep_terminal_shift_caps.append(
                float(plan_summary.get(
                    'terminal_shift_max_s',
                    self.timetable_planner.terminal_shift_max_s)))
            self._ep_terminal_shift_mins.append(
                float(plan_summary.get(
                    'terminal_shift_min_s',
                    self.timetable_planner.terminal_shift_min_s)))
            self._ep_terminal_feedback_biases.append(
                float(plan_summary.get('terminal_shift_bias_s', 0.0)))
            floor_n = int(plan_summary.get('terminal_headway_floor_n', 0))
            if floor_n > 0:
                self._ep_terminal_headway_floors.extend(
                    [float(plan_summary.get(
                        'terminal_headway_floor_mean', 0.0))] * floor_n)
            if upper_decision_taken:
                plan_penalty = (
                    self.upper_plan_penalty_weight
                    * self.timetable_planner.smoothness_penalty(action_vec))
                self._ep_upper_plan_penalties.append(plan_penalty)
            self._ep_upper_plan_targets.append(
                float(plan_summary.get('planned_mean', trip.target_headway)))
            if upper_decision_taken:
                self._ep_upper_plan_raw_delta_means.append(float(
                    plan_summary.get('raw_headway_delta_mean_s', 0.0)))
                self._ep_upper_plan_projected_delta_means.append(float(
                    plan_summary.get('projected_headway_delta_mean_s', 0.0)))
                self._ep_upper_plan_projected_delta_sums.append(float(
                    plan_summary.get('projected_headway_delta_sum_s', 0.0)))
        self._ep_upper_deltas.append(delta_t)
        self._ep_upper_deltas_by_dir[bool(trip.direction)].append(delta_t)
        if getattr(self.env, 'frequency_tracker', None) is not None:
            freq_summary = self.env.frequency_tracker.summary()
            self._ep_upper_demand_action.append((
                float(freq_summary.get('freq_low_demand', 0.0)),
                float(freq_summary.get('freq_high_energy', 0.0)),
                float(delta_t),
            ))

        # Action channel:
        #   default (channels/haar): δ_t directly shifts launch time, target_headway
        #                            communicated to lower stays at the baseline value
        #   hiro mode:               δ_t shifts target_headway only; launch time stays
        #                            at the baseline schedule. The lower's Lagrangian
        #                            cost is then on (realised_headway - (H_base + δ_t))^2,
        #                            i.e. goal-conditioned holding control.
        if not hasattr(trip, '_original_launch'):
            trip._original_launch = trip.launch_time
        if self.coupling_mode == 'hiro':
            trip._delta_t = 0  # no launch shift
            if snapshot_write_terminal_dispatch:
                trip._freqduet_terminal_dispatch = True
                if not hasattr(trip, '_freqduet_scheduled_launch'):
                    trip._freqduet_scheduled_launch = int(round(trip.launch_time))
            else:
                for attr in (
                        '_freqduet_scheduled_launch',
                        '_freqduet_terminal_dispatch',
                        '_freqduet_min_dispatch_headway'):
                    if hasattr(trip, attr):
                        delattr(trip, attr)
            if plan_summary is None:
                # Override the trip's target_headway so the lower's cost is goal-conditioned.
                base_hw = (trip.target_headway
                           if hasattr(trip, 'target_headway') else 360.0)
                trip.target_headway = base_hw + float(delta_t)
        else:
            trip._delta_t = int(delta_t)

        # TPC-Lower: store dispatch metadata for IS weight lookup. Use a global
        # trip id (episode * 1000 + local tid) so cross-episode samples in the
        # replay buffer can still be reweighted (or default to 1 if pruned).
        if log_mu is not None:
            global_tid = self._current_ep * 1000 + int(trip.launch_turn)
            self.dispatch_meta[global_tid] = {
                'z': s_upper.astype(np.float32),
                'delta': self.upper_plan_execution.replay_action(
                    policy_command_vec, action_vec),
                'log_mu': log_mu,
            }
            # Bound metadata size: drop oldest when over budget
            if len(self.dispatch_meta) > self._dispatch_meta_max:
                # delete the oldest 10% in one pass
                k_drop = max(1, self._dispatch_meta_max // 10)
                for old_key in sorted(self.dispatch_meta.keys())[:k_drop]:
                    del self.dispatch_meta[old_key]

        # Store only actual upper decisions. Cached timetable points are
        # executions of the previous low-frequency plan, not new policy actions.
        if upper_decision_taken:
            self._close_previous_upper_transition(
                s_upper,
                done=False,
                decision_time_s=decision_time_s,
                planner_key=planner_key,
            )

            value_cost, value_active = self._upper_residual_value_cost(delta_t)
            replay_action = self.upper_plan_execution.replay_action(
                policy_command_vec, action_vec)
            stream_key = self._upper_transition_stream_key(planner_key)
            self._prev_upper_states[stream_key] = {
                's': s_upper.copy(),
                'a': replay_action,
                'tid': int(trip.launch_turn),
                'dir': bool(trip.direction),
                'a_eff': float(delta_t),
                'plan_penalty': float(plan_penalty),
                'upper_value_cost': float(value_cost),
                'upper_value_active': float(value_active),
                'upper_residual_selector_x': selector_x,
                'terminal_value_selector_x': terminal_selector_x,
                'headway_value_planner_x': headway_selector_x,
                'decision_time_s': decision_time_s,
            }
            self.upper_interval_credit.begin(
                stream_key, start_time_s=decision_time_s)

        # Record dispatch info (actual launch time captured post-episode from env).
        # Terminal-dispatch mode uses the planner's executable launch schedule.
        # Otherwise channels/haar apply _delta_t to launch time and HIRO keeps the
        # baseline launch schedule.
        planned_launch = None
        if snapshot_write_terminal_dispatch:
            planned_launch = getattr(trip, '_freqduet_scheduled_launch', None)
        if planned_launch is None:
            planned_launch = (
                trip._original_launch + int(getattr(trip, '_delta_t', 0)))
        launch_shift = int(planned_launch) - int(trip._original_launch)
        self._ep_terminal_launch_shifts.append(float(launch_shift))
        dir_key = 'up' if trip.direction else 'down'
        self._ep_dispatch_times[dir_key].append({
            'tid': trip.launch_turn,
            'scheduled': trip._original_launch,
            'delta_t': float(delta_t),                 # effective current target shift
            'launch_shift': launch_shift,
            'effective_launch': int(planned_launch),
            'target_headway': float(getattr(trip, 'target_headway', base_hw)),
        })

        try:
            freq_summary = self.env.frequency_summary()
        except Exception:
            freq_summary = {}
        concurrent, n_fleet, fleet_pressure = self._fleet_pressure()
        try:
            waiting_total = sum(
                len(st.waiting_passengers)
                for st in getattr(self.env, 'stations', []))
        except Exception:
            waiting_total = 0
        now = float(getattr(self.env, 'current_time', trip.launch_time))
        last_dispatch = float(getattr(
            self.env, '_last_dispatch_time', {}).get(
                bool(trip.direction), -9999.0))
        terminal_gap_now = now - last_dispatch
        target_for_gap = float(getattr(trip, 'target_headway', base_hw))
        if terminal_gap_now > 9000.0 or terminal_gap_now < 0.0:
            terminal_gap_now = target_for_gap
        terminal_short_gap = max(0.0, target_for_gap - terminal_gap_now)
        terminal_over_gap = max(0.0, terminal_gap_now - target_for_gap)

        # Per-trip record for step-level diagnostics
        hour = 6 + trip.launch_time // 3600
        period = 'peak' if (7 <= hour <= 9 or 17 <= hour <= 19) else (
            'off' if 9 < hour < 17 else 'trans')
        self._ep_trip_records.append({
            'tid': trip.launch_turn,
            'dir': int(trip.direction),
            'hour': hour,
            'period': period,
            'delta_t': round(delta_t, 1),
            'base_hw': round(base_hw, 0),
            'eff_hw': round(float(getattr(trip, 'target_headway', base_hw)), 0),
            's_hold_mean': round(s_upper[5] * 60, 1),
            's_hold_std': round(s_upper[6] * 60, 1),
            'upper_decision': int(bool(upper_decision_taken)),
            'promotion_replan': int(bool(promotion_replan)),
            'launch_shift': float(launch_shift),
            'effective_launch': int(planned_launch),
            'terminal_gap_now': round(float(terminal_gap_now), 3),
            'terminal_short_gap': round(float(terminal_short_gap), 3),
            'terminal_over_gap': round(float(terminal_over_gap), 3),
            'fleet_concurrent': float(concurrent),
            'fleet_target': float(n_fleet),
            'fleet_pressure': float(fleet_pressure),
            'waiting_total': float(waiting_total),
            'freq_low_demand': float(freq_summary.get('freq_low_demand', 0.0)),
            'freq_low_forecast': float(freq_summary.get('freq_low_forecast', 0.0)),
            'freq_high_energy': float(freq_summary.get('freq_high_energy', 0.0)),
            'freq_middle_energy': float(freq_summary.get('freq_middle_energy', 0.0)),
            'freq_od_entropy': float(freq_summary.get('freq_od_entropy', 0.0)),
            'freq_promotion_strength': float(freq_summary.get('freq_promotion_strength', 0.0)),
            'freq_promotion_active': float(freq_summary.get('freq_promotion_active', 0.0)),
            'cf_action_selector_active': float(
                (cf_action_selector_info or {}).get('active', 0.0)),
            'cf_action_selector_method': str(
                (cf_action_selector_info or {}).get('selected_method', '')),
            'cf_action_selector_delta_s': float(
                (cf_action_selector_info or {}).get('selected_delta_s', 0.0)),
            'cf_action_selector_actor_delta_s': float(
                (cf_action_selector_info or {}).get('actor_delta_s', 0.0)),
            'cf_action_selector_changed': float(
                (cf_action_selector_info or {}).get('changed', 0.0)),
            'cf_action_selector_terminal_dispatch': float(
                (cf_action_selector_info or {}).get('terminal_dispatch', 0.0)),
            'cf_action_selector_confidence': float(
                (cf_action_selector_info or {}).get('confidence', 0.0)),
            'cf_action_selector_node_id': int(
                (cf_action_selector_info or {}).get('node_id', -1)),
            'snapshot_value_active': float(
                (snapshot_selector_info or {}).get('active', 0.0)),
            'snapshot_value_method': str(
                (snapshot_selector_info or {}).get('selected_method', '')),
            'snapshot_value_override_action': float(
                (snapshot_selector_info or {}).get('override_action', 0.0)),
            'snapshot_value_terminal_dispatch': float(
                (snapshot_selector_info or {}).get('terminal_dispatch', 0.0)),
            'snapshot_value_terminal_bias_s': float(
                (snapshot_selector_info or {}).get('terminal_bias_s', 0.0)),
            'snapshot_value_pred': float(
                (snapshot_selector_info or {}).get('selected_pred', 0.0)),
            'snapshot_value_baseline_pred': float(
                (snapshot_selector_info or {}).get('baseline_pred', 0.0)),
            'snapshot_value_margin': float(
                (snapshot_selector_info or {}).get('margin', 0.0)),
            'snapshot_value_candidate_gate_cap_s': float(
                (snapshot_selector_info or {}).get(
                    'candidate_gate_cap_s', 0.0)),
            'snapshot_value_candidate_gate_filtered': float(
                (snapshot_selector_info or {}).get(
                    'candidate_gate_filtered', 0.0)),
            'snapshot_value_risk_score': float(
                (snapshot_selector_info or {}).get('risk_score', 0.0)),
            'snapshot_value_risk_penalty': float(
                (snapshot_selector_info or {}).get('risk_penalty', 0.0)),
            'snapshot_value_risk_penalty_max': float(
                (snapshot_selector_info or {}).get('risk_penalty_max', 0.0)),
            'snapshot_value_guard_blocked': float(
                (snapshot_selector_info or {}).get('guard_blocked', 0.0)),
            'snapshot_value_guard_negative_target': float(
                (snapshot_selector_info or {}).get('guard_negative_target', 0.0)),
            'snapshot_value_guard_negative_target_blocked': float(
                (snapshot_selector_info or {}).get(
                    'guard_negative_target_blocked', 0.0)),
            'snapshot_value_guard_prev_overshoot_norm': float(
                (snapshot_selector_info or {}).get(
                    'guard_prev_overshoot_norm', 0.0)),
            'snapshot_value_guard_fleet_pressure_norm': float(
                (snapshot_selector_info or {}).get(
                    'guard_fleet_pressure_norm', 0.0)),
            'snapshot_value_guard_primary_bias_s': float(
                (snapshot_selector_info or {}).get(
                    'guard_primary_terminal_bias_s', 0.0)),
        })
        self._update_upper_state_history(
            action_vec=action_vec,
            direction=bool(trip.direction),
            delta_t=float(delta_t),
            launch_shift=float(launch_shift),
            plan_penalty=float(plan_penalty),
            upper_decision_taken=bool(upper_decision_taken),
            promotion_replan=bool(promotion_replan),
        )

        # In HIRO mode the target headway has been adjusted to base_hw + δ_t (above);
        # in default/HAAR mode the base headway is unchanged because δ_t was applied
        # via launch-time gating instead. Either way, return the trip's current target.
        return trip.target_headway if hasattr(trip, 'target_headway') else base_hw

    # ────────────────── Episode ──────────────────

    def run_episode(self, ep, training=True, N_fleet_override=None,
                    scenario_seed=None, record_diagnostics=None):
        t0 = time.time()
        self._episode_training = bool(training)
        self.env._freqduet_training = bool(training)
        self.env._freqduet_episode = int(ep)
        if scenario_seed is None:
            scenario_seed_base = int(
                self.cfg.get('env', {}).get('scenario_seed', self.base_seed))
            scenario_seed = scenario_seed_base * 1000003 + int(ep)
        self.env.scenario_seed = int(scenario_seed)
        if record_diagnostics is None:
            record_diagnostics = bool(training)
        self.env.reset()
        self.holding_feedback.clear(reset_history=(
            self.upper_holding_state_source == 'trip_lifecycle'
            and self.upper_holding_state_episode_local
        ))
        self.lower_lifecycle.reset_episode()
        self._current_ep = ep
        self._episode_upper_transitions = []
        self._prev_upper_states = {}
        self._upper_state_history = deque(
            maxlen=max(1, self.upper_state_history_len))
        self._ep_lower_actions = []
        self._ep_lower_context_gate_values = []
        self._ep_lower_action_bins_gate_values = []
        self._ep_lower_rewards = []
        self._ep_lower_trip_boundary_resets = 0
        self._ep_lower_pending_states_dropped = 0
        self._ep_lower_pending_actions_dropped = 0
        self._ep_lower_pending_states_consumed = 0
        self._ep_lower_pending_actions_consumed = 0
        self._ep_lower_terminal_action_masks = 0
        self._ep_lower_terminal_transitions = 0
        self._ep_lower_terminal_outcomes_missing = 0
        self._ep_hold_feedback_trip_finalizations = 0
        self._ep_upper_deltas = []
        self._ep_upper_rewards = []
        self._ep_upper_system_rewards = []
        self._ep_upper_gap_credits = []
        self._ep_upper_reliability_rewards = []
        self._ep_upper_interval_rewards = []
        self._ep_upper_interval_wait_costs = []
        self._ep_upper_interval_onboard_costs = []
        self._ep_upper_interval_dispatch_backlog_costs = []
        self._ep_upper_interval_headway_costs = []
        self._ep_upper_interval_fleet_costs = []
        self._ep_upper_interval_coverages = []
        self._ep_lower_actions_by_dir = {True: [], False: []}
        self._ep_lower_load_hold_penalties = []
        self._ep_lower_load_ratios = []
        self._ep_lower_normalized_person_delays = []
        self._ep_lower_causal_guard_active = []
        self._ep_lower_causal_guard_limits = []
        self._ep_lower_causal_guard_adjustments = []
        self._ep_upper_deltas_by_dir = {True: [], False: []}
        self._ep_upper_demand_action = []
        self._ep_lower_demand_action = []
        self._ep_shock_response_events = []
        self._freq_holdfb_events = {
            True: deque(maxlen=self.freq_holdfb_window),
            False: deque(maxlen=self.freq_holdfb_window),
        }
        self._ep_freq_holdfb_features = []
        self._ep_freq_driftfb_features = []
        self._ep_lower_wait_penalties = []
        self._ep_lower_board_credits = []
        self._ep_lower_board_credit_gates = []
        self._ep_lower_high_hold_penalties = []
        self._ep_lower_wait_net = []
        self._ep_upper_wait_credits = []
        self._ep_freq_wait_low_shares = []
        self._ep_freq_wait_lower_high_shares = []
        self._ep_freq_wait_lower_raw_credit_weights = []
        self._ep_freq_wait_boarded_pax = 0
        self._ep_trip_wait_stats = defaultdict(lambda: {
            'pax': 0,
            'wait_s': 0.0,
            'upper_wait_norm_sum': 0.0,
            'low_share_sum': 0.0,
            'events': 0,
        })
        self._ep_trip_records = []
        self._ep_dispatch_times = {'up': [], 'down': []}
        self._lower_drift_by_dir = {
            True: deque(maxlen=max(1, self.lower_drift_window)),
            False: deque(maxlen=max(1, self.lower_drift_window)),
        }
        self._ep_lower_drift_penalties = []
        self._ep_lower_drift_costs = []
        self._ep_lower_drift_loads = []
        self._ep_lower_drift_cost_adaptive_gate = []
        self._ep_upper_hf_penalties = []
        self._ep_upper_residual_value_costs = []
        self._ep_upper_residual_value_cost_active = []
        self._ep_upper_residual_selector_active = []
        self._ep_upper_residual_selector_adjusts = []
        self._ep_upper_residual_selector_margins = []
        self._ep_upper_residual_selector_actor_preds = []
        self._ep_upper_residual_selector_selected_preds = []
        self._ep_upper_residual_selector_feature_norms = []
        self._ep_headway_value_planner_active = []
        self._ep_headway_value_planner_adjusts = []
        self._ep_headway_value_planner_deltas = []
        self._ep_headway_value_planner_margins = []
        self._ep_headway_value_planner_actor_preds = []
        self._ep_headway_value_planner_selected_preds = []
        self._ep_headway_value_planner_priors = []
        self._ep_headway_value_planner_target_costs = []
        self._ep_headway_value_planner_feature_norms = []
        self._ep_upper_plan_penalties = []
        self._ep_upper_plan_targets = []
        self._ep_upper_plan_raw_delta_means = []
        self._ep_upper_plan_projected_delta_means = []
        self._ep_upper_plan_projected_delta_sums = []
        self._ep_upper_plan_decisions = 0
        self._ep_upper_plan_reuses = 0
        self._ep_terminal_launch_shifts = []
        self._ep_terminal_shift_caps = []
        self._ep_terminal_shift_mins = []
        self._ep_terminal_feedback_biases = []
        self._ep_terminal_value_selector_active = []
        self._ep_terminal_value_selector_biases = []
        self._ep_terminal_value_selector_margins = []
        self._ep_terminal_value_selector_actor_preds = []
        self._ep_terminal_value_selector_selected_preds = []
        self._ep_terminal_value_selector_feature_norms = []
        self._ep_terminal_value_selector_target_costs = []
        self._ep_cf_action_selector_active = []
        self._ep_cf_action_selector_changed = []
        self._ep_cf_action_selector_terminal_dispatch = []
        self._ep_cf_action_selector_deltas = []
        self._ep_cf_action_selector_confidences = []
        self._ep_terminal_headway_floors = []
        self._ep_fleet_noharm_upper_pressures = []
        self._ep_fleet_noharm_upper_adjusts = []
        self._ep_fleet_noharm_upper_gate_active = []
        self._ep_fleet_noharm_lower_pressures = []
        self._ep_fleet_noharm_lower_adjusts = []
        self._ep_fleet_noharm_lower_gate_active = []
        self._ep_fleet_noharm_lower_proactive_adjusts = []
        self._ep_fleet_noharm_lower_proactive_gate_active = []
        self._ep_fleet_noharm_lower_value_guard_adjusts = []
        self._ep_fleet_noharm_lower_value_guard_active = []
        self._ep_fleet_noharm_lower_value_guard_values = []
        self._ep_fleet_noharm_lower_value_guard_headway_values = []
        self._ep_fleet_noharm_lower_value_guard_costs = []
        self._ep_fleet_noharm_lower_value_soft_costs = []
        self._ep_fleet_noharm_lower_value_soft_active = []
        self._ep_fleet_noharm_lower_value_soft_values = []
        self._ep_fleet_noharm_lower_value_soft_headway_values = []
        self._ep_fleet_noharm_lower_value_soft_risks = []
        self._ep_fleet_noharm_lower_value_soft_violations = []
        self._active_timetable_plans = {}
        self._last_promotion_replan_launch = {}
        self._fixed_expert_active = self._select_fixed_expert_for_episode(
            ep, training=training)

        # v2k: elastic fleet sampling per-episode
        if N_fleet_override is not None:
            self._current_N_fleet = int(N_fleet_override)
        elif self.fleet_mode == 'elastic' and training:
            self._current_N_fleet = int(self.fleet_rng.randint(
                self.fleet_min, self.fleet_max + 1))
        else:
            self._current_N_fleet = self.N_fleet_default
        self.env._n_fleet_target = self._current_N_fleet

        learned_training = training and not self._fixed_expert_active
        upper_active = (
            ep >= self.upper_warmup and not self._fixed_expert_active)
        upper_training_active = upper_active and training
        if self._fixed_expert_active:
            self.env._upper_policy_callback = self._fixed_headway_callback
        else:
            self.env._upper_policy_callback = (
                self._upper_callback_v2 if upper_active else None)

        state_dict, reward_dict, _ = self.env.initialize_state()
        action_dict = {k: None for k in range(self.env.max_agent_num)}
        lower_last_action = {k: 0.0 for k in range(self.env.max_agent_num)}
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        while not self.env.done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        action_dict[key] = self._lower_action_for_agent(
                            state_dict[key][0],
                            key,
                            last_action=lower_last_action.get(key, 0.0),
                            deterministic=not training)

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        raw_state = np.array(state_dict[key][0], dtype=np.float32)
                        raw_next_state = np.array(state_dict[key][1], dtype=np.float32)
                        cur_bus = self._bus_for_agent(int(raw_state[0]))
                        transition_done = self._lower_terminal_action_masked(
                            cur_bus)
                        shaped_reward, transition_cost, act_val = (
                            self._record_lower_transition(
                                key=key,
                                raw_state=raw_state,
                                raw_next_state=raw_next_state,
                                action=action_dict[key],
                                reward=reward_dict[key],
                                cost=self.env.cost.get(key, 0.0),
                                previous_action=lower_last_action.get(key, 0.0),
                                transition_done=transition_done,
                                learned_training=learned_training,
                                bus=cur_bus,
                            )
                        )
                        episode_reward += shaped_reward
                        episode_cost += transition_cost
                        episode_steps += 1
                        lower_last_action[key] = act_val

                    state_dict[key] = state_dict[key][1:]
                    action_dict[key] = self._lower_action_for_agent(
                        state_dict[key][0],
                        key,
                        last_action=lower_last_action.get(key, 0.0),
                        deterministic=not training)

            state_dict, reward_dict, cost_dict, done = self.env.step(
                action_dict, render=False)
            completed_events = self.lower_lifecycle.process(
                self.env.bus_all,
                state_dict,
                action_dict,
                lower_last_action,
                self.holding_feedback,
            )
            for event in completed_events:
                self._ep_hold_feedback_trip_finalizations += int(
                    event.feedback_finalized)
                terminal_transition_recorded = False
                if self.lower_terminal_action_mode == 'transition':
                    has_pending = (
                        event.pending_state is not None
                        and event.pending_action is not None)
                    has_outcome = (
                        event.terminal_reward is not None
                        and event.terminal_cost is not None)
                    if has_pending and has_outcome:
                        terminal_reward, terminal_cost, _ = (
                            self._record_lower_transition(
                                key=event.bus_id,
                                raw_state=event.pending_state,
                                raw_next_state=None,
                                action=event.pending_action,
                                reward=event.terminal_reward,
                                cost=event.terminal_cost,
                                previous_action=event.previous_action_s,
                                transition_done=True,
                                learned_training=learned_training,
                                bus=event,
                                trip_id=event.trip_id,
                                direction=event.direction,
                                station_id=event.last_board_station_id,
                                board_wait_sum_s=event.last_board_wait_sum_s,
                                board_lf_wait_sum_s=(
                                    event.last_board_lf_wait_sum_s),
                                board_hf_wait_sum_s=(
                                    event.last_board_hf_wait_sum_s),
                                board_lf_mass=event.last_board_lf_mass,
                                board_hf_mass=event.last_board_hf_mass,
                                board_count=event.last_board_count,
                                record_holding_action=(
                                    not event.feedback_finalized),
                            )
                        )
                        episode_reward += terminal_reward
                        episode_cost += terminal_cost
                        episode_steps += 1
                        terminal_transition_recorded = True
                    elif event.pending_state is not None or event.pending_action is not None:
                        self._ep_lower_terminal_outcomes_missing += 1
                if self.lower_lifecycle.boundary_mode == 'reset':
                    self._ep_lower_trip_boundary_resets += 1
                    if terminal_transition_recorded:
                        self._ep_lower_pending_states_consumed += int(
                            event.pending_states_dropped)
                        self._ep_lower_pending_actions_consumed += int(
                            event.pending_action_dropped)
                    else:
                        self._ep_lower_pending_states_dropped += int(
                            event.pending_states_dropped)
                        self._ep_lower_pending_actions_dropped += int(
                            event.pending_action_dropped)

        env_time = time.time() - t0

        # ── Finalize trip holdings ──
        if self.lower_lifecycle.feedback_mode == 'episode_end':
            for bus in self.env.bus_all:
                if (not bus.on_route and hasattr(bus, 'applied_actions')
                        and bus.applied_actions):
                    self.holding_feedback.finalize_trip(
                        bus.trip_id, bus.direction)

        # ── Finalize last upper transition ──
        episode_end_time_s = float(getattr(self.env, 'current_time', 0.0))
        for stream_key, prev in list(self._prev_upper_states.items()):
            self._close_upper_transition_stream(
                stream_key,
                next_state=prev['s'],
                done=True,
                decision_time_s=episode_end_time_s,
            )

        # ── Hindsight Credit Assignment (v2g: gap-based, not holding-based) ──
        # Old: credit based on holding magnitude → corr(δ_t, hold)=0, BROKEN
        # New: credit based on dispatch gap uniformity → directly causal
        #   δ_t → dispatch timing → gap to neighbors → gap deviation = credit
        z = self.env.measurement_vector
        env_details = self.env.measurement_details
        N_fleet = self._current_N_fleet  # v2k: use episode's sampled budget
        episode_overshoot = max(0.0, float(z[1]) - float(N_fleet))
        service_cost_by_wait = service_cost_views(
            env_details,
            peak_fleet=float(z[1]),
            headway_cv=float(z[2]),
            n_fleet=float(N_fleet),
            weights=self.objective_weights,
        )
        observed_service_cost = service_cost_by_wait['observed']
        restricted_service_cost = service_cost_by_wait['restricted']
        episode_composite_cost = service_cost_by_wait[
            self.objective_wait_metric]
        # v2j: belief-weighted multi-objective scalarization (Option 1 BAMOR)
        sys_r, adj_w = self.compute_belief_weighted_reward(z, N_fleet)
        self._last_adj_weights = adj_w

        # Compute per-trip gap deviation using ACTUAL launch times from env
        trip_gap_devs = {}
        for dir_key_bool in [True, False]:  # direction
            dir_key = 'up' if dir_key_bool else 'down'
            # Get actual launch times from timetable objects
            launched = [(tt.launch_turn, tt._actual_launch_time)
                        for tt in self.env.timetables
                        if tt.launched and tt.direction == dir_key_bool
                        and hasattr(tt, '_actual_launch_time')]
            if len(launched) < 2:
                continue
            launched.sort(key=lambda x: x[1])  # sort by actual time
            tids = [l[0] for l in launched]
            times = [l[1] for l in launched]
            gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
            if not gaps:
                continue
            mean_gap = np.mean(gaps)
            std_gap = max(np.std(gaps), 1.0)
            # Assign deviation to each trip (trip i created gap[i-1])
            for i in range(len(tids)):
                if i == 0:
                    dev = abs(gaps[0] - mean_gap) / std_gap if gaps else 0.0
                elif i < len(gaps):
                    dev = (abs(gaps[i-1] - mean_gap) + abs(gaps[i] - mean_gap)) / (2 * std_gap)
                else:
                    dev = abs(gaps[-1] - mean_gap) / std_gap
                trip_gap_devs[tids[i]] = dev

        owner_by_tid = {
            int(tt.launch_turn): int(getattr(
                tt, '_freqduet_planned_by', tt.launch_turn))
            for tt in self.env.timetables
        }
        gap_values_by_owner = defaultdict(list)
        for tid, gap_dev in trip_gap_devs.items():
            gap_values_by_owner[owner_by_tid.get(int(tid), int(tid))].append(
                float(gap_dev))
        plan_gap_devs = {
            int(owner): float(np.mean(values))
            for owner, values in gap_values_by_owner.items()
            if values
        }

        backfilled = []
        prev_delta_by_dir = {}
        upper_wait_credits = self._upper_frequency_wait_credits(
            self._episode_upper_transitions)
        transition_ids = [
            int(trans['tid']) for trans in self._episode_upper_transitions]
        gap_credits = self.upper_credit_assignment.gap_credits(
            plan_gap_devs, transition_ids)
        if self.ablate_hindsight_credit:
            gap_credits = {tid: 0.0 for tid in transition_ids}
        system_rewards = self.upper_credit_assignment.system_rewards(
            sys_r, len(self._episode_upper_transitions))
        reliability_rewards = self.upper_credit_assignment.reliability_rewards(
            unserved_rate=float(env_details['passenger_unserved_rate']),
            incomplete_rate=(
                1.0 - float(env_details['trip_completion_rate'])),
            count=len(self._episode_upper_transitions),
        )
        interval_scores = self.upper_interval_credit.score_many(
            [
                trans.get('interval_outcome')
                for trans in self._episode_upper_transitions
            ],
            passengers_generated=int(env_details['passengers_generated']),
            episode_headway_samples=int(env_details['headway_sample_count']),
            episode_duration_s=float(
                self.env.protocol.evaluation_end_time_s),
            n_fleet_target=float(N_fleet),
        )
        for trans, system_reward, reliability_reward, interval_score in zip(
                self._episode_upper_transitions,
                system_rewards,
                reliability_rewards,
                interval_scores):
            tid = trans['tid']
            credit = float(gap_credits.get(int(tid), 0.0))
            a_u = float(trans.get(
                'a_eff', float(np.asarray(trans['a']).reshape(-1)[0])))
            upper_hf_pen = self._upper_delta_hf_penalty(
                trans.get('dir', True), a_u, prev_delta_by_dir)
            self._ep_upper_hf_penalties.append(upper_hf_pen)
            plan_pen = float(trans.get('plan_penalty', 0.0))
            upper_value_cost = float(trans.get('upper_value_cost', 0.0))
            upper_value_active = float(trans.get('upper_value_active', 0.0))
            self._ep_upper_residual_value_costs.append(upper_value_cost)
            self._ep_upper_residual_value_cost_active.append(
                upper_value_active)
            wait_credit = float(upper_wait_credits.get(int(tid), 0.0))
            self._ep_upper_wait_credits.append(wait_credit)
            interval_reward = float(interval_score['reward'])
            interval_wait_cost = float(interval_score['wait_cost'])
            interval_onboard_cost = float(interval_score['onboard_cost'])
            interval_dispatch_backlog_cost = float(
                interval_score['dispatch_backlog_cost'])
            interval_headway_cost = float(interval_score['headway_cost'])
            interval_fleet_cost = float(interval_score['fleet_cost'])
            interval_coverage = float(
                (trans.get('interval_outcome') or {}).get('coverage', 0.0))
            r = (
                float(system_reward) + float(reliability_reward)
                + credit + wait_credit + interval_reward
                - upper_hf_pen - plan_pen - upper_value_cost)
            upper_local_credit_cost = (
                -float(credit)
                - wait_credit
                - interval_reward
                - float(reliability_reward)
                + upper_hf_pen
                + plan_pen
                + upper_value_cost)
            terminal_value_target_cost = (
                self._terminal_value_selector_target_cost(
                    episode_composite_cost,
                    transition_reward=r,
                    local_credit_cost=upper_local_credit_cost))
            headway_value_target_cost = (
                self._headway_value_planner_target_cost(
                    episode_composite_cost,
                    transition_reward=r,
                    local_credit_cost=upper_local_credit_cost))
            if (training and int(self._current_ep) >= int(
                    self.upper_residual_selector_learn_start_ep)):
                self._update_upper_residual_selector(
                    trans.get('upper_residual_selector_x'), r)
            if (training and int(self._current_ep) >= int(
                    self.timetable_terminal_value_selector_learn_start_ep)):
                self._update_terminal_value_selector(
                    trans.get('terminal_value_selector_x'),
                    terminal_value_target_cost)
            if (training and int(self._current_ep) >= int(
                    self.timetable_headway_value_planner_learn_start_ep)):
                self._update_headway_value_planner(
                    trans.get('headway_value_planner_x'),
                    headway_value_target_cost)
            backfilled.append({
                's': trans['s'], 'a': trans['a'], 'r': r,
                'ns': trans['ns'], 'done': trans['done'], 'tid': tid,
                'dir': trans.get('dir', True),
                'duration_steps': trans.get('duration_steps', 1.0),
                'duration_s': trans.get('duration_s', 0.0),
                'system_reward': float(system_reward),
                'reliability_reward': float(reliability_reward),
                'gap_credit': credit,
                'transition_stream_key': trans.get(
                    'transition_stream_key', '__legacy_global__'),
                'interval_reward': interval_reward,
                'interval_wait_cost': interval_wait_cost,
                'interval_onboard_cost': interval_onboard_cost,
                'interval_dispatch_backlog_cost': (
                    interval_dispatch_backlog_cost),
                'interval_headway_cost': interval_headway_cost,
                'interval_fleet_cost': interval_fleet_cost,
                'interval_coverage': interval_coverage,
            })
            self._ep_upper_rewards.append(r)
            self._ep_upper_system_rewards.append(float(system_reward))
            self._ep_upper_gap_credits.append(credit)
            self._ep_upper_reliability_rewards.append(
                float(reliability_reward))
            self._ep_upper_interval_rewards.append(interval_reward)
            self._ep_upper_interval_wait_costs.append(interval_wait_cost)
            self._ep_upper_interval_onboard_costs.append(
                interval_onboard_cost)
            self._ep_upper_interval_dispatch_backlog_costs.append(
                interval_dispatch_backlog_cost)
            self._ep_upper_interval_headway_costs.append(
                interval_headway_cost)
            self._ep_upper_interval_fleet_costs.append(interval_fleet_cost)
            self._ep_upper_interval_coverages.append(interval_coverage)
        self._episode_upper_transitions = backfilled

        # ── Enrich per-trip records with holding + gap deviation ──
        upper_reward_by_owner = {
            int(trans['tid']): float(trans['r'])
            for trans in self._episode_upper_transitions
        }
        for rec in self._ep_trip_records:
            tid = rec['tid']
            stats = self.holding_feedback.get_trip_stats(tid)
            if stats:
                rec['hold_mean'] = round(stats['mean'], 1)
                rec['hold_std'] = round(stats['std'], 1)
                rec['hold_max'] = round(stats['max'], 1)
                rec['hold_n'] = stats['n_stops']
            else:
                rec['hold_mean'] = 0.0
                rec['hold_std'] = 0.0
                rec['hold_max'] = 0.0
                rec['hold_n'] = 0
            gap_dev = trip_gap_devs.get(tid, 0.0)
            rec['gap_dev'] = round(gap_dev, 3)
            rec['penalty'] = round(gap_dev, 3)  # now gap-based
            owner = owner_by_tid.get(int(tid), int(tid))
            rec['reward'] = round(
                upper_reward_by_owner.get(int(owner), 0.0), 3)

        # ══════════════ CS-BAPR: Belief Update ══════════════
        # Detect non-stationarity from upper-level timetable changes
        ep_reward_mean = episode_reward / max(episode_steps, 1)
        # Use Q_std from previous episode's training (history), or 0 if first ep
        prev_q_stds = self.history.get('lower_q_std', [])
        ep_q_std = prev_q_stds[-1] if prev_q_stds else 0.0
        ep_delta_mean = (np.mean(self._ep_upper_deltas)
                         if self._ep_upper_deltas else 0.0)

        if self._fixed_expert_active or not training:
            surprise = 0.0
        else:
            surprise = self.surprise_computer.compute(
                ep_reward_mean, ep_q_std, ep_delta_mean)
            self.belief_tracker.update(surprise)

        # Adaptive alpha: boost exploration after detected changepoint
        base_alpha = self.lower_trainer.alpha
        boosted_alpha = self.belief_tracker.adaptive_alpha_boost(
            base_alpha, max_boost=self.belief_alpha_boost_max)
        # Temporarily set alpha for this episode's training
        # (auto-entropy will correct it over time, this just gives a nudge)
        if (not self.ablate_csbapr and surprise > 0.5
                and upper_training_active):
            self.lower_trainer.alpha = min(boosted_alpha,
                                           self.lower_trainer.maximum_alpha)

        # ══════════════ Training ══════════════
        t1 = time.time()
        lower_m = {}
        upper_m = {}

        # ─── TPC: lazy-init EMA target upper at start of Phase 1 ───
        # We snapshot the current upper at end of warmup; subsequent Polyak
        # averaging keeps this "deployment" copy as a slow-moving anchor for
        # importance reweighting on the lower SAC.
        if (self.tpc_enable and upper_training_active
                and self.target_upper_trainer is None):
            self.target_upper_trainer = copy.deepcopy(self.upper_trainer)
            self.target_upper_trainer.replay_buffer.buffer.clear()
            print(f"  [TPC] initialised EMA target upper at ep {ep}")

        # Build per-sample IS weight function for lower SAC
        weight_fn = self._build_tpc_weight_fn() if self.tpc_enable else None

        # ─── v3: HAAR/PIPER tap signal for lower reward shaping ───
        # Each completed trip k gets a per-trip bonus β · clip(A_U(s_k, δ_k), -c, c) · f_k
        # where A_U is the upper advantage and f_k is the reachability gate.
        haar_tap_signal = None
        if self.coupling_mode == 'haar' and upper_training_active:
            haar_tap_signal = self._build_haar_tap_signal(trip_gap_devs)

        # Lower
        lower_policy_frozen = (
            self.freeze_lower_policy_after_ep is not None
            and ep >= self.freeze_lower_policy_after_ep)
        lower_critic_frozen = (
            self.freeze_lower_critic_after_ep is not None
            and ep >= self.freeze_lower_critic_after_ep)
        upper_policy_frozen = (
            self.freeze_upper_after_ep is not None
            and ep >= self.freeze_upper_after_ep)

        if (learned_training and not lower_critic_frozen
                and len(self.replay_buffer) > self.batch_size):
            for _ in range(self.updates_per_episode):
                lower_m = self.lower_trainer.update(
                    self.replay_buffer, self.batch_size, reward_scale=1.0,
                    weight_fn=weight_fn,
                    tap_signal=haar_tap_signal,
                    update_policy=not lower_policy_frozen)
        lower_m['lower_policy_frozen'] = 1.0 if lower_policy_frozen else 0.0
        lower_m['lower_critic_frozen'] = 1.0 if lower_critic_frozen else 0.0

        # Train reachability classifier (HAAR mode only)
        if (self.coupling_mode == 'haar' and self.haar_use_reach_gate
                and upper_training_active and self.reach_net is not None):
            self._train_reach_classifier(trip_gap_devs)

        # Upper
        if upper_training_active:
            for trans in self._episode_upper_transitions:
                self.upper_trainer.replay_buffer.push(
                    trans['s'], trans['a'], trans['r'], trans['ns'],
                    trans['done'],
                    duration_steps=trans.get('duration_steps', 1.0))
            if (not upper_policy_frozen
                    and len(self.upper_trainer.replay_buffer)
                    > self.upper_batch_size):
                for _ in range(self.upper_updates):
                    upper_m = self.upper_trainer.update(self.upper_batch_size)

            # ─── TPC: Polyak update EMA target after each upper training step ───
            if (not upper_policy_frozen and self.tpc_enable
                    and self.target_upper_trainer is not None):
                with torch.no_grad():
                    for p_t, p in zip(self.target_upper_trainer.policy_net.parameters(),
                                      self.upper_trainer.policy_net.parameters()):
                        p_t.data.mul_(1.0 - self.tpc_ema_tau).add_(
                            p.data, alpha=self.tpc_ema_tau)
        upper_m['upper_policy_frozen'] = 1.0 if upper_policy_frozen else 0.0

        # Measurement projection (z already computed above for upper reward)
        if learned_training:
            self.measurement_proj.update(z)
        theta_w = self.measurement_proj.get_reward_weights()

        train_time = time.time() - t1

        # ══════════════ Diagnostics ══════════════
        stage = "Warmup" if ep < self.upper_warmup else "BiLevel"
        hold_summary = self.holding_feedback.episode_summary
        hold_dir0 = self.holding_feedback.get_direction_stats(False)
        hold_dir1 = self.holding_feedback.get_direction_stats(True)
        if self._ep_freq_holdfb_features:
            freq_holdfb_arr = np.vstack(self._ep_freq_holdfb_features)
            freq_holdfb_mean = freq_holdfb_arr.mean(axis=0)
        else:
            freq_holdfb_mean = np.zeros(4, dtype=np.float64)
        if self._ep_freq_driftfb_features:
            freq_driftfb_arr = np.vstack(self._ep_freq_driftfb_features)
            freq_driftfb_mean = freq_driftfb_arr.mean(axis=0)
        else:
            freq_driftfb_mean = np.zeros(4, dtype=np.float64)
        la_stat = _stat(self._ep_lower_actions)
        lower_context_gate_stat = _stat(self._ep_lower_context_gate_values)
        lower_action_bins_gate_stat = _stat(
            self._ep_lower_action_bins_gate_values)
        lr_stat = _stat(self._ep_lower_rewards)
        ud_stat = _stat(self._ep_upper_deltas)
        ur_stat = _stat(self._ep_upper_rewards)
        upper_system_reward_stat = _stat(self._ep_upper_system_rewards)
        upper_gap_credit_stat = _stat(self._ep_upper_gap_credits)
        upper_interval_reward_stat = _stat(
            self._ep_upper_interval_rewards)
        upper_interval_coverage_stat = _stat(
            self._ep_upper_interval_coverages)
        upper_transition_duration_stat = _stat([
            float(trans.get('duration_steps', 1.0))
            for trans in self._episode_upper_transitions
        ])
        upper_transition_stream_count = len({
            str(trans.get('transition_stream_key', '__legacy_global__'))
            for trans in self._episode_upper_transitions
        })
        upper_transition_short_ratio = float(np.mean([
            float(trans.get('duration_steps', 1.0)) <= 0.250001
            for trans in self._episode_upper_transitions
        ])) if self._episode_upper_transitions else 0.0
        freq_summary = self.env.frequency_summary()
        lower_drift_stat = _stat(self._ep_lower_drift_penalties)
        lower_drift_cost_stat = _stat(self._ep_lower_drift_costs)
        lower_drift_load_stat = _stat(self._ep_lower_drift_loads)
        lower_load_hold_penalty_stat = _stat(
            self._ep_lower_load_hold_penalties)
        lower_load_ratio_stat = _stat(self._ep_lower_load_ratios)
        lower_normalized_person_delay_stat = _stat(
            self._ep_lower_normalized_person_delays)
        lower_causal_guard_active_stat = _stat(
            self._ep_lower_causal_guard_active)
        lower_causal_guard_limit_stat = _stat(
            self._ep_lower_causal_guard_limits)
        lower_causal_guard_adjustment_stat = _stat(
            self._ep_lower_causal_guard_adjustments)
        lower_drift_cost_adaptive_gate_stat = _stat(
            self._ep_lower_drift_cost_adaptive_gate)
        upper_hf_stat = _stat(self._ep_upper_hf_penalties)
        upper_value_cost_stat = _stat(
            self._ep_upper_residual_value_costs)
        upper_value_active_stat = _stat(
            self._ep_upper_residual_value_cost_active)
        upper_selector_active_stat = _stat(
            self._ep_upper_residual_selector_active)
        upper_selector_adjust_stat = _stat(
            self._ep_upper_residual_selector_adjusts)
        upper_selector_margin_stat = _stat(
            self._ep_upper_residual_selector_margins)
        upper_selector_actor_pred_stat = _stat(
            self._ep_upper_residual_selector_actor_preds)
        upper_selector_selected_pred_stat = _stat(
            self._ep_upper_residual_selector_selected_preds)
        upper_selector_feature_norm_stat = _stat(
            self._ep_upper_residual_selector_feature_norms)
        headway_planner_active_stat = _stat(
            self._ep_headway_value_planner_active)
        headway_planner_adjust_stat = _stat(
            self._ep_headway_value_planner_adjusts)
        headway_planner_delta_stat = _stat(
            self._ep_headway_value_planner_deltas)
        headway_planner_margin_stat = _stat(
            self._ep_headway_value_planner_margins)
        headway_planner_actor_pred_stat = _stat(
            self._ep_headway_value_planner_actor_preds)
        headway_planner_selected_pred_stat = _stat(
            self._ep_headway_value_planner_selected_preds)
        headway_planner_prior_stat = _stat(
            self._ep_headway_value_planner_priors)
        headway_planner_target_cost_stat = _stat(
            self._ep_headway_value_planner_target_costs)
        headway_planner_feature_norm_stat = _stat(
            self._ep_headway_value_planner_feature_norms)
        lower_wait_stat = _stat(self._ep_lower_wait_penalties)
        lower_board_credit_stat = _stat(self._ep_lower_board_credits)
        lower_board_credit_gate_stat = _stat(
            self._ep_lower_board_credit_gates)
        lower_hold_penalty_stat = _stat(self._ep_lower_high_hold_penalties)
        lower_wait_net_stat = _stat(self._ep_lower_wait_net)
        upper_wait_credit_stat = _stat(self._ep_upper_wait_credits)
        wait_low_share_stat = _stat(self._ep_freq_wait_low_shares)
        lower_high_share_stat = _stat(self._ep_freq_wait_lower_high_shares)
        lower_raw_weight_stat = _stat(
            self._ep_freq_wait_lower_raw_credit_weights)
        upper_plan_penalty_stat = _stat(self._ep_upper_plan_penalties)
        upper_plan_target_stat = _stat(self._ep_upper_plan_targets)
        upper_plan_raw_delta_stat = _stat(
            self._ep_upper_plan_raw_delta_means)
        upper_plan_projected_delta_stat = _stat(
            self._ep_upper_plan_projected_delta_means)
        upper_plan_projected_delta_sum_abs_stat = _stat([
            abs(value) for value in self._ep_upper_plan_projected_delta_sums
        ])
        terminal_launch_shift_stat = _stat(self._ep_terminal_launch_shifts)
        terminal_shift_cap_stat = _stat(self._ep_terminal_shift_caps)
        terminal_shift_min_stat = _stat(self._ep_terminal_shift_mins)
        terminal_feedback_bias_stat = _stat(
            self._ep_terminal_feedback_biases)
        terminal_selector_active_stat = _stat(
            self._ep_terminal_value_selector_active)
        terminal_selector_bias_stat = _stat(
            self._ep_terminal_value_selector_biases)
        terminal_selector_margin_stat = _stat(
            self._ep_terminal_value_selector_margins)
        terminal_selector_actor_pred_stat = _stat(
            self._ep_terminal_value_selector_actor_preds)
        terminal_selector_selected_pred_stat = _stat(
            self._ep_terminal_value_selector_selected_preds)
        terminal_selector_feature_norm_stat = _stat(
            self._ep_terminal_value_selector_feature_norms)
        terminal_selector_target_cost_stat = _stat(
            self._ep_terminal_value_selector_target_costs)
        snapshot_value_active_stat = _stat([
            float(rec.get('snapshot_value_active', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_changed_values = [
            1.0 if (
                float(rec.get('snapshot_value_active', 0.0)) > 0.5
                and str(rec.get('snapshot_value_method', ''))
                != str(self.snapshot_value_selector_fallback_method)
            ) else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_changed_stat = _stat(snapshot_value_changed_values)
        snapshot_value_override_values = [
            float(rec.get('snapshot_value_override_action', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_override_stat = _stat(snapshot_value_override_values)
        snapshot_value_terminal_dispatch_values = [
            float(rec.get('snapshot_value_terminal_dispatch', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_terminal_dispatch_stat = _stat(
            snapshot_value_terminal_dispatch_values)
        snapshot_value_terminal_bias_values = [
            float(rec.get('snapshot_value_terminal_bias_s', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_terminal_bias_stat = _stat(
            snapshot_value_terminal_bias_values)
        snapshot_value_margin_stat = _stat([
            float(rec.get('snapshot_value_margin', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_pred_stat = _stat([
            float(rec.get('snapshot_value_pred', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_baseline_pred_stat = _stat([
            float(rec.get('snapshot_value_baseline_pred', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_candidate_gate_cap_stat = _stat([
            float(rec.get('snapshot_value_candidate_gate_cap_s', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_candidate_gate_filtered_stat = _stat([
            float(rec.get('snapshot_value_candidate_gate_filtered', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_risk_score_stat = _stat([
            float(rec.get('snapshot_value_risk_score', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_risk_penalty_stat = _stat([
            float(rec.get('snapshot_value_risk_penalty', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_risk_penalty_max_stat = _stat([
            float(rec.get('snapshot_value_risk_penalty_max', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_guard_blocked_values = [
            float(rec.get('snapshot_value_guard_blocked', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_guard_blocked_stat = _stat(
            snapshot_value_guard_blocked_values)
        snapshot_value_guard_negative_target_values = [
            float(rec.get('snapshot_value_guard_negative_target', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_guard_negative_target_blocked_values = [
            float(rec.get('snapshot_value_guard_negative_target_blocked', 0.0))
            if float(rec.get('snapshot_value_active', 0.0)) > 0.5 else 0.0
            for rec in self._ep_trip_records
        ]
        snapshot_value_guard_negative_target_stat = _stat(
            snapshot_value_guard_negative_target_values)
        snapshot_value_guard_negative_target_blocked_stat = _stat(
            snapshot_value_guard_negative_target_blocked_values)
        snapshot_value_guard_prev_overshoot_stat = _stat([
            float(rec.get('snapshot_value_guard_prev_overshoot_norm', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_guard_fleet_pressure_stat = _stat([
            float(rec.get('snapshot_value_guard_fleet_pressure_norm', 0.0))
            for rec in self._ep_trip_records
        ])
        snapshot_value_guard_primary_bias_stat = _stat([
            float(rec.get('snapshot_value_guard_primary_bias_s', 0.0))
            for rec in self._ep_trip_records
        ])
        cf_action_active_stat = _stat(self._ep_cf_action_selector_active)
        cf_action_changed_stat = _stat(self._ep_cf_action_selector_changed)
        cf_action_terminal_dispatch_stat = _stat(
            self._ep_cf_action_selector_terminal_dispatch)
        cf_action_delta_stat = _stat(self._ep_cf_action_selector_deltas)
        cf_action_confidence_stat = _stat(
            self._ep_cf_action_selector_confidences)
        terminal_headway_floor_stat = _stat(
            self._ep_terminal_headway_floors)
        fleet_noharm_upper_pressure_stat = _stat(
            self._ep_fleet_noharm_upper_pressures)
        fleet_noharm_upper_adjust_stat = _stat(
            self._ep_fleet_noharm_upper_adjusts)
        fleet_noharm_upper_gate_stat = _stat(
            self._ep_fleet_noharm_upper_gate_active)
        fleet_noharm_lower_pressure_stat = _stat(
            self._ep_fleet_noharm_lower_pressures)
        fleet_noharm_lower_adjust_stat = _stat(
            self._ep_fleet_noharm_lower_adjusts)
        fleet_noharm_lower_gate_stat = _stat(
            self._ep_fleet_noharm_lower_gate_active)
        fleet_noharm_lower_proactive_adjust_stat = _stat(
            self._ep_fleet_noharm_lower_proactive_adjusts)
        fleet_noharm_lower_proactive_gate_stat = _stat(
            self._ep_fleet_noharm_lower_proactive_gate_active)
        fleet_noharm_lower_value_guard_adjust_stat = _stat(
            self._ep_fleet_noharm_lower_value_guard_adjusts)
        fleet_noharm_lower_value_guard_active_stat = _stat(
            self._ep_fleet_noharm_lower_value_guard_active)
        fleet_noharm_lower_value_guard_value_stat = _stat(
            self._ep_fleet_noharm_lower_value_guard_values)
        fleet_noharm_lower_value_guard_headway_stat = _stat(
            self._ep_fleet_noharm_lower_value_guard_headway_values)
        fleet_noharm_lower_value_guard_cost_stat = _stat(
            self._ep_fleet_noharm_lower_value_guard_costs)
        fleet_noharm_lower_value_soft_cost_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_costs)
        fleet_noharm_lower_value_soft_active_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_active)
        fleet_noharm_lower_value_soft_value_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_values)
        fleet_noharm_lower_value_soft_headway_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_headway_values)
        fleet_noharm_lower_value_soft_risk_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_risks)
        fleet_noharm_lower_value_soft_violation_stat = _stat(
            self._ep_fleet_noharm_lower_value_soft_violations)
        upper_hf_power_ratio = _upper_hf_power_ratio(
            self._ep_upper_deltas_by_dir, self.upper_lpf_window)
        lower_lf_drift_ratio = _lower_lf_drift_ratio(
            self._ep_lower_actions_by_dir, self.lower_drift_window)
        demand_attr_score = _demand_attribution_score(
            self._ep_upper_demand_action, self._ep_lower_demand_action)
        demand_attr_mi = demand_attribution_mi(
            self._ep_upper_demand_action,
            self._ep_lower_demand_action,
            bins=self.freq_diag_mi_bins)
        shock_metrics = shock_response_metrics(
            self._ep_shock_response_events,
            shock_threshold=self.freq_diag_shock_threshold,
            action_threshold_s=self.freq_diag_shock_action_threshold_s,
            response_window_s=self.freq_diag_shock_response_window_s,
            same_station=self.freq_diag_shock_same_station)
        plan_total = self._ep_upper_plan_decisions + self._ep_upper_plan_reuses
        plan_reuse_ratio = (
            self._ep_upper_plan_reuses / max(plan_total, 1)
            if self.timetable_planner is not None else 0.0)

        # Holding penalties across all trips this episode
        hold_pens = []
        for tid in self.holding_feedback._trip_actions:
            hold_pens.append(self.holding_feedback.holding_penalty(tid))
        hp_stat = _stat(hold_pens)

        row = {
            'ep': ep, 'stage': stage,
            'wall_env_s': round(env_time, 1),
            'wall_train_s': round(train_time, 1),
            # env
            'protocol_version': self.protocol_version,
            'config_fingerprint_sha256': self.config_fingerprint_sha256,
            'randomness_contract': self.randomness.mode,
            'randomness_fingerprint_sha256': self.randomness_manifest[
                'fingerprint_sha256'],
            'avg_wait_min': round(z[0], 3),
            'avg_wait_observed_min': round(
                float(env_details['avg_wait_observed_min']), 3),
            'restricted_wait_horizon_min': round(
                float(env_details['restricted_wait_horizon_min']), 3),
            'avg_wait_lf_observed_min': round(float(
                env_details.get('avg_wait_lf_observed_min', 0.0)), 6),
            'avg_wait_hf_observed_min': round(float(
                env_details.get('avg_wait_hf_observed_min', 0.0)), 6),
            'restricted_wait_lf_horizon_min': round(float(
                env_details.get('restricted_wait_lf_horizon_min', 0.0)), 6),
            'restricted_wait_hf_horizon_min': round(float(
                env_details.get('restricted_wait_hf_horizon_min', 0.0)), 6),
            'frequency_share_max_error': float(
                env_details.get('frequency_share_max_error', 0.0)),
            'avg_in_vehicle_observed_min': round(float(
                env_details.get('avg_in_vehicle_observed_min', 0.0)), 6),
            'restricted_in_vehicle_horizon_min': round(float(
                env_details.get('restricted_in_vehicle_horizon_min', 0.0)), 6),
            'avg_total_journey_observed_min': round(float(
                env_details.get('avg_total_journey_observed_min', 0.0)), 6),
            'restricted_total_journey_horizon_min': round(float(
                env_details.get('restricted_total_journey_horizon_min', 0.0)),
                6),
            'passengers_generated': int(
                env_details['passengers_generated']),
            'passengers_unserved': int(
                env_details['passengers_unserved']),
            'passenger_unserved_rate': round(
                float(env_details['passenger_unserved_rate']), 6),
            'headway_sample_count': int(
                env_details['headway_sample_count']),
            'headway_state_arrival_event_count': int(
                env_details.get('headway_state_arrival_event_count', 0)),
            'headway_state_spatial_fallback_count': int(
                env_details.get('headway_state_spatial_fallback_count', 0)),
            'headway_state_target_default_count': int(
                env_details.get('headway_state_target_default_count', 0)),
            'headway_state_arrival_event_rate': round(float(
                env_details.get('headway_state_arrival_event_rate', 0.0)), 6),
            'trips_unlaunched': int(env_details['trips_unlaunched']),
            'trip_launch_rate': round(
                float(env_details['trip_launch_rate']), 6),
            'trips_completed': int(env_details['trips_completed']),
            'trips_incomplete': int(env_details['trips_incomplete']),
            'trip_completion_rate': round(
                float(env_details['trip_completion_rate']), 6),
            'simulation_end_time_s': int(
                env_details['simulation_end_time_s']),
            'done_reason': str(env_details['done_reason']),
            'scenario_tape_id': str(env_details['scenario_tape_id']),
            'peak_fleet': int(z[1]),
            'fleet_inventory_mode': str(
                env_details.get('fleet_inventory_mode', 'elastic_legacy')),
            'physical_vehicle_count': int(
                env_details.get('physical_vehicle_count', len(self.env.bus_all))),
            'fleet_capacity': int(env_details.get(
                'fleet_capacity', self.env._fleet_capacity())),
            'fleet_ready_up': int(env_details.get('fleet_ready_up', 0)),
            'fleet_ready_down': int(env_details.get('fleet_ready_down', 0)),
            'fleet_denied_dispatch_events': int(
                env_details.get('fleet_denied_dispatch_events', 0)),
            'fleet_denied_retry_trip_seconds': round(float(
                env_details.get('fleet_denied_retry_trip_seconds', 0.0)), 6),
            'fleet_denied_trips': int(
                env_details.get('fleet_denied_trips', 0)),
            'fleet_denied_trip_rate': round(float(
                env_details.get('fleet_denied_trip_rate', 0.0)), 6),
            'fleet_readiness_delay_mean_s': round(float(
                env_details.get('fleet_readiness_delay_mean_s', 0.0)), 6),
            'fleet_readiness_delay_max_s': round(float(
                env_details.get('fleet_readiness_delay_max_s', 0.0)), 6),
            'holding_vehicle_seconds': round(float(
                env_details.get('holding_vehicle_seconds', 0.0)), 6),
            'holding_vehicle_seconds_per_launched_trip': round(float(
                env_details.get(
                    'holding_vehicle_seconds_per_launched_trip', 0.0)), 6),
            'holding_passenger_seconds': round(float(
                env_details.get('holding_passenger_seconds', 0.0)), 6),
            'holding_passenger_min_per_generated': round(float(
                env_details.get(
                    'holding_passenger_min_per_generated', 0.0)), 6),
            'commanded_holding_vehicle_seconds': round(float(
                env_details.get(
                    'commanded_holding_vehicle_seconds', 0.0)), 6),
            'commanded_holding_passenger_seconds': round(float(
                env_details.get(
                    'commanded_holding_passenger_seconds', 0.0)), 6),
            'commanded_holding_passenger_min_per_generated': round(float(
                env_details.get(
                    'commanded_holding_passenger_min_per_generated', 0.0)), 6),
            'terminal_actual_dispatch_gap_mean_s': round(float(
                env_details.get(
                    'terminal_actual_dispatch_gap_mean_s', 0.0)), 6),
            'terminal_dispatch_execution_error_mean_s': round(float(
                env_details.get(
                    'terminal_dispatch_execution_error_mean_s', 0.0)), 6),
            'terminal_dispatch_execution_error_abs_mean_s': round(float(
                env_details.get(
                    'terminal_dispatch_execution_error_abs_mean_s', 0.0)), 6),
            'invalid_headway_decisions_masked': int(
                env_details.get('invalid_headway_decisions_masked', 0)),
            'lower_observation_contract': str(
                env_details.get('lower_observation_contract',
                                self.lower_observation_contract)),
            'headway_reward_mode': str(
                env_details.get('headway_reward_mode',
                                self.lower_headway_reward_mode)),
            'frequency_observation_source': str(
                env_details.get('frequency_observation_source',
                                self.env.frequency_observation_source)),
            'lower_observation_ledger_hash': (
                self.lower_observation_spec.fingerprint),
            'headway_cv': round(z[2], 4),
            'service_cost': round(episode_composite_cost, 6),
            'service_cost_wait_metric': self.objective_wait_metric,
            'service_cost_observed': round(observed_service_cost, 6),
            'service_cost_restricted': round(restricted_service_cost, 6),
            'ep_reward': round(episode_reward, 3),
            'ep_cost': round(episode_cost, 3),
            'ep_steps': episode_steps,
            'n_dispatches': len(self._ep_upper_deltas) if upper_active else 0,
            # lower policy
            'lower_action_mean': round(la_stat['mean'], 2),
            'lower_action_std': round(la_stat['std'], 2),
            'lower_action_min': round(la_stat['min'], 2),
            'lower_action_max': round(la_stat['max'], 2),
            'lower_headway_state_mode': self.lower_headway_state_mode,
            'lower_state_input_schema': self.lower_state_input_schema,
            'lower_context_gate_enabled': int(getattr(
                self.env, 'lower_context_gate_enabled', False)),
            'lower_context_gate_active_mean': round(
                lower_context_gate_stat['mean'], 4),
            'lower_action_bins_gate_enabled': int(
                self.lower_action_bins_gate_enabled),
            'lower_action_bins_gate_active_mean': round(
                lower_action_bins_gate_stat['mean'], 4),
            'lower_reward_mean': round(lr_stat['mean'], 4),
            'lower_reward_std': round(lr_stat['std'], 4),
            'lower_load_hold_penalty_mean': round(
                lower_load_hold_penalty_stat['mean'], 6),
            'lower_load_hold_penalty_max': round(
                lower_load_hold_penalty_stat['max'], 6),
            'lower_load_ratio_mean': round(
                lower_load_ratio_stat['mean'], 6),
            'lower_normalized_person_delay_mean': round(
                lower_normalized_person_delay_stat['mean'], 6),
            'lower_causal_guard_enabled': int(
                self.lower_causal_holding_guard.enabled),
            'lower_causal_guard_evidence_mode': (
                self.lower_causal_holding_guard.evidence_mode),
            'lower_causal_guard_active_mean': round(
                lower_causal_guard_active_stat['mean'], 6),
            'lower_causal_guard_limit_mean_s': round(
                lower_causal_guard_limit_stat['mean'], 6),
            'lower_causal_guard_adjustment_mean_s': round(
                lower_causal_guard_adjustment_stat['mean'], 6),
            # lower training
            'lower_q_mean': lower_m.get('q_mean', 0.),
            'lower_q_std': lower_m.get('q_std', 0.),
            'lower_q_loss': lower_m.get('q_loss', 0.),
            'lower_q_mse': lower_m.get('q_mse', 0.),
            'lower_ood_loss': lower_m.get('ood_loss', 0.),
            'lower_q_l1': lower_m.get('q_l1', 0.),
            'lower_q_l1_penalty': lower_m.get('q_l1_penalty', 0.),
            'lower_cost_q_mean': lower_m.get('cost_q_mean', 0.),
            'lower_cost_q_loss': lower_m.get('cost_q_loss', 0.),
            'lower_policy_loss': lower_m.get('policy_loss', 0.),
            'lower_pi_grad_norm': lower_m.get('pi_grad_norm', 0.),
            'lower_q_grad_norm': lower_m.get('q_grad_norm', 0.),
            'lower_alpha': lower_m.get('alpha', 0.),
            'lower_lambda': lower_m.get('lambda', self.lower_trainer.lambda_param),
            'lower_replay_size': len(self.replay_buffer),
            'lower_trip_boundary_resets':
                self._ep_lower_trip_boundary_resets,
            'lower_pending_states_dropped':
                self._ep_lower_pending_states_dropped,
            'lower_pending_actions_dropped':
                self._ep_lower_pending_actions_dropped,
            'lower_pending_states_consumed':
                self._ep_lower_pending_states_consumed,
            'lower_pending_actions_consumed':
                self._ep_lower_pending_actions_consumed,
            'lower_terminal_action_masks':
                self._ep_lower_terminal_action_masks,
            'lower_terminal_transitions':
                self._ep_lower_terminal_transitions,
            'lower_terminal_outcomes_missing':
                self._ep_lower_terminal_outcomes_missing,
            'lower_policy_frozen': lower_m.get('lower_policy_frozen', 0.),
            'lower_critic_frozen': lower_m.get('lower_critic_frozen', 0.),
            # upper policy
            'upper_delta_mean': round(ud_stat['mean'], 2),
            'upper_delta_std': round(ud_stat['std'], 2),
            'upper_delta_min': round(ud_stat['min'], 2),
            'upper_delta_max': round(ud_stat['max'], 2),
            'upper_reward_mean': round(ur_stat['mean'], 4),
            'upper_reward_std': round(ur_stat['std'], 4),
            'upper_system_reward_mean': round(
                upper_system_reward_stat['mean'], 4),
            'upper_system_reward_sum': round(
                float(sum(self._ep_upper_system_rewards)), 4),
            'upper_reliability_reward_sum': round(
                float(sum(self._ep_upper_reliability_rewards)), 6),
            'upper_gap_credit_mean': round(
                upper_gap_credit_stat['mean'], 4),
            'upper_gap_credit_std': round(
                upper_gap_credit_stat['std'], 4),
            'upper_interval_reward_mean': round(
                upper_interval_reward_stat['mean'], 6),
            'upper_interval_reward_sum': round(
                float(sum(self._ep_upper_interval_rewards)), 6),
            'upper_interval_wait_cost_sum': round(
                float(sum(self._ep_upper_interval_wait_costs)), 6),
            'upper_interval_onboard_cost_sum': round(
                float(sum(self._ep_upper_interval_onboard_costs)), 6),
            'upper_interval_dispatch_backlog_cost_sum': round(
                float(sum(
                    self._ep_upper_interval_dispatch_backlog_costs)), 6),
            'upper_interval_headway_cost_sum': round(
                float(sum(self._ep_upper_interval_headway_costs)), 6),
            'upper_interval_fleet_cost_sum': round(
                float(sum(self._ep_upper_interval_fleet_costs)), 6),
            'upper_interval_coverage_mean': round(
                upper_interval_coverage_stat['mean'], 6),
            # upper training
            'upper_q_mean': upper_m.get('upper_q_mean', 0.),
            'upper_q_std': upper_m.get('upper_q_std', 0.),
            'upper_q_loss': upper_m.get('upper_q_loss', 0.),
            'upper_q_mse': upper_m.get('upper_q_mse', 0.),
            'upper_ood_loss': upper_m.get('upper_ood_loss', 0.),
            'upper_q_l1': upper_m.get('upper_q_l1', 0.),
            'upper_q_l1_penalty': upper_m.get(
                'upper_q_l1_penalty', 0.),
            'upper_duration_steps_mean': upper_m.get(
                'upper_duration_steps_mean', 0.),
            'upper_transition_duration_steps_mean':
                upper_transition_duration_stat['mean'],
            'upper_transition_stream_count':
                upper_transition_stream_count,
            'upper_transition_short_ratio':
                upper_transition_short_ratio,
            'upper_policy_loss': upper_m.get('upper_policy_loss', 0.),
            'upper_pi_grad_norm': upper_m.get('upper_pi_grad_norm', 0.),
            'upper_q_grad_norm': upper_m.get('upper_q_grad_norm', 0.),
            'upper_alpha': upper_m.get('upper_alpha', 0.),
            'upper_replay_size': len(self.upper_trainer.replay_buffer),
            'upper_policy_frozen': upper_m.get('upper_policy_frozen', 0.),
            # coupling
            'hold_fb_mean': hold_summary.get('mean', 0.),
            'hold_fb_std': hold_summary.get('std', 0.),
            'hold_fb_n_trips': hold_summary.get('n_trips', 0),
            'hold_fb_trip_finalizations':
                self._ep_hold_feedback_trip_finalizations,
            'hold_fb_dir0_mean': hold_dir0['rolling_mean'],
            'hold_fb_dir1_mean': hold_dir1['rolling_mean'],
            'hold_penalty_mean': hp_stat['mean'],
            'freq_holdfb_same_hold': float(freq_holdfb_mean[0]),
            'freq_holdfb_same_wait': float(freq_holdfb_mean[1]),
            'freq_holdfb_other_hold': float(freq_holdfb_mean[2]),
            'freq_holdfb_other_wait': float(freq_holdfb_mean[3]),
            'freq_holdfb_decisions': len(self._ep_freq_holdfb_features),
            'freq_driftfb_same_drift': float(freq_driftfb_mean[0]),
            'freq_driftfb_same_excess': float(freq_driftfb_mean[1]),
            'freq_driftfb_other_drift': float(freq_driftfb_mean[2]),
            'freq_driftfb_other_excess': float(freq_driftfb_mean[3]),
            'freq_driftfb_decisions': len(self._ep_freq_driftfb_features),
            'theta_wait': float(theta_w[0]),
            'theta_fleet': float(theta_w[1]),
            'theta_cv': float(theta_w[2]),
            # CS-BAPR belief
            'surprise': round(surprise, 4),
            'belief_window': round(self.belief_tracker.effective_window, 2),
            'belief_cp_prob': round(self.belief_tracker.changepoint_prob, 4),
            'belief_entropy': round(self.belief_tracker.entropy, 3),
            # v2j: belief-weighted MORL weights
            'w_wait': round(float(self._last_adj_weights[0]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            'w_fleet': round(float(self._last_adj_weights[1]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            'w_cv': round(float(self._last_adj_weights[2]), 3) if hasattr(self, '_last_adj_weights') else 0.,
            # v2k: elastic fleet
            'N_fleet': self._current_N_fleet,
            'fleet_overshoot': max(0, int(z[1]) - self._current_N_fleet),
            # FreqDuet
            'freq_low_demand': freq_summary['freq_low_demand'],
            'freq_low_slope': freq_summary['freq_low_slope'],
            'freq_low_forecast': freq_summary.get('freq_low_forecast', 0.0),
            'freq_high_energy': freq_summary['freq_high_energy'],
            'freq_middle': freq_summary.get('freq_middle', 0.0),
            'freq_middle_energy': freq_summary.get('freq_middle_energy', 0.0),
            'freq_od_entropy': freq_summary.get('freq_od_entropy', 0.0),
            'freq_od_high_energy': freq_summary.get('freq_od_high_energy', 0.0),
            'freq_od_active': freq_summary.get('freq_od_active', 0),
            'freq_updates': freq_summary['freq_updates'],
            'freq_promotion_flag': freq_summary.get('freq_promotion_flag', 0.0),
            'freq_promotion_strength': freq_summary.get('freq_promotion_strength', 0.0),
            'freq_promotion_age': freq_summary.get('freq_promotion_age', 0.0),
            'freq_promotion_score': freq_summary.get('freq_promotion_score', 0.0),
            'freq_promotion_active': freq_summary.get(
                'freq_promotion_active', 0.0),
            'freq_promotion_persistent': freq_summary.get(
                'freq_promotion_persistent', 0.0),
            'freq_promotion_ratio': freq_summary.get(
                'freq_promotion_ratio', 0.0),
            'freq_promotion_absorptions': freq_summary.get(
                'freq_promotion_absorptions', 0),
            'freq_promotion_absorbed': freq_summary.get(
                'freq_promotion_absorbed', 0.0),
            'lower_drift_signal_mode': self.lower_drift_signal_mode,
            'lower_drift_load_mean': lower_drift_load_stat['mean'],
            'lower_drift_load_max': lower_drift_load_stat['max'],
            'lower_drift_penalty_mean': lower_drift_stat['mean'],
            'lower_drift_penalty_max': lower_drift_stat['max'],
            'lower_drift_cost_mean': lower_drift_cost_stat['mean'],
            'lower_drift_cost_max': lower_drift_cost_stat['max'],
            'lower_drift_cost_adaptive_gate_mean':
                lower_drift_cost_adaptive_gate_stat['mean'],
            'lower_trip_hold_total_mean': hold_summary.get(
                'trip_total_mean', 0.0),
            'lower_trip_hold_total_std': hold_summary.get(
                'trip_total_std', 0.0),
            'lower_trip_hold_total_max': hold_summary.get(
                'trip_total_max', 0.0),
            'upper_hf_penalty_mean': upper_hf_stat['mean'],
            'upper_hf_penalty_max': upper_hf_stat['max'],
            'upper_residual_value_cost_mean':
                upper_value_cost_stat['mean'],
            'upper_residual_value_cost_max':
                upper_value_cost_stat['max'],
            'upper_residual_value_cost_active_mean':
                upper_value_active_stat['mean'],
            'upper_residual_selector_enabled':
                1.0 if self.upper_residual_selector_enable else 0.0,
            'upper_residual_selector_active_mean':
                upper_selector_active_stat['mean'],
            'upper_residual_selector_adjust_mean':
                upper_selector_adjust_stat['mean'],
            'upper_residual_selector_adjust_max':
                upper_selector_adjust_stat['max'],
            'upper_residual_selector_margin_mean':
                upper_selector_margin_stat['mean'],
            'upper_residual_selector_actor_pred_mean':
                upper_selector_actor_pred_stat['mean'],
            'upper_residual_selector_selected_pred_mean':
                upper_selector_selected_pred_stat['mean'],
            'upper_residual_selector_feature_norm_mean':
                upper_selector_feature_norm_stat['mean'],
            'upper_residual_selector_updates':
                int(self.upper_residual_selector_updates),
            'headway_value_planner_enabled':
                1.0 if self.timetable_headway_value_planner_enable else 0.0,
            'headway_value_planner_active_mean':
                headway_planner_active_stat['mean'],
            'headway_value_planner_adjust_mean':
                headway_planner_adjust_stat['mean'],
            'headway_value_planner_adjust_max':
                headway_planner_adjust_stat['max'],
            'headway_value_planner_delta_mean':
                headway_planner_delta_stat['mean'],
            'headway_value_planner_delta_max':
                headway_planner_delta_stat['max'],
            'headway_value_planner_margin_mean':
                headway_planner_margin_stat['mean'],
            'headway_value_planner_actor_pred_mean':
                headway_planner_actor_pred_stat['mean'],
            'headway_value_planner_selected_pred_mean':
                headway_planner_selected_pred_stat['mean'],
            'headway_value_planner_prior_mean':
                headway_planner_prior_stat['mean'],
            'headway_value_planner_target_cost_mean':
                headway_planner_target_cost_stat['mean'],
            'headway_value_planner_target_cost_max':
                headway_planner_target_cost_stat['max'],
            'headway_value_planner_feature_norm_mean':
                headway_planner_feature_norm_stat['mean'],
            'headway_value_planner_updates':
                int(self.timetable_headway_value_planner_updates),
            'upper_hf_power_ratio': upper_hf_power_ratio,
            'lower_lf_drift_ratio': lower_lf_drift_ratio,
            'demand_attr_score': demand_attr_score,
            'demand_attr_mi_score': demand_attr_mi['demand_attr_mi_score'],
            'demand_attr_mi_upper_low': demand_attr_mi[
                'demand_attr_mi_upper_low'],
            'demand_attr_mi_upper_high': demand_attr_mi[
                'demand_attr_mi_upper_high'],
            'demand_attr_mi_lower_high': demand_attr_mi[
                'demand_attr_mi_lower_high'],
            'demand_attr_mi_lower_low': demand_attr_mi[
                'demand_attr_mi_lower_low'],
            'shock_response_time_mean_s': shock_metrics[
                'shock_response_time_mean_s'],
            'shock_response_time_std_s': shock_metrics[
                'shock_response_time_std_s'],
            'shock_response_hit_rate': shock_metrics[
                'shock_response_hit_rate'],
            'shock_events': shock_metrics['shock_events'],
            'shock_action_mean_s': shock_metrics['shock_action_mean_s'],
            'freq_wait_lower_penalty_mean': lower_wait_stat['mean'],
            'freq_wait_lower_penalty_max': lower_wait_stat['max'],
            'freq_wait_lower_board_credit_mean': lower_board_credit_stat['mean'],
            'freq_wait_lower_board_credit_max': lower_board_credit_stat['max'],
            'freq_wait_lower_board_credit_gate_mean':
                lower_board_credit_gate_stat['mean'],
            'freq_wait_lower_hold_penalty_mean': lower_hold_penalty_stat['mean'],
            'freq_wait_lower_hold_penalty_max': lower_hold_penalty_stat['max'],
            'freq_wait_lower_net_mean': lower_wait_net_stat['mean'],
            'freq_wait_upper_credit_mean': upper_wait_credit_stat['mean'],
            'freq_wait_upper_credit_std': upper_wait_credit_stat['std'],
            'freq_wait_low_share_mean': wait_low_share_stat['mean'],
            'freq_wait_lower_high_share_mean': lower_high_share_stat['mean'],
            'freq_wait_lower_raw_credit_weight_mean': lower_raw_weight_stat['mean'],
            'freq_wait_boarded_pax': int(self._ep_freq_wait_boarded_pax),
            'upper_plan_penalty_mean': upper_plan_penalty_stat['mean'],
            'upper_plan_penalty_max': upper_plan_penalty_stat['max'],
            'upper_plan_target_mean': upper_plan_target_stat['mean'],
            'upper_plan_target_std': upper_plan_target_stat['std'],
            'upper_plan_decisions': self._ep_upper_plan_decisions,
            'upper_plan_reuse_ratio': plan_reuse_ratio,
            'upper_plan_projection_mode': (
                self.timetable_planner.terminal_schedule_mode
                if self.timetable_planner is not None else 'disabled'),
            'upper_interval_wait_ownership': (
                self.upper_interval_credit.wait_ownership
                if self.upper_interval_credit.enabled else 'disabled'),
            'upper_plan_headway_budget_mode': (
                self.timetable_planner.headway_budget_mode
                if self.timetable_planner is not None else 'disabled'),
            'upper_plan_raw_delta_mean_s': round(
                upper_plan_raw_delta_stat['mean'], 6),
            'upper_plan_projected_delta_mean_s': round(
                upper_plan_projected_delta_stat['mean'], 6),
            'upper_plan_projected_delta_sum_abs_mean_s': round(
                upper_plan_projected_delta_sum_abs_stat['mean'], 6),
            'terminal_launch_shift_mean': terminal_launch_shift_stat['mean'],
            'terminal_launch_shift_std': terminal_launch_shift_stat['std'],
            'terminal_shift_cap_mean': terminal_shift_cap_stat['mean'],
            'terminal_shift_cap_max': terminal_shift_cap_stat['max'],
            'terminal_shift_min_mean': terminal_shift_min_stat['mean'],
            'terminal_shift_min_min': terminal_shift_min_stat['min'],
            'terminal_feedback_bias_mean': terminal_feedback_bias_stat['mean'],
            'terminal_feedback_bias_max': terminal_feedback_bias_stat['max'],
            'terminal_feedback_events': int(terminal_feedback_bias_stat['n']),
            'terminal_value_selector_enabled':
                1.0 if self.timetable_terminal_value_selector_enable else 0.0,
            'terminal_value_selector_active_mean':
                terminal_selector_active_stat['mean'],
            'terminal_value_selector_bias_mean':
                terminal_selector_bias_stat['mean'],
            'terminal_value_selector_bias_max':
                terminal_selector_bias_stat['max'],
            'terminal_value_selector_margin_mean':
                terminal_selector_margin_stat['mean'],
            'terminal_value_selector_actor_pred_mean':
                terminal_selector_actor_pred_stat['mean'],
            'terminal_value_selector_selected_pred_mean':
                terminal_selector_selected_pred_stat['mean'],
            'terminal_value_selector_feature_norm_mean':
                terminal_selector_feature_norm_stat['mean'],
            'terminal_value_selector_target_cost_mean':
                terminal_selector_target_cost_stat['mean'],
            'terminal_value_selector_target_cost_max':
                terminal_selector_target_cost_stat['max'],
            'terminal_value_selector_updates':
                int(self.timetable_terminal_value_selector_updates),
            'snapshot_value_selector_enabled':
                1.0 if self.snapshot_value_selector_enable else 0.0,
            'snapshot_value_active_mean':
                snapshot_value_active_stat['mean'],
            'snapshot_value_events': int(sum(
                1 for rec in self._ep_trip_records
                if float(rec.get('snapshot_value_active', 0.0)) > 0.5)),
            'snapshot_value_changed_mean':
                snapshot_value_changed_stat['mean'],
            'snapshot_value_changed_events': int(sum(
                1 for value in snapshot_value_changed_values
                if float(value) > 0.5)),
            'snapshot_value_override_mean':
                snapshot_value_override_stat['mean'],
            'snapshot_value_override_events': int(sum(
                1 for value in snapshot_value_override_values
                if float(value) > 0.5)),
            'snapshot_value_terminal_dispatch_mean':
                snapshot_value_terminal_dispatch_stat['mean'],
            'snapshot_value_terminal_dispatch_events': int(sum(
                1 for value in snapshot_value_terminal_dispatch_values
                if float(value) > 0.5)),
            'snapshot_value_terminal_bias_mean':
                snapshot_value_terminal_bias_stat['mean'],
            'snapshot_value_terminal_bias_max':
                snapshot_value_terminal_bias_stat['max'],
            'snapshot_value_terminal_bias_events': int(sum(
                1 for value in snapshot_value_terminal_bias_values
                if float(value) > 1e-9)),
            'snapshot_value_margin_mean':
                snapshot_value_margin_stat['mean'],
            'snapshot_value_margin_max':
                snapshot_value_margin_stat['max'],
            'snapshot_value_pred_mean':
                snapshot_value_pred_stat['mean'],
            'snapshot_value_baseline_pred_mean':
                snapshot_value_baseline_pred_stat['mean'],
            'snapshot_value_candidate_gate_cap_mean':
                snapshot_value_candidate_gate_cap_stat['mean'],
            'snapshot_value_candidate_gate_filtered_mean':
                snapshot_value_candidate_gate_filtered_stat['mean'],
            'snapshot_value_risk_score_mean':
                snapshot_value_risk_score_stat['mean'],
            'snapshot_value_risk_penalty_mean':
                snapshot_value_risk_penalty_stat['mean'],
            'snapshot_value_risk_penalty_max_mean':
                snapshot_value_risk_penalty_max_stat['mean'],
            'snapshot_value_guard_blocked_mean':
                snapshot_value_guard_blocked_stat['mean'],
            'snapshot_value_guard_blocked_events': int(sum(
                1 for value in snapshot_value_guard_blocked_values
                if float(value) > 0.5)),
            'snapshot_value_guard_negative_target_mean':
                snapshot_value_guard_negative_target_stat['mean'],
            'snapshot_value_guard_negative_target_events': int(sum(
                1 for value in snapshot_value_guard_negative_target_values
                if float(value) > 0.5)),
            'snapshot_value_guard_negative_target_blocked_mean':
                snapshot_value_guard_negative_target_blocked_stat['mean'],
            'snapshot_value_guard_negative_target_blocked_events': int(sum(
                1 for value in snapshot_value_guard_negative_target_blocked_values
                if float(value) > 0.5)),
            'snapshot_value_guard_prev_overshoot_norm_mean':
                snapshot_value_guard_prev_overshoot_stat['mean'],
            'snapshot_value_guard_fleet_pressure_norm_mean':
                snapshot_value_guard_fleet_pressure_stat['mean'],
            'snapshot_value_guard_primary_bias_mean':
                snapshot_value_guard_primary_bias_stat['mean'],
            'cf_action_selector_enabled':
                1.0 if self.cf_action_selector_enable else 0.0,
            'cf_action_selector_active_mean':
                cf_action_active_stat['mean'],
            'cf_action_selector_events': int(sum(
                1 for value in self._ep_cf_action_selector_active
                if float(value) > 0.5)),
            'cf_action_selector_changed_mean':
                cf_action_changed_stat['mean'],
            'cf_action_selector_terminal_dispatch_mean':
                cf_action_terminal_dispatch_stat['mean'],
            'cf_action_selector_delta_mean':
                cf_action_delta_stat['mean'],
            'cf_action_selector_delta_std':
                cf_action_delta_stat['std'],
            'cf_action_selector_confidence_mean':
                cf_action_confidence_stat['mean'],
            'terminal_headway_floor_mean':
                terminal_headway_floor_stat['mean'],
            'terminal_headway_floor_events':
                int(terminal_headway_floor_stat['n']),
            'fleet_noharm_upper_pressure_mean':
                fleet_noharm_upper_pressure_stat['mean'],
            'fleet_noharm_upper_adjust_mean':
                fleet_noharm_upper_adjust_stat['mean'],
            'fleet_noharm_upper_events': int(sum(
                1 for v in self._ep_fleet_noharm_upper_adjusts
                if float(v) > 1e-6)),
            'fleet_noharm_upper_gate_active_mean':
                fleet_noharm_upper_gate_stat['mean'],
            'fleet_noharm_lower_pressure_mean':
                fleet_noharm_lower_pressure_stat['mean'],
            'fleet_noharm_lower_adjust_mean':
                fleet_noharm_lower_adjust_stat['mean'],
            'fleet_noharm_lower_events': int(sum(
                1 for v in self._ep_fleet_noharm_lower_adjusts
                if float(v) > 1e-6)),
            'fleet_noharm_lower_gate_active_mean':
                fleet_noharm_lower_gate_stat['mean'],
            'fleet_noharm_lower_proactive_adjust_mean':
                fleet_noharm_lower_proactive_adjust_stat['mean'],
            'fleet_noharm_lower_proactive_events': int(sum(
                1 for v in self._ep_fleet_noharm_lower_proactive_adjusts
                if float(v) > 1e-6)),
            'fleet_noharm_lower_proactive_gate_active_mean':
                fleet_noharm_lower_proactive_gate_stat['mean'],
            'fleet_noharm_lower_value_guard_adjust_mean':
                fleet_noharm_lower_value_guard_adjust_stat['mean'],
            'fleet_noharm_lower_value_guard_events': int(sum(
                1 for v in self._ep_fleet_noharm_lower_value_guard_adjusts
                if float(v) > 1e-6)),
            'fleet_noharm_lower_value_guard_active_mean':
                fleet_noharm_lower_value_guard_active_stat['mean'],
            'fleet_noharm_lower_value_guard_value_mean':
                fleet_noharm_lower_value_guard_value_stat['mean'],
            'fleet_noharm_lower_value_guard_headway_mean':
                fleet_noharm_lower_value_guard_headway_stat['mean'],
            'fleet_noharm_lower_value_guard_cost_mean':
                fleet_noharm_lower_value_guard_cost_stat['mean'],
            'fleet_noharm_lower_value_soft_cost_mean':
                fleet_noharm_lower_value_soft_cost_stat['mean'],
            'fleet_noharm_lower_value_soft_cost_max':
                fleet_noharm_lower_value_soft_cost_stat['max'],
            'fleet_noharm_lower_value_soft_events': int(sum(
                1 for v in self._ep_fleet_noharm_lower_value_soft_costs
                if float(v) > 1e-6)),
            'fleet_noharm_lower_value_soft_active_mean':
                fleet_noharm_lower_value_soft_active_stat['mean'],
            'fleet_noharm_lower_value_soft_value_mean':
                fleet_noharm_lower_value_soft_value_stat['mean'],
            'fleet_noharm_lower_value_soft_headway_mean':
                fleet_noharm_lower_value_soft_headway_stat['mean'],
            'fleet_noharm_lower_value_soft_risk_mean':
                fleet_noharm_lower_value_soft_risk_stat['mean'],
            'fleet_noharm_lower_value_soft_violation_mean':
                fleet_noharm_lower_value_soft_violation_stat['mean'],
        }
        composite_cost = episode_composite_cost
        if (training
                and int(ep) >= int(self._fixed_selector_update_start_ep())):
            self._update_fixed_expert_selector(
                self._fixed_expert_active, composite_cost)
        row['fixed_selector_fixed_active'] = (
            1.0 if self._fixed_expert_active else 0.0)
        row['fixed_selector_learned_cost_ema'] = (
            self.fixed_selector_cost_ema['learned']
            if self.fixed_selector_cost_ema['learned'] is not None else 0.0)
        row['fixed_selector_fixed_cost_ema'] = (
            self.fixed_selector_cost_ema['fixed']
            if self.fixed_selector_cost_ema['fixed'] is not None else 0.0)
        row['fixed_selector_learned_count'] = int(
            self.fixed_selector_counts['learned'])
        row['fixed_selector_fixed_count'] = int(
            self.fixed_selector_counts['fixed'])
        row['fixed_selector_context_enabled'] = (
            1.0 if self.fixed_selector_context_enable else 0.0)
        row['fixed_selector_context_learned_value'] = float(
            self._fixed_selector_context_learned_value)
        row['fixed_selector_context_fixed_value'] = float(
            self._fixed_selector_context_fixed_value)
        row['fixed_selector_context_margin'] = float(
            self._fixed_selector_context_margin)
        context = self._fixed_selector_current_context
        row['fixed_selector_context_feature_norm'] = (
            float(np.linalg.norm(context)) if context is not None else 0.0)
        if training:
            self._fixed_selector_prev_diag = dict(row)
        if record_diagnostics and self.diag is not None:
            self.diag.append(row)

        # Also keep lightweight history for quick plotting
        if record_diagnostics:
            for k in ['avg_wait_min', 'peak_fleet', 'headway_cv',
                   'lower_lambda', 'lower_alpha', 'lower_q_mean', 'lower_q_std',
                   'upper_delta_mean', 'upper_q_mean',
                   'hold_fb_mean', 'hold_penalty_mean',
                   'freq_holdfb_same_hold', 'freq_holdfb_same_wait',
                   'freq_driftfb_same_drift', 'freq_driftfb_same_excess',
                   'theta_wait', 'theta_fleet',
                   'surprise', 'belief_window',
                   'upper_hf_power_ratio', 'lower_lf_drift_ratio',
                   'upper_residual_value_cost_mean',
                   'upper_residual_selector_active_mean',
                   'upper_residual_selector_adjust_mean',
                   'upper_residual_selector_margin_mean',
                   'demand_attr_score', 'demand_attr_mi_score',
                   'shock_response_time_mean_s',
                   'shock_response_hit_rate',
                   'upper_plan_penalty_mean',
                   'freq_wait_lower_penalty_mean',
                   'freq_wait_lower_board_credit_mean',
                   'freq_wait_lower_board_credit_gate_mean',
                   'freq_wait_lower_hold_penalty_mean',
                   'freq_wait_lower_net_mean',
                   'freq_wait_upper_credit_mean',
                   'freq_wait_low_share_mean',
                   'freq_wait_lower_high_share_mean',
                   'freq_wait_lower_raw_credit_weight_mean',
                   'freq_middle', 'freq_middle_energy',
                   'upper_plan_target_mean', 'upper_plan_decisions',
                   'upper_plan_reuse_ratio', 'terminal_launch_shift_mean',
                   'terminal_feedback_bias_mean',
                   'snapshot_value_terminal_bias_mean',
                   'freq_promotion_flag', 'freq_promotion_strength',
                   'freq_promotion_active', 'freq_promotion_persistent',
                   'freq_promotion_ratio',
                    'freq_promotion_absorbed']:
                self.history[k].append(row[k])

        return row

    # ────────────────── Periodic deep dump ──────────────────

    def _print_diagnostic_block(self, row):
        """Print a detailed diagnostic block (every N episodes)."""
        ep = row['ep']
        print(f"\n{'─'*90}")
        print(f"  DIAGNOSTIC  ep={ep}  stage={row['stage']}  "
              f"wall={row['wall_env_s']}+{row['wall_train_s']}s")
        print(f"{'─'*90}")

        print(f"  ENV      wait={row['avg_wait_min']:.2f}m  fleet={row['peak_fleet']}  "
              f"cv={row['headway_cv']:.3f}  "
              f"R={row['ep_reward']:.2f}  C={row['ep_cost']:.2f}  "
              f"steps={row['ep_steps']}")

        print(f"  LOWER π  action μ={row['lower_action_mean']:.1f} "
              f"σ={row['lower_action_std']:.1f} "
              f"[{row['lower_action_min']:.1f}, {row['lower_action_max']:.1f}]  "
              f"reward μ={row['lower_reward_mean']:.3f} "
              f"σ={row['lower_reward_std']:.3f}")

        print(f"  LOWER Q  Q={row['lower_q_mean']:.2f}±{row['lower_q_std']:.2f}  "
              f"loss={row['lower_q_loss']:.4f} (mse={row['lower_q_mse']:.4f} "
              f"ood={row['lower_ood_loss']:.4f})  "
              f"CQ={row['lower_cost_q_mean']:.3f} "
              f"CQ_loss={row['lower_cost_q_loss']:.4f}")

        print(f"  LOWER ∇  π_grad={row['lower_pi_grad_norm']:.4f}  "
              f"Q_grad={row['lower_q_grad_norm']:.4f}  "
              f"α={row['lower_alpha']:.4f}  λ={row['lower_lambda']:.3f}  "
              f"buf={row['lower_replay_size']}")

        if row['stage'] == 'BiLevel':
            print(f"  UPPER δ  δ_t μ={row['upper_delta_mean']:.1f} "
                  f"σ={row['upper_delta_std']:.1f} "
                  f"[{row['upper_delta_min']:.1f}, {row['upper_delta_max']:.1f}]  "
                  f"reward μ={row['upper_reward_mean']:.3f} "
                  f"σ={row['upper_reward_std']:.3f}  "
                  f"n={row['n_dispatches']}")

            print(f"  UPPER Q  Q={row['upper_q_mean']:.3f}±{row['upper_q_std']:.3f}  "
                  f"loss={row['upper_q_loss']:.4f} (mse={row['upper_q_mse']:.4f} "
                  f"ood={row['upper_ood_loss']:.4f})  "
                  f"π_loss={row['upper_policy_loss']:.4f}")

            print(f"  UPPER ∇  π_grad={row['upper_pi_grad_norm']:.4f}  "
                  f"Q_grad={row['upper_q_grad_norm']:.4f}  "
                  f"α={row['upper_alpha']:.4f}  buf={row['upper_replay_size']}")

        print(f"  COUPLE   hold μ={row['hold_fb_mean']:.1f}s  "
              f"σ={row['hold_fb_std']:.1f}  "
              f"n_trips={row['hold_fb_n_trips']}  "
              f"dir0={row['hold_fb_dir0_mean']:.1f}  "
              f"dir1={row['hold_fb_dir1_mean']:.1f}  "
              f"penalty={row['hold_penalty_mean']:.3f}")
        if self.freq_holdfb_enable:
            print(f"  F-HOLDFB same={row.get('freq_holdfb_same_hold',0):.3f}/"
                  f"{row.get('freq_holdfb_same_wait',0):.3f}  "
                  f"other={row.get('freq_holdfb_other_hold',0):.3f}/"
                  f"{row.get('freq_holdfb_other_wait',0):.3f}  "
                  f"n={int(row.get('freq_holdfb_decisions',0))}")
        if self.freq_driftfb_enable:
            print(f"  D-FB     same={row.get('freq_driftfb_same_drift',0):.3f}/"
                  f"{row.get('freq_driftfb_same_excess',0):.3f}  "
                  f"other={row.get('freq_driftfb_other_drift',0):.3f}/"
                  f"{row.get('freq_driftfb_other_excess',0):.3f}  "
                  f"n={int(row.get('freq_driftfb_decisions',0))}")

        print(f"  θ-OGD    w=[{row['theta_wait']:.3f}, "
              f"{row['theta_fleet']:.3f}, {row['theta_cv']:.3f}]")

        print(f"  BELIEF   surprise={row.get('surprise',0):.3f}  "
              f"window={row.get('belief_window',0):.1f}  "
              f"cp_prob={row.get('belief_cp_prob',0):.3f}  "
              f"entropy={row.get('belief_entropy',0):.2f}")
        print(f"  FREQ     low={row.get('freq_low_demand',0):.3f}  "
              f"forecast={row.get('freq_low_forecast',0):.3f}  "
              f"hf_energy={row.get('freq_high_energy',0):.3f}  "
              f"mid={row.get('freq_middle',0):+.3f}/"
              f"{row.get('freq_middle_energy',0):.3f}  "
              f"odH={row.get('freq_od_entropy',0):.3f}  "
              f"U_HF={row.get('upper_hf_power_ratio',0):.3f}  "
              f"L_LF={row.get('lower_lf_drift_ratio',0):.3f}  "
              f"attr={row.get('demand_attr_score',0):.3f}  "
              f"MI={row.get('demand_attr_mi_score',0):+.3f}  "
              f"shock={row.get('shock_response_time_mean_s',0):.0f}s/"
              f"{row.get('shock_response_hit_rate',0):.2f}  "
              f"prom={row.get('freq_promotion_flag',0):.0f}/"
              f"{row.get('freq_promotion_strength',0):.2f}  "
              f"absorb={row.get('freq_promotion_absorbed',0):+.3f}")
        if self.freq_wait_enable:
            print(f"  WAIT-F   lower_pen={row.get('freq_wait_lower_penalty_mean',0):.4f}  "
                  f"board_credit={row.get('freq_wait_lower_board_credit_mean',0):+.4f}  "
                  f"hold_pen={row.get('freq_wait_lower_hold_penalty_mean',0):.4f}  "
                  f"net={row.get('freq_wait_lower_net_mean',0):+.4f}  "
                  f"upper_credit={row.get('freq_wait_upper_credit_mean',0):+.4f}"
                  f"±{row.get('freq_wait_upper_credit_std',0):.4f}  "
                  f"low_share={row.get('freq_wait_low_share_mean',0):.3f}  "
                  f"lower_hshare={row.get('freq_wait_lower_high_share_mean',0):.3f}  "
                  f"pax={int(row.get('freq_wait_boarded_pax',0))}")
        if self.timetable_terminal_dispatch:
            print(f"  TERM     launch_shift={row.get('terminal_launch_shift_mean',0):+.1f}"
                  f"±{row.get('terminal_launch_shift_std',0):.1f}s  "
                  f"cap={row.get('terminal_shift_cap_mean',0):.1f}s  "
                  f"fb_bias={row.get('terminal_feedback_bias_mean',0):.1f}s")
        print(f"{'─'*90}\n")

    # ────────────────── Per-trip dump ──────────────────

    def _dump_trip_breakdown(self, ep):
        """Write per-trip detail to CSV and print summary."""
        if not self._ep_trip_records:
            return

        trip_csv = os.path.join(self.log_dir, 'trip_details.csv')
        write_header = not os.path.exists(trip_csv)
        fields = ['ep', 'tid', 'dir', 'hour', 'period', 'delta_t',
                  'base_hw', 'eff_hw', 's_hold_mean', 's_hold_std',
                  'upper_decision', 'promotion_replan', 'launch_shift',
                  'effective_launch', 'terminal_gap_now',
                  'terminal_short_gap', 'terminal_over_gap',
                  'fleet_concurrent', 'fleet_target', 'fleet_pressure',
                  'waiting_total', 'freq_low_demand', 'freq_low_forecast',
                  'freq_high_energy', 'freq_middle_energy',
                  'freq_od_entropy', 'freq_promotion_strength',
                  'freq_promotion_active',
                  'cf_action_selector_active',
                  'cf_action_selector_method',
                  'cf_action_selector_delta_s',
                  'cf_action_selector_actor_delta_s',
                  'cf_action_selector_changed',
                  'cf_action_selector_terminal_dispatch',
                  'cf_action_selector_confidence',
                  'cf_action_selector_node_id',
                  'snapshot_value_active', 'snapshot_value_method',
                  'snapshot_value_override_action',
                  'snapshot_value_terminal_dispatch',
                  'snapshot_value_terminal_bias_s',
                  'snapshot_value_pred', 'snapshot_value_baseline_pred',
                  'snapshot_value_margin',
                  'snapshot_value_guard_blocked',
                  'snapshot_value_guard_negative_target',
                  'snapshot_value_guard_negative_target_blocked',
                  'snapshot_value_guard_prev_overshoot_norm',
                  'snapshot_value_guard_fleet_pressure_norm',
                  'snapshot_value_guard_primary_bias_s',
                  'hold_mean', 'hold_std', 'hold_max', 'hold_n',
                  'gap_dev', 'penalty', 'reward']

        with open(trip_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            if write_header:
                w.writeheader()
            for rec in self._ep_trip_records:
                rec['ep'] = ep
                w.writerow(rec)

        # Print per-period summary
        from collections import defaultdict
        by_period = defaultdict(list)
        for r in self._ep_trip_records:
            by_period[r['period']].append(r)

        print(f"  TRIPS    ep={ep}  n={len(self._ep_trip_records)}")
        for period in ['peak', 'off', 'trans']:
            recs = by_period.get(period, [])
            if not recs:
                continue
            deltas = [r['delta_t'] for r in recs]
            holds = [r['hold_mean'] for r in recs]
            pens = [r['penalty'] for r in recs]
            print(f"    {period:5s} n={len(recs):3d}  "
                  f"δ={np.mean(deltas):+5.1f}±{np.std(deltas):4.1f}  "
                  f"hold={np.mean(holds):+5.1f}±{np.std(holds):4.1f}  "
                  f"pen={np.mean(pens):.3f}")

            # Flag worst trips
            worst = sorted(recs, key=lambda r: -abs(r['hold_mean']))[:3]
            for w in worst:
                if abs(w['hold_mean']) > 20:
                    print(f"      ⚠ tid={w['tid']} h={w['hour']}:00 "
                          f"δ={w['delta_t']:+.0f}s hold_μ={w['hold_mean']:+.0f}s "
                          f"pen={w['penalty']:.2f}")

    # ────────────────── Train loop ──────────────────

    def maybe_resume(self):
        """Scan checkpoints/ for latest ep; if found, load networks and set resume_from_ep.
        Returns the ep to start from (0 if no resume, last_ep+1 otherwise).
        """
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            return 0
        if self.checkpoint_contract == 'exact_training_state_v4':
            training_path = os.path.join(ckpt_dir, 'training_latest.pt')
            if not os.path.exists(training_path):
                if any(name.startswith('lower_ep')
                       for name in os.listdir(ckpt_dir)):
                    raise RuntimeError(
                        'v4 checkpoint directory contains deployment weights '
                        'but no exact training state')
                return 0
            last_ep = self._load_training_state(training_path)
            self.resume_from_ep = last_ep + 1
            print(
                f"  [Exact resume] Loaded ep{last_ep}. "
                f"Resuming from ep{self.resume_from_ep}.")
            return self.resume_from_ep
        import re
        eps = []
        for fn in os.listdir(ckpt_dir):
            m = re.match(r'lower_ep(\d+)\.pt$', fn)
            if m and os.path.exists(os.path.join(ckpt_dir, f'upper_ep{m.group(1)}.pt')):
                eps.append(int(m.group(1)))
        if not eps:
            return 0
        last_ep = max(eps)
        try:
            self.load_checkpoint(ckpt_dir, ep=last_ep)
        except Exception as e:
            print(f"  [Resume] Failed to load ep{last_ep} checkpoint: {e}. Starting fresh.")
            return 0
        self.resume_from_ep = last_ep + 1
        print(f"  [Resume] Loaded checkpoint ep{last_ep}. Resuming from ep{self.resume_from_ep}.")
        return self.resume_from_ep

    def _deployment_state_dict(self, ep):
        adaptive_names = [
            'upper_residual_selector_A',
            'upper_residual_selector_b',
            'upper_residual_selector_updates',
            'timetable_terminal_value_selector_A',
            'timetable_terminal_value_selector_b',
            'timetable_terminal_value_selector_updates',
            'timetable_headway_value_planner_A',
            'timetable_headway_value_planner_b',
            'timetable_headway_value_planner_updates',
        ]
        adaptive = {
            name: copy.deepcopy(getattr(self, name))
            for name in adaptive_names
            if hasattr(self, name)
        }
        return {
            'protocol_version': self.protocol_version,
            'config_fingerprint_sha256': self.config_fingerprint_sha256,
            'randomness': copy.deepcopy(self.randomness_manifest),
            'episode': int(ep),
            'measurement_theta': self.measurement_proj.theta.copy(),
            'measurement_iter': int(self.measurement_proj._iter),
            'belief': self.belief_tracker.belief.copy(),
            'surprise': {
                'ema_surprise': float(self.surprise_computer.ema_surprise),
                'reward_history': list(self.surprise_computer.reward_history),
                'reward_ema': float(self.surprise_computer.reward_ema),
                'reward_var_ema': float(self.surprise_computer.reward_var_ema),
                'prev_q_std': self.surprise_computer.prev_q_std,
                'prev_delta_mean': float(
                    self.surprise_computer.prev_delta_mean),
            },
            'fixed_selector_cost_ema': copy.deepcopy(
                self.fixed_selector_cost_ema),
            'fixed_selector_counts': copy.deepcopy(
                self.fixed_selector_counts),
            'fixed_selector_context_A': copy.deepcopy(
                self.fixed_selector_context_A),
            'fixed_selector_context_b': copy.deepcopy(
                self.fixed_selector_context_b),
            'fixed_selector_prev_diag': copy.deepcopy(
                self._fixed_selector_prev_diag),
            'lower_context_gate': {
                'history_count': int(
                    self.env.lower_context_gate_history_count),
                'history_last_episode': (
                    self.env.lower_context_gate_history_last_episode),
                'history_summary': copy.deepcopy(
                    self.env.lower_context_gate_history_summary),
            },
            'adaptive_selectors': adaptive,
        }

    def _restore_deployment_state(self, state):
        saved_randomness = state.get('randomness')
        if self.randomness.isolated:
            if saved_randomness is None:
                raise ValueError(
                    'isolated_streams_v4 requires randomness checkpoint metadata')
            if (saved_randomness.get('fingerprint_sha256')
                    != self.randomness_manifest['fingerprint_sha256']):
                raise ValueError('checkpoint randomness contract mismatch')
        self.measurement_proj.theta = np.asarray(
            state.get('measurement_theta', self.measurement_proj.theta),
            dtype=np.float64)
        self.measurement_proj._iter = int(
            state.get('measurement_iter', self.measurement_proj._iter))
        belief = state.get('belief')
        if belief is not None:
            self.belief_tracker.belief = np.asarray(
                belief, dtype=np.float64)
        surprise = state.get('surprise', {}) or {}
        for name in [
                'ema_surprise', 'reward_ema', 'reward_var_ema',
                'prev_q_std', 'prev_delta_mean']:
            if name in surprise:
                setattr(self.surprise_computer, name, surprise[name])
        if 'reward_history' in surprise:
            self.surprise_computer.reward_history.clear()
            self.surprise_computer.reward_history.extend(
                surprise['reward_history'])
        for name in [
                'fixed_selector_cost_ema', 'fixed_selector_counts',
                'fixed_selector_context_A', 'fixed_selector_context_b']:
            if name in state:
                setattr(self, name, copy.deepcopy(state[name]))
        self._fixed_selector_prev_diag = copy.deepcopy(
            state.get('fixed_selector_prev_diag'))
        gate = state.get('lower_context_gate', {}) or {}
        self.env.lower_context_gate_history_count = int(
            gate.get('history_count', 0))
        self.env.lower_context_gate_history_last_episode = gate.get(
            'history_last_episode')
        self.env.lower_context_gate_history_summary = copy.deepcopy(
            gate.get('history_summary', {}))
        for name, value in (state.get('adaptive_selectors', {}) or {}).items():
            if hasattr(self, name):
                setattr(self, name, copy.deepcopy(value))
        self.loaded_checkpoint_ep = int(state.get('episode', -1))
        self.loaded_checkpoint_protocol = str(
            state.get('protocol_version', 'legacy'))
        self._deployment_state_loaded = True

    def _load_deployment_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        self._restore_deployment_state(state)

    def _runtime_numpy_streams(self):
        return {
            'fleet': self.fleet_rng,
            'upper_residual_selector': self._upper_residual_selector_rng,
            'headway_value_selector': self._headway_value_selector_rng,
            'terminal_value_selector': self._terminal_value_selector_rng,
            'fixed_expert_selector': self._fixed_expert_selector_rng,
            'tpc_mixture': self._tpc_rng,
            'reachability_replay': self._reachability_rng,
        }

    def _training_state_dict(self, ep):
        target_upper = None
        if self.target_upper_trainer is not None:
            target_upper = {
                'policy': copy.deepcopy(
                    self.target_upper_trainer.policy_net.state_dict()),
                'policy_sampling_state': (
                    self.target_upper_trainer.policy_net.sampling_state()),
            }
        reachability = None
        if self.reach_net is not None:
            reachability = {
                'network': self.reach_net.state_dict(),
                'optimizer': self.reach_optimizer.state_dict(),
                'buffer': copy.deepcopy(self._reach_buffer),
            }
        return {
            'format': 'freqduet-exact-training-state-v4',
            'protocol_version': self.protocol_version,
            'config_fingerprint_sha256': self.config_fingerprint_sha256,
            'episode': int(ep),
            'randomness': copy.deepcopy(self.randomness_manifest),
            'deployment': self._deployment_state_dict(ep),
            'lower_trainer': self.lower_trainer.training_state_dict(),
            'upper_trainer': self.upper_trainer.training_state_dict(),
            'lower_replay_buffer': self.replay_buffer.state_dict(),
            'numpy_stream_states': {
                name: rng.get_state()
                for name, rng in self._runtime_numpy_streams().items()
            },
            'global_random_state': random.getstate(),
            'global_numpy_state': np.random.get_state(),
            'global_torch_state': torch.random.get_rng_state(),
            'history': copy.deepcopy(dict(self.history)),
            'dispatch_meta': copy.deepcopy(self.dispatch_meta),
            'target_upper': target_upper,
            'reachability': reachability,
            'current_N_fleet': int(self._current_N_fleet),
        }

    def _load_training_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format') != 'freqduet-exact-training-state-v4':
            raise ValueError('not a FreqDuet exact v4 training checkpoint')
        if state.get('protocol_version') != self.protocol_version:
            raise ValueError('training checkpoint protocol mismatch')
        if (state.get('config_fingerprint_sha256')
                != self.config_fingerprint_sha256):
            raise ValueError('training checkpoint resolved-config mismatch')
        saved_randomness = state.get('randomness', {})
        if (saved_randomness.get('fingerprint_sha256')
                != self.randomness_manifest['fingerprint_sha256']):
            raise ValueError('training checkpoint randomness mismatch')

        self.lower_trainer.load_training_state_dict(state['lower_trainer'])
        self.upper_trainer.load_training_state_dict(state['upper_trainer'])
        self.replay_buffer.load_state_dict(state['lower_replay_buffer'])
        self._restore_deployment_state(state['deployment'])
        for name, rng in self._runtime_numpy_streams().items():
            rng.set_state(state['numpy_stream_states'][name])
        random.setstate(state['global_random_state'])
        np.random.set_state(state['global_numpy_state'])
        torch.random.set_rng_state(state['global_torch_state'])
        self.history = defaultdict(list, copy.deepcopy(state.get('history', {})))
        self.dispatch_meta = copy.deepcopy(state.get('dispatch_meta', {}))
        self._current_N_fleet = int(state.get(
            'current_N_fleet', self.N_fleet_default))

        target_upper = state.get('target_upper')
        if target_upper is None:
            self.target_upper_trainer = None
        else:
            self.target_upper_trainer = copy.deepcopy(self.upper_trainer)
            self.target_upper_trainer.replay_buffer.buffer.clear()
            self.target_upper_trainer.policy_net.load_state_dict(
                target_upper['policy'])
            self.target_upper_trainer.policy_net.set_sampling_state(
                target_upper.get('policy_sampling_state'))

        reachability = state.get('reachability')
        if reachability is None:
            self.reach_net = None
            self.reach_optimizer = None
            self._reach_buffer = []
        else:
            with self.randomness.torch_initialization('reachability_init'):
                self.reach_net = ReachabilityMLP(
                    self.upper_state_dim).to(self.device)
            self.reach_optimizer = torch.optim.Adam(
                self.reach_net.parameters(), lr=self.haar_reach_lr)
            self.reach_net.load_state_dict(reachability['network'])
            self.reach_optimizer.load_state_dict(reachability['optimizer'])
            self._reach_buffer = copy.deepcopy(reachability['buffer'])
        self.loaded_checkpoint_ep = int(state['episode'])
        return self.loaded_checkpoint_ep

    def load_checkpoint(self, checkpoint_dir=None, ep=None,
                        require_deployment_state=False):
        ckpt_dir = Path(checkpoint_dir or (Path(self.log_dir) / 'checkpoints'))
        if ckpt_dir.name != 'checkpoints' and (ckpt_dir / 'checkpoints').is_dir():
            ckpt_dir = ckpt_dir / 'checkpoints'
        if ep is None:
            eps = []
            for path in ckpt_dir.glob('lower_ep*.pt'):
                try:
                    candidate = int(path.stem.replace('lower_ep', ''))
                except ValueError:
                    continue
                if (ckpt_dir / f'upper_ep{candidate}.pt').exists():
                    eps.append(candidate)
            if not eps:
                raise FileNotFoundError(f'no paired checkpoints in {ckpt_dir}')
            ep = max(eps)
        self.lower_trainer.load(str(ckpt_dir / f'lower_ep{int(ep)}.pt'))
        self.upper_trainer.load(str(ckpt_dir / f'upper_ep{int(ep)}.pt'))
        state_path = ckpt_dir / f'runner_ep{int(ep)}.pt'
        if state_path.exists():
            self._load_deployment_state(state_path)
        else:
            if require_deployment_state:
                raise FileNotFoundError(
                    f'missing deployment checkpoint {state_path}')
            self.loaded_checkpoint_ep = int(ep)
            self.loaded_checkpoint_protocol = 'legacy'
            self._deployment_state_loaded = False
        if (require_deployment_state
                and self.loaded_checkpoint_protocol != self.protocol_version):
            raise ValueError(
                'checkpoint deployment protocol mismatch: '
                f'{self.loaded_checkpoint_protocol} != {self.protocol_version}')
        return int(ep)

    def _policy_digest(self):
        digest = hashlib.sha256()
        modules = [
            self.lower_trainer.policy_net,
            self.lower_trainer.q_net,
            self.lower_trainer.target_q_net,
            self.lower_trainer.cost_q_net,
            self.lower_trainer.target_cost_q_net,
            self.upper_trainer.policy_net,
            self.upper_trainer.q_net,
            self.upper_trainer.target_q_net,
        ]
        for module in modules:
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(name.encode('utf-8'))
                digest.update(
                    tensor.detach().cpu().contiguous().numpy().tobytes())
        deployment_state = self._deployment_state_dict(
            int(getattr(self, 'loaded_checkpoint_ep', -1)))
        deployment_state.pop('episode', None)
        # Hash fields independently so semantically irrelevant aliasing between
        # objects before and after torch deserialisation cannot change the
        # deployment digest.
        for name, value in sorted(deployment_state.items()):
            digest.update(name.encode('utf-8'))
            digest.update(pickle.dumps(value, protocol=5))
        return digest.hexdigest()

    def evaluate(self, scenario_seeds, output_dir=None, policy_ep=None):
        seeds = [int(seed) for seed in scenario_seeds]
        if not seeds:
            raise ValueError('evaluate requires at least one scenario seed')
        if policy_ep is None:
            policy_ep = max(
                int(getattr(self, 'loaded_checkpoint_ep', self._current_ep)),
                int(self.upper_warmup),
            )
        before = self._policy_digest()
        rows = []
        for seed in seeds:
            row = self.run_episode(
                ep=int(policy_ep),
                training=False,
                scenario_seed=seed,
                record_diagnostics=False,
            )
            row = dict(row)
            row['eval_seed'] = seed
            row['checkpoint_ep'] = int(
                getattr(self, 'loaded_checkpoint_ep', policy_ep))
            row['policy_digest'] = before
            rows.append(row)
            after_seed = self._policy_digest()
            if after_seed != before:
                raise RuntimeError(
                    'frozen evaluation mutated deployment policy state '
                    f'on scenario seed {seed}')

        destination = Path(
            output_dir
            or (Path(self.log_dir) / 'frozen_evaluation'
                / f'checkpoint_ep{int(getattr(self, "loaded_checkpoint_ep", policy_ep))}'))
        destination.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        evaluation_path = destination / 'evaluation.csv'
        with evaluation_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        evaluation_sha256 = hashlib.sha256(
            evaluation_path.read_bytes()).hexdigest()
        primary_key_records = [
            {'eval_seed': int(row['eval_seed'])}
            for row in sorted(rows, key=lambda value: int(value['eval_seed']))
        ]
        primary_key_sha256 = hashlib.sha256(json.dumps(
            primary_key_records,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')).hexdigest()
        with (destination / 'evaluation_manifest.json').open('w') as f:
            json.dump({
                'manifest_version': 'freqduet-evaluation-manifest-v2',
                'protocol_version': self.protocol_version,
                'config_fingerprint_sha256': self.config_fingerprint_sha256,
                'config_name': self.exp_name,
                'training_seed': self.base_seed,
                'checkpoint_ep': int(
                    getattr(self, 'loaded_checkpoint_ep', policy_ep)),
                'policy_episode': int(policy_ep),
                'scenario_seeds': seeds,
                'policy_digest': before,
                'n_episodes': len(rows),
                'randomness': self.randomness_manifest,
                'artifacts': {
                    'evaluation_csv': {
                        'sha256': evaluation_sha256,
                        'size_bytes': int(evaluation_path.stat().st_size),
                        'n_rows': len(rows),
                        'columns': fieldnames,
                        'primary_key': ['eval_seed'],
                        'primary_key_sha256': primary_key_sha256,
                    },
                },
            }, f, indent=2)
        return rows, destination

    def train(self, total_episodes=300):
        training_cfg = self.cfg.get('training', {})
        diag_freq = training_cfg.get('diag_freq', 10)
        trip_dump_freq = int(training_cfg.get('trip_dump_freq', 25))
        save_checkpoints = bool(training_cfg.get('save_checkpoints', True))
        checkpoint_freq = int(training_cfg.get('checkpoint_freq', 50))
        suppress_heavy_artifacts = str(
            os.environ.get('FREQDUET_SUPPRESS_HEAVY_ARTIFACTS', '')
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        if suppress_heavy_artifacts:
            trip_dump_freq = 0
            checkpoint_freq = 0
            save_checkpoints = True
        # Init diag now (after resume decision so CSV header handled correctly)
        if self.diag is None:
            self.diag = DiagnosticLog(self.log_dir, resume=(self.resume_from_ep > 0))

        print(f"TransitDuet v3 [{self.coupling_mode}] | eps={total_episodes} | "
              f"warmup={self.upper_warmup} | "
              f"δ∈[{self.upper_action_low.min():+.0f},{self.upper_action_high.max():+.0f}] "
              f"| α_hold={self.alpha_holding} | "
              f"dev={self.device}")
        bins_note = (
            f"  bins={self.lower_action_bins.tolist()}"
            if self.lower_action_bins is not None else "")
        if self.lower_action_bins_gate_enabled:
            bins_note += (
                f"  bins_gate={self.lower_action_bins_gate_source}"
                f">={self.lower_action_bins_gate_threshold:g}")
        last_note = "  +last_action" if self.lower_use_last_action_feature else ""
        print(f"  Lower: state={self.lower_state_dim}  K={self.lower_trainer.ensemble_size}  "
              f"batch={self.batch_size}  updates/ep={self.updates_per_episode}"
              f"{bins_note}{last_note}")
        if self.lower_state_encoder is not None:
            print("    lower_state_encoder=physical_dimensionless_v1")
        print(f"  Upper: state={self.upper_state_dim}  K={self.upper_trainer.ensemble_size}  "
              f"batch={self.upper_batch_size}  updates/ep={self.upper_updates}")
        if self.upper_action_bins is not None:
            print(f"    upper_bins={self.upper_action_bins.tolist()}")
        if self.upper_action_candidates is not None:
            print(
                f"    upper_action_library={len(self.upper_action_candidates)} "
                f"curves  critic={self.upper_trainer.discrete_critic}")
        print(
            "    upper_credit="
            f"{self.upper_credit_assignment.system_reward_mode}/"
            f"{self.upper_credit_assignment.gap_credit_mode}  "
            f"stream={self.upper_transition_stream_mode}  "
            f"interval={self.upper_interval_credit.assignment_mode if self.upper_interval_credit.enabled else 'off'}")
        freeze_notes = []
        if self.freeze_lower_policy_after_ep is not None:
            freeze_notes.append(
                f"lower_policy@{self.freeze_lower_policy_after_ep}")
        if self.freeze_lower_critic_after_ep is not None:
            freeze_notes.append(
                f"lower_critic@{self.freeze_lower_critic_after_ep}")
        if self.freeze_upper_after_ep is not None:
            freeze_notes.append(f"upper@{self.freeze_upper_after_ep}")
        if freeze_notes:
            print(f"  Longtrain stability freeze: {', '.join(freeze_notes)}")
        if suppress_heavy_artifacts:
            print("  Heavy artifacts suppressed: no trip details; final deployment checkpoint retained")
        print(f"  Diag CSV: {self.diag.csv_path}")
        print("=" * 90)
        if self.fixed_selector_reset_env_rng:
            random.seed(self.base_seed)
            np.random.seed(self.base_seed)

        for ep in range(self.resume_from_ep, total_episodes):
            row = self.run_episode(ep, training=True)

            # ── Compact per-episode line ──
            if ep % 5 == 0 or ep < 5:
                line = (f"[{ep:3d}] {row['stage']:7s} N={row.get('N_fleet',12):2d} | "
                        f"w={row['avg_wait_min']:4.1f} f={row['peak_fleet']:2d} "
                        f"cv={row['headway_cv']:.2f} | "
                        f"Lπ a={row['lower_action_mean']:+5.1f}±{row['lower_action_std']:4.1f} "
                        f"Q={row['lower_q_mean']:+6.1f} λ={row['lower_lambda']:.2f} | ")
                if row['stage'] == 'BiLevel':
                    line += (f"Uδ={row['upper_delta_mean']:+5.1f}±{row['upper_delta_std']:4.1f} "
                             f"Q={row['upper_q_mean']:+6.3f} | "
                             f"h_μ={row['hold_fb_mean']:+5.1f} "
                             f"pen={row['hold_penalty_mean']:.2f} | ")
                line += f"{row['wall_env_s']:.0f}+{row['wall_train_s']:.0f}s"
                print(line)

            # ── Detailed diagnostic block ──
            if ep % diag_freq == 0 or ep == total_episodes - 1:
                self._print_diagnostic_block(row)

            # ── Per-trip breakdown ──
            if trip_dump_freq > 0 and ep % trip_dump_freq == 0 and row['stage'] == 'BiLevel':
                self._dump_trip_breakdown(ep)

            # ── Checkpoint ──
            if save_checkpoints and checkpoint_freq > 0 and (ep + 1) % checkpoint_freq == 0:
                self._save_checkpoint(ep)

        if save_checkpoints:
            self._save_checkpoint(total_episodes - 1)
        self.diag.save_json()
        self._save_history()
        print(f"\nDone. Results in {self.log_dir}/")

    def _save_checkpoint(self, ep):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.lower_trainer.save(os.path.join(ckpt_dir, f'lower_ep{ep}.pt'))
        self.upper_trainer.save(os.path.join(ckpt_dir, f'upper_ep{ep}.pt'))
        torch.save(
            self._deployment_state_dict(ep),
            os.path.join(ckpt_dir, f'runner_ep{ep}.pt'),
        )
        if self.checkpoint_contract == 'exact_training_state_v4':
            training_path = os.path.join(ckpt_dir, 'training_latest.pt')
            temporary_path = training_path + '.tmp'
            torch.save(self._training_state_dict(ep), temporary_path)
            os.replace(temporary_path, training_path)
        self.loaded_checkpoint_ep = int(ep)
        with open(os.path.join(ckpt_dir, 'checkpoint_meta.json'), 'w') as f:
            json.dump({
                'protocol_version': self.protocol_version,
                'config_fingerprint_sha256': self.config_fingerprint_sha256,
                'latest_episode': int(ep),
                'config_name': self.exp_name,
                'seed': self.base_seed,
                'randomness': self.randomness_manifest,
                'checkpoint_contract': self.checkpoint_contract,
            }, f, indent=2)
        print(f"  [Checkpoint ep {ep}]")

    def _save_history(self):
        results = {}
        for key, values in self.history.items():
            try:
                results[key] = [float(x) if not isinstance(x, list) else x
                                for x in values]
            except (TypeError, ValueError):
                results[key] = [str(x) for x in values]
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            json.dump(results, f)


def eval_pareto_frontier(runner, n_eval=10, fleet_values=None,
                         scenario_seeds=None):
    """v2k: Sweep N_fleet values and record (fleet, wait, cv) Pareto points."""
    if fleet_values is None:
        fleet_values = list(range(8, 17))
    if scenario_seeds is None:
        start = 20000000 + int(runner.base_seed) * 1000
        scenario_seeds = list(range(start, start + int(n_eval)))
    else:
        scenario_seeds = [int(seed) for seed in scenario_seeds]
    policy_ep = max(
        int(getattr(runner, 'loaded_checkpoint_ep', runner._current_ep)),
        int(runner.upper_warmup),
    )
    policy_digest = runner._policy_digest()
    results = []
    for N in fleet_values:
        waits, restricted_waits, cvs, overshoots = [], [], [], []
        for seed in scenario_seeds:
            row = runner.run_episode(
                ep=policy_ep,
                training=False,
                N_fleet_override=N,
                scenario_seed=seed,
                record_diagnostics=False,
            )
            waits.append(row['avg_wait_observed_min'])
            restricted_waits.append(row['restricted_wait_horizon_min'])
            cvs.append(row['headway_cv'])
            overshoots.append(row.get('fleet_overshoot', 0))
        results.append({
            'N_fleet': N,
            'wait_mean': float(np.mean(waits)),
            'wait_std': float(np.std(waits)),
            'restricted_wait_mean': float(np.mean(restricted_waits)),
            'restricted_wait_std': float(np.std(restricted_waits)),
            'cv_mean': float(np.mean(cvs)),
            'cv_std': float(np.std(cvs)),
            'overshoot_mean': float(np.mean(overshoots)),
        })
        print(f"  N_fleet={N:2d}: wait={np.mean(waits):4.1f}±{np.std(waits):.1f}m  "
              f"cv={np.mean(cvs):.2f}  overshoot={np.mean(overshoots):.1f}")
    if runner._policy_digest() != policy_digest:
        raise RuntimeError('Pareto evaluation mutated deployment policy state')
    return results


def main():
    parser = argparse.ArgumentParser(
        description='TransitDuet v3 (HIRO/HAAR/channels coupling-mode runner)')
    parser.add_argument('--config', type=str, default='config_v2.yaml')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--eval_pareto', action='store_true',
                        help='After training, evaluate Pareto frontier over N_fleet ∈ [8,16]')
    parser.add_argument('--n_eval', type=int, default=5, help='eps per N_fleet for eval')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load a checkpoint and run frozen deterministic evaluation')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Run directory or checkpoints directory for --eval-only')
    parser.add_argument('--checkpoint-ep', type=int, default=None,
                        help='Checkpoint episode; defaults to latest paired checkpoint')
    parser.add_argument('--eval-seeds', type=str, default=None,
                        help='Comma-separated independent scenario seeds')
    parser.add_argument('--eval-output-dir', type=str, default=None,
                        help='Output directory for frozen evaluation')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                        help='Resume from latest checkpoint if found (default: on)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start from scratch even if checkpoints exist')
    parser.add_argument('--upper-warmup-eps', type=int, default=None,
                        help='Override coupling.upper_warmup_eps for short experiments')
    parser.add_argument('--logs-dir', type=str, default=None,
                        help='Override base directory for run logs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config_path = os.path.join(str(SCRIPT_DIR), args.config)
    config = load_config(config_path)
    config['seed'] = args.seed
    if args.upper_warmup_eps is not None:
        config.setdefault('coupling', {})['upper_warmup_eps'] = int(args.upper_warmup_eps)
    if args.logs_dir is not None:
        config.setdefault('logging', {})['logs_dir'] = args.logs_dir

    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'

    runner = TransitDuetV2Runner(config, device=device)
    if args.eval_only:
        checkpoint_ep = runner.load_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            ep=args.checkpoint_ep,
            require_deployment_state=True,
        )
        if args.eval_seeds:
            eval_seeds = [
                int(value.strip())
                for value in args.eval_seeds.split(',')
                if value.strip()
            ]
        else:
            start = 10000000 + int(args.seed) * 1000
            eval_seeds = list(range(start, start + int(args.n_eval)))
        rows, destination = runner.evaluate(
            eval_seeds,
            output_dir=args.eval_output_dir,
            policy_ep=max(checkpoint_ep, runner.upper_warmup),
        )
        print(
            f"Frozen evaluation complete: {len(rows)} episodes -> "
            f"{destination}")
        return
    if args.resume:
        runner.maybe_resume()
    runner.train(total_episodes=args.episodes)

    if args.eval_pareto:
        print("\n" + "="*80)
        print("  PARETO FRONTIER EVALUATION")
        print("="*80)
        results = eval_pareto_frontier(runner, n_eval=args.n_eval)
        with open(os.path.join(runner.log_dir, 'pareto_frontier.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {runner.log_dir}/pareto_frontier.json")


if __name__ == '__main__':
    main()
