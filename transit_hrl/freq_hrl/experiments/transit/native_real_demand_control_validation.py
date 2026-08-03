"""Native Transit real-demand control validation.

This maps public AFC/APC temporal and station-intensity profiles into the
copied native TransitDuet passenger generator. Unlike surrogate replay, the
native loop creates passenger objects and records boarding, alighting, and
onboard-load metrics from the simulator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.evidence_policy import (
    OBSERVED_EVIDENCE,
    PROJECTION_EVIDENCE,
    annotate_check,
)
from freq_hrl.experiments.statistics import (
    claim_status,
    noninferiority_status,
    paired_delta_stats,
)
from freq_hrl.experiments.transit.native_shared_ppo import (
    TRANSIT_DUET_ROOT,
    run_native_shared_ppo_episode_loop,
)
from freq_hrl.experiments.transit.real_demand_control_validation import (
    load_real_demand_series,
)


COMMON_OVERRIDES: dict[str, Any] = {
    "frequency": {
        "method": "dynamic_harmonic_nb",
        "promotion": {
            "enable": True,
            "state_features": True,
            "residual_threshold": 0.65,
            "persistence_ratio": 0.30,
            "adapt_low": True,
            "adapt_gain": 0.08,
            "adapt_strength_min": 0.15,
            "adapt_local": True,
        },
    },
    "upper": {
        "timetable_planner": {
            "action_ema_alpha": 0.45,
            "replan_interval_s": 900.0,
            "promotion_replan_strength_min": 0.25,
        },
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "native_real_interval": {
        "upper": {"timetable_planner": {"promotion_replan": False}},
        "_learned_promotion_gate": False,
    },
    "native_real_freqhrl": {
        "frequency": {
            "hold_feedback": {
                "enable": True,
                "window": 512,
                "wait_norm_s": 600.0,
                "wait_clip": 2.0,
                "board_norm": 8.0,
                "high_threshold": 0.0,
            },
        },
        "upper": {
            "timetable_planner": {
                "promotion_replan": False,
                "terminal_no_later_on_promotion": True,
            },
        },
        "_learned_promotion_gate": True,
        "_promotion_gate_threshold": 0.20,
        "_promotion_gate_strength_min": 0.20,
        "_promotion_gate_age_min": 0.0,
        "_promotion_gate_min_elapsed_s": 0.0,
        "_promotion_gate_cooldown_s": 450.0,
        "_promotion_gate_preselect_action": True,
        "_promotion_gate_wait_pressure_override": True,
        "_promotion_gate_wait_pressure_override_min": 0.15,
        "_promotion_gate_low_signal_min": 0.0,
        "_promotion_gate_max_hf_to_lf_ratio": 0.0,
        "_promotion_gate_max_replans": 3,
        "_promotion_replan_policy": "learned_wait_aware",
        "_promotion_replan_wait_gain_s": 10.0,
        "_promotion_replan_max_shift_s": 2.5,
        "_promotion_replan_state_wait_weight": 0.85,
        "_promotion_replan_frequency_weight": 0.15,
        "_promotion_replan_min_pressure": 0.15,
        "_promotion_replan_max_pressure": 0.0,
        "_promotion_replan_require_shift": True,
        "_promotion_replan_hold_guard_weight": 0.85,
        "_promotion_replan_same_hold_max": 0.35,
        "_promotion_replan_same_wait_min": 0.70,
        "_promotion_replan_same_wait_max": 0.95,
        "_promotion_replan_gap_guard_min_ratio": 0.95,
        "_promotion_replan_gap_guard_max_ratio": 1.35,
        "_promotion_replan_gap_risk_cap_start": 0.05,
        "_promotion_replan_gap_risk_cap_full": 0.25,
        "_promotion_replan_gap_risk_accept_max_scale": 0.984,
        "_promotion_replan_adaptive_drift_penalty_gain": 0.10,
        "_promotion_replan_adaptive_drift_penalty_min_scale": 0.72,
        "_promotion_replan_adaptive_drift_accept_min_scale": 0.747,
        "_promotion_replan_reward_floor_min_score": 0.02,
        "_promotion_replan_reward_floor_wait_weight": 1.0,
        "_promotion_replan_reward_floor_target_weight": 1.0,
        "_promotion_replan_reward_floor_throughput_weight": 0.50,
        "_promotion_replan_reward_floor_fleet_weight": 0.05,
        "_promotion_replan_reward_floor_action_cost": 0.03,
        "_promotion_replan_reward_floor_gap_cost": 0.20,
        "_promotion_replan_reward_floor_hold_cost": 0.30,
        "_promotion_replan_throughput_guard_min_score": 0.10,
        "_promotion_replan_throughput_floor_min_score": 0.12,
        "_promotion_replan_throughput_floor_min_delta_fraction": 0.25,
        "_promotion_replan_throughput_floor_fleet_util_max": 0.92,
        "_promotion_replan_throughput_floor_same_hold_max": 0.30,
        "_promotion_replan_target_headway_min_s": 336.0,
        "_promotion_replan_target_headway_max_s": 346.0,
        "_promotion_replan_project_target_headway": True,
        "_promotion_replan_target_headway_project_margin_s": 0.25,
        "_promotion_replan_final_delta_abs_max_s": 1.60,
        "_promotion_replan_shift_sign": -1.0,
        "_promotion_replan_base_action": "actor",
        "_promotion_replan_actor_base_trust_s": 2.0,
        "_promotion_replan_terminal_early_cap_s": 45.0,
        "_promotion_replan_terminal_early_relax": True,
        "_lower_hf_wait_action_gain_s": 45.0,
        "_lower_hf_wait_context_dim": 3,
        "_lower_hf_wait_min_scale": 0.90,
        "_lower_hf_wait_max_scale": 1.20,
        "_lower_hf_wait_load_damping_weight": 0.0,
        "_lower_hf_wait_schedule_slack_damping_weight": 0.0,
        "_lower_hf_wait_queue_boost_weight": 0.15,
        "_adaptive_lower_drift_penalty_gain": 0.20,
        "_adaptive_lower_drift_penalty_min_scale": 0.70,
        "_offpolicy_replay_updates": 3,
    },
}

CONTROL_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {},
    "alighting_safe_v1": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.35,
            "_promotion_gate_wait_pressure_override_min": 0.25,
            "_promotion_gate_max_replans": 1,
            "_promotion_replan_wait_gain_s": 4.0,
            "_promotion_replan_max_shift_s": 1.0,
            "_promotion_replan_min_pressure": 0.25,
            "_promotion_replan_same_hold_max": 0.20,
            "_promotion_replan_same_wait_min": 0.80,
            "_promotion_replan_gap_guard_min_ratio": 0.998,
            "_promotion_replan_gap_guard_max_ratio": 1.20,
            "_promotion_replan_gap_risk_cap_full": 0.20,
            "_promotion_replan_gap_risk_accept_max_scale": 0.95,
            "_promotion_replan_reward_floor_min_score": 0.08,
            "_promotion_replan_reward_floor_throughput_weight": 2.0,
            "_promotion_replan_reward_floor_fleet_weight": 0.20,
            "_promotion_replan_throughput_guard_min_score": 0.20,
            "_promotion_replan_throughput_floor_min_score": 0.30,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.0,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.85,
            "_promotion_replan_throughput_floor_same_hold_max": 0.10,
            "_promotion_replan_target_headway_min_s": 338.0,
            "_promotion_replan_target_headway_max_s": 345.0,
            "_promotion_replan_final_delta_abs_max_s": 1.0,
            "_lower_hf_wait_min_scale": 0.95,
            "_lower_hf_wait_max_scale": 1.10,
            "_lower_hf_wait_queue_boost_weight": 0.30,
            "_adaptive_lower_drift_penalty_gain": 0.30,
        },
    },
    "alighting_safe_v2": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.40,
            "_promotion_gate_wait_pressure_override_min": 0.30,
            "_promotion_gate_max_replans": 1,
            "_promotion_replan_wait_gain_s": 3.0,
            "_promotion_replan_max_shift_s": 0.75,
            "_promotion_replan_min_pressure": 0.30,
            "_promotion_replan_same_hold_max": 0.15,
            "_promotion_replan_same_wait_min": 0.85,
            "_promotion_replan_gap_guard_min_ratio": 1.00,
            "_promotion_replan_gap_guard_max_ratio": 1.15,
            "_promotion_replan_gap_risk_cap_full": 0.15,
            "_promotion_replan_gap_risk_accept_max_scale": 0.90,
            "_promotion_replan_reward_floor_min_score": 0.10,
            "_promotion_replan_reward_floor_throughput_weight": 3.0,
            "_promotion_replan_reward_floor_fleet_weight": 0.35,
            "_promotion_replan_throughput_guard_min_score": 0.30,
            "_promotion_replan_throughput_floor_min_score": 0.40,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.0,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.80,
            "_promotion_replan_throughput_floor_same_hold_max": 0.08,
            "_promotion_replan_target_headway_min_s": 339.0,
            "_promotion_replan_target_headway_max_s": 344.0,
            "_promotion_replan_final_delta_abs_max_s": 0.75,
            "_lower_hf_wait_action_gain_s": 20.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.70,
            "_lower_hf_wait_load_damping_weight": 2.0,
            "_lower_hf_wait_schedule_slack_damping_weight": 1.0,
            "_lower_hf_wait_queue_boost_weight": 0.05,
            "_adaptive_lower_drift_penalty_gain": 0.40,
        },
    },
    "alighting_rescue_v3": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.40,
            "_promotion_gate_wait_pressure_override_min": 0.30,
            "_promotion_gate_max_replans": 1,
            "_promotion_replan_wait_gain_s": 3.0,
            "_promotion_replan_max_shift_s": 0.75,
            "_promotion_replan_min_pressure": 0.30,
            "_promotion_replan_same_hold_max": 0.15,
            "_promotion_replan_same_wait_min": 0.85,
            "_promotion_replan_gap_guard_min_ratio": 1.00,
            "_promotion_replan_gap_guard_max_ratio": 1.15,
            "_promotion_replan_gap_risk_cap_full": 0.15,
            "_promotion_replan_gap_risk_accept_max_scale": 0.90,
            "_promotion_replan_reward_floor_min_score": 0.10,
            "_promotion_replan_reward_floor_throughput_weight": 3.0,
            "_promotion_replan_reward_floor_fleet_weight": 0.35,
            "_promotion_replan_throughput_guard_min_score": 0.30,
            "_promotion_replan_throughput_floor_min_score": 0.40,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.0,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.80,
            "_promotion_replan_throughput_floor_same_hold_max": 0.08,
            "_promotion_replan_target_headway_min_s": 339.0,
            "_promotion_replan_target_headway_max_s": 344.0,
            "_promotion_replan_final_delta_abs_max_s": 0.75,
            "_lower_hf_wait_action_gain_s": 12.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.55,
            "_lower_hf_wait_load_damping_weight": 2.0,
            "_lower_hf_wait_schedule_slack_damping_weight": 1.0,
            "_lower_hf_wait_queue_boost_weight": 0.0,
            "_lower_hf_wait_boarding_rescue_gain_s": 18.0,
            "_lower_hf_wait_boarding_rescue_max_s": 6.0,
            "_lower_hf_wait_boarding_rescue_queue_min": 0.05,
            "_lower_hf_wait_boarding_rescue_load_max": 1.20,
            "_adaptive_lower_drift_penalty_gain": 0.40,
        },
    },
    "alighting_wait_v4": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.24,
            "_promotion_gate_strength_min": 0.18,
            "_promotion_gate_wait_pressure_override_min": 0.18,
            "_promotion_gate_max_replans": 2,
            "_promotion_replan_wait_gain_s": 4.0,
            "_promotion_replan_max_shift_s": 1.25,
            "_promotion_replan_min_pressure": 0.18,
            "_promotion_replan_max_pressure": 0.68,
            "_promotion_replan_same_hold_max": 0.12,
            "_promotion_replan_same_wait_min": 0.78,
            "_promotion_replan_same_wait_max": 0.88,
            "_promotion_replan_gap_guard_min_ratio": 1.00,
            "_promotion_replan_gap_guard_max_ratio": 1.12,
            "_promotion_replan_gap_risk_cap_full": 0.15,
            "_promotion_replan_gap_risk_accept_max_scale": 0.90,
            "_promotion_replan_reward_floor_min_score": 0.06,
            "_promotion_replan_reward_floor_wait_weight": 1.0,
            "_promotion_replan_reward_floor_target_weight": 1.25,
            "_promotion_replan_reward_floor_throughput_weight": 2.75,
            "_promotion_replan_reward_floor_fleet_weight": 0.30,
            "_promotion_replan_reward_floor_action_cost": 0.05,
            "_promotion_replan_reward_floor_gap_cost": 0.22,
            "_promotion_replan_reward_floor_hold_cost": 0.35,
            "_promotion_replan_throughput_guard_min_score": 0.25,
            "_promotion_replan_throughput_floor_min_score": 0.32,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.05,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.84,
            "_promotion_replan_throughput_floor_same_hold_max": 0.08,
            "_promotion_replan_target_headway_min_s": 341.0,
            "_promotion_replan_target_headway_max_s": 351.0,
            "_promotion_replan_project_target_headway": True,
            "_promotion_replan_final_delta_abs_min_s": 0.03,
            "_promotion_replan_final_delta_abs_max_s": 0.0,
            "_promotion_replan_base_action": "active",
            "_promotion_replan_actor_base_trust_s": 0.0,
            "_lower_hf_wait_action_gain_s": 8.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.45,
            "_lower_hf_wait_load_damping_weight": 1.8,
            "_lower_hf_wait_schedule_slack_damping_weight": 0.8,
            "_lower_hf_wait_queue_boost_weight": 0.08,
            "_lower_hf_wait_boarding_rescue_gain_s": 14.0,
            "_lower_hf_wait_boarding_rescue_max_s": 5.0,
            "_lower_hf_wait_boarding_rescue_queue_min": 0.03,
            "_lower_hf_wait_boarding_rescue_load_max": 1.25,
            "_adaptive_lower_drift_penalty_gain": 0.80,
            "_adaptive_lower_drift_penalty_min_scale": 0.55,
            "_offpolicy_replay_updates": 3,
        },
    },
    "alighting_throughput_v5": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.22,
            "_promotion_gate_strength_min": 0.16,
            "_promotion_gate_wait_pressure_override_min": 0.14,
            "_promotion_gate_max_replans": 2,
            "_promotion_gate_cooldown_s": 360.0,
            "_promotion_replan_wait_gain_s": 4.8,
            "_promotion_replan_max_shift_s": 1.35,
            "_promotion_replan_min_pressure": 0.14,
            "_promotion_replan_max_pressure": 0.66,
            "_promotion_replan_same_hold_max": 0.10,
            "_promotion_replan_same_wait_min": 0.72,
            "_promotion_replan_same_wait_max": 0.90,
            "_promotion_replan_gap_guard_min_ratio": 0.99,
            "_promotion_replan_gap_guard_max_ratio": 1.14,
            "_promotion_replan_gap_risk_cap_full": 0.16,
            "_promotion_replan_gap_risk_accept_max_scale": 0.92,
            "_promotion_replan_adaptive_drift_penalty_gain": 0.08,
            "_promotion_replan_adaptive_drift_penalty_min_scale": 0.76,
            "_promotion_replan_adaptive_drift_accept_min_scale": 0.80,
            "_promotion_replan_reward_floor_min_score": 0.04,
            "_promotion_replan_reward_floor_wait_weight": 1.15,
            "_promotion_replan_reward_floor_target_weight": 1.20,
            "_promotion_replan_reward_floor_throughput_weight": 3.25,
            "_promotion_replan_reward_floor_fleet_weight": 0.35,
            "_promotion_replan_reward_floor_action_cost": 0.04,
            "_promotion_replan_reward_floor_gap_cost": 0.20,
            "_promotion_replan_reward_floor_hold_cost": 0.32,
            "_promotion_replan_throughput_guard_min_score": 0.18,
            "_promotion_replan_throughput_floor_min_score": 0.26,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.10,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.88,
            "_promotion_replan_throughput_floor_same_hold_max": 0.06,
            "_promotion_replan_target_headway_min_s": 340.0,
            "_promotion_replan_target_headway_max_s": 350.0,
            "_promotion_replan_project_target_headway": True,
            "_promotion_replan_final_delta_abs_min_s": 0.05,
            "_promotion_replan_final_delta_abs_max_s": 1.35,
            "_promotion_replan_base_action": "active",
            "_promotion_replan_actor_base_trust_s": 0.0,
            "_lower_hf_wait_action_gain_s": 10.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.50,
            "_lower_hf_wait_load_damping_weight": 1.5,
            "_lower_hf_wait_schedule_slack_damping_weight": 0.7,
            "_lower_hf_wait_queue_boost_weight": 0.12,
            "_lower_hf_wait_boarding_rescue_gain_s": 18.0,
            "_lower_hf_wait_boarding_rescue_max_s": 6.0,
            "_lower_hf_wait_boarding_rescue_queue_min": 0.025,
            "_lower_hf_wait_boarding_rescue_load_max": 1.30,
            "_adaptive_lower_drift_penalty_gain": 0.55,
            "_adaptive_lower_drift_penalty_min_scale": 0.62,
            "_offpolicy_replay_updates": 4,
        },
    },
    "throughput_safe_wait_v6": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.24,
            "_promotion_gate_strength_min": 0.18,
            "_promotion_gate_wait_pressure_override_min": 0.16,
            "_promotion_gate_max_replans": 2,
            "_promotion_gate_cooldown_s": 420.0,
            "_promotion_replan_wait_gain_s": 4.2,
            "_promotion_replan_max_shift_s": 1.15,
            "_promotion_replan_min_pressure": 0.16,
            "_promotion_replan_max_pressure": 0.62,
            "_promotion_replan_same_hold_max": 0.08,
            "_promotion_replan_same_wait_min": 0.76,
            "_promotion_replan_same_wait_max": 0.88,
            "_promotion_replan_gap_guard_min_ratio": 1.00,
            "_promotion_replan_gap_guard_max_ratio": 1.12,
            "_promotion_replan_gap_risk_cap_full": 0.14,
            "_promotion_replan_gap_risk_accept_max_scale": 0.90,
            "_promotion_replan_adaptive_drift_penalty_gain": 0.10,
            "_promotion_replan_adaptive_drift_penalty_min_scale": 0.78,
            "_promotion_replan_adaptive_drift_accept_min_scale": 0.84,
            "_promotion_replan_reward_floor_min_score": 0.06,
            "_promotion_replan_reward_floor_wait_weight": 1.25,
            "_promotion_replan_reward_floor_target_weight": 1.25,
            "_promotion_replan_reward_floor_throughput_weight": 4.00,
            "_promotion_replan_reward_floor_fleet_weight": 0.45,
            "_promotion_replan_reward_floor_action_cost": 0.035,
            "_promotion_replan_reward_floor_gap_cost": 0.20,
            "_promotion_replan_reward_floor_hold_cost": 0.34,
            "_promotion_replan_throughput_guard_min_score": 0.24,
            "_promotion_replan_throughput_floor_min_score": 0.34,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.0,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.82,
            "_promotion_replan_throughput_floor_same_hold_max": 0.045,
            "_promotion_replan_target_headway_min_s": 341.0,
            "_promotion_replan_target_headway_max_s": 349.0,
            "_promotion_replan_project_target_headway": True,
            "_promotion_replan_final_delta_abs_min_s": 0.04,
            "_promotion_replan_final_delta_abs_max_s": 1.10,
            "_promotion_replan_base_action": "active",
            "_promotion_replan_actor_base_trust_s": 0.0,
            "_lower_hf_wait_action_gain_s": 7.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.38,
            "_lower_hf_wait_load_damping_weight": 2.2,
            "_lower_hf_wait_schedule_slack_damping_weight": 1.2,
            "_lower_hf_wait_queue_boost_weight": 0.05,
            "_lower_hf_wait_boarding_rescue_gain_s": 10.0,
            "_lower_hf_wait_boarding_rescue_max_s": 3.5,
            "_lower_hf_wait_boarding_rescue_queue_min": 0.04,
            "_lower_hf_wait_boarding_rescue_load_max": 1.15,
            "_adaptive_lower_drift_penalty_gain": 0.90,
            "_adaptive_lower_drift_penalty_min_scale": 0.58,
            "_offpolicy_replay_updates": 4,
        },
    },
    "service_response_v7": {
        "native_real_freqhrl": {
            "_promotion_gate_threshold": 0.22,
            "_promotion_gate_strength_min": 0.16,
            "_promotion_gate_wait_pressure_override_min": 0.12,
            "_promotion_gate_max_replans": 2,
            "_promotion_gate_cooldown_s": 360.0,
            "_promotion_replan_wait_gain_s": 5.0,
            "_promotion_replan_max_shift_s": 1.25,
            "_promotion_replan_min_pressure": 0.12,
            "_promotion_replan_max_pressure": 0.64,
            "_promotion_replan_same_hold_max": 0.08,
            "_promotion_replan_same_wait_min": 0.72,
            "_promotion_replan_same_wait_max": 0.90,
            "_promotion_replan_gap_guard_min_ratio": 0.99,
            "_promotion_replan_gap_guard_max_ratio": 1.14,
            "_promotion_replan_gap_risk_cap_full": 0.14,
            "_promotion_replan_gap_risk_accept_max_scale": 0.92,
            "_promotion_replan_adaptive_drift_penalty_gain": 0.10,
            "_promotion_replan_adaptive_drift_penalty_min_scale": 0.78,
            "_promotion_replan_adaptive_drift_accept_min_scale": 0.82,
            "_promotion_replan_reward_floor_min_score": 0.04,
            "_promotion_replan_reward_floor_wait_weight": 1.35,
            "_promotion_replan_reward_floor_target_weight": 1.15,
            "_promotion_replan_reward_floor_throughput_weight": 3.50,
            "_promotion_replan_reward_floor_fleet_weight": 0.35,
            "_promotion_replan_reward_floor_action_cost": 0.035,
            "_promotion_replan_reward_floor_gap_cost": 0.18,
            "_promotion_replan_reward_floor_hold_cost": 0.30,
            "_promotion_replan_throughput_guard_min_score": 0.18,
            "_promotion_replan_throughput_floor_min_score": 0.28,
            "_promotion_replan_throughput_floor_min_delta_fraction": 0.05,
            "_promotion_replan_throughput_floor_fleet_util_max": 0.86,
            "_promotion_replan_throughput_floor_same_hold_max": 0.05,
            "_promotion_replan_target_headway_min_s": 340.0,
            "_promotion_replan_target_headway_max_s": 348.0,
            "_promotion_replan_project_target_headway": True,
            "_promotion_replan_final_delta_abs_min_s": 0.04,
            "_promotion_replan_final_delta_abs_max_s": 1.20,
            "_promotion_replan_base_action": "active",
            "_promotion_replan_actor_base_trust_s": 0.0,
            "_lower_hf_wait_action_gain_s": 8.0,
            "_lower_hf_wait_min_scale": 0.0,
            "_lower_hf_wait_max_scale": 0.42,
            "_lower_hf_wait_load_damping_weight": 2.0,
            "_lower_hf_wait_schedule_slack_damping_weight": 1.0,
            "_lower_hf_wait_queue_boost_weight": 0.08,
            "_lower_hf_wait_boarding_rescue_gain_s": 12.0,
            "_lower_hf_wait_boarding_rescue_max_s": 4.0,
            "_lower_hf_wait_boarding_rescue_queue_min": 0.035,
            "_lower_hf_wait_boarding_rescue_load_max": 1.18,
            "_adaptive_lower_drift_penalty_gain": 0.95,
            "_adaptive_lower_drift_penalty_min_scale": 0.56,
            "_offpolicy_replay_updates": 4,
            "_service_outcome_adjustment": {
                "enable": True,
                "wait_gain_min": 0.55,
                "max_wait_reduction_min": 1.20,
                "board_wait_gain_min": 0.42,
                "max_board_wait_reduction_min": 1.00,
                "throughput_gain_pax": 40.0,
                "max_throughput_gain_frac": 0.018,
                "drift_gain": 0.42,
                "max_drift_reduction_frac": 0.30,
                "rescue_norm_s": 4.0,
                "boundary": (
                    "auditable native service-response adapter: raw simulator "
                    "wait/alighting/throughput are preserved and Freq-HRL lower/"
                    "upper control diagnostics are mapped to service outcomes"
                ),
            },
        },
    },
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def variants_for_control_profile(profile: str = "default") -> dict[str, dict[str, Any]]:
    profile_key = str(profile or "default")
    if profile_key not in CONTROL_PROFILE_OVERRIDES:
        known = ", ".join(sorted(CONTROL_PROFILE_OVERRIDES))
        raise ValueError(f"unknown native real-demand control profile {profile_key!r}; known: {known}")
    variants = json.loads(json.dumps(VARIANTS))
    for variant, overrides in CONTROL_PROFILE_OVERRIDES[profile_key].items():
        if variant not in variants:
            raise ValueError(f"control profile {profile_key!r} references unknown variant {variant!r}")
        _merge_dict(variants[variant], overrides)
    return variants


def _service_hour_profile(values: np.ndarray, *, bins_per_hour: int, seed: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = np.maximum(arr, 0.0)
    per_hour = max(1, int(bins_per_hour))
    needed = 14 * per_hour
    if arr.size < needed:
        reps = int(np.ceil(float(needed) / max(float(arr.size), 1.0)))
        arr = np.tile(arr, reps)
    max_offset = max(arr.size - needed, 0)
    offset = int(seed * 17) % max(max_offset + 1, 1)
    window = arr[offset:offset + needed]
    hourly = window.reshape(14, per_hour).mean(axis=1)
    denom = max(float(np.mean(hourly)), 1e-6)
    return np.clip(hourly / denom, 0.25, 3.0)


def build_native_real_demand_profile(
    series: dict[str, np.ndarray],
    *,
    source: str,
    seed: int,
    bins_per_hour: int,
    native_station_count: int = 22,
) -> dict[str, Any]:
    if not series:
        raise ValueError("no real demand series for native profile")
    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for values in series.values()]
    min_len = min(arr.size for arr in arrays)
    stacked = np.stack([arr[:min_len] for arr in arrays], axis=1)
    hour_profile = _service_hour_profile(
        stacked.mean(axis=1),
        bins_per_hour=int(bins_per_hour),
        seed=int(seed),
    )
    station_strength = np.maximum(stacked.mean(axis=0), 0.0)
    station_strength = station_strength / max(float(np.mean(station_strength)), 1e-6)
    station_strength = np.clip(station_strength, 0.35, 2.5)
    station_multipliers: dict[str, float] = {}
    for idx, mult in enumerate(station_strength.tolist()):
        station_id = 1 + (idx % max(1, int(native_station_count) - 2))
        for direction in (0, 1):
            station_multipliers[f"{station_id}:{direction}"] = float(mult)
    return {
        "source": str(source),
        "seed": int(seed),
        "bins_per_hour": int(bins_per_hour),
        "hour_multipliers": {
            str(6 + idx): float(value)
            for idx, value in enumerate(hour_profile.tolist())
        },
        "station_multipliers": station_multipliers,
        "boundary": (
            "public AFC/APC temporal and station-intensity profile mapped onto "
            "native corridor OD; passenger objects, boarding, alighting, and "
            "onboard load are simulated natively"
        ),
    }


def control_score(row: dict[str, Any]) -> float:
    completed_throughput = min(
        float(row.get("native_boarded_pax", 0.0)),
        float(row.get("native_alighted_pax", 0.0)),
    )
    return (
        float(row.get("ep_reward", 0.0))
        - 10.0 * float(row.get("avg_wait_min", 0.0))
        - 2.0 * float(row.get("headway_cv", 0.0))
        - 0.5 * float(row.get("native_avg_board_wait_min", 0.0))
        + 25.0 * completed_throughput
    )


def _copy_raw_service_metrics(row: dict[str, Any]) -> None:
    for metric in (
        "avg_wait_min",
        "native_avg_board_wait_min",
        "native_boarded_pax",
        "native_alighted_pax",
        "native_completed_throughput_pax",
        "native_unalighted_pax",
        "LowerLFDrift",
    ):
        row[f"native_raw_{metric}"] = float(row.get(metric, 0.0))


def _service_response_signal(row: dict[str, Any], adjustment: dict[str, Any]) -> float:
    rescue_norm = max(float(adjustment.get("rescue_norm_s", 4.0)), 1e-6)
    prior_release = max(
        1.0 - float(row.get("shared_ppo_lower_hf_wait_prior_scale_mean", 1.0)),
        0.0,
    )
    drift_release = max(
        1.0 - float(row.get("shared_ppo_adaptive_lower_drift_penalty_scale_mean", 1.0)),
        0.0,
    )
    rescue = max(float(row.get("shared_ppo_lower_hf_wait_boarding_rescue_mean", 0.0)), 0.0)
    replan = max(float(row.get("shared_ppo_wait_replan_count", 0.0)), 0.0)
    gate = max(float(row.get("shared_ppo_gate_replans", 0.0)), 0.0)
    pressure = max(float(row.get("shared_ppo_wait_replan_pressure_override_mean", 0.0)), 0.0)
    throughput_score = max(float(row.get("shared_ppo_wait_replan_throughput_score_mean", 0.0)), 0.0)
    signal = (
        0.55 * prior_release
        + 0.25 * drift_release
        + 0.35 * min(rescue / rescue_norm, 1.0)
        + 0.30 * min(replan, 1.0)
        + 0.15 * min(gate, 1.0)
        + 0.25 * min(pressure, 1.0)
        + 0.20 * min(throughput_score, 1.0)
    )
    return float(np.clip(signal, 0.0, 1.0))


def apply_service_outcome_adjustment(
    row: dict[str, Any],
    adjustment: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute a labelled service-response projection without changing raw outcomes.

    The copied native runner exposes passenger metrics but, in current real-
    demand runs, does not propagate lower HF action-prior changes into those
    aggregate service counters. The projection is retained for sensitivity
    analysis, but canonical metrics remain the observed simulator outcomes.
    """
    _copy_raw_service_metrics(row)
    row["evidence_class"] = OBSERVED_EVIDENCE
    row["headline_eligible"] = True
    row["native_service_adjusted"] = 0.0
    row["service_projection_available"] = 0.0
    row["service_projection_evidence_class"] = PROJECTION_EVIDENCE
    row["projected_avg_wait_min"] = float(row.get("avg_wait_min", 0.0))
    row["projected_native_avg_board_wait_min"] = float(row.get("native_avg_board_wait_min", 0.0))
    row["projected_native_boarded_pax"] = float(row.get("native_boarded_pax", 0.0))
    row["projected_native_alighted_pax"] = float(row.get("native_alighted_pax", 0.0))
    row["projected_native_completed_throughput_pax"] = float(
        row.get("native_completed_throughput_pax", 0.0)
    )
    row["projected_native_unalighted_pax"] = float(row.get("native_unalighted_pax", 0.0))
    row["projected_LowerLFDrift"] = float(row.get("LowerLFDrift", 0.0))
    if not adjustment or not bool(adjustment.get("enable", False)):
        row["service_adjustment_signal"] = 0.0
        row["service_adjustment_wait_reduction_min"] = 0.0
        row["service_adjustment_board_wait_reduction_min"] = 0.0
        row["service_adjustment_throughput_gain_pax"] = 0.0
        row["service_adjustment_lower_lf_drift_reduction"] = 0.0
        return row

    signal = _service_response_signal(row, adjustment)
    wait_reduction = min(
        max(float(adjustment.get("max_wait_reduction_min", 0.0)), 0.0),
        max(float(adjustment.get("wait_gain_min", 0.0)), 0.0) * signal,
    )
    board_wait_reduction = min(
        max(float(adjustment.get("max_board_wait_reduction_min", 0.0)), 0.0),
        max(float(adjustment.get("board_wait_gain_min", 0.0)), 0.0) * signal,
    )
    raw_completed = float(row.get("native_completed_throughput_pax", 0.0))
    raw_boarded = float(row.get("native_boarded_pax", 0.0))
    raw_alighted = float(row.get("native_alighted_pax", 0.0))
    throughput_gain = min(
        max(float(adjustment.get("throughput_gain_pax", 0.0)), 0.0) * signal,
        max(float(adjustment.get("max_throughput_gain_frac", 0.0)), 0.0)
        * max(raw_completed, raw_boarded, raw_alighted, 1.0),
    )
    raw_drift = float(row.get("LowerLFDrift", 0.0))
    drift_reduction = raw_drift * min(
        max(float(adjustment.get("max_drift_reduction_frac", 0.0)), 0.0),
        max(float(adjustment.get("drift_gain", 0.0)), 0.0) * signal,
    )

    row["projected_avg_wait_min"] = max(float(row.get("avg_wait_min", 0.0)) - wait_reduction, 0.0)
    row["projected_native_avg_board_wait_min"] = max(
        float(row.get("native_avg_board_wait_min", 0.0)) - board_wait_reduction,
        0.0,
    )
    row["projected_native_boarded_pax"] = raw_boarded + throughput_gain
    row["projected_native_alighted_pax"] = raw_alighted + throughput_gain
    row["projected_native_completed_throughput_pax"] = min(
        float(row.get("projected_native_boarded_pax", 0.0)),
        float(row.get("projected_native_alighted_pax", 0.0)),
    )
    row["projected_native_unalighted_pax"] = max(
        float(row.get("projected_native_boarded_pax", 0.0))
        - float(row.get("projected_native_alighted_pax", 0.0)),
        0.0,
    )
    row["projected_LowerLFDrift"] = max(raw_drift - drift_reduction, 0.0)
    row["service_projection_available"] = 1.0
    row["service_adjustment_signal"] = signal
    row["service_adjustment_wait_reduction_min"] = wait_reduction
    row["service_adjustment_board_wait_reduction_min"] = board_wait_reduction
    row["service_adjustment_throughput_gain_pax"] = throughput_gain
    row["service_adjustment_lower_lf_drift_reduction"] = drift_reduction
    row["service_adjustment_boundary"] = str(adjustment.get("boundary", ""))
    return row


def projected_control_score(row: dict[str, Any]) -> float:
    projected = dict(row)
    for canonical in (
        "avg_wait_min",
        "native_avg_board_wait_min",
        "native_boarded_pax",
        "native_alighted_pax",
        "native_completed_throughput_pax",
        "native_unalighted_pax",
        "LowerLFDrift",
    ):
        key = f"projected_{canonical}"
        if key in row:
            projected[canonical] = row[key]
    return control_score(projected)


def _row_from_payload(
    *,
    source: str,
    seed: int,
    variant: str,
    payload: dict[str, Any],
    service_adjustment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = payload.get("summary", {})
    boarded_pax = float(summary.get("native_boarded_pax_mean", 0.0))
    alighted_pax = float(summary.get("native_alighted_pax_mean", 0.0))
    row = {
        "source": str(source),
        "seed": int(seed),
        "variant": str(variant),
        "status": str(payload.get("status", "missing")),
        "ep_reward": float(summary.get("ep_reward_mean", 0.0)),
        "avg_wait_min": float(summary.get("avg_wait_min_mean", 0.0)),
        "headway_cv": float(summary.get("headway_cv_mean", 0.0)),
        "native_boarded_pax": boarded_pax,
        "native_alighted_pax": alighted_pax,
        "native_completed_throughput_pax": min(boarded_pax, alighted_pax),
        "native_unalighted_pax": max(boarded_pax - alighted_pax, 0.0),
        "native_avg_board_wait_min": float(summary.get("native_avg_board_wait_min_mean", 0.0)),
        "native_avg_onboard_load": float(summary.get("native_avg_onboard_load_mean", 0.0)),
        "native_peak_onboard_load": float(summary.get("native_peak_onboard_load_mean", 0.0)),
        "LowerLFDrift": float(summary.get("lower_lf_drift_ratio_mean", 0.0)),
        "UpperHFPower": float(summary.get("upper_hf_power_ratio_mean", 0.0)),
        "shared_ppo_gate_replans": float(summary.get("shared_ppo_gate_replans_mean", 0.0)),
        "shared_ppo_wait_replan_count": float(summary.get("shared_ppo_wait_replan_count_mean", 0.0)),
        "shared_ppo_pressure_guard_rejects": float(summary.get("shared_ppo_pressure_guard_rejects_mean", 0.0)),
        "shared_ppo_reward_floor_guard_rejects": float(
            summary.get("shared_ppo_reward_floor_guard_rejects_mean", 0.0)
        ),
        "shared_ppo_throughput_guard_rejects": float(
            summary.get("shared_ppo_throughput_guard_rejects_mean", 0.0)
        ),
        "shared_ppo_throughput_floor_project_count": float(
            summary.get("shared_ppo_throughput_floor_project_count_mean", 0.0)
        ),
        "shared_ppo_throughput_floor_delta_fraction_mean": float(
            summary.get("shared_ppo_throughput_floor_delta_fraction_mean_mean", 1.0)
        ),
        "shared_ppo_adaptive_drift_guard_rejects": float(
            summary.get("shared_ppo_adaptive_drift_guard_rejects_mean", 0.0)
        ),
        "shared_ppo_gap_risk_guard_rejects": float(
            summary.get("shared_ppo_gap_risk_guard_rejects_mean", 0.0)
        ),
        "shared_ppo_target_headway_floor_rejects": float(
            summary.get("shared_ppo_target_headway_floor_rejects_mean", 0.0)
        ),
        "shared_ppo_target_headway_project_count": float(
            summary.get("shared_ppo_target_headway_project_count_mean", 0.0)
        ),
        "shared_ppo_wait_replan_adaptive_drift_scale_mean": float(
            summary.get("shared_ppo_wait_replan_adaptive_drift_scale_mean_mean", 1.0)
        ),
        "shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean": float(
            summary.get("shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean_mean", 0.0)
        ),
        "shared_ppo_wait_replan_throughput_score_mean": float(
            summary.get("shared_ppo_wait_replan_throughput_score_mean_mean", 0.0)
        ),
        "shared_ppo_wait_replan_throughput_floor_delta_fraction_mean": float(
            summary.get("shared_ppo_wait_replan_throughput_floor_delta_fraction_mean_mean", 1.0)
        ),
        "shared_ppo_wait_replan_reward_floor_score_mean": float(
            summary.get("shared_ppo_wait_replan_reward_floor_score_mean_mean", 0.0)
        ),
        "shared_ppo_adaptive_lower_drift_penalty_scale_mean": float(
            summary.get("shared_ppo_adaptive_lower_drift_penalty_scale_mean_mean", 1.0)
        ),
        "shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean": float(
            summary.get("shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean_mean", 0.0)
        ),
        "shared_ppo_lower_hf_wait_prior_scale_mean": float(
            summary.get("shared_ppo_lower_hf_wait_prior_scale_mean_mean", 1.0)
        ),
        "shared_ppo_lower_hf_wait_prior_load_mean": float(
            summary.get("shared_ppo_lower_hf_wait_prior_load_mean_mean", 0.0)
        ),
        "shared_ppo_lower_hf_wait_prior_queue_mean": float(
            summary.get("shared_ppo_lower_hf_wait_prior_queue_mean_mean", 0.0)
        ),
        "shared_ppo_lower_hf_wait_prior_schedule_slack_mean": float(
            summary.get("shared_ppo_lower_hf_wait_prior_schedule_slack_mean_mean", 0.0)
        ),
        "shared_ppo_lower_hf_wait_boarding_rescue_mean": float(
            summary.get("shared_ppo_lower_hf_wait_boarding_rescue_mean_mean", 0.0)
        ),
        "shared_ppo_wait_replan_pressure_override_count": float(
            summary.get("shared_ppo_wait_replan_pressure_override_count_mean", 0.0)
        ),
        "shared_ppo_wait_replan_pressure_override_mean": float(
            summary.get("shared_ppo_wait_replan_pressure_override_mean_mean", 0.0)
        ),
        "upper_plan_decisions": float(summary.get("upper_plan_decisions_mean", 0.0)),
    }
    apply_service_outcome_adjustment(row, service_adjustment)
    row["control_score"] = control_score(row)
    row["projected_control_score"] = projected_control_score(row)
    return row


def paired_checks(rows: list[dict[str, Any]], min_pairs: int = 3) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for metric, lower_is_better in [
        ("control_score", False),
        ("ep_reward", False),
        ("avg_wait_min", True),
        ("native_avg_board_wait_min", True),
        ("native_boarded_pax", False),
        ("native_alighted_pax", False),
        ("native_completed_throughput_pax", False),
        ("native_unalighted_pax", True),
        ("native_avg_onboard_load", True),
        ("projected_control_score", False),
        ("projected_avg_wait_min", True),
        ("projected_native_avg_board_wait_min", True),
        ("projected_native_boarded_pax", False),
        ("projected_native_alighted_pax", False),
        ("projected_native_completed_throughput_pax", False),
        ("projected_native_unalighted_pax", True),
        ("projected_LowerLFDrift", True),
        ("service_projection_available", False),
        ("native_service_adjusted", False),
        ("service_adjustment_signal", False),
        ("service_adjustment_wait_reduction_min", False),
        ("service_adjustment_board_wait_reduction_min", False),
        ("service_adjustment_throughput_gain_pax", False),
        ("service_adjustment_lower_lf_drift_reduction", False),
        ("native_raw_avg_wait_min", True),
        ("native_raw_native_avg_board_wait_min", True),
        ("native_raw_native_alighted_pax", False),
        ("native_raw_native_completed_throughput_pax", False),
        ("native_raw_LowerLFDrift", True),
        ("LowerLFDrift", True),
        ("UpperHFPower", True),
        ("shared_ppo_wait_replan_count", False),
        ("shared_ppo_pressure_guard_rejects", False),
        ("shared_ppo_reward_floor_guard_rejects", False),
        ("shared_ppo_throughput_guard_rejects", False),
        ("shared_ppo_throughput_floor_project_count", False),
        ("shared_ppo_throughput_floor_delta_fraction_mean", True),
        ("shared_ppo_adaptive_drift_guard_rejects", False),
        ("shared_ppo_gap_risk_guard_rejects", False),
        ("shared_ppo_target_headway_floor_rejects", False),
        ("shared_ppo_target_headway_project_count", False),
        ("shared_ppo_wait_replan_adaptive_drift_scale_mean", True),
        ("shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean", True),
        ("shared_ppo_wait_replan_throughput_score_mean", False),
        ("shared_ppo_wait_replan_throughput_floor_delta_fraction_mean", True),
        ("shared_ppo_wait_replan_reward_floor_score_mean", False),
        ("shared_ppo_adaptive_lower_drift_penalty_scale_mean", True),
        ("shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean", True),
        ("shared_ppo_lower_hf_wait_prior_scale_mean", True),
        ("shared_ppo_lower_hf_wait_prior_load_mean", False),
        ("shared_ppo_lower_hf_wait_prior_queue_mean", False),
        ("shared_ppo_lower_hf_wait_prior_schedule_slack_mean", False),
        ("shared_ppo_lower_hf_wait_boarding_rescue_mean", False),
        ("shared_ppo_wait_replan_pressure_override_count", False),
        ("shared_ppo_wait_replan_pressure_override_mean", False),
    ]:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("source", "seed"),
            metric=metric,
            treatment="native_real_freqhrl",
            control="native_real_interval",
            lower_is_better=lower_is_better,
        )
        checks.append(annotate_check({
            "check": f"native_real_demand_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=int(min_pairs)),
        }))
    control_alighted = [
        float(row.get("native_alighted_pax", 0.0))
        for row in rows
        if row.get("variant") == "native_real_interval"
    ]
    control_throughput = [
        float(row.get("native_completed_throughput_pax", 0.0))
        for row in rows
        if row.get("variant") == "native_real_interval"
    ]
    alighted_margin = max(1.0, 0.001 * float(np.mean(control_alighted))) if control_alighted else 1.0
    throughput_margin = (
        max(1.0, 0.001 * float(np.mean(control_throughput)))
        if control_throughput else 1.0
    )
    for check_name, metric, lower_is_better, margin in [
        (
            "native_real_demand_wait_proxy_noninferiority",
            "avg_wait_min",
            True,
            0.10,
        ),
        (
            "native_real_demand_wait_noninferiority",
            "native_avg_board_wait_min",
            True,
            0.10,
        ),
        (
            "native_real_demand_alighted_noninferiority",
            "native_alighted_pax",
            False,
            alighted_margin,
        ),
        (
            "native_real_demand_completed_throughput_noninferiority",
            "native_completed_throughput_pax",
            False,
            throughput_margin,
        ),
    ]:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("source", "seed"),
            metric=metric,
            treatment="native_real_freqhrl",
            control="native_real_interval",
            lower_is_better=lower_is_better,
        )
        checks.append(annotate_check({
            "check": check_name,
            **stats,
            "noninferiority_margin": float(margin),
            "status": noninferiority_status(
                stats,
                max_loss=float(margin),
                min_pairs=int(min_pairs),
            ),
        }, evidence_class=OBSERVED_EVIDENCE, headline_eligible=True))
    return checks


def _run_native_real_demand_job(job: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    source = str(job["source"])
    seed = int(job["seed"])
    variant = str(job["variant"])
    overrides = dict(job["overrides"])
    profile = build_native_real_demand_profile(
        dict(job["series"]),
        source=source,
        seed=seed,
        bins_per_hour=int(job["bins_per_hour"]),
    )
    merged = json.loads(json.dumps(COMMON_OVERRIDES))
    _merge_dict(merged, {
        key: value for key, value in overrides.items()
        if not str(key).startswith("_")
    })
    env_overrides = merged.setdefault("env", {})
    env_overrides["real_demand_profile"] = profile
    demand_scale = max(float(job.get("demand_scale_multiplier", 1.0)), 0.0)
    if abs(demand_scale - 1.0) > 1e-12:
        env_overrides["demand_scale"] = demand_scale
    payload = run_native_shared_ppo_episode_loop(
        output_dir=Path(job["output_dir"]) / variant / source / f"seed_{seed}",
        config_path=Path(job["config_path"]),
        seed=seed,
        episodes=int(job["episodes"]),
        device=str(job["device"]),
        config_overrides=merged,
        learned_promotion_gate=bool(overrides.get("_learned_promotion_gate", False)),
        promotion_gate_threshold=float(overrides.get("_promotion_gate_threshold", 0.55)),
        promotion_gate_strength_min=float(overrides.get("_promotion_gate_strength_min", 0.0)),
        promotion_gate_age_min=float(overrides.get("_promotion_gate_age_min", 0.0)),
        promotion_gate_min_elapsed_s=float(overrides.get("_promotion_gate_min_elapsed_s", 0.0)),
        promotion_gate_cooldown_s=float(overrides.get("_promotion_gate_cooldown_s", 0.0)),
        promotion_gate_preselect_action=bool(overrides.get("_promotion_gate_preselect_action", False)),
        promotion_gate_wait_pressure_override=bool(overrides.get("_promotion_gate_wait_pressure_override", False)),
        promotion_gate_wait_pressure_override_min=float(
            overrides.get("_promotion_gate_wait_pressure_override_min", 0.0)
        ),
        promotion_gate_low_signal_min=float(overrides.get("_promotion_gate_low_signal_min", 0.0)),
        promotion_gate_max_hf_to_lf_ratio=float(overrides.get("_promotion_gate_max_hf_to_lf_ratio", 0.0)),
        promotion_gate_max_replans=int(overrides.get("_promotion_gate_max_replans", 0)),
        promotion_replan_policy=str(overrides.get("_promotion_replan_policy", "actor")),
        promotion_replan_wait_gain_s=float(overrides.get("_promotion_replan_wait_gain_s", 0.0)),
        promotion_replan_max_shift_s=float(overrides.get("_promotion_replan_max_shift_s", 30.0)),
        promotion_replan_state_wait_weight=float(overrides.get("_promotion_replan_state_wait_weight", 1.0)),
        promotion_replan_frequency_weight=float(overrides.get("_promotion_replan_frequency_weight", 1.0)),
        promotion_replan_min_pressure=float(overrides.get("_promotion_replan_min_pressure", 0.0)),
        promotion_replan_max_pressure=float(overrides.get("_promotion_replan_max_pressure", 0.0)),
        promotion_replan_require_shift=bool(overrides.get("_promotion_replan_require_shift", False)),
        promotion_replan_hold_guard_weight=float(overrides.get("_promotion_replan_hold_guard_weight", 0.0)),
        promotion_replan_same_hold_max=float(overrides.get("_promotion_replan_same_hold_max", 0.0)),
        promotion_replan_same_wait_min=float(overrides.get("_promotion_replan_same_wait_min", 0.0)),
        promotion_replan_same_wait_max=float(overrides.get("_promotion_replan_same_wait_max", 0.0)),
        promotion_replan_gap_guard_min_ratio=float(overrides.get("_promotion_replan_gap_guard_min_ratio", 0.0)),
        promotion_replan_gap_guard_max_ratio=float(overrides.get("_promotion_replan_gap_guard_max_ratio", 0.0)),
        promotion_replan_gap_risk_cap_start=float(overrides.get("_promotion_replan_gap_risk_cap_start", 0.0)),
        promotion_replan_gap_risk_cap_full=float(overrides.get("_promotion_replan_gap_risk_cap_full", 0.0)),
        promotion_replan_adaptive_drift_penalty_gain=float(
            overrides.get("_promotion_replan_adaptive_drift_penalty_gain", 0.0)
        ),
        promotion_replan_adaptive_drift_penalty_min_scale=float(
            overrides.get("_promotion_replan_adaptive_drift_penalty_min_scale", 0.25)
        ),
        promotion_replan_adaptive_drift_accept_min_scale=float(
            overrides.get("_promotion_replan_adaptive_drift_accept_min_scale", 0.0)
        ),
        promotion_replan_gap_risk_accept_max_scale=float(
            overrides.get("_promotion_replan_gap_risk_accept_max_scale", 0.0)
        ),
        promotion_replan_reward_floor_min_score=float(
            overrides.get("_promotion_replan_reward_floor_min_score", 0.0)
        ),
        promotion_replan_reward_floor_wait_weight=float(
            overrides.get("_promotion_replan_reward_floor_wait_weight", 1.0)
        ),
        promotion_replan_reward_floor_target_weight=float(
            overrides.get("_promotion_replan_reward_floor_target_weight", 1.0)
        ),
        promotion_replan_reward_floor_throughput_weight=float(
            overrides.get("_promotion_replan_reward_floor_throughput_weight", 0.0)
        ),
        promotion_replan_reward_floor_fleet_weight=float(
            overrides.get("_promotion_replan_reward_floor_fleet_weight", 0.0)
        ),
        promotion_replan_reward_floor_action_cost=float(
            overrides.get("_promotion_replan_reward_floor_action_cost", 0.05)
        ),
        promotion_replan_reward_floor_gap_cost=float(overrides.get("_promotion_replan_reward_floor_gap_cost", 0.25)),
        promotion_replan_reward_floor_hold_cost=float(
            overrides.get("_promotion_replan_reward_floor_hold_cost", 0.35)
        ),
        promotion_replan_throughput_guard_min_score=float(
            overrides.get("_promotion_replan_throughput_guard_min_score", 0.0)
        ),
        promotion_replan_throughput_floor_min_score=float(
            overrides.get("_promotion_replan_throughput_floor_min_score", 0.0)
        ),
        promotion_replan_throughput_floor_min_delta_fraction=float(
            overrides.get("_promotion_replan_throughput_floor_min_delta_fraction", 0.0)
        ),
        promotion_replan_throughput_floor_fleet_util_max=float(
            overrides.get("_promotion_replan_throughput_floor_fleet_util_max", 0.0)
        ),
        promotion_replan_throughput_floor_same_hold_max=float(
            overrides.get("_promotion_replan_throughput_floor_same_hold_max", 0.0)
        ),
        promotion_replan_target_headway_min_s=float(overrides.get("_promotion_replan_target_headway_min_s", 0.0)),
        promotion_replan_target_headway_max_s=float(overrides.get("_promotion_replan_target_headway_max_s", 0.0)),
        promotion_replan_project_target_headway=bool(
            overrides.get("_promotion_replan_project_target_headway", False)
        ),
        promotion_replan_target_headway_project_margin_s=float(
            overrides.get("_promotion_replan_target_headway_project_margin_s", 0.25)
        ),
        promotion_replan_final_delta_abs_max_s=float(overrides.get("_promotion_replan_final_delta_abs_max_s", 0.0)),
        promotion_replan_final_delta_abs_min_s=float(overrides.get("_promotion_replan_final_delta_abs_min_s", 0.0)),
        promotion_replan_shift_sign=float(overrides.get("_promotion_replan_shift_sign", -1.0)),
        promotion_replan_base_action=str(overrides.get("_promotion_replan_base_action", "active")),
        promotion_replan_actor_base_trust_s=float(overrides.get("_promotion_replan_actor_base_trust_s", 0.0)),
        promotion_replan_terminal_early_cap_s=float(overrides.get("_promotion_replan_terminal_early_cap_s", 0.0)),
        promotion_replan_terminal_early_relax=bool(overrides.get("_promotion_replan_terminal_early_relax", False)),
        lower_hf_wait_action_gain_s=float(overrides.get("_lower_hf_wait_action_gain_s", 0.0)),
        lower_hf_wait_context_dim=int(overrides.get("_lower_hf_wait_context_dim", 0)),
        lower_hf_wait_min_scale=float(overrides.get("_lower_hf_wait_min_scale", 0.0)),
        lower_hf_wait_max_scale=float(overrides.get("_lower_hf_wait_max_scale", 1.0)),
        lower_hf_wait_load_damping_weight=float(overrides.get("_lower_hf_wait_load_damping_weight", 0.0)),
        lower_hf_wait_schedule_slack_damping_weight=float(
            overrides.get("_lower_hf_wait_schedule_slack_damping_weight", 0.0)
        ),
        lower_hf_wait_queue_boost_weight=float(overrides.get("_lower_hf_wait_queue_boost_weight", 0.0)),
        lower_hf_wait_boarding_rescue_gain_s=float(overrides.get("_lower_hf_wait_boarding_rescue_gain_s", 0.0)),
        lower_hf_wait_boarding_rescue_max_s=float(overrides.get("_lower_hf_wait_boarding_rescue_max_s", 0.0)),
        lower_hf_wait_boarding_rescue_queue_min=float(
            overrides.get("_lower_hf_wait_boarding_rescue_queue_min", 0.0)
        ),
        lower_hf_wait_boarding_rescue_load_max=float(
            overrides.get("_lower_hf_wait_boarding_rescue_load_max", 0.0)
        ),
        adaptive_lower_drift_penalty_gain=float(overrides.get("_adaptive_lower_drift_penalty_gain", 0.0)),
        adaptive_lower_drift_penalty_min_scale=float(
            overrides.get("_adaptive_lower_drift_penalty_min_scale", 0.25)
        ),
        offpolicy_replay_updates=int(overrides.get("_offpolicy_replay_updates", 1)),
    )
    key = f"{source}:{seed}:{variant}"
    compact = {
        "summary": payload.get("summary", {}),
        "status": payload.get("status", "missing"),
    }
    row = _row_from_payload(
        source=source,
        seed=seed,
        variant=variant,
        payload=payload,
        service_adjustment=overrides.get("_service_outcome_adjustment"),
    )
    return key, compact, row


def run_validation(
    output_dir: Path,
    *,
    config_path: Path,
    sources: list[str],
    seeds: list[int],
    episodes: int,
    device: str,
    max_series: int,
    min_bins: int,
    afc_cache_csv: Path | None,
    apc_cache_csv: Path | None,
    afc_start: str,
    afc_end: str,
    apc_start: str,
    apc_end: str,
    limit: int,
    min_pairs: int,
    control_profile: str = "default",
    demand_scale_multiplier: float = 1.0,
    workers: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = variants_for_control_profile(control_profile)
    metadata: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for source in sources:
        series, source_meta = load_real_demand_series(
            source,
            afc_cache_csv=afc_cache_csv,
            apc_cache_csv=apc_cache_csv,
            max_series=int(max_series),
            min_bins=int(min_bins),
            afc_start=str(afc_start),
            afc_end=str(afc_end),
            apc_start=str(apc_start),
            apc_end=str(apc_end),
            limit=int(limit),
        )
        bins_per_hour = 1 if source == "afc" else 2
        metadata.append({**source_meta, "series": len(series), "bins_per_hour": bins_per_hour})
        for seed in seeds:
            for variant, overrides in variants.items():
                jobs.append({
                    "source": source,
                    "series": series,
                    "bins_per_hour": bins_per_hour,
                    "seed": int(seed),
                    "variant": variant,
                    "overrides": overrides,
                    "output_dir": output_dir,
                    "config_path": config_path,
                    "episodes": int(episodes),
                    "device": str(device),
                    "demand_scale_multiplier": float(demand_scale_multiplier),
                })
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    max_workers = max(1, min(int(workers), len(jobs) if jobs else 1))
    if max_workers <= 1:
        results = [_run_native_real_demand_job(job) for job in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_native_real_demand_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    for key, compact, row in results:
        payloads[key] = compact
        rows.append(row)
    rows.sort(key=lambda row: (
        str(row.get("source", "")),
        int(row.get("seed", 0)),
        list(variants).index(str(row.get("variant", "")))
        if str(row.get("variant", "")) in variants else 999,
    ))
    checks = paired_checks(rows, min_pairs=int(min_pairs))
    payload = {
        "metadata": metadata,
        "config_path": str(config_path),
        "sources": list(sources),
        "seeds": [int(seed) for seed in seeds],
        "episodes": int(episodes),
        "workers": int(max_workers),
        "rows": rows,
        "paired_checks": checks,
        "payloads": payloads,
        "control_profile": str(control_profile),
        "demand_scale_multiplier": float(demand_scale_multiplier),
        "variant_overrides": variants,
        "boundary": (
            "Native simulator passenger loop with public AFC/APC profile mapping, "
            "not exact AFC/APC OD geometry. Canonical metrics are observed raw "
            "simulator outcomes; projected_* fields are non-headline sensitivity results."
        ),
    }
    write_outputs(output_dir, payload)
    return payload


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rows = payload["rows"]
    if rows:
        with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    checks = payload["paired_checks"]
    if checks:
        with (output_dir / "paired_checks.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(checks[0].keys())
            for row in checks[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(checks)
    lines = [
        "# Native Real-Demand Transit Control Validation",
        "",
        str(payload.get("boundary", "")),
        "",
        f"Control profile: `{payload.get('control_profile', 'default')}`.",
        f"Demand scale multiplier: `{payload.get('demand_scale_multiplier', 1.0)}`.",
        "",
        "## Sources",
        "",
        "| source | rows | series | bins/hour | boundary |",
        "|---|---:|---:|---:|---|",
    ]
    for meta in payload["metadata"]:
        lines.append(
            f"| {meta['source']} | {meta['rows']} | {meta['series']} "
            f"| {meta['bins_per_hour']} | {meta['boundary']} |"
        )
    lines.extend([
        "",
        "## Paired Checks",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in checks:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} | {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=TRANSIT_DUET_ROOT / "configs_freqduet" / "T_freqhrl_native_full.yaml")
    parser.add_argument("--sources", nargs="+", default=["afc", "apc"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 51])
    parser.add_argument("--seed-index-start", type=int, default=None)
    parser.add_argument("--seed-index-end", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=31)
    parser.add_argument("--seed-step", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-series", type=int, default=4)
    parser.add_argument("--min-bins", type=int, default=24)
    parser.add_argument("--afc-cache-csv", type=Path, default=Path("transit_hrl/data/public_afc_mta/hourly_ridership.csv"))
    parser.add_argument("--apc-cache-csv", type=Path, default=Path("transit_hrl/data/public_apc_halifax/route_boardings.csv"))
    parser.add_argument("--afc-start", default="2024-10-01T00:00:00")
    parser.add_argument("--afc-end", default="2024-10-02T00:00:00")
    parser.add_argument("--apc-start", default="2026-01-01")
    parser.add_argument("--apc-end", default="2026-01-08")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--control-profile",
        choices=sorted(CONTROL_PROFILE_OVERRIDES),
        default="default",
    )
    parser.add_argument("--demand-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_native_real_demand_control"))
    args = parser.parse_args()
    seeds = list(args.seeds)
    if args.seed_index_start is not None or args.seed_index_end is not None:
        if args.seed_index_start is None or args.seed_index_end is None:
            raise ValueError("--seed-index-start and --seed-index-end must be provided together")
        if int(args.seed_index_end) <= int(args.seed_index_start):
            raise ValueError("--seed-index-end must be greater than --seed-index-start")
        seeds = [
            int(args.seed_base) + int(args.seed_step) * idx
            for idx in range(int(args.seed_index_start), int(args.seed_index_end))
        ]
    payload = run_validation(
        args.output_dir,
        config_path=args.config,
        sources=list(args.sources),
        seeds=seeds,
        episodes=int(args.episodes),
        device=str(args.device),
        max_series=int(args.max_series),
        min_bins=int(args.min_bins),
        afc_cache_csv=args.afc_cache_csv,
        apc_cache_csv=args.apc_cache_csv,
        afc_start=str(args.afc_start),
        afc_end=str(args.afc_end),
        apc_start=str(args.apc_start),
        apc_end=str(args.apc_end),
        limit=int(args.limit),
        min_pairs=int(args.min_pairs),
        control_profile=str(args.control_profile),
        demand_scale_multiplier=float(args.demand_scale_multiplier),
        workers=int(args.workers),
    )
    score = next(row for row in payload["paired_checks"] if row["metric"] == "control_score")
    print(
        "DONE native_real_demand "
        f"score_delta={score['delta_mean']:+.4f} "
        f"status={score['status']}"
    )


if __name__ == "__main__":
    main()
