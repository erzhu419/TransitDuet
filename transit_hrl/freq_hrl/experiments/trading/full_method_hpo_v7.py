"""Support-only nested HPO for the frozen-checkpoint Freq-HRL v7.3.2 protocol.

One model is trained per variant/candidate/training replicate on four complete,
independently reset support-regime episodes. Candidate selection never loads
``ood_period`` or ``promotion_recovery``. The five mechanism ablations are not
tuned independently: they inherit the selected full-method candidate and are
created only for confirmatory evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.experiments.reproducibility import (
    derive_seed,
    is_hex_digest,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)
from freq_hrl.rl import SMDPPPOConfig

from .metrics import (
    DEFAULT_TRAINING_REWARD_SCALE,
    METRIC_CONTRACT_VERSION,
    SELECTION_OBJECTIVE_VERSION,
    validation_utility,
)
from . import full_method_budget_plan_v732 as budget_plan
from . import full_method_confirmatory_plan_v732 as confirmatory_plan
from .offpolicy_baseline_validation import (
    run_offpolicy_episode,
    train_flat_offpolicy_baseline,
)
from .performance_validation import (
    SCENARIOS,
    SUPPORT_MIXTURE_COMPONENTS,
    SUPPORT_MIXTURE_SCENARIO,
)
from .ppo_actor_critic import (
    FLAT_PPO_MODES,
    FULL_METHOD_V7_IMPLEMENTATION_VERSION,
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
    V7_METHOD_CONTRACTS,
    evaluate_hf_lower_intervention,
    joint_flat_rollout,
    make_plan_mapper,
    promotion_gate_state_dim,
    resolve_method_contract,
    smdp_parameter_count,
    smdp_rollout,
    train_ppo_actor_critic,
)
from .strong_learned_baseline_validation import count_parameters


FULL_METHOD_TUNING_PROTOCOL_VERSION = "full_method_support_only_hpo_v7_3_2"
FULL_METHOD_HPO_IMPLEMENTATION_VERSION = (
    "full_method_hpo_null_rate_calibrated_promotion_v7_3_2_2026_08_08"
)
PROMOTION_CALIBRATION_PROTOCOL_VERSION = (
    "paired_advantage_bias_null_quantile_calibration_v2"
)
FULL_METHOD_IMPLEMENTATION_VERSION = FULL_METHOD_V7_IMPLEMENTATION_VERSION
EXECUTION_TIMELINE_CONTRACT = "causal_post_trade_v3"
CAPACITY_REFERENCE_METHOD_CONTRACT = "full_freq_hrl_v7"
TRAINING_SCENARIO = SUPPORT_MIXTURE_SCENARIO
SELECTION_SCENARIOS = tuple(SUPPORT_MIXTURE_COMPONENTS)
FORBIDDEN_HPO_SCENARIOS = frozenset({"ood_period", "promotion_recovery"})
DEFAULT_VOLUME_IMPACT_BPS = 10.0
DEFAULT_PLAN_BASIS_DIM = 3
DEFAULT_PLAN_HORIZON_S = 1800.0
DEFAULT_PLAN_EVAL_OFFSET_S = 300.0
DEFAULT_PLAN_COEFFICIENT_SCALE = 0.75
DEFAULT_TRAIN_SEEDS = (170003, 170021, 170029)
DEFAULT_PROMOTION_CALIBRATION_SEEDS = (180001, 180013, 180023)
DEFAULT_CHECKPOINT_VALIDATION_SEEDS = (190027, 190031, 190051)
DEFAULT_TUNING_SEEDS = (200003, 200009, 200017, 200023, 200029)
DEFAULT_PILOT_OPTIMIZER_SEEDS = (5003, 5009, 5011)
DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS = (6007, 6011, 6029, 6037, 6043)
ABLATION_PARENT_VARIANT = "freq_hrl_full_v7"
MIN_MECHANISM_REPLICATE_FRACTION = 0.8
MIN_HF_ACTION_SENSITIVITY = 1e-8
MIN_PLAN_ACTION_EFFECT = 1e-8
MAX_HF_BUDGET_RATIO = 1.0 + 1e-8
MAX_LOW_NOISE_PROMOTION_RATE = 0.25
MAX_STRESS_PROMOTION_RATE = 0.90
MIN_STRESS_PROMOTION_RATE = 0.02
MIN_STRESS_LOW_RATE_LIFT = 0.02
MIN_STRESS_LOW_PROBABILITY_LIFT = 1e-4
MIN_STRESS_LOW_ADVANTAGE_LIFT = 1e-3
MIN_ADVANTAGE_DECISION_ACCURACY = 0.55
PROMOTION_CALIBRATION_NULL_RATE_CAP = 0.20
CHECKPOINT_BOUNDARY_WINDOW_FRACTION = 0.125
MAX_CHECKPOINT_BOUNDARY_REPLICATE_FRACTION = 0.40


@dataclass(frozen=True)
class Variant:
    variant_id: str
    trainer_family: str
    candidate_family: str
    policy_mode: str
    method_contract: str
    scientific_role: str
    ablation_of: str = ""

    @property
    def inherits_full_selection(self) -> bool:
        return bool(self.ablation_of)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]


VARIANTS = (
    Variant("freq_hrl_full_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "full_freq_hrl_v7", "proposed_full_method"),
    Variant("freq_hrl_no_promotion_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "ablate_promotion_v7", "one_factor_ablation", ABLATION_PARENT_VARIANT),
    Variant("freq_hrl_no_hf_lower_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "ablate_hf_lower_v7", "one_factor_ablation", ABLATION_PARENT_VARIANT),
    Variant("freq_hrl_no_leakage_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "ablate_leakage_v7", "one_factor_ablation", ABLATION_PARENT_VARIANT),
    Variant("freq_hrl_no_lf_reference_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "ablate_lf_reference_v7", "one_factor_ablation", ABLATION_PARENT_VARIANT),
    Variant("freq_hrl_anchor_only_v7", "ppo", "frequency_ppo_v7_3", "freq_hrl", "ablate_upper_residual_v7", "reference_only_control", ABLATION_PARENT_VARIANT),
    Variant("flat_ppo_matched_v7", "ppo", "baseline_ppo_v7", "flat_ppo", "routing_core_v2", "capacity_matched_flat_baseline"),
    Variant("flat_gru_ppo_matched_v7", "ppo", "baseline_ppo_v7", "flat_gru_ppo", "routing_core_v2", "capacity_matched_flat_recurrent_baseline"),
    Variant("generic_hrl_ppo_matched_v7", "ppo", "baseline_ppo_v7", "generic_hrl_ppo", "curve_credit_control_v3", "capacity_matched_nonfrequency_hrl_baseline"),
    Variant("generic_hrl_gru_ppo_matched_v7", "ppo", "baseline_ppo_v7", "generic_hrl_gru_ppo", "curve_credit_control_v3", "capacity_matched_nonfrequency_recurrent_hrl_baseline"),
    Variant("flat_sac_matched_v7", "offpolicy", "offpolicy_v7", "flat_sac", "routing_core_v2", "capacity_matched_offpolicy_baseline"),
    Variant("flat_td3_matched_v7", "offpolicy", "offpolicy_v7", "flat_td3", "routing_core_v2", "capacity_matched_offpolicy_baseline"),
)
VARIANTS_BY_ID = {variant.variant_id: variant for variant in VARIANTS}
ALL_VARIANT_IDS = tuple(variant.variant_id for variant in VARIANTS)
HPO_VARIANT_IDS = tuple(
    variant.variant_id for variant in VARIANTS
    if not variant.inherits_full_selection
)


def _optimizer_parameters(learning_rate: float, init_log_std: float) -> dict[str, Any]:
    return {
        "hidden_dim": 64,
        "learning_rate": float(learning_rate),
        "epochs": 4,
        "minibatch_size": 512,
        "init_log_std": float(init_log_std),
        "reward_scale": DEFAULT_TRAINING_REWARD_SCALE,
    }


def _frequency_candidate(
    candidate_id: str,
    *,
    upper_lr: float,
    lower_lr: float,
    hf_lr: float,
    promotion_lr: float,
    init_log_std: float,
    leakage_scale: float,
    constraint_init: float,
    dual_lr: float,
    objective_weight: float,
    lower_budget: float,
    hf_budget: float,
    promotion_init_logit: float,
    promotion_replan_cost: float,
    lower_hf_order_scale: float,
    upper_period: int,
    min_upper_duration: int,
    reference_gain: float = 1.0,
    forecast_blend: float = 0.0,
    promotion_threshold: float = 0.4,
    promotion_adapt_gain: float = 0.20,
    promotion_cooldown_steps: int = 15,
    promotion_gate_interval_steps: int = 3,
    promotion_entropy_coef: float = 5e-5,
    promotion_rate_budget: float = 0.20,
    promotion_rate_coef: float = 0.10,
    promotion_counterfactual_coef: float = 1.0,
    promotion_credit_scale: float = 500.0,
    promotion_advantage_learning_rate: float = 1e-3,
    promotion_advantage_coef: float = 1.0,
    promotion_advantage_huber_delta: float = 0.05,
    promotion_advantage_threshold: float = 0.0,
) -> Candidate:
    return Candidate(candidate_id, "frequency_ppo_v7_3", {
        **_optimizer_parameters(lower_lr, init_log_std),
        "upper_learning_rate": float(upper_lr),
        "lower_learning_rate": float(lower_lr),
        "hf_learning_rate": float(hf_lr),
        "promotion_learning_rate": float(promotion_lr),
        "leakage_scale": float(leakage_scale),
        "lower_lf_constraint_coef": float(constraint_init),
        "lower_lf_constraint_target": 1.0,
        "lower_lf_dual_lr": float(dual_lr),
        "lower_lf_objective_weight": float(objective_weight),
        "lower_lf_budget_rms": float(lower_budget),
        "hf_lf_budget_rms": float(hf_budget),
        "lower_lf_effect_filter_window": 0,
        "lower_lf_effect_filter_gain": 1.0,
        "lower_lf_raw_recenter_gain": 0.0,
        "lower_lf_raw_recenter_scale": 0.10,
        "plan_smoothness_weight": 1e-5,
        "promotion_init_logit": float(promotion_init_logit),
        "promotion_entropy_coef": float(promotion_entropy_coef),
        "promotion_rate_budget": float(promotion_rate_budget),
        "promotion_rate_coef": float(promotion_rate_coef),
        "promotion_counterfactual_coef": float(
            promotion_counterfactual_coef
        ),
        "promotion_advantage_learning_rate": float(
            promotion_advantage_learning_rate
        ),
        "promotion_advantage_coef": float(promotion_advantage_coef),
        "promotion_advantage_huber_delta": float(
            promotion_advantage_huber_delta
        ),
        "promotion_replan_cost": float(promotion_replan_cost),
        "lower_hf_order_scale": float(lower_hf_order_scale),
        "upper_period": int(upper_period),
        "min_upper_duration": int(min_upper_duration),
        "promotion_credit_mode": "paired_plan_advantage",
        "promotion_credit_scale": float(promotion_credit_scale),
        "leakage_cost_mode": "fixed_rms_budget",
        "include_hf_predictability": True,
        "upper_plan_reference_mode": "causal_lf",
        "upper_plan_reference_gain": float(reference_gain),
        "upper_plan_reference_forecast_blend": float(forecast_blend),
        "hard_hf_budget_projection": True,
        "promotion_deterministic_threshold": float(promotion_threshold),
        "promotion_adapt_gain": float(promotion_adapt_gain),
        "promotion_cooldown_steps": int(promotion_cooldown_steps),
        "promotion_gate_interval_steps": int(promotion_gate_interval_steps),
        "promotion_deterministic_mode": "counterfactual_advantage",
        "promotion_advantage_threshold": float(
            promotion_advantage_threshold
        ),
        "promotion_advantage_target_threshold": float(
            promotion_advantage_threshold
        ),
        "upper_residual_action_scale": 1.0,
    })


def _baseline_candidate(candidate_id: str, learning_rate: float, init_log_std: float) -> Candidate:
    return Candidate(
        candidate_id,
        "baseline_ppo_v7",
        _optimizer_parameters(learning_rate, init_log_std),
    )


def _offpolicy_candidate(
    candidate_id: str,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
) -> Candidate:
    return Candidate(candidate_id, "offpolicy_v7", {
        "hidden_dim": 64,
        "learning_rate": float(learning_rate),
        "replay_capacity": 100_000,
        "warmup_steps": int(warmup_steps),
        "batch_size": int(batch_size),
        "updates_per_step": 1,
        "reward_scale": DEFAULT_TRAINING_REWARD_SCALE,
    })


FREQUENCY_CANDIDATES = (
    _frequency_candidate(
        "v73_conservative_zero",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=1e-4, promotion_lr=2e-4,
        init_log_std=-1.0, leakage_scale=5e-4, constraint_init=0.05,
        dual_lr=5e-4, objective_weight=0.0, lower_budget=0.0025,
        hf_budget=0.00075, promotion_init_logit=-2.3,
        promotion_replan_cost=2e-5, lower_hf_order_scale=0.0075,
        upper_period=45, min_upper_duration=10, reference_gain=1.0,
        forecast_blend=0.0, promotion_threshold=0.10,
        promotion_adapt_gain=0.20, promotion_cooldown_steps=12,
        promotion_gate_interval_steps=12, promotion_entropy_coef=3e-5,
        promotion_rate_budget=0.14, promotion_rate_coef=0.15,
        promotion_advantage_threshold=0.0,
    ),
    _frequency_candidate(
        "v73_conservative_margin",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=1e-4, promotion_lr=2e-4,
        init_log_std=-1.0, leakage_scale=5e-4, constraint_init=0.05,
        dual_lr=5e-4, objective_weight=0.0, lower_budget=0.0025,
        hf_budget=0.00075, promotion_init_logit=-2.3,
        promotion_replan_cost=2e-5, lower_hf_order_scale=0.0075,
        upper_period=45, min_upper_duration=10, reference_gain=1.0,
        forecast_blend=0.0, promotion_threshold=0.10,
        promotion_adapt_gain=0.20, promotion_cooldown_steps=12,
        promotion_gate_interval_steps=12, promotion_entropy_coef=3e-5,
        promotion_rate_budget=0.14, promotion_rate_coef=0.15,
        promotion_advantage_threshold=0.01,
    ),
    _frequency_candidate(
        "v73_forecast_small_margin",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=3e-4,
        init_log_std=-1.0, leakage_scale=5e-4, constraint_init=0.05,
        dual_lr=1e-3, objective_weight=1e-5, lower_budget=0.0020,
        hf_budget=0.00100, promotion_init_logit=-1.9,
        promotion_replan_cost=2.5e-5, lower_hf_order_scale=0.0075,
        upper_period=30, min_upper_duration=8, reference_gain=1.0,
        forecast_blend=0.25, promotion_threshold=0.14,
        promotion_adapt_gain=0.20, promotion_cooldown_steps=10,
        promotion_gate_interval_steps=10, promotion_entropy_coef=5e-5,
        promotion_rate_budget=0.20, promotion_rate_coef=0.10,
        promotion_advantage_threshold=0.005,
    ),
    _frequency_candidate(
        "v73_forecast_margin",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=3e-4,
        init_log_std=-1.0, leakage_scale=5e-4, constraint_init=0.05,
        dual_lr=1e-3, objective_weight=1e-5, lower_budget=0.0020,
        hf_budget=0.00100, promotion_init_logit=-1.9,
        promotion_replan_cost=2.5e-5, lower_hf_order_scale=0.0075,
        upper_period=30, min_upper_duration=8, reference_gain=1.0,
        forecast_blend=0.25, promotion_threshold=0.14,
        promotion_adapt_gain=0.20, promotion_cooldown_steps=10,
        promotion_gate_interval_steps=10, promotion_entropy_coef=5e-5,
        promotion_rate_budget=0.20, promotion_rate_coef=0.10,
        promotion_advantage_threshold=0.015,
    ),
    _frequency_candidate(
        "v73_balanced_margin",
        upper_lr=5e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=3e-4,
        init_log_std=-1.0, leakage_scale=7.5e-4, constraint_init=0.10,
        dual_lr=1e-3, objective_weight=2.5e-5, lower_budget=0.0025,
        hf_budget=0.00150, promotion_init_logit=-2.0,
        promotion_replan_cost=2.5e-5, lower_hf_order_scale=0.0100,
        upper_period=30, min_upper_duration=8, reference_gain=1.25,
        forecast_blend=0.0, promotion_threshold=0.13,
        promotion_adapt_gain=0.30, promotion_cooldown_steps=10,
        promotion_gate_interval_steps=10, promotion_entropy_coef=5e-5,
        promotion_rate_budget=0.18, promotion_rate_coef=0.10,
        promotion_advantage_threshold=0.015,
    ),
    _frequency_candidate(
        "v73_balanced_strict",
        upper_lr=5e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=3e-4,
        init_log_std=-1.0, leakage_scale=7.5e-4, constraint_init=0.10,
        dual_lr=1e-3, objective_weight=2.5e-5, lower_budget=0.0025,
        hf_budget=0.00150, promotion_init_logit=-2.0,
        promotion_replan_cost=2.5e-5, lower_hf_order_scale=0.0100,
        upper_period=30, min_upper_duration=8, reference_gain=1.25,
        forecast_blend=0.0, promotion_threshold=0.13,
        promotion_adapt_gain=0.30, promotion_cooldown_steps=10,
        promotion_gate_interval_steps=10, promotion_entropy_coef=5e-5,
        promotion_rate_budget=0.18, promotion_rate_coef=0.10,
        promotion_advantage_threshold=0.03,
    ),
)
BASELINE_CANDIDATES = (
    _baseline_candidate("ppo_lr1e4_std15", 1e-4, -1.5),
    _baseline_candidate("ppo_lr1e4_std10", 1e-4, -1.0),
    _baseline_candidate("ppo_lr3e4_std15", 3e-4, -1.5),
    _baseline_candidate("ppo_lr3e4_std10", 3e-4, -1.0),
    _baseline_candidate("ppo_lr1e4_std05", 1e-4, -0.5),
    _baseline_candidate("ppo_lr3e4_std05", 3e-4, -0.5),
)
OFFPOLICY_CANDIDATES = (
    _offpolicy_candidate("off_lr1e4_w1024_b64", 1e-4, 1024, 64),
    _offpolicy_candidate("off_lr1e4_w2048_b64", 1e-4, 2048, 64),
    _offpolicy_candidate("off_lr3e4_w1024_b64", 3e-4, 1024, 64),
    _offpolicy_candidate("off_lr3e4_w2048_b64", 3e-4, 2048, 64),
    _offpolicy_candidate("off_lr3e4_w2048_b128", 3e-4, 2048, 128),
    _offpolicy_candidate("off_lr1e3_w4096_b64", 1e-3, 4096, 64),
)
ALL_CANDIDATES = FREQUENCY_CANDIDATES + BASELINE_CANDIDATES + OFFPOLICY_CANDIDATES
CANDIDATES_BY_ID = {candidate.candidate_id: candidate for candidate in ALL_CANDIDATES}


def candidate_ids_for_variant(variant_id: str) -> list[str]:
    variant = VARIANTS_BY_ID.get(str(variant_id))
    if variant is None:
        raise ValueError(f"unknown variant_id: {variant_id}")
    return [
        candidate.candidate_id for candidate in ALL_CANDIDATES
        if candidate.family == variant.candidate_family
    ]


def canonical_full_method_parameter_count(
    assets: int,
    *,
    hidden_dim: int = 64,
    plan_basis_dim: int = DEFAULT_PLAN_BASIS_DIM,
) -> int:
    assets = int(assets)
    if assets <= 0 or int(plan_basis_dim) < 2:
        raise ValueError("assets must be positive and plan_basis_dim at least two")
    lower_dim = 9 * assets + 1
    return smdp_parameter_count(SMDPPPOConfig(
        upper_state_dim=6 * assets + 5,
        lower_state_dim=lower_dim,
        upper_action_dim=(int(plan_basis_dim) - 1) * assets,
        lower_action_dim=assets,
        hf_state_dim=lower_dim,
        hf_action_dim=assets,
        promotion_state_dim=promotion_gate_state_dim(assets),
        promotion_advantage_coef=1.0,
        hidden_dim=int(hidden_dim),
    ))


def effective_parameters_for_variant(variant_id: str, candidate_id: str) -> dict[str, Any]:
    variant = VARIANTS_BY_ID.get(str(variant_id))
    candidate = CANDIDATES_BY_ID.get(str(candidate_id))
    if variant is None or candidate is None:
        raise ValueError("unknown variant or candidate")
    if candidate.family != variant.candidate_family:
        raise ValueError(f"candidate {candidate_id} does not apply to {variant_id}")
    params = dict(candidate.parameters)
    params.update({
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
        "volume_impact_bps": DEFAULT_VOLUME_IMPACT_BPS,
        "method_contract": variant.method_contract,
        "policy_mode": variant.policy_mode,
    })
    if variant.trainer_family == "ppo":
        params.update({
            "plan_basis_dim": DEFAULT_PLAN_BASIS_DIM,
            "plan_horizon_s": DEFAULT_PLAN_HORIZON_S,
            "plan_eval_offset_s": DEFAULT_PLAN_EVAL_OFFSET_S,
            "plan_coefficient_scale": DEFAULT_PLAN_COEFFICIENT_SCALE,
            "upper_period": int(params.get("upper_period", 30)),
            "min_upper_duration": int(params.get("min_upper_duration", 5)),
            "use_handcrafted_frequency_prior": False,
        })
    if variant.candidate_family == "baseline_ppo_v7":
        params.update({
            "leakage_scale": 0.0,
            "lower_lf_constraint_coef": 0.0,
            "lower_lf_constraint_target": 0.0,
            "lower_lf_dual_lr": 0.0,
            "lower_lf_objective_weight": 0.0,
            "lower_lf_effect_filter_window": 0,
            "lower_lf_effect_filter_gain": 0.0,
            "lower_lf_raw_recenter_gain": 0.0,
            "lower_lf_raw_recenter_scale": 0.0,
            "plan_smoothness_weight": 0.0,
            "promotion_init_logit": -2.0,
            "promotion_entropy_coef": 0.001,
            "promotion_rate_budget": 1.0,
            "promotion_rate_coef": 0.0,
            "promotion_counterfactual_coef": 0.0,
            "promotion_advantage_learning_rate": float(
                params["learning_rate"]
            ),
            "promotion_advantage_coef": 0.0,
            "promotion_advantage_huber_delta": 0.1,
            "promotion_replan_cost": 0.0,
            "lower_hf_order_scale": 0.0,
            "promotion_credit_mode": "auto",
            "promotion_credit_scale": DEFAULT_TRAINING_REWARD_SCALE,
            "leakage_cost_mode": "auto",
            "lower_lf_budget_rms": 0.0025,
            "hf_lf_budget_rms": 0.00025,
            "include_hf_predictability": None,
            "upper_plan_reference_mode": "none",
            "upper_plan_reference_gain": 1.0,
            "upper_plan_reference_forecast_blend": 0.0,
            "hard_hf_budget_projection": False,
            "promotion_deterministic_threshold": 0.5,
            "promotion_adapt_gain": 0.05,
            "promotion_cooldown_steps": 0,
            "promotion_gate_interval_steps": 1,
            "promotion_deterministic_mode": "actor_probability",
            "promotion_advantage_threshold": 0.0,
            "promotion_advantage_target_threshold": 0.0,
            "upper_residual_action_scale": 1.0,
        })
    elif variant.method_contract == "ablate_promotion_v7":
        params["promotion_replan_cost"] = 0.0
    elif variant.method_contract == "ablate_hf_lower_v7":
        params["lower_hf_order_scale"] = 0.0
        params["hard_hf_budget_projection"] = False
    elif variant.method_contract == "ablate_leakage_v7":
        params.update({
            "leakage_scale": 0.0,
            "lower_lf_constraint_coef": 0.0,
            "lower_lf_constraint_target": 1.0,
            "lower_lf_dual_lr": 0.0,
            "lower_lf_objective_weight": 0.0,
            "hard_hf_budget_projection": False,
        })
    elif variant.method_contract == "ablate_lf_reference_v7":
        params["upper_plan_reference_mode"] = "none"
    elif variant.method_contract == "ablate_upper_residual_v7":
        params["upper_residual_action_scale"] = 0.0
    return params


def _ppo_training_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "hidden_dim", "learning_rate", "epochs", "minibatch_size",
        "init_log_std", "reward_scale", "leakage_scale", "plan_basis_dim",
        "plan_horizon_s", "plan_eval_offset_s", "plan_coefficient_scale",
        "lower_lf_constraint_coef", "lower_lf_constraint_target",
        "lower_lf_dual_lr", "lower_lf_objective_weight",
        "lower_lf_effect_filter_window", "lower_lf_effect_filter_gain",
        "lower_lf_raw_recenter_gain", "lower_lf_raw_recenter_scale",
        "policy_mode", "upper_period", "min_upper_duration",
        "use_handcrafted_frequency_prior", "execution_timeline_contract",
        "method_contract", "capacity_reference_method_contract",
        "volume_impact_bps", "plan_smoothness_weight",
        "promotion_replan_cost", "promotion_init_logit",
        "promotion_entropy_coef", "promotion_rate_budget",
        "promotion_rate_coef", "promotion_counterfactual_coef",
        "promotion_advantage_learning_rate", "promotion_advantage_coef",
        "promotion_advantage_huber_delta",
        "lower_hf_order_scale", "promotion_credit_mode",
        "promotion_credit_scale",
        "leakage_cost_mode", "lower_lf_budget_rms", "hf_lf_budget_rms",
        "include_hf_predictability", "upper_plan_reference_mode",
        "upper_plan_reference_gain", "upper_plan_reference_forecast_blend",
        "hard_hf_budget_projection", "promotion_deterministic_threshold",
        "promotion_adapt_gain", "promotion_cooldown_steps",
        "promotion_gate_interval_steps",
        "promotion_deterministic_mode", "promotion_advantage_threshold",
        "promotion_advantage_target_threshold",
        "upper_residual_action_scale",
    )
    result = {key: params[key] for key in keys}
    result["ppo_epochs"] = result.pop("epochs")
    for key in (
        "upper_learning_rate", "lower_learning_rate", "hf_learning_rate",
        "promotion_learning_rate",
    ):
        if key in params:
            result[key] = params[key]
    return result


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"ndarray\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict\0")
            for key in sorted(value, key=lambda item: repr(item)):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii") + b"\0")
            for item in value:
                update(item)
        elif isinstance(value, np.generic):
            update(value.item())
        else:
            digest.update(type(value).__name__.encode("ascii") + b"\0")
            digest.update(repr(value).encode("utf-8"))

    update(model.state_dict())
    return digest.hexdigest()


def _evaluate_ppo(
    model: Any,
    *,
    params: dict[str, Any],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    steps: int,
    assets: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flags = resolve_method_contract(str(params["method_contract"]))
    mapper = make_plan_mapper(
        assets=int(assets),
        plan_basis_dim=int(params["plan_basis_dim"]),
        plan_horizon_s=float(params["plan_horizon_s"]),
        plan_eval_offset_s=float(params["plan_eval_offset_s"]),
        plan_coefficient_scale=float(params["plan_coefficient_scale"]),
        anchor_first_coefficient=True,
    )
    for scenario in scenarios:
        for seed in seeds:
            if str(params["policy_mode"]) in FLAT_PPO_MODES:
                _, row = joint_flat_rollout(
                    model,
                    seed=int(seed), steps=int(steps), assets=int(assets),
                    scenario=str(scenario), sample=False, leakage_scale=0.0,
                    policy_mode=str(params["policy_mode"]),
                    reward_scale=float(params["reward_scale"]),
                    mark_to_market_timing="post_trade",
                    volume_impact_bps=float(params["volume_impact_bps"]),
                    execution_timeline_contract=str(params["execution_timeline_contract"]),
                    method_contract=str(params["method_contract"]),
                )
            else:
                _, row = smdp_rollout(
                    model,
                    seed=int(seed), steps=int(steps), assets=int(assets),
                    scenario=str(scenario), sample=False, leakage_scale=0.0,
                    plan_mapper=mapper,
                    lower_lf_effect_filter_window=int(params["lower_lf_effect_filter_window"]),
                    lower_lf_effect_filter_gain=float(params["lower_lf_effect_filter_gain"]),
                    lower_lf_raw_recenter_gain=float(params["lower_lf_raw_recenter_gain"]),
                    lower_lf_raw_recenter_scale=float(params["lower_lf_raw_recenter_scale"]),
                    policy_mode=str(params["policy_mode"]),
                    upper_period=int(params["upper_period"]),
                    min_upper_duration=int(params["min_upper_duration"]),
                    reward_scale=float(params["reward_scale"]),
                    mark_to_market_timing="post_trade",
                    volume_impact_bps=float(params["volume_impact_bps"]),
                    execute_plan_curve=bool(flags["execute_plan_curve"]),
                    use_additive_frequency_credit=bool(flags["use_additive_frequency_credit"]),
                    constrain_raw_lower_effect=bool(flags["constrain_raw_lower_effect"]),
                    plan_smoothness_weight=float(params["plan_smoothness_weight"]),
                    learned_promotion_gate=bool(flags["learned_promotion_gate"]),
                    heuristic_promotion_gate=bool(flags["heuristic_promotion_gate"]),
                    promotion_replan_cost=float(params["promotion_replan_cost"]),
                    enable_hf_lower=bool(flags["lower_hf_overlay"]),
                    separate_hf_tactical=bool(flags["separate_hf_tactical"]),
                    lower_hf_order_scale=float(params["lower_hf_order_scale"]),
                    execution_timeline_contract=str(params["execution_timeline_contract"]),
                    method_contract=str(params["method_contract"]),
                    promotion_credit_mode=str(params["promotion_credit_mode"]),
                    promotion_credit_scale=float(
                        params["promotion_credit_scale"]
                    ),
                    leakage_cost_mode=str(params["leakage_cost_mode"]),
                    lower_lf_budget_rms=float(params["lower_lf_budget_rms"]),
                    hf_lf_budget_rms=float(params["hf_lf_budget_rms"]),
                    include_hf_predictability=params["include_hf_predictability"],
                    allow_inactive_mechanism_modules=(
                        str(params["method_contract"]) in V7_METHOD_CONTRACTS
                    ),
                    upper_plan_reference_mode=str(
                        params["upper_plan_reference_mode"]
                    ),
                    upper_plan_reference_gain=float(
                        params["upper_plan_reference_gain"]
                    ),
                    upper_plan_reference_forecast_blend=float(
                        params["upper_plan_reference_forecast_blend"]
                    ),
                    hard_hf_budget_projection=bool(
                        params["hard_hf_budget_projection"]
                    ),
                    promotion_deterministic_threshold=float(
                        params["promotion_deterministic_threshold"]
                    ),
                    promotion_adapt_gain=float(params["promotion_adapt_gain"]),
                    promotion_cooldown_steps=int(
                        params["promotion_cooldown_steps"]
                    ),
                    promotion_gate_interval_steps=int(
                        params["promotion_gate_interval_steps"]
                    ),
                    promotion_deterministic_mode=str(
                        params["promotion_deterministic_mode"]
                    ),
                    promotion_advantage_threshold=float(
                        params["promotion_advantage_threshold"]
                    ),
                    promotion_advantage_target_threshold=float(
                        params["promotion_advantage_target_threshold"]
                    ),
                    upper_residual_action_scale=float(
                        params["upper_residual_action_scale"]
                    ),
                )
            rows.append(row)
    return rows


def _evaluate_offpolicy(
    model: Any,
    *,
    params: dict[str, Any],
    scenarios: Iterable[str],
    seeds: Iterable[int],
    steps: int,
    assets: int,
) -> list[dict[str, Any]]:
    return [
        run_offpolicy_episode(
            model, seed=int(seed), steps=int(steps), assets=int(assets),
            scenario=str(scenario), policy_mode=str(params["policy_mode"]),
            training=False,
            execution_timeline_contract=str(params["execution_timeline_contract"]),
            volume_impact_bps=float(params["volume_impact_bps"]),
        )[0]
        for scenario in scenarios for seed in seeds
    ]


def _hf_intervention_kwargs(
    params: dict[str, Any], *, steps: int, assets: int, scenario: str
) -> dict[str, Any]:
    flags = resolve_method_contract(str(params["method_contract"]))
    return {
        "steps": int(steps), "assets": int(assets), "scenario": str(scenario),
        "leakage_scale": 0.0,
        "plan_mapper": make_plan_mapper(
            assets=int(assets), plan_basis_dim=int(params["plan_basis_dim"]),
            plan_horizon_s=float(params["plan_horizon_s"]),
            plan_eval_offset_s=float(params["plan_eval_offset_s"]),
            plan_coefficient_scale=float(params["plan_coefficient_scale"]),
            anchor_first_coefficient=True,
        ),
        "lower_lf_effect_filter_window": int(params["lower_lf_effect_filter_window"]),
        "lower_lf_effect_filter_gain": float(params["lower_lf_effect_filter_gain"]),
        "lower_lf_raw_recenter_gain": float(params["lower_lf_raw_recenter_gain"]),
        "lower_lf_raw_recenter_scale": float(params["lower_lf_raw_recenter_scale"]),
        "upper_period": int(params["upper_period"]),
        "min_upper_duration": int(params["min_upper_duration"]),
        "policy_mode": str(params["policy_mode"]),
        "reward_scale": float(params["reward_scale"]),
        "mark_to_market_timing": "post_trade",
        "volume_impact_bps": float(params["volume_impact_bps"]),
        "execute_plan_curve": bool(flags["execute_plan_curve"]),
        "use_additive_frequency_credit": bool(flags["use_additive_frequency_credit"]),
        "constrain_raw_lower_effect": bool(flags["constrain_raw_lower_effect"]),
        "plan_smoothness_weight": float(params["plan_smoothness_weight"]),
        "learned_promotion_gate": bool(flags["learned_promotion_gate"]),
        "heuristic_promotion_gate": bool(flags["heuristic_promotion_gate"]),
        "promotion_replan_cost": float(params["promotion_replan_cost"]),
        "enable_hf_lower": bool(flags["lower_hf_overlay"]),
        "separate_hf_tactical": bool(flags["separate_hf_tactical"]),
        "lower_hf_order_scale": float(params["lower_hf_order_scale"]),
        "execution_timeline_contract": str(params["execution_timeline_contract"]),
        "method_contract": str(params["method_contract"]),
        "promotion_credit_mode": str(params["promotion_credit_mode"]),
        "promotion_credit_scale": float(params["promotion_credit_scale"]),
        "leakage_cost_mode": str(params["leakage_cost_mode"]),
        "lower_lf_budget_rms": float(params["lower_lf_budget_rms"]),
        "hf_lf_budget_rms": float(params["hf_lf_budget_rms"]),
        "include_hf_predictability": params["include_hf_predictability"],
        "allow_inactive_mechanism_modules": True,
        "upper_plan_reference_mode": str(params["upper_plan_reference_mode"]),
        "upper_plan_reference_gain": float(params["upper_plan_reference_gain"]),
        "upper_plan_reference_forecast_blend": float(
            params["upper_plan_reference_forecast_blend"]
        ),
        "hard_hf_budget_projection": bool(params["hard_hf_budget_projection"]),
        "promotion_deterministic_threshold": float(
            params["promotion_deterministic_threshold"]
        ),
        "promotion_adapt_gain": float(params["promotion_adapt_gain"]),
        "promotion_cooldown_steps": int(params["promotion_cooldown_steps"]),
        "promotion_gate_interval_steps": int(
            params["promotion_gate_interval_steps"]
        ),
        "promotion_deterministic_mode": str(
            params["promotion_deterministic_mode"]
        ),
        "promotion_advantage_threshold": float(
            params["promotion_advantage_threshold"]
        ),
        "promotion_advantage_target_threshold": float(
            params["promotion_advantage_target_threshold"]
        ),
        "upper_residual_action_scale": float(
            params["upper_residual_action_scale"]
        ),
    }


def _paired_advantage_bias_calibration(
    predictions: Iterable[float],
    targets: Iterable[float],
    *,
    target_threshold: float,
) -> dict[str, Any]:
    """Map an economic advantage margin through a critic's robust bias."""

    predicted = np.asarray(list(predictions), dtype=np.float64).reshape(-1)
    observed = np.asarray(list(targets), dtype=np.float64).reshape(-1)
    if (
        predicted.size == 0
        or predicted.shape != observed.shape
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(observed))
        or not np.isfinite(float(target_threshold))
    ):
        raise ValueError("advantage calibration requires aligned finite samples")
    residual = predicted - observed
    median_bias = float(np.median(residual))
    calibrated_threshold = float(target_threshold) + median_bias
    target_positive = observed >= float(target_threshold)
    uncalibrated_positive = predicted >= float(target_threshold)
    calibrated_positive = predicted >= calibrated_threshold
    return {
        "status": "calibrated",
        "protocol_version": PROMOTION_CALIBRATION_PROTOCOL_VERSION,
        "sample_count": int(predicted.size),
        "target_threshold": float(target_threshold),
        "uncalibrated_decision_threshold": float(target_threshold),
        "calibrated_decision_threshold": calibrated_threshold,
        "median_prediction_bias": median_bias,
        "prediction_target_mae_before": float(np.mean(np.abs(residual))),
        "prediction_target_mae_after": float(np.mean(np.abs(
            predicted - median_bias - observed
        ))),
        "decision_accuracy_before": float(np.mean(
            uncalibrated_positive == target_positive
        )),
        "decision_accuracy_after": float(np.mean(
            calibrated_positive == target_positive
        )),
        "target_positive_rate": float(np.mean(target_positive)),
        "decision_positive_rate_before": float(np.mean(
            uncalibrated_positive
        )),
        "decision_positive_rate_after": float(np.mean(
            calibrated_positive
        )),
    }


def _apply_low_noise_rate_calibration(
    calibration: dict[str, Any],
    *,
    predictions: Iterable[float],
    targets: Iterable[float],
    low_noise_predictions: Iterable[float],
    max_low_noise_rate: float,
) -> dict[str, Any]:
    """Raise the decision threshold to a support-only null-rate quantile."""

    predicted = np.asarray(list(predictions), dtype=np.float64).reshape(-1)
    observed = np.asarray(list(targets), dtype=np.float64).reshape(-1)
    low_noise = np.asarray(
        list(low_noise_predictions), dtype=np.float64
    ).reshape(-1)
    rate = float(max_low_noise_rate)
    if (
        predicted.size == 0
        or predicted.shape != observed.shape
        or low_noise.size == 0
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(low_noise))
        or not 0.0 < rate < 1.0
    ):
        raise ValueError("null-rate calibration requires finite aligned samples")
    allowed = int(np.floor(rate * low_noise.size))
    ordered = np.sort(low_noise)
    pivot = max(0, low_noise.size - allowed - 1)
    null_threshold = float(np.nextafter(ordered[pivot], np.inf))
    bias_threshold = float(calibration["calibrated_decision_threshold"])
    final_threshold = max(bias_threshold, null_threshold)
    target_threshold = float(calibration["target_threshold"])
    target_positive = observed >= target_threshold
    final_positive = predicted >= final_threshold
    result = dict(calibration)
    result.update({
        "bias_calibrated_decision_threshold": bias_threshold,
        "null_rate_decision_threshold": null_threshold,
        "calibrated_decision_threshold": final_threshold,
        "low_noise_rate_ceiling": rate,
        "low_noise_sample_count": int(low_noise.size),
        "low_noise_action_rate_before": float(np.mean(
            low_noise >= bias_threshold
        )),
        "low_noise_action_rate_after": float(np.mean(
            low_noise >= final_threshold
        )),
        "null_rate_constraint_active": bool(
            final_threshold > bias_threshold
        ),
        "decision_accuracy_after": float(np.mean(
            final_positive == target_positive
        )),
        "decision_positive_rate_after": float(np.mean(final_positive)),
    })
    if result["low_noise_action_rate_after"] > rate + 1e-12:
        raise RuntimeError("low-noise calibration failed its empirical rate cap")
    return result


def calibrate_promotion_advantage_threshold(
    model: Any,
    *,
    params: dict[str, Any],
    calibration_seeds: Iterable[int],
    steps: int,
    assets: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calibrate critic intercept on a role-separated support-path set."""

    seeds = validate_unique_seeds(
        calibration_seeds, role="promotion_calibration_seeds"
    )
    if str(params["promotion_deterministic_mode"]) != (
        "counterfactual_advantage"
    ):
        raise ValueError("advantage calibration requires advantage decisions")
    target_threshold = float(params["promotion_advantage_target_threshold"])
    predictions: list[float] = []
    targets: list[float] = []
    low_noise_predictions: list[float] = []
    predictions_by_row: dict[tuple[str, int], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for scenario in SELECTION_SCENARIOS:
        for seed in seeds:
            trajectory, row = smdp_rollout(
                model,
                seed=int(seed),
                sample=False,
                return_trajectory=True,
                **_hf_intervention_kwargs(
                    params,
                    steps=int(steps),
                    assets=int(assets),
                    scenario=str(scenario),
                ),
            )
            if (
                trajectory is None
                or trajectory.promotion is None
                or trajectory.promotion.counterfactual_advantage is None
            ):
                rows.append({
                    "scenario": str(scenario),
                    "seed": int(seed),
                    "transition_count": 0,
                    "prediction_mean": 0.0,
                    "target_mean": 0.0,
                    "prediction_bias_median": 0.0,
                    "uncalibrated_action_rate": float(
                        row["promotion_gate_action_rate"]
                    ),
                    "target_threshold": target_threshold,
                    "calibration_status": "no_eligible_transition",
                })
                continue
            state = np.asarray(trajectory.promotion.state, dtype=np.float32)
            predicted = model.predict_promotion_advantage(state)
            observed = np.asarray(
                trajectory.promotion.counterfactual_advantage,
                dtype=np.float64,
            ).reshape(-1)
            if predicted.shape != observed.shape or predicted.size == 0:
                raise RuntimeError("promotion calibration labels are misaligned")
            if not np.isclose(
                float(np.mean(predicted)),
                float(row["promotion_gate_advantage_mean"]),
                atol=1e-7,
                rtol=1e-6,
            ):
                raise RuntimeError("promotion calibration prediction trace drifted")
            predictions.extend(map(float, predicted))
            targets.extend(map(float, observed))
            predictions_by_row[(str(scenario), int(seed))] = predicted.copy()
            if str(scenario) == "stationary_low_noise":
                low_noise_predictions.extend(map(float, predicted))
            rows.append({
                "scenario": str(scenario),
                "seed": int(seed),
                "transition_count": int(predicted.size),
                "prediction_mean": float(np.mean(predicted)),
                "target_mean": float(np.mean(observed)),
                "prediction_bias_median": float(np.median(
                    predicted - observed
                )),
                "uncalibrated_action_rate": float(
                    row["promotion_gate_action_rate"]
                ),
                "target_threshold": target_threshold,
                "calibration_status": "used",
            })
    if not predictions:
        raise RuntimeError("promotion calibration produced no paired labels")
    calibration = _paired_advantage_bias_calibration(
        predictions, targets, target_threshold=target_threshold
    )
    calibration = _apply_low_noise_rate_calibration(
        calibration,
        predictions=predictions,
        targets=targets,
        low_noise_predictions=low_noise_predictions,
        max_low_noise_rate=min(
            float(params["promotion_rate_budget"]),
            PROMOTION_CALIBRATION_NULL_RATE_CAP,
        ),
    )
    calibrated_threshold = float(
        calibration["calibrated_decision_threshold"]
    )
    for row in rows:
        predicted = predictions_by_row.get(
            (str(row["scenario"]), int(row["seed"]))
        )
        row["calibrated_decision_threshold"] = calibrated_threshold
        row["calibrated_action_rate"] = (
            float(np.mean(predicted >= calibrated_threshold))
            if predicted is not None else 0.0
        )
    calibration.update({
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "scenario_count": len(SELECTION_SCENARIOS),
        "scenarios": list(SELECTION_SCENARIOS),
        "evaluation_role": "promotion_calibration_support_only",
        "ood_period_access_status": "not_loaded",
        "promotion_recovery_access_status": "not_loaded",
    })
    return calibration, rows


def run_hpo_cell(
    *,
    candidate_id: str,
    variant_id: str,
    training_replicate_seed: int,
    train_seeds: list[int],
    promotion_calibration_seeds: list[int] | None = None,
    checkpoint_validation_seeds: list[int],
    tuning_validation_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
    training_budget_plan_sha256: str = "",
    training_budget_decision_sha256: str = "",
    training_budget_selected_iterations: int = 0,
) -> dict[str, Any]:
    variant = VARIANTS_BY_ID.get(str(variant_id))
    candidate = CANDIDATES_BY_ID.get(str(candidate_id))
    if variant is None or variant.variant_id not in HPO_VARIANT_IDS:
        raise ValueError("HPO accepts only independently tuned non-ablation variants")
    if candidate is None or candidate.family != variant.candidate_family:
        raise ValueError(f"candidate {candidate_id} does not apply to {variant_id}")
    rollout_roots = validate_unique_seeds(train_seeds, role="rollout_seed_roots")
    calibration_seeds = validate_unique_seeds(
        (
            DEFAULT_PROMOTION_CALIBRATION_SEEDS
            if promotion_calibration_seeds is None
            else promotion_calibration_seeds
        ),
        role="promotion_calibration_seeds",
    )
    checkpoint_seeds, tuning_seeds = validate_evaluation_seed_roles(
        checkpoint_validation_seeds, tuning_validation_seeds
    )
    seed_roles = {
        "training": set(rollout_roots),
        "promotion_calibration": set(calibration_seeds),
        "checkpoint_validation": set(checkpoint_seeds),
        "tuning_validation": set(tuning_seeds),
    }
    role_names = list(seed_roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1:]:
            overlap = seed_roles[left].intersection(seed_roles[right])
            if overlap:
                raise ValueError(
                    f"seed roles {left} and {right} overlap: {sorted(overlap)}"
                )
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(code_revision),
        expected_source_manifest_sha256=str(expected_source_manifest_sha256),
    )
    budget_plan_hash = str(training_budget_plan_sha256).strip().lower()
    budget_decision_hash = str(training_budget_decision_sha256).strip().lower()
    if bool(budget_plan_hash) != bool(budget_decision_hash):
        raise ValueError("budget plan and decision hashes must be supplied together")
    if budget_plan_hash:
        if budget_plan_hash != budget_plan.plan_sha256():
            raise ValueError("training-budget plan hash drifted")
        if not is_hex_digest(budget_decision_hash, length=64):
            raise ValueError("training-budget decision hash is invalid")
        if int(iterations) < budget_plan.MIN_FINAL_ITERATIONS:
            raise ValueError("final HPO budget is below the registered minimum")
        if int(training_budget_selected_iterations) != int(iterations):
            raise ValueError("HPO iterations differ from the budget decision")
    elif int(training_budget_selected_iterations) != 0:
        raise ValueError("unbound diagnostic cell cannot claim a selected budget")
    optimizer_seed = derive_seed(
        "freq_hrl_v7_support_only_optimizer",
        int(training_replicate_seed),
    )
    params = effective_parameters_for_variant(variant.variant_id, candidate.candidate_id)
    started = time.perf_counter()
    if variant.trainer_family == "ppo":
        model_payload, _, model = train_ppo_actor_critic(
            train_seeds=rollout_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps), assets=int(assets), scenario=TRAINING_SCENARIO,
            training_scenarios=SELECTION_SCENARIOS,
            iterations=int(iterations), seed=int(optimizer_seed),
            resample_training_paths=True, evaluation_role="tuning_validation",
            **_ppo_training_kwargs(params),
        )
    else:
        capacity_target = canonical_full_method_parameter_count(
            int(assets), hidden_dim=int(params["hidden_dim"])
        )
        model_payload, _, model = train_flat_offpolicy_baseline(
            policy_mode=variant.policy_mode,
            train_seeds=rollout_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps), assets=int(assets), scenario=TRAINING_SCENARIO,
            iterations=int(iterations), seed=int(optimizer_seed),
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            replay_capacity=int(params["replay_capacity"]),
            warmup_steps=int(params["warmup_steps"]),
            batch_size=int(params["batch_size"]),
            updates_per_step=int(params["updates_per_step"]),
            resample_training_paths=True, evaluation_role="tuning_validation",
            reward_scale=float(params["reward_scale"]),
            execution_timeline_contract=EXECUTION_TIMELINE_CONTRACT,
            volume_impact_bps=DEFAULT_VOLUME_IMPACT_BPS,
            capacity_target_parameter_count=int(capacity_target),
            capacity_reference_method_contract=CAPACITY_REFERENCE_METHOD_CONTRACT,
            training_scenarios=SELECTION_SCENARIOS,
        )
    if model_payload.get("heldout_test_seeds"):
        raise RuntimeError("HPO loaded held-out seeds")
    checkpoint_hash = _state_dict_sha256(model)
    evaluation_params = dict(params)
    promotion_calibration_rows: list[dict[str, Any]] = []
    promotion_target_threshold = float(
        params.get("promotion_advantage_target_threshold", 0.0)
    )
    promotion_decision_threshold = float(
        params.get("promotion_advantage_threshold", promotion_target_threshold)
    )
    promotion_calibration = {
        "status": "not_applicable",
        "protocol_version": PROMOTION_CALIBRATION_PROTOCOL_VERSION,
        "sample_count": 0,
        "seed_count": 0,
        "seeds": [],
        "scenario_count": 0,
        "scenarios": [],
        "target_threshold": promotion_target_threshold,
        "uncalibrated_decision_threshold": promotion_decision_threshold,
        "bias_calibrated_decision_threshold": promotion_decision_threshold,
        "null_rate_decision_threshold": promotion_decision_threshold,
        "calibrated_decision_threshold": promotion_decision_threshold,
        "median_prediction_bias": 0.0,
        "prediction_target_mae_before": 0.0,
        "prediction_target_mae_after": 0.0,
        "decision_accuracy_before": 0.0,
        "decision_accuracy_after": 0.0,
        "target_positive_rate": 0.0,
        "decision_positive_rate_before": 0.0,
        "decision_positive_rate_after": 0.0,
        "low_noise_rate_ceiling": 0.0,
        "low_noise_sample_count": 0,
        "low_noise_action_rate_before": 0.0,
        "low_noise_action_rate_after": 0.0,
        "null_rate_constraint_active": False,
        "evaluation_role": "not_applicable",
        "ood_period_access_status": "not_loaded",
        "promotion_recovery_access_status": "not_loaded",
    }
    if variant.variant_id == ABLATION_PARENT_VARIANT:
        promotion_calibration, promotion_calibration_rows = (
            calibrate_promotion_advantage_threshold(
                model,
                params=params,
                calibration_seeds=calibration_seeds,
                steps=int(steps),
                assets=int(assets),
            )
        )
        evaluation_params["promotion_advantage_threshold"] = float(
            promotion_calibration["calibrated_decision_threshold"]
        )
    if checkpoint_hash != _state_dict_sha256(model):
        raise RuntimeError("promotion calibration mutated the checkpoint")
    tuning_rows = (
        _evaluate_ppo(
            model, params=evaluation_params, scenarios=SELECTION_SCENARIOS,
            seeds=tuning_seeds, steps=int(steps), assets=int(assets),
        )
        if variant.trainer_family == "ppo" else
        _evaluate_offpolicy(
            model, params=params, scenarios=SELECTION_SCENARIOS,
            seeds=tuning_seeds, steps=int(steps), assets=int(assets),
        )
    )
    if checkpoint_hash != _state_dict_sha256(model):
        raise RuntimeError("deterministic cross-regime evaluation mutated the checkpoint")
    annotated: list[dict[str, Any]] = []
    utilities: list[float] = []
    for row in tuning_rows:
        if str(row["scenario"]) in FORBIDDEN_HPO_SCENARIOS:
            raise RuntimeError("HPO accessed a forbidden confirmatory scenario")
        utility = float(validation_utility(row))
        if not np.isfinite(utility):
            raise RuntimeError("non-finite tuning utility")
        utilities.append(utility)
        annotated.append({
            **row,
            "variant_id": variant.variant_id,
            "candidate_id": candidate.candidate_id,
            "candidate_family": candidate.family,
            "scientific_role": variant.scientific_role,
            "training_replicate_seed": int(training_replicate_seed),
            "optimizer_seed": int(optimizer_seed),
            "evaluation_role": "support_only_tuning_validation",
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "promotion_calibration_protocol_version": (
                PROMOTION_CALIBRATION_PROTOCOL_VERSION
            ),
            "promotion_calibration_status": promotion_calibration["status"],
            "promotion_calibration_sample_count": int(
                promotion_calibration["sample_count"]
            ),
            "promotion_calibrated_bias": float(
                promotion_calibration["median_prediction_bias"]
            ),
            "frozen_checkpoint_sha256": checkpoint_hash,
            "selection_utility": utility,
        })
    expected_coverage = {
        (scenario, int(seed))
        for scenario in SELECTION_SCENARIOS for seed in tuning_seeds
    }
    observed_coverage = {
        (str(row["scenario"]), int(row["seed"])) for row in annotated
    }
    if observed_coverage != expected_coverage:
        raise RuntimeError("support-only tuning coverage is incomplete")

    hf_rows: list[dict[str, Any]] = []
    if variant.variant_id == ABLATION_PARENT_VARIANT:
        for scenario in SELECTION_SCENARIOS:
            scenario_rows = evaluate_hf_lower_intervention(
                model,
                eval_seeds=list(tuning_seeds),
                rollout_kwargs=_hf_intervention_kwargs(
                    evaluation_params,
                    steps=int(steps), assets=int(assets), scenario=scenario
                ),
            )
            hf_rows.extend({
                **row,
                "variant_id": variant.variant_id,
                "candidate_id": candidate.candidate_id,
                "training_replicate_seed": int(training_replicate_seed),
                "evaluation_role": "support_only_mechanism_diagnostic",
                "frozen_checkpoint_sha256": checkpoint_hash,
            } for row in scenario_rows)
    if checkpoint_hash != _state_dict_sha256(model):
        raise RuntimeError("HF intervention mutated the checkpoint")

    parameter_count = count_parameters(model)
    target_count = int(model_payload.get("capacity_target_parameter_count", parameter_count))
    actual_count = int(model_payload.get("capacity_actual_parameter_count", parameter_count))
    if actual_count != parameter_count:
        raise RuntimeError("reported parameter count differs from checkpoint")
    summary = {
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": candidate.candidate_id,
        "candidate_family": candidate.family,
        "candidate_parameters": dict(candidate.parameters),
        "effective_parameters": evaluation_params,
        "uncalibrated_effective_parameters": params,
        "trainer_family": variant.trainer_family,
        "policy_mode": variant.policy_mode,
        "method_contract": variant.method_contract,
        "training_scenario": TRAINING_SCENARIO,
        "training_support_components": list(SELECTION_SCENARIOS),
        "training_episode_protocol": "independent_full_episode_support_batch_v1",
        "training_episode_count_per_root": len(SELECTION_SCENARIOS),
        "selection_scenarios": list(SELECTION_SCENARIOS),
        "forbidden_hpo_scenarios": sorted(FORBIDDEN_HPO_SCENARIOS),
        "ood_period_access_status": "not_loaded",
        "promotion_recovery_access_status": "not_loaded",
        "training_replicate_seed": int(training_replicate_seed),
        "optimizer_seed": int(optimizer_seed),
        "rollout_seed_roots": list(rollout_roots),
        "promotion_calibration_seeds": list(calibration_seeds),
        "promotion_calibration_protocol_version": (
            PROMOTION_CALIBRATION_PROTOCOL_VERSION
        ),
        "promotion_calibration": promotion_calibration,
        "checkpoint_validation_seeds": list(checkpoint_seeds),
        "tuning_validation_seeds": list(tuning_seeds),
        "heldout_test_seeds": [],
        "heldout_test_access_status": "not_loaded",
        "evaluation_role": "support_only_tuning_validation",
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "learned_baseline_implementation_version": LEARNED_BASELINE_IMPLEMENTATION_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "volume_impact_bps": DEFAULT_VOLUME_IMPACT_BPS,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
        "capacity_target_parameter_count": target_count,
        "capacity_actual_parameter_count": actual_count,
        "capacity_ratio": float(actual_count / max(target_count, 1)),
        "parameter_count": int(parameter_count),
        "steps": int(steps), "assets": int(assets), "iterations": int(iterations),
        "training_budget_plan_version": budget_plan.BUDGET_PLAN_VERSION,
        "training_budget_plan_sha256": budget_plan_hash,
        "training_budget_decision_sha256": budget_decision_hash,
        "training_budget_selected_iterations": int(
            training_budget_selected_iterations
        ),
        "training_budget_binding_status": (
            "source_bound_decision" if budget_plan_hash else "diagnostic_unbound"
        ),
        "selection_utility_mean": float(np.mean(utilities)),
        "selection_utility_min": float(np.min(utilities)),
        "selection_utility_std": float(np.std(utilities, ddof=1)) if len(utilities) > 1 else 0.0,
        "initial_checkpoint_validation_score": float(model_payload.get("initial_validation_score", 0.0)),
        "best_checkpoint_inner_validation_score": float(model_payload.get("best_score", 0.0)),
        "validation_learning_gain": float(model_payload.get("validation_learning_gain", 0.0)),
        "selected_checkpoint_iteration": int(model_payload.get("selected_checkpoint_iteration", -1)),
        "environment_steps_train": int(model_payload.get("environment_steps_train", 0)),
        "environment_steps_validation": int(
            model_payload.get("environment_steps_validation", 0)
        ),
        "environment_steps_promotion_calibration": int(
            len(promotion_calibration_rows) * int(steps)
        ),
        "frozen_checkpoint_sha256": checkpoint_hash,
        "hf_intervention_pair_count": len(hf_rows),
        "hf_action_sensitivity_mean": float(np.mean([
            float(row["lower_hf_action_sensitivity"]) for row in hf_rows
        ])) if hf_rows else 0.0,
        "elapsed_sec": float(time.perf_counter() - started),
        "source_identity_status": source_identity["source_identity_status"],
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "cell_status": "valid",
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_payload.get("config", {}),
        "variant_id": variant.variant_id,
        "candidate_id": candidate.candidate_id,
        "effective_parameters": evaluation_params,
        "promotion_calibration": promotion_calibration,
        "training_replicate_seed": int(training_replicate_seed),
        "frozen_checkpoint_sha256": checkpoint_hash,
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "training_budget_plan_sha256": budget_plan_hash,
        "training_budget_decision_sha256": budget_decision_hash,
        "training_budget_selected_iterations": int(
            training_budget_selected_iterations
        ),
        "heldout_test_seeds": [],
    }
    return {
        "tuning_rows": annotated,
        "promotion_calibration_rows": promotion_calibration_rows,
        "hf_intervention_rows": hf_rows,
        "cell_summary": summary,
        "checkpoint": checkpoint,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_hpo_cell(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tuning_rows.csv", payload["tuning_rows"])
    _write_csv(
        output_dir / "promotion_calibration_rows.csv",
        payload["promotion_calibration_rows"],
    )
    _write_csv(output_dir / "hf_intervention_rows.csv", payload["hf_intervention_rows"])
    (output_dir / "cell_summary.json").write_text(
        json.dumps(payload["cell_summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")
    summary = payload["cell_summary"]
    (output_dir / "report.md").write_text(
        "\n".join([
            "# Freq-HRL v7.3.2 Support-Only HPO Cell", "",
            f"- variant: `{summary['variant_id']}`",
            f"- candidate: `{summary['candidate_id']}`",
            f"- checkpoint: `{summary['frozen_checkpoint_sha256']}`",
            f"- support utility: `{summary['selection_utility_mean']:.8f}`",
            "- promotion calibration: "
            f"`{summary['promotion_calibration']['status']}`",
            "- promotion decision threshold: "
            f"`{summary['promotion_calibration']['calibrated_decision_threshold']:.8f}`",
            "- OOD access: `not_loaded`",
        ]) + "\n",
        encoding="utf-8",
    )


def _bootstrap_mean_ci(values: Iterable[float], *, seed: int, draws: int = 10_000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    means = array[rng.integers(0, array.size, size=(int(draws), array.size))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def frozen_config_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _promotion_replicate_diagnostics(
    candidate_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(candidate_rows)
    diagnostics = []
    for replicate in sorted({
        int(float(row["training_replicate_seed"])) for row in rows
    }):
        replicate_rows = [
            row for row in rows
            if int(float(row["training_replicate_seed"])) == replicate
        ]
        low_rows = [
            row for row in replicate_rows
            if str(row.get("scenario", "")) == "stationary_low_noise"
        ]
        stress_rows = [
            row for row in replicate_rows
            if str(row.get("scenario", "")) in {
                "localized_burst", "persistent_shift"
            }
        ]

        def mean(field: str, subset: list[dict[str, Any]]) -> float:
            return float(np.mean([
                float(row.get(field, 0.0) or 0.0) for row in subset
            ] or [0.0]))

        executed = float(np.sum([
            float(row.get("promotion_count", 0.0) or 0.0)
            for row in replicate_rows
        ]))
        replans = float(np.sum([
            float(row.get("promotion_replan_count", 0.0) or 0.0)
            for row in replicate_rows
        ]))
        low_rate = mean("promotion_gate_action_rate", low_rows)
        stress_rate = mean("promotion_gate_action_rate", stress_rows)
        probability_lift = (
            mean("promotion_gate_probability_mean", stress_rows)
            - mean("promotion_gate_probability_mean", low_rows)
        )
        advantage_lift = (
            mean("promotion_gate_advantage_mean", stress_rows)
            - mean("promotion_gate_advantage_mean", low_rows)
        )
        modes = {
            str(row.get(
                "promotion_deterministic_mode", "actor_probability"
            ))
            for row in replicate_rows
        }
        decision_mode = next(iter(modes)) if len(modes) == 1 else "mixed"
        decision_score_lift = (
            advantage_lift
            if decision_mode == "counterfactual_advantage"
            else probability_lift
        )
        minimum_score_lift = (
            MIN_STRESS_LOW_ADVANTAGE_LIFT
            if decision_mode == "counterfactual_advantage"
            else MIN_STRESS_LOW_PROBABILITY_LIFT
        )
        advantage_mode = decision_mode == "counterfactual_advantage"
        advantage_head_enabled = bool(replicate_rows) and all(
            float(row.get(
                "promotion_gate_advantage_head_enabled", 0.0
            ) or 0.0) == 1.0
            for row in replicate_rows
        )
        advantage_alignment_valid = bool(replicate_rows) and all(
            float(row.get(
                "promotion_gate_advantage_alignment_valid", 0.0
            ) or 0.0) == 1.0
            for row in replicate_rows
        )
        advantage_decision_accuracy = mean(
            "promotion_gate_advantage_sign_accuracy", replicate_rows
        )
        advantage_target_correlation = mean(
            "promotion_gate_advantage_target_correlation", replicate_rows
        )
        checks = {
            "complete_scenario_support": bool(low_rows) and bool(stress_rows),
            "promotion_executed": executed > 0.0 and replans > 0.0,
            "low_noise_rate_within_limit": (
                low_rate <= MAX_LOW_NOISE_PROMOTION_RATE
            ),
            "stress_rate_above_minimum": (
                stress_rate >= MIN_STRESS_PROMOTION_RATE
            ),
            "stress_rate_within_limit": (
                stress_rate <= MAX_STRESS_PROMOTION_RATE
            ),
            "stress_low_rate_lift_supported": (
                stress_rate - low_rate >= MIN_STRESS_LOW_RATE_LIFT
            ),
            "stress_low_decision_score_lift_supported": (
                decision_score_lift >= minimum_score_lift
            ),
            "decision_mode_consistent": decision_mode != "mixed",
            "advantage_head_enabled": (
                not advantage_mode or advantage_head_enabled
            ),
            "advantage_targets_aligned": (
                not advantage_mode or advantage_alignment_valid
            ),
            "advantage_decision_accuracy_supported": (
                not advantage_mode
                or advantage_decision_accuracy
                >= MIN_ADVANTAGE_DECISION_ACCURACY
            ),
        }
        diagnostics.append({
            "variant_id": str(replicate_rows[0].get("variant_id", "")),
            "candidate_id": str(replicate_rows[0].get("candidate_id", "")),
            "training_replicate_seed": replicate,
            "promotion_deterministic_mode": decision_mode,
            "promotion_execution_count": executed,
            "promotion_replan_count": replans,
            "low_noise_promotion_rate": low_rate,
            "stress_promotion_rate": stress_rate,
            "stress_low_promotion_rate_lift": stress_rate - low_rate,
            "stress_low_promotion_probability_lift": probability_lift,
            "stress_low_promotion_advantage_lift": advantage_lift,
            "stress_low_promotion_decision_score_lift": decision_score_lift,
            "minimum_decision_score_lift": minimum_score_lift,
            "advantage_decision_accuracy": advantage_decision_accuracy,
            "advantage_target_correlation": advantage_target_correlation,
            **checks,
            "selective": all(checks.values()),
            "failure_codes": ";".join(
                key for key, passed in checks.items() if not passed
            ),
        })
    return diagnostics


def _mechanism_activity_summary(
    candidate_rows: Iterable[dict[str, Any]],
    hf_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Require executed mechanisms, not merely policy-decision records."""

    candidate_rows = list(candidate_rows)
    hf_rows = list(hf_rows)
    replicate_ids = sorted({
        int(float(row["training_replicate_seed"])) for row in candidate_rows
    })
    if not replicate_ids:
        return {
            "status": "ineligible",
            "promotion_execution_count": 0.0,
            "promotion_replan_count": 0.0,
            "promotion_active_replicate_fraction": 0.0,
            "promotion_selective_replicate_fraction": 0.0,
            "low_noise_promotion_rate_mean": 0.0,
            "stress_promotion_rate_mean": 0.0,
            "stress_low_promotion_rate_lift_mean": 0.0,
            "stress_low_promotion_probability_lift_mean": 0.0,
            "stress_low_promotion_advantage_lift_mean": 0.0,
            "stress_low_promotion_decision_score_lift_mean": 0.0,
            "advantage_decision_accuracy_mean": 0.0,
            "advantage_target_correlation_mean": 0.0,
            "hf_action_sensitivity_mean": 0.0,
            "hf_active_replicate_fraction": 0.0,
            "reference_active_replicate_fraction": 0.0,
            "upper_residual_active_replicate_fraction": 0.0,
            "hard_budget_compliant_replicate_fraction": 0.0,
        }

    promotion_active = []
    promotion_selective = []
    low_noise_rates = []
    stress_rates = []
    stress_low_rate_lifts = []
    stress_low_probability_lifts = []
    stress_low_advantage_lifts = []
    stress_low_decision_score_lifts = []
    advantage_decision_accuracies = []
    advantage_target_correlations = []
    hf_active = []
    reference_active = []
    residual_active = []
    hard_budget_compliant = []
    promotion_diagnostics = {
        int(row["training_replicate_seed"]): row
        for row in _promotion_replicate_diagnostics(candidate_rows)
    }
    for replicate in replicate_ids:
        replicate_rows = [
            row for row in candidate_rows
            if int(float(row["training_replicate_seed"])) == replicate
        ]
        promotion_diagnostic = promotion_diagnostics[replicate]
        promotion_active.append(bool(
            promotion_diagnostic["promotion_executed"]
        ))
        promotion_selective.append(bool(promotion_diagnostic["selective"]))
        low_noise_rates.append(float(
            promotion_diagnostic["low_noise_promotion_rate"]
        ))
        stress_rates.append(float(
            promotion_diagnostic["stress_promotion_rate"]
        ))
        stress_low_rate_lifts.append(float(
            promotion_diagnostic["stress_low_promotion_rate_lift"]
        ))
        stress_low_probability_lifts.append(float(
            promotion_diagnostic["stress_low_promotion_probability_lift"]
        ))
        stress_low_advantage_lifts.append(float(
            promotion_diagnostic["stress_low_promotion_advantage_lift"]
        ))
        stress_low_decision_score_lifts.append(float(
            promotion_diagnostic[
                "stress_low_promotion_decision_score_lift"
            ]
        ))
        advantage_decision_accuracies.append(float(
            promotion_diagnostic["advantage_decision_accuracy"]
        ))
        advantage_target_correlations.append(float(
            promotion_diagnostic["advantage_target_correlation"]
        ))

        replicate_hf = [
            row for row in hf_rows
            if int(float(row["training_replicate_seed"])) == replicate
        ]
        sensitivity = float(np.mean([
            float(row.get("lower_hf_action_sensitivity", 0.0) or 0.0)
            for row in replicate_hf
        ] or [0.0]))
        paired_paths_valid = bool(replicate_hf) and all(
            str(row.get("paired_exogenous_path_identity", "")).lower()
            in {"1", "1.0", "true"}
            for row in replicate_hf
        )
        hf_active.append(
            paired_paths_valid
            and sensitivity > MIN_HF_ACTION_SENSITIVITY
            and max(
                float(row.get("hf_overlay_rms_after_projection_max", 0.0) or 0.0)
                for row in replicate_rows
            ) > MIN_PLAN_ACTION_EFFECT
        )
        reference_active.append(
            float(np.mean([
                float(row.get("upper_plan_reference_coeff_abs", 0.0) or 0.0)
                for row in replicate_rows
            ])) > MIN_PLAN_ACTION_EFFECT
        )
        residual_active.append(
            float(np.mean([
                float(row.get("upper_plan_residual_coeff_abs", 0.0) or 0.0)
                for row in replicate_rows
            ])) > MIN_PLAN_ACTION_EFFECT
        )
        hard_budget_compliant.append(
            all(
                str(row.get("hard_hf_budget_projection", "")).lower()
                in {"1", "1.0", "true"}
                and float(row.get("hf_leakage_budget_ratio_max", float("inf")))
                <= MAX_HF_BUDGET_RATIO
                for row in replicate_rows
            )
        )

    promotion_execution_count = float(np.sum([
        float(row.get("promotion_count", 0.0) or 0.0)
        for row in candidate_rows
    ]))
    promotion_replan_count = float(np.sum([
        float(row.get("promotion_replan_count", 0.0) or 0.0)
        for row in candidate_rows
    ]))
    hf_action_sensitivity = float(np.mean([
        float(row.get("lower_hf_action_sensitivity", 0.0) or 0.0)
        for row in hf_rows
    ] or [0.0]))
    promotion_fraction = float(np.mean(promotion_active))
    promotion_selective_fraction = float(np.mean(promotion_selective))
    hf_fraction = float(np.mean(hf_active))
    reference_fraction = float(np.mean(reference_active))
    residual_fraction = float(np.mean(residual_active))
    budget_fraction = float(np.mean(hard_budget_compliant))
    eligible = (
        promotion_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
        and promotion_selective_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
        and hf_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
        and reference_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
        and residual_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
        and budget_fraction >= MIN_MECHANISM_REPLICATE_FRACTION
    )
    return {
        "status": "eligible" if eligible else "ineligible",
        "promotion_execution_count": promotion_execution_count,
        "promotion_replan_count": promotion_replan_count,
        "promotion_active_replicate_fraction": promotion_fraction,
        "promotion_selective_replicate_fraction": (
            promotion_selective_fraction
        ),
        "low_noise_promotion_rate_mean": float(np.mean(low_noise_rates)),
        "stress_promotion_rate_mean": float(np.mean(stress_rates)),
        "stress_low_promotion_rate_lift_mean": float(np.mean(
            stress_low_rate_lifts
        )),
        "stress_low_promotion_probability_lift_mean": float(np.mean(
            stress_low_probability_lifts
        )),
        "stress_low_promotion_advantage_lift_mean": float(np.mean(
            stress_low_advantage_lifts
        )),
        "stress_low_promotion_decision_score_lift_mean": float(np.mean(
            stress_low_decision_score_lifts
        )),
        "advantage_decision_accuracy_mean": float(np.mean(
            advantage_decision_accuracies
        )),
        "advantage_target_correlation_mean": float(np.mean(
            advantage_target_correlations
        )),
        "hf_action_sensitivity_mean": hf_action_sensitivity,
        "hf_active_replicate_fraction": hf_fraction,
        "reference_active_replicate_fraction": reference_fraction,
        "upper_residual_active_replicate_fraction": residual_fraction,
        "hard_budget_compliant_replicate_fraction": budget_fraction,
    }


def _load_validated_hpo_cells(
    input_dirs: Iterable[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    set[tuple[str, str, int]],
]:
    """Load cells under the support-only protocol and reject mixed evidence."""

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    hf_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, int]] = set()
    for directory in input_dirs:
        base = Path(directory)
        summary = json.loads(
            (base / "cell_summary.json").read_text(encoding="utf-8")
        )
        key = (
            str(summary["variant_id"]), str(summary["candidate_id"]),
            int(summary["training_replicate_seed"]),
        )
        if key in seen or summary.get("cell_status") != "valid":
            raise ValueError(f"duplicate or invalid HPO cell: {key}")
        if summary.get("tuning_protocol_version") != FULL_METHOD_TUNING_PROTOCOL_VERSION:
            raise ValueError(f"HPO protocol mismatch: {key}")
        if summary.get("ood_period_access_status") != "not_loaded":
            raise ValueError(f"HPO cell accessed OOD: {key}")
        if set(summary.get("selection_scenarios", [])) != set(SELECTION_SCENARIOS):
            raise ValueError(f"support scenario contract mismatch: {key}")
        if summary.get("training_episode_protocol") != (
            "independent_full_episode_support_batch_v1"
        ):
            raise ValueError(f"stitched support training is forbidden: {key}")
        calibration = dict(summary.get("promotion_calibration", {}))
        calibration_rows = _read_csv(
            base / "promotion_calibration_rows.csv"
        )
        if str(summary["variant_id"]) == ABLATION_PARENT_VARIANT:
            if (
                summary.get("promotion_calibration_protocol_version")
                != PROMOTION_CALIBRATION_PROTOCOL_VERSION
                or calibration.get("status") != "calibrated"
            ):
                raise ValueError(f"promotion calibration mismatch: {key}")
            expected_calibration_coverage = {
                (scenario, int(seed))
                for scenario in SELECTION_SCENARIOS
                for seed in summary["promotion_calibration_seeds"]
            }
            observed_calibration_coverage = {
                (str(row["scenario"]), int(float(row["seed"])))
                for row in calibration_rows
            }
            if observed_calibration_coverage != expected_calibration_coverage:
                raise ValueError(
                    f"incomplete promotion calibration coverage: {key}"
                )
            if int(calibration.get("sample_count", 0)) <= 0:
                raise ValueError(f"empty promotion calibration: {key}")
        elif calibration_rows or calibration.get("status") != "not_applicable":
            raise ValueError(f"unexpected baseline promotion calibration: {key}")
        cell_rows = _read_csv(base / "tuning_rows.csv")
        expected_calibration_status = (
            "calibrated"
            if str(summary["variant_id"]) == ABLATION_PARENT_VARIANT
            else "not_applicable"
        )
        if any(
            row.get("promotion_calibration_protocol_version")
            != PROMOTION_CALIBRATION_PROTOCOL_VERSION
            or row.get("promotion_calibration_status")
            != expected_calibration_status
            for row in cell_rows
        ):
            raise ValueError(f"tuning calibration provenance mismatch: {key}")
        expected_coverage = {
            (scenario, int(seed))
            for scenario in SELECTION_SCENARIOS
            for seed in summary["tuning_validation_seeds"]
        }
        observed = {
            (str(row["scenario"]), int(float(row["seed"])))
            for row in cell_rows
        }
        if observed != expected_coverage:
            raise ValueError(f"incomplete support coverage: {key}")
        hashes = {str(row["frozen_checkpoint_sha256"]) for row in cell_rows}
        if hashes != {str(summary["frozen_checkpoint_sha256"])}:
            raise ValueError(f"mixed checkpoints within HPO cell: {key}")
        seen.add(key)
        summaries.append(summary)
        rows.extend(cell_rows)
        hf_rows.extend(_read_csv(base / "hf_intervention_rows.csv"))
    return summaries, rows, hf_rows, seen


def _validate_common_hpo_fields(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        raise ValueError("HPO merge requires at least one cell")
    for field in (
        "rollout_seed_roots", "checkpoint_validation_seeds",
        "promotion_calibration_seeds", "tuning_validation_seeds",
        "steps", "assets", "iterations",
        "training_episode_protocol",
        "training_budget_plan_sha256", "training_budget_decision_sha256",
        "training_budget_selected_iterations", "training_budget_binding_status",
    ):
        values = {json.dumps(summary[field], sort_keys=True) for summary in summaries}
        if len(values) != 1:
            raise ValueError(f"HPO matrix mixes {field}")


def checkpoint_boundary_start(iterations: int) -> int:
    count = int(iterations)
    if count < 1:
        raise ValueError("iterations must be positive")
    window = max(2, int(np.ceil(
        CHECKPOINT_BOUNDARY_WINDOW_FRACTION * count
    )))
    return max(0, count - window)


def _candidate_leaderboard_row(
    *,
    variant_id: str,
    candidate_id: str,
    replicates: list[int],
    summaries: list[dict[str, Any]],
    rows: list[dict[str, str]],
    hf_rows: list[dict[str, str]],
) -> dict[str, Any]:
    replicate_scores = []
    for replicate in replicates:
        values = [
            float(row["selection_utility"])
            for row in rows
            if row["variant_id"] == variant_id
            and row["candidate_id"] == candidate_id
            and int(float(row["training_replicate_seed"])) == replicate
        ]
        if not values:
            raise ValueError(
                "missing candidate support rows: "
                f"{(variant_id, candidate_id, replicate)}"
            )
        replicate_scores.append(float(np.mean(values)))
    ci_low, ci_high = _bootstrap_mean_ci(
        replicate_scores,
        seed=derive_seed("freq_hrl_v7_hpo_bootstrap", variant_id, candidate_id),
    )
    matching = [
        summary for summary in summaries
        if summary["variant_id"] == variant_id
        and summary["candidate_id"] == candidate_id
    ]
    trained_fraction = float(np.mean([
        int(summary["selected_checkpoint_iteration"]) >= 0
        for summary in matching
    ]))
    gain_mean = float(np.mean([
        float(summary["validation_learning_gain"]) for summary in matching
    ]))
    boundary_fraction = float(np.mean([
        int(summary["selected_checkpoint_iteration"]) >= checkpoint_boundary_start(
            int(summary["iterations"])
        )
        for summary in matching
    ]))
    calibrations = [
        dict(summary.get("promotion_calibration", {}))
        for summary in matching
        if summary.get("promotion_calibration", {}).get("status")
        == "calibrated"
    ]

    def calibration_mean(field: str) -> float:
        return float(np.mean([
            float(calibration[field]) for calibration in calibrations
        ] or [0.0]))

    mechanism_evidence = {
        "status": "not_applicable",
        "promotion_execution_count": 0.0,
        "promotion_replan_count": 0.0,
        "promotion_active_replicate_fraction": 0.0,
        "promotion_selective_replicate_fraction": 0.0,
        "low_noise_promotion_rate_mean": 0.0,
        "stress_promotion_rate_mean": 0.0,
        "stress_low_promotion_rate_lift_mean": 0.0,
        "stress_low_promotion_probability_lift_mean": 0.0,
        "stress_low_promotion_advantage_lift_mean": 0.0,
        "stress_low_promotion_decision_score_lift_mean": 0.0,
        "advantage_decision_accuracy_mean": 0.0,
        "advantage_target_correlation_mean": 0.0,
        "hf_action_sensitivity_mean": 0.0,
        "hf_active_replicate_fraction": 0.0,
        "reference_active_replicate_fraction": 0.0,
        "upper_residual_active_replicate_fraction": 0.0,
        "hard_budget_compliant_replicate_fraction": 0.0,
    }
    if variant_id == ABLATION_PARENT_VARIANT:
        candidate_rows = [
            row for row in rows
            if row["variant_id"] == variant_id
            and row["candidate_id"] == candidate_id
        ]
        candidate_hf_rows = [
            row for row in hf_rows
            if row["variant_id"] == variant_id
            and row["candidate_id"] == candidate_id
        ]
        mechanism_evidence = _mechanism_activity_summary(
            candidate_rows, candidate_hf_rows
        )
    return {
        "variant_id": variant_id,
        "candidate_id": candidate_id,
        "independent_training_replicates": len(replicates),
        "support_scenario_count": len(SELECTION_SCENARIOS),
        "tuning_utility_mean": float(np.mean(replicate_scores)),
        "tuning_utility_std_across_replicates": (
            float(np.std(replicate_scores, ddof=1))
            if len(replicate_scores) > 1 else 0.0
        ),
        "tuning_utility_ci95_low": ci_low,
        "tuning_utility_ci95_high": ci_high,
        "robust_selection_score": ci_low,
        "trained_checkpoint_fraction": trained_fraction,
        "validation_learning_gain_mean": gain_mean,
        "checkpoint_boundary_replicate_fraction": boundary_fraction,
        "checkpoint_boundary_start_iteration": checkpoint_boundary_start(
            int(matching[0]["iterations"])
        ),
        "training_budget_status": (
            "sufficient"
            if boundary_fraction <= MAX_CHECKPOINT_BOUNDARY_REPLICATE_FRACTION
            else "boundary_limited"
        ),
        "promotion_calibrated_replicate_fraction": float(
            len(calibrations) / max(len(matching), 1)
        ),
        "promotion_calibrated_threshold_mean": calibration_mean(
            "calibrated_decision_threshold"
        ),
        "promotion_calibrated_bias_mean": calibration_mean(
            "median_prediction_bias"
        ),
        "promotion_calibration_accuracy_before_mean": calibration_mean(
            "decision_accuracy_before"
        ),
        "promotion_calibration_accuracy_after_mean": calibration_mean(
            "decision_accuracy_after"
        ),
        "learning_gate_status": (
            "eligible"
            if trained_fraction >= 0.8 and gain_mean > 0.0 else "ineligible"
        ),
        "mechanism_activity_status": mechanism_evidence["status"],
        "promotion_execution_count": mechanism_evidence[
            "promotion_execution_count"
        ],
        "promotion_replan_count": mechanism_evidence[
            "promotion_replan_count"
        ],
        "promotion_active_replicate_fraction": mechanism_evidence[
            "promotion_active_replicate_fraction"
        ],
        "promotion_selective_replicate_fraction": mechanism_evidence[
            "promotion_selective_replicate_fraction"
        ],
        "low_noise_promotion_rate_mean": mechanism_evidence[
            "low_noise_promotion_rate_mean"
        ],
        "stress_promotion_rate_mean": mechanism_evidence[
            "stress_promotion_rate_mean"
        ],
        "stress_low_promotion_rate_lift_mean": mechanism_evidence[
            "stress_low_promotion_rate_lift_mean"
        ],
        "stress_low_promotion_probability_lift_mean": mechanism_evidence[
            "stress_low_promotion_probability_lift_mean"
        ],
        "stress_low_promotion_advantage_lift_mean": mechanism_evidence[
            "stress_low_promotion_advantage_lift_mean"
        ],
        "stress_low_promotion_decision_score_lift_mean": mechanism_evidence[
            "stress_low_promotion_decision_score_lift_mean"
        ],
        "advantage_decision_accuracy_mean": mechanism_evidence[
            "advantage_decision_accuracy_mean"
        ],
        "advantage_target_correlation_mean": mechanism_evidence[
            "advantage_target_correlation_mean"
        ],
        "hf_action_sensitivity_mean": mechanism_evidence[
            "hf_action_sensitivity_mean"
        ],
        "hf_active_replicate_fraction": mechanism_evidence[
            "hf_active_replicate_fraction"
        ],
        "reference_active_replicate_fraction": mechanism_evidence[
            "reference_active_replicate_fraction"
        ],
        "upper_residual_active_replicate_fraction": mechanism_evidence[
            "upper_residual_active_replicate_fraction"
        ],
        "hard_budget_compliant_replicate_fraction": mechanism_evidence[
            "hard_budget_compliant_replicate_fraction"
        ],
    }


def merge_hpo_cells(
    input_dirs: list[Path],
    *,
    expected_variant_ids: list[str],
    expected_candidate_ids: list[str],
    expected_replicate_seeds: list[int],
    top_k: int = 3,
    stage: str = "pilot",
) -> dict[str, Any]:
    summaries, rows, hf_rows, seen = _load_validated_hpo_cells(input_dirs)

    variants = list(expected_variant_ids)
    candidates = list(expected_candidate_ids)
    replicates = list(map(int, expected_replicate_seeds))
    if set(variants) != set(HPO_VARIANT_IDS):
        raise ValueError("HPO merge must cover all independently tuned variants")
    expected = {
        (variant_id, candidate_id, replicate)
        for variant_id in variants
        for candidate_id in candidates
        if candidate_id in candidate_ids_for_variant(variant_id)
        for replicate in replicates
    }
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise ValueError(f"incomplete HPO matrix: {(missing + unexpected)[:6]}")
    _validate_common_hpo_fields(summaries)

    leaderboard: list[dict[str, Any]] = []
    ranked_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_id in variants:
        ranked = [
            _candidate_leaderboard_row(
                variant_id=variant_id,
                candidate_id=candidate_id,
                replicates=replicates,
                summaries=summaries,
                rows=rows,
                hf_rows=hf_rows,
            )
            for candidate_id in candidate_ids_for_variant(variant_id)
        ]
        ranked.sort(key=lambda row: (
            0 if row["learning_gate_status"] == "eligible" else 1,
            0 if row["mechanism_activity_status"] in {"eligible", "not_applicable"} else 1,
            -float(row["robust_selection_score"]),
            -float(row["tuning_utility_mean"]),
        ))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        ranked_by_variant[variant_id] = ranked
        leaderboard.extend(ranked)

    selected: dict[str, dict[str, Any]] = {}
    top_candidates: dict[str, list[str]] = {}
    for variant_id in variants:
        ranked = ranked_by_variant[variant_id]
        eligible = [
            row for row in ranked
            if row["learning_gate_status"] == "eligible"
            and row["mechanism_activity_status"] in {"eligible", "not_applicable"}
        ]
        pool = eligible or ranked
        winners = pool[:max(1, min(int(top_k), len(pool)))]
        top_candidates[variant_id] = [str(row["candidate_id"]) for row in winners]
        winner = winners[0]
        candidate_id = str(winner["candidate_id"])
        selected[variant_id] = {
            "candidate_id": candidate_id,
            "candidate_parameters": dict(CANDIDATES_BY_ID[candidate_id].parameters),
            "effective_parameters": effective_parameters_for_variant(variant_id, candidate_id),
            "selection_source_variant": variant_id,
            "selection_rule": "support_only_training_replicate_lcb",
            "robust_selection_score": float(winner["robust_selection_score"]),
            "learning_gate_status": str(winner["learning_gate_status"]),
            "mechanism_activity_status": str(winner["mechanism_activity_status"]),
            "training_budget_status": str(winner["training_budget_status"]),
            "checkpoint_boundary_replicate_fraction": float(
                winner["checkpoint_boundary_replicate_fraction"]
            ),
        }
    parent_candidate = str(selected[ABLATION_PARENT_VARIANT]["candidate_id"])
    for variant in VARIANTS:
        if not variant.inherits_full_selection:
            continue
        selected[variant.variant_id] = {
            "candidate_id": parent_candidate,
            "candidate_parameters": dict(CANDIDATES_BY_ID[parent_candidate].parameters),
            "effective_parameters": effective_parameters_for_variant(
                variant.variant_id, parent_candidate
            ),
            "selection_source_variant": ABLATION_PARENT_VARIANT,
            "selection_rule": "inherit_full_candidate_disable_one_mechanism",
            "learning_gate_status": selected[ABLATION_PARENT_VARIANT]["learning_gate_status"],
            "mechanism_activity_status": "not_applicable",
            "training_budget_status": selected[ABLATION_PARENT_VARIANT][
                "training_budget_status"
            ],
            "checkpoint_boundary_replicate_fraction": selected[
                ABLATION_PARENT_VARIANT
            ]["checkpoint_boundary_replicate_fraction"],
        }

    revisions = {str(summary["code_revision"]).lower() for summary in summaries}
    manifests = {str(summary["source_manifest_sha256"]).lower() for summary in summaries}
    statuses = {str(summary["source_identity_status"]) for summary in summaries}
    source_verified = (
        len(revisions) == 1 and len(manifests) == 1 and statuses == {"verified"}
        and is_hex_digest(next(iter(revisions)), length=40)
        and is_hex_digest(next(iter(manifests)), length=64)
    )
    first = summaries[0]
    budget_binding_verified = (
        first.get("training_budget_binding_status") == "source_bound_decision"
        and first.get("training_budget_plan_sha256") == budget_plan.plan_sha256()
        and is_hex_digest(
            first.get("training_budget_decision_sha256"), length=64
        )
        and int(first.get("training_budget_selected_iterations", 0))
        == int(first["iterations"])
        and int(first["iterations"]) >= budget_plan.MIN_FINAL_ITERATIONS
    )
    final_design_complete = (
        set(variants) == set(HPO_VARIANT_IDS)
        and set(DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS).issubset(replicates)
        and not set(DEFAULT_PILOT_OPTIMIZER_SEEDS).intersection(replicates)
        and source_verified
        and budget_binding_verified
    )
    all_learning_eligible = all(
        entry["learning_gate_status"] == "eligible"
        for variant_id, entry in selected.items()
        if variant_id in HPO_VARIANT_IDS
    )
    full_mechanism_eligible = (
        selected[ABLATION_PARENT_VARIANT]["mechanism_activity_status"] == "eligible"
    )
    all_budget_sufficient = all(
        entry["training_budget_status"] == "sufficient"
        for variant_id, entry in selected.items()
        if variant_id in HPO_VARIANT_IDS
    )
    freeze_status = (
        "frozen_from_support_validation_only"
        if stage == "final" and final_design_complete
        and all_learning_eligible and full_mechanism_eligible
        and all_budget_sufficient
        else "provisional_support_validation_only"
    )
    confirmatory_plan_audit = confirmatory_plan.validate_plan()
    development_seeds = set(map(int, first["rollout_seed_roots"]))
    development_seeds.update(map(
        int, first["promotion_calibration_seeds"]
    ))
    development_seeds.update(map(
        int, first["checkpoint_validation_seeds"]
    ))
    development_seeds.update(map(int, first["tuning_validation_seeds"]))
    if development_seeds.intersection(
        confirmatory_plan.DEFAULT_HELDOUT_SEEDS
    ):
        raise ValueError("confirmatory held-out seeds overlap HPO development")
    if set(replicates).intersection(
        confirmatory_plan.DEFAULT_CONFIRMATORY_REPLICATES
    ):
        raise ValueError("confirmatory training replicates overlap HPO")
    frozen = {
        "status": freeze_status,
        "stage": str(stage),
        "final_design_complete": bool(final_design_complete),
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "training_scenario": TRAINING_SCENARIO,
        "training_support_components": list(SELECTION_SCENARIOS),
        "training_episode_protocol": "independent_full_episode_support_batch_v1",
        "promotion_calibration_protocol_version": (
            PROMOTION_CALIBRATION_PROTOCOL_VERSION
        ),
        "selection_scenarios": list(SELECTION_SCENARIOS),
        "ood_period_access_status": "not_loaded",
        "promotion_recovery_access_status": "not_loaded",
        "heldout_test_access_status": "not_loaded",
        "heldout_test_seeds": [],
        "confirmatory_plan_version": confirmatory_plan.CONFIRMATORY_PLAN_VERSION,
        "confirmatory_plan_sha256": confirmatory_plan_audit["sha256"],
        "confirmatory_plan_commitment_status": (
            "source_bound_before_heldout_access"
        ),
        "confirmatory_training_replicate_count": len(
            confirmatory_plan.DEFAULT_CONFIRMATORY_REPLICATES
        ),
        "confirmatory_heldout_path_count": len(
            confirmatory_plan.DEFAULT_HELDOUT_SEEDS
        ),
        "rollout_seed_roots": first["rollout_seed_roots"],
        "promotion_calibration_seeds": first[
            "promotion_calibration_seeds"
        ],
        "checkpoint_validation_seeds": first["checkpoint_validation_seeds"],
        "tuning_validation_seeds": first["tuning_validation_seeds"],
        "training_replicate_seeds": sorted(set(replicates)),
        "checkpoint_boundary_window_fraction": (
            CHECKPOINT_BOUNDARY_WINDOW_FRACTION
        ),
        "max_checkpoint_boundary_replicate_fraction": (
            MAX_CHECKPOINT_BOUNDARY_REPLICATE_FRACTION
        ),
        "training_budget_status": (
            "sufficient" if all_budget_sufficient else "boundary_limited"
        ),
        "training_budget_plan_version": budget_plan.BUDGET_PLAN_VERSION,
        "training_budget_plan_sha256": first[
            "training_budget_plan_sha256"
        ],
        "training_budget_decision_sha256": first[
            "training_budget_decision_sha256"
        ],
        "training_budget_selected_iterations": int(
            first["training_budget_selected_iterations"]
        ),
        "training_budget_binding_status": first[
            "training_budget_binding_status"
        ],
        "pilot_optimizer_seed_overlap": sorted(
            set(DEFAULT_PILOT_OPTIMIZER_SEEDS).intersection(replicates)
        ),
        "steps": int(first["steps"]), "assets": int(first["assets"]),
        "iterations": int(first["iterations"]),
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
        "volume_impact_bps": DEFAULT_VOLUME_IMPACT_BPS,
        "ablation_selection_rule": "inherit_full_candidate_then_disable_exactly_one_registered_mechanism",
        "selected": selected,
        "top_candidates": top_candidates,
        "code_revision": next(iter(revisions)) if len(revisions) == 1 else "",
        "source_manifest_sha256": next(iter(manifests)) if len(manifests) == 1 else "",
        "source_identity_status": "verified" if source_verified else "unregistered_or_incomplete",
    }
    return {
        "summary": {
            "cell_count": len(summaries),
            "tuning_row_count": len(rows),
            "independent_hpo_variant_count": len(variants),
            "ablation_variant_count": len(ALL_VARIANT_IDS) - len(HPO_VARIANT_IDS),
            "training_replicate_count": len(set(replicates)),
            "freeze_status": freeze_status,
            "training_budget_status": (
                "sufficient" if all_budget_sufficient else "boundary_limited"
            ),
            "training_budget_binding_status": (
                "source_bound_decision"
                if budget_binding_verified else "missing_or_invalid"
            ),
            "budget_sufficient_selected_count": sum(
                entry["training_budget_status"] == "sufficient"
                for variant_id, entry in selected.items()
                if variant_id in HPO_VARIANT_IDS
            ),
        },
        "leaderboard": leaderboard,
        "frozen_config": frozen,
    }


def summarize_selective_promotion_pilot(
    input_dirs: list[Path],
    *,
    expected_candidate_ids: list[str],
    expected_replicate_seeds: list[int],
) -> dict[str, Any]:
    """Diagnose promotion candidates without creating a frozen HPO config."""

    candidates = list(expected_candidate_ids)
    replicates = list(map(int, expected_replicate_seeds))
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("promotion pilot candidates must be non-empty and unique")
    if not replicates or len(replicates) != len(set(replicates)):
        raise ValueError("promotion pilot replicates must be non-empty and unique")
    allowed = set(candidate_ids_for_variant(ABLATION_PARENT_VARIANT))
    if not set(candidates).issubset(allowed):
        raise ValueError("promotion pilot accepts only full-method candidates")

    summaries, rows, hf_rows, seen = _load_validated_hpo_cells(input_dirs)
    expected = {
        (ABLATION_PARENT_VARIANT, candidate_id, replicate)
        for candidate_id in candidates
        for replicate in replicates
    }
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise ValueError(
            "incomplete promotion pilot matrix: "
            f"{(missing + unexpected)[:6]}"
        )
    _validate_common_hpo_fields(summaries)

    leaderboard = [
        _candidate_leaderboard_row(
            variant_id=ABLATION_PARENT_VARIANT,
            candidate_id=candidate_id,
            replicates=replicates,
            summaries=summaries,
            rows=rows,
            hf_rows=hf_rows,
        )
        for candidate_id in candidates
    ]
    for row in leaderboard:
        params = CANDIDATES_BY_ID[str(row["candidate_id"])].parameters
        row.update({
            "promotion_initial_probability": float(
                1.0 / (1.0 + np.exp(-float(params["promotion_init_logit"])))
            ),
            "promotion_deterministic_threshold": float(
                params["promotion_deterministic_threshold"]
            ),
            "promotion_gate_interval_steps": int(
                params["promotion_gate_interval_steps"]
            ),
            "promotion_rate_budget": float(params["promotion_rate_budget"]),
            "promotion_replan_cost": float(params["promotion_replan_cost"]),
            "promotion_deterministic_mode": str(
                params["promotion_deterministic_mode"]
            ),
            "promotion_advantage_threshold": float(
                params["promotion_advantage_threshold"]
            ),
            "promotion_advantage_target_threshold": float(
                params["promotion_advantage_target_threshold"]
            ),
            "promotion_advantage_learning_rate": float(
                params["promotion_advantage_learning_rate"]
            ),
            "promotion_advantage_coef": float(
                params["promotion_advantage_coef"]
            ),
        })
    leaderboard.sort(key=lambda row: (
        0 if row["learning_gate_status"] == "eligible" else 1,
        0 if row["mechanism_activity_status"] == "eligible" else 1,
        -float(row["robust_selection_score"]),
        -float(row["tuning_utility_mean"]),
    ))
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank

    jointly_eligible = [
        str(row["candidate_id"])
        for row in leaderboard
        if row["learning_gate_status"] == "eligible"
        and row["mechanism_activity_status"] == "eligible"
    ]
    revisions = {str(summary["code_revision"]).lower() for summary in summaries}
    manifests = {
        str(summary["source_manifest_sha256"]).lower() for summary in summaries
    }
    source_statuses = {
        str(summary["source_identity_status"]) for summary in summaries
    }
    source_verified = (
        len(revisions) == 1
        and len(manifests) == 1
        and source_statuses == {"verified"}
        and is_hex_digest(next(iter(revisions)), length=40)
        and is_hex_digest(next(iter(manifests)), length=64)
    )
    if not source_verified:
        status = "invalid_source_identity"
    elif jointly_eligible:
        status = "selective_candidate_found"
    else:
        status = "no_selective_candidate"
    first = summaries[0]
    replicate_diagnostics = []
    for candidate_id in candidates:
        replicate_diagnostics.extend(_promotion_replicate_diagnostics([
            row for row in rows
            if row["variant_id"] == ABLATION_PARENT_VARIANT
            and row["candidate_id"] == candidate_id
        ]))
    return {
        "summary": {
            "status": status,
            "diagnostic_only": True,
            "frozen_config_created": False,
            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
            "variant_id": ABLATION_PARENT_VARIANT,
            "candidate_count": len(candidates),
            "cell_count": len(summaries),
            "training_replicate_count": len(replicates),
            "training_replicate_seeds": sorted(replicates),
            "promotion_calibration_protocol_version": (
                PROMOTION_CALIBRATION_PROTOCOL_VERSION
            ),
            "promotion_calibration_seeds": first[
                "promotion_calibration_seeds"
            ],
            "support_scenarios": list(SELECTION_SCENARIOS),
            "ood_period_access_status": "not_loaded",
            "promotion_recovery_access_status": "not_loaded",
            "learning_eligible_candidate_count": sum(
                row["learning_gate_status"] == "eligible" for row in leaderboard
            ),
            "mechanism_eligible_candidate_count": sum(
                row["mechanism_activity_status"] == "eligible"
                for row in leaderboard
            ),
            "jointly_eligible_candidate_ids": jointly_eligible,
            "best_diagnostic_candidate_id": str(
                leaderboard[0]["candidate_id"]
            ),
            "steps": int(first["steps"]),
            "assets": int(first["assets"]),
            "iterations": int(first["iterations"]),
            "source_identity_status": (
                "verified" if source_verified else "unregistered_or_incomplete"
            ),
            "code_revision": next(iter(revisions)) if len(revisions) == 1 else "",
            "source_manifest_sha256": (
                next(iter(manifests)) if len(manifests) == 1 else ""
            ),
        },
        "leaderboard": leaderboard,
        "replicate_diagnostics": replicate_diagnostics,
    }


def write_hpo_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", payload["leaderboard"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "frozen_config.json").write_text(
        json.dumps(payload["frozen_config"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_selective_promotion_pilot(
    output_dir: Path, payload: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", payload["leaderboard"])
    _write_csv(
        output_dir / "replicate_diagnostics.csv",
        payload["replicate_diagnostics"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_frozen_config(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "frozen_from_support_validation_only":
        raise ValueError("v7 config is not final and frozen")
    if payload.get("stage") != "final" or not bool(
        payload.get("final_design_complete")
    ):
        raise ValueError("v7 frozen config did not complete the final design")
    if payload.get("tuning_protocol_version") != FULL_METHOD_TUNING_PROTOCOL_VERSION:
        raise ValueError("v7 tuning protocol mismatch")
    if payload.get("ood_period_access_status") != "not_loaded":
        raise ValueError("v7 HPO accessed OOD")
    if payload.get("promotion_recovery_access_status") != "not_loaded":
        raise ValueError("v7 HPO accessed confirmatory promotion recovery")
    if payload.get("promotion_calibration_protocol_version") != (
        PROMOTION_CALIBRATION_PROTOCOL_VERSION
    ):
        raise ValueError("v7 promotion calibration protocol mismatch")
    plan_audit = confirmatory_plan.validate_plan()
    if (
        payload.get("confirmatory_plan_version")
        != confirmatory_plan.CONFIRMATORY_PLAN_VERSION
        or payload.get("confirmatory_plan_sha256") != plan_audit["sha256"]
        or payload.get("confirmatory_plan_commitment_status")
        != "source_bound_before_heldout_access"
    ):
        raise ValueError("v7 confirmatory plan commitment drifted")
    if int(payload.get("confirmatory_training_replicate_count", 0)) != len(
        confirmatory_plan.DEFAULT_CONFIRMATORY_REPLICATES
    ) or int(payload.get("confirmatory_heldout_path_count", 0)) != len(
        confirmatory_plan.DEFAULT_HELDOUT_SEEDS
    ):
        raise ValueError("v7 confirmatory design count drifted")
    if payload.get("training_budget_status") != "sufficient":
        raise ValueError("v7 frozen config is checkpoint-boundary limited")
    if (
        payload.get("training_budget_plan_version")
        != budget_plan.BUDGET_PLAN_VERSION
        or payload.get("training_budget_plan_sha256")
        != budget_plan.plan_sha256()
        or payload.get("training_budget_binding_status")
        != "source_bound_decision"
        or not is_hex_digest(
            payload.get("training_budget_decision_sha256"), length=64
        )
        or int(payload.get("training_budget_selected_iterations", 0))
        != int(payload.get("iterations", -1))
        or int(payload.get("iterations", 0))
        < budget_plan.MIN_FINAL_ITERATIONS
    ):
        raise ValueError("v7 source-bound training-budget decision drifted")
    if not np.isclose(
        float(payload.get("checkpoint_boundary_window_fraction", -1.0)),
        CHECKPOINT_BOUNDARY_WINDOW_FRACTION,
    ) or not np.isclose(
        float(payload.get(
            "max_checkpoint_boundary_replicate_fraction", -1.0
        )),
        MAX_CHECKPOINT_BOUNDARY_REPLICATE_FRACTION,
    ):
        raise ValueError("v7 checkpoint-boundary contract drifted")
    validate_unique_seeds(
        payload.get("promotion_calibration_seeds", []),
        role="promotion_calibration_seeds",
    )
    role_seeds = {
        "training": set(map(int, payload.get("rollout_seed_roots", []))),
        "promotion_calibration": set(map(
            int, payload.get("promotion_calibration_seeds", [])
        )),
        "checkpoint_validation": set(map(
            int, payload.get("checkpoint_validation_seeds", [])
        )),
        "tuning_validation": set(map(
            int, payload.get("tuning_validation_seeds", [])
        )),
    }
    names = list(role_seeds)
    if any(
        role_seeds[left].intersection(role_seeds[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    ):
        raise ValueError("v7 frozen seed roles overlap")
    if payload.get("heldout_test_access_status") != "not_loaded" or payload.get(
        "heldout_test_seeds"
    ):
        raise ValueError("v7 HPO accessed held-out seeds")
    if set(payload.get("selection_scenarios", [])) != set(SELECTION_SCENARIOS):
        raise ValueError("v7 support selection scenarios drifted")
    if payload.get("source_identity_status") != "verified":
        raise ValueError("v7 frozen source identity is not verified")
    if not is_hex_digest(payload.get("code_revision"), length=40) or not is_hex_digest(
        payload.get("source_manifest_sha256"), length=64
    ):
        raise ValueError("v7 frozen source hashes are invalid")
    if len(set(map(int, payload.get("training_replicate_seeds", [])))) < 5:
        raise ValueError("v7 final HPO requires at least five training replicates")
    final_replicates = set(map(int, payload["training_replicate_seeds"]))
    if not set(DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS).issubset(final_replicates):
        raise ValueError("v7 final HPO is missing registered optimizer seeds")
    if set(DEFAULT_PILOT_OPTIMIZER_SEEDS).intersection(final_replicates):
        raise ValueError("v7 final HPO reused pilot optimizer seeds")
    if final_replicates.intersection(
        confirmatory_plan.DEFAULT_CONFIRMATORY_REPLICATES
    ):
        raise ValueError("v7 confirmatory training replicates overlap HPO")
    development_seeds = set(map(int, payload.get("rollout_seed_roots", [])))
    development_seeds.update(map(
        int, payload.get("promotion_calibration_seeds", [])
    ))
    development_seeds.update(map(
        int, payload.get("checkpoint_validation_seeds", [])
    ))
    development_seeds.update(map(
        int, payload.get("tuning_validation_seeds", [])
    ))
    if development_seeds.intersection(
        confirmatory_plan.DEFAULT_HELDOUT_SEEDS
    ):
        raise ValueError("v7 confirmatory held-out seeds overlap development")
    if set(payload.get("selected", {})) != set(ALL_VARIANT_IDS):
        raise ValueError("v7 frozen config is missing variants")
    parent = payload["selected"][ABLATION_PARENT_VARIANT]["candidate_id"]
    for variant in VARIANTS:
        entry = payload["selected"][variant.variant_id]
        if variant.inherits_full_selection and entry["candidate_id"] != parent:
            raise ValueError("v7 ablation did not inherit the full candidate")
        expected = effective_parameters_for_variant(
            variant.variant_id, str(entry["candidate_id"])
        )
        if entry.get("effective_parameters") != expected:
            raise ValueError(f"v7 effective parameters drifted: {variant.variant_id}")
        if entry.get("learning_gate_status") != "eligible":
            raise ValueError(f"v7 selected model did not learn: {variant.variant_id}")
        if entry.get("training_budget_status") != "sufficient":
            raise ValueError(
                f"v7 selected model is checkpoint-boundary limited: "
                f"{variant.variant_id}"
            )
        if variant.inherits_full_selection and entry.get(
            "selection_source_variant"
        ) != ABLATION_PARENT_VARIANT:
            raise ValueError("v7 ablation selection provenance drifted")
    if payload["selected"][ABLATION_PARENT_VARIANT].get(
        "mechanism_activity_status"
    ) != "eligible":
        raise ValueError("v7 full method failed the mechanism activity gate")
    return {"status": "valid", "sha256": frozen_config_sha256(payload)}


def load_frozen_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload, validate_frozen_config(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", choices=sorted(CANDIDATES_BY_ID))
    parser.add_argument("--variant-id", choices=HPO_VARIANT_IDS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument(
        "--promotion-calibration-seeds", type=int, nargs="+",
        default=list(DEFAULT_PROMOTION_CALIBRATION_SEEDS),
    )
    parser.add_argument("--checkpoint-validation-seeds", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_VALIDATION_SEEDS))
    parser.add_argument("--tuning-validation-seeds", type=int, nargs="+", default=list(DEFAULT_TUNING_SEEDS))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--source-manifest-sha256", default="")
    parser.add_argument("--training-budget-plan-sha256", default="")
    parser.add_argument("--training-budget-decision-sha256", default="")
    parser.add_argument("--training-budget-selected-iterations", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument("--expected-variant-ids", nargs="*", choices=HPO_VARIANT_IDS)
    parser.add_argument("--expected-candidate-ids", nargs="*", choices=sorted(CANDIDATES_BY_ID))
    parser.add_argument("--expected-replicate-seeds", type=int, nargs="*")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_hpo_cells(
            list(args.merge_inputs),
            expected_variant_ids=list(args.expected_variant_ids or HPO_VARIANT_IDS),
            expected_candidate_ids=list(args.expected_candidate_ids or CANDIDATES_BY_ID),
            expected_replicate_seeds=list(args.expected_replicate_seeds or []),
            top_k=int(args.top_k), stage=str(args.stage),
        )
        write_hpo_merge(args.output_dir, payload)
        print(f"full_method_hpo_v7_merge cells={payload['summary']['cell_count']} output={args.output_dir}")
        return
    required = (args.candidate_id, args.variant_id, args.training_replicate_seed)
    if any(value is None for value in required):
        parser.error("cell mode requires candidate, variant, and training replicate")
    payload = run_hpo_cell(
        candidate_id=str(args.candidate_id), variant_id=str(args.variant_id),
        training_replicate_seed=int(args.training_replicate_seed),
        train_seeds=list(args.train_seeds),
        promotion_calibration_seeds=list(args.promotion_calibration_seeds),
        checkpoint_validation_seeds=list(args.checkpoint_validation_seeds),
        tuning_validation_seeds=list(args.tuning_validation_seeds),
        steps=int(args.steps), assets=int(args.assets), iterations=int(args.iterations),
        code_revision=str(args.code_revision),
        expected_source_manifest_sha256=str(args.source_manifest_sha256),
        training_budget_plan_sha256=str(args.training_budget_plan_sha256),
        training_budget_decision_sha256=str(
            args.training_budget_decision_sha256
        ),
        training_budget_selected_iterations=int(
            args.training_budget_selected_iterations
        ),
    )
    write_hpo_cell(args.output_dir, payload)
    print(f"full_method_hpo_v7_cell status=valid variant={args.variant_id} candidate={args.candidate_id} output={args.output_dir}")


if __name__ == "__main__":
    main()
