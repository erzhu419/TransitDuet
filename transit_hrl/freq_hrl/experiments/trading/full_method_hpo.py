"""Nested-validation HPO for the complete causal Freq-HRL method.

This protocol is intentionally independent from ``nested_validation_hpo_v1``.
The older protocol tunes the routing/SMDP core and must remain reproducible.
Here every method is evaluated on the causal post-trade environment with the
same nonlinear volume-impact contract and a capacity reference to full v4.

The three ablations inherit the validation-selected full-method candidate.
They are never tuned to separate optima, so the confirmatory comparison changes
one registered mechanism rather than both mechanism and hyperparameters.
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
from .offpolicy_baseline_validation import train_flat_offpolicy_baseline
from .performance_validation import SCENARIOS
from .ppo_actor_critic import (
    FULL_METHOD_IMPLEMENTATION_VERSION,
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
    evaluate_hf_lower_intervention,
    make_plan_mapper,
    promotion_gate_state_dim,
    resolve_method_contract,
    smdp_parameter_count,
    train_ppo_actor_critic,
)
from .strong_learned_baseline_validation import (
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
    count_parameters,
    scenario_optimizer_seed,
)


FULL_METHOD_TUNING_PROTOCOL_VERSION = "full_method_nested_hpo_v2"
FULL_METHOD_HPO_IMPLEMENTATION_VERSION = (
    "full_method_hpo_capacity_ablation_hf_intervention_v2_2026_08_03"
)
DEFAULT_TUNING_SEEDS = (68207, 68209, 68213, 68219, 68227)
DEFAULT_PILOT_SCENARIOS = (
    "stationary_low_noise",
    "persistent_shift",
    "ood_period",
)
EXECUTION_TIMELINE_CONTRACT = "causal_post_trade_v3"
CAPACITY_REFERENCE_METHOD_CONTRACT = "full_freq_hrl_v4"
DEFAULT_VOLUME_IMPACT_BPS = 10.0
DEFAULT_PLAN_BASIS_DIM = 3
DEFAULT_PLAN_HORIZON_S = 1800.0
DEFAULT_PLAN_EVAL_OFFSET_S = 300.0
DEFAULT_PLAN_COEFFICIENT_SCALE = 0.75
ABLATION_PARENT_VARIANT = "freq_hrl_full_v4"


@dataclass(frozen=True)
class FullMethodVariant:
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


VARIANTS = (
    FullMethodVariant(
        "freq_hrl_full_v4",
        "ppo",
        "frequency_ppo",
        "freq_hrl",
        "full_freq_hrl_v4",
        "proposed_full_method",
    ),
    FullMethodVariant(
        "freq_hrl_no_promotion_v4",
        "ppo",
        "frequency_ppo",
        "freq_hrl",
        "ablate_promotion_v4",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "freq_hrl_no_hf_lower_v4",
        "ppo",
        "frequency_ppo",
        "freq_hrl",
        "ablate_hf_lower_v4",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "freq_hrl_no_leakage_v4",
        "ppo",
        "frequency_ppo",
        "freq_hrl",
        "ablate_leakage_v4",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "flat_ppo_matched_v4",
        "ppo",
        "baseline_ppo",
        "flat_ppo",
        "routing_core_v2",
        "capacity_matched_flat_baseline",
    ),
    FullMethodVariant(
        "flat_gru_ppo_matched_v4",
        "ppo",
        "baseline_ppo",
        "flat_gru_ppo",
        "routing_core_v2",
        "capacity_matched_flat_recurrent_baseline",
    ),
    FullMethodVariant(
        "generic_hrl_ppo_matched_v4",
        "ppo",
        "baseline_ppo",
        "generic_hrl_ppo",
        "curve_credit_control_v3",
        "capacity_matched_nonfrequency_hrl_baseline",
    ),
    FullMethodVariant(
        "generic_hrl_gru_ppo_matched_v4",
        "ppo",
        "baseline_ppo",
        "generic_hrl_gru_ppo",
        "curve_credit_control_v3",
        "capacity_matched_nonfrequency_recurrent_hrl_baseline",
    ),
    FullMethodVariant(
        "flat_sac_matched_v4",
        "offpolicy",
        "offpolicy",
        "flat_sac",
        "routing_core_v2",
        "capacity_matched_offpolicy_baseline",
    ),
    FullMethodVariant(
        "flat_td3_matched_v4",
        "offpolicy",
        "offpolicy",
        "flat_td3",
        "routing_core_v2",
        "capacity_matched_offpolicy_baseline",
    ),
)
VARIANTS_BY_ID = {variant.variant_id: variant for variant in VARIANTS}
ALL_VARIANT_IDS = tuple(variant.variant_id for variant in VARIANTS)


@dataclass(frozen=True)
class FullMethodCandidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]

    def applies_to(self, variant_id: str) -> bool:
        variant = VARIANTS_BY_ID.get(str(variant_id))
        return variant is not None and variant.candidate_family == self.family


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
    learning_rate: float,
    init_log_std: float,
    leakage_scale: float,
    constraint_init: float,
    constraint_target: float,
    dual_lr: float,
    objective_weight: float,
    promotion_init_logit: float,
    promotion_replan_cost: float,
    lower_hf_order_scale: float,
) -> FullMethodCandidate:
    return FullMethodCandidate(
        candidate_id,
        "frequency_ppo",
        {
            **_optimizer_parameters(learning_rate, init_log_std),
            "leakage_scale": float(leakage_scale),
            "lower_lf_constraint_coef": float(constraint_init),
            "lower_lf_constraint_target": float(constraint_target),
            "lower_lf_dual_lr": float(dual_lr),
            "lower_lf_objective_weight": float(objective_weight),
            "lower_lf_effect_filter_window": 0,
            "lower_lf_effect_filter_gain": 1.0,
            "lower_lf_raw_recenter_gain": 0.0,
            "lower_lf_raw_recenter_scale": 0.10,
            "plan_smoothness_weight": 1e-5,
            "promotion_init_logit": float(promotion_init_logit),
            "promotion_replan_cost": float(promotion_replan_cost),
            "lower_hf_order_scale": float(lower_hf_order_scale),
        },
    )


def _baseline_ppo_candidate(
    candidate_id: str,
    learning_rate: float,
    init_log_std: float,
) -> FullMethodCandidate:
    return FullMethodCandidate(
        candidate_id,
        "baseline_ppo",
        _optimizer_parameters(learning_rate, init_log_std),
    )


def _offpolicy_candidate(
    candidate_id: str,
    *,
    learning_rate: float,
    warmup_steps: int,
    batch_size: int,
) -> FullMethodCandidate:
    return FullMethodCandidate(
        candidate_id,
        "offpolicy",
        {
            "hidden_dim": 64,
            "learning_rate": float(learning_rate),
            "replay_capacity": 100_000,
            "warmup_steps": int(warmup_steps),
            "batch_size": int(batch_size),
            "updates_per_step": 1,
            "reward_scale": DEFAULT_TRAINING_REWARD_SCALE,
        },
    )


FULL_METHOD_CANDIDATES = (
    _frequency_candidate(
        "freq_lr1e4_std15_conservative",
        learning_rate=1e-4,
        init_log_std=-1.5,
        leakage_scale=2.5e-4,
        constraint_init=0.02,
        constraint_target=0.20,
        dual_lr=5e-4,
        objective_weight=2.5e-4,
        promotion_init_logit=-1.5,
        promotion_replan_cost=1e-5,
        lower_hf_order_scale=0.015,
    ),
    _frequency_candidate(
        "freq_lr1e4_std10_balanced",
        learning_rate=1e-4,
        init_log_std=-1.0,
        leakage_scale=5e-4,
        constraint_init=0.05,
        constraint_target=0.15,
        dual_lr=5e-4,
        objective_weight=5e-4,
        promotion_init_logit=-1.0,
        promotion_replan_cost=2.5e-5,
        lower_hf_order_scale=0.020,
    ),
    _frequency_candidate(
        "freq_lr3e4_std15_lowgate",
        learning_rate=3e-4,
        init_log_std=-1.5,
        leakage_scale=5e-4,
        constraint_init=0.05,
        constraint_target=0.10,
        dual_lr=1e-3,
        objective_weight=5e-4,
        promotion_init_logit=-0.75,
        promotion_replan_cost=5e-5,
        lower_hf_order_scale=0.020,
    ),
    _frequency_candidate(
        "freq_lr3e4_std10_balanced",
        learning_rate=3e-4,
        init_log_std=-1.0,
        leakage_scale=7.5e-4,
        constraint_init=0.10,
        constraint_target=0.10,
        dual_lr=1e-3,
        objective_weight=7.5e-4,
        promotion_init_logit=-0.5,
        promotion_replan_cost=7.5e-5,
        lower_hf_order_scale=0.025,
    ),
    _frequency_candidate(
        "freq_lr1e4_std05_exploratory",
        learning_rate=1e-4,
        init_log_std=-0.5,
        leakage_scale=1e-3,
        constraint_init=0.10,
        constraint_target=0.075,
        dual_lr=2e-3,
        objective_weight=1e-3,
        promotion_init_logit=-0.25,
        promotion_replan_cost=1e-4,
        lower_hf_order_scale=0.030,
    ),
    _frequency_candidate(
        "freq_lr3e4_std05_activegate",
        learning_rate=3e-4,
        init_log_std=-0.5,
        leakage_scale=1e-3,
        constraint_init=0.15,
        constraint_target=0.05,
        dual_lr=2e-3,
        objective_weight=1e-3,
        promotion_init_logit=0.0,
        promotion_replan_cost=1.5e-4,
        lower_hf_order_scale=0.035,
    ),
    _baseline_ppo_candidate("ppo_lr1e4_std15", 1e-4, -1.5),
    _baseline_ppo_candidate("ppo_lr1e4_std10", 1e-4, -1.0),
    _baseline_ppo_candidate("ppo_lr3e4_std15", 3e-4, -1.5),
    _baseline_ppo_candidate("ppo_lr3e4_std10", 3e-4, -1.0),
    _baseline_ppo_candidate("ppo_lr1e4_std05", 1e-4, -0.5),
    _baseline_ppo_candidate("ppo_lr3e4_std05", 3e-4, -0.5),
    _offpolicy_candidate(
        "off_lr1e4_w1024_b64",
        learning_rate=1e-4,
        warmup_steps=1024,
        batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr1e4_w2048_b64",
        learning_rate=1e-4,
        warmup_steps=2048,
        batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w1024_b64",
        learning_rate=3e-4,
        warmup_steps=1024,
        batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w2048_b64",
        learning_rate=3e-4,
        warmup_steps=2048,
        batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w2048_b128",
        learning_rate=3e-4,
        warmup_steps=2048,
        batch_size=128,
    ),
    _offpolicy_candidate(
        "off_lr1e3_w4096_b64",
        learning_rate=1e-3,
        warmup_steps=4096,
        batch_size=64,
    ),
)
CANDIDATES_BY_ID = {
    candidate.candidate_id: candidate for candidate in FULL_METHOD_CANDIDATES
}


def candidate_ids_for_variant(variant_id: str) -> list[str]:
    if variant_id not in VARIANTS_BY_ID:
        raise ValueError(f"unknown variant_id: {variant_id}")
    return [
        candidate.candidate_id
        for candidate in FULL_METHOD_CANDIDATES
        if candidate.applies_to(variant_id)
    ]


def canonical_full_method_parameter_count(
    assets: int,
    *,
    hidden_dim: int = 64,
    plan_basis_dim: int = DEFAULT_PLAN_BASIS_DIM,
) -> int:
    if int(assets) <= 0:
        raise ValueError("assets must be positive")
    if int(plan_basis_dim) < 2:
        raise ValueError("full-method plan_basis_dim must be at least two")
    config = SMDPPPOConfig(
        upper_state_dim=6 * int(assets) + 5,
        lower_state_dim=8 * int(assets) + 1,
        upper_action_dim=(int(plan_basis_dim) - 1) * int(assets),
        lower_action_dim=2 * int(assets),
        promotion_state_dim=promotion_gate_state_dim(int(assets)),
        hidden_dim=int(hidden_dim),
    )
    return smdp_parameter_count(config)


def effective_parameters_for_variant(
    variant_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    variant = VARIANTS_BY_ID.get(str(variant_id))
    candidate = CANDIDATES_BY_ID.get(str(candidate_id))
    if variant is None:
        raise ValueError(f"unknown variant_id: {variant_id}")
    if candidate is None or not candidate.applies_to(variant.variant_id):
        raise ValueError(
            f"candidate {candidate_id} does not apply to {variant.variant_id}"
        )
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
            "upper_period": 30,
            "min_upper_duration": 5,
            "use_handcrafted_frequency_prior": False,
        })
    if variant.candidate_family == "baseline_ppo":
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
            "promotion_replan_cost": 0.0,
            "lower_hf_order_scale": 0.0,
        })
    elif variant.method_contract == "ablate_promotion_v4":
        params["promotion_replan_cost"] = 0.0
    elif variant.method_contract == "ablate_hf_lower_v4":
        params["lower_hf_order_scale"] = 0.0
    elif variant.method_contract == "ablate_leakage_v4":
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
        })
    return params


def frozen_config_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_frozen_config(
    payload: dict[str, Any],
    *,
    required_variant_ids: Iterable[str] = ALL_VARIANT_IDS,
) -> dict[str, Any]:
    required = list(required_variant_ids)
    if payload.get("status") != "frozen_from_validation_only":
        raise ValueError("frozen config status must be frozen_from_validation_only")
    if payload.get("stage") != "final" or not bool(payload.get("final_design_complete")):
        raise ValueError("frozen config must come from a complete final design")
    if payload.get("tuning_protocol_version") != FULL_METHOD_TUNING_PROTOCOL_VERSION:
        raise ValueError("frozen config full-method tuning protocol mismatch")
    if payload.get("hpo_implementation_version") != FULL_METHOD_HPO_IMPLEMENTATION_VERSION:
        raise ValueError("frozen config HPO implementation mismatch")
    if payload.get("full_method_implementation_version") != FULL_METHOD_IMPLEMENTATION_VERSION:
        raise ValueError("frozen config full-method implementation mismatch")
    if payload.get("learned_baseline_implementation_version") != (
        LEARNED_BASELINE_IMPLEMENTATION_VERSION
    ):
        raise ValueError("frozen config learned-baseline implementation mismatch")
    if payload.get("selection_objective_version") != SELECTION_OBJECTIVE_VERSION:
        raise ValueError("frozen config selection objective mismatch")
    if payload.get("source_identity_status") != "verified":
        raise ValueError("frozen config source identity is not verified")
    if not is_hex_digest(payload.get("code_revision"), length=40):
        raise ValueError("frozen config requires a full Git revision")
    if not is_hex_digest(payload.get("source_manifest_sha256"), length=64):
        raise ValueError("frozen config requires a source manifest SHA-256")
    if payload.get("heldout_test_access_status") != "not_loaded":
        raise ValueError("frozen config accessed held-out test data")
    if payload.get("heldout_test_seeds"):
        raise ValueError("frozen config contains held-out test seeds")
    checkpoint = {int(seed) for seed in payload.get("checkpoint_validation_seeds", [])}
    tuning = {int(seed) for seed in payload.get("tuning_validation_seeds", [])}
    if not checkpoint or not tuning or checkpoint & tuning:
        raise ValueError("frozen config validation splits must be non-empty and disjoint")
    if set(payload.get("scenarios", [])) != set(DEFAULT_SCENARIOS):
        raise ValueError("frozen config must cover the preregistered five scenarios")
    if len(set(map(int, payload.get("training_replicate_seeds", [])))) < 5:
        raise ValueError("frozen config requires at least five training replicates")
    if payload.get("ablation_selection_rule") != (
        "inherit_full_candidate_then_disable_exactly_one_registered_mechanism"
    ):
        raise ValueError("frozen config has an invalid ablation selection rule")
    if payload.get("execution_timeline_contract") != EXECUTION_TIMELINE_CONTRACT:
        raise ValueError("frozen config execution timeline contract drifted")
    if payload.get("capacity_reference_method_contract") != (
        CAPACITY_REFERENCE_METHOD_CONTRACT
    ):
        raise ValueError("frozen config capacity reference contract drifted")
    if float(payload.get("volume_impact_bps", -1.0)) != DEFAULT_VOLUME_IMPACT_BPS:
        raise ValueError("frozen config environment impact contract drifted")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("frozen config selected variants are missing")
    full_candidate_id = str(
        selected.get(ABLATION_PARENT_VARIANT, {}).get("candidate_id", "")
    )
    for variant_id in required:
        variant = VARIANTS_BY_ID.get(variant_id)
        if variant is None:
            raise ValueError(f"unknown required variant: {variant_id}")
        entry = selected.get(variant_id)
        if not isinstance(entry, dict):
            raise ValueError(f"missing frozen variant: {variant_id}")
        candidate_id = str(entry.get("candidate_id", ""))
        candidate = CANDIDATES_BY_ID.get(candidate_id)
        if candidate is None or not candidate.applies_to(variant_id):
            raise ValueError(f"invalid frozen candidate for {variant_id}")
        if entry.get("candidate_parameters") != candidate.parameters:
            raise ValueError(f"frozen candidate parameters drifted for {variant_id}")
        expected_effective = effective_parameters_for_variant(variant_id, candidate_id)
        if entry.get("effective_parameters") != expected_effective:
            raise ValueError(f"frozen effective parameters drifted for {variant_id}")
        if entry.get("learning_gate_status") != "eligible":
            raise ValueError(f"frozen variant failed learning gate: {variant_id}")
        if variant.inherits_full_selection:
            if candidate_id != full_candidate_id:
                raise ValueError("ablation does not inherit the full-method candidate")
            if entry.get("selection_source_variant") != ABLATION_PARENT_VARIANT:
                raise ValueError("ablation selection provenance is invalid")
    full_entry = selected.get(ABLATION_PARENT_VARIANT, {})
    if full_entry.get("mechanism_activity_status") != "eligible":
        raise ValueError("full method failed its mechanism-activity gate")
    budgets = payload.get("search_budget_candidates_per_variant", {})
    if not budgets or len({int(budgets.get(variant_id, 0)) for variant_id in required}) != 1:
        raise ValueError("full-method HPO search budgets are not equal")
    if min(int(budgets.get(variant_id, 0)) for variant_id in required) < 2:
        raise ValueError("full-method HPO search budget is too small")
    return {
        "status": "valid",
        "sha256": frozen_config_sha256(payload),
        "selected": {variant_id: selected[variant_id] for variant_id in required},
        "code_revision": str(payload["code_revision"]).lower(),
        "source_manifest_sha256": str(payload["source_manifest_sha256"]).lower(),
    }


def load_frozen_config(
    path: Path,
    *,
    required_variant_ids: Iterable[str] = ALL_VARIANT_IDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("full-method frozen config must be a JSON object")
    return payload, validate_frozen_config(
        payload, required_variant_ids=required_variant_ids
    )


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ppo_training_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "hidden_dim",
        "learning_rate",
        "epochs",
        "minibatch_size",
        "init_log_std",
        "reward_scale",
        "leakage_scale",
        "plan_basis_dim",
        "plan_horizon_s",
        "plan_eval_offset_s",
        "plan_coefficient_scale",
        "lower_lf_constraint_coef",
        "lower_lf_constraint_target",
        "lower_lf_dual_lr",
        "lower_lf_objective_weight",
        "lower_lf_effect_filter_window",
        "lower_lf_effect_filter_gain",
        "lower_lf_raw_recenter_gain",
        "lower_lf_raw_recenter_scale",
        "policy_mode",
        "upper_period",
        "min_upper_duration",
        "use_handcrafted_frequency_prior",
        "execution_timeline_contract",
        "method_contract",
        "capacity_reference_method_contract",
        "volume_impact_bps",
        "plan_smoothness_weight",
        "promotion_replan_cost",
        "promotion_init_logit",
        "lower_hf_order_scale",
    )
    result = {key: params[key] for key in keys}
    result["ppo_epochs"] = result.pop("epochs")
    return result


def _hf_intervention_kwargs(
    *,
    params: dict[str, Any],
    steps: int,
    assets: int,
    scenario: str,
) -> dict[str, Any]:
    flags = resolve_method_contract(str(params["method_contract"]))
    mapper = make_plan_mapper(
        assets=int(assets),
        plan_basis_dim=int(params["plan_basis_dim"]),
        plan_horizon_s=float(params["plan_horizon_s"]),
        plan_eval_offset_s=float(params["plan_eval_offset_s"]),
        plan_coefficient_scale=float(params["plan_coefficient_scale"]),
        anchor_first_coefficient=True,
    )
    return {
        "steps": int(steps),
        "assets": int(assets),
        "scenario": str(scenario),
        "leakage_scale": 0.0,
        "plan_mapper": mapper,
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
        "lower_hf_order_scale": float(params["lower_hf_order_scale"]),
        "execution_timeline_contract": str(params["execution_timeline_contract"]),
        "method_contract": str(params["method_contract"]),
    }


def run_hpo_cell(
    *,
    candidate_id: str,
    variant_id: str,
    scenario: str,
    training_replicate_seed: int,
    train_seeds: list[int],
    checkpoint_validation_seeds: list[int],
    tuning_validation_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
) -> dict[str, Any]:
    variant = VARIANTS_BY_ID.get(str(variant_id))
    candidate = CANDIDATES_BY_ID.get(str(candidate_id))
    if variant is None:
        raise ValueError(f"unknown variant_id: {variant_id}")
    if candidate is None or not candidate.applies_to(variant.variant_id):
        raise ValueError(f"candidate {candidate_id} does not apply to {variant_id}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    rollout_seed_roots = validate_unique_seeds(train_seeds, role="rollout_seed_roots")
    checkpoint_seeds, tuning_seeds = validate_evaluation_seed_roles(
        checkpoint_validation_seeds, tuning_validation_seeds
    )
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(code_revision),
        expected_source_manifest_sha256=str(expected_source_manifest_sha256),
    )
    run_seed = scenario_optimizer_seed(int(training_replicate_seed), scenario)
    params = effective_parameters_for_variant(variant.variant_id, candidate.candidate_id)
    started = time.perf_counter()
    if variant.trainer_family == "ppo":
        model_payload, tuning_rows, model = train_ppo_actor_critic(
            train_seeds=rollout_seed_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps),
            assets=int(assets),
            scenario=str(scenario),
            iterations=int(iterations),
            seed=int(run_seed),
            resample_training_paths=True,
            evaluation_role="tuning_validation",
            **_ppo_training_kwargs(params),
        )
    else:
        capacity_target = canonical_full_method_parameter_count(
            int(assets), hidden_dim=int(params["hidden_dim"])
        )
        model_payload, tuning_rows, model = train_flat_offpolicy_baseline(
            policy_mode=variant.policy_mode,
            train_seeds=rollout_seed_roots,
            validation_seeds=checkpoint_seeds,
            eval_seeds=tuning_seeds,
            steps=int(steps),
            assets=int(assets),
            scenario=str(scenario),
            iterations=int(iterations),
            seed=int(run_seed),
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            replay_capacity=int(params["replay_capacity"]),
            warmup_steps=int(params["warmup_steps"]),
            batch_size=int(params["batch_size"]),
            updates_per_step=int(params["updates_per_step"]),
            resample_training_paths=True,
            evaluation_role="tuning_validation",
            reward_scale=float(params["reward_scale"]),
            execution_timeline_contract=str(params["execution_timeline_contract"]),
            volume_impact_bps=float(params["volume_impact_bps"]),
            capacity_target_parameter_count=int(capacity_target),
            capacity_reference_method_contract=str(
                params["capacity_reference_method_contract"]
            ),
        )
    if model_payload.get("evaluation_role") != "tuning_validation":
        raise RuntimeError("full-method HPO trainer exposed the wrong evaluation role")
    if model_payload.get("heldout_test_seeds"):
        raise RuntimeError("full-method HPO must not load held-out test seeds")
    if str(model_payload.get("execution_timeline_contract")) != EXECUTION_TIMELINE_CONTRACT:
        raise RuntimeError("full-method HPO did not use the causal execution contract")
    if float(model_payload.get("volume_impact_bps", -1.0)) != DEFAULT_VOLUME_IMPACT_BPS:
        raise RuntimeError("full-method HPO environment impact contract drifted")

    annotated_rows: list[dict[str, Any]] = []
    utilities: list[float] = []
    for row in tuning_rows:
        utility = float(validation_utility(row))
        if not np.isfinite(utility):
            raise RuntimeError("full-method HPO tuning utility is not finite")
        utilities.append(utility)
        annotated_rows.append({
            **row,
            "variant_id": variant.variant_id,
            "candidate_id": candidate.candidate_id,
            "candidate_family": candidate.family,
            "scientific_role": variant.scientific_role,
            "scenario": str(scenario),
            "training_replicate_seed": int(training_replicate_seed),
            "optimizer_seed": int(run_seed),
            "evaluation_role": "tuning_validation",
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
            "code_revision": source_identity["code_revision"],
            "source_manifest_sha256": source_identity["source_manifest_sha256"],
            "source_identity_status": source_identity["source_identity_status"],
            "selection_utility": utility,
        })
    if len(annotated_rows) != len(tuning_seeds):
        raise RuntimeError("full-method HPO must emit one tuning row per tuning seed")
    if {int(row["seed"]) for row in annotated_rows} != set(tuning_seeds):
        raise RuntimeError("full-method HPO tuning rows do not match tuning seeds")

    method_flags = resolve_method_contract(variant.method_contract)
    hf_rows: list[dict[str, Any]] = []
    if variant.policy_mode == "freq_hrl" and method_flags["lower_hf_overlay"]:
        hf_rows = evaluate_hf_lower_intervention(
            model,
            eval_seeds=list(tuning_seeds),
            rollout_kwargs=_hf_intervention_kwargs(
                params=params,
                steps=int(steps),
                assets=int(assets),
                scenario=str(scenario),
            ),
        )
        hf_rows = [{
            **row,
            "variant_id": variant.variant_id,
            "candidate_id": candidate.candidate_id,
            "training_replicate_seed": int(training_replicate_seed),
            "optimizer_seed": int(run_seed),
            "evaluation_role": "tuning_validation_mechanism_diagnostic",
            "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
            "code_revision": source_identity["code_revision"],
            "source_manifest_sha256": source_identity["source_manifest_sha256"],
        } for row in hf_rows]
        if len(hf_rows) != len(tuning_seeds):
            raise RuntimeError("HF intervention must emit one paired row per tuning seed")

    parameter_count = count_parameters(model)
    target_count = int(model_payload.get("capacity_target_parameter_count", parameter_count))
    actual_count = int(model_payload.get("capacity_actual_parameter_count", parameter_count))
    if actual_count != parameter_count:
        raise RuntimeError("reported active parameter count does not match the checkpoint")
    elapsed = float(time.perf_counter() - started)
    summary = {
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "ablation_of": variant.ablation_of,
        "candidate_id": candidate.candidate_id,
        "candidate_family": candidate.family,
        "candidate_parameters": dict(candidate.parameters),
        "effective_parameters": params,
        "trainer_family": variant.trainer_family,
        "policy_mode": variant.policy_mode,
        "method_contract": variant.method_contract,
        "scenario": str(scenario),
        "training_replicate_seed": int(training_replicate_seed),
        "optimizer_seed": int(run_seed),
        "rollout_seed_roots": list(rollout_seed_roots),
        "checkpoint_validation_seeds": list(checkpoint_seeds),
        "tuning_validation_seeds": list(tuning_seeds),
        "heldout_test_seeds": [],
        "heldout_test_access_status": "not_loaded",
        "evaluation_role": "tuning_validation",
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "learned_baseline_implementation_version": LEARNED_BASELINE_IMPLEMENTATION_VERSION,
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "source_identity_status": source_identity["source_identity_status"],
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "mark_to_market_timing": "post_trade",
        "volume_impact_bps": DEFAULT_VOLUME_IMPACT_BPS,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
        "capacity_target_parameter_count": target_count,
        "capacity_actual_parameter_count": actual_count,
        "capacity_ratio": float(actual_count / max(target_count, 1)),
        "capacity_match_status": str(model_payload.get("capacity_match_status", "")),
        "steps": int(steps),
        "assets": int(assets),
        "iterations": int(iterations),
        "parameter_count": parameter_count,
        "environment_steps_train": int(model_payload.get("environment_steps_train", 0)),
        "environment_steps_checkpoint_validation": int(
            model_payload.get("environment_steps_validation", 0)
        ),
        "environment_steps_tuning_validation": int(
            model_payload.get("environment_steps_eval", 0)
        ),
        "selection_utility_mean": float(np.mean(utilities)),
        "selection_utility_min": float(np.min(utilities)),
        "selection_utility_std": (
            float(np.std(utilities, ddof=1)) if len(utilities) > 1 else 0.0
        ),
        "best_checkpoint_inner_validation_score": float(
            model_payload.get("best_score", 0.0)
        ),
        "initial_checkpoint_validation_score": float(
            model_payload.get("initial_validation_score", 0.0)
        ),
        "validation_learning_gain": float(
            model_payload.get("validation_learning_gain", 0.0)
        ),
        "selected_checkpoint_iteration": int(
            model_payload.get("selected_checkpoint_iteration", -1)
        ),
        "hf_intervention_status": "paired" if hf_rows else "not_applicable",
        "hf_intervention_pair_count": len(hf_rows),
        "hf_action_sensitivity_mean": (
            float(np.mean([float(row["lower_hf_action_sensitivity"]) for row in hf_rows]))
            if hf_rows else 0.0
        ),
        "hf_total_return_delta_mean": (
            float(np.mean([float(row["total_return_delta"]) for row in hf_rows]))
            if hf_rows else 0.0
        ),
        "elapsed_sec": elapsed,
        "cell_status": "valid",
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_payload.get("config", {}),
        "variant_id": variant.variant_id,
        "candidate_id": candidate.candidate_id,
        "candidate_parameters": dict(candidate.parameters),
        "effective_parameters": params,
        "scenario": str(scenario),
        "training_replicate_seed": int(training_replicate_seed),
        "checkpoint_validation_seeds": list(checkpoint_seeds),
        "tuning_validation_seeds": list(tuning_seeds),
        "heldout_test_seeds": [],
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "source_identity_status": source_identity["source_identity_status"],
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
    }
    return {
        "tuning_rows": annotated_rows,
        "hf_intervention_rows": hf_rows,
        "cell_summary": summary,
        "checkpoint": checkpoint,
    }


def write_hpo_cell(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tuning_rows.csv", payload["tuning_rows"])
    _write_csv(
        output_dir / "hf_intervention_rows.csv",
        payload["hf_intervention_rows"],
    )
    with (output_dir / "cell_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["cell_summary"], handle, indent=2, sort_keys=True)
    torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")
    summary = payload["cell_summary"]
    lines = [
        "# Full-Method Nested-Validation HPO Cell",
        "",
        f"- protocol: `{FULL_METHOD_TUNING_PROTOCOL_VERSION}`",
        f"- status: `{summary['cell_status']}`",
        f"- variant: `{summary['variant_id']}`",
        f"- candidate: `{summary['candidate_id']}`",
        f"- scenario: `{summary['scenario']}`",
        f"- training replicate: `{summary['training_replicate_seed']}`",
        f"- tuning utility: `{summary['selection_utility_mean']:.8f}`",
        f"- HF intervention: `{summary['hf_intervention_status']}`",
        f"- held-out test access: `{summary['heldout_test_access_status']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_mean_ci(
    values: Iterable[float], *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _cell_key(summary: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(summary["variant_id"]),
        str(summary["candidate_id"]),
        str(summary["scenario"]),
        int(summary["training_replicate_seed"]),
    )


def _validate_cell_summary(summary: dict[str, Any]) -> None:
    key = _cell_key(summary)
    variant = VARIANTS_BY_ID.get(key[0])
    candidate = CANDIDATES_BY_ID.get(key[1])
    if variant is None or candidate is None or not candidate.applies_to(variant.variant_id):
        raise ValueError(f"invalid full-method HPO cell identity: {key}")
    if summary.get("cell_status") != "valid":
        raise ValueError(f"invalid full-method HPO cell: {key}")
    if summary.get("heldout_test_seeds"):
        raise ValueError(f"HPO cell accessed held-out test seeds: {key}")
    if summary.get("heldout_test_access_status") != "not_loaded":
        raise ValueError(f"HPO cell has invalid held-out access status: {key}")
    expected_versions = {
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "learned_baseline_implementation_version": LEARNED_BASELINE_IMPLEMENTATION_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
    }
    for field, expected in expected_versions.items():
        if summary.get(field) != expected:
            raise ValueError(f"HPO cell {field} mismatch: {key}")
    if float(summary.get("volume_impact_bps", -1.0)) != DEFAULT_VOLUME_IMPACT_BPS:
        raise ValueError(f"HPO cell environment impact mismatch: {key}")
    if summary.get("candidate_parameters") != candidate.parameters:
        raise ValueError(f"HPO cell candidate parameters drifted: {key}")
    if summary.get("effective_parameters") != effective_parameters_for_variant(*key[:2]):
        raise ValueError(f"HPO cell effective parameters drifted: {key}")
    actual = int(summary.get("capacity_actual_parameter_count", 0))
    reported = int(summary.get("parameter_count", -1))
    if actual <= 0 or actual != reported:
        raise ValueError(f"HPO cell active parameter count is invalid: {key}")
    ratio = float(summary.get("capacity_ratio", float("nan")))
    if not np.isfinite(ratio) or abs(ratio - 1.0) > 0.05:
        raise ValueError(f"HPO cell is not capacity matched within 5%: {key}")


def merge_hpo_cells(
    input_dirs: list[Path],
    *,
    expected_variant_ids: list[str] | None = None,
    expected_candidate_ids: list[str] | None = None,
    expected_scenarios: list[str] | None = None,
    expected_replicate_seeds: list[int] | None = None,
    top_k: int = 3,
    stage: str = "pilot",
) -> dict[str, Any]:
    cell_summaries: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    hf_rows: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, str, int]] = set()
    for directory in input_dirs:
        base = Path(directory)
        summary_path = base / "cell_summary.json"
        if not summary_path.exists():
            raise ValueError(f"missing full-method HPO cell summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_cell_summary(summary)
        key = _cell_key(summary)
        if key in seen_cells:
            raise ValueError(f"duplicate full-method HPO cell: {key}")
        seen_cells.add(key)
        cell_tuning_rows = _read_csv(base / "tuning_rows.csv")
        if len(cell_tuning_rows) != len(summary["tuning_validation_seeds"]):
            raise ValueError(f"HPO cell tuning-row count mismatch: {key}")
        if {int(float(row["seed"])) for row in cell_tuning_rows} != {
            int(seed) for seed in summary["tuning_validation_seeds"]
        }:
            raise ValueError(f"HPO cell tuning-seed coverage mismatch: {key}")
        cell_hf_rows = _read_csv(base / "hf_intervention_rows.csv")
        expected_hf = int(summary.get("hf_intervention_pair_count", 0))
        if len(cell_hf_rows) != expected_hf:
            raise ValueError(f"HPO cell HF-row count mismatch: {key}")
        cell_summaries.append(summary)
        tuning_rows.extend(cell_tuning_rows)
        hf_rows.extend(cell_hf_rows)

    variants = list(expected_variant_ids or sorted({key[0] for key in seen_cells}))
    scenarios = list(expected_scenarios or sorted({key[2] for key in seen_cells}))
    replicates = list(expected_replicate_seeds or sorted({key[3] for key in seen_cells}))
    candidates = list(expected_candidate_ids or sorted({key[1] for key in seen_cells}))
    for variant_id in variants:
        if variant_id not in VARIANTS_BY_ID:
            raise ValueError(f"unknown expected full-method variant: {variant_id}")
    for candidate_id in candidates:
        if candidate_id not in CANDIDATES_BY_ID:
            raise ValueError(f"unknown expected full-method candidate: {candidate_id}")
    expected_cells = {
        (variant_id, candidate_id, scenario, int(replicate))
        for variant_id in variants
        for candidate_id in candidates
        if CANDIDATES_BY_ID[candidate_id].applies_to(variant_id)
        for scenario in scenarios
        for replicate in replicates
    }
    missing = sorted(expected_cells - seen_cells)
    unexpected = sorted(seen_cells - expected_cells)
    if missing or unexpected:
        preview = ", ".join(map(str, (missing + unexpected)[:6]))
        raise ValueError(f"incomplete full-method HPO matrix: {preview}")

    matrix_contract_fields = (
        "rollout_seed_roots",
        "checkpoint_validation_seeds",
        "tuning_validation_seeds",
        "steps",
        "assets",
        "iterations",
    )
    for field in matrix_contract_fields:
        values = {json.dumps(summary[field], sort_keys=True) for summary in cell_summaries}
        if len(values) != 1:
            raise ValueError(f"full-method HPO matrix mixes {field}")
    code_revisions = {str(summary.get("code_revision", "")).lower() for summary in cell_summaries}
    source_manifests = {
        str(summary.get("source_manifest_sha256", "")).lower()
        for summary in cell_summaries
    }
    source_statuses = {
        str(summary.get("source_identity_status", "")) for summary in cell_summaries
    }
    if len(code_revisions) != 1 or len(source_manifests) != 1:
        raise ValueError("full-method HPO matrix mixes source identities")
    code_revision = next(iter(code_revisions))
    source_manifest = next(iter(source_manifests))
    source_identity_status = (
        "verified"
        if source_statuses == {"verified"}
        and is_hex_digest(code_revision, length=40)
        and is_hex_digest(source_manifest, length=64)
        else "unregistered_or_incomplete"
    )

    utility_by_cell: dict[tuple[str, str, str, int], list[float]] = {}
    tuning_by_cell: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    for row in tuning_rows:
        if str(row.get("evaluation_role")) != "tuning_validation":
            raise ValueError("full-method HPO merge found a non-tuning row")
        key = (
            str(row["variant_id"]),
            str(row["candidate_id"]),
            str(row["scenario"]),
            int(float(row["training_replicate_seed"])),
        )
        if key not in expected_cells:
            raise ValueError(f"unexpected full-method tuning row: {key}")
        utility_by_cell.setdefault(key, []).append(float(row["selection_utility"]))
        tuning_by_cell.setdefault(key, []).append(row)
    for key in expected_cells:
        if not utility_by_cell.get(key):
            raise ValueError(f"full-method HPO cell has no tuning utilities: {key}")

    hf_by_cell: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    for row in hf_rows:
        if str(row.get("evaluation_role")) != "tuning_validation_mechanism_diagnostic":
            raise ValueError("full-method HPO merge found an invalid HF diagnostic row")
        key = (
            str(row["variant_id"]),
            str(row["candidate_id"]),
            str(row["scenario"]),
            int(float(row["training_replicate_seed"])),
        )
        if key not in expected_cells:
            raise ValueError(f"unexpected full-method HF row: {key}")
        if str(row.get("paired_exogenous_path_identity", "")).lower() not in {"true", "1"}:
            raise ValueError(f"HF intervention is not exogenously paired: {key}")
        hf_by_cell.setdefault(key, []).append(row)

    leaderboard: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant_id in variants:
        variant_rows: list[dict[str, Any]] = []
        applicable = [
            candidate_id for candidate_id in candidates
            if CANDIDATES_BY_ID[candidate_id].applies_to(variant_id)
        ]
        for candidate_id in applicable:
            replicate_scores: list[float] = []
            for replicate in replicates:
                scenario_scores = [
                    float(np.mean(utility_by_cell[(variant_id, candidate_id, scenario, int(replicate))]))
                    for scenario in scenarios
                ]
                replicate_scores.append(float(np.mean(scenario_scores)))
            ci_low, ci_high = _bootstrap_mean_ci(
                replicate_scores,
                seed=scenario_optimizer_seed(int(replicates[0]), scenarios[0]),
            )
            matching_summaries = [
                summary for summary in cell_summaries
                if summary["variant_id"] == variant_id
                and summary["candidate_id"] == candidate_id
            ]
            trained_fraction = float(np.mean([
                int(summary.get("selected_checkpoint_iteration", -1)) >= 0
                for summary in matching_summaries
            ]))
            gain_mean = float(np.mean([
                float(summary.get("validation_learning_gain", 0.0))
                for summary in matching_summaries
            ]))
            learning_status = (
                "eligible" if trained_fraction >= 0.80 and gain_mean > 0.0 else "ineligible"
            )
            candidate_tuning_rows = [
                row
                for key, rows in tuning_by_cell.items()
                if key[0] == variant_id and key[1] == candidate_id
                for row in rows
            ]
            promotion_transition_count = float(np.sum([
                float(row.get("promotion_gate_transition_count", 0.0) or 0.0)
                for row in candidate_tuning_rows
            ]))
            stress_replan_indicators = [
                float(row.get("promotion_replan_count", 0.0) or 0.0) > 0.0
                for row in candidate_tuning_rows
                if str(row.get("scenario")) in {"localized_burst", "persistent_shift", "ood_period"}
            ]
            promotion_replan_fraction = (
                float(np.mean(stress_replan_indicators))
                if stress_replan_indicators else 0.0
            )
            candidate_hf_rows = [
                row
                for key, rows in hf_by_cell.items()
                if key[0] == variant_id and key[1] == candidate_id
                for row in rows
            ]
            hf_sensitivity = (
                float(np.mean([
                    float(row["lower_hf_action_sensitivity"])
                    for row in candidate_hf_rows
                ])) if candidate_hf_rows else 0.0
            )
            if variant_id == ABLATION_PARENT_VARIANT:
                mechanism_status = (
                    "eligible"
                    if promotion_transition_count > 0.0
                    and promotion_replan_fraction > 0.0
                    and hf_sensitivity > 1e-8
                    else "ineligible"
                )
            else:
                mechanism_status = "not_applicable"
            row = {
                "variant_id": variant_id,
                "candidate_id": candidate_id,
                "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
                "scientific_role": VARIANTS_BY_ID[variant_id].scientific_role,
                "independent_training_replicates": len(replicate_scores),
                "scenario_count": len(scenarios),
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
                "learning_gate_status": learning_status,
                "promotion_gate_transition_count": promotion_transition_count,
                "promotion_replan_episode_fraction_stress": promotion_replan_fraction,
                "hf_action_sensitivity_mean": hf_sensitivity,
                "mechanism_activity_status": mechanism_status,
            }
            variant_rows.append(row)
        variant_rows.sort(key=lambda row: (
            0 if row["learning_gate_status"] == "eligible" else 1,
            0 if row["mechanism_activity_status"] in {"eligible", "not_applicable"} else 1,
            -float(row["robust_selection_score"]),
            -float(row["tuning_utility_mean"]),
            str(row["candidate_id"]),
        ))
        for rank, row in enumerate(variant_rows, start=1):
            row["rank"] = rank
        rows_by_variant[variant_id] = variant_rows
        leaderboard.extend(variant_rows)

    selected: dict[str, dict[str, Any]] = {}
    top_candidates: dict[str, list[str]] = {}
    for variant_id in variants:
        variant = VARIANTS_BY_ID[variant_id]
        if variant.inherits_full_selection:
            continue
        ranked = rows_by_variant[variant_id]
        eligible = [
            row for row in ranked
            if row["learning_gate_status"] == "eligible"
            and row["mechanism_activity_status"] in {"eligible", "not_applicable"}
        ]
        pool = eligible if eligible else ranked[:1]
        winners = pool[:max(1, min(int(top_k), len(pool)))]
        top_candidates[variant_id] = [str(row["candidate_id"]) for row in winners]
        winner = winners[0]
        candidate_id = str(winner["candidate_id"])
        selected[variant_id] = {
            "candidate_id": candidate_id,
            "candidate_family": CANDIDATES_BY_ID[candidate_id].family,
            "candidate_parameters": dict(CANDIDATES_BY_ID[candidate_id].parameters),
            "effective_parameters": effective_parameters_for_variant(variant_id, candidate_id),
            "selection_source_variant": variant_id,
            "selection_rule": "validation_cluster_lcb",
            "robust_selection_score": float(winner["robust_selection_score"]),
            "tuning_utility_mean": float(winner["tuning_utility_mean"]),
            "learning_gate_status": str(winner["learning_gate_status"]),
            "mechanism_activity_status": str(winner["mechanism_activity_status"]),
            "trained_checkpoint_fraction": float(winner["trained_checkpoint_fraction"]),
            "validation_learning_gain_mean": float(winner["validation_learning_gain_mean"]),
        }

    if ABLATION_PARENT_VARIANT in selected:
        parent_candidate_id = str(selected[ABLATION_PARENT_VARIANT]["candidate_id"])
        for variant_id in variants:
            variant = VARIANTS_BY_ID[variant_id]
            if not variant.inherits_full_selection:
                continue
            matching = next(
                row for row in rows_by_variant[variant_id]
                if row["candidate_id"] == parent_candidate_id
            )
            top_candidates[variant_id] = [parent_candidate_id]
            selected[variant_id] = {
                "candidate_id": parent_candidate_id,
                "candidate_family": CANDIDATES_BY_ID[parent_candidate_id].family,
                "candidate_parameters": dict(CANDIDATES_BY_ID[parent_candidate_id].parameters),
                "effective_parameters": effective_parameters_for_variant(
                    variant_id, parent_candidate_id
                ),
                "selection_source_variant": ABLATION_PARENT_VARIANT,
                "selection_rule": "inherited_full_method_one_factor_ablation",
                "robust_selection_score": float(matching["robust_selection_score"]),
                "tuning_utility_mean": float(matching["tuning_utility_mean"]),
                "learning_gate_status": str(matching["learning_gate_status"]),
                "mechanism_activity_status": "not_applicable",
                "trained_checkpoint_fraction": float(matching["trained_checkpoint_fraction"]),
                "validation_learning_gain_mean": float(
                    matching["validation_learning_gain_mean"]
                ),
            }

    candidate_counts = {
        variant_id: len(rows_by_variant[variant_id]) for variant_id in variants
    }
    equal_search_budget = len(set(candidate_counts.values())) == 1
    all_selected_eligible = bool(selected) and all(
        entry["learning_gate_status"] == "eligible" for entry in selected.values()
    )
    full_mechanism_eligible = (
        selected.get(ABLATION_PARENT_VARIANT, {}).get("mechanism_activity_status")
        == "eligible"
    )
    final_design_complete = (
        set(variants) == set(ALL_VARIANT_IDS)
        and set(scenarios) == set(DEFAULT_SCENARIOS)
        and len(replicates) >= 5
        and equal_search_budget
        and min(candidate_counts.values(), default=0) >= 2
        and source_identity_status == "verified"
    )
    if (
        str(stage) == "final"
        and final_design_complete
        and all_selected_eligible
        and full_mechanism_eligible
    ):
        freeze_status = "frozen_from_validation_only"
    elif not all_selected_eligible:
        freeze_status = "not_freezable_learning_gate"
    elif not full_mechanism_eligible:
        freeze_status = "not_freezable_mechanism_activity_gate"
    elif str(stage) == "final":
        freeze_status = "not_freezable_incomplete_final_design"
    else:
        freeze_status = "provisional_validation_only"
    first_summary = cell_summaries[0]
    freeze = {
        "status": freeze_status,
        "stage": str(stage),
        "tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "learned_baseline_implementation_version": LEARNED_BASELINE_IMPLEMENTATION_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "source_identity_status": source_identity_status,
        "code_revision": code_revision,
        "source_manifest_sha256": source_manifest,
        "heldout_test_access_status": "not_loaded",
        "heldout_test_seeds": [],
        "rollout_seed_roots": list(first_summary["rollout_seed_roots"]),
        "checkpoint_validation_seeds": list(first_summary["checkpoint_validation_seeds"]),
        "tuning_validation_seeds": list(first_summary["tuning_validation_seeds"]),
        "scenarios": list(scenarios),
        "training_replicate_seeds": [int(seed) for seed in replicates],
        "search_budget_candidates_per_variant": candidate_counts,
        "equal_search_budget": bool(equal_search_budget),
        "ablation_selection_rule": (
            "inherit_full_candidate_then_disable_exactly_one_registered_mechanism"
        ),
        "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        "capacity_reference_method_contract": CAPACITY_REFERENCE_METHOD_CONTRACT,
        "volume_impact_bps": DEFAULT_VOLUME_IMPACT_BPS,
        "final_design_complete": bool(final_design_complete),
        "selected": selected,
        "top_candidates": top_candidates,
    }
    return {
        "cell_summaries": cell_summaries,
        "tuning_rows": tuning_rows,
        "hf_intervention_rows": hf_rows,
        "leaderboard": leaderboard,
        "frozen_config": freeze,
        "summary": {
            "stage": str(stage),
            "cell_count": len(cell_summaries),
            "expected_cell_count": len(expected_cells),
            "matrix_coverage_status": "complete",
            "variant_count": len(variants),
            "scenario_count": len(scenarios),
            "training_replicate_count": len(replicates),
            "heldout_test_access_count": 0,
            "source_identity_status": source_identity_status,
            "code_revision": code_revision,
            "source_manifest_sha256": source_manifest,
            "equal_search_budget_status": (
                "supported" if equal_search_budget else "not_supported"
            ),
            "learning_gate_status": (
                "supported" if all_selected_eligible else "not_supported"
            ),
            "mechanism_activity_status": (
                "supported" if full_mechanism_eligible else "not_supported"
            ),
            "final_design_status": (
                "complete" if final_design_complete else "incomplete"
            ),
        },
    }


def write_hpo_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "leaderboard.csv", payload["leaderboard"])
    _write_csv(output_dir / "cell_summaries.csv", payload["cell_summaries"])
    _write_csv(
        output_dir / "hf_intervention_rows.csv",
        payload["hf_intervention_rows"],
    )
    with (output_dir / "frozen_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["frozen_config"], handle, indent=2, sort_keys=True)
    serializable = {
        key: value
        for key, value in payload.items()
        if key not in {"tuning_rows", "hf_intervention_rows"}
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
    lines = [
        "# Full-Method Nested-Validation HPO",
        "",
        f"- protocol: `{FULL_METHOD_TUNING_PROTOCOL_VERSION}`",
        f"- coverage: `{payload['summary']['matrix_coverage_status']}`",
        f"- equal search budget: `{payload['summary']['equal_search_budget_status']}`",
        f"- mechanism activity: `{payload['summary']['mechanism_activity_status']}`",
        f"- held-out test accesses: `{payload['summary']['heldout_test_access_count']}`",
        "",
        "| variant | candidate | rank | robust score | utility | learning | mechanism |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["leaderboard"]:
        lines.append(
            f"| {row['variant_id']} | {row['candidate_id']} | {row['rank']} "
            f"| {float(row['robust_selection_score']):+.8f} "
            f"| {float(row['tuning_utility_mean']):+.8f} "
            f"| {row['learning_gate_status']} | {row['mechanism_activity_status']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", choices=sorted(CANDIDATES_BY_ID))
    parser.add_argument("--variant-id", choices=ALL_VARIANT_IDS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument(
        "--train-seeds", type=int, nargs="+", default=list(DEFAULT_ROLLOUT_SEED_ROOTS)
    )
    parser.add_argument(
        "--checkpoint-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument(
        "--tuning-validation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TUNING_SEEDS),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--source-manifest-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument("--expected-variant-ids", nargs="*", choices=ALL_VARIANT_IDS)
    parser.add_argument(
        "--expected-candidate-ids", nargs="*", choices=sorted(CANDIDATES_BY_ID)
    )
    parser.add_argument("--expected-scenarios", nargs="*", choices=SCENARIOS)
    parser.add_argument("--expected-replicate-seeds", type=int, nargs="*")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_hpo_cells(
            list(args.merge_inputs),
            expected_variant_ids=list(args.expected_variant_ids or []),
            expected_candidate_ids=list(args.expected_candidate_ids or []),
            expected_scenarios=list(args.expected_scenarios or []),
            expected_replicate_seeds=list(args.expected_replicate_seeds or []),
            top_k=int(args.top_k),
            stage=str(args.stage),
        )
        write_hpo_merge(args.output_dir, payload)
        print(
            f"full_method_hpo_merge cells={payload['summary']['cell_count']} "
            f"status={payload['frozen_config']['status']} output={args.output_dir}"
        )
        return
    required = {
        "candidate_id": args.candidate_id,
        "variant_id": args.variant_id,
        "scenario": args.scenario,
        "training_replicate_seed": args.training_replicate_seed,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        parser.error(
            "cell mode requires "
            + ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        )
    payload = run_hpo_cell(
        candidate_id=str(args.candidate_id),
        variant_id=str(args.variant_id),
        scenario=str(args.scenario),
        training_replicate_seed=int(args.training_replicate_seed),
        train_seeds=list(args.train_seeds),
        checkpoint_validation_seeds=list(args.checkpoint_validation_seeds),
        tuning_validation_seeds=list(args.tuning_validation_seeds),
        steps=int(args.steps),
        assets=int(args.assets),
        iterations=int(args.iterations),
        code_revision=str(args.code_revision),
        expected_source_manifest_sha256=str(args.source_manifest_sha256),
    )
    write_hpo_cell(args.output_dir, payload)
    print(
        f"full_method_hpo_cell status=valid variant={args.variant_id} "
        f"candidate={args.candidate_id} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
