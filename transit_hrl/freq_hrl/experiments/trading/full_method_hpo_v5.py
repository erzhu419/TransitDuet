"""Versioned nested-validation HPO for independent-HF Freq-HRL v5.

The v4 HPO module remains immutable for reproducing its registered pilot. This
module reuses its audited cell/merge/freeze machinery under a scoped registry
override, while supplying v5 variants, capacity accounting, interventions, and
independently tunable policy-level learning rates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from freq_hrl.rl import SMDPPPOConfig

from . import full_method_hpo as _v4
from .metrics import DEFAULT_TRAINING_REWARD_SCALE
from .performance_validation import SCENARIOS
from .ppo_actor_critic import (
    FULL_METHOD_V5_IMPLEMENTATION_VERSION,
    make_plan_mapper,
    promotion_gate_state_dim,
    resolve_method_contract,
    smdp_parameter_count,
)
from .strong_learned_baseline_validation import (
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_VALIDATION_SEEDS,
)


FullMethodVariant = _v4.FullMethodVariant
FullMethodCandidate = _v4.FullMethodCandidate

FULL_METHOD_TUNING_PROTOCOL_VERSION = "full_method_nested_hpo_v3"
FULL_METHOD_HPO_IMPLEMENTATION_VERSION = (
    "full_method_hpo_independent_hf_credit_v3_2026_08_03"
)
FULL_METHOD_IMPLEMENTATION_VERSION = FULL_METHOD_V5_IMPLEMENTATION_VERSION
DEFAULT_TUNING_SEEDS = _v4.DEFAULT_TUNING_SEEDS
DEFAULT_PILOT_SCENARIOS = _v4.DEFAULT_PILOT_SCENARIOS
EXECUTION_TIMELINE_CONTRACT = _v4.EXECUTION_TIMELINE_CONTRACT
CAPACITY_REFERENCE_METHOD_CONTRACT = "full_freq_hrl_v5"
DEFAULT_VOLUME_IMPACT_BPS = _v4.DEFAULT_VOLUME_IMPACT_BPS
DEFAULT_PLAN_BASIS_DIM = _v4.DEFAULT_PLAN_BASIS_DIM
DEFAULT_PLAN_HORIZON_S = _v4.DEFAULT_PLAN_HORIZON_S
DEFAULT_PLAN_EVAL_OFFSET_S = _v4.DEFAULT_PLAN_EVAL_OFFSET_S
DEFAULT_PLAN_COEFFICIENT_SCALE = _v4.DEFAULT_PLAN_COEFFICIENT_SCALE
ABLATION_PARENT_VARIANT = "freq_hrl_full_v5"


VARIANTS = (
    FullMethodVariant(
        "freq_hrl_full_v5",
        "ppo",
        "frequency_ppo_v5",
        "freq_hrl",
        "full_freq_hrl_v5",
        "proposed_full_method",
    ),
    FullMethodVariant(
        "freq_hrl_no_promotion_v5",
        "ppo",
        "frequency_ppo_v5",
        "freq_hrl",
        "ablate_promotion_v5",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "freq_hrl_no_hf_lower_v5",
        "ppo",
        "frequency_ppo_v5",
        "freq_hrl",
        "ablate_hf_lower_v5",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "freq_hrl_no_leakage_v5",
        "ppo",
        "frequency_ppo_v5",
        "freq_hrl",
        "ablate_leakage_v5",
        "one_factor_ablation",
        ABLATION_PARENT_VARIANT,
    ),
    FullMethodVariant(
        "flat_ppo_matched_v5",
        "ppo",
        "baseline_ppo",
        "flat_ppo",
        "routing_core_v2",
        "capacity_matched_flat_baseline",
    ),
    FullMethodVariant(
        "flat_gru_ppo_matched_v5",
        "ppo",
        "baseline_ppo",
        "flat_gru_ppo",
        "routing_core_v2",
        "capacity_matched_flat_recurrent_baseline",
    ),
    FullMethodVariant(
        "generic_hrl_ppo_matched_v5",
        "ppo",
        "baseline_ppo",
        "generic_hrl_ppo",
        "curve_credit_control_v3",
        "capacity_matched_nonfrequency_hrl_baseline",
    ),
    FullMethodVariant(
        "generic_hrl_gru_ppo_matched_v5",
        "ppo",
        "baseline_ppo",
        "generic_hrl_gru_ppo",
        "curve_credit_control_v3",
        "capacity_matched_nonfrequency_recurrent_hrl_baseline",
    ),
    FullMethodVariant(
        "flat_sac_matched_v5",
        "offpolicy",
        "offpolicy",
        "flat_sac",
        "routing_core_v2",
        "capacity_matched_offpolicy_baseline",
    ),
    FullMethodVariant(
        "flat_td3_matched_v5",
        "offpolicy",
        "offpolicy",
        "flat_td3",
        "routing_core_v2",
        "capacity_matched_offpolicy_baseline",
    ),
)
VARIANTS_BY_ID = {variant.variant_id: variant for variant in VARIANTS}
ALL_VARIANT_IDS = tuple(variant.variant_id for variant in VARIANTS)


def _optimizer_parameters(
    learning_rate: float,
    init_log_std: float,
) -> dict[str, Any]:
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
    constraint_target: float,
    dual_lr: float,
    objective_weight: float,
    promotion_init_logit: float,
    promotion_replan_cost: float,
    lower_hf_order_scale: float,
    upper_period: int,
    min_upper_duration: int,
) -> FullMethodCandidate:
    return FullMethodCandidate(
        candidate_id,
        "frequency_ppo_v5",
        {
            **_optimizer_parameters(lower_lr, init_log_std),
            "upper_learning_rate": float(upper_lr),
            "lower_learning_rate": float(lower_lr),
            "hf_learning_rate": float(hf_lr),
            "promotion_learning_rate": float(promotion_lr),
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
            "upper_period": int(upper_period),
            "min_upper_duration": int(min_upper_duration),
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
        "v5_u3_l3_h003_p003_conservative",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=3e-5, promotion_lr=3e-5,
        init_log_std=-1.5, leakage_scale=2.5e-4,
        constraint_init=0.02, constraint_target=0.20, dual_lr=5e-4,
        objective_weight=2.5e-4, promotion_init_logit=-1.5,
        promotion_replan_cost=5e-5, lower_hf_order_scale=0.0075,
        upper_period=45, min_upper_duration=15,
    ),
    _frequency_candidate(
        "v5_u3_l3_h01_p005_balanced",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=1e-4, promotion_lr=5e-5,
        init_log_std=-1.0, leakage_scale=5e-4,
        constraint_init=0.05, constraint_target=0.15, dual_lr=5e-4,
        objective_weight=5e-4, promotion_init_logit=-1.0,
        promotion_replan_cost=7.5e-5, lower_hf_order_scale=0.010,
        upper_period=45, min_upper_duration=10,
    ),
    _frequency_candidate(
        "v5_u1_l3_h005_p01_tracking",
        upper_lr=1e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=1e-4,
        init_log_std=-1.0, leakage_scale=5e-4,
        constraint_init=0.05, constraint_target=0.10, dual_lr=1e-3,
        objective_weight=5e-4, promotion_init_logit=-0.75,
        promotion_replan_cost=1e-4, lower_hf_order_scale=0.010,
        upper_period=30, min_upper_duration=10,
    ),
    _frequency_candidate(
        "v5_u5_l3_h005_p01_macro",
        upper_lr=5e-4, lower_lr=3e-4, hf_lr=5e-5, promotion_lr=1e-4,
        init_log_std=-1.0, leakage_scale=7.5e-4,
        constraint_init=0.10, constraint_target=0.10, dual_lr=1e-3,
        objective_weight=7.5e-4, promotion_init_logit=-0.5,
        promotion_replan_cost=1.5e-4, lower_hf_order_scale=0.010,
        upper_period=30, min_upper_duration=10,
    ),
    _frequency_candidate(
        "v5_u3_l1_h3_p01_tactical",
        upper_lr=3e-4, lower_lr=1e-4, hf_lr=3e-4, promotion_lr=1e-4,
        init_log_std=-0.75, leakage_scale=7.5e-4,
        constraint_init=0.10, constraint_target=0.075, dual_lr=2e-3,
        objective_weight=7.5e-4, promotion_init_logit=-0.25,
        promotion_replan_cost=2e-4, lower_hf_order_scale=0.015,
        upper_period=30, min_upper_duration=10,
    ),
    _frequency_candidate(
        "v5_u3_l3_h1_p3_activegate",
        upper_lr=3e-4, lower_lr=3e-4, hf_lr=1e-4, promotion_lr=3e-4,
        init_log_std=-0.75, leakage_scale=1e-3,
        constraint_init=0.10, constraint_target=0.075, dual_lr=2e-3,
        objective_weight=1e-3, promotion_init_logit=0.0,
        promotion_replan_cost=2.5e-4, lower_hf_order_scale=0.0125,
        upper_period=45, min_upper_duration=15,
    ),
    _baseline_ppo_candidate("ppo_lr1e4_std15", 1e-4, -1.5),
    _baseline_ppo_candidate("ppo_lr1e4_std10", 1e-4, -1.0),
    _baseline_ppo_candidate("ppo_lr3e4_std15", 3e-4, -1.5),
    _baseline_ppo_candidate("ppo_lr3e4_std10", 3e-4, -1.0),
    _baseline_ppo_candidate("ppo_lr1e4_std05", 1e-4, -0.5),
    _baseline_ppo_candidate("ppo_lr3e4_std05", 3e-4, -0.5),
    _offpolicy_candidate(
        "off_lr1e4_w1024_b64", learning_rate=1e-4,
        warmup_steps=1024, batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr1e4_w2048_b64", learning_rate=1e-4,
        warmup_steps=2048, batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w1024_b64", learning_rate=3e-4,
        warmup_steps=1024, batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w2048_b64", learning_rate=3e-4,
        warmup_steps=2048, batch_size=64,
    ),
    _offpolicy_candidate(
        "off_lr3e4_w2048_b128", learning_rate=3e-4,
        warmup_steps=2048, batch_size=128,
    ),
    _offpolicy_candidate(
        "off_lr1e3_w4096_b64", learning_rate=1e-3,
        warmup_steps=4096, batch_size=64,
    ),
)
CANDIDATES_BY_ID = {
    candidate.candidate_id: candidate for candidate in FULL_METHOD_CANDIDATES
}


_LEGACY_EFFECTIVE_PARAMETERS = _v4.effective_parameters_for_variant
_LEGACY_PPO_KWARGS = _v4._ppo_training_kwargs
_LEGACY_HF_KWARGS = _v4._hf_intervention_kwargs


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
        lower_action_dim=int(assets),
        hf_state_dim=8 * int(assets) + 1,
        hf_action_dim=int(assets),
        promotion_state_dim=promotion_gate_state_dim(int(assets)),
        hidden_dim=int(hidden_dim),
    )
    return smdp_parameter_count(config)


_REGISTRY_PATCH = {
    "FULL_METHOD_TUNING_PROTOCOL_VERSION": FULL_METHOD_TUNING_PROTOCOL_VERSION,
    "FULL_METHOD_HPO_IMPLEMENTATION_VERSION": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
    "FULL_METHOD_IMPLEMENTATION_VERSION": FULL_METHOD_IMPLEMENTATION_VERSION,
    "CAPACITY_REFERENCE_METHOD_CONTRACT": CAPACITY_REFERENCE_METHOD_CONTRACT,
    "ABLATION_PARENT_VARIANT": ABLATION_PARENT_VARIANT,
    "VARIANTS": VARIANTS,
    "VARIANTS_BY_ID": VARIANTS_BY_ID,
    "ALL_VARIANT_IDS": ALL_VARIANT_IDS,
    "FULL_METHOD_CANDIDATES": FULL_METHOD_CANDIDATES,
    "CANDIDATES_BY_ID": CANDIDATES_BY_ID,
}


@contextmanager
def _activated() -> Iterator[None]:
    patch = {
        **_REGISTRY_PATCH,
        "canonical_full_method_parameter_count": (
            canonical_full_method_parameter_count
        ),
        "effective_parameters_for_variant": effective_parameters_for_variant,
        "_ppo_training_kwargs": _ppo_training_kwargs,
        "_hf_intervention_kwargs": _hf_intervention_kwargs,
    }
    previous = {name: getattr(_v4, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(_v4, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(_v4, name, value)


def candidate_ids_for_variant(variant_id: str) -> list[str]:
    with _activated():
        return _v4.candidate_ids_for_variant(variant_id)


def effective_parameters_for_variant(
    variant_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    with _activated():
        params = _LEGACY_EFFECTIVE_PARAMETERS(variant_id, candidate_id)
    variant = VARIANTS_BY_ID[str(variant_id)]
    candidate = CANDIDATES_BY_ID[str(candidate_id)]
    for key in ("upper_period", "min_upper_duration"):
        if key in candidate.parameters:
            params[key] = candidate.parameters[key]
    if variant.method_contract == "ablate_promotion_v5":
        params["promotion_replan_cost"] = 0.0
    elif variant.method_contract == "ablate_hf_lower_v5":
        params["lower_hf_order_scale"] = 0.0
    elif variant.method_contract == "ablate_leakage_v5":
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


def _ppo_training_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    result = _LEGACY_PPO_KWARGS(params)
    for key in (
        "upper_learning_rate",
        "lower_learning_rate",
        "hf_learning_rate",
        "promotion_learning_rate",
    ):
        if key in params:
            result[key] = params[key]
    return result


def _hf_intervention_kwargs(
    *,
    params: dict[str, Any],
    steps: int,
    assets: int,
    scenario: str,
) -> dict[str, Any]:
    result = _LEGACY_HF_KWARGS(
        params=params,
        steps=steps,
        assets=assets,
        scenario=scenario,
    )
    flags = resolve_method_contract(str(params["method_contract"]))
    result["separate_hf_tactical"] = bool(flags["separate_hf_tactical"])
    return result


def run_hpo_cell(**kwargs: Any) -> dict[str, Any]:
    with _activated():
        return _v4.run_hpo_cell(**kwargs)


def write_hpo_cell(output_dir: Path, payload: dict[str, Any]) -> None:
    with _activated():
        _v4.write_hpo_cell(output_dir, payload)


def merge_hpo_cells(*args: Any, **kwargs: Any) -> dict[str, Any]:
    with _activated():
        return _v4.merge_hpo_cells(*args, **kwargs)


def write_hpo_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    with _activated():
        _v4.write_hpo_merge(output_dir, payload)


def frozen_config_sha256(payload: dict[str, Any]) -> str:
    return _v4.frozen_config_sha256(payload)


def validate_frozen_config(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    with _activated():
        return _v4.validate_frozen_config(payload, **kwargs)


def load_frozen_config(path: Path, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    with _activated():
        return _v4.load_frozen_config(path, **kwargs)


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
            f"full_method_hpo_v5_merge cells={payload['summary']['cell_count']} "
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
        f"full_method_hpo_v5_cell status=valid variant={args.variant_id} "
        f"candidate={args.candidate_id} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
