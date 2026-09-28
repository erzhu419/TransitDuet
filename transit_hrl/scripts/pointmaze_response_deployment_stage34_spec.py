"""Frozen full-episode transfer of the qualified Stage-33 linear response."""

from pathlib import Path

from scripts import pointmaze_root_response_stage33_spec as source
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PROTOCOL = "pointmaze_response_deployment_stage34_v1"
POLICY = source.POLICY
OPTIMIZER_ROOTS = source.OPTIMIZER_ROOTS
PREFLIGHT_ROOTS = source.PREFLIGHT_ROOTS
RUNNER_SCRIPT = "scripts/run_pointmaze_response_deployment_stage34.py"
SOURCE_RUN = "pointmaze_root_response_stage33_v1_qualification_20260928_r1"
PREFLIGHT_SOURCE_RUN = "pointmaze_root_response_stage33_v1_preflight_20260928_r1"
CONTROLS = (*response.METHODS[1:], "always_keep", "always_renew")
METHODS = ("history", *CONTROLS, "fixed50")
ENDPOINTS = tuple(f"{metric}:{control}" for metric in ("ise", "return") for control in CONTROLS)
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = (34, 34039)


def roots(*, preflight):
    return PREFLIGHT_ROOTS if preflight else OPTIMIZER_ROOTS


def evaluation_paths(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-34 root")
    base = 4_590_000 if preflight else 4_600_000 + roster.index(root) * 10000
    return list(range(base + 1, base + (3 if preflight else 33)))


def source_result(root, *, preflight):
    run = PREFLIGHT_SOURCE_RUN if preflight else SOURCE_RUN
    return ROOT / "results" / run / "cells" / POLICY / f"replicate_{root}" / "result.json"


def checks(horizon):
    return tuple(range(100, horizon - hold.SETTLEMENT_STEPS + 1, hold.SETTLEMENT_STEPS))


def budget(root, *, preflight):
    horizon = source.arguments(root, preflight=preflight).horizon
    paths = evaluation_paths(root, preflight=preflight)
    return {"evaluation_episodes": len(paths) * len(METHODS),
            "evaluation_primitive_steps": len(paths) * len(METHODS) * horizon,
            "factual_replay_primitive_steps": horizon,
            "total_primitive_steps": (len(paths) * len(METHODS) + 1) * horizon,
            "controller_updates": 0, "motion_updates": 0, "response_fits": 0}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL,
            "methods": list(METHODS), "decision": "settled_rate_strictly_positive",
            "warmup_steps": 100, "hold_steps": hold.HOLD_STEPS,
            "block_steps": hold.SETTLEMENT_STEPS,
            "nonoverlapping_blocks": True, "lower_feedback": "every_primitive_step",
            "delayed_plan": "fresh_policy_at_current_state_not_cached_preview",
            "candidate_preview": "actual_policy_call_reused_only_if_immediately_executed",
            "tail": "fixed50_from_last_settlement",
            "statistical_unit": "optimizer_seed_root", "root_weighting": "equal",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": list(BOOTSTRAP_SEED), "familywise_alpha": .05,
            "interval": "two_sided_percentile_bonferroni_16_endpoints",
            "gate": "all_16_adjusted_lower_bounds_strictly_positive",
            "fixed50_role": "original_frequency_reference_not_joint_gate",
            "cost_role": "measured_not_equalized_with_unused_calls",
            "timing_role": "descriptive_not_a_speedup_test",
            "evidence_role": "fresh_path_transfer_conditional_on_stage33_frozen_models",
            "root_exclusion": "forbidden", "sequential_extension": "forbidden"}
