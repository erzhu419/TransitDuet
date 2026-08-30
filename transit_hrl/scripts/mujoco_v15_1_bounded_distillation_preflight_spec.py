"""Fresh-root development protocol for saturation-bounded distillation."""

from __future__ import annotations

from scripts import mujoco_v14_29_fresh_anchor_spec as anchors
from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as previous


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v15_1_saturation_bounded_raw_policy_preflight_v1"
)
EVIDENCE_ROLE = (
    "post_v15_failure_saturation_bounded_distillation_development_"
    "not_confirmatory"
)
FROZEN_ALGORITHM_REVISION = "d80b364053f0c205be04131390ea352bda3afe00"
PROBE_VERSION = "mujoco_saturation_bounded_raw_policy_distillation_probe_v1"
ANCHOR_RUN_NAME = previous.ANCHOR_RUN_NAME
ENVIRONMENTS = previous.ENVIRONMENTS
OPTIMIZER_SEEDS = previous.OPTIMIZER_SEEDS
DISTILL_ROOTS = (2236520307, 2734605137, 2604001809, 3269003295)
DESIGN_ROOTS = (
    591125992,
    910715664,
    1798492165,
    578352522,
    3765638021,
    124728730,
    3738944975,
    341553073,
    1465039579,
    3512968005,
)
VALIDATION_ROOTS = (
    2188163909,
    587279417,
    4189486343,
    3263497054,
    567678356,
    67531923,
    2420049904,
    2142345880,
    2006603077,
    2442456859,
)
DISTURBANCE_MODES = previous.DISTURBANCE_MODES
DESIGN_FOLD_COUNT = 2
EPISODE_HORIZON = previous.EPISODE_HORIZON
LEAKAGE_COST_MODE = previous.LEAKAGE_COST_MODE
RISK_MODE = previous.RISK_MODE
CVAR_ALPHA = previous.CVAR_ALPHA
MINIMUM_MERIT_REDUCTION = previous.MINIMUM_MERIT_REDUCTION
FUNNEL_MULTIPLIER = previous.FUNNEL_MULTIPLIER
RIDGE = 1e-2
BLEND = 1.0
ROUTER_STRENGTHS = (0.5, 0.75, 1.0)
WORKERS = 108
CPU_PER_TASK = 108
RAM_MB_PER_TASK = 49152
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)
ANALYSIS_VERSION = "mujoco_v15_1_bounded_distillation_analysis_v1"
SUPPORTED_ANALYSIS_STATUS = (
    "bounded_raw_policy_preflight_supported_all_environments"
)
NOT_SUPPORTED_ANALYSIS_STATUS = "bounded_raw_policy_preflight_not_supported"
ANALYSIS_JSON_NAME = "bounded_raw_policy_preflight.json"
ANALYSIS_CSV_NAME = "bounded_raw_policy_cells.csv"

CANDIDATES = tuple(
    {
        "slow_alpha": slow_alpha,
        "transfer_strength": transfer_strength,
        "blend": BLEND,
        "ridge": RIDGE,
        "raw_target_limit": raw_target_limit,
        "head_delta_rms_limit": head_delta_rms_limit,
        "router_strength": router_strength,
    }
    for slow_alpha in (0.5, 0.75, 1.0)
    for transfer_strength in (0.75, 1.0)
    for raw_target_limit in (2.5, 3.5)
    for head_delta_rms_limit in (0.02, 0.05, 0.1)
    for router_strength in ROUTER_STRENGTHS
)

SELECTION_CONTRACT = {
    "teacher": (
        "strictly_causal_previous_macro_total_action_ema_with_exact_"
        "feasibility_projection_and_bounded_inverse_tanh_targets"
    ),
    "student": (
        "upper_then_counterfactual_lower_actor_head_ridge_fit_with_"
        "parameter_rms_trust_region_followed_by_function_preserving_"
        "responsibility_router_selection"
    ),
    "design": (
        "pooled_and_two_disjoint_five_root_folds_require_reward_floor_"
        "and_zero_normalized_violation_on_all_five_frequency_endpoints"
    ),
    "validation": "one_evaluation_of_the_design_selected_candidate_only",
    "failure": "no_design_candidate_or_validation_failure_is_an_abstention",
}


def validate() -> None:
    roles = (DISTILL_ROOTS, DESIGN_ROOTS, VALIDATION_ROOTS)
    flattened = tuple(root for role in roles for root in role)
    consumed = set(
        anchors.OPTIMIZER_SEEDS
        + anchors.PRETRAIN_SEEDS
        + anchors.PRETRAIN_SELECTION_SEEDS
        + anchors.DEVELOPMENT_EVALUATION_SEEDS
        + previous.DISTILL_ROOTS
        + previous.DESIGN_ROOTS
        + previous.VALIDATION_ROOTS
    )
    if len(flattened) != 24 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v15.1 requires twenty-four unique fresh roots")
    if consumed & set(flattened):
        raise RuntimeError("v15.1 roots overlap prior optimizer or trajectory roles")
    if len(DESIGN_ROOTS) % DESIGN_FOLD_COUNT:
        raise RuntimeError("v15.1 design roots must divide across folds")
    if len(CANDIDATES) != 108 or len({
        tuple(sorted(candidate.items())) for candidate in CANDIDATES
    }) != len(CANDIDATES):
        raise RuntimeError("v15.1 requires one hundred eight unique candidates")
    if WORKERS != len(CANDIDATES) or CPU_PER_TASK != WORKERS:
        raise RuntimeError("v15.1 allocates one physical core per candidate")


validate()
