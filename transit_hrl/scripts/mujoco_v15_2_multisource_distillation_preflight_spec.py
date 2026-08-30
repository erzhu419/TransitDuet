"""Fresh-root multi-source causal responsibility distillation protocol."""

from __future__ import annotations

from scripts import mujoco_v14_29_fresh_anchor_spec as anchors
from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as v15
from scripts import mujoco_v15_1_bounded_distillation_preflight_spec as previous


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v15_2_multisource_raw_policy_preflight_v1"
)
EVIDENCE_ROLE = (
    "post_v15_1_failure_multisource_causal_teacher_development_not_confirmatory"
)
FROZEN_ALGORITHM_REVISION = "68cdc2aad1ad49d81165658809b7300273c0d137"
PROBE_VERSION = "mujoco_multisource_raw_policy_distillation_probe_v1"
ANCHOR_RUN_NAME = previous.ANCHOR_RUN_NAME
ENVIRONMENTS = previous.ENVIRONMENTS
OPTIMIZER_SEEDS = previous.OPTIMIZER_SEEDS
DISTILL_ROOTS = (2426144706, 1931411560, 3503818430, 536001399)
DESIGN_ROOTS = (
    3611963113,
    257078613,
    309473628,
    2559248157,
    2751398596,
    3603222630,
    404538985,
    2619333497,
    3123932549,
    3913178714,
)
VALIDATION_ROOTS = (
    503974763,
    3135342327,
    907928770,
    2236194949,
    1290650001,
    1948301590,
    378309775,
    2882886999,
    2800616586,
    3324661803,
)
DISTURBANCE_MODES = previous.DISTURBANCE_MODES
DESIGN_FOLD_COUNT = previous.DESIGN_FOLD_COUNT
EPISODE_HORIZON = previous.EPISODE_HORIZON
LEAKAGE_COST_MODE = previous.LEAKAGE_COST_MODE
RISK_MODE = previous.RISK_MODE
CVAR_ALPHA = previous.CVAR_ALPHA
MINIMUM_MERIT_REDUCTION = previous.MINIMUM_MERIT_REDUCTION
FUNNEL_MULTIPLIER = previous.FUNNEL_MULTIPLIER
RIDGE = previous.RIDGE
BLEND = previous.BLEND
ROUTER_STRENGTHS = previous.ROUTER_STRENGTHS
SLOW_SOURCES = ("total_action", "upper_action")
WORKERS = 108
CPU_PER_TASK = 108
RAM_MB_PER_TASK = 81920
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)
ANALYSIS_VERSION = "mujoco_v15_2_multisource_distillation_analysis_v1"
SUPPORTED_ANALYSIS_STATUS = (
    "multisource_raw_policy_preflight_supported_all_environments"
)
NOT_SUPPORTED_ANALYSIS_STATUS = "multisource_raw_policy_preflight_not_supported"
ANALYSIS_JSON_NAME = "multisource_raw_policy_preflight.json"
ANALYSIS_CSV_NAME = "multisource_raw_policy_cells.csv"

CANDIDATES = tuple(
    {
        "slow_alpha": slow_alpha,
        "transfer_strength": transfer_strength,
        "blend": BLEND,
        "ridge": RIDGE,
        "raw_target_limit": raw_target_limit,
        "head_delta_rms_limit": head_delta_rms_limit,
        "router_strength": router_strength,
        "slow_source": slow_source,
    }
    for slow_source in SLOW_SOURCES
    for slow_alpha in (0.5, 0.75, 1.0)
    for transfer_strength in (0.75, 1.0)
    for raw_target_limit in (2.5, 3.5)
    for head_delta_rms_limit in (0.02, 0.05, 0.1)
    for router_strength in ROUTER_STRENGTHS
)

SELECTION_CONTRACT = {
    "teacher": (
        "design_selects_between_strictly_causal_completed_macro_total_"
        "and_upper_action_ema_targets_with_exact_feasibility_projection"
    ),
    "student": (
        "bounded_logit_upper_then_counterfactual_lower_actor_head_fit_"
        "with_parameter_rms_trust_region_and_function_preserving_router"
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
        + v15.DISTILL_ROOTS
        + v15.DESIGN_ROOTS
        + v15.VALIDATION_ROOTS
        + previous.DISTILL_ROOTS
        + previous.DESIGN_ROOTS
        + previous.VALIDATION_ROOTS
    )
    if len(flattened) != 24 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v15.2 requires twenty-four unique fresh roots")
    if consumed & set(flattened):
        raise RuntimeError("v15.2 roots overlap prior optimizer or trajectory roles")
    if len(DESIGN_ROOTS) % DESIGN_FOLD_COUNT:
        raise RuntimeError("v15.2 design roots must divide across folds")
    if len(CANDIDATES) != 216 or len({
        tuple(sorted(candidate.items())) for candidate in CANDIDATES
    }) != len(CANDIDATES):
        raise RuntimeError("v15.2 requires two hundred sixteen unique candidates")
    if WORKERS > len(CANDIDATES) or CPU_PER_TASK != WORKERS:
        raise RuntimeError("v15.2 worker allocation is inconsistent")


validate()
