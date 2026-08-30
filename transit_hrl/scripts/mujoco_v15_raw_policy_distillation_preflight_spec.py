"""Development-only specification for raw-policy responsibility distillation."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as base
from scripts import mujoco_v14_29_fresh_anchor_spec as anchors


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v15_raw_policy_distillation_preflight_v3"
EVIDENCE_ROLE = "post_v14_29_raw_policy_distillation_development_not_confirmatory"
ANCHOR_RUN_NAME = "mujoco_v14_29_fresh_anchor_bank_20260830_r1"
ENVIRONMENTS = anchors.ENVIRONMENTS
OPTIMIZER_SEEDS = (anchors.OPTIMIZER_SEEDS[0],)
DISTILL_ROOTS = (586756208, 1062465530, 3980369529, 2087212635)
DESIGN_ROOTS = (
    2617714361,
    2320213826,
    1621012606,
    523802056,
    2871897286,
    1194129291,
    702322608,
    1650629248,
)
VALIDATION_ROOTS = (
    1271831274,
    1856364544,
    1978685983,
    3984224271,
    2932771227,
    2732972439,
    1573218729,
    1141712102,
)
DISTURBANCE_MODES = base.TRAINING_DISTURBANCE_MODES
DESIGN_FOLD_COUNT = 2
EPISODE_HORIZON = 1000
LEAKAGE_COST_MODE = "power_excess"
RISK_MODE = "mode_mean"
CVAR_ALPHA = 0.5
MINIMUM_MERIT_REDUCTION = 0.01
FUNNEL_MULTIPLIER = 1.0
RIDGE = 1e-2
WORKERS = 12

CANDIDATES = tuple(
    {
        "slow_alpha": slow_alpha,
        "transfer_strength": transfer_strength,
        "blend": blend,
        "ridge": RIDGE,
    }
    for slow_alpha in (0.5, 0.75, 1.0)
    for transfer_strength in (0.75, 1.0)
    for blend in (0.4, 0.5, 0.6)
)

SELECTION_CONTRACT = {
    "teacher": (
        "strictly_causal_previous_macro_total_action_ema_projected_to_the_"
        "exact_upper_lower_action_feasibility_intersection"
    ),
    "student": (
        "ridge_fit_of_the_upper_mlp_output_head_then_lower_head_fit_on_"
        "counterfactual_states_and_exact_complements_of_the_fitted_upper"
    ),
    "design": (
        "pooled_and_two_disjoint_root_folds_require_reward_floor_and_zero_"
        "normalized_violation_on_responsibility_raw_lower_latent_lower_"
        "effective_upper_and_latent_upper_frequency_endpoints"
    ),
    "validation": "one_evaluation_of_the_design_selected_candidate_only",
    "failure": "no_design_candidate_or_validation_failure_is_an_abstention",
}


def validate() -> None:
    roles = (DISTILL_ROOTS, DESIGN_ROOTS, VALIDATION_ROOTS)
    flattened = tuple(root for role in roles for root in role)
    historical = set(
        anchors.OPTIMIZER_SEEDS
        + anchors.PRETRAIN_SEEDS
        + anchors.PRETRAIN_SELECTION_SEEDS
        + anchors.DEVELOPMENT_EVALUATION_SEEDS
    )
    if len(flattened) != 20 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v15 requires twenty unique trajectory roots")
    if historical & set(flattened):
        raise RuntimeError("v15 trajectory roots overlap v14.29 anchor roles")
    if len(DESIGN_ROOTS) % DESIGN_FOLD_COUNT:
        raise RuntimeError("v15 design roots must divide across folds")
    if len(CANDIDATES) != 18 or len({
        tuple(sorted(candidate.items())) for candidate in CANDIDATES
    }) != len(CANDIDATES):
        raise RuntimeError("v15 requires eighteen unique distillation candidates")
    if tuple(DISTURBANCE_MODES) != (
        "standard", "low_frequency", "high_frequency", "mixed"
    ):
        raise RuntimeError("v15 development modes drifted")


validate()
