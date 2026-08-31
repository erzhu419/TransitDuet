"""RL trainers for domain-agnostic Freq-HRL experiments."""

from .dual_actor_critic import (
    DualActorCriticPPO,
    DualPPOConfig,
    TrajectoryBatch,
)
from .causal_sequence import (
    CausalGRUGaussianActor,
    CausalGRUStateEncoder,
    CausalGRUValueNet,
    causal_gru_actor_parameter_count,
    causal_gru_encoder_parameter_count,
    causal_gru_value_parameter_count,
)
from .plan_actions import (
    LearnedPlanActionMapper,
    LearnedPlanCurveState,
    PlanActionResult,
)
from .joint_actor_critic import (
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    concat_joint_batches,
)
from .checkpoint_selection import (
    RobustValidationCheckpointSelector,
    StateAlignedLexicographicCheckpointSelector,
)
from .deployment_frequency import (
    DeploymentFrequencyStats,
    deployment_frequency_stats,
    deterministic_actor_action,
)
from .action_cost_critic import (
    ActionCostCritic,
    discounted_smdp_cost_returns,
    transform_latent_action,
)
from .restoration_portfolio import (
    RestorationPortfolioDecision,
    fold_guarded_restoration_eligibility,
    paired_trace_invariance_diagnostics,
    restoration_snapshot_eligible,
    select_guarded_restoration_portfolio,
)
from .responsibility_distillation import (
    ResponsibilityDistillationTargets,
    causal_macro_responsibility_targets,
    distill_hierarchical_actor_heads,
    fit_actor_output_head,
)
from .smdp_actor_critic import (
    DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES,
    PROJECTION_CONSISTENCY_UPDATE_MODES,
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    HierarchicalTrajectoryBatch,
    LevelTrajectoryBatch,
    PromotionRolloutBuilder,
    SMDPPPOConfig,
    TemporalDecisionScheduler,
    concat_hierarchical_batches,
    concat_level_batches,
)
from .training import (
    PROJECTION_CONSISTENCY_SCHEDULES,
    apply_replay_updates,
    apply_smdp_updates,
    concat_batches,
    projection_consistency_schedule_scale,
    summarize_numeric_rows,
    train_dual_ppo,
    train_frequency_separated_ppo,
    train_joint_ppo,
)
from .offpolicy_actor_critic import (
    FlatOffPolicyActorCritic,
    OffPolicyConfig,
    ReplayBuffer,
)

__all__ = [
    "DualActorCriticPPO",
    "DualPPOConfig",
    "DeploymentFrequencyStats",
    "DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES",
    "PROJECTION_CONSISTENCY_SCHEDULES",
    "PROJECTION_CONSISTENCY_UPDATE_MODES",
    "CausalGRUGaussianActor",
    "CausalGRUStateEncoder",
    "CausalGRUValueNet",
    "FrequencySeparatedActorCriticPPO",
    "HierarchicalRolloutBuilder",
    "HierarchicalTrajectoryBatch",
    "LearnedPlanActionMapper",
    "LearnedPlanCurveState",
    "JointActorCriticPPO",
    "JointPPOConfig",
    "JointTrajectoryBatch",
    "LevelTrajectoryBatch",
    "PlanActionResult",
    "PromotionRolloutBuilder",
    "RobustValidationCheckpointSelector",
    "StateAlignedLexicographicCheckpointSelector",
    "SMDPPPOConfig",
    "TemporalDecisionScheduler",
    "TrajectoryBatch",
    "apply_replay_updates",
    "apply_smdp_updates",
    "concat_batches",
    "causal_gru_actor_parameter_count",
    "causal_gru_encoder_parameter_count",
    "causal_gru_value_parameter_count",
    "concat_hierarchical_batches",
    "concat_joint_batches",
    "concat_level_batches",
    "deployment_frequency_stats",
    "ActionCostCritic",
    "RestorationPortfolioDecision",
    "ResponsibilityDistillationTargets",
    "causal_macro_responsibility_targets",
    "distill_hierarchical_actor_heads",
    "discounted_smdp_cost_returns",
    "transform_latent_action",
    "fold_guarded_restoration_eligibility",
    "fit_actor_output_head",
    "paired_trace_invariance_diagnostics",
    "projection_consistency_schedule_scale",
    "restoration_snapshot_eligible",
    "select_guarded_restoration_portfolio",
    "deterministic_actor_action",
    "summarize_numeric_rows",
    "train_dual_ppo",
    "train_frequency_separated_ppo",
    "train_joint_ppo",
    "FlatOffPolicyActorCritic",
    "OffPolicyConfig",
    "ReplayBuffer",
]
