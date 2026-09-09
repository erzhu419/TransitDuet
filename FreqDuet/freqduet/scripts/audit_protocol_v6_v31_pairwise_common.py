"""Frozen model and confirmation contract for V31 pairwise planning."""

from __future__ import annotations

from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_EVAL_EPISODE,
    DISCOVERY_REPLAY_SEED,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPANDED_CONTEXT_COLUMNS,
)


MODEL_PROTOCOL_VERSION = "freqduet-v31-pairwise-safe-value-development-v1"
FEATURE_CONTRACT = "causal_foldlocal_pca_pairwise_signed_quadratic_v1"
SERVICE_REFERENCE_METHOD = "actor_firstknot_p30"
CONTEXT_RANKS = [0, 1, 2, 4, 8]
RISK_THRESHOLD = 0.0

# Frozen before the V31 model is fitted. V31 uses V30 labels only for
# development; confirmation requires new policies and new rollout contexts.
CONFIRMATION_POLICY_SEEDS = [35013, 35031, 35057, 35081]
CONFIRMATION_SCENARIO_SEEDS = [70011, 70029, 70047, 70071]
CONFIRMATION_DECISION_INDICES = [6, 18, 30, 42, 54, 66, 78, 86]
CONFIRMATION_EVAL_EPISODE = 400000
CONFIRMATION_REPLAY_SEED = 31001


def frozen_confirmation_roster() -> dict[str, object]:
    return {
        "policy_seeds": list(CONFIRMATION_POLICY_SEEDS),
        "scenario_seeds": list(CONFIRMATION_SCENARIO_SEEDS),
        "decision_indices": list(CONFIRMATION_DECISION_INDICES),
        "eval_episode": CONFIRMATION_EVAL_EPISODE,
        "replay_seed": CONFIRMATION_REPLAY_SEED,
    }


def confirmation_is_disjoint_from_development() -> bool:
    return bool(
        set(CONFIRMATION_POLICY_SEEDS).isdisjoint(DISCOVERY_TRAIN_SEEDS)
        and set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            DISCOVERY_SCENARIO_SEEDS
        )
        and set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            DISCOVERY_DECISION_INDICES
        )
        and CONFIRMATION_EVAL_EPISODE != DISCOVERY_EVAL_EPISODE
        and CONFIRMATION_REPLAY_SEED != DISCOVERY_REPLAY_SEED
    )


__all__ = [
    "CONFIRMATION_DECISION_INDICES",
    "CONFIRMATION_EVAL_EPISODE",
    "CONFIRMATION_POLICY_SEEDS",
    "CONFIRMATION_REPLAY_SEED",
    "CONFIRMATION_SCENARIO_SEEDS",
    "CONTEXT_RANKS",
    "EXPANDED_CONTEXT_COLUMNS",
    "FEATURE_CONTRACT",
    "MODEL_PROTOCOL_VERSION",
    "RISK_THRESHOLD",
    "SERVICE_REFERENCE_METHOD",
    "confirmation_is_disjoint_from_development",
    "frozen_confirmation_roster",
]
