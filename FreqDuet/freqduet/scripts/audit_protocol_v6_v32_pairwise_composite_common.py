"""Frozen model and confirmation contract for V32 composite planning."""

from __future__ import annotations

from scripts.audit_protocol_v6_v31_pairwise_common import (
    CONFIRMATION_DECISION_INDICES as V31_CONFIRMATION_DECISION_INDICES,
    CONFIRMATION_EVAL_EPISODE as V31_CONFIRMATION_EVAL_EPISODE,
    CONFIRMATION_POLICY_SEEDS as V31_CONFIRMATION_POLICY_SEEDS,
    CONFIRMATION_REPLAY_SEED as V31_CONFIRMATION_REPLAY_SEED,
    CONFIRMATION_SCENARIO_SEEDS as V31_CONFIRMATION_SCENARIO_SEEDS,
    CONTEXT_RANKS,
    SERVICE_REFERENCE_METHOD,
)


MODEL_PROTOCOL_VERSION = "freqduet-v32-pairwise-composite-development-v1"
FEATURE_CONTRACT = "causal_foldlocal_pca_pairwise_composite_quadratic_v1"

# Frozen before V32 is fitted on the already designated V30 development data.
CONFIRMATION_POLICY_SEEDS = [36013, 36031, 36057, 36081]
CONFIRMATION_SCENARIO_SEEDS = [71011, 71029, 71047, 71071]
CONFIRMATION_DECISION_INDICES = [12, 24, 36, 48, 60, 72, 81, 87]
CONFIRMATION_EVAL_EPISODE = 500000
CONFIRMATION_REPLAY_SEED = 32001


def frozen_confirmation_roster() -> dict[str, object]:
    return {
        "policy_seeds": list(CONFIRMATION_POLICY_SEEDS),
        "scenario_seeds": list(CONFIRMATION_SCENARIO_SEEDS),
        "decision_indices": list(CONFIRMATION_DECISION_INDICES),
        "eval_episode": CONFIRMATION_EVAL_EPISODE,
        "replay_seed": CONFIRMATION_REPLAY_SEED,
    }


def confirmation_is_fresh_from_v31() -> bool:
    return bool(
        set(CONFIRMATION_POLICY_SEEDS).isdisjoint(
            V31_CONFIRMATION_POLICY_SEEDS
        )
        and set(CONFIRMATION_SCENARIO_SEEDS).isdisjoint(
            V31_CONFIRMATION_SCENARIO_SEEDS
        )
        and set(CONFIRMATION_DECISION_INDICES).isdisjoint(
            V31_CONFIRMATION_DECISION_INDICES
        )
        and CONFIRMATION_EVAL_EPISODE != V31_CONFIRMATION_EVAL_EPISODE
        and CONFIRMATION_REPLAY_SEED != V31_CONFIRMATION_REPLAY_SEED
    )


__all__ = [
    "CONFIRMATION_DECISION_INDICES",
    "CONFIRMATION_EVAL_EPISODE",
    "CONFIRMATION_POLICY_SEEDS",
    "CONFIRMATION_REPLAY_SEED",
    "CONFIRMATION_SCENARIO_SEEDS",
    "CONTEXT_RANKS",
    "FEATURE_CONTRACT",
    "MODEL_PROTOCOL_VERSION",
    "SERVICE_REFERENCE_METHOD",
    "confirmation_is_fresh_from_v31",
    "frozen_confirmation_roster",
]
