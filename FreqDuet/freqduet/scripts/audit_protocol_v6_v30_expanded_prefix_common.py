"""Frozen discovery and confirmation rosters for V30 prefix labels."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from scripts.audit_protocol_v6_v28_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    EXPECTED_METHODS,
    OFFSETS_S,
    OUTCOME_DELTAS,
    PRIMARY_DELTA,
    checkpoint_dir,
)


LABEL_PROTOCOL_VERSION = (
    "freqduet-v30-expanded-observation-prefix-counterfactual-v1"
)
MATRIX_PROTOCOL_VERSION = "freqduet-v30-expanded-observation-matrix-v1"
MODEL_PROTOCOL_VERSION = "freqduet-v30-expanded-observation-development-v1"

DISCOVERY_TRAIN_SEEDS = [31013, 31031, 31057, 31081]
DISCOVERY_SCENARIO_SEEDS = [
    68011, 68029, 68047, 68071, 68101, 68129, 68147, 68171,
]
DISCOVERY_DECISION_INDICES = [
    2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 82,
]
DISCOVERY_EVAL_EPISODE = 200000
DISCOVERY_REPLAY_SEED = 30001

# Frozen before discovery labels are opened. These require newly trained policy
# checkpoints and are used only if the discovery gate passes unchanged.
CONFIRMATION_POLICY_SEEDS = [34013, 34031, 34057, 34081]
CONFIRMATION_SCENARIO_SEEDS = [69011, 69029, 69047, 69071]
CONFIRMATION_DECISION_INDICES = [10, 22, 34, 46, 58, 70, 80, 83]
CONFIRMATION_EVAL_EPISODE = 300000
CONFIRMATION_REPLAY_SEED = 30029

EXPANDED_CONTEXT_COLUMNS = ["waiting_total_pre", "headway_cv_active_pre"]


def discovery_checkpoint_dir(train_seed: int) -> Path:
    return checkpoint_dir(int(train_seed))


def expected_discovery_jobs() -> list[tuple[int, int, int]]:
    return list(product(
        DISCOVERY_TRAIN_SEEDS,
        DISCOVERY_SCENARIO_SEEDS,
        DISCOVERY_DECISION_INDICES,
    ))


def discovery_job_name(
    train_seed: int,
    scenario_seed: int,
    decision_index: int,
) -> str:
    return (
        f"s{int(train_seed)}_e{int(scenario_seed)}_"
        f"d{int(decision_index):02d}"
    )
