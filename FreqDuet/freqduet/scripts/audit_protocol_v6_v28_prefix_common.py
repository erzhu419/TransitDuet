"""Frozen roster and shared contracts for the V28 prefix counterfactual screen."""

from __future__ import annotations

from itertools import product
from pathlib import Path


PROTOCOL_VERSION = "freqduet-v28-exact-prefix-counterfactual-v1"
MATRIX_PROTOCOL_VERSION = "freqduet-v28-prefix-matrix-v1"
MODEL_PROTOCOL_VERSION = "freqduet-v28-prefix-value-model-v1"

CONFIG = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
TRAIN_SEEDS = [31013, 31031, 31057, 31081]
EVAL_SEEDS = [65011, 65029, 65047, 65071]
DECISION_INDICES = [4, 16, 28, 40, 52, 64, 76, 84]
OFFSETS_S = [-30.0, -15.0, 0.0, 15.0, 30.0]
CHECKPOINT_EP = 39
EVAL_EPISODE = 100000
REPLAY_SEED = 28001
UPPER_DECISIONS_PER_EPISODE = 88

V27_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/results/"
    "protocol_v6_v27_msvalue_screen_2dfe51e04c_provenance_rerun1"
)
V13_SHARD_BY_TRAIN_SEED = {
    31013: "shard_0008_0009",
    31031: "shard_0009_0010",
    31057: "shard_0010_0011",
    31081: "shard_0011_0012",
}

EXPECTED_METHODS = [
    "actor",
    "actor_firstknot_m30",
    "actor_firstknot_m15",
    "actor_firstknot_0",
    "actor_firstknot_p15",
    "actor_firstknot_p30",
]

PRIMARY_DELTA = "episode_service_cost_restricted_delta_vs_actor"
OUTCOME_DELTAS = {
    "service_cost_restricted": PRIMARY_DELTA,
    "journey_min": "episode_restricted_total_journey_horizon_min_delta_vs_actor",
    "headway_cv": "episode_headway_cv_delta_vs_actor",
    "fleet_overshoot": "episode_fleet_overshoot_delta_vs_actor",
    "holding_vehicle_seconds": "episode_holding_vehicle_seconds_delta_vs_actor",
    "trip_completion_rate": "episode_trip_completion_rate_delta_vs_actor",
    "passenger_unserved_rate": "episode_passenger_unserved_rate_delta_vs_actor",
}


def checkpoint_dir(train_seed: int) -> Path:
    seed = int(train_seed)
    try:
        shard = V13_SHARD_BY_TRAIN_SEED[seed]
    except KeyError as exc:
        raise ValueError(f"unregistered V28 train seed: {seed}") from exc
    return (
        V27_ROOT
        / "logs_shards"
        / shard
        / f"{CONFIG}_seed{seed}"
        / "checkpoints"
    )


def expected_jobs() -> list[tuple[int, int, int]]:
    return list(product(TRAIN_SEEDS, EVAL_SEEDS, DECISION_INDICES))


def job_name(train_seed: int, eval_seed: int, decision_index: int) -> str:
    return f"s{int(train_seed)}_e{int(eval_seed)}_d{int(decision_index):02d}"

