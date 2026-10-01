"""Read-only pre-update credit diagnosis after Stage65's inconclusive reward."""

import math
from scripts import pointmaze_normalized_update_stage65_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_actor_credit_stage66_v1"
POLICY = "preupdate_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_actor_credit_stage66.py"
PERIODS, TRAIN_POLICIES, TREATMENTS = source.PERIODS, source.TRAIN_POLICIES, source.TREATMENTS
SCORE_CHUNK_SIZE = 1024


def roots(*, preflight):
    return source.roots(preflight=preflight)


def source_result(root, *, preflight):
    run = "pointmaze_normalized_update_stage65_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "checkpoint": "Stage64 pre-actor critic and cloned actor/Adam",
        "batch": "exact Stage65 first training archive", "credits": ["episode_GAE", "MC_minus_same_value"],
        "gradient": "pre-update clipped-PPO loss, credit-only and with unchanged entropy",
        "parts": ["all", "mean", "log_std"], "windows": "first/last episode decile",
        "interpretation": "descriptive surrogate gradients, not true reward gradients",
        "selection": "all roots, periods, arms and both critics retained",
        "new_native_steps": 0, "optimizer_steps": 0, "critic_fits": 0, "workers": 2,
        "score_chunk_size": SCORE_CHUNK_SIZE}


def budget(*, preflight):
    opt = source.options(preflight=preflight)
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    fits = cases * len(TREATMENTS)
    episodes = cases * opt["rollouts_per_iteration"]
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_loads": fits,
        "archive_episodes": episodes, "reconstructed_lower_calls": episodes * horizon,
        "reconstructed_upper_calls": opt["rollouts_per_iteration"] * len(TRAIN_POLICIES) * sum(horizon // p for p in PERIODS),
        "extra_critic_scalar_calls": episodes * horizon, "archive_network_checks": 2 * episodes,
        "source_probe_checks": fits, "gae_calls": fits, "mc_calls": cases,
        "actor_score_forward_batches": fits * math.ceil(horizon * opt["rollouts_per_iteration"] / SCORE_CHUNK_SIZE),
        "actor_score_backward_batches": 3 * fits * math.ceil(horizon * opt["rollouts_per_iteration"] / SCORE_CHUNK_SIZE),
        "frozen_model_snapshots": fits, "frozen_model_checks": fits}
