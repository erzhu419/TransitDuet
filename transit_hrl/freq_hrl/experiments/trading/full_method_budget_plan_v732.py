"""Source-bound training-budget plan for Freq-HRL v7.3.2.

The 32-iteration v7.3.1 diagnostic selected checkpoints at the search boundary.
This plan fixes the representative candidates, independent optimizer seeds,
budget ladder, and pass rule before any v7.3.2 budget outcome is observed.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


BUDGET_PLAN_VERSION = "freq_hrl_v7_3_2_training_budget_plan_v1"
BUDGET_LADDER = (64, 96, 128)
MANDATORY_BUDGETS = (64, 96)
MIN_FINAL_ITERATIONS = 96
DEFAULT_BUDGET_OPTIMIZER_SEEDS = (7103, 7111, 7121, 7127, 7151)
REPRESENTATIVE_CANDIDATES = {
    "freq_hrl_full_v7": (
        "v73_balanced_margin",
        "v73_balanced_strict",
        "v73_forecast_margin",
    ),
    "flat_ppo_matched_v7": ("ppo_lr1e4_std05",),
    "flat_gru_ppo_matched_v7": ("ppo_lr1e4_std05",),
    "generic_hrl_ppo_matched_v7": ("ppo_lr3e4_std05",),
    "generic_hrl_gru_ppo_matched_v7": ("ppo_lr3e4_std05",),
    "flat_sac_matched_v7": ("off_lr1e4_w1024_b64",),
    "flat_td3_matched_v7": ("off_lr1e3_w4096_b64",),
}
MIN_TRAINED_REPLICATE_FRACTION = 0.80
MIN_MEAN_VALIDATION_LEARNING_GAIN = 0.0
MAX_BOUNDARY_REPLICATE_FRACTION = 0.40
BOUNDARY_WINDOW_FRACTION = 0.125
SELECTION_RULE = (
    "evaluate_64_and_96; select_96_if_every_representative_passes; "
    "otherwise_evaluate_128_and_select_only_if_every_representative_passes"
)


def experiment_cells(
    budgets: tuple[int, ...] | list[int],
) -> list[tuple[int, str, str, int]]:
    return [
        (int(budget), variant_id, candidate_id, int(seed))
        for budget in budgets
        for variant_id, candidate_ids in REPRESENTATIVE_CANDIDATES.items()
        for candidate_id in candidate_ids
        for seed in DEFAULT_BUDGET_OPTIMIZER_SEEDS
    ]


def plan_payload() -> dict[str, Any]:
    return {
        "version": BUDGET_PLAN_VERSION,
        "budget_ladder": list(BUDGET_LADDER),
        "mandatory_budgets": list(MANDATORY_BUDGETS),
        "minimum_final_iterations": MIN_FINAL_ITERATIONS,
        "optimizer_seeds": list(DEFAULT_BUDGET_OPTIMIZER_SEEDS),
        "representative_candidates": {
            variant: list(candidates)
            for variant, candidates in REPRESENTATIVE_CANDIDATES.items()
        },
        "minimum_trained_replicate_fraction": (
            MIN_TRAINED_REPLICATE_FRACTION
        ),
        "minimum_mean_validation_learning_gain": (
            MIN_MEAN_VALIDATION_LEARNING_GAIN
        ),
        "maximum_boundary_replicate_fraction": (
            MAX_BOUNDARY_REPLICATE_FRACTION
        ),
        "boundary_window_fraction": BOUNDARY_WINDOW_FRACTION,
        "selection_rule": SELECTION_RULE,
    }


def plan_sha256() -> str:
    encoded = json.dumps(
        plan_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_plan() -> dict[str, Any]:
    payload = plan_payload()
    ladder = payload["budget_ladder"]
    seeds = payload["optimizer_seeds"]
    if ladder != sorted(set(ladder)) or ladder[-1] != 128:
        raise ValueError("budget ladder must be unique and increasing through 128")
    if not set(payload["mandatory_budgets"]).issubset(ladder):
        raise ValueError("mandatory budgets are outside the ladder")
    if payload["minimum_final_iterations"] not in ladder:
        raise ValueError("minimum final budget is outside the ladder")
    if len(seeds) < 5 or len(seeds) != len(set(seeds)):
        raise ValueError("budget plan requires at least five unique replicates")
    if not payload["representative_candidates"]:
        raise ValueError("budget plan has no representative candidates")
    if any(not candidates for candidates in payload["representative_candidates"].values()):
        raise ValueError("every budget family needs a representative candidate")
    return {"status": "valid", "sha256": plan_sha256(), **payload}
