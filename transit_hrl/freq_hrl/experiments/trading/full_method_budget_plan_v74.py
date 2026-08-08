"""Source-bound robust-checkpoint budget plan for Freq-HRL v7.4.

The v7.3.2 ladder remained boundary-limited at 128 iterations and used noisy
single-observation checkpoint selection.  This plan is committed before v7.4
outcomes and fixes fresh optimizer seeds, a 192/256 ladder, and a plateau gate.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


BUDGET_PLAN_VERSION = "freq_hrl_v7_4_robust_checkpoint_budget_plan_v1"
BUDGET_LADDER = (192, 256)
MANDATORY_BUDGETS = (192,)
MIN_FINAL_ITERATIONS = 192
DEFAULT_BUDGET_OPTIMIZER_SEEDS = (7207, 7211, 7213, 7219, 7229)
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
MIN_PLATEAU_REPLICATE_FRACTION = 0.80
MIN_PLATEAU_TAIL_ITERATIONS = 32
CHECKPOINT_SELECTION_PROTOCOL = "trailing_mean_material_improvement_v1"
CHECKPOINT_SMOOTHING_WINDOW = 8
CHECKPOINT_MIN_DELTA = 5e-4
SELECTION_RULE = (
    "evaluate_192; select_192_if_every_representative_has_80pct_material_"
    "learning_and_80pct_plateau_for_32_iterations; otherwise_evaluate_256_"
    "and_select_only_if_every_representative_passes_the_same_gate"
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
        "minimum_plateau_replicate_fraction": (
            MIN_PLATEAU_REPLICATE_FRACTION
        ),
        "minimum_plateau_tail_iterations": MIN_PLATEAU_TAIL_ITERATIONS,
        "checkpoint_selection_protocol": CHECKPOINT_SELECTION_PROTOCOL,
        "checkpoint_smoothing_window": CHECKPOINT_SMOOTHING_WINDOW,
        "checkpoint_min_delta": CHECKPOINT_MIN_DELTA,
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
    if ladder != sorted(set(ladder)) or ladder != [192, 256]:
        raise ValueError("v7.4 budget ladder must be exactly [192, 256]")
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
