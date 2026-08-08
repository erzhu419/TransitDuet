"""Pre-registered held-out analysis plan for Freq-HRL v7.3.2."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


CONFIRMATORY_PLAN_VERSION = "freq_hrl_v7_3_2_confirmatory_plan_v1"
DEFAULT_CONFIRMATORY_REPLICATES = (
    81013, 81031, 81041, 81043, 81047, 81049,
    81071, 81077, 81083, 81097, 81101, 81119,
    81131, 81157, 81163, 81173, 81181, 81197,
    81203, 81223, 81233, 81239, 81281, 81283,
)
DEFAULT_HELDOUT_SEEDS = (
    91009, 91019, 91033, 91079, 91121, 91139, 91153, 91159,
)
EVALUATION_SCENARIOS = (
    "persistent_shift",
    "promotion_recovery",
    "stationary_low_noise",
    "stationary_high_noise",
    "localized_burst",
    "ood_period",
)
PRIMARY_BASELINE_COMPARATORS = (
    "flat_ppo_matched_v7",
    "flat_gru_ppo_matched_v7",
    "generic_hrl_ppo_matched_v7",
    "generic_hrl_gru_ppo_matched_v7",
    "flat_sac_matched_v7",
    "flat_td3_matched_v7",
)
MECHANISM_ABLATION_COMPARATORS = (
    "freq_hrl_no_promotion_v7",
    "freq_hrl_no_hf_lower_v7",
    "freq_hrl_no_leakage_v7",
    "freq_hrl_no_lf_reference_v7",
    "freq_hrl_anchor_only_v7",
)
PRIMARY_METRICS = ("total_return", "LowerLFDriftAbs")
REPORT_METRIC_DIRECTIONS = {
    "total_return": True,
    "sharpe": True,
    "max_drawdown": False,
    "turnover": False,
    "LowerLFDriftAbs": False,
}
INFERENCE_UNIT = "independent_training_replicate"
PATH_AGGREGATION = "mean_within_training_replicate_and_scenario"
POOLED_SCENARIO_AGGREGATION = "equal_weight_mean_across_registered_scenarios"
PRIMARY_MULTIPLICITY = "holm_across_pooled_baseline_metric_family"
SECONDARY_MULTIPLICITY = "holm_within_metric_and_analysis_scope"
ALPHA = 0.05
PRIMARY_BOOTSTRAP_DRAWS = 50_000
PRIMARY_RANDOMIZATION_DRAWS = 200_000
SECONDARY_BOOTSTRAP_DRAWS = 20_000
SECONDARY_RANDOMIZATION_DRAWS = 50_000


def plan_payload() -> dict[str, Any]:
    return {
        "version": CONFIRMATORY_PLAN_VERSION,
        "training_replicates": list(DEFAULT_CONFIRMATORY_REPLICATES),
        "heldout_path_seeds": list(DEFAULT_HELDOUT_SEEDS),
        "evaluation_scenarios": list(EVALUATION_SCENARIOS),
        "primary_baseline_comparators": list(PRIMARY_BASELINE_COMPARATORS),
        "mechanism_ablation_comparators": list(
            MECHANISM_ABLATION_COMPARATORS
        ),
        "primary_metrics": list(PRIMARY_METRICS),
        "report_metric_directions": dict(REPORT_METRIC_DIRECTIONS),
        "inference_unit": INFERENCE_UNIT,
        "path_aggregation": PATH_AGGREGATION,
        "pooled_scenario_aggregation": POOLED_SCENARIO_AGGREGATION,
        "primary_multiplicity": PRIMARY_MULTIPLICITY,
        "secondary_multiplicity": SECONDARY_MULTIPLICITY,
        "alpha": ALPHA,
        "primary_bootstrap_draws": PRIMARY_BOOTSTRAP_DRAWS,
        "primary_randomization_draws": PRIMARY_RANDOMIZATION_DRAWS,
        "secondary_bootstrap_draws": SECONDARY_BOOTSTRAP_DRAWS,
        "secondary_randomization_draws": SECONDARY_RANDOMIZATION_DRAWS,
    }


def plan_sha256() -> str:
    encoded = json.dumps(
        plan_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_plan() -> dict[str, Any]:
    payload = plan_payload()
    replicates = payload["training_replicates"]
    heldout = payload["heldout_path_seeds"]
    scenarios = payload["evaluation_scenarios"]
    if len(replicates) < 20 or len(replicates) != len(set(replicates)):
        raise ValueError("confirmatory plan requires at least 20 unique replicates")
    if len(heldout) < 5 or len(heldout) != len(set(heldout)):
        raise ValueError("confirmatory plan requires at least five unique paths")
    if set(replicates).intersection(heldout):
        raise ValueError("confirmatory training and held-out seeds overlap")
    if len(scenarios) != len(set(scenarios)) or len(scenarios) != 6:
        raise ValueError("confirmatory scenario registry drifted")
    if not set(PRIMARY_METRICS).issubset(REPORT_METRIC_DIRECTIONS):
        raise ValueError("primary metric direction is missing")
    return {"status": "valid", "sha256": plan_sha256(), **payload}
