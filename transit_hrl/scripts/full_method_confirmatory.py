#!/usr/bin/env python3
"""Freeze and execute held-out confirmation for complete Freq-HRL v4.

This runner lives outside the registered ``freq_hrl`` source manifest. It can
therefore analyze a validation-frozen algorithm revision without changing the
algorithm bytes that produced the HPO freeze. Held-out seeds are accepted only
through a preregistered protocol and never by the HPO code path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import (  # noqa: E402
    validate_evaluation_seed_roles,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)
from freq_hrl.experiments.statistics import (  # noqa: E402
    apply_holm_correction,
    bootstrap_mean_ci,
    finite_float,
    paired_delta_stats,
)
from freq_hrl.experiments.trading.full_method_hpo import (  # noqa: E402
    ALL_VARIANT_IDS,
    CAPACITY_REFERENCE_METHOD_CONTRACT,
    EXECUTION_TIMELINE_CONTRACT,
    FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
    FULL_METHOD_TUNING_PROTOCOL_VERSION,
    VARIANTS_BY_ID,
    _hf_intervention_kwargs,
    _ppo_training_kwargs,
    canonical_full_method_parameter_count,
    frozen_config_sha256,
    load_frozen_config,
)
from freq_hrl.experiments.trading.metrics import (  # noqa: E402
    METRIC_CONTRACT_VERSION,
    SELECTION_OBJECTIVE_VERSION,
)
from freq_hrl.experiments.trading.offpolicy_baseline_validation import (  # noqa: E402
    train_flat_offpolicy_baseline,
)
from freq_hrl.experiments.trading.ppo_actor_critic import (  # noqa: E402
    FULL_METHOD_IMPLEMENTATION_VERSION,
    evaluate_hf_lower_intervention,
    resolve_method_contract,
    train_ppo_actor_critic,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    DEFAULT_EVAL_SEEDS,
    DEFAULT_SCENARIOS,
    count_parameters,
    scenario_optimizer_seed,
)


CONFIRMATORY_PROTOCOL_VERSION = "full_method_confirmatory_protocol_v1"
CONFIRMATORY_ANALYSIS_VERSION = "full_method_paired_cluster_holm_v1_2026_08_03"
PRACTICAL_THRESHOLD_RULE = (
    "max(metric_floor, 0.2 * validation_cluster_delta_sample_sd)"
)
MINIMUM_STANDARDIZED_EFFECT = 0.20
MINIMUM_CONFIRMATORY_REPLICATES = 10
MINIMUM_HELDOUT_TEST_SEEDS = 10
DEFAULT_CONFIRMATORY_REPLICATE_SEEDS = (
    4279159473,
    4021320081,
    2404912296,
    1878064000,
    3487050751,
    924100023,
    4239992016,
    3032623095,
    4010426863,
    2551209601,
)
DEFAULT_FRESH_HELDOUT_TEST_SEEDS = (
    4051639434,
    2900967337,
    2776444091,
    983015312,
    465593443,
    50980312,
    819253249,
    3458787619,
    608989062,
    3973227933,
)
LEGACY_EXPOSED_TEST_SEEDS = tuple(DEFAULT_EVAL_SEEDS)
SHOCK_SCENARIOS = ("localized_burst", "persistent_shift", "ood_period")
BASELINE_VARIANTS = tuple(
    variant_id
    for variant_id in ALL_VARIANT_IDS
    if VARIANTS_BY_ID[variant_id].scientific_role.endswith("baseline")
)
ABLATION_VARIANTS = (
    "freq_hrl_no_promotion_v4",
    "freq_hrl_no_hf_lower_v4",
    "freq_hrl_no_leakage_v4",
)
FULL_VARIANT = "freq_hrl_full_v4"
METRIC_DIRECTIONS = {
    "total_return": False,
    "episode_information_ratio": False,
    "max_drawdown": True,
    "FocusScore": False,
    "LowerLFDrift": True,
}
METRIC_FLOORS = {
    "total_return": 1e-6,
    "episode_information_ratio": 0.05,
    "max_drawdown": 1e-6,
    "FocusScore": 0.005,
    "LowerLFDrift": 0.01,
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _comparison_id(
    treatment: str,
    control: str,
    metric: str,
    scope: str,
) -> str:
    return f"{scope}__{treatment}__vs__{control}__{metric}"


def _protocol_comparisons() -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for control in BASELINE_VARIANTS:
        for metric in ("total_return", "LowerLFDrift"):
            comparisons.append({
                "comparison_id": _comparison_id(
                    FULL_VARIANT, control, metric, "all_preregistered"
                ),
                "treatment": FULL_VARIANT,
                "control": control,
                "metric": metric,
                "lower_is_better": bool(METRIC_DIRECTIONS[metric]),
                "scenario_scope": "all_preregistered",
                "claim_class": "strong_baseline_superiority",
                "multiplicity_family": f"baseline_{metric}",
                "test_type": "superiority",
            })
    comparisons.extend([
        {
            "comparison_id": _comparison_id(
                FULL_VARIANT,
                "freq_hrl_no_promotion_v4",
                "total_return",
                "shock_regimes",
            ),
            "treatment": FULL_VARIANT,
            "control": "freq_hrl_no_promotion_v4",
            "metric": "total_return",
            "lower_is_better": False,
            "scenario_scope": "shock_regimes",
            "claim_class": "promotion_ablation",
            "multiplicity_family": "registered_ablation_superiority",
            "test_type": "superiority",
        },
        {
            "comparison_id": _comparison_id(
                FULL_VARIANT,
                "freq_hrl_no_hf_lower_v4",
                "total_return",
                "all_preregistered",
            ),
            "treatment": FULL_VARIANT,
            "control": "freq_hrl_no_hf_lower_v4",
            "metric": "total_return",
            "lower_is_better": False,
            "scenario_scope": "all_preregistered",
            "claim_class": "hf_lower_ablation",
            "multiplicity_family": "registered_ablation_superiority",
            "test_type": "superiority",
        },
        {
            "comparison_id": _comparison_id(
                FULL_VARIANT,
                "freq_hrl_no_leakage_v4",
                "LowerLFDrift",
                "all_preregistered",
            ),
            "treatment": FULL_VARIANT,
            "control": "freq_hrl_no_leakage_v4",
            "metric": "LowerLFDrift",
            "lower_is_better": True,
            "scenario_scope": "all_preregistered",
            "claim_class": "leakage_ablation",
            "multiplicity_family": "registered_ablation_superiority",
            "test_type": "superiority",
        },
        {
            "comparison_id": _comparison_id(
                FULL_VARIANT,
                "freq_hrl_no_leakage_v4",
                "total_return",
                "all_preregistered",
            ),
            "treatment": FULL_VARIANT,
            "control": "freq_hrl_no_leakage_v4",
            "metric": "total_return",
            "lower_is_better": False,
            "scenario_scope": "all_preregistered",
            "claim_class": "leakage_no_tradeoff",
            "multiplicity_family": "registered_noninferiority",
            "test_type": "noninferiority",
        },
    ])
    return comparisons


def _scope_rows(
    rows: list[dict[str, Any]], scenario_scope: str
) -> list[dict[str, Any]]:
    if str(scenario_scope) == "all_preregistered":
        return rows
    if str(scenario_scope) == "shock_regimes":
        return [
            row for row in rows if str(row.get("scenario")) in SHOCK_SCENARIOS
        ]
    if str(scenario_scope).startswith("scenario:"):
        scenario = str(scenario_scope).split(":", 1)[1]
        return [row for row in rows if str(row.get("scenario")) == scenario]
    raise ValueError(f"unknown scenario scope: {scenario_scope}")


def _selected_validation_rows(
    *,
    hpo_cells_root: Path,
    frozen: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = list(frozen["scenarios"])
    replicates = [int(seed) for seed in frozen["training_replicate_seeds"]]
    for variant_id in ALL_VARIANT_IDS:
        candidate_id = str(frozen["selected"][variant_id]["candidate_id"])
        for scenario in scenarios:
            for replicate in replicates:
                path = (
                    Path(hpo_cells_root)
                    / variant_id
                    / candidate_id
                    / scenario
                    / f"replicate_{replicate}"
                    / "tuning_rows.csv"
                )
                cell_rows = _read_csv(path)
                if len(cell_rows) != len(frozen["tuning_validation_seeds"]):
                    raise ValueError(
                        f"selected HPO validation rows are incomplete: {path}"
                    )
                rows.extend(cell_rows)
    return rows


def _validation_threshold(
    rows: list[dict[str, Any]], comparison: dict[str, Any]
) -> dict[str, Any]:
    stats = paired_delta_stats(
        _scope_rows(rows, str(comparison["scenario_scope"])),
        variant_key="variant_id",
        pair_keys=("scenario", "training_replicate_seed", "seed"),
        metric=str(comparison["metric"]),
        treatment=str(comparison["treatment"]),
        control=str(comparison["control"]),
        lower_is_better=bool(comparison["lower_is_better"]),
        cluster_keys=("training_replicate_seed",),
        n_boot=10_000,
        seed=20260803,
    )
    n_independent = int(stats.get("n_independent", 0))
    if n_independent < 5:
        raise ValueError(
            "practical-effect calibration requires at least five independent "
            f"validation replicates: {comparison['comparison_id']}"
        )
    standard_error = finite_float(stats.get("delta_standard_error"))
    sample_sd = (
        float(standard_error * math.sqrt(n_independent))
        if standard_error is not None and n_independent > 1 else 0.0
    )
    floor = float(METRIC_FLOORS[str(comparison["metric"])])
    threshold = max(floor, MINIMUM_STANDARDIZED_EFFECT * sample_sd)
    return {
        "practical_effect_threshold": float(threshold),
        "validation_cluster_delta_sample_sd": float(sample_sd),
        "validation_independent_training_replicates": n_independent,
        "threshold_rule": PRACTICAL_THRESHOLD_RULE,
        "metric_floor": floor,
    }


def prepare_confirmatory_protocol(
    *,
    frozen_config_path: Path,
    hpo_summary_path: Path,
    hpo_cells_root: Path,
    heldout_test_seeds: Iterable[int] = DEFAULT_FRESH_HELDOUT_TEST_SEEDS,
    confirmatory_replicate_seeds: Iterable[int] = DEFAULT_CONFIRMATORY_REPLICATE_SEEDS,
) -> dict[str, Any]:
    frozen, audit = load_frozen_config(Path(frozen_config_path))
    hpo_summary = json.loads(Path(hpo_summary_path).read_text(encoding="utf-8"))
    if canonical_sha256(hpo_summary.get("frozen_config")) != canonical_sha256(frozen):
        raise ValueError("HPO summary and frozen config disagree")
    cell_summaries = list(hpo_summary.get("cell_summaries") or [])
    if not cell_summaries:
        raise ValueError("HPO summary has no cell summaries")
    budget_fields = ("steps", "assets", "iterations", "rollout_seed_roots")
    budget: dict[str, Any] = {}
    for field in budget_fields:
        values = {
            json.dumps(row.get(field), sort_keys=True) for row in cell_summaries
        }
        if len(values) != 1:
            raise ValueError(f"HPO final matrix mixes {field}")
        budget[field] = json.loads(next(iter(values)))
    heldout = validate_unique_seeds(heldout_test_seeds, role="heldout_test_seeds")
    if len(heldout) < MINIMUM_HELDOUT_TEST_SEEDS:
        raise ValueError(
            f"confirmatory protocol requires {MINIMUM_HELDOUT_TEST_SEEDS} "
            "fresh held-out test seeds"
        )
    checkpoint = {int(seed) for seed in frozen["checkpoint_validation_seeds"]}
    tuning = {int(seed) for seed in frozen["tuning_validation_seeds"]}
    overlap = sorted(set(heldout) & (checkpoint | tuning))
    if overlap:
        raise ValueError(f"held-out seeds overlap HPO validation seeds: {overlap}")
    legacy_overlap = sorted(set(heldout) & set(LEGACY_EXPOSED_TEST_SEEDS))
    if legacy_overlap:
        raise ValueError(
            f"held-out seeds were exposed by legacy experiments: {legacy_overlap}"
        )
    replicates = validate_unique_seeds(
        confirmatory_replicate_seeds, role="confirmatory_replicate_seeds"
    )
    if len(replicates) < MINIMUM_CONFIRMATORY_REPLICATES:
        raise ValueError(
            f"confirmatory protocol requires {MINIMUM_CONFIRMATORY_REPLICATES} replicates"
        )
    overlap_replicates = sorted(
        set(replicates) & set(map(int, frozen["training_replicate_seeds"]))
    )
    if overlap_replicates:
        raise ValueError(
            "confirmatory training replicates overlap HPO replicates: "
            f"{overlap_replicates}"
        )
    validation_rows = _selected_validation_rows(
        hpo_cells_root=Path(hpo_cells_root), frozen=frozen
    )
    comparisons = []
    for comparison in _protocol_comparisons():
        comparisons.append({
            **comparison,
            **_validation_threshold(validation_rows, comparison),
        })
    protocol = {
        "status": "preregistered_from_validation_before_heldout",
        "protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
        "hpo_tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "frozen_config_sha256": audit["sha256"],
        "algorithm_code_revision": frozen["code_revision"],
        "algorithm_source_manifest_sha256": frozen["source_manifest_sha256"],
        "heldout_test_access_status_at_freeze": "not_loaded",
        "heldout_test_seeds": heldout,
        "heldout_seed_derivation": (
            "derive_seed('full_method_confirmatory_heldout_market_v1', index=0..9)"
            if tuple(heldout) == DEFAULT_FRESH_HELDOUT_TEST_SEEDS
            else "caller_supplied_before_protocol_hash"
        ),
        "checkpoint_validation_seeds": list(frozen["checkpoint_validation_seeds"]),
        "tuning_validation_seeds": list(frozen["tuning_validation_seeds"]),
        "confirmatory_replicate_seeds": replicates,
        "confirmatory_replicate_seed_derivation": (
            "derive_seed('full_method_confirmatory_training_replicate_v1', index=0..9)"
        ),
        "scenarios": list(DEFAULT_SCENARIOS),
        "variant_ids": list(ALL_VARIANT_IDS),
        "selected": frozen["selected"],
        "training_budget": {
            "steps": int(budget["steps"]),
            "assets": int(budget["assets"]),
            "iterations": int(budget["iterations"]),
            "rollout_seed_roots": [int(seed) for seed in budget["rollout_seed_roots"]],
        },
        "independent_sampling_unit": "training_replicate_seed",
        "pair_keys": ["scenario", "training_replicate_seed", "seed"],
        "minimum_independent_replicates": MINIMUM_CONFIRMATORY_REPLICATES,
        "minimum_standardized_effect_dz": MINIMUM_STANDARDIZED_EFFECT,
        "familywise_alpha": 0.05,
        "multiplicity_method": "Holm-Bonferroni within preregistered family",
        "practical_threshold_rule": PRACTICAL_THRESHOLD_RULE,
        "comparisons": comparisons,
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    return protocol


def validate_confirmatory_protocol(
    protocol: dict[str, Any],
    *,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    if protocol.get("status") != "preregistered_from_validation_before_heldout":
        raise ValueError("confirmatory protocol is not preregistered")
    if protocol.get("protocol_version") != CONFIRMATORY_PROTOCOL_VERSION:
        raise ValueError("confirmatory protocol version mismatch")
    if protocol.get("analysis_version") != CONFIRMATORY_ANALYSIS_VERSION:
        raise ValueError("confirmatory analysis version mismatch")
    expected_versions = {
        "hpo_tuning_protocol_version": FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "hpo_implementation_version": FULL_METHOD_HPO_IMPLEMENTATION_VERSION,
        "full_method_implementation_version": FULL_METHOD_IMPLEMENTATION_VERSION,
        "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }
    for field, expected in expected_versions.items():
        if protocol.get(field) != expected:
            raise ValueError(f"confirmatory protocol {field} mismatch")
    expected_sha = str(protocol.get("protocol_sha256", ""))
    without_sha = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if expected_sha != canonical_sha256(without_sha):
        raise ValueError("confirmatory protocol content drifted after preregistration")
    if protocol.get("frozen_config_sha256") != frozen_config_sha256(frozen):
        raise ValueError("confirmatory protocol frozen-config binding mismatch")
    if protocol.get("algorithm_code_revision") != frozen["code_revision"]:
        raise ValueError("confirmatory protocol algorithm revision mismatch")
    if protocol.get("algorithm_source_manifest_sha256") != frozen["source_manifest_sha256"]:
        raise ValueError("confirmatory protocol source manifest mismatch")
    if protocol.get("heldout_test_access_status_at_freeze") != "not_loaded":
        raise ValueError("confirmatory protocol was frozen after held-out access")
    if set(protocol.get("variant_ids", [])) != set(ALL_VARIANT_IDS):
        raise ValueError("confirmatory protocol variant matrix is incomplete")
    if set(protocol.get("scenarios", [])) != set(DEFAULT_SCENARIOS):
        raise ValueError("confirmatory protocol scenario matrix is incomplete")
    replicates = validate_unique_seeds(
        protocol.get("confirmatory_replicate_seeds", []),
        role="confirmatory_replicate_seeds",
    )
    if len(replicates) < MINIMUM_CONFIRMATORY_REPLICATES:
        raise ValueError("confirmatory protocol has too few independent replicates")
    if set(replicates) & set(map(int, frozen["training_replicate_seeds"])):
        raise ValueError("confirmatory training replicates overlap HPO selection")
    heldout = validate_unique_seeds(
        protocol.get("heldout_test_seeds", []), role="heldout_test_seeds"
    )
    if len(heldout) < MINIMUM_HELDOUT_TEST_SEEDS:
        raise ValueError("confirmatory protocol has too few held-out test seeds")
    if set(heldout) & set(LEGACY_EXPOSED_TEST_SEEDS):
        raise ValueError("confirmatory held-out seeds were exposed by legacy runs")
    checkpoint, heldout_checked = validate_evaluation_seed_roles(
        protocol.get("checkpoint_validation_seeds", []), heldout
    )
    if set(heldout_checked) & set(protocol.get("tuning_validation_seeds", [])):
        raise ValueError("confirmatory held-out seeds overlap tuning validation")
    if protocol.get("selected") != frozen["selected"]:
        raise ValueError("confirmatory selected configurations drifted")
    if float(protocol.get("minimum_standardized_effect_dz", -1.0)) != (
        MINIMUM_STANDARDIZED_EFFECT
    ):
        raise ValueError("confirmatory standardized effect threshold drifted")
    if float(protocol.get("familywise_alpha", -1.0)) != 0.05:
        raise ValueError("confirmatory familywise alpha drifted")
    if protocol.get("multiplicity_method") != (
        "Holm-Bonferroni within preregistered family"
    ):
        raise ValueError("confirmatory multiplicity method drifted")
    if protocol.get("practical_threshold_rule") != PRACTICAL_THRESHOLD_RULE:
        raise ValueError("confirmatory practical threshold rule drifted")
    budget = protocol.get("training_budget") or {}
    for field in ("steps", "assets", "iterations"):
        if int(budget.get(field, 0)) <= 0:
            raise ValueError(f"confirmatory training budget lacks {field}")
    validate_unique_seeds(budget.get("rollout_seed_roots", []), role="rollout_seed_roots")
    registered = _protocol_comparisons()
    comparisons = list(protocol.get("comparisons") or [])
    if [row["comparison_id"] for row in comparisons] != [
        row["comparison_id"] for row in registered
    ]:
        raise ValueError("confirmatory comparison registry drifted")
    for actual, expected in zip(comparisons, registered):
        for field, value in expected.items():
            if actual.get(field) != value:
                raise ValueError(f"confirmatory comparison drifted: {field}")
        threshold = finite_float(actual.get("practical_effect_threshold"))
        if threshold is None or threshold <= 0.0:
            raise ValueError("confirmatory practical threshold is invalid")
        if actual.get("threshold_rule") != PRACTICAL_THRESHOLD_RULE:
            raise ValueError("confirmatory comparison threshold rule drifted")
    return {
        "status": "valid",
        "protocol_sha256": expected_sha,
        "frozen_config_sha256": frozen_config_sha256(frozen),
        "checkpoint_validation_seeds": checkpoint,
        "heldout_test_seeds": heldout_checked,
    }


def load_protocol_and_frozen(
    protocol_path: Path,
    frozen_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen, _ = load_frozen_config(Path(frozen_config_path))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("confirmatory protocol must be a JSON object")
    audit = validate_confirmatory_protocol(protocol, frozen=frozen)
    return protocol, frozen, audit


def run_confirmatory_cell(
    *,
    protocol_path: Path,
    frozen_config_path: Path,
    variant_id: str,
    scenario: str,
    training_replicate_seed: int,
) -> dict[str, Any]:
    protocol, frozen, protocol_audit = load_protocol_and_frozen(
        protocol_path, frozen_config_path
    )
    if variant_id not in protocol["variant_ids"]:
        raise ValueError(f"variant not registered for confirmation: {variant_id}")
    if scenario not in protocol["scenarios"]:
        raise ValueError(f"scenario not registered for confirmation: {scenario}")
    replicate = int(training_replicate_seed)
    if replicate not in set(protocol["confirmatory_replicate_seeds"]):
        raise ValueError("training replicate is not preregistered")
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(frozen["code_revision"]),
        expected_source_manifest_sha256=str(frozen["source_manifest_sha256"]),
        require_verified=True,
    )
    variant = VARIANTS_BY_ID[variant_id]
    selected = frozen["selected"][variant_id]
    candidate_id = str(selected["candidate_id"])
    params = dict(selected["effective_parameters"])
    budget = protocol["training_budget"]
    run_seed = scenario_optimizer_seed(replicate, scenario)
    started = time.perf_counter()
    if variant.trainer_family == "ppo":
        model_payload, heldout_rows, model = train_ppo_actor_critic(
            train_seeds=list(budget["rollout_seed_roots"]),
            validation_seeds=list(protocol["checkpoint_validation_seeds"]),
            eval_seeds=list(protocol["heldout_test_seeds"]),
            steps=int(budget["steps"]),
            assets=int(budget["assets"]),
            scenario=str(scenario),
            iterations=int(budget["iterations"]),
            seed=int(run_seed),
            resample_training_paths=True,
            evaluation_role="heldout_test",
            **_ppo_training_kwargs(params),
        )
    else:
        capacity_target = canonical_full_method_parameter_count(
            int(budget["assets"]), hidden_dim=int(params["hidden_dim"])
        )
        model_payload, heldout_rows, model = train_flat_offpolicy_baseline(
            policy_mode=variant.policy_mode,
            train_seeds=list(budget["rollout_seed_roots"]),
            validation_seeds=list(protocol["checkpoint_validation_seeds"]),
            eval_seeds=list(protocol["heldout_test_seeds"]),
            steps=int(budget["steps"]),
            assets=int(budget["assets"]),
            scenario=str(scenario),
            iterations=int(budget["iterations"]),
            seed=int(run_seed),
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            replay_capacity=int(params["replay_capacity"]),
            warmup_steps=int(params["warmup_steps"]),
            batch_size=int(params["batch_size"]),
            updates_per_step=int(params["updates_per_step"]),
            resample_training_paths=True,
            evaluation_role="heldout_test",
            reward_scale=float(params["reward_scale"]),
            execution_timeline_contract=str(params["execution_timeline_contract"]),
            volume_impact_bps=float(params["volume_impact_bps"]),
            capacity_target_parameter_count=int(capacity_target),
            capacity_reference_method_contract=str(
                params["capacity_reference_method_contract"]
            ),
        )
    if model_payload.get("evaluation_role") != "heldout_test":
        raise RuntimeError("confirmatory trainer did not use heldout_test role")
    if list(model_payload.get("heldout_test_seeds", [])) != list(
        protocol["heldout_test_seeds"]
    ):
        raise RuntimeError("confirmatory trainer held-out seed set drifted")
    parameter_count = count_parameters(model)
    actual_count = int(model_payload.get("capacity_actual_parameter_count", parameter_count))
    target_count = int(model_payload.get("capacity_target_parameter_count", parameter_count))
    if parameter_count != actual_count:
        raise RuntimeError("confirmatory active parameter audit failed")
    ratio = float(actual_count / max(target_count, 1))
    if abs(ratio - 1.0) > 0.05:
        raise RuntimeError("confirmatory checkpoint is not capacity matched")
    common = {
        "variant_id": variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": candidate_id,
        "scenario": scenario,
        "training_replicate_seed": replicate,
        "optimizer_seed": int(run_seed),
        "evaluation_role": "heldout_test",
        "confirmatory_protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "confirmatory_analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
        "confirmatory_protocol_sha256": protocol_audit["protocol_sha256"],
        "frozen_config_sha256": protocol_audit["frozen_config_sha256"],
        "algorithm_code_revision": source_identity["code_revision"],
        "algorithm_source_manifest_sha256": source_identity["source_manifest_sha256"],
        "actual_effective_parameters_sha256": canonical_sha256(params),
    }
    annotated_rows = [{**row, **common} for row in heldout_rows]
    if len(annotated_rows) != len(protocol["heldout_test_seeds"]):
        raise RuntimeError("confirmatory cell emitted an incomplete held-out row set")
    if {int(row["seed"]) for row in annotated_rows} != set(
        protocol["heldout_test_seeds"]
    ):
        raise RuntimeError("confirmatory held-out row seed coverage drifted")

    flags = resolve_method_contract(variant.method_contract)
    hf_rows: list[dict[str, Any]] = []
    if variant.policy_mode == "freq_hrl" and flags["lower_hf_overlay"]:
        hf_rows = evaluate_hf_lower_intervention(
            model,
            eval_seeds=list(protocol["heldout_test_seeds"]),
            rollout_kwargs=_hf_intervention_kwargs(
                params=params,
                steps=int(budget["steps"]),
                assets=int(budget["assets"]),
                scenario=str(scenario),
            ),
        )
        hf_rows = [{
            **row,
            **common,
            "evaluation_role": "heldout_hf_mechanism_diagnostic",
        } for row in hf_rows]
    elapsed = float(time.perf_counter() - started)
    summary = {
        **common,
        "trainer_family": variant.trainer_family,
        "policy_mode": variant.policy_mode,
        "method_contract": variant.method_contract,
        "effective_parameters": params,
        "steps": int(budget["steps"]),
        "assets": int(budget["assets"]),
        "iterations": int(budget["iterations"]),
        "rollout_seed_roots": list(budget["rollout_seed_roots"]),
        "checkpoint_validation_seeds": list(protocol["checkpoint_validation_seeds"]),
        "tuning_validation_seeds_loaded": [],
        "heldout_test_seeds": list(protocol["heldout_test_seeds"]),
        "heldout_test_access_status": "loaded_only_in_confirmatory_cell",
        "parameter_count": parameter_count,
        "capacity_target_parameter_count": target_count,
        "capacity_actual_parameter_count": actual_count,
        "capacity_ratio": ratio,
        "selected_checkpoint_iteration": int(
            model_payload.get("selected_checkpoint_iteration", -1)
        ),
        "validation_learning_gain": float(
            model_payload.get("validation_learning_gain", 0.0)
        ),
        "environment_steps_train": int(model_payload.get("environment_steps_train", 0)),
        "environment_steps_validation": int(
            model_payload.get("environment_steps_validation", 0)
        ),
        "environment_steps_heldout": int(model_payload.get("environment_steps_eval", 0)),
        "heldout_row_count": len(annotated_rows),
        "hf_intervention_row_count": len(hf_rows),
        "elapsed_sec": elapsed,
        "cell_status": "valid",
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_payload.get("config", {}),
        "cell_summary": summary,
    }
    return {
        "heldout_rows": annotated_rows,
        "hf_intervention_rows": hf_rows,
        "cell_summary": summary,
        "checkpoint": checkpoint,
    }


def write_confirmatory_cell(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "heldout_rows.csv", payload["heldout_rows"])
    _write_csv(
        output_dir / "hf_intervention_rows.csv", payload["hf_intervention_rows"]
    )
    (output_dir / "cell_summary.json").write_text(
        json.dumps(payload["cell_summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")


def _check_status(row: dict[str, Any]) -> str:
    if int(row.get("n_independent", 0)) < MINIMUM_CONFIRMATORY_REPLICATES:
        return "underpowered"
    ci_low = finite_float(row.get("improvement_ci95_low"))
    effect = finite_float(row.get("paired_effect_size_dz"))
    standard_error = finite_float(row.get("delta_standard_error"))
    improvement_mean = finite_float(row.get("improvement_mean"))
    threshold = float(row.get("practical_effect_threshold", 0.0))
    if row.get("test_type") == "noninferiority":
        if ci_low is not None and ci_low >= -threshold:
            return "supported"
        return "not_supported"
    if (
        ci_low is not None
        and ci_low > threshold
        and (
            (effect is not None and effect >= MINIMUM_STANDARDIZED_EFFECT)
            or (
                standard_error == 0.0
                and improvement_mean is not None
                and improvement_mean > threshold
            )
        )
        and bool(row.get("holm_reject", False))
    ):
        return "supported"
    if ci_low is not None and ci_low > 0.0 and bool(row.get("holm_reject", False)):
        return "statistically_supported_below_practical_threshold"
    if improvement_mean is not None and improvement_mean > threshold:
        return "positive_mixed"
    if improvement_mean is not None and improvement_mean > 0.0:
        return "positive_below_practical_threshold"
    return "not_supported"


def build_confirmatory_checks(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for comparison in protocol["comparisons"]:
        stats = paired_delta_stats(
            _scope_rows(rows, str(comparison["scenario_scope"])),
            variant_key="variant_id",
            pair_keys=("scenario", "training_replicate_seed", "seed"),
            metric=str(comparison["metric"]),
            treatment=str(comparison["treatment"]),
            control=str(comparison["control"]),
            lower_is_better=bool(comparison["lower_is_better"]),
            cluster_keys=("training_replicate_seed",),
            n_boot=10_000,
            seed=20260803,
        )
        checks.append({**comparison, **stats})
    corrected = apply_holm_correction(
        [row for row in checks if row["test_type"] == "superiority"],
        family_key="multiplicity_family",
        p_key="sign_p_value",
        alpha=float(protocol["familywise_alpha"]),
    )
    corrected_by_id = {row["comparison_id"]: row for row in corrected}
    out = []
    for row in checks:
        merged = corrected_by_id.get(row["comparison_id"], row)
        if row["test_type"] == "noninferiority":
            merged = {
                **merged,
                "holm_adjusted_p_value": float("nan"),
                "multiplicity_family_size": 1,
                "holm_reject": False,
            }
        merged["status"] = _check_status(merged)
        merged["confirmatory_gate"] = (
            "cluster bootstrap over independent training replicates; superiority "
            "also requires Holm-adjusted sign test and validation-frozen minimum effect"
        )
        out.append(merged)
    return out


def _hf_diagnostic_summary(hf_rows: list[dict[str, Any]]) -> dict[str, Any]:
    full_rows = [row for row in hf_rows if row.get("variant_id") == FULL_VARIANT]
    by_replicate: dict[int, list[dict[str, Any]]] = {}
    for row in full_rows:
        by_replicate.setdefault(int(float(row["training_replicate_seed"])), []).append(row)
    sensitivity = [
        float(np.mean([float(row["lower_hf_action_sensitivity"]) for row in group]))
        for group in by_replicate.values()
    ]
    outcome_delta = [
        float(np.mean([float(row["total_return_delta"]) for row in group]))
        for group in by_replicate.values()
    ]
    sens_low, sens_high = bootstrap_mean_ci(sensitivity, n_boot=10_000, seed=20260803)
    out_low, out_high = bootstrap_mean_ci(outcome_delta, n_boot=10_000, seed=20260804)
    return {
        "independent_training_replicates": len(by_replicate),
        "hf_action_sensitivity_mean": float(np.mean(sensitivity)) if sensitivity else float("nan"),
        "hf_action_sensitivity_ci95_low": sens_low,
        "hf_action_sensitivity_ci95_high": sens_high,
        "hf_total_system_return_delta_mean": (
            float(np.mean(outcome_delta)) if outcome_delta else float("nan")
        ),
        "hf_total_system_return_delta_ci95_low": out_low,
        "hf_total_system_return_delta_ci95_high": out_high,
        "direct_hf_action_identifiability_status": (
            "supported"
            if len(by_replicate) >= MINIMUM_CONFIRMATORY_REPLICATES and sens_low > 0.0
            else "not_supported"
        ),
        "hf_total_system_effect_status": (
            "supported"
            if len(by_replicate) >= MINIMUM_CONFIRMATORY_REPLICATES and out_low > 0.0
            else "not_supported"
        ),
    }


def merge_confirmatory_cells(
    input_dirs: list[Path],
    *,
    protocol_path: Path,
    frozen_config_path: Path,
) -> dict[str, Any]:
    protocol, frozen, audit = load_protocol_and_frozen(
        protocol_path, frozen_config_path
    )
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    hf_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for directory in input_dirs:
        base = Path(directory)
        summary = json.loads((base / "cell_summary.json").read_text(encoding="utf-8"))
        key = (
            str(summary["variant_id"]),
            str(summary["scenario"]),
            int(summary["training_replicate_seed"]),
        )
        if key in seen:
            raise ValueError(f"duplicate confirmatory cell: {key}")
        seen.add(key)
        if summary.get("cell_status") != "valid":
            raise ValueError(f"invalid confirmatory cell: {key}")
        expected_entry = frozen["selected"][key[0]]
        expected_values = {
            "candidate_id": expected_entry["candidate_id"],
            "effective_parameters": expected_entry["effective_parameters"],
            "confirmatory_protocol_sha256": audit["protocol_sha256"],
            "frozen_config_sha256": audit["frozen_config_sha256"],
            "algorithm_code_revision": frozen["code_revision"],
            "algorithm_source_manifest_sha256": frozen["source_manifest_sha256"],
        }
        for field, expected in expected_values.items():
            if summary.get(field) != expected:
                raise ValueError(f"confirmatory cell {field} drifted: {key}")
        if summary.get("tuning_validation_seeds_loaded"):
            raise ValueError(f"confirmatory cell reloaded tuning seeds: {key}")
        if abs(float(summary.get("capacity_ratio", 0.0)) - 1.0) > 0.05:
            raise ValueError(f"confirmatory cell capacity mismatch: {key}")
        cell_rows = _read_csv(base / "heldout_rows.csv")
        if {int(float(row["seed"])) for row in cell_rows} != set(
            protocol["heldout_test_seeds"]
        ):
            raise ValueError(f"confirmatory held-out coverage mismatch: {key}")
        row_expected = {
            "variant_id": key[0],
            "candidate_id": expected_entry["candidate_id"],
            "scenario": key[1],
            "evaluation_role": "heldout_test",
            "confirmatory_protocol_sha256": audit["protocol_sha256"],
            "frozen_config_sha256": audit["frozen_config_sha256"],
            "algorithm_code_revision": frozen["code_revision"],
            "algorithm_source_manifest_sha256": frozen["source_manifest_sha256"],
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            "execution_timeline_contract": EXECUTION_TIMELINE_CONTRACT,
        }
        for row in cell_rows:
            for field, expected_value in row_expected.items():
                if str(row.get(field, "")) != str(expected_value):
                    raise ValueError(
                        f"confirmatory held-out row {field} drifted: {key}"
                    )
            if float(row.get("volume_impact_bps", -1.0)) != 10.0:
                raise ValueError(f"confirmatory environment impact drifted: {key}")
        cell_hf = _read_csv(base / "hf_intervention_rows.csv")
        for row in cell_hf:
            if row.get("evaluation_role") != "heldout_hf_mechanism_diagnostic":
                raise ValueError(f"confirmatory HF role drifted: {key}")
            if str(row.get("paired_exogenous_path_identity", "")).lower() not in {
                "true",
                "1",
            }:
                raise ValueError(f"confirmatory HF intervention is unpaired: {key}")
        summaries.append(summary)
        rows.extend(cell_rows)
        hf_rows.extend(cell_hf)
    expected = {
        (variant_id, scenario, int(replicate))
        for variant_id in protocol["variant_ids"]
        for scenario in protocol["scenarios"]
        for replicate in protocol["confirmatory_replicate_seeds"]
    }
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise ValueError(
            "incomplete confirmatory matrix: "
            + ", ".join(map(str, (missing + unexpected)[:6]))
        )
    checks = build_confirmatory_checks(rows, protocol)
    learning_rows = []
    for variant_id in protocol["variant_ids"]:
        group = [row for row in summaries if row["variant_id"] == variant_id]
        trained_fraction = float(np.mean([
            int(row["selected_checkpoint_iteration"]) >= 0 for row in group
        ]))
        gain_mean = float(np.mean([float(row["validation_learning_gain"]) for row in group]))
        learning_rows.append({
            "variant_id": variant_id,
            "cell_count": len(group),
            "trained_checkpoint_fraction": trained_fraction,
            "validation_learning_gain_mean": gain_mean,
            "status": (
                "supported" if trained_fraction >= 0.80 and gain_mean > 0.0
                else "not_supported"
            ),
        })
    return {
        "summary": {
            "protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
            "analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
            "matrix_coverage_status": "complete",
            "cell_count": len(summaries),
            "heldout_row_count": len(rows),
            "independent_training_replicates": len(
                protocol["confirmatory_replicate_seeds"]
            ),
            "heldout_seed_count": len(protocol["heldout_test_seeds"]),
            "source_binding_status": "verified",
            "frozen_config_sha256": audit["frozen_config_sha256"],
            "confirmatory_protocol_sha256": audit["protocol_sha256"],
            "all_primary_superiority_status": (
                "supported"
                if all(
                    row["status"] == "supported"
                    for row in checks
                    if row["test_type"] == "superiority"
                ) else "not_supported"
            ),
            "all_learning_dynamics_status": (
                "supported"
                if all(row["status"] == "supported" for row in learning_rows)
                else "not_supported"
            ),
        },
        "checks": checks,
        "learning_dynamics": learning_rows,
        "hf_diagnostic": _hf_diagnostic_summary(hf_rows),
        "cell_summaries": summaries,
        "heldout_rows": rows,
        "hf_intervention_rows": hf_rows,
    }


def write_confirmatory_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "confirmatory_checks.csv", payload["checks"])
    _write_csv(output_dir / "learning_dynamics.csv", payload["learning_dynamics"])
    _write_csv(output_dir / "cell_summaries.csv", payload["cell_summaries"])
    _write_csv(output_dir / "heldout_rows.csv", payload["heldout_rows"])
    _write_csv(
        output_dir / "hf_intervention_rows.csv", payload["hf_intervention_rows"]
    )
    serializable = {
        key: value
        for key, value in payload.items()
        if key not in {"heldout_rows", "hf_intervention_rows"}
    }
    (output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "# Full-Method Confirmatory Validation",
        "",
        f"- matrix: `{payload['summary']['matrix_coverage_status']}`",
        f"- source binding: `{payload['summary']['source_binding_status']}`",
        f"- independent training replicates: `{payload['summary']['independent_training_replicates']}`",
        f"- held-out seeds per cell: `{payload['summary']['heldout_seed_count']}`",
        f"- all superiority claims: `{payload['summary']['all_primary_superiority_status']}`",
        "",
        "| claim | metric | control | improvement | CI low | Holm p | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["checks"]:
        lines.append(
            f"| {row['claim_class']} | {row['metric']} | {row['control']} "
            f"| {float(row['improvement_mean']):+.6g} "
            f"| {float(row['improvement_ci95_low']):+.6g} "
            f"| {float(row.get('holm_adjusted_p_value', float('nan'))):.4g} "
            f"| {row['status']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--prepare-protocol", action="store_true")
    parser.add_argument("--hpo-summary", type=Path)
    parser.add_argument("--hpo-cells-root", type=Path)
    parser.add_argument(
        "--heldout-test-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_FRESH_HELDOUT_TEST_SEEDS),
    )
    parser.add_argument(
        "--confirmatory-replicate-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIRMATORY_REPLICATE_SEEDS),
    )
    parser.add_argument("--variant-id", choices=ALL_VARIANT_IDS)
    parser.add_argument("--scenario", choices=DEFAULT_SCENARIOS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare_protocol:
        if args.hpo_summary is None or args.hpo_cells_root is None:
            parser.error("--prepare-protocol requires --hpo-summary and --hpo-cells-root")
        protocol = prepare_confirmatory_protocol(
            frozen_config_path=args.frozen_config,
            hpo_summary_path=args.hpo_summary,
            hpo_cells_root=args.hpo_cells_root,
            heldout_test_seeds=args.heldout_test_seeds,
            confirmatory_replicate_seeds=args.confirmatory_replicate_seeds,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "confirmatory_protocol.json"
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
        print(f"prepared confirmatory protocol sha256={protocol['protocol_sha256']} path={path}")
        return
    if args.protocol is None:
        parser.error("cell/merge mode requires --protocol")
    if args.merge_inputs:
        payload = merge_confirmatory_cells(
            list(args.merge_inputs),
            protocol_path=args.protocol,
            frozen_config_path=args.frozen_config,
        )
        write_confirmatory_merge(args.output_dir, payload)
        print(f"merged confirmatory cells={payload['summary']['cell_count']} output={args.output_dir}")
        return
    if args.variant_id is None or args.scenario is None or args.training_replicate_seed is None:
        parser.error("cell mode requires --variant-id, --scenario, and --training-replicate-seed")
    payload = run_confirmatory_cell(
        protocol_path=args.protocol,
        frozen_config_path=args.frozen_config,
        variant_id=args.variant_id,
        scenario=args.scenario,
        training_replicate_seed=args.training_replicate_seed,
    )
    write_confirmatory_cell(args.output_dir, payload)
    print(
        f"confirmatory_cell status=valid variant={args.variant_id} "
        f"scenario={args.scenario} replicate={args.training_replicate_seed}"
    )


if __name__ == "__main__":
    main()
