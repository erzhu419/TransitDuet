"""Held-out confirmatory evaluation for frozen Freq-HRL v7.3.2.

The source-bound analysis plan fixes training replicates, held-out path seeds,
scenarios, endpoints, aggregation, and multiplicity handling before any
confirmatory outcome is loaded. Path seeds are repeated measures; independent
training replicates are the inferential units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.experiments.reproducibility import (
    derive_seed,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)

from . import full_method_confirmatory_plan_v732 as plan
from . import full_method_hpo_v7 as hpo
from .offpolicy_baseline_validation import train_flat_offpolicy_baseline
from .ppo_actor_critic import resolve_method_contract, train_ppo_actor_critic
from .strong_learned_baseline_validation import count_parameters


CONFIRMATORY_PROTOCOL_VERSION = (
    "full_method_confirmatory_v7_3_2_source_bound_training_replicate_v1"
)
CONFIRMATORY_IMPLEMENTATION_VERSION = (
    "full_method_confirmatory_calibrated_hierarchical_stats_v7_3_2_2026_08_08"
)
EVALUATION_SCENARIOS = tuple(plan.EVALUATION_SCENARIOS)
DEFAULT_HELDOUT_SEEDS = tuple(plan.DEFAULT_HELDOUT_SEEDS)
DEFAULT_CONFIRMATORY_REPLICATES = tuple(
    plan.DEFAULT_CONFIRMATORY_REPLICATES
)


def _validate_confirmatory_roles(
    frozen: dict[str, Any],
    *,
    training_replicate_seed: int,
    heldout_seeds: Iterable[int],
) -> list[int]:
    replicate = int(training_replicate_seed)
    heldout = validate_unique_seeds(
        heldout_seeds, role="confirmatory_heldout_seeds"
    )
    if replicate not in DEFAULT_CONFIRMATORY_REPLICATES:
        raise ValueError("training replicate is outside the registered plan")
    if tuple(heldout) != DEFAULT_HELDOUT_SEEDS:
        raise ValueError("held-out path seeds differ from the registered plan")
    if replicate in set(map(int, frozen["training_replicate_seeds"])):
        raise ValueError("confirmatory training replicate overlaps HPO")
    development_seeds = set(map(int, frozen["rollout_seed_roots"]))
    development_seeds.update(map(
        int, frozen["promotion_calibration_seeds"]
    ))
    development_seeds.update(map(
        int, frozen["checkpoint_validation_seeds"]
    ))
    development_seeds.update(map(int, frozen["tuning_validation_seeds"]))
    overlap = development_seeds.intersection(heldout)
    if overlap:
        raise ValueError(
            f"confirmatory held-out seeds overlap development: {sorted(overlap)}"
        )
    plan_audit = plan.validate_plan()
    if (
        frozen.get("confirmatory_plan_version")
        != plan.CONFIRMATORY_PLAN_VERSION
        or frozen.get("confirmatory_plan_sha256") != plan_audit["sha256"]
    ):
        raise ValueError("frozen config is not bound to this confirmatory plan")
    return heldout


def _not_applicable_calibration(params: dict[str, Any]) -> dict[str, Any]:
    threshold = float(params.get("promotion_advantage_threshold", 0.0))
    return {
        "status": "not_applicable",
        "protocol_version": hpo.PROMOTION_CALIBRATION_PROTOCOL_VERSION,
        "sample_count": 0,
        "calibrated_decision_threshold": threshold,
        "evaluation_role": "not_applicable",
    }


def _train_frozen_variant(
    *,
    frozen: dict[str, Any],
    variant_id: str,
    training_replicate_seed: int,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    variant = hpo.VARIANTS_BY_ID[str(variant_id)]
    selected = frozen["selected"][variant.variant_id]
    params = dict(selected["effective_parameters"])
    expected = hpo.effective_parameters_for_variant(
        variant.variant_id, str(selected["candidate_id"])
    )
    if params != expected:
        raise ValueError("frozen effective parameters drifted")
    optimizer_seed = derive_seed(
        "freq_hrl_v732_confirmatory_optimizer",
        int(training_replicate_seed),
    )
    if variant.trainer_family == "ppo":
        payload, _, model = train_ppo_actor_critic(
            train_seeds=list(map(int, frozen["rollout_seed_roots"])),
            validation_seeds=list(map(
                int, frozen["checkpoint_validation_seeds"]
            )),
            eval_seeds=list(map(int, frozen["tuning_validation_seeds"])),
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
            scenario=hpo.TRAINING_SCENARIO,
            training_scenarios=hpo.SELECTION_SCENARIOS,
            iterations=int(frozen["iterations"]),
            seed=int(optimizer_seed),
            resample_training_paths=True,
            evaluation_role="tuning_validation",
            **hpo._ppo_training_kwargs(params),
        )
    else:
        target = hpo.canonical_full_method_parameter_count(
            int(frozen["assets"]), hidden_dim=int(params["hidden_dim"])
        )
        payload, _, model = train_flat_offpolicy_baseline(
            policy_mode=variant.policy_mode,
            train_seeds=list(map(int, frozen["rollout_seed_roots"])),
            validation_seeds=list(map(
                int, frozen["checkpoint_validation_seeds"]
            )),
            eval_seeds=list(map(int, frozen["tuning_validation_seeds"])),
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
            scenario=hpo.TRAINING_SCENARIO,
            training_scenarios=hpo.SELECTION_SCENARIOS,
            iterations=int(frozen["iterations"]),
            seed=int(optimizer_seed),
            hidden_dim=int(params["hidden_dim"]),
            learning_rate=float(params["learning_rate"]),
            replay_capacity=int(params["replay_capacity"]),
            warmup_steps=int(params["warmup_steps"]),
            batch_size=int(params["batch_size"]),
            updates_per_step=int(params["updates_per_step"]),
            resample_training_paths=True,
            evaluation_role="tuning_validation",
            reward_scale=float(params["reward_scale"]),
            execution_timeline_contract=hpo.EXECUTION_TIMELINE_CONTRACT,
            volume_impact_bps=hpo.DEFAULT_VOLUME_IMPACT_BPS,
            capacity_target_parameter_count=int(target),
            capacity_reference_method_contract=(
                hpo.CAPACITY_REFERENCE_METHOD_CONTRACT
            ),
        )
    if payload.get("heldout_test_seeds"):
        raise RuntimeError("trainer accessed held-out paths during fitting")
    checkpoint_hash = hpo._state_dict_sha256(model)
    evaluation_params = dict(params)
    calibration = _not_applicable_calibration(params)
    calibration_rows: list[dict[str, Any]] = []
    method_flags = resolve_method_contract(str(params["method_contract"]))
    if variant.trainer_family == "ppo" and method_flags[
        "learned_promotion_gate"
    ]:
        calibration, calibration_rows = (
            hpo.calibrate_promotion_advantage_threshold(
                model,
                params=params,
                calibration_seeds=list(map(
                    int, frozen["promotion_calibration_seeds"]
                )),
                steps=int(frozen["steps"]),
                assets=int(frozen["assets"]),
            )
        )
        evaluation_params["promotion_advantage_threshold"] = float(
            calibration["calibrated_decision_threshold"]
        )
    if checkpoint_hash != hpo._state_dict_sha256(model):
        raise RuntimeError("support-only calibration mutated the checkpoint")
    return payload, model, evaluation_params, calibration, calibration_rows


def run_confirmatory_cell(
    *,
    frozen_config_path: Path,
    variant_id: str,
    training_replicate_seed: int,
    heldout_seeds: list[int],
) -> dict[str, Any]:
    frozen, frozen_audit = hpo.load_frozen_config(Path(frozen_config_path))
    if variant_id not in hpo.ALL_VARIANT_IDS:
        raise ValueError(f"unknown confirmatory variant: {variant_id}")
    heldout = _validate_confirmatory_roles(
        frozen,
        training_replicate_seed=int(training_replicate_seed),
        heldout_seeds=heldout_seeds,
    )
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(frozen["code_revision"]),
        expected_source_manifest_sha256=str(frozen["source_manifest_sha256"]),
        require_verified=True,
    )
    started = time.perf_counter()
    model_payload, model, params, calibration, calibration_rows = (
        _train_frozen_variant(
            frozen=frozen,
            variant_id=str(variant_id),
            training_replicate_seed=int(training_replicate_seed),
        )
    )
    checkpoint_hash = hpo._state_dict_sha256(model)
    variant = hpo.VARIANTS_BY_ID[str(variant_id)]
    rows = (
        hpo._evaluate_ppo(
            model,
            params=params,
            scenarios=EVALUATION_SCENARIOS,
            seeds=heldout,
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
        )
        if variant.trainer_family == "ppo" else
        hpo._evaluate_offpolicy(
            model,
            params=params,
            scenarios=EVALUATION_SCENARIOS,
            seeds=heldout,
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
        )
    )
    if checkpoint_hash != hpo._state_dict_sha256(model):
        raise RuntimeError("held-out evaluation mutated the checkpoint")
    annotated = [{
        **row,
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": frozen["selected"][variant.variant_id]["candidate_id"],
        "training_replicate_seed": int(training_replicate_seed),
        "evaluation_role": "heldout_confirmatory",
        "confirmatory_protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "confirmatory_plan_sha256": plan.plan_sha256(),
        "frozen_config_sha256": frozen_audit["sha256"],
        "frozen_checkpoint_sha256": checkpoint_hash,
        "inferential_unit": plan.INFERENCE_UNIT,
    } for row in rows]
    expected_coverage = {
        (scenario, int(seed))
        for scenario in EVALUATION_SCENARIOS for seed in heldout
    }
    observed_coverage = {
        (str(row["scenario"]), int(row["seed"])) for row in annotated
    }
    if observed_coverage != expected_coverage:
        raise RuntimeError("confirmatory scenario/path coverage is incomplete")

    hf_rows: list[dict[str, Any]] = []
    if variant.variant_id == hpo.ABLATION_PARENT_VARIANT:
        for scenario in EVALUATION_SCENARIOS:
            scenario_rows = hpo.evaluate_hf_lower_intervention(
                model,
                eval_seeds=heldout,
                rollout_kwargs=hpo._hf_intervention_kwargs(
                    params,
                    steps=int(frozen["steps"]),
                    assets=int(frozen["assets"]),
                    scenario=scenario,
                ),
            )
            hf_rows.extend({
                **row,
                "variant_id": variant.variant_id,
                "training_replicate_seed": int(training_replicate_seed),
                "evaluation_role": "heldout_mechanism_diagnostic",
                "confirmatory_plan_sha256": plan.plan_sha256(),
                "frozen_config_sha256": frozen_audit["sha256"],
                "frozen_checkpoint_sha256": checkpoint_hash,
            } for row in scenario_rows)
    if checkpoint_hash != hpo._state_dict_sha256(model):
        raise RuntimeError("HF intervention mutated the checkpoint")

    parameter_count = count_parameters(model)
    summary = {
        "cell_status": "valid",
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": frozen["selected"][variant.variant_id]["candidate_id"],
        "training_replicate_seed": int(training_replicate_seed),
        "heldout_test_seeds": list(heldout),
        "heldout_test_access_status": "loaded_once_after_source_bound_freeze",
        "evaluation_scenarios": list(EVALUATION_SCENARIOS),
        "training_scenario": hpo.TRAINING_SCENARIO,
        "training_support_components": list(hpo.SELECTION_SCENARIOS),
        "ood_training_status": "excluded",
        "confirmatory_protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "confirmatory_implementation_version": (
            CONFIRMATORY_IMPLEMENTATION_VERSION
        ),
        "confirmatory_plan_version": plan.CONFIRMATORY_PLAN_VERSION,
        "confirmatory_plan_sha256": plan.plan_sha256(),
        "tuning_protocol_version": hpo.FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "full_method_implementation_version": (
            hpo.FULL_METHOD_IMPLEMENTATION_VERSION
        ),
        "promotion_calibration_protocol_version": (
            hpo.PROMOTION_CALIBRATION_PROTOCOL_VERSION
        ),
        "promotion_calibration": calibration,
        "promotion_calibration_row_count": len(calibration_rows),
        "frozen_config_sha256": frozen_audit["sha256"],
        "frozen_checkpoint_sha256": checkpoint_hash,
        "parameter_count": int(parameter_count),
        "capacity_target_parameter_count": int(
            model_payload.get("capacity_target_parameter_count", parameter_count)
        ),
        "capacity_actual_parameter_count": int(
            model_payload.get("capacity_actual_parameter_count", parameter_count)
        ),
        "selected_checkpoint_iteration": int(
            model_payload.get("selected_checkpoint_iteration", -1)
        ),
        "validation_learning_gain": float(
            model_payload.get("validation_learning_gain", 0.0)
        ),
        "row_count": len(annotated),
        "hf_intervention_pair_count": len(hf_rows),
        "inferential_unit": plan.INFERENCE_UNIT,
        "path_seeds_are_repeated_measures": True,
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "source_identity_status": source_identity["source_identity_status"],
        "elapsed_sec": float(time.perf_counter() - started),
    }
    return {
        "confirmatory_rows": annotated,
        "promotion_calibration_rows": calibration_rows,
        "hf_intervention_rows": hf_rows,
        "cell_summary": summary,
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": model_payload.get("config", {}),
            "effective_evaluation_parameters": params,
            "promotion_calibration": calibration,
            "variant_id": variant.variant_id,
            "training_replicate_seed": int(training_replicate_seed),
            "confirmatory_plan_sha256": plan.plan_sha256(),
            "frozen_config_sha256": frozen_audit["sha256"],
            "frozen_checkpoint_sha256": checkpoint_hash,
        },
    }


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_confirmatory_cell(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    save_checkpoint: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "confirmatory_rows.csv", payload["confirmatory_rows"])
    _write_csv(
        output_dir / "promotion_calibration_rows.csv",
        payload["promotion_calibration_rows"],
    )
    _write_csv(
        output_dir / "hf_intervention_rows.csv",
        payload["hf_intervention_rows"],
    )
    (output_dir / "cell_summary.json").write_text(
        json.dumps(payload["cell_summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if save_checkpoint:
        torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")


def _bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = plan.PRIMARY_BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    sampled = values[
        rng.integers(0, values.size, size=(int(draws), values.size))
    ].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def paired_randomization_p(
    values: Iterable[float],
    *,
    seed: int = 0,
    draws: int = plan.PRIMARY_RANDOMIZATION_DRAWS,
) -> float:
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    observed = abs(float(np.mean(values)))
    if observed == 0.0 and np.all(values == 0.0):
        return 1.0
    if values.size <= 16:
        assignments = np.arange(1 << values.size, dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(values.size, dtype=np.uint64)) & 1
        signs = 2.0 * bits.astype(np.float64) - 1.0
        null_stats = np.abs(np.mean(signs * values[None, :], axis=1))
        return float(np.mean(null_stats >= observed - 1e-15))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice((-1.0, 1.0), size=(int(draws), values.size))
    null_stats = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((np.sum(null_stats >= observed - 1e-15) + 1) / (draws + 1))


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row["multiplicity_family"]), []).append(row)
    for family_rows in families.values():
        ordered = sorted(family_rows, key=lambda row: float(row["p_value_raw"]))
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(row["p_value_raw"]))
            running = max(running, adjusted)
            row["p_value_holm"] = float(running)


def _effect_row(
    index: dict[tuple[str, int, str, int], dict[str, str]],
    *,
    comparator: str,
    metric: str,
    scenarios: tuple[str, ...],
    training_replicates: list[int],
    heldout_seeds: list[int],
    analysis_scope: str,
    hypothesis_role: str,
    multiplicity_family: str,
) -> dict[str, Any]:
    higher_is_better = bool(plan.REPORT_METRIC_DIRECTIONS[metric])
    direction = 1.0 if higher_is_better else -1.0
    raw_deltas = []
    for replicate in training_replicates:
        scenario_deltas = []
        for scenario in scenarios:
            path_deltas = [
                float(index[(
                    hpo.ABLATION_PARENT_VARIANT, replicate, scenario, seed
                )][metric])
                - float(index[(comparator, replicate, scenario, seed)][metric])
                for seed in heldout_seeds
            ]
            scenario_deltas.append(float(np.mean(path_deltas)))
        raw_deltas.append(float(np.mean(scenario_deltas)))
    raw = np.asarray(raw_deltas, dtype=np.float64)
    directional = direction * raw
    primary = hypothesis_role == "primary_baseline"
    bootstrap_draws = (
        plan.PRIMARY_BOOTSTRAP_DRAWS
        if primary else plan.SECONDARY_BOOTSTRAP_DRAWS
    )
    randomization_draws = (
        plan.PRIMARY_RANDOMIZATION_DRAWS
        if primary else plan.SECONDARY_RANDOMIZATION_DRAWS
    )
    ci_low, ci_high = _bootstrap_ci(
        directional,
        seed=derive_seed(
            "freq_hrl_v732_confirmatory_bootstrap",
            comparator,
            metric,
            analysis_scope,
            *scenarios,
        ),
        draws=bootstrap_draws,
    )
    std = float(np.std(directional, ddof=1)) if directional.size > 1 else 0.0
    return {
        "full_variant_id": hpo.ABLATION_PARENT_VARIANT,
        "comparator_variant_id": comparator,
        "analysis_scope": analysis_scope,
        "hypothesis_role": hypothesis_role,
        "scenarios": ";".join(scenarios),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "raw_full_minus_comparator_mean": float(np.mean(raw)),
        "directional_improvement_mean": float(np.mean(directional)),
        "directional_ci95_low": ci_low,
        "directional_ci95_high": ci_high,
        "paired_effect_size_dz": (
            float(np.mean(directional) / std)
            if std > 0.0 else
            (math.inf if float(np.mean(directional)) > 0.0 else 0.0)
        ),
        "p_value_raw": paired_randomization_p(
            directional,
            seed=derive_seed(
                "freq_hrl_v732_confirmatory_randomization",
                comparator,
                metric,
                analysis_scope,
                *scenarios,
            ),
            draws=randomization_draws,
        ),
        "bootstrap_draws": bootstrap_draws,
        "randomization_draws": randomization_draws,
        "independent_training_replicates": len(training_replicates),
        "heldout_paths_per_replicate": len(heldout_seeds),
        "scenario_count": len(scenarios),
        "inferential_unit": plan.INFERENCE_UNIT,
        "path_seed_role": "within_replicate_repeated_measure",
        "multiplicity_family": multiplicity_family,
    }


def _paired_effect_rows(
    rows: list[dict[str, str]],
    *,
    training_replicates: list[int],
    heldout_seeds: list[int],
) -> list[dict[str, Any]]:
    index = {
        (
            str(row["variant_id"]),
            int(float(row["training_replicate_seed"])),
            str(row["scenario"]),
            int(float(row["seed"])),
        ): row
        for row in rows
    }
    effects: list[dict[str, Any]] = []
    for comparator in plan.PRIMARY_BASELINE_COMPARATORS:
        for metric in plan.PRIMARY_METRICS:
            effects.append(_effect_row(
                index,
                comparator=comparator,
                metric=metric,
                scenarios=EVALUATION_SCENARIOS,
                training_replicates=training_replicates,
                heldout_seeds=heldout_seeds,
                analysis_scope="pooled_registered_scenarios",
                hypothesis_role="primary_baseline",
                multiplicity_family="primary_pooled_baseline",
            ))
    for comparator in plan.PRIMARY_BASELINE_COMPARATORS:
        for scenario in EVALUATION_SCENARIOS:
            for metric in plan.REPORT_METRIC_DIRECTIONS:
                effects.append(_effect_row(
                    index,
                    comparator=comparator,
                    metric=metric,
                    scenarios=(scenario,),
                    training_replicates=training_replicates,
                    heldout_seeds=heldout_seeds,
                    analysis_scope=f"scenario:{scenario}",
                    hypothesis_role="secondary_baseline",
                    multiplicity_family=f"secondary_baseline:{metric}",
                ))
    for comparator in plan.MECHANISM_ABLATION_COMPARATORS:
        for metric in plan.REPORT_METRIC_DIRECTIONS:
            effects.append(_effect_row(
                index,
                comparator=comparator,
                metric=metric,
                scenarios=EVALUATION_SCENARIOS,
                training_replicates=training_replicates,
                heldout_seeds=heldout_seeds,
                analysis_scope="pooled_registered_scenarios",
                hypothesis_role="secondary_ablation",
                multiplicity_family=f"secondary_ablation:{metric}",
            ))
    _holm_adjust(effects)
    for row in effects:
        if (
            float(row["directional_ci95_low"]) > 0.0
            and float(row["p_value_holm"]) < plan.ALPHA
        ):
            row["claim_status"] = "supported_improvement"
        elif (
            float(row["directional_ci95_high"]) < 0.0
            and float(row["p_value_holm"]) < plan.ALPHA
        ):
            row["claim_status"] = "supported_harm"
        else:
            row["claim_status"] = "inconclusive"
    return effects


def merge_confirmatory_cells(
    input_dirs: list[Path],
    *,
    expected_variant_ids: list[str],
    expected_training_replicates: list[int],
    expected_heldout_seeds: list[int],
) -> dict[str, Any]:
    variants = list(expected_variant_ids)
    replicates = list(map(int, expected_training_replicates))
    heldout = list(map(int, expected_heldout_seeds))
    if tuple(variants) != tuple(hpo.ALL_VARIANT_IDS):
        raise ValueError("confirmatory merge requires every registered variant")
    if tuple(replicates) != DEFAULT_CONFIRMATORY_REPLICATES:
        raise ValueError("confirmatory replicate registry drifted")
    if tuple(heldout) != DEFAULT_HELDOUT_SEEDS:
        raise ValueError("confirmatory held-out registry drifted")
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    hf_rows: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    expected_coverage = {
        (scenario, seed)
        for scenario in EVALUATION_SCENARIOS for seed in heldout
    }
    for directory in input_dirs:
        base = Path(directory)
        summary = json.loads((base / "cell_summary.json").read_text(
            encoding="utf-8"
        ))
        key = (
            str(summary["variant_id"]),
            int(summary["training_replicate_seed"]),
        )
        if key in seen or summary.get("cell_status") != "valid":
            raise ValueError(f"duplicate or invalid confirmatory cell: {key}")
        if summary.get("confirmatory_protocol_version") != (
            CONFIRMATORY_PROTOCOL_VERSION
        ):
            raise ValueError(f"confirmatory protocol mismatch: {key}")
        if summary.get("confirmatory_plan_sha256") != plan.plan_sha256():
            raise ValueError(f"confirmatory plan mismatch: {key}")
        if summary.get("heldout_test_access_status") != (
            "loaded_once_after_source_bound_freeze"
        ):
            raise ValueError(f"invalid held-out access status: {key}")
        cell_rows = _read_csv(base / "confirmatory_rows.csv")
        coverage = {
            (str(row["scenario"]), int(float(row["seed"])))
            for row in cell_rows
        }
        if coverage != expected_coverage:
            raise ValueError(f"confirmatory coverage mismatch: {key}")
        hashes = {row["frozen_checkpoint_sha256"] for row in cell_rows}
        if hashes != {summary["frozen_checkpoint_sha256"]}:
            raise ValueError(f"confirmatory cell mixed checkpoints: {key}")
        seen.add(key)
        summaries.append(summary)
        rows.extend(cell_rows)
        hf_rows.extend(_read_csv(base / "hf_intervention_rows.csv"))
    expected = {
        (variant, replicate) for variant in variants for replicate in replicates
    }
    if seen != expected:
        raise ValueError(
            f"incomplete confirmatory matrix: {sorted(expected - seen)[:6]}"
        )
    frozen_hashes = {str(row["frozen_config_sha256"]) for row in summaries}
    revisions = {str(row["code_revision"]) for row in summaries}
    manifests = {str(row["source_manifest_sha256"]) for row in summaries}
    if len(frozen_hashes) != 1 or len(revisions) != 1 or len(manifests) != 1:
        raise ValueError("confirmatory matrix mixes frozen source identities")
    effects = _paired_effect_rows(
        rows,
        training_replicates=replicates,
        heldout_seeds=heldout,
    )
    primary = [
        row for row in effects if row["hypothesis_role"] == "primary_baseline"
    ]
    return {
        "summary": {
            "status": "valid",
            "cell_count": len(summaries),
            "row_count": len(rows),
            "variant_count": len(variants),
            "scenario_count": len(EVALUATION_SCENARIOS),
            "independent_training_replicates": len(replicates),
            "heldout_paths_per_replicate": len(heldout),
            "inferential_unit": plan.INFERENCE_UNIT,
            "pseudo_replication_guard": (
                "path seeds and scenarios averaged within training replicate"
            ),
            "confirmatory_plan_version": plan.CONFIRMATORY_PLAN_VERSION,
            "confirmatory_plan_sha256": plan.plan_sha256(),
            "primary_hypothesis_count": len(primary),
            "primary_supported_improvement_count": sum(
                row["claim_status"] == "supported_improvement"
                for row in primary
            ),
            "primary_supported_harm_count": sum(
                row["claim_status"] == "supported_harm" for row in primary
            ),
            "frozen_config_sha256": next(iter(frozen_hashes)),
            "code_revision": next(iter(revisions)),
            "source_manifest_sha256": next(iter(manifests)),
        },
        "paired_effects": effects,
        "confirmatory_rows": rows,
        "hf_intervention_rows": hf_rows,
        "analysis_plan": plan.plan_payload(),
    }


def write_confirmatory_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_effects.csv", payload["paired_effects"])
    _write_csv(
        output_dir / "confirmatory_rows.csv", payload["confirmatory_rows"]
    )
    _write_csv(
        output_dir / "hf_intervention_rows.csv",
        payload["hf_intervention_rows"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_plan.json").write_text(
        json.dumps(payload["analysis_plan"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    primary = [
        row for row in payload["paired_effects"]
        if row["hypothesis_role"] == "primary_baseline"
    ]
    lines = [
        "# Freq-HRL v7.3.2 Confirmatory Results",
        "",
        f"- status: `{payload['summary']['status']}`",
        "- independent training replicates: "
        f"`{payload['summary']['independent_training_replicates']}`",
        "- held-out paths per replicate: "
        f"`{payload['summary']['heldout_paths_per_replicate']}`",
        f"- plan SHA-256: `{payload['summary']['confirmatory_plan_sha256']}`",
        "",
        "## Primary Pooled Contrasts",
        "",
    ]
    lines.extend(
        f"- {row['comparator_variant_id']} / {row['metric']}: "
        f"`{row['claim_status']}` (directional delta "
        f"{float(row['directional_improvement_mean']):.6g}, Holm p="
        f"{float(row['p_value_holm']):.4g})"
        for row in primary
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--variant-id", choices=hpo.ALL_VARIANT_IDS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument(
        "--heldout-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_HELDOUT_SEEDS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument(
        "--expected-variant-ids", nargs="*", choices=hpo.ALL_VARIANT_IDS
    )
    parser.add_argument(
        "--expected-training-replicates", type=int, nargs="*"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_confirmatory_cells(
            list(args.merge_inputs),
            expected_variant_ids=list(
                args.expected_variant_ids or hpo.ALL_VARIANT_IDS
            ),
            expected_training_replicates=list(
                args.expected_training_replicates
                or DEFAULT_CONFIRMATORY_REPLICATES
            ),
            expected_heldout_seeds=list(args.heldout_seeds),
        )
        write_confirmatory_merge(args.output_dir, payload)
        print(
            "confirmatory_v732_merge "
            f"cells={payload['summary']['cell_count']} output={args.output_dir}"
        )
        return
    if (
        args.frozen_config is None
        or args.variant_id is None
        or args.training_replicate_seed is None
    ):
        parser.error(
            "cell mode requires frozen config, variant, and training replicate"
        )
    payload = run_confirmatory_cell(
        frozen_config_path=args.frozen_config,
        variant_id=str(args.variant_id),
        training_replicate_seed=int(args.training_replicate_seed),
        heldout_seeds=list(args.heldout_seeds),
    )
    write_confirmatory_cell(
        args.output_dir, payload, save_checkpoint=bool(args.save_checkpoint)
    )
    print(
        "confirmatory_v732_cell status=valid "
        f"variant={args.variant_id} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
