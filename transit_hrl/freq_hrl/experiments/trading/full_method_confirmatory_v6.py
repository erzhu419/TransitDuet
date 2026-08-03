"""Held-out confirmatory evaluation for a frozen Freq-HRL v6 design.

Hyperparameters are loaded from the support-only HPO freeze. Confirmatory
training replicates must be disjoint from HPO replicates. The same trained
checkpoint is evaluated on every preregistered regime and held-out path seed.
Inference averages path seeds within each training replicate, then uses the
training replicate as the independent statistical unit.
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

from . import full_method_hpo_v6 as hpo
from .performance_validation import SCENARIOS
from .ppo_actor_critic import train_ppo_actor_critic
from .offpolicy_baseline_validation import train_flat_offpolicy_baseline
from .strong_learned_baseline_validation import count_parameters


CONFIRMATORY_PROTOCOL_VERSION = "full_method_confirmatory_v6_training_replicate_inference"
CONFIRMATORY_IMPLEMENTATION_VERSION = (
    "full_method_confirmatory_frozen_checkpoint_hierarchical_stats_v6_2026_08_03"
)
EVALUATION_SCENARIOS = tuple(SCENARIOS)
DEFAULT_HELDOUT_SEEDS = (31415, 27182, 16180, 14142, 17320)
DEFAULT_CONFIRMATORY_REPLICATES = tuple(7001 + 17 * index for index in range(20))
PRIMARY_METRICS = {
    "total_return": True,
    "sharpe": True,
    "max_drawdown": False,
    "turnover": False,
    "LowerLFDriftAbs": False,
}


def _validate_confirmatory_roles(
    frozen: dict[str, Any],
    *,
    training_replicate_seed: int,
    heldout_seeds: Iterable[int],
) -> list[int]:
    heldout = validate_unique_seeds(heldout_seeds, role="confirmatory_heldout_seeds")
    hpo_replicates = set(map(int, frozen["training_replicate_seeds"]))
    if int(training_replicate_seed) in hpo_replicates:
        raise ValueError("confirmatory training replicate overlaps HPO")
    development_seeds = set(map(int, frozen["rollout_seed_roots"]))
    development_seeds.update(map(int, frozen["checkpoint_validation_seeds"]))
    development_seeds.update(map(int, frozen["tuning_validation_seeds"]))
    overlap = development_seeds.intersection(heldout)
    if overlap:
        raise ValueError(f"confirmatory held-out seeds overlap development: {sorted(overlap)}")
    return heldout


def _train_frozen_variant(
    *,
    frozen: dict[str, Any],
    variant_id: str,
    training_replicate_seed: int,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    variant = hpo.VARIANTS_BY_ID[str(variant_id)]
    selected = frozen["selected"][variant.variant_id]
    params = dict(selected["effective_parameters"])
    expected = hpo.effective_parameters_for_variant(
        variant.variant_id, str(selected["candidate_id"])
    )
    if params != expected:
        raise ValueError("frozen effective parameters drifted before confirmatory training")
    optimizer_seed = derive_seed(
        "freq_hrl_v6_confirmatory_optimizer",
        int(training_replicate_seed),
    )
    if variant.trainer_family == "ppo":
        payload, _, model = train_ppo_actor_critic(
            train_seeds=list(map(int, frozen["rollout_seed_roots"])),
            validation_seeds=list(map(int, frozen["checkpoint_validation_seeds"])),
            eval_seeds=list(map(int, frozen["tuning_validation_seeds"])),
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
            scenario=hpo.TRAINING_SCENARIO,
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
            validation_seeds=list(map(int, frozen["checkpoint_validation_seeds"])),
            eval_seeds=list(map(int, frozen["tuning_validation_seeds"])),
            steps=int(frozen["steps"]),
            assets=int(frozen["assets"]),
            scenario=hpo.TRAINING_SCENARIO,
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
            capacity_reference_method_contract=hpo.CAPACITY_REFERENCE_METHOD_CONTRACT,
        )
    if payload.get("heldout_test_seeds"):
        raise RuntimeError("confirmatory trainer accessed held-out paths during fitting")
    return payload, model, params


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
    )
    if source_identity["source_identity_status"] != "verified":
        raise RuntimeError("confirmatory source does not match the frozen HPO source")
    started = time.perf_counter()
    model_payload, model, params = _train_frozen_variant(
        frozen=frozen,
        variant_id=str(variant_id),
        training_replicate_seed=int(training_replicate_seed),
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
        raise RuntimeError("confirmatory regime evaluation mutated the checkpoint")
    annotated = [{
        **row,
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": frozen["selected"][variant.variant_id]["candidate_id"],
        "training_replicate_seed": int(training_replicate_seed),
        "evaluation_role": "heldout_confirmatory",
        "confirmatory_protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "frozen_config_sha256": frozen_audit["sha256"],
        "frozen_checkpoint_sha256": checkpoint_hash,
        "inferential_unit": "training_replicate",
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
                "frozen_config_sha256": frozen_audit["sha256"],
                "frozen_checkpoint_sha256": checkpoint_hash,
            } for row in scenario_rows)
    if checkpoint_hash != hpo._state_dict_sha256(model):
        raise RuntimeError("confirmatory HF intervention mutated the checkpoint")

    parameter_count = count_parameters(model)
    summary = {
        "cell_status": "valid",
        "variant_id": variant.variant_id,
        "scientific_role": variant.scientific_role,
        "candidate_id": frozen["selected"][variant.variant_id]["candidate_id"],
        "training_replicate_seed": int(training_replicate_seed),
        "heldout_test_seeds": list(heldout),
        "heldout_test_access_status": "loaded_once_after_freeze",
        "evaluation_scenarios": list(EVALUATION_SCENARIOS),
        "training_scenario": hpo.TRAINING_SCENARIO,
        "training_support_components": list(hpo.SELECTION_SCENARIOS),
        "ood_training_status": "excluded",
        "confirmatory_protocol_version": CONFIRMATORY_PROTOCOL_VERSION,
        "confirmatory_implementation_version": CONFIRMATORY_IMPLEMENTATION_VERSION,
        "tuning_protocol_version": hpo.FULL_METHOD_TUNING_PROTOCOL_VERSION,
        "full_method_implementation_version": hpo.FULL_METHOD_IMPLEMENTATION_VERSION,
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
        "inferential_unit": "training_replicate",
        "path_seeds_are_repeated_measures": True,
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "source_identity_status": source_identity["source_identity_status"],
        "elapsed_sec": float(time.perf_counter() - started),
    }
    return {
        "confirmatory_rows": annotated,
        "hf_intervention_rows": hf_rows,
        "cell_summary": summary,
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": model_payload.get("config", {}),
            "variant_id": variant.variant_id,
            "training_replicate_seed": int(training_replicate_seed),
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
    _write_csv(output_dir / "hf_intervention_rows.csv", payload["hf_intervention_rows"])
    (output_dir / "cell_summary.json").write_text(
        json.dumps(payload["cell_summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if save_checkpoint:
        torch.save(payload["checkpoint"], output_dir / "checkpoint.pt")


def _bootstrap_ci(
    values: np.ndarray, *, seed: int, draws: int = 20_000
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
    values: Iterable[float], *, seed: int = 0, draws: int = 100_000
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


def holm_adjust(rows: list[dict[str, Any]], *, family_key: str = "metric") -> None:
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row[family_key]), []).append(row)
    for family_rows in families.values():
        ordered = sorted(family_rows, key=lambda row: float(row["p_value_raw"]))
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(row["p_value_raw"]))
            running = max(running, adjusted)
            row["p_value_holm"] = float(running)


def _paired_effect_rows(
    rows: list[dict[str, str]],
    *,
    training_replicates: list[int],
    heldout_seeds: list[int],
) -> list[dict[str, Any]]:
    index = {
        (
            str(row["variant_id"]), int(float(row["training_replicate_seed"])),
            str(row["scenario"]), int(float(row["seed"])),
        ): row
        for row in rows
    }
    full_id = hpo.ABLATION_PARENT_VARIANT
    effects: list[dict[str, Any]] = []
    for comparator in hpo.ALL_VARIANT_IDS:
        if comparator == full_id:
            continue
        for scenario in EVALUATION_SCENARIOS:
            for metric, higher_is_better in PRIMARY_METRICS.items():
                replicate_deltas = []
                for replicate in training_replicates:
                    path_deltas = [
                        float(index[(full_id, replicate, scenario, seed)][metric])
                        - float(index[(comparator, replicate, scenario, seed)][metric])
                        for seed in heldout_seeds
                    ]
                    replicate_deltas.append(float(np.mean(path_deltas)))
                raw = np.asarray(replicate_deltas, dtype=np.float64)
                direction = 1.0 if higher_is_better else -1.0
                directional = direction * raw
                ci_low, ci_high = _bootstrap_ci(
                    raw,
                    seed=derive_seed(
                        "freq_hrl_v6_confirmatory_bootstrap",
                        comparator, scenario, metric,
                    ),
                )
                directional_ci_low = ci_low if direction > 0 else -ci_high
                directional_ci_high = ci_high if direction > 0 else -ci_low
                std = float(np.std(directional, ddof=1)) if directional.size > 1 else 0.0
                effects.append({
                    "full_variant_id": full_id,
                    "comparator_variant_id": comparator,
                    "scenario": scenario,
                    "metric": metric,
                    "higher_is_better": bool(higher_is_better),
                    "raw_full_minus_comparator_mean": float(np.mean(raw)),
                    "raw_ci95_low": ci_low,
                    "raw_ci95_high": ci_high,
                    "directional_improvement_mean": float(np.mean(directional)),
                    "directional_ci95_low": float(directional_ci_low),
                    "directional_ci95_high": float(directional_ci_high),
                    "paired_effect_size_dz": (
                        float(np.mean(directional) / std) if std > 0.0 else
                        (math.inf if float(np.mean(directional)) > 0.0 else 0.0)
                    ),
                    "p_value_raw": paired_randomization_p(
                        directional,
                        seed=derive_seed(
                            "freq_hrl_v6_confirmatory_randomization",
                            comparator, scenario, metric,
                        ),
                    ),
                    "independent_training_replicates": len(training_replicates),
                    "heldout_paths_per_replicate": len(heldout_seeds),
                    "inferential_unit": "training_replicate",
                    "path_seed_role": "within_replicate_repeated_measure",
                    "multiplicity_family": metric,
                })
    holm_adjust(effects, family_key="multiplicity_family")
    for row in effects:
        if float(row["directional_ci95_low"]) > 0.0 and float(row["p_value_holm"]) < 0.05:
            row["claim_status"] = "supported_improvement"
        elif float(row["directional_ci95_high"]) < 0.0 and float(row["p_value_holm"]) < 0.05:
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
    if set(variants) != set(hpo.ALL_VARIANT_IDS):
        raise ValueError("confirmatory merge requires all variants")
    if len(set(replicates)) < 12:
        raise ValueError(
            "confirmatory inference requires at least twelve training replicates "
            "for the preregistered randomization tests"
        )
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    hf_rows: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for directory in input_dirs:
        base = Path(directory)
        summary = json.loads((base / "cell_summary.json").read_text(encoding="utf-8"))
        key = (str(summary["variant_id"]), int(summary["training_replicate_seed"]))
        if key in seen or summary.get("cell_status") != "valid":
            raise ValueError(f"duplicate or invalid confirmatory cell: {key}")
        if summary.get("confirmatory_protocol_version") != CONFIRMATORY_PROTOCOL_VERSION:
            raise ValueError(f"confirmatory protocol mismatch: {key}")
        if summary.get("heldout_test_access_status") != "loaded_once_after_freeze":
            raise ValueError(f"invalid held-out access status: {key}")
        cell_rows = _read_csv(base / "confirmatory_rows.csv")
        coverage = {
            (str(row["scenario"]), int(float(row["seed"]))) for row in cell_rows
        }
        expected_coverage = {
            (scenario, seed) for scenario in EVALUATION_SCENARIOS for seed in heldout
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
    expected = {(variant, replicate) for variant in variants for replicate in replicates}
    if seen != expected:
        raise ValueError(f"incomplete confirmatory matrix: {sorted(expected - seen)[:6]}")
    frozen_hashes = {str(summary["frozen_config_sha256"]) for summary in summaries}
    revisions = {str(summary["code_revision"]) for summary in summaries}
    manifests = {str(summary["source_manifest_sha256"]) for summary in summaries}
    if len(frozen_hashes) != 1 or len(revisions) != 1 or len(manifests) != 1:
        raise ValueError("confirmatory matrix mixes frozen designs or source identities")
    effects = _paired_effect_rows(
        rows,
        training_replicates=replicates,
        heldout_seeds=heldout,
    )
    return {
        "summary": {
            "status": "valid",
            "cell_count": len(summaries),
            "row_count": len(rows),
            "variant_count": len(variants),
            "scenario_count": len(EVALUATION_SCENARIOS),
            "independent_training_replicates": len(replicates),
            "heldout_paths_per_replicate": len(heldout),
            "inferential_unit": "training_replicate",
            "pseudo_replication_guard": "path seeds averaged within training replicate",
            "multiplicity_correction": "Holm within each metric family",
            "supported_improvement_count": sum(
                row["claim_status"] == "supported_improvement" for row in effects
            ),
            "supported_harm_count": sum(
                row["claim_status"] == "supported_harm" for row in effects
            ),
            "frozen_config_sha256": next(iter(frozen_hashes)),
            "code_revision": next(iter(revisions)),
            "source_manifest_sha256": next(iter(manifests)),
        },
        "paired_effects": effects,
        "confirmatory_rows": rows,
        "hf_intervention_rows": hf_rows,
    }


def write_confirmatory_merge(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_effects.csv", payload["paired_effects"])
    _write_csv(output_dir / "confirmatory_rows.csv", payload["confirmatory_rows"])
    _write_csv(output_dir / "hf_intervention_rows.csv", payload["hf_intervention_rows"])
    (output_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    supported = [
        row for row in payload["paired_effects"]
        if row["claim_status"] != "inconclusive"
    ]
    lines = [
        "# Freq-HRL v6 Confirmatory Results", "",
        f"- status: `{payload['summary']['status']}`",
        f"- independent training replicates: `{payload['summary']['independent_training_replicates']}`",
        f"- held-out paths per replicate: `{payload['summary']['heldout_paths_per_replicate']}`",
        f"- multiplicity correction: `{payload['summary']['multiplicity_correction']}`",
        "", "## Non-Inconclusive Effects", "",
    ]
    lines.extend(
        f"- {row['comparator_variant_id']} / {row['scenario']} / {row['metric']}: "
        f"`{row['claim_status']}` (directional delta "
        f"{float(row['directional_improvement_mean']):.6g}, Holm p="
        f"{float(row['p_value_holm']):.4g})"
        for row in supported
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--variant-id", choices=hpo.ALL_VARIANT_IDS)
    parser.add_argument("--training-replicate-seed", type=int)
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=list(DEFAULT_HELDOUT_SEEDS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--merge-inputs", type=Path, nargs="*")
    parser.add_argument("--expected-variant-ids", nargs="*", choices=hpo.ALL_VARIANT_IDS)
    parser.add_argument("--expected-training-replicates", type=int, nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_confirmatory_cells(
            list(args.merge_inputs),
            expected_variant_ids=list(args.expected_variant_ids or hpo.ALL_VARIANT_IDS),
            expected_training_replicates=list(args.expected_training_replicates or []),
            expected_heldout_seeds=list(args.heldout_seeds),
        )
        write_confirmatory_merge(args.output_dir, payload)
        print(f"confirmatory_v6_merge cells={payload['summary']['cell_count']} output={args.output_dir}")
        return
    if args.frozen_config is None or args.variant_id is None or args.training_replicate_seed is None:
        parser.error("cell mode requires frozen config, variant, and training replicate")
    payload = run_confirmatory_cell(
        frozen_config_path=args.frozen_config,
        variant_id=str(args.variant_id),
        training_replicate_seed=int(args.training_replicate_seed),
        heldout_seeds=list(args.heldout_seeds),
    )
    write_confirmatory_cell(
        args.output_dir, payload, save_checkpoint=bool(args.save_checkpoint)
    )
    print(f"confirmatory_v6_cell status=valid variant={args.variant_id} output={args.output_dir}")


if __name__ == "__main__":
    main()
