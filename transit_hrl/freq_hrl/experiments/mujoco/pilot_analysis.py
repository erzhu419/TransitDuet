"""Integrity-first analysis for source-bound MuJoCo development pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.domains.mujoco import DISTURBANCE_MODES
from freq_hrl.experiments.mujoco.control_validation import (
    DEFAULT_ENV_IDS,
    DEFAULT_EVAL_SEEDS,
    METHODS,
)
from freq_hrl.experiments.statistics import paired_delta_stats


ANALYSIS_VERSION = "mujoco_development_pilot_audit_v1"
PRIMARY_METRIC = "episode_return"
PAIR_COMPARISONS = (
    ("freq_hrl", "flat_ppo"),
    ("freq_hrl", "generic_hrl"),
    ("freq_hrl", "freq_hrl_no_leakage"),
    ("freq_hrl_no_leakage", "generic_hrl"),
    ("generic_hrl", "flat_ppo"),
)


@dataclass(frozen=True)
class PilotCell:
    path: Path
    environment: str
    method: str
    optimizer_seed: int
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    rows: list[dict[str, str]]
    checkpoint_file_sha256: str

    @property
    def key(self) -> tuple[str, str, int]:
        return self.environment, self.method, self.optimizer_seed


def _parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float:
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.mean(finite)) if finite else float("nan")


def _sample_std(values: Iterable[Any]) -> float:
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def load_cells(run_dir: Path) -> list[PilotCell]:
    cell_root = Path(run_dir) / "cells"
    summaries = sorted(cell_root.glob("*/*/replicate_*/cell_summary.json"))
    cells: list[PilotCell] = []
    for summary_path in summaries:
        cell_dir = summary_path.parent
        history_path = cell_dir / "training_history.json"
        rows_path = cell_dir / "evaluation_rows.csv"
        checkpoint_path = cell_dir / "checkpoint.pt"
        missing = [
            path.name for path in (history_path, rows_path, checkpoint_path)
            if not path.is_file()
        ]
        if missing:
            raise ValueError(
                f"incomplete MuJoCo cell {cell_dir}: missing {','.join(missing)}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError(f"training history is not a list: {history_path}")
        try:
            optimizer_seed = int(cell_dir.name.removeprefix("replicate_"))
        except ValueError as exc:
            raise ValueError(f"invalid replicate directory: {cell_dir}") from exc
        cells.append(PilotCell(
            path=cell_dir,
            environment=str(summary.get("environment", cell_dir.parents[1].name)),
            method=str(summary.get("method", cell_dir.parent.name)),
            optimizer_seed=optimizer_seed,
            summary=summary,
            history=history,
            rows=_read_csv(rows_path),
            checkpoint_file_sha256=_sha256(checkpoint_path),
        ))
    return cells


def audit_cells(
    cells: list[PilotCell],
    *,
    expected_protocol_version: str,
    environments: Iterable[str],
    methods: Iterable[str],
    optimizer_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
    disturbance_modes: Iterable[str],
) -> dict[str, Any]:
    environments = tuple(map(str, environments))
    methods = tuple(map(str, methods))
    optimizer_seeds = tuple(map(int, optimizer_seeds))
    evaluation_seeds = tuple(map(int, evaluation_seeds))
    disturbance_modes = tuple(map(str, disturbance_modes))
    expected_keys = {
        (environment, method, optimizer_seed)
        for environment in environments
        for method in methods
        for optimizer_seed in optimizer_seeds
    }
    actual_keys = [cell.key for cell in cells]
    issues: list[str] = []
    warnings: list[str] = []
    if len(actual_keys) != len(set(actual_keys)):
        issues.append("duplicate environment/method/optimizer cell keys")
    missing_keys = sorted(expected_keys - set(actual_keys))
    unexpected_keys = sorted(set(actual_keys) - expected_keys)
    if missing_keys:
        issues.append(f"missing {len(missing_keys)} expected cells; first={missing_keys[0]}")
    if unexpected_keys:
        issues.append(
            f"found {len(unexpected_keys)} unexpected cells; first={unexpected_keys[0]}"
        )

    source_identities: set[tuple[str, str, str]] = set()
    cell_audit: list[dict[str, Any]] = []
    for cell in cells:
        prefix = "/".join(map(str, cell.key))
        summary = cell.summary
        cell_issues: list[str] = []
        cell_warnings: list[str] = []
        if str(summary.get("protocol_version")) != str(expected_protocol_version):
            cell_issues.append("protocol_version_mismatch")
        if str(summary.get("source_identity_status")) != "verified":
            cell_issues.append("source_identity_not_verified")
        source_identities.add((
            str(summary.get("code_revision", "")),
            str(summary.get("source_manifest_sha256", "")),
            str(summary.get("source_identity_status", "")),
        ))
        iterations = int(summary.get("iterations", -1))
        expected_iterations = list(range(-1, iterations))
        observed_iterations = [int(row.get("iteration", -999)) for row in cell.history]
        if observed_iterations != expected_iterations:
            cell_issues.append("training_history_incomplete_or_unordered")
        selected = [
            int(row["iteration"]) for row in cell.history
            if bool(row.get("checkpoint_selected", False))
        ]
        if not selected or selected[-1] != int(
            summary.get("selected_checkpoint_iteration", -999)
        ):
            cell_issues.append("selected_checkpoint_history_mismatch")
        learning_gain = _finite(summary.get("validation_learning_gain"))
        initial = _finite(summary.get("initial_validation_score"))
        best = _finite(summary.get("checkpoint_selection_score"))
        if None in (learning_gain, initial, best) or not math.isclose(
            float(learning_gain), float(best) - float(initial),
            rel_tol=1e-6, abs_tol=1e-8,
        ):
            cell_issues.append("validation_learning_gain_mismatch")

        expected_paths = {
            (mode, seed) for mode in disturbance_modes for seed in evaluation_seeds
        }
        observed_paths: list[tuple[str, int]] = []
        for row in cell.rows:
            try:
                observed_paths.append((
                    str(row["disturbance_mode"]), int(row["seed"])
                ))
            except (KeyError, ValueError):
                cell_issues.append("malformed_evaluation_row")
                continue
            if str(row.get("environment")) != cell.environment:
                cell_issues.append("evaluation_environment_mismatch")
            if str(row.get("method")) != cell.method:
                cell_issues.append("evaluation_method_mismatch")
            if int(row.get("training_replicate_seed", -1)) != cell.optimizer_seed:
                cell_issues.append("evaluation_optimizer_seed_mismatch")
            if str(row.get("evaluation_role")) != "heldout_test":
                cell_issues.append("evaluation_role_not_heldout")
            if str(row.get("protocol_version")) != str(expected_protocol_version):
                cell_issues.append("evaluation_protocol_mismatch")
            if _finite(row.get("protocol_valid")) != 1.0:
                cell_issues.append("evaluation_protocol_invalid")
        if len(observed_paths) != len(set(observed_paths)):
            cell_issues.append("duplicate_evaluation_paths")
        if set(observed_paths) != expected_paths:
            cell_issues.append("evaluation_seed_or_mode_matrix_mismatch")

        checkpoint_path = cell.path / "checkpoint.pt"
        try:
            checkpoint = _checkpoint_metadata(checkpoint_path)
        except Exception as exc:  # pragma: no cover - corrupt artifact path
            cell_issues.append(f"checkpoint_load_failed:{type(exc).__name__}")
            checkpoint = {}
        embedded_parameter_hash = str(
            checkpoint.get(
                "frozen_parameter_sha256",
                checkpoint.get("frozen_checkpoint_sha256", ""),
            )
        )
        summary_parameter_hash = str(
            summary.get(
                "frozen_parameter_sha256",
                summary.get("frozen_checkpoint_sha256", ""),
            )
        )
        if not summary_parameter_hash or embedded_parameter_hash != summary_parameter_hash:
            cell_issues.append("checkpoint_parameter_hash_metadata_mismatch")
        recorded_file_hash = str(summary.get("checkpoint_file_sha256", ""))
        if recorded_file_hash:
            if recorded_file_hash != cell.checkpoint_file_sha256:
                cell_issues.append("checkpoint_file_hash_mismatch")
        else:
            cell_warnings.append("checkpoint_file_hash_not_recorded_by_legacy_protocol")
        if abs(float(summary.get("capacity_ratio", 0.0)) - 1.0) > 0.03:
            cell_issues.append("capacity_ratio_outside_three_percent")

        cell_issues = sorted(set(cell_issues))
        cell_warnings = sorted(set(cell_warnings))
        issues.extend(f"{prefix}:{item}" for item in cell_issues)
        warnings.extend(f"{prefix}:{item}" for item in cell_warnings)
        cell_audit.append({
            "environment": cell.environment,
            "method": cell.method,
            "optimizer_seed": cell.optimizer_seed,
            "status": "valid" if not cell_issues else "invalid",
            "issues": cell_issues,
            "warnings": cell_warnings,
            "checkpoint_file_sha256": cell.checkpoint_file_sha256,
            "parameter_sha256_metadata": summary_parameter_hash,
            "evaluation_row_count": len(cell.rows),
            "training_history_row_count": len(cell.history),
        })
    if len(source_identities) != 1:
        issues.append("source identity is not uniform across cells")
    return {
        "analysis_version": ANALYSIS_VERSION,
        "evidence_role": "development_only_not_claim_eligible",
        "status": "valid" if not issues else "invalid",
        "expected_protocol_version": str(expected_protocol_version),
        "expected_cell_count": len(expected_keys),
        "observed_cell_count": len(cells),
        "independent_optimizer_replicates_per_method_environment": len(
            optimizer_seeds
        ),
        "heldout_paths_per_cell": len(evaluation_seeds) * len(disturbance_modes),
        "source_identities": [list(item) for item in sorted(source_identities)],
        "issues": sorted(set(issues)),
        "warnings": sorted(set(warnings)),
        "cells": cell_audit,
    }


def replicate_rows(cells: list[PilotCell]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric_metrics = (
        "episode_return",
        "LowerLFDriftAbs",
        "UpperHFPowerAbs",
        "LowerLFBudgetViolationRate",
    )
    for cell in cells:
        modes = sorted({str(row["disturbance_mode"]) for row in cell.rows})
        for mode in modes:
            rows = [
                row for row in cell.rows
                if str(row["disturbance_mode"]) == mode
            ]
            aggregate: dict[str, Any] = {
                "environment": cell.environment,
                "method": cell.method,
                "optimizer_seed": cell.optimizer_seed,
                "disturbance_mode": mode,
                "heldout_path_count": len(rows),
            }
            for metric in numeric_metrics:
                if rows and metric in rows[0]:
                    aggregate[metric] = _mean(row.get(metric) for row in rows)
            output.append(aggregate)
    return output


def method_summaries(
    cells: list[PilotCell], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cell_index = {cell.key: cell for cell in cells}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["environment"]),
            str(row["method"]),
            str(row["disturbance_mode"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        gains = [
            cell_index[(key[0], key[1], int(row["optimizer_seed"]))].summary[
                "validation_learning_gain"
            ]
            for row in group
        ]
        selected = [
            cell_index[(key[0], key[1], int(row["optimizer_seed"]))].summary[
                "selected_checkpoint_iteration"
            ]
            for row in group
        ]
        output.append({
            "environment": key[0],
            "method": key[1],
            "disturbance_mode": key[2],
            "n_independent_optimizer_replicates": len(group),
            "episode_return_mean": _mean(row.get(PRIMARY_METRIC) for row in group),
            "episode_return_sample_std": _sample_std(
                row.get(PRIMARY_METRIC) for row in group
            ),
            "episode_return_min": min(float(row[PRIMARY_METRIC]) for row in group),
            "episode_return_max": max(float(row[PRIMARY_METRIC]) for row in group),
            "validation_learning_gain_mean": _mean(gains),
            "selected_checkpoint_iteration_mean": _mean(selected),
            "LowerLFDriftAbs_mean": _mean(
                row.get("LowerLFDriftAbs") for row in group
            ),
            "LowerLFBudgetViolationRate_mean": _mean(
                row.get("LowerLFBudgetViolationRate") for row in group
            ),
            "evidence_role": "development_only_not_claim_eligible",
        })
    return output


def paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    environments = sorted({str(row["environment"]) for row in rows})
    modes = sorted({str(row["disturbance_mode"]) for row in rows})
    available_methods = {str(row["method"]) for row in rows}
    for environment in environments:
        for mode in modes:
            subset = [
                row for row in rows
                if str(row["environment"]) == environment
                and str(row["disturbance_mode"]) == mode
            ]
            for treatment, control in PAIR_COMPARISONS:
                if treatment not in available_methods or control not in available_methods:
                    continue
                for metric, lower_is_better in (
                    ("episode_return", False),
                    ("LowerLFDriftAbs", True),
                ):
                    if metric == "LowerLFDriftAbs" and "flat_ppo" in {
                        treatment, control
                    }:
                        continue
                    if not all(metric in row for row in subset):
                        continue
                    stats = paired_delta_stats(
                        subset,
                        variant_key="method",
                        pair_keys=("environment", "optimizer_seed"),
                        cluster_keys=("environment", "optimizer_seed"),
                        metric=metric,
                        treatment=treatment,
                        control=control,
                        lower_is_better=lower_is_better,
                        n_boot=10_000,
                        seed=20260808,
                    )
                    output.append({
                        "environment": environment,
                        "disturbance_mode": mode,
                        **stats,
                        "evidence_role": "development_only_not_claim_eligible",
                        "claim_status": "development_diagnostic",
                    })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    finite = _finite(value)
    return "NA" if finite is None else f"{finite:.3f}"


def _markdown_report(
    audit: dict[str, Any], summaries: list[dict[str, Any]], effects: list[dict[str, Any]]
) -> str:
    lines = [
        "# MuJoCo Development Pilot Audit",
        "",
        f"Analysis: `{ANALYSIS_VERSION}`",
        "",
        "This report is development-only. Held-out paths inside one optimizer "
        "replicate are repeated measurements, not independent training replicates.",
        "",
        "## Integrity",
        "",
        f"- Status: **{audit['status']}**",
        f"- Cells: {audit['observed_cell_count']}/{audit['expected_cell_count']}",
        "- Independent optimizer replicates per method/environment: "
        f"{audit['independent_optimizer_replicates_per_method_environment']}",
        f"- Issues: {len(audit['issues'])}",
        f"- Warnings: {len(audit['warnings'])}",
        "",
        "## Standard Task",
        "",
        "| Environment | Method | Independent n | Return mean | Return SD | "
        "Validation gain | Selected iteration |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if str(row["disturbance_mode"]) != "standard":
            continue
        lines.append(
            f"| {row['environment']} | {row['method']} | "
            f"{row['n_independent_optimizer_replicates']} | "
            f"{_format(row['episode_return_mean'])} | "
            f"{_format(row['episode_return_sample_std'])} | "
            f"{_format(row['validation_learning_gain_mean'])} | "
            f"{_format(row['selected_checkpoint_iteration_mean'])} |"
        )
    lines.extend([
        "",
        "## Primary Paired Diagnostics",
        "",
        "Positive improvement favors the treatment. These rows are not paper claims.",
        "",
        "| Environment | Treatment | Control | Independent n | Mean improvement | "
        "Win rate |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in effects:
        if (
            str(row["disturbance_mode"]) != "standard"
            or str(row["metric"]) != PRIMARY_METRIC
        ):
            continue
        lines.append(
            f"| {row['environment']} | {row['treatment']} | {row['control']} | "
            f"{row['n_independent']} | {_format(row['improvement_mean'])} | "
            f"{_format(row['win_rate'])} |"
        )
    if audit["issues"]:
        lines.extend(["", "## Blocking Issues", ""])
        lines.extend(f"- `{issue}`" for issue in audit["issues"][:50])
    if audit["warnings"]:
        lines.extend(["", "## Evidence Warnings", ""])
        unique = sorted({warning.split(":", 3)[-1] for warning in audit["warnings"]})
        lines.extend(f"- `{warning}`" for warning in unique)
    lines.extend([
        "",
        "## Gate",
        "",
        "A valid pilot may select protocol and compute budget only. Formal evaluation "
        "requires fresh optimizer seeds, untouched evaluation seeds, a frozen source "
        "manifest, and multiplicity-controlled confirmatory analysis.",
        "",
    ])
    return "\n".join(lines)


def write_analysis(
    run_dir: Path,
    output_dir: Path,
    *,
    expected_protocol_version: str,
    environments: Iterable[str],
    methods: Iterable[str],
    optimizer_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
    disturbance_modes: Iterable[str],
) -> dict[str, Any]:
    cells = load_cells(run_dir)
    audit = audit_cells(
        cells,
        expected_protocol_version=expected_protocol_version,
        environments=environments,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        evaluation_seeds=evaluation_seeds,
        disturbance_modes=disturbance_modes,
    )
    replicates = replicate_rows(cells)
    summaries = method_summaries(cells, replicates)
    effects = paired_effects(replicates)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output / "replicate_aggregates.csv", replicates)
    _write_csv(output / "method_summary.csv", summaries)
    _write_csv(output / "paired_effects.csv", effects)
    (output / "report.md").write_text(
        _markdown_report(audit, summaries, effects), encoding="utf-8"
    )
    if audit["status"] != "valid":
        raise ValueError(
            f"MuJoCo pilot integrity audit failed with {len(audit['issues'])} issues"
        )
    return {
        "audit": audit,
        "replicate_rows": replicates,
        "method_summaries": summaries,
        "paired_effects": effects,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-protocol-version", required=True)
    parser.add_argument("--environments", default=",".join(DEFAULT_ENV_IDS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--optimizer-seeds", required=True)
    parser.add_argument(
        "--evaluation-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS))
    )
    parser.add_argument(
        "--disturbance-modes", default=",".join(DISTURBANCE_MODES)
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = write_analysis(
        args.run_dir,
        args.output_dir,
        expected_protocol_version=args.expected_protocol_version,
        environments=_parse_csv(args.environments),
        methods=_parse_csv(args.methods),
        optimizer_seeds=_parse_csv(args.optimizer_seeds, int),
        evaluation_seeds=_parse_csv(args.evaluation_seeds, int),
        disturbance_modes=_parse_csv(args.disturbance_modes),
    )
    print(
        "mujoco_pilot_audit "
        f"status={payload['audit']['status']} "
        f"cells={payload['audit']['observed_cell_count']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
