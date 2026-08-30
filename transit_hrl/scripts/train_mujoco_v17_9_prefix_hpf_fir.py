#!/usr/bin/env python3
"""Run grouped v17.9 prefix-HPF FIR selection on reused server paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import mujoco_v17_9_prefix_hpf_fir_spec as spec  # noqa: E402
from scripts.mujoco_v17_8_causal_fir import (  # noqa: E402
    evaluate_causal_fir_prefix_split,
    fit_causal_fir,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    fit_final_models,
    load_reused_panel,
    reused_advancement_gate,
    selection_key,
    summarize_candidate,
)


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": (
                f"prefix_hpf_fir_w{int(window)}_"
                f"ridge{float(penalty):.0e}_gain1.00"
            ),
            "window": int(window),
            "ridge_penalty": float(penalty),
            "output_gain": spec.OUTPUT_GAIN,
            "router_mode": "prefix_hpf_innovation_projection",
        }
        for window in spec.FIR_WINDOWS
        for penalty in spec.RIDGE_PENALTIES
    ]


def grouped_out_of_fold_rows(
    panel: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    configs = candidate_configs()
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {
        str(config["candidate_id"]): [] for config in configs
    }
    configs_by_base: dict[tuple[int, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for config in configs:
        configs_by_base[
            (int(config["window"]), float(config["ridge_penalty"]))
        ].append(config)

    for environment in spec.ENVIRONMENTS:
        environment_rows = [
            row for row in panel if row["environment"] == environment
        ]
        for held_seed in spec.REUSED_SELECTION_SEEDS:
            fit_rows = [
                row for row in environment_rows
                if row["evaluation_seed"] != int(held_seed)
            ]
            held_rows = [
                row for row in environment_rows
                if row["evaluation_seed"] == int(held_seed)
            ]
            if (
                len(fit_rows)
                != (len(spec.REUSED_SELECTION_SEEDS) - 1)
                * len(spec.DISTURBANCE_MODES)
                or len(held_rows) != len(spec.DISTURBANCE_MODES)
            ):
                raise RuntimeError("v17.9 grouped fold construction mismatch")
            for (window, penalty), base_configs in configs_by_base.items():
                model = fit_causal_fir(
                    [row["total_action"] for row in fit_rows],
                    [row["oracle_upper_action"] for row in fit_rows],
                    window=window,
                    ridge_penalty=penalty,
                    feature_scale_floor=spec.FEATURE_SCALE_FLOOR,
                )
                for config in base_configs:
                    candidate = str(config["candidate_id"])
                    for row in held_rows:
                        metrics = evaluate_causal_fir_prefix_split(
                            row["total_action"],
                            model,
                            output_gain=spec.OUTPUT_GAIN,
                            upper_action_limit=spec.UPPER_ACTION_LIMIT,
                            lower_action_limit=spec.LOWER_ACTION_LIMIT,
                            upper_window=spec.UPPER_WINDOW,
                            lower_window=spec.LOWER_WINDOW,
                            upper_rms_budget=spec.UPPER_RMS_BUDGET,
                            lower_rms_budget=spec.LOWER_RMS_BUDGET,
                            power_tolerance=spec.POWER_TOLERANCE,
                        )
                        valid = bool(
                            metrics["finite"]
                            and metrics["reconstruction_error_max"]
                            <= spec.RECONSTRUCTION_TOLERANCE
                            and metrics["bound_violation_max"]
                            <= spec.BOUND_TOLERANCE
                            and metrics[
                                "prefix_upper_budget_feasible_rate"
                            ] == 1.0
                        )
                        recoverable = bool(
                            row["oracle_joint_feasible"]
                            and not row["baseline_joint_feasible"]
                        )
                        rows_by_candidate[candidate].append({
                            "candidate_id": candidate,
                            "environment": str(environment),
                            "disturbance_mode": row["disturbance_mode"],
                            "evaluation_seed": int(held_seed),
                            "fit_seed_count": len(
                                spec.REUSED_SELECTION_SEEDS
                            ) - 1,
                            "valid": valid,
                            "baseline_joint_feasible": bool(
                                row["baseline_joint_feasible"]
                            ),
                            "baseline_lower_power": float(
                                row["baseline_lower_power"]
                            ),
                            "oracle_joint_feasible": bool(
                                row["oracle_joint_feasible"]
                            ),
                            "oracle_recoverable_failure": recoverable,
                            "recovers_oracle_recoverable_failure": bool(
                                recoverable and metrics["joint_budget_pass"]
                            ),
                            "preserves_baseline_feasible_path": bool(
                                row["baseline_joint_feasible"]
                                and metrics["joint_budget_pass"]
                            ),
                            **metrics,
                        })
    for candidate, rows in rows_by_candidate.items():
        if len(rows) != spec.REUSED_EXPECTED_PATH_COUNT:
            raise RuntimeError(
                f"v17.9 OOF path count mismatch for {candidate}"
            )
    return rows_by_candidate


def run_selection(dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    panel = load_reused_panel(dataset_root)
    rows_by_candidate = grouped_out_of_fold_rows(panel)
    configs = {row["candidate_id"]: row for row in candidate_configs()}
    summaries = [
        summarize_candidate(configs[candidate], rows)
        for candidate, rows in rows_by_candidate.items()
    ]
    summaries.sort(key=lambda row: str(row["candidate_id"]))
    selected = max(summaries, key=selection_key)
    gate = reused_advancement_gate(selected)
    advances = bool(all(gate.values()))
    selected_rows = rows_by_candidate[str(selected["candidate_id"])]
    summary = {
        "status": (
            "prefix_hpf_fir_advances_to_fresh_path_validation"
            if advances
            else "prefix_hpf_fir_stops_before_fresh_path_access"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "dataset_root": str(dataset_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "candidate_count": len(summaries),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "advancement_gate": gate,
        "fresh_path_access_allowed": advances,
        "selected_out_of_fold_rows": selected_rows,
        "candidate_summaries": summaries,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    model = {
        "status": "v17_9_selected_prefix_hpf_fir_model_fitted",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selected_candidate_id": selected["candidate_id"],
        "window": int(selected["window"]),
        "ridge_penalty": float(selected["ridge_penalty"]),
        "output_gain": spec.OUTPUT_GAIN,
        "router_mode": "prefix_hpf_innovation_projection",
        "upper_action_limit": spec.UPPER_ACTION_LIMIT,
        "lower_action_limit": spec.LOWER_ACTION_LIMIT,
        "upper_window": spec.UPPER_WINDOW,
        "upper_rms_budget": spec.UPPER_RMS_BUDGET,
        "fresh_validation_eligible": advances,
        "environment_models": fit_final_models(panel, selected),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    return summary, model


def write_outputs(
    summary: dict[str, Any], model: dict[str, Any], output_dir: Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "selected_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary, model = run_selection(args.dataset_root)
    summary["source_identity"] = source_identity
    model["source_identity"] = source_identity
    write_outputs(summary, model, args.output_dir)
    print(
        f"DONE v17.9 selection status={summary['status']} "
        f"candidate={summary['selected_candidate_id']} "
        f"recovered={summary['selected_candidate']['recovered_failure_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
