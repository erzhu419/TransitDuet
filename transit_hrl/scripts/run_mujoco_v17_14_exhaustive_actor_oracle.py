#!/usr/bin/env python3
"""Evaluate the unexamined remainder of the v17.13 actor-adapter grid."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import mujoco_v17_14_exhaustive_actor_oracle_spec as spec  # noqa: E402
from scripts.train_mujoco_v17_13_causal_actor_adapter import (  # noqa: E402
    attach_actor_targets,
    build_statistics_cache,
    candidate_configs,
    fit_final_models,
    grouped_candidate_rows,
    reused_advancement_gate,
    selection_key,
    summarize_candidate,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def load_v17_13_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("v17.13 source summary must be a JSON object")
    prefilter = list(summary.get("prefilter_candidate_summaries") or [])
    exact = list(summary.get("full_oracle_candidate_summaries") or [])
    selected_rows = list(summary.get("selected_out_of_fold_rows") or [])
    prefilter_ids = {str(row.get("candidate_id")) for row in prefilter}
    exact_ids = {str(row.get("candidate_id")) for row in exact}
    exact_gains = sorted({float(row.get("output_gain")) for row in exact})
    source_identity = dict(summary.get("source_identity") or {})
    if (
        summary.get("status")
        != "causal_actor_adapter_stops_before_fresh_path_access"
        or summary.get("development_protocol_version")
        != "mujoco_v17_13_causal_actor_adapter_v1"
        or int(summary.get("candidate_count", -1))
        != spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT
        or int(summary.get("full_oracle_candidate_count", -1))
        != spec.EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT
        or len(prefilter) != spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT
        or len(prefilter_ids) != spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT
        or len(exact) != spec.EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT
        or len(exact_ids) != spec.EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT
        or not exact_ids.issubset(prefilter_ids)
        or exact_gains != [0.5, 1.0]
        or len(selected_rows) != spec.v17_13.EXPECTED_PATH_COUNT
        or summary.get("fresh_validation_paths_accessed") is not False
        or summary.get("fresh_path_access_allowed") is not False
        or source_identity.get("source_identity_status") != "verified"
        or source_identity.get("code_revision")
        != spec.v17_13.FROZEN_CORE_REVISION
        or source_identity.get("source_manifest_sha256")
        != spec.v17_13.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v17.13 source summary failed frozen validation")
    registered_selected = max(
        sorted(exact, key=lambda row: str(row["candidate_id"])),
        key=selection_key,
    )
    if registered_selected["candidate_id"] != summary["selected_candidate_id"]:
        raise ValueError("v17.13 registered selection ordering drifted")
    return summary


def remainder_candidate_configs(
    v17_13_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    examined = {
        str(row["candidate_id"])
        for row in v17_13_summary["full_oracle_candidate_summaries"]
    }
    configs = [
        row for row in candidate_configs()
        if str(row["candidate_id"]) not in examined
    ]
    configs.sort(key=lambda row: str(row["candidate_id"]))
    if (
        len(configs) != spec.EXPECTED_REMAINDER_CANDIDATE_COUNT
        or len(examined) != spec.EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT
        or len({str(row["candidate_id"]) for row in configs}) != len(configs)
    ):
        raise RuntimeError("v17.14 remainder candidate partition mismatch")
    return configs


def run_audit(
    dataset_root: Path,
    target_root: Path,
    v17_13_summary_path: Path,
    *,
    oracle_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    source_summary = load_v17_13_summary(v17_13_summary_path)
    panel = attach_actor_targets(load_reused_panel(dataset_root), target_root)
    statistics_cache = build_statistics_cache(panel)
    remainder = remainder_candidate_configs(source_summary)
    new_summaries = []
    best_new_summary = None
    best_new_rows = None
    worker_count = int(oracle_workers)
    if worker_count < 1:
        raise ValueError("v17.14 oracle worker count must be positive")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for index, config in enumerate(remainder, start=1):
            rows = grouped_candidate_rows(
                panel, config, statistics_cache, executor
            )
            candidate = summarize_candidate(config, rows)
            new_summaries.append(candidate)
            if (
                best_new_summary is None
                or selection_key(candidate) > selection_key(best_new_summary)
            ):
                best_new_summary = candidate
                best_new_rows = rows
            if (
                index % spec.PROGRESS_INTERVAL == 0
                or index == len(remainder)
            ):
                best_floor = best_new_summary[
                    "actor_floor_recovered_path_count"
                ]
                best_joint = best_new_summary[
                    "corrected_joint_feasible_path_count"
                ]
                print(
                    f"PROGRESS v17.14 {index}/{len(remainder)} "
                    f"best_floor={best_floor} best_joint={best_joint}",
                    flush=True,
                )
    prior_summaries = [
        dict(row)
        for row in source_summary["full_oracle_candidate_summaries"]
    ]
    combined = [*prior_summaries, *new_summaries]
    combined.sort(key=lambda row: str(row["candidate_id"]))
    if (
        len(combined) != spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT
        or len({str(row["candidate_id"]) for row in combined}) != len(combined)
    ):
        raise RuntimeError("v17.14 combined full-grid frontier mismatch")
    selected = max(combined, key=selection_key)
    if selected["candidate_id"] == source_summary["selected_candidate_id"]:
        selected_rows = source_summary["selected_out_of_fold_rows"]
    elif (
        best_new_summary is not None
        and selected["candidate_id"] == best_new_summary["candidate_id"]
        and best_new_rows is not None
    ):
        selected_rows = best_new_rows
    else:
        raise RuntimeError("v17.14 selected rows are unavailable")
    floor = [row for row in panel if row["actor_floor"]]
    target_nonzero_count = sum(
        row["target_executed_correction_rms"]
        >= spec.v17_13.EXECUTED_CORRECTION_RMS_MIN_GATE
        for row in floor
    )
    gate = reused_advancement_gate(selected)
    gate["all_actor_floor_targets_change_executed_action"] = bool(
        target_nonzero_count == spec.v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT
    )
    advances = bool(all(gate.values()))
    recovery_distribution = Counter(
        int(row["actor_floor_recovered_path_count"]) for row in combined
    )
    by_gain = {}
    for gain in spec.v17_13.OUTPUT_GAINS:
        gain_rows = [
            row for row in combined if float(row["output_gain"]) == float(gain)
        ]
        by_gain[str(gain)] = {
            "candidate_count": len(gain_rows),
            "maximum_actor_floor_recovered_path_count": max(
                int(row["actor_floor_recovered_path_count"])
                for row in gain_rows
            ),
            "maximum_corrected_joint_feasible_path_count": max(
                int(row["corrected_joint_feasible_path_count"])
                for row in gain_rows
            ),
        }
    passing_candidates = [
        str(row["candidate_id"])
        for row in combined if all(reused_advancement_gate(row).values())
    ]
    config_by_id = {
        str(row["candidate_id"]): row for row in candidate_configs()
    }
    selected_config = config_by_id[str(selected["candidate_id"])]
    summary = {
        "status": (
            "exhaustive_actor_oracle_authorizes_closed_loop_fresh_validation"
            if advances
            else "exhaustive_actor_oracle_closes_frozen_linear_fir_grid"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_v17_13_selection_run": spec.SOURCE_V17_13_SELECTION_RUN,
        "dataset_root": str(dataset_root),
        "target_root": str(target_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "v17_13_exact_candidate_count": len(prior_summaries),
        "new_exact_candidate_count": len(new_summaries),
        "combined_exact_candidate_count": len(combined),
        "oracle_worker_count": worker_count,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "selected_out_of_fold_rows": selected_rows,
        "advancement_gate": gate,
        "passing_candidate_count": len(passing_candidates),
        "passing_candidate_ids": passing_candidates,
        "full_grid_frontier": {
            "candidate_count_by_actor_floor_recovery": {
                str(key): int(value)
                for key, value in sorted(recovery_distribution.items())
            },
            "maximum_actor_floor_recovered_path_count": max(
                int(row["actor_floor_recovered_path_count"])
                for row in combined
            ),
            "maximum_corrected_joint_feasible_path_count": max(
                int(row["corrected_joint_feasible_path_count"])
                for row in combined
            ),
            "all_reference_feasible_preserved_candidate_count": sum(
                int(row["reference_feasible_preserved_path_count"])
                == spec.v17_13.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
                for row in combined
            ),
            "by_output_gain": by_gain,
        },
        "combined_exact_candidate_summaries": combined,
        "fresh_path_access_allowed": advances,
        "fresh_validation_paths_accessed": False,
        "frozen_linear_fir_grid_closed": not advances,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    model = {
        "status": "v17_14_selected_full_grid_causal_actor_adapter_fitted",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selected_candidate_id": selected["candidate_id"],
        **{
            key: selected_config[key]
            for key in (
                "window",
                "ridge_penalty",
                "actor_floor_path_weight",
                "output_gain",
                "correction_abs_limit",
            )
        },
        "component_sum_limit": spec.v17_13.COMPONENT_SUM_LIMIT,
        "executed_action_limit": spec.v17_13.EXECUTED_ACTION_LIMIT,
        "fresh_validation_eligible": advances,
        "environment_models": fit_final_models(
            panel, selected_config, statistics_cache
        ),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    return summary, model


def write_outputs(
    summary: dict[str, Any], model: dict[str, Any], output_dir: Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "exhaustive_oracle_summary.json").write_text(
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
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--v17-13-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-workers", type=int, default=1)
    args = parser.parse_args()
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary, model = run_audit(
        args.dataset_root,
        args.target_root,
        args.v17_13_summary,
        oracle_workers=int(args.oracle_workers),
    )
    summary["source_identity"] = source_identity
    model["source_identity"] = source_identity
    write_outputs(summary, model, args.output_dir)
    selected = summary["selected_candidate"]
    print(
        f"DONE v17.14 status={summary['status']} "
        f"candidate={summary['selected_candidate_id']} "
        f"floor={selected['actor_floor_recovered_path_count']}/"
        f"{spec.v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT} "
        f"preserved={selected['reference_feasible_preserved_path_count']}/"
        f"{spec.v17_13.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
