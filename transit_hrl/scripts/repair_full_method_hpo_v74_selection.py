#!/usr/bin/env python3
"""Repair the v7.4 HPO selector by enforcing its frozen budget gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import (  # noqa: E402
    current_freq_hrl_source_manifest_sha256,
)
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo  # noqa: E402


REPAIR_PROTOCOL_VERSION = "v74_budget_eligible_selection_repair_v1"
FROZEN_ALGORITHM_REVISION = "5f54bb2323e5cbeb1a5beea6548324c01c131085"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "56c663db9a391a05dbab3e097c305de60b7e251a36451633dd4934853644aa30"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _verify_runtime(expected_revision: str) -> str:
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if revision != str(expected_revision).strip().lower():
        raise ValueError(
            f"selection-repair revision mismatch: expected {expected_revision}, "
            f"got {revision}"
        )
    git_root = Path(subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()).resolve()
    relative = str(Path(__file__).resolve().relative_to(git_root))
    subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--error-unmatch", relative],
        check=True,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["git", "-C", str(git_root), "diff", "--quiet", "HEAD", "--", relative]
    )
    if clean.returncode != 0:
        raise ValueError("selection-repair runtime does not match committed bytes")
    if current_freq_hrl_source_manifest_sha256() != FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError("selection repair is not running over the frozen v7.4 source")
    return revision


def select_budget_eligible_rows(
    leaderboard: Iterable[dict[str, Any]],
    *,
    variant_ids: Iterable[str] = hpo.HPO_VARIANT_IDS,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Select the best LCB row only after every frozen gate is satisfied."""

    rows = list(leaderboard)
    selected: dict[str, dict[str, Any]] = {}
    top_candidates: dict[str, list[str]] = {}
    for variant_id in map(str, variant_ids):
        ranked = sorted(
            (row for row in rows if str(row["variant_id"]) == variant_id),
            key=lambda row: int(row["rank"]),
        )
        eligible = [
            row for row in ranked
            if row["learning_gate_status"] == "eligible"
            and row["mechanism_activity_status"] in {
                "eligible", "not_applicable"
            }
            and row["training_budget_status"] == "sufficient"
        ]
        if not eligible:
            raise ValueError(
                f"no learning/mechanism/budget-eligible candidate for {variant_id}"
            )
        top_candidates[variant_id] = [
            str(row["candidate_id"]) for row in eligible[:2]
        ]
        winner = eligible[0]
        candidate_id = str(winner["candidate_id"])
        candidate = hpo.CANDIDATES_BY_ID[candidate_id]
        selected[variant_id] = {
            "candidate_id": candidate_id,
            "candidate_parameters": dict(candidate.parameters),
            "effective_parameters": hpo.effective_parameters_for_variant(
                variant_id, candidate_id
            ),
            "selection_source_variant": variant_id,
            "selection_rule": (
                "support_only_training_replicate_lcb_after_frozen_"
                "learning_mechanism_budget_gates"
            ),
            "robust_selection_score": float(winner["robust_selection_score"]),
            "learning_gate_status": str(winner["learning_gate_status"]),
            "mechanism_activity_status": str(
                winner["mechanism_activity_status"]
            ),
            "training_budget_status": str(winner["training_budget_status"]),
            "checkpoint_boundary_replicate_fraction": float(
                winner["checkpoint_boundary_replicate_fraction"]
            ),
        }
    return selected, top_candidates


def repair_merge_payload(
    raw_payload: dict[str, Any],
    *,
    runtime_revision: str,
    input_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the missing eligibility filter without changing scores or gates."""

    payload = copy.deepcopy(raw_payload)
    before = copy.deepcopy(payload["frozen_config"])
    if before.get("status") != "provisional_support_validation_only":
        raise ValueError("repair requires the preserved provisional v7.4 merge")
    if before.get("heldout_test_access_status") != "not_loaded" or before.get(
        "heldout_test_seeds"
    ):
        raise ValueError("selection repair refuses HPO data with held-out access")
    if before.get("code_revision") != FROZEN_ALGORITHM_REVISION:
        raise ValueError("selection repair input algorithm revision drifted")
    if before.get("source_manifest_sha256") != FROZEN_SOURCE_MANIFEST_SHA256:
        raise ValueError("selection repair input source manifest drifted")
    selected, top_candidates = select_budget_eligible_rows(
        payload["leaderboard"]
    )
    parent = selected[hpo.ABLATION_PARENT_VARIANT]
    parent_candidate = str(parent["candidate_id"])
    for variant in hpo.VARIANTS:
        if not variant.inherits_full_selection:
            continue
        selected[variant.variant_id] = {
            "candidate_id": parent_candidate,
            "candidate_parameters": dict(
                hpo.CANDIDATES_BY_ID[parent_candidate].parameters
            ),
            "effective_parameters": hpo.effective_parameters_for_variant(
                variant.variant_id, parent_candidate
            ),
            "selection_source_variant": hpo.ABLATION_PARENT_VARIANT,
            "selection_rule": "inherit_full_candidate_disable_one_mechanism",
            "learning_gate_status": parent["learning_gate_status"],
            "mechanism_activity_status": "not_applicable",
            "training_budget_status": parent["training_budget_status"],
            "checkpoint_boundary_replicate_fraction": parent[
                "checkpoint_boundary_replicate_fraction"
            ],
        }
    frozen = payload["frozen_config"]
    frozen.update({
        "status": "frozen_from_support_validation_only",
        "training_budget_status": "sufficient",
        "selected": selected,
        "top_candidates": top_candidates,
        "selection_repair_protocol_version": REPAIR_PROTOCOL_VERSION,
        "selection_repair_runtime_revision": str(runtime_revision).lower(),
        "selection_repair_runtime_sha256": _sha256(Path(__file__)),
        "selection_repair_input_manifest_sha256": input_manifest_sha256,
        "selection_repair_heldout_access_status": "not_loaded",
    })
    payload["summary"].update({
        "freeze_status": "frozen_from_support_validation_only",
        "training_budget_status": "sufficient",
        "budget_sufficient_selected_count": len(hpo.HPO_VARIANT_IDS),
        "selection_repair_protocol_version": REPAIR_PROTOCOL_VERSION,
        "selection_repair_status": "valid",
    })
    validation = hpo.validate_frozen_config(frozen)
    audit = {
        "status": "valid",
        "repair_protocol_version": REPAIR_PROTOCOL_VERSION,
        "runtime_revision": str(runtime_revision).lower(),
        "runtime_sha256": _sha256(Path(__file__)),
        "input_manifest_sha256": input_manifest_sha256,
        "heldout_access_status": "not_loaded",
        "original_freeze_status": before["status"],
        "repaired_freeze_status": frozen["status"],
        "original_selected": {
            variant_id: entry["candidate_id"]
            for variant_id, entry in before["selected"].items()
            if variant_id in hpo.HPO_VARIANT_IDS
        },
        "repaired_selected": {
            variant_id: entry["candidate_id"]
            for variant_id, entry in selected.items()
            if variant_id in hpo.HPO_VARIANT_IDS
        },
        "changed_variants": sorted(
            variant_id for variant_id in hpo.HPO_VARIANT_IDS
            if before["selected"][variant_id]["candidate_id"]
            != selected[variant_id]["candidate_id"]
        ),
        "frozen_config_validation": validation,
    }
    return payload, audit


def _input_manifest(cell_dirs: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for directory in sorted(cell_dirs):
        for name in (
            "cell_summary.json",
            "tuning_rows.csv",
            "hf_intervention_rows.csv",
        ):
            path = directory / name
            relative = path.relative_to(root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runtime-revision", required=True)
    args = parser.parse_args()
    runtime_revision = _verify_runtime(args.expected_runtime_revision)
    run = args.input_run.resolve()
    cell_dirs = sorted(
        path.parent for path in run.glob(
            "cells/*/*/replicate_*/cell_summary.json"
        )
    )
    if len(cell_dirs) != 210:
        raise ValueError(
            f"selection repair requires 210 HPO cells, found {len(cell_dirs)}"
        )
    candidates = sorted({
        candidate_id
        for variant_id in hpo.HPO_VARIANT_IDS
        for candidate_id in hpo.candidate_ids_for_variant(variant_id)
    })
    raw = hpo.merge_hpo_cells(
        cell_dirs,
        expected_variant_ids=list(hpo.HPO_VARIANT_IDS),
        expected_candidate_ids=candidates,
        expected_replicate_seeds=list(hpo.DEFAULT_FINAL_HPO_OPTIMIZER_SEEDS),
        top_k=2,
        stage="final",
    )
    manifest = _input_manifest(cell_dirs, run)
    repaired, audit = repair_merge_payload(
        raw,
        runtime_revision=runtime_revision,
        input_manifest_sha256=manifest,
    )
    output = args.output_dir.resolve()
    hpo.write_hpo_merge(output, repaired)
    (output / "selection_repair_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "v74_hpo_selection_repair status=valid "
        f"changed={','.join(audit['changed_variants'])} output={output}"
    )


if __name__ == "__main__":
    main()
