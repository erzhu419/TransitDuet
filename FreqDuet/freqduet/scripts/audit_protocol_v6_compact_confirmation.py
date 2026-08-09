#!/usr/bin/env python3
"""Audit the preregistered V8 compact-state primary confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_incremental_selection import (
    DEFAULT_MAIN,
    DEFAULT_REFERENCE,
    evaluate_selection,
    sha256_file,
)


DEFAULT_PRIMARY = "F_freqduet_protocol_v6_avlcompact_w2_hiro"
DEFAULT_SENSITIVITY = "F_freqduet_protocol_v6_avlcompact_w4_hiro"
DEFAULT_MATCHED_CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def evaluate_primary_confirmation(
    aggregate_dir: Path,
    *,
    selection_gate_path: Path,
    selection_manifest_path: Path | None = None,
    primary: str = DEFAULT_PRIMARY,
    sensitivity: str = DEFAULT_SENSITIVITY,
    matched_context: str = DEFAULT_MATCHED_CONTEXT,
    main: str = DEFAULT_MAIN,
    reference: str = DEFAULT_REFERENCE,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    selection_gate_path = Path(selection_gate_path).resolve()
    selection_gate = _load_json(selection_gate_path)
    embedded_manifest = Path(
        selection_gate["input_artifacts"]["manifest"]["path"])
    selection_manifest_path = Path(
        selection_manifest_path or embedded_manifest).resolve()
    confirmation_manifest_path = aggregate_dir / "matrix_manifest.json"
    selection_manifest = _load_json(selection_manifest_path)
    confirmation_manifest = _load_json(confirmation_manifest_path)

    expected_candidates = {primary, sensitivity}
    observed_candidates = set(selection_gate.get("passing_candidates", []))
    recorded_selection_manifest_sha = str(
        selection_gate["input_artifacts"]["manifest"]["sha256"])
    selection_source_sha = str(
        selection_gate["matrix_provenance"][
            "source_fingerprint_sha256"])
    selection_scenario_sha = str(
        selection_gate["matrix_provenance"][
            "scenario_contract_sha256"])
    confirmation_source_sha = str(
        confirmation_manifest.get("run_source_fingerprint", {}).get(
            "sha256"))
    confirmation_scenario_sha = str(
        confirmation_manifest.get("scenario_contract", {}).get("sha256"))
    selection_train_seeds = set(selection_manifest.get("train_seeds", []))
    selection_eval_seeds = set(selection_manifest.get("eval_seeds", []))
    confirmation_train_seeds = set(
        confirmation_manifest.get("train_seeds", []))
    confirmation_eval_seeds = set(
        confirmation_manifest.get("eval_seeds", []))

    lineage_checks = {
        "selection_gate_v3": (
            selection_gate.get("gate_version")
            == "freqduet-v6-incremental-selection-v3"),
        "selection_was_exploratory": (
            selection_gate.get("audit_stage") == "exploratory"),
        "selection_was_ambiguous": (
            selection_gate.get("status") == "ambiguous_multiple_passes"),
        "selection_had_no_chosen_candidate": (
            selection_gate.get("selected_candidate") is None),
        "passing_candidates_match_preregistration": (
            observed_candidates == expected_candidates),
        "matched_context_unchanged": (
            selection_gate.get("matched_context") == matched_context),
        "selection_manifest_hash_verified": (
            sha256_file(selection_manifest_path)
            == recorded_selection_manifest_sha),
        "selection_manifest_is_exploratory": (
            selection_manifest.get("stage") == "exploratory"
            and selection_manifest.get("independent_confirmation") is False),
        "confirmation_manifest_is_independent": (
            confirmation_manifest.get("stage") == "confirmation"
            and confirmation_manifest.get("independent_confirmation") is True),
        "train_seeds_are_disjoint": not (
            selection_train_seeds & confirmation_train_seeds),
        "eval_seeds_are_disjoint": not (
            selection_eval_seeds & confirmation_eval_seeds),
        "model_source_fingerprint_unchanged": (
            confirmation_source_sha == selection_source_sha),
        "scenario_contract_unchanged": (
            confirmation_scenario_sha == selection_scenario_sha),
        "confirmation_source_is_clean": (
            confirmation_manifest.get("run_git_provenance", {}).get(
                "tracked_dirty") is False),
    }
    if not all(lineage_checks.values()):
        raise ValueError(
            f"compact confirmation lineage checks failed: {lineage_checks}")

    common = {
        "main": main,
        "reference": reference,
        "matched_context": matched_context,
        "expected_stage": "confirmation",
    }
    primary_result = evaluate_selection(
        aggregate_dir, candidates=[primary], **common)
    sensitivity_result = evaluate_selection(
        aggregate_dir, candidates=[sensitivity], **common)
    primary_pass = primary_result["status"] == "unique_pass"
    sensitivity_pass = sensitivity_result["status"] == "unique_pass"
    return {
        "gate_version": "freqduet-v8-compact-primary-confirmation-v1",
        "status": (
            "primary_confirmed" if primary_pass
            else "primary_not_confirmed"),
        "primary_claim_eligible": primary_pass,
        "primary": primary,
        "sensitivity": sensitivity,
        "sensitivity_confirmed": sensitivity_pass,
        "sensitivity_can_rescue_primary": False,
        "matched_context": matched_context,
        "lineage_checks": lineage_checks,
        "selection_lineage": {
            "selection_gate_path": str(selection_gate_path),
            "selection_gate_sha256": sha256_file(selection_gate_path),
            "selection_manifest_path": str(selection_manifest_path),
            "selection_manifest_sha256": sha256_file(
                selection_manifest_path),
            "selection_train_seeds": sorted(selection_train_seeds),
            "selection_eval_seeds": sorted(selection_eval_seeds),
        },
        "confirmation_design": {
            "confirmation_manifest_path": str(
                confirmation_manifest_path),
            "confirmation_manifest_sha256": sha256_file(
                confirmation_manifest_path),
            "confirmation_train_seeds": sorted(confirmation_train_seeds),
            "confirmation_eval_seeds": sorted(confirmation_eval_seeds),
        },
        "primary_result": primary_result,
        "sensitivity_result": sensitivity_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--selection-gate", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--primary", default=DEFAULT_PRIMARY)
    parser.add_argument("--sensitivity", default=DEFAULT_SENSITIVITY)
    parser.add_argument("--matched-context", default=DEFAULT_MATCHED_CONTEXT)
    parser.add_argument("--main", default=DEFAULT_MAIN)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-primary-pass", action="store_true")
    args = parser.parse_args()

    result = evaluate_primary_confirmation(
        args.aggregate_dir,
        selection_gate_path=args.selection_gate,
        selection_manifest_path=args.selection_manifest,
        primary=args.primary,
        sensitivity=args.sensitivity,
        matched_context=args.matched_context,
        main=args.main,
        reference=args.reference,
    )
    out = args.out or Path(args.aggregate_dir) / "confirmation_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_primary_pass and not result["primary_claim_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
