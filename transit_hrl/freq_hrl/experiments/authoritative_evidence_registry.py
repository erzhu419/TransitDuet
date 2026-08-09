"""Fail-closed evidence registry for the Freq-HRL manuscript.

The project historically accumulated several independent claim matrices.  This
module provides one auditable reporting boundary: an artifact is not eligible
for manuscript claims unless it is registered here, hash verified, and assigned
an explicit paper disposition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "freq_hrl_authoritative_evidence_registry_v1"
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = Path("transit_hrl/evidence/authoritative_registry_v1.json")
DEFAULT_OUTPUT_DIR = Path(
    "transit_hrl/results/authoritative_evidence_registry_latest"
)
DEFAULT_MD_OUTPUT = Path(
    "transit_hrl/md/freq_hrl_authoritative_claim_ledger_2026-08-09.md"
)

PAPER_DISPOSITIONS = {
    "positive_main_or_si",
    "mixed_or_negative_main_or_si",
    "development_only",
    "excluded_legacy",
}
EVIDENCE_STAGES = {"confirmatory", "development", "legacy"}
CONFIRMATORY_DECISIONS = {"supported", "mixed", "not_supported"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_registry(path: Path) -> dict[str, Any]:
    registry = _read_json(Path(path))
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("authoritative evidence registry schema mismatch")
    return registry


def _verify_artifacts(
    record: dict[str, Any], repository_root: Path
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{record.get('evidence_id')}: no source artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact declarations must be JSON objects")
        role = str(artifact.get("role", ""))
        relative = Path(str(artifact.get("path", "")))
        expected = str(artifact.get("sha256", ""))
        if not role or role in resolved:
            raise ValueError(
                f"{record.get('evidence_id')}: duplicate or empty artifact role"
            )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{record.get('evidence_id')}: artifact path must be repository-relative"
            )
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if len(expected) != 64 or actual != expected:
            raise ValueError(
                f"{record.get('evidence_id')}: SHA-256 mismatch for {relative}"
            )
        resolved[role] = path
    return resolved


def _mujoco_v12_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    rows = list(decision.get("environment_rows") or [])
    if (
        decision.get("status") != "confirmatory_supported"
        or decision.get("primary_gate_pass") is not True
        or decision.get("integrity_status") != "valid"
        or len(rows) != 3
    ):
        raise ValueError("MuJoCo v12 registered decision no longer matches")
    return {
        "decision_status": "supported",
        "source_decision_status": str(decision["status"]),
        "primary_gate_pass": True,
        "primary_gate_count": int(decision["primary_gate_count"]),
        "independent_optimizer_replicates_per_arm": int(
            decision["optimizer_replicates_per_environment_arm"]
        ),
        "heldout_paths_per_cell": int(decision["heldout_paths_per_cell"]),
        "environments": [
            {
                "environment": str(row["environment"]),
                "return_difference": float(row["episode_return_difference"]),
                "return_ci95": [
                    float(row["episode_return_difference_ci95_lower"]),
                    float(row["episode_return_difference_ci95_upper"]),
                ],
                "responsibility_drift_reduction": float(
                    row["relative_drift_reduction"]
                ),
                "familywise_drift_reduction_lower": float(
                    row["drift_reduction_familywise_lower_bound"]
                ),
            }
            for row in rows
        ],
    }


def _mujoco_v13_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    rows = list(decision.get("environment_rows") or [])
    if (
        decision.get("status") != "confirmatory_primary_gate_failed"
        or decision.get("primary_gate_pass") is not False
        or decision.get("integrity_status") != "valid"
        or len(rows) != 3
    ):
        raise ValueError("MuJoCo v13 registered decision no longer matches")
    failed = []
    for row in rows:
        gates = {
            "return_noninferiority": bool(row["return_noninferiority_pass"]),
            "responsibility_drift": bool(
                row["minimum_responsibility_drift_reduction_pass"]
            ),
            "raw_lower_drift": bool(
                row["minimum_raw_lower_drift_reduction_pass"]
            ),
            "upper_hf_budget": bool(row["upper_hf_budget_pass"]),
        }
        failed.append({
            "environment": str(row["environment"]),
            "environment_gate_pass": bool(row["environment_primary_gate_pass"]),
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "return_difference": float(row["episode_return_difference"]),
            "raw_lower_drift_reduction": float(
                row["relative_raw_lower_drift_reduction"]
            ),
            "upper_hf_rms_familywise_upper": float(
                row["upper_hf_rms_familywise_upper_bound"]
            ),
        })
    if all(row["environment_gate_pass"] for row in failed):
        raise ValueError("MuJoCo v13 must retain its registered failed gate")
    return {
        "decision_status": "not_supported",
        "source_decision_status": str(decision["status"]),
        "primary_gate_pass": False,
        "primary_gate_count": int(decision["primary_gate_count"]),
        "independent_optimizer_replicates_per_arm": int(
            decision["optimizer_replicates_per_environment_arm"]
        ),
        "heldout_paths_per_cell": int(decision["heldout_paths_per_cell"]),
        "environments": failed,
    }


def _quant_v74_facts(paths: dict[str, Path]) -> dict[str, Any]:
    summary = _read_json(paths["summary"])
    rows = [
        row for row in _read_csv(paths["paired_effects"])
        if row.get("hypothesis_role") == "primary_baseline"
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["claim_status"])
        counts[status] = counts.get(status, 0) + 1
    expected = {
        "supported_improvement": int(
            summary["primary_supported_improvement_count"]
        ),
        "supported_harm": int(summary["primary_supported_harm_count"]),
    }
    if (
        summary.get("status") != "valid"
        or len(rows) != int(summary["primary_hypothesis_count"])
        or counts.get("supported_improvement", 0)
        != expected["supported_improvement"]
        or counts.get("supported_harm", 0) != expected["supported_harm"]
    ):
        raise ValueError("Quant v7.4 registered primary analysis no longer matches")
    primary_rows = []
    for row in rows:
        primary_rows.append({
            "comparator": str(row["comparator_variant_id"]),
            "metric": str(row["metric"]),
            "directional_improvement_mean": float(
                row["directional_improvement_mean"]
            ),
            "ci95": [
                float(row["directional_ci95_low"]),
                float(row["directional_ci95_high"]),
            ],
            "effect_size_dz": float(row["paired_effect_size_dz"]),
            "holm_p": float(row["p_value_holm"]),
            "status": str(row["claim_status"]),
        })
    return {
        "decision_status": "mixed",
        "primary_hypothesis_count": len(rows),
        "primary_status_counts": {
            "supported_improvement": counts.get("supported_improvement", 0),
            "supported_harm": counts.get("supported_harm", 0),
            "inconclusive": counts.get("inconclusive", 0),
        },
        "independent_training_replicates": int(
            summary["independent_training_replicates"]
        ),
        "heldout_paths_per_replicate": int(
            summary["heldout_paths_per_replicate"]
        ),
        "scenario_count": int(summary["scenario_count"]),
        "primary_rows": primary_rows,
    }


def _mujoco_v14_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("selected_arm") is not None
    ):
        raise ValueError("MuJoCo v14 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": dict(decision.get("arm_status") or {}),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
    }


def _mujoco_v14_1_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or not arm_status
        or any(
            int(status.get("passed_gate_count", -1)) != 0
            or int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
    ):
        raise ValueError("MuJoCo v14.1 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
    }


def _mujoco_v14_2_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_2_physical_router_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or len(arm_status) != 8
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 2
    ):
        raise ValueError("MuJoCo v14.2 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": max(
            int(status["passed_gate_count"])
            for status in arm_status.values()
        ),
    }


PARSERS = {
    "mujoco_v12": _mujoco_v12_facts,
    "mujoco_v13": _mujoco_v13_facts,
    "quant_v74": _quant_v74_facts,
    "mujoco_v14": _mujoco_v14_facts,
    "mujoco_v14_1": _mujoco_v14_1_facts,
    "mujoco_v14_2": _mujoco_v14_2_facts,
    "opaque_legacy": lambda paths: {
        "decision_status": "excluded_legacy",
        "artifact_count": len(paths),
    },
}


def _paper_eligibility(record: dict[str, Any]) -> tuple[bool, bool]:
    disposition = str(record["paper_disposition"])
    stage = str(record["evidence_stage"])
    decision = str(record["decision"])
    integrity = str(record["integrity_status"])
    selection = str(record["selection_access"])
    confirmatory_ready = bool(
        stage == "confirmatory"
        and integrity == "valid"
        and selection == "frozen_before_heldout"
        and decision in CONFIRMATORY_DECISIONS
    )
    reportable = bool(
        confirmatory_ready
        and disposition in {
            "positive_main_or_si",
            "mixed_or_negative_main_or_si",
        }
    )
    positive = bool(
        reportable
        and disposition == "positive_main_or_si"
        and decision == "supported"
    )
    return reportable, positive


def validate_registry(
    registry: dict[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("authoritative evidence registry schema mismatch")
    policy = registry.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("registry policy is missing")
    if policy.get("unregistered_artifacts") != "excluded":
        raise ValueError("the authoritative registry must fail closed")
    records = registry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("the authoritative registry has no records")
    seen: set[str] = set()
    enriched: list[dict[str, Any]] = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("registry records must be JSON objects")
        record = dict(source)
        evidence_id = str(record.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            raise ValueError("registry evidence_id values must be unique")
        seen.add(evidence_id)
        if record.get("evidence_stage") not in EVIDENCE_STAGES:
            raise ValueError(f"{evidence_id}: invalid evidence stage")
        if record.get("paper_disposition") not in PAPER_DISPOSITIONS:
            raise ValueError(f"{evidence_id}: invalid paper disposition")
        parser_name = str(record.get("parser", ""))
        if parser_name not in PARSERS:
            raise ValueError(f"{evidence_id}: unknown evidence parser")
        paths = _verify_artifacts(record, Path(repository_root))
        facts = PARSERS[parser_name](paths)
        if str(facts["decision_status"]) != str(record["decision"]):
            raise ValueError(f"{evidence_id}: registered decision drifted")
        reportable, positive = _paper_eligibility(record)
        disposition = str(record["paper_disposition"])
        if disposition == "positive_main_or_si" and not positive:
            raise ValueError(
                f"{evidence_id}: positive disposition lacks confirmatory support"
            )
        if disposition == "mixed_or_negative_main_or_si" and not reportable:
            raise ValueError(
                f"{evidence_id}: reportable negative/mixed result is not valid"
            )
        if disposition == "development_only" and record["evidence_stage"] != "development":
            raise ValueError(f"{evidence_id}: development disposition/stage mismatch")
        if disposition == "excluded_legacy" and reportable:
            raise ValueError(f"{evidence_id}: legacy evidence cannot be reportable")
        record.update({
            "artifact_hashes_verified": True,
            "manuscript_reportable": reportable,
            "positive_claim_supported": positive,
            "facts": facts,
        })
        enriched.append(record)
    return enriched


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "evidence_id",
        "claim_id",
        "domain",
        "evidence_stage",
        "decision",
        "paper_disposition",
        "manuscript_reportable",
        "positive_claim_supported",
        "allowed_wording",
        "forbidden_wording",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_registry_outputs(
    *,
    registry_path: Path,
    repository_root: Path,
    output_dir: Path,
    md_output: Path,
) -> dict[str, Any]:
    registry = load_registry(Path(registry_path))
    records = validate_registry(registry, Path(repository_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": _sha256(Path(registry_path)),
        "policy": registry["policy"],
        "record_count": len(records),
        "reportable_record_count": sum(
            bool(row["manuscript_reportable"]) for row in records
        ),
        "positive_supported_record_count": sum(
            bool(row["positive_claim_supported"]) for row in records
        ),
        "mixed_or_negative_record_count": sum(
            row["paper_disposition"] == "mixed_or_negative_main_or_si"
            for row in records
        ),
        "development_record_count": sum(
            row["paper_disposition"] == "development_only" for row in records
        ),
        "excluded_legacy_record_count": sum(
            row["paper_disposition"] == "excluded_legacy" for row in records
        ),
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "records.csv", records)
    lines = [
        "# Freq-HRL Authoritative Evidence Ledger",
        "",
        "Date: 2026-08-09",
        "",
        "This is the only manuscript claim ledger. Unregistered artifacts and the old independent claim generators are excluded by default.",
        "",
        "| Evidence | Domain | Stage | Decision | Paper use | Positive claim |",
        "|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['evidence_id']} | {row['domain']} | "
            f"{row['evidence_stage']} | {row['decision']} | "
            f"{row['paper_disposition']} | "
            f"{str(bool(row['positive_claim_supported'])).lower()} |"
        )
    lines.extend(["", "## Allowed Wording", ""])
    for row in records:
        lines.extend([
            f"### {row['evidence_id']}",
            "",
            str(row["allowed_wording"]),
            "",
            f"Forbidden: {row['forbidden_wording']}",
            "",
        ])
    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    md_output.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    summary = build_registry_outputs(
        registry_path=args.registry,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        md_output=args.md_output,
    )
    print(
        "authoritative_evidence_registry "
        f"records={summary['record_count']} "
        f"reportable={summary['reportable_record_count']} "
        f"positive={summary['positive_supported_record_count']}"
    )


if __name__ == "__main__":
    main()
