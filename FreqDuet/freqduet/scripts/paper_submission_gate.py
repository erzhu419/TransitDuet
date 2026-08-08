"""Shared fail-fast gate for paper artifacts that are not submission-ready."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from scripts.analysis_provenance import validate_csv_artifact
from scripts.decide_freqduet_protocol_v6_screen import (
    ARTIFACT_PRIMARY_KEYS,
    CONFIGS,
    DECISION_CONTRACT,
    DECISION_CONTRACT_SHA256,
    REFERENCE,
)
from scripts.run_freqduet_protocol_v2_matrix import (
    analysis_fingerprint,
    scenario_contract,
    source_fingerprint,
)

READY_STATUS = "ready_protocol_v6_confirmed"
HISTORICAL_HOLD_STATUS = "hold_pending_protocol_v5"
ACTIVE_PROTOCOL = "freqduet-eval-v6"
PRIMARY_METRIC = "restricted_total_journey_horizon_min"
MATRIX_MANIFEST_VERSION = "freqduet-matrix-manifest-v2"
CONFIRMATION_DECISION_STATUS = "confirmation_supported"
REQUIRED_MATRIX_ARTIFACTS = (
    "frozen_per_eval.csv",
    "frozen_summary.csv",
    "frozen_paired_deltas.csv",
)
ARTIFACT_BINDING_FIELDS = (
    "sha256", "size_bytes", "n_rows", "columns",
    "primary_key", "primary_key_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def read_submission_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"paper manifest must be a mapping: {path}")
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"submission manifest requires mapping {label!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise RuntimeError(f"submission manifest has invalid SHA256 for {label}")
    return digest


def _artifact_path(
    item: Mapping[str, Any], *, label: str, manifest_path: Path
) -> tuple[Path, str]:
    raw_path = str(item.get("path", "")).strip()
    if not raw_path:
        raise RuntimeError(f"submission manifest requires {label}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"submission {label} does not exist: {path}")
    expected = _require_sha256(item.get("sha256"), f"{label}.sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"submission {label} SHA256 mismatch: expected {expected}, "
            f"observed {actual}"
        )
    return path, actual


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    return _require_mapping(payload, label)


def _validate_matrix_artifacts(
    matrix: Mapping[str, Any], *, matrix_path: Path
) -> Mapping[str, Any]:
    artifacts = _require_mapping(
        matrix.get("artifacts"), "confirmation matrix artifacts"
    )
    for filename in REQUIRED_MATRIX_ARTIFACTS:
        record = _require_mapping(
            artifacts.get(filename),
            f"confirmation matrix artifact {filename}",
        )
        artifact_path = (matrix_path.parent / filename).resolve()
        try:
            validate_csv_artifact(
                artifact_path,
                dict(record),
                expected_primary_key=ARTIFACT_PRIMARY_KEYS[filename],
            )
        except ValueError as exc:
            raise RuntimeError(
                f"confirmation matrix artifact validation failed for "
                f"{filename}: {exc}") from exc
    return artifacts


def _validate_matrix_manifest(
    matrix: Mapping[str, Any], *, matrix_path: Path, source_commit: str
) -> None:
    expected_fields = {
        "manifest_version": MATRIX_MANIFEST_VERSION,
        "protocol_version": ACTIVE_PROTOCOL,
        "stage": "confirmation",
    }
    for field, expected in expected_fields.items():
        if str(matrix.get(field, "")).strip() != expected:
            raise RuntimeError(
                f"confirmation matrix {field}={matrix.get(field)!r}, "
                f"expected {expected!r}"
            )
    for field in (
        "strict_complete",
        "run_manifests_verified",
        "common_random_numbers_verified",
        "independent_confirmation",
    ):
        if matrix.get(field) is not True:
            raise RuntimeError(
                f"confirmation matrix requires {field}=true"
            )
    configs = matrix.get("configs")
    if (not isinstance(configs, list)
            or len(configs) != len(CONFIGS)
            or set(str(value) for value in configs) != set(CONFIGS)):
        raise RuntimeError("confirmation matrix does not use locked V6 configs")
    if matrix.get("reference") != REFERENCE:
        raise RuntimeError("confirmation matrix reference is not locked V6 main")
    if matrix.get("primary_metric") != PRIMARY_METRIC:
        raise RuntimeError("confirmation matrix primary metric is not locked")
    train_seeds = matrix.get("train_seeds")
    eval_seeds = matrix.get("eval_seeds")
    if (not isinstance(train_seeds, list) or not train_seeds
            or not isinstance(eval_seeds, list) or not eval_seeds
            or len(train_seeds) != len(set(train_seeds))
            or len(eval_seeds) != len(set(eval_seeds))
            or set(train_seeds) & set(eval_seeds)):
        raise RuntimeError("confirmation matrix seed contract is invalid")
    train_episodes = matrix.get("train_episodes")
    checkpoint_ep = matrix.get("checkpoint_ep")
    if (not isinstance(train_episodes, int) or train_episodes <= 0
            or checkpoint_ep != train_episodes - 1):
        raise RuntimeError("confirmation matrix checkpoint contract is invalid")
    expected_rollouts = len(CONFIGS) * len(train_seeds) * len(eval_seeds)
    if matrix.get("expected_rollouts") != expected_rollouts:
        raise RuntimeError("confirmation matrix rollout grid is incomplete")
    locked_fingerprints = {
        "run_source_fingerprint": source_fingerprint(),
        "scenario_contract": scenario_contract(REFERENCE),
        "analysis_fingerprint": analysis_fingerprint(),
    }
    for label, current in locked_fingerprints.items():
        item = _require_mapping(matrix.get(label), f"matrix {label}")
        _require_sha256(item.get("sha256"), f"matrix {label}.sha256")
        if dict(item) != current:
            raise RuntimeError(
                f"confirmation matrix {label} does not match current "
                "locked source"
            )
    if matrix.get("launch_analysis_sha256") != locked_fingerprints[
            "analysis_fingerprint"]["sha256"]:
        raise RuntimeError(
            "confirmation matrix launch analysis differs from aggregation"
        )
    run_git = _require_mapping(
        matrix.get("run_git_provenance"), "matrix run_git_provenance")
    if str(run_git.get("commit", "")).strip().lower() != source_commit:
        raise RuntimeError(
            "confirmation run Git commit does not match source")
    if run_git.get("tracked_dirty") is not False:
        raise RuntimeError("confirmation runs used tracked-dirty source")
    git = _require_mapping(matrix.get("git"), "matrix git")
    if str(git.get("commit", "")).strip().lower() != source_commit:
        raise RuntimeError("confirmation matrix git commit does not match source")
    if git.get("tracked_dirty") is not False:
        raise RuntimeError("confirmation matrix source was tracked-dirty")
    _validate_matrix_artifacts(matrix, matrix_path=matrix_path)


def _validate_ready_manifest(
    manifest: Mapping[str, Any], *, manifest_path: str | Path | None
) -> None:
    if manifest_path is None:
        raise RuntimeError(
            "ready submission validation requires the paper manifest path"
        )
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise RuntimeError(f"paper manifest does not exist: {manifest_file}")

    if str(manifest.get("active_protocol", "")).strip() != ACTIVE_PROTOCOL:
        raise RuntimeError(
            f"ready submission requires active_protocol={ACTIVE_PROTOCOL!r}"
        )
    version = str(manifest.get("version", "")).strip()
    if not version or "historical" in version.lower():
        raise RuntimeError(
            "ready submission requires a non-historical manifest version"
        )
    active_metrics = _require_mapping(
        manifest.get("active_metrics"), "active_metrics"
    )
    if str(active_metrics.get("primary", "")).strip() != PRIMARY_METRIC:
        raise RuntimeError(
            f"ready submission requires primary metric {PRIMARY_METRIC!r}"
        )

    source_commit = str(manifest.get("active_source_commit", "")).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(source_commit):
        raise RuntimeError(
            "ready submission requires a full 40-character active_source_commit"
        )

    confirmation = _require_mapping(
        manifest.get("confirmation"), "confirmation"
    )
    if str(confirmation.get("status", "")).strip() != "confirmed":
        raise RuntimeError("ready submission requires confirmation.status='confirmed'")
    if str(confirmation.get("stage", "")).strip() != "confirmation":
        raise RuntimeError(
            "ready submission requires confirmation.stage='confirmation'"
        )
    if str(confirmation.get("source_commit", "")).strip().lower() != source_commit:
        raise RuntimeError(
            "confirmation source_commit must match active_source_commit"
        )
    decision_spec = _require_mapping(
        confirmation.get("decision"), "confirmation.decision"
    )
    source_manifest_spec = _require_mapping(
        confirmation.get("source_manifest"), "confirmation.source_manifest"
    )
    decision_path, _decision_sha256 = _artifact_path(
        decision_spec,
        label="confirmation.decision",
        manifest_path=manifest_file,
    )
    matrix_path, matrix_sha256 = _artifact_path(
        source_manifest_spec,
        label="confirmation.source_manifest",
        manifest_path=manifest_file,
    )

    decision = _read_json_object(
        decision_path, "confirmation decision payload"
    )
    expected_decision = {
        "decision_contract": DECISION_CONTRACT,
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "status": CONFIRMATION_DECISION_STATUS,
        "protocol": ACTIVE_PROTOCOL,
        "stage": "confirmation",
        "primary_metric": PRIMARY_METRIC,
    }
    for key, expected in expected_decision.items():
        observed = str(decision.get(key, "")).strip().lower()
        if observed != str(expected).lower():
            raise RuntimeError(
                f"confirmation decision {key}={decision.get(key)!r}, "
                f"expected {expected!r}"
            )
    if "candidate_config" in decision:
        raise RuntimeError(
            "development/candidate decisions are never submission-ready"
        )
    selected_config = str(decision.get("selected_config", "")).strip()
    if not selected_config:
        raise RuntimeError("confirmation decision requires selected_config")
    if selected_config not in CONFIGS:
        raise RuntimeError("confirmation decision selected_config is not locked")

    matrix_binding = _require_mapping(
        decision.get("matrix_manifest"),
        "confirmation decision matrix_manifest",
    )
    bound_matrix_sha256 = _require_sha256(
        matrix_binding.get("sha256"),
        "confirmation decision matrix_manifest.sha256",
    )
    if bound_matrix_sha256 != matrix_sha256:
        raise RuntimeError(
            "confirmation decision matrix_manifest_sha256 does not match "
            "confirmation.source_manifest"
        )
    if str(matrix_binding.get("manifest_version", "")).strip() != (
        MATRIX_MANIFEST_VERSION
    ):
        raise RuntimeError(
            "confirmation decision matrix manifest version is not locked V2"
        )

    matrix = _read_json_object(
        matrix_path, "confirmation matrix manifest"
    )
    _validate_matrix_manifest(
        matrix,
        matrix_path=matrix_path,
        source_commit=source_commit,
    )
    decision_artifacts = _require_mapping(
        decision.get("input_artifacts"),
        "confirmation decision input_artifacts",
    )
    matrix_artifacts = _require_mapping(
        matrix.get("artifacts"), "confirmation matrix artifacts")
    for filename in REQUIRED_MATRIX_ARTIFACTS:
        decision_record = _require_mapping(
            decision_artifacts.get(filename),
            f"confirmation decision artifact {filename}",
        )
        matrix_record = _require_mapping(
            matrix_artifacts.get(filename),
            f"confirmation matrix artifact {filename}",
        )
        expected_binding = {
            key: matrix_record.get(key) for key in ARTIFACT_BINDING_FIELDS
        }
        if dict(decision_record) != expected_binding:
            raise RuntimeError(
                f"confirmation decision artifact binding differs for "
                f"{filename}")


def require_submission_ready(
    manifest: Mapping[str, Any], *, allow_historical: bool = False,
    manifest_path: str | Path | None = None,
) -> None:
    status = str(manifest.get("submission_status", "")).strip().lower()
    if status.startswith("hold"):
        if allow_historical:
            if status != HISTORICAL_HOLD_STATUS:
                raise RuntimeError(
                    "--allow-historical only accepts the locked historical "
                    f"hold status {HISTORICAL_HOLD_STATUS!r}"
                )
            version = str(manifest.get("version", "")).strip().lower()
            if "historical" not in version:
                raise RuntimeError(
                    "--allow-historical requires an explicitly historical "
                    "manifest version"
                )
            return
        protocol = str(manifest.get("active_protocol", "unknown"))
        raise RuntimeError(
            "paper artifact generation is blocked by submission_status="
            f"{status!r}; active protocol is {protocol!r}. Pass "
            "--allow-historical only to reproduce the explicitly historical "
            "package."
        )
    if status != READY_STATUS:
        raise RuntimeError(
            "paper artifact generation is fail-closed: submission_status must "
            f"be exactly {READY_STATUS!r}; observed {status or '<missing>'!r}"
        )
    _validate_ready_manifest(manifest, manifest_path=manifest_path)


def require_no_missing_artifacts(missing: list[str]) -> None:
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"submission package is missing {len(missing)} required artifact(s): "
            f"{preview}{suffix}"
        )
