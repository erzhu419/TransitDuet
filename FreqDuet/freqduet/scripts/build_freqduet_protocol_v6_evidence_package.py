#!/usr/bin/env python3
"""Build the current-best Protocol V6 paper evidence bundle.

This is deliberately an evidence package, not a submission-ready override.
It keeps the successful V8 confirmation, the failed V9 long-training gate,
the source-identical V9 external comparison, and the failed V33 timescale
repair together so manuscript claims cannot silently select only favorable
experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = (
    ROOT / "paper_evidence" / "protocol_v6" / "current_best"
)
DEFAULT_OUT = ROOT / "results_freqduet" / "paper_package" / "protocol_v6_current_best"

PAPER_CONTROLLER = "F_freqduet_protocol_v6_confirmed_main_hiro"
CONFIRMED_SOURCE_CONFIG = "F_freqduet_protocol_v6_avlcompact_w2_hiro"
NOGUARD_REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
PROTOCOL = "freqduet-eval-v6"
MATRIX_MANIFEST_VERSION = "freqduet-matrix-manifest-v2"
EXTERNAL_MANIFEST_VERSION = "freqduet-external-comparison-v6"
EXTERNAL_METHODS = ("fixed_headway", "rule_holding", "rule_mpc")
V8_SOURCE_COMMIT = "54f4a8e66763059274b2d6d5c9f4b2bc5e7ad92a"
V9_SOURCE_COMMIT = "673355cae10640b2737b6d51d541a9685059c342"
V33_SOURCE_COMMIT = "955352a9afb446ff3699c396936352a82983c8fb"
V33_GATE_VERSION = "freqduet-v33-timescale-stability-development-v2"
V33_RUN_NAME = "protocol_v6_v33_lower_timescale_ep200_s4_e4"
V8_TRAIN_SEEDS = (809, 827, 853, 877, 907, 929)
V8_EVAL_SEEDS = (44011, 44017, 44023, 44029)
V9_TRAIN_SEEDS = (12011, 12037, 12049, 12071, 12097, 12109, 12143, 12161)
V9_EVAL_SEEDS = (45007, 45013, 45053, 45061, 45077, 45119, 45131, 45137)
V33_TRAIN_SEEDS = (933013, 933031, 933059, 933083)
V33_EVAL_SEEDS = (934011, 934029, 934047, 934071)
V33_CONFIRM_TRAIN_SEEDS = (
    935013, 935037, 935063, 935087, 935111, 935133, 935159, 935181,
)
V33_CONFIRM_EVAL_SEEDS = (
    936011, 936029, 936047, 936071, 936101, 936119, 936143, 936167,
)
V33_CANDIDATE_CONTRACTS = {
    "F_freqduet_protocol_v6_v33_lowerfreeze100_hiro": {
        "freeze_lower_critic_after_ep": -1,
        "freeze_lower_policy_after_ep": 100,
        "freeze_upper_after_ep": -1,
    },
    "F_freqduet_protocol_v6_v33_lowerfreeze80_hiro": {
        "freeze_lower_critic_after_ep": -1,
        "freeze_lower_policy_after_ep": 80,
        "freeze_upper_after_ep": -1,
    },
    "F_freqduet_protocol_v6_v33_lowerfreeze40_hiro": {
        "freeze_lower_critic_after_ep": -1,
        "freeze_lower_policy_after_ep": 40,
        "freeze_upper_after_ep": -1,
    },
}
V33_CANDIDATES = tuple(V33_CANDIDATE_CONTRACTS)
V33_CONFIGS = (
    "F_freqduet_protocol_v6_main_hiro",
    NOGUARD_REFERENCE,
    "F_freqduet_protocol_v6_avlcompact_hiro",
    PAPER_CONTROLLER,
    *V33_CANDIDATES,
)

PAIRWISE_METRICS = (
    ("restricted_total_journey_horizon_min", "passenger_journey_min"),
    ("restricted_wait_horizon_min", "passenger_wait_min"),
    ("restricted_in_vehicle_horizon_min", "in_vehicle_min"),
    ("headway_cv", "headway_cv"),
    ("passenger_unserved_rate", "unserved_rate"),
    ("holding_vehicle_seconds_per_launched_trip", "holding_s_per_trip"),
    ("fleet_denied_trip_rate", "denied_trip_rate"),
    ("service_cost_restricted", "restricted_service_cost"),
)

V8_FILES = (
    "confirmation_gate.json",
    "matrix_manifest.json",
    "frozen_per_eval.csv",
    "frozen_summary.csv",
    "frozen_paired_deltas.csv",
)
V9_FILES = (
    "confirmed_longtrain_gate.json",
    "matrix_manifest.json",
    "frozen_per_eval.csv",
    "frozen_summary.csv",
    "frozen_paired_deltas.csv",
)
EXTERNAL_FILES = (
    "learned_vs_external_manifest.json",
    "learned_vs_external_per_pair.csv",
    "learned_vs_external_summary.csv",
    "external_baselines_per_seed.csv",
    "external_baselines_summary.csv",
    "external_baselines_summary.json",
)
V33_FILES = (
    "v33_timescale_gate.json",
    "matrix_manifest.json",
    "frozen_per_eval.csv",
    "frozen_summary.csv",
    "frozen_paired_deltas.csv",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_files(directory: Path, filenames: Iterable[str]) -> None:
    missing = [name for name in filenames if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing evidence files in {directory}: {', '.join(missing)}"
        )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_matrix_artifacts(directory: Path, matrix: dict[str, Any]) -> None:
    records = matrix.get("artifacts", {})
    for filename in ("frozen_per_eval.csv", "frozen_summary.csv",
                     "frozen_paired_deltas.csv"):
        expected = records.get(filename, {}).get("sha256")
        if not expected:
            raise ValueError(f"matrix lacks artifact binding for {filename}")
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"matrix artifact SHA256 mismatch: {filename}")


def validate_matrix_contract(
    label: str,
    matrix: dict[str, Any],
    *,
    source_commit: str,
    train_episodes: int,
    train_seeds: tuple[int, ...],
    eval_seeds: tuple[int, ...],
    required_config: str,
    stage: str = "confirmation",
    independent_confirmation: bool = True,
    exact_configs: tuple[str, ...] | None = None,
) -> None:
    if matrix.get("manifest_version") != MATRIX_MANIFEST_VERSION:
        raise ValueError(f"{label} matrix manifest version mismatch")
    if matrix.get("protocol_version") != PROTOCOL:
        raise ValueError(f"{label} matrix is not Protocol V6")
    if matrix.get("stage") != stage:
        raise ValueError(f"{label} matrix stage mismatch")
    if matrix.get("independent_confirmation") is not independent_confirmation:
        raise ValueError(f"{label} matrix confirmation status mismatch")
    if matrix.get("reference") != NOGUARD_REFERENCE:
        raise ValueError(f"{label} matrix uses a different no-guard reference")
    if matrix.get("strict_complete") is not True:
        raise ValueError(f"{label} matrix is not strict-complete")
    if matrix.get("common_random_numbers_verified") is not True:
        raise ValueError(f"{label} matrix lacks verified common random numbers")
    provenance = matrix.get("run_git_provenance", {})
    if provenance.get("commit") != source_commit:
        raise ValueError(f"{label} matrix source commit mismatch")
    if provenance.get("tracked_dirty") is not False:
        raise ValueError(f"{label} matrix source was not clean")
    if matrix.get("train_episodes") != train_episodes:
        raise ValueError(f"{label} matrix training horizon mismatch")
    if matrix.get("checkpoint_ep") != train_episodes - 1:
        raise ValueError(f"{label} matrix checkpoint mismatch")
    if tuple(matrix.get("train_seeds", ())) != train_seeds:
        raise ValueError(f"{label} matrix training-seed contract mismatch")
    if tuple(matrix.get("eval_seeds", ())) != eval_seeds:
        raise ValueError(f"{label} matrix evaluation-seed contract mismatch")
    configs = tuple(matrix.get("configs", ()))
    if required_config not in configs:
        raise ValueError(f"{label} matrix lacks required config {required_config}")
    if exact_configs is not None and configs != exact_configs:
        raise ValueError(f"{label} matrix config roster mismatch")
    expected_rollouts = len(configs) * len(train_seeds) * len(eval_seeds)
    if matrix.get("expected_rollouts") != expected_rollouts:
        raise ValueError(f"{label} matrix rollout contract mismatch")


def _resolve_parent(path: Path, parent: str, config_root: Path) -> Path:
    parent_path = Path(parent)
    if parent_path.is_absolute():
        return parent_path.resolve()
    candidates = (
        path.parent / parent_path,
        path.parent.parent / parent_path,
        config_root / parent_path,
    )
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[-1].resolve(),
    )


def config_lineage(
    path: Path,
    config_root: Path,
    seen: set[Path] | None = None,
) -> list[Path]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic config inheritance at {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing config in lineage: {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    lineage: list[Path] = []
    if "_extends" in payload:
        lineage.extend(config_lineage(
            _resolve_parent(path, str(payload["_extends"]), config_root),
            config_root,
            seen,
        ))
    lineage.append(path)
    return lineage


def config_fingerprint(name: str, config_root: Path) -> dict[str, Any]:
    lineage = config_lineage(
        config_root / "configs_freqduet" / f"{name}.yaml",
        config_root,
    )
    digest = hashlib.sha256()
    labels: list[str] = []
    for path in lineage:
        try:
            label = str(path.relative_to(config_root.resolve()))
        except ValueError as exc:
            raise ValueError(f"config lineage leaves config root: {path}") from exc
        labels.append(label)
        digest.update(label.encode("utf-8"))
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "lineage": labels}


def validate_config_fingerprints(
    v8_matrix: dict[str, Any],
    v9_matrix: dict[str, Any],
    v33_matrix: dict[str, Any],
    config_root: Path,
) -> dict[str, dict[str, Any]]:
    requested = [
        ("V8", v8_matrix, CONFIRMED_SOURCE_CONFIG),
        ("V8", v8_matrix, NOGUARD_REFERENCE),
        ("V9", v9_matrix, PAPER_CONTROLLER),
        ("V9", v9_matrix, NOGUARD_REFERENCE),
    ]
    requested.extend(("V33", v33_matrix, name) for name in V33_CONFIGS)
    verified: dict[str, dict[str, Any]] = {}
    for phase, matrix, name in requested:
        expected = matrix.get("config_fingerprints", {}).get(name)
        if not expected:
            raise ValueError(f"{phase} matrix lacks config fingerprint for {name}")
        actual = config_fingerprint(name, config_root)
        if actual != expected:
            raise ValueError(f"{phase} config fingerprint mismatch for {name}")
        prior = verified.get(name)
        if prior is not None and prior != actual:
            raise ValueError(f"inconsistent config fingerprint across phases: {name}")
        verified[name] = actual
    return verified


def one_row(rows: list[dict[str, str]], **matches: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if all(row.get(key) == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {matches}, found {len(selected)}")
    return selected[0]


def validate_v33_evidence(
    directory: Path,
    gate: dict[str, Any],
    matrix: dict[str, Any],
) -> None:
    if gate.get("gate_version") != V33_GATE_VERSION:
        raise ValueError("V33 gate version mismatch")
    if gate.get("status") != "no_pass":
        raise ValueError("V33 development decision is not the registered no-pass")
    if gate.get("claim_eligible") is not False:
        raise ValueError("V33 development must remain non-claim-eligible")
    if gate.get("confirmation_authorized") is not False:
        raise ValueError("V33 confirmation was not authorized")
    if gate.get("selected_for_confirmation") is not None:
        raise ValueError("V33 no-pass unexpectedly selects a confirmation candidate")
    if gate.get("passing_candidates") != []:
        raise ValueError("V33 no-pass unexpectedly records a passing candidate")
    if tuple(gate.get("candidate_priority", ())) != V33_CANDIDATES:
        raise ValueError("V33 candidate priority mismatch")
    if gate.get("candidate_contracts") != V33_CANDIDATE_CONTRACTS:
        raise ValueError("V33 candidate freeze contracts mismatch")

    results = gate.get("candidate_results")
    if not isinstance(results, list):
        raise ValueError("V33 gate lacks candidate results")
    if tuple(result.get("candidate") for result in results) != V33_CANDIDATES:
        raise ValueError("V33 gate candidate-result roster mismatch")
    if any(result.get("passes") is not False for result in results):
        raise ValueError("V33 gate does not preserve the registered no-pass")

    strict_checks = gate.get("strict_checks")
    if not isinstance(strict_checks, dict) or not strict_checks:
        raise ValueError("V33 gate lacks strict checks")
    if any(value is not True for value in strict_checks.values()):
        raise ValueError("V33 strict checks did not all pass")
    contract_checks = gate.get("contract_checks")
    if not isinstance(contract_checks, dict) or not contract_checks:
        raise ValueError("V33 gate lacks candidate-contract checks")
    if any(value is not True for value in contract_checks.values()):
        raise ValueError("V33 candidate-contract checks did not all pass")

    expected_development = {
        "checkpoint_ep": 199,
        "configs": list(V33_CONFIGS),
        "eval_seeds": list(V33_EVAL_SEEDS),
        "run_name": V33_RUN_NAME,
        "train_episodes": 200,
        "train_seeds": list(V33_TRAIN_SEEDS),
    }
    if gate.get("development_design") != expected_development:
        raise ValueError("V33 development design mismatch")
    expected_confirmation = {
        "checkpoint_ep": 199,
        "eval_seeds": list(V33_CONFIRM_EVAL_SEEDS),
        "single_use": True,
        "train_episodes": 200,
        "train_seeds": list(V33_CONFIRM_TRAIN_SEEDS),
    }
    if gate.get("confirmation_design") != expected_confirmation:
        raise ValueError("V33 frozen confirmation design mismatch")
    if gate.get("thresholds") != {
        "base_gate": "unchanged V8 effect and mechanism thresholds",
        "max_journey_ci_high_min": 0.15,
        "min_negative_train_seed_fraction": 0.75,
    }:
        raise ValueError("V33 registered thresholds mismatch")

    validate_matrix_contract(
        "V33",
        matrix,
        source_commit=V33_SOURCE_COMMIT,
        train_episodes=200,
        train_seeds=V33_TRAIN_SEEDS,
        eval_seeds=V33_EVAL_SEEDS,
        required_config=V33_CANDIDATES[0],
        stage="exploratory",
        independent_confirmation=False,
        exact_configs=V33_CONFIGS,
    )
    if matrix.get("run_manifests_verified") is not True:
        raise ValueError("V33 matrix lacks verified run manifests")
    validate_matrix_artifacts(directory, matrix)

    inputs = gate.get("input_artifacts", {})
    bound_inputs = (
        (directory / "matrix_manifest.json", inputs.get("manifest", {})),
        (directory / "frozen_per_eval.csv", inputs.get("per_eval", {})),
    )
    for path, record in bound_inputs:
        if not record.get("sha256") or sha256_file(path) != record["sha256"]:
            raise ValueError(f"V33 gate input artifact mismatch: {path.name}")

    rows = read_rows(directory / "frozen_paired_deltas.csv")
    for candidate in V33_CANDIDATES:
        row = one_row(rows, candidate=candidate, reference=NOGUARD_REFERENCE)
        if int(row.get("n_pairs", 0)) != (
            len(V33_TRAIN_SEEDS) * len(V33_EVAL_SEEDS)
        ):
            raise ValueError(f"V33 paired-rollout contract mismatch: {candidate}")


def validate_evidence(
    v8_dir: Path,
    v9_dir: Path,
    v33_dir: Path,
    external_dir: Path,
    config_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    require_files(v8_dir, V8_FILES)
    require_files(v9_dir, V9_FILES)
    require_files(v33_dir, V33_FILES)
    require_files(external_dir, EXTERNAL_FILES)

    v8_gate = read_json(v8_dir / "confirmation_gate.json")
    v9_gate = read_json(v9_dir / "confirmed_longtrain_gate.json")
    v33_gate = read_json(v33_dir / "v33_timescale_gate.json")
    v8_matrix = read_json(v8_dir / "matrix_manifest.json")
    v9_matrix = read_json(v9_dir / "matrix_manifest.json")
    v33_matrix = read_json(v33_dir / "matrix_manifest.json")

    if v8_gate.get("primary") != CONFIRMED_SOURCE_CONFIG:
        raise ValueError("V8 gate does not identify the confirmed source config")
    if v8_gate.get("primary_claim_eligible") is not True:
        raise ValueError("V8 primary confirmation is not claim eligible")
    if v8_gate.get("primary_result", {}).get("status") != "unique_pass":
        raise ValueError("V8 primary confirmation did not uniquely pass")
    if v9_gate.get("candidate") != PAPER_CONTROLLER:
        raise ValueError("V9 gate does not evaluate the paper controller")
    if v9_gate.get("status") != "longtrain_not_confirmed":
        raise ValueError("V9 long-training decision is not the registered result")
    validate_matrix_contract(
        "V8",
        v8_matrix,
        source_commit=V8_SOURCE_COMMIT,
        train_episodes=40,
        train_seeds=V8_TRAIN_SEEDS,
        eval_seeds=V8_EVAL_SEEDS,
        required_config=CONFIRMED_SOURCE_CONFIG,
    )
    validate_matrix_contract(
        "V9",
        v9_matrix,
        source_commit=V9_SOURCE_COMMIT,
        train_episodes=200,
        train_seeds=V9_TRAIN_SEEDS,
        eval_seeds=V9_EVAL_SEEDS,
        required_config=PAPER_CONTROLLER,
    )
    validate_matrix_artifacts(v8_dir, v8_matrix)
    validate_matrix_artifacts(v9_dir, v9_matrix)
    validate_v33_evidence(v33_dir, v33_gate, v33_matrix)
    config_records = validate_config_fingerprints(
        v8_matrix, v9_matrix, v33_matrix, config_root
    )

    external_manifest = read_json(
        external_dir / "learned_vs_external_manifest.json"
    )
    if external_manifest.get("manifest_version") != EXTERNAL_MANIFEST_VERSION:
        raise ValueError("external comparison manifest version mismatch")
    if external_manifest.get("protocol_version") != PROTOCOL:
        raise ValueError("external comparison is not Protocol V6")
    if external_manifest.get("strict_complete") is not True:
        raise ValueError("external comparison is not strict-complete")
    if external_manifest.get("common_random_numbers_verified") is not True:
        raise ValueError("external comparison lacks verified common random numbers")
    if external_manifest.get("learned_config") != PAPER_CONTROLLER:
        raise ValueError("external manifest uses a different learned controller")
    if external_manifest.get("baseline_config") != PAPER_CONTROLLER:
        raise ValueError("external baseline was not generated for the paper controller")
    if tuple(external_manifest.get("baseline_methods", ())) != EXTERNAL_METHODS:
        raise ValueError("external manifest baseline roster mismatch")
    if tuple(external_manifest.get(
        "required_external_method_family", ()
    )) != EXTERNAL_METHODS:
        raise ValueError("external required-method roster mismatch")
    external_rows = read_rows(external_dir / "learned_vs_external_summary.csv")
    methods = {row.get("baseline_method") for row in external_rows}
    required_methods = set(EXTERNAL_METHODS)
    if methods != required_methods:
        raise ValueError(
            "external comparison must contain exactly fixed_headway, "
            "rule_holding, and rule_mpc"
        )
    if any(row.get("learned_config") != PAPER_CONTROLLER for row in external_rows):
        raise ValueError("external comparison uses a different learned controller")
    if any(int(row.get("n_pairs", 0)) != 64 for row in external_rows):
        raise ValueError("external comparison does not contain 64 paired rollouts")

    pair_contracts = (
        (
            v8_dir / "frozen_paired_deltas.csv",
            CONFIRMED_SOURCE_CONFIG,
            len(V8_TRAIN_SEEDS) * len(V8_EVAL_SEEDS),
        ),
        (
            v9_dir / "frozen_paired_deltas.csv",
            PAPER_CONTROLLER,
            len(V9_TRAIN_SEEDS) * len(V9_EVAL_SEEDS),
        ),
    )
    for path, candidate, expected_pairs in pair_contracts:
        row = one_row(
            read_rows(path),
            candidate=candidate,
            reference=NOGUARD_REFERENCE,
        )
        if int(row.get("n_pairs", 0)) != expected_pairs:
            raise ValueError(f"paired-rollout contract mismatch: {path}")

    v9_commit = v9_matrix.get("run_git_provenance", {}).get("commit")
    provenance = external_manifest.get("source_provenance", {})
    if provenance.get("git", {}).get("commit") != v9_commit:
        raise ValueError("external comparison source commit differs from V9")
    if provenance.get("core_source_sha256") != v9_matrix.get(
        "run_source_fingerprint", {}
    ).get("sha256"):
        raise ValueError("external comparison source fingerprint differs from V9")
    if provenance.get("scenario_contract_sha256") != v9_matrix.get(
        "scenario_contract", {}
    ).get("sha256"):
        raise ValueError("external comparison scenario contract differs from V9")
    input_artifacts = external_manifest.get("input_artifacts", {})
    artifact_pairs = (
        (v9_dir / "frozen_per_eval.csv", input_artifacts.get("learned", {})),
        (
            external_dir / "external_baselines_per_seed.csv",
            input_artifacts.get("external", {}),
        ),
    )
    for path, record in artifact_pairs:
        if not record.get("sha256") or sha256_file(path) != record["sha256"]:
            raise ValueError(f"external input artifact mismatch: {path.name}")

    return (
        v8_gate,
        v9_gate,
        v33_gate,
        v8_matrix,
        v9_matrix,
        v33_matrix,
        external_manifest,
        config_records,
    )


def pairwise_table(
    csv_path: Path,
    *,
    candidate: str,
    reference: str,
    paper_controller: str,
    phase: str,
) -> list[dict[str, Any]]:
    row = one_row(
        read_rows(csv_path), candidate=candidate, reference=reference
    )
    output: list[dict[str, Any]] = []
    for source_name, paper_name in PAIRWISE_METRICS:
        prefix = f"delta_{source_name}"
        if f"{prefix}_mean" not in row:
            continue
        output.append({
            "phase": phase,
            "paper_controller": paper_controller,
            "source_candidate": candidate,
            "reference": reference,
            "metric": paper_name,
            "delta_candidate_minus_reference": row[f"{prefix}_mean"],
            "ci95_low": row.get(f"{prefix}_ci_low", ""),
            "ci95_high": row.get(f"{prefix}_ci_high", ""),
            "paired_signflip_p": row.get(f"{prefix}_signflip_p", ""),
            "paired_signflip_p_holm": row.get(
                f"{prefix}_signflip_p_holm", ""
            ),
            "n_pairs": row["n_pairs"],
        })
    return output


def external_table(path: Path) -> list[dict[str, Any]]:
    rows = read_rows(path)
    output: list[dict[str, Any]] = []
    for row in rows:
        for source_name, paper_name in PAIRWISE_METRICS:
            delta = f"delta_{source_name}_mean"
            if delta not in row:
                continue
            output.append({
                "paper_controller": PAPER_CONTROLLER,
                "baseline": row["baseline_method"],
                "metric": paper_name,
                "learned_mean": row.get(f"{source_name}_learned_mean", ""),
                "baseline_mean": row.get(f"{source_name}_baseline_mean", ""),
                "delta_learned_minus_baseline": row[delta],
                "ci95_low": row.get(f"delta_{source_name}_ci_low", ""),
                "ci95_high": row.get(f"delta_{source_name}_ci_high", ""),
                "paired_signflip_p": row.get(
                    f"delta_{source_name}_signflip_p", ""
                ),
                "paired_signflip_p_holm": row.get(
                    f"delta_{source_name}_signflip_p_holm", ""
                ),
                "n_pairs": row["n_pairs"],
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    selected = [row for row in rows if row["metric"] == metric]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {metric}")
    return selected[0]


def number(value: Any) -> float:
    return float(str(value))


def fmt(value: Any, digits: int = 4) -> str:
    return f"{number(value):+.{digits}f}"


def copy_inputs(src_dir: Path, names: Iterable[str], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(src_dir / name, dst_dir / name)


def copy_config_snapshots(
    config_root: Path,
    records: dict[str, dict[str, Any]],
    dst_dir: Path,
) -> None:
    labels = {
        label
        for record in records.values()
        for label in record["lineage"]
    }
    for label in sorted(labels):
        source = config_root / label
        target = dst_dir / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (dst_dir / "config_snapshot_manifest.json").write_text(
        json.dumps({
            "fingerprint_version": "freqduet-config-lineage-v1",
            "configs": records,
        }, indent=2, sort_keys=True) + "\n"
    )


def write_readme(
    path: Path,
    v8_commit: str,
    v9_commit: str,
    v33_commit: str,
) -> None:
    path.write_text(f"""# FreqDuet Protocol V6 Current-Best Evidence

This bundle is the paper-facing evidence for
`{PAPER_CONTROLLER}`. It deliberately keeps the successful V8 independent
confirmation, the failed V9 long-training gate, and the failed V33 timescale
repair together.

## Scientific Status

- V8 confirms lower headway CV than the Protocol V6 reference config named
  `{NOGUARD_REFERENCE}`, with the registered passenger-journey no-harm
  condition. Both policies disable the legacy holding guard; the confirmed
  policy adds compact APC/AVL context and a two-sided regularity objective.
- V9 does not confirm the preregistered long-training headway effect and is a
  mandatory negative robustness result.
- V33 development finds that freezing lower-actor updates at episode 40, 80,
  or 100 does not restore the registered long-training headway effect. No V33
  candidate passed and confirmation was not authorized.
- The V9 external comparison supports a service-regularity trade-off: FreqDuet
  is more regular than fixed headway but has higher passenger journey time.
- `submission_ready` is therefore `false`. This package must not be used to
  claim long-run confirmation or passenger-journey superiority to fixed
  headway.

## Contents

- `tables/`: normalized manuscript source tables.
- `manuscript/`: a concise results narrative with the claim boundary.
- `source_artifacts/`: the small immutable JSON/CSV evidence inputs.
- `configs/`: the exact verified YAML inheritance chains for the evaluated
  controller and protocol reference.
- `evidence_status.json` and `package_manifest.json`: machine-readable status
  and inventory.

V8 was generated from commit `{v8_commit}`, V9 plus the external comparison
from commit `{v9_commit}`, and V33 from commit `{v33_commit}`. The package
builder verifies result bindings, protocol/CRN status, source/scenario identity
for the V9 external comparison, and config fingerprints before writing output.
Checkpoints and full training logs are intentionally excluded; rerunning
training requires the cited source commits and seed contracts preserved in the
source manifests.
""")


def write_manuscript_results(
    path: Path,
    v8_rows: list[dict[str, Any]],
    v9_rows: list[dict[str, Any]],
    v33_rows: list[dict[str, Any]],
    v33_gate: dict[str, Any],
    external_rows: list[dict[str, Any]],
) -> None:
    v8_journey = metric_row(v8_rows, "passenger_journey_min")
    v8_cv = metric_row(v8_rows, "headway_cv")
    v9_journey = metric_row(v9_rows, "passenger_journey_min")
    v9_cv = metric_row(v9_rows, "headway_cv")
    fixed_journey = one_row(external_rows, baseline="fixed_headway",
                            metric="passenger_journey_min")
    fixed_service = one_row(external_rows, baseline="fixed_headway",
                            metric="restricted_service_cost")
    fixed_cv = one_row(external_rows, baseline="fixed_headway",
                       metric="headway_cv")
    rule_journey = one_row(external_rows, baseline="rule_holding",
                           metric="passenger_journey_min")
    mpc_journey = one_row(external_rows, baseline="rule_mpc",
                          metric="passenger_journey_min")
    v33_metrics = []
    for candidate in V33_CANDIDATES:
        v33_metrics.append((
            candidate,
            one_row(v33_rows, source_candidate=candidate, metric="headway_cv"),
            one_row(
                v33_rows,
                source_candidate=candidate,
                metric="passenger_journey_min",
            ),
        ))
    v33_lines = "\n".join(
        "- `{candidate}`: headway-CV delta {cv} (95% CI [{cv_low}, {cv_high}]); "
        "journey delta {journey} min (95% CI [{journey_low}, {journey_high}]).".format(
            candidate=candidate,
            cv=fmt(cv_row["delta_candidate_minus_reference"]),
            cv_low=fmt(cv_row["ci95_low"]),
            cv_high=fmt(cv_row["ci95_high"]),
            journey=fmt(journey_row["delta_candidate_minus_reference"]),
            journey_low=fmt(journey_row["ci95_low"]),
            journey_high=fmt(journey_row["ci95_high"]),
        )
        for candidate, cv_row, journey_row in v33_metrics
    )
    unfrozen = v33_gate["unfrozen_control_result"]["candidate_results"][0]

    text = f"""# Protocol V6 Current-Best Results

## Confirmed Effect

The canonical controller is `{PAPER_CONTROLLER}`, a naming alias of the V8
confirmed compact-AVL weight-two policy. In the preregistered 40-episode
independent confirmation (six training seeds crossed with four untouched
evaluation seeds; 24 paired rollouts), it reduced headway CV relative to the
Protocol V6 reference config named `{NOGUARD_REFERENCE}` by
{fmt(v8_cv['delta_candidate_minus_reference'])}
(95% CI [{fmt(v8_cv['ci95_low'])}, {fmt(v8_cv['ci95_high'])}]). Restricted
passenger journey time changed by {fmt(v8_journey['delta_candidate_minus_reference'])}
min (95% CI [{fmt(v8_journey['ci95_low'])},
{fmt(v8_journey['ci95_high'])}]), satisfying the registered journey no-harm
criterion but not establishing a significant journey-time reduction. Both
configurations disable the legacy causal holding guard; the confirmed policy
adds compact APC/AVL context and the two-sided departure-regularity objective.
The paired contrast is therefore a combined-policy comparison, not an isolated
guard effect.

## Long-Training Robustness

In the independent 200-episode V9 matrix (eight training seeds crossed with
eight evaluation seeds; 64 paired rollouts), the same controller improved
restricted journey time versus the same protocol reference by
{fmt(v9_journey['delta_candidate_minus_reference'])} min (95% CI
[{fmt(v9_journey['ci95_low'])}, {fmt(v9_journey['ci95_high'])}]). Its headway-CV
delta was {fmt(v9_cv['delta_candidate_minus_reference'])} (95% CI
[{fmt(v9_cv['ci95_low'])}, {fmt(v9_cv['ci95_high'])}]), which did not meet the
preregistered magnitude and interval gate. V9 is therefore a valid negative
long-training result, not a second confirmation claim.

## External Baselines

Under the source-identical V9 comparison, FreqDuet reduced restricted service
cost versus fixed headway by
{fmt(fixed_service['delta_learned_minus_baseline'])} (95% CI
[{fmt(fixed_service['ci95_low'])}, {fmt(fixed_service['ci95_high'])}]) and
headway CV by {fmt(fixed_cv['delta_learned_minus_baseline'])} (95% CI
[{fmt(fixed_cv['ci95_low'])}, {fmt(fixed_cv['ci95_high'])}]), but increased
restricted passenger journey time by
{fmt(fixed_journey['delta_learned_minus_baseline'])} min (95% CI
[{fmt(fixed_journey['ci95_low'])}, {fmt(fixed_journey['ci95_high'])}]). It
reduced journey time relative to rule holding by
{fmt(rule_journey['delta_learned_minus_baseline'])} min and relative to rule
MPC by {fmt(mpc_journey['delta_learned_minus_baseline'])} min. The supported
interpretation is a service-regularity trade-off: the current controller is
better than the weaker rule baselines and more regular than fixed headway, but
it does not outperform fixed headway on passenger journey time.

## Registered Timescale Repair

V33 tested whether stopping lower-actor updates at episode 40, 80, or 100 could
restore the registered 200-episode regularity effect while both critics and the
upper actor continued training. The four-policy-seed by four-evaluation-seed
development matrix produced no passing candidate:

{v33_lines}

All three candidates preserved the registered journey no-harm condition, but
none achieved a headway-CV reduction of at least 0.02 with its interval fully
below zero. The episode-40 candidate improved headway CV in all four training
seeds, yet its aggregate effect was only
{fmt(v33_metrics[2][1]['delta_candidate_minus_reference'])} and its interval
still crossed zero. V33 status is `{v33_gate['status']}`; confirmation was not
authorized. The unfrozen current controller happened to pass the incremental
gate on these development seeds (headway-CV delta
{fmt(unfrozen['headway_cv_delta_vs_reference'])}, 95% CI
[{fmt(unfrozen['headway_cv_delta_ci_low'])},
{fmt(unfrozen['headway_cv_delta_ci_high'])}]), but it was a control rather than
a registered V33 candidate and cannot replace the failed independent V9 gate.

## Evidence Boundary

The legacy domain-selected V1/composite package is excluded from current
headline evidence because the later physical and causal audit invalidated that
protocol for passenger-journey claims. V28-V32 and V33 are retained as negative
development evidence and do not modify the controller above. This evidence
bundle remains draft-facing because the registered V9 long-training gate did
not confirm; it must not be relabelled as submission-ready.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def refresh_package_manifest(out_dir: Path) -> dict[str, Any]:
    status = read_json(out_dir / "evidence_status.json")
    files = sorted(
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "package_manifest.json"
    )
    package_manifest = {
        **status,
        "payload_file_count": len(files),
        "files": files,
    }
    (out_dir / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"
    )
    return package_manifest


def build_package(
    v8_dir: Path,
    v9_dir: Path,
    v33_dir: Path,
    external_dir: Path,
    out_dir: Path,
    config_root: Path = ROOT,
) -> dict[str, Any]:
    (
        v8_gate,
        v9_gate,
        v33_gate,
        v8_matrix,
        v9_matrix,
        v33_matrix,
        external_manifest,
        config_records,
    ) = validate_evidence(v8_dir, v9_dir, v33_dir, external_dir, config_root)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    v8_rows = pairwise_table(
        v8_dir / "frozen_paired_deltas.csv",
        candidate=CONFIRMED_SOURCE_CONFIG,
        reference=NOGUARD_REFERENCE,
        paper_controller=PAPER_CONTROLLER,
        phase="v8_independent_confirmation_ep40",
    )
    v9_rows = pairwise_table(
        v9_dir / "frozen_paired_deltas.csv",
        candidate=PAPER_CONTROLLER,
        reference=NOGUARD_REFERENCE,
        paper_controller=PAPER_CONTROLLER,
        phase="v9_independent_longtrain_ep200",
    )
    v33_rows: list[dict[str, Any]] = []
    v33_results = {
        result["candidate"]: result for result in v33_gate["candidate_results"]
    }
    for candidate in V33_CANDIDATES:
        rows = pairwise_table(
            v33_dir / "frozen_paired_deltas.csv",
            candidate=candidate,
            reference=NOGUARD_REFERENCE,
            paper_controller=PAPER_CONTROLLER,
            phase="v33_lower_actor_timescale_development_ep200",
        )
        for row in rows:
            row["registered_gate_pass"] = v33_results[candidate]["passes"]
        v33_rows.extend(rows)
    ext_rows = external_table(external_dir / "learned_vs_external_summary.csv")

    write_csv(out_dir / "tables" / "table1_v8_confirmation.csv", v8_rows)
    write_csv(out_dir / "tables" / "table2_v9_longtrain.csv", v9_rows)
    write_csv(out_dir / "tables" / "table3_v9_external_baselines.csv", ext_rows)
    write_csv(out_dir / "tables" / "table5_v33_timescale_development.csv", v33_rows)
    write_csv(out_dir / "tables" / "table4_evidence_decisions.csv", [
        {
            "phase": "v8_independent_confirmation_ep40",
            "decision": "primary_confirmed",
            "claim_eligible": True,
            "controller": PAPER_CONTROLLER,
            "train_seeds": 6,
            "evaluation_seeds": 4,
            "paired_rollouts": 24,
        },
        {
            "phase": "v9_independent_longtrain_ep200",
            "decision": v9_gate["status"],
            "claim_eligible": v9_gate["longtrain_claim_eligible"],
            "controller": PAPER_CONTROLLER,
            "train_seeds": 8,
            "evaluation_seeds": 8,
            "paired_rollouts": 64,
        },
    ])

    copy_inputs(v8_dir, V8_FILES, out_dir / "source_artifacts" / "v8")
    copy_inputs(v9_dir, V9_FILES, out_dir / "source_artifacts" / "v9")
    copy_inputs(v33_dir, V33_FILES, out_dir / "source_artifacts" / "v33_timescale")
    copy_inputs(
        external_dir, EXTERNAL_FILES, out_dir / "source_artifacts" / "v9_external"
    )
    copy_config_snapshots(config_root, config_records, out_dir / "configs")
    write_manuscript_results(
        out_dir / "manuscript" / "current_best_results.md",
        v8_rows,
        v9_rows,
        v33_rows,
        v33_gate,
        ext_rows,
    )

    v8_commit = v8_matrix["run_git_provenance"]["commit"]
    v9_commit = v9_matrix["run_git_provenance"]["commit"]
    v33_commit = v33_matrix["run_git_provenance"]["commit"]
    write_readme(out_dir / "README.md", v8_commit, v9_commit, v33_commit)
    status = {
        "package_version": "freqduet-protocol-v6-current-best-evidence-v4",
        "submission_ready": False,
        "submission_blocker": "v9_longtrain_not_confirmed",
        "protocol": PROTOCOL,
        "paper_controller": PAPER_CONTROLLER,
        "confirmed_source_config": CONFIRMED_SOURCE_CONFIG,
        "v8_source_commit": v8_commit,
        "v9_source_commit": v9_commit,
        "v33_source_commit": v33_commit,
        "v8_confirmation_status": v8_gate["primary_result"]["status"],
        "v9_longtrain_status": v9_gate["status"],
        "v33_timescale_status": v33_gate["status"],
        "v33_claim_eligible": v33_gate["claim_eligible"],
        "v33_confirmation_authorized": v33_gate["confirmation_authorized"],
        "config_fingerprints_verified": True,
        "external_source_identity_verified": True,
        "external_comparison_version": external_manifest.get(
            "manifest_version", "unknown"
        ),
    }
    (out_dir / "evidence_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    return refresh_package_manifest(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v8-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "v8_confirmation",
    )
    parser.add_argument(
        "--v9-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "v9_longtrain",
    )
    parser.add_argument(
        "--v33-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "v33_timescale",
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "v9_external",
    )
    parser.add_argument("--config-root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_package(
        args.v8_dir.resolve(),
        args.v9_dir.resolve(),
        args.v33_dir.resolve(),
        args.external_dir.resolve(),
        args.out_dir.resolve(),
        args.config_root.resolve(),
    )
    print(json.dumps({
        "status": "evidence_package_complete",
        "out_dir": str(args.out_dir.resolve()),
        "payload_file_count": manifest["payload_file_count"],
        "submission_ready": manifest["submission_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
