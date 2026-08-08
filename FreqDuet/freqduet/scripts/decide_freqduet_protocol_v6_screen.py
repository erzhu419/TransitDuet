#!/usr/bin/env python3
"""Apply the fail-closed staged decision contract for protocol V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scripts.analysis_provenance import (
        canonical_json_sha256,
        sha256_file,
        validate_csv_artifact,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from analysis_provenance import (  # type: ignore
        canonical_json_sha256,
        sha256_file,
        validate_csv_artifact,
    )


MATRIX_MANIFEST_VERSION = "freqduet-matrix-manifest-v2"
PROTOCOL = "freqduet-eval-v6"
DECISION_CONTRACT = "freqduet-protocol-v6-staged-decision-v1"
PRIMARY = "restricted_total_journey_horizon_min"
REFERENCE = "F_freqduet_protocol_v6_main_hiro"

CONFIGS = (
    REFERENCE,
    "F_freqduet_protocol_v6_nofreq_hiro",
    "F_freqduet_protocol_v6_rawhistory_hiro",
    "F_freqduet_protocol_v6_allfreq_hiro",
    "F_freqduet_protocol_v6_upperonly_hiro",
    "F_freqduet_protocol_v6_loweronly_hiro",
    "F_freqduet_protocol_v6_swapped_hiro",
    "F_freqduet_protocol_v6_nobudget_hiro",
    "F_freqduet_protocol_v6_noguard_hiro",
    "F_freqduet_protocol_v6_noloadcost_hiro",
    "F_freqduet_protocol_v6_waitonlycredit_hiro",
    "F_freqduet_protocol_v6_csac_hiro",
)
FREQUENCY_CONTROLS = frozenset({
    "F_freqduet_protocol_v6_nofreq_hiro",
    "F_freqduet_protocol_v6_rawhistory_hiro",
})
ALLOCATION_CONTROLS = frozenset({
    "F_freqduet_protocol_v6_allfreq_hiro",
    "F_freqduet_protocol_v6_upperonly_hiro",
    "F_freqduet_protocol_v6_loweronly_hiro",
    "F_freqduet_protocol_v6_swapped_hiro",
})
MECHANISM_ABLATIONS = frozenset({
    "F_freqduet_protocol_v6_nobudget_hiro",
    "F_freqduet_protocol_v6_noguard_hiro",
    "F_freqduet_protocol_v6_noloadcost_hiro",
    "F_freqduet_protocol_v6_waitonlycredit_hiro",
})
SIMPLE_CONFIG = "F_freqduet_protocol_v6_csac_hiro"

FREQUENCY_MIN_ADVANTAGE_MIN = 0.25
ALLOCATION_MIN_ADVANTAGE_MIN = 0.10
MECHANISM_MIN_EFFECT_MIN = 0.10
SIMPLE_NONINFERIORITY_MARGIN_MIN = 0.25
REFERENCE_NO_HARM_LIMITS = {
    "restricted_wait_horizon_min": ("min", 0.50),
    "passenger_unserved_rate": ("min", 0.005),
    "headway_cv": ("min", 0.02),
    "fleet_denied_trip_rate": ("min", 0.005),
    "fleet_readiness_delay_mean_s": ("min", 15.0),
    "holding_passenger_min_per_generated": ("min", 0.10),
    "trip_launch_rate": ("max", 0.005),
    "trip_completion_rate": ("max", 0.005),
}

SUMMARY_FILE = "frozen_summary.csv"
PAIRED_FILE = "frozen_paired_deltas.csv"
PER_EVAL_FILE = "frozen_per_eval.csv"
ARTIFACT_PRIMARY_KEYS = {
    SUMMARY_FILE: ("config",),
    PAIRED_FILE: ("candidate", "reference"),
    PER_EVAL_FILE: ("config", "train_seed", "eval_seed"),
}
ROLLOUT_INVARIANT_COLUMNS = (
    PRIMARY,
    "upper_plan_projected_delta_sum_abs_mean_s",
    "lower_causal_guard_enabled",
    "passengers_generated",
    "passengers_unserved",
    "passenger_unserved_rate",
    "physical_vehicle_count",
    "fleet_capacity",
    "peak_fleet",
    "fleet_denied_dispatch_events",
    "fleet_denied_retry_trip_seconds",
    "fleet_denied_trips",
    "fleet_readiness_delay_mean_s",
    "fleet_readiness_delay_max_s",
    "fleet_denied_trip_rate",
)


def _contract_payload() -> dict[str, object]:
    return {
        "contract": DECISION_CONTRACT,
        "manifest_version": MATRIX_MANIFEST_VERSION,
        "protocol": PROTOCOL,
        "configs": list(CONFIGS),
        "reference": REFERENCE,
        "primary_metric": PRIMARY,
        "frequency_controls": sorted(FREQUENCY_CONTROLS),
        "allocation_controls": sorted(ALLOCATION_CONTROLS),
        "mechanism_ablations": sorted(MECHANISM_ABLATIONS),
        "simple_config": SIMPLE_CONFIG,
        "thresholds": {
            "frequency_min_advantage_min": FREQUENCY_MIN_ADVANTAGE_MIN,
            "allocation_min_advantage_min": ALLOCATION_MIN_ADVANTAGE_MIN,
            "mechanism_min_effect_min": MECHANISM_MIN_EFFECT_MIN,
            "simple_noninferiority_margin_min": (
                SIMPLE_NONINFERIORITY_MARGIN_MIN),
            "reference_no_harm_limits": REFERENCE_NO_HARM_LIMITS,
        },
        "gate_rules": {
            "frequency": (
                "all controls clear mean threshold, CI low > 0, and "
                "reference no-harm"),
            "allocation": (
                "all controls clear mean threshold, CI low > 0, and "
                "reference no-harm"),
            "mechanism": (
                "all ablations clear mean threshold and CI low > 0"),
            "simple_optimizer": (
                "CI high <= noninferiority margin and candidate no-harm"),
        },
        "artifact_primary_keys": {
            name: list(key) for name, key in ARTIFACT_PRIMARY_KEYS.items()
        },
        "rollout_invariants": {
            "budget_sum_abs_max": 1e-5,
            "guard_enabled": 1.0,
            "guard_evidence_mode": "pre_action_departure_v6",
            "finite_columns": list(ROLLOUT_INVARIANT_COLUMNS),
        },
        "stage_rules": {
            "development": "candidate only",
            "confirmation": {
                "independent_confirmation": True,
                "development_manifest_required": True,
                "seed_disjointness": "union of train and eval seed sets",
            },
        },
    }


DECISION_CONTRACT_SHA256 = canonical_json_sha256(_contract_payload())


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _seed_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON list")
    seeds = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{label} must contain only integer seeds")
        seeds.append(int(item))
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{label} contains duplicate seeds")
    return seeds


def _require_exact_configs(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of config names")
    configs = [str(item) for item in value]
    if len(configs) != len(set(configs)):
        raise ValueError(f"{label} contains duplicate configs")
    missing = sorted(set(CONFIGS) - set(configs))
    extra = sorted(set(configs) - set(CONFIGS))
    if missing or extra:
        raise ValueError(
            f"{label} does not match the locked V6 configs; "
            f"missing={missing}, extra={extra}")
    return configs


def _validate_manifest_contract(
    manifest: dict[str, object],
    *,
    stage: str,
) -> tuple[list[int], list[int], list[str]]:
    if manifest.get("manifest_version") != MATRIX_MANIFEST_VERSION:
        raise ValueError("matrix manifest version is not locked V2")
    if manifest.get("protocol_version") != PROTOCOL:
        raise ValueError("matrix manifest protocol is not locked V6")
    if manifest.get("stage") != stage:
        raise ValueError(
            f"matrix stage {manifest.get('stage')!r} does not match "
            f"command stage {stage!r}")
    for field in (
        "strict_complete", "run_manifests_verified",
        "common_random_numbers_verified",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"matrix manifest requires {field}=true")
    if stage == "confirmation" and (
            manifest.get("independent_confirmation") is not True):
        raise ValueError(
            "confirmation matrix lacks independent_confirmation=true")
    configs = _require_exact_configs(manifest.get("configs"), "manifest configs")
    if manifest.get("reference") != REFERENCE:
        raise ValueError("matrix reference is not the locked V6 main")
    if manifest.get("primary_metric") != PRIMARY:
        raise ValueError("matrix primary metric is not the locked journey metric")

    metrics = manifest.get("metrics")
    if not isinstance(metrics, list) or any(
            not isinstance(metric, str) for metric in metrics):
        raise ValueError("matrix metrics must be a list of names")
    metrics = [str(metric) for metric in metrics]
    if len(metrics) != len(set(metrics)):
        raise ValueError("matrix metrics contain duplicates")
    required_metrics = {PRIMARY, *REFERENCE_NO_HARM_LIMITS}
    missing_metrics = sorted(required_metrics - set(metrics))
    if missing_metrics:
        raise ValueError(
            f"matrix is missing locked decision metrics: {missing_metrics}")

    train_seeds = _seed_list(manifest.get("train_seeds"), "train_seeds")
    eval_seeds = _seed_list(manifest.get("eval_seeds"), "eval_seeds")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("matrix train and evaluation seeds overlap")
    expected_rollouts = len(CONFIGS) * len(train_seeds) * len(eval_seeds)
    if manifest.get("expected_rollouts") != expected_rollouts:
        raise ValueError(
            "matrix expected_rollouts does not match the locked Cartesian grid")
    return train_seeds, eval_seeds, metrics


def _artifact_binding(record: dict[str, object]) -> dict[str, object]:
    fields = (
        "sha256", "size_bytes", "n_rows", "columns",
        "primary_key", "primary_key_sha256",
    )
    return {field: record[field] for field in fields}


def _load_artifacts(
    manifest_path: Path,
    manifest: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("matrix manifest has no artifact provenance map")
    frames = {}
    bindings = {}
    for name, primary_key in ARTIFACT_PRIMARY_KEYS.items():
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"matrix manifest is missing artifact {name}")
        path = manifest_path.parent / name
        frame = validate_csv_artifact(
            path,
            record,
            expected_primary_key=primary_key,
        )
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"{name}: size does not match its manifest")
        frames[name] = frame
        bindings[name] = _artifact_binding(record)
    return frames, bindings


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"per-rollout artifact is missing {column}")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"per-rollout {column} values must be finite integers")
    return numeric.astype(np.int64)


def _validate_table_contracts(
    frames: dict[str, pd.DataFrame],
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
) -> None:
    summary = frames[SUMMARY_FILE]
    paired = frames[PAIRED_FILE]
    per_eval = frames[PER_EVAL_FILE]

    required_columns = {
        SUMMARY_FILE: {"config"},
        PAIRED_FILE: {"candidate", "reference"},
        PER_EVAL_FILE: {
            "config", "train_seed", "eval_seed", "protocol_version",
            "scenario_tape_id",
        },
    }
    for name, required in required_columns.items():
        missing = sorted(required - set(frames[name].columns))
        if missing:
            raise ValueError(f"{name} lacks contract columns: {missing}")

    _require_exact_configs(summary["config"].tolist(), "summary configs")
    _require_exact_configs(per_eval["config"].drop_duplicates().tolist(),
                           "per-rollout configs")
    expected_candidates = set(CONFIGS) - {REFERENCE}
    candidates = set(paired["candidate"].astype(str))
    if candidates != expected_candidates or len(paired) != len(expected_candidates):
        raise ValueError("paired artifact does not contain every locked candidate")
    if set(paired["reference"].astype(str)) != {REFERENCE}:
        raise ValueError("paired artifact does not use the locked reference")

    observed_train = _integer_series(per_eval, "train_seed")
    observed_eval = _integer_series(per_eval, "eval_seed")
    if set(observed_train) != set(train_seeds):
        raise ValueError("per-rollout train seeds do not match the manifest")
    if set(observed_eval) != set(eval_seeds):
        raise ValueError("per-rollout evaluation seeds do not match the manifest")
    observed_keys = set(zip(
        per_eval["config"].astype(str),
        observed_train.astype(int),
        observed_eval.astype(int),
    ))
    expected_keys = {
        (config, int(train_seed), int(eval_seed))
        for config in CONFIGS
        for train_seed in train_seeds
        for eval_seed in eval_seeds
    }
    if observed_keys != expected_keys or len(per_eval) != len(expected_keys):
        raise ValueError("per-rollout artifact is not the complete locked grid")
    if set(per_eval["protocol_version"].astype(str)) != {PROTOCOL}:
        raise ValueError("per-rollout protocol_version is not locked V6")
    scenario_ids = per_eval["scenario_tape_id"]
    if scenario_ids.isna().any() or scenario_ids.astype(str).str.strip().eq("").any():
        raise ValueError("per-rollout scenario_tape_id values must be non-empty")
    scenario_counts = per_eval.groupby("eval_seed")["scenario_tape_id"].nunique(
        dropna=False)
    if not scenario_counts.eq(1).all():
        raise ValueError("common random scenario tapes differ across policies")


def _delta_column(metric: str, suffix: str) -> str:
    return f"delta_{metric}_ci_{suffix}"


def _required_delta_columns() -> set[str]:
    columns = {
        "candidate",
        "reference",
        f"delta_{PRIMARY}_mean",
        _delta_column(PRIMARY, "low"),
        _delta_column(PRIMARY, "high"),
    }
    for metric in REFERENCE_NO_HARM_LIMITS:
        columns.add(_delta_column(metric, "low"))
        columns.add(_delta_column(metric, "high"))
    return columns


def _validate_decision_numbers(paired: pd.DataFrame) -> None:
    missing = sorted(_required_delta_columns() - set(paired.columns))
    if missing:
        raise ValueError(f"paired artifact lacks decision columns: {missing}")
    numeric = sorted(_required_delta_columns() - {"candidate", "reference"})
    values = paired[numeric].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("paired decision columns must all be finite")


def _violation_keys(frame: pd.DataFrame, mask: np.ndarray) -> list[dict[str, object]]:
    columns = ["train_seed", "eval_seed"]
    return frame.loc[mask, columns].head(20).astype(int).to_dict(orient="records")


def _main_invariants(
    per_eval: pd.DataFrame,
    metrics: Sequence[str],
) -> tuple[bool, dict[str, object]]:
    required = set(ROLLOUT_INVARIANT_COLUMNS) | {
        "config", "train_seed", "eval_seed", "lower_causal_guard_evidence_mode",
    }
    missing = sorted(required - set(per_eval.columns))
    if missing:
        raise ValueError(f"per-rollout artifact lacks invariant columns: {missing}")
    main = per_eval[per_eval["config"].astype(str).eq(REFERENCE)].copy()
    if main.empty:
        raise ValueError("per-rollout artifact has no V6 main rows")

    finite_columns = list(dict.fromkeys([
        *ROLLOUT_INVARIANT_COLUMNS,
        *(str(metric) for metric in metrics),
    ]))
    missing_finite = sorted(set(finite_columns) - set(main.columns))
    if missing_finite:
        raise ValueError(
            f"per-rollout artifact lacks finite metric columns: {missing_finite}")
    finite_masks = {}
    for column in finite_columns:
        values = pd.to_numeric(main[column], errors="coerce").to_numpy(dtype=float)
        finite_masks[column] = np.isfinite(values)
    finite_all = np.logical_and.reduce(list(finite_masks.values()))

    budget = pd.to_numeric(
        main["upper_plan_projected_delta_sum_abs_mean_s"],
        errors="coerce",
    ).to_numpy(dtype=float)
    budget_ok = np.isfinite(budget) & (budget >= -1e-12) & (budget <= 1e-5)
    guard = pd.to_numeric(
        main["lower_causal_guard_enabled"], errors="coerce",
    ).to_numpy(dtype=float)
    guard_ok = np.isfinite(guard) & (np.abs(guard - 1.0) <= 1e-9)
    evidence = main["lower_causal_guard_evidence_mode"].astype(str).to_numpy()
    evidence_ok = evidence == "pre_action_departure_v6"
    passengers = pd.to_numeric(
        main["passengers_generated"], errors="coerce",
    ).to_numpy(dtype=float)
    passengers_ok = np.isfinite(passengers) & (passengers > 0.0)
    fleet_capacity = pd.to_numeric(
        main["fleet_capacity"], errors="coerce",
    ).to_numpy(dtype=float)
    physical_vehicles = pd.to_numeric(
        main["physical_vehicle_count"], errors="coerce",
    ).to_numpy(dtype=float)
    fleet_ok = (
        np.isfinite(fleet_capacity)
        & np.isfinite(physical_vehicles)
        & (fleet_capacity > 0.0)
        & (physical_vehicles > 0.0)
    )

    checks = {
        "rollout_count": int(len(main)),
        "budget_sum_abs": {
            "pass": bool(budget_ok.all()),
            "limit": 1e-5,
            "violations": _violation_keys(main, ~budget_ok),
        },
        "guard_enabled": {
            "pass": bool(guard_ok.all()),
            "expected": 1.0,
            "violations": _violation_keys(main, ~guard_ok),
        },
        "guard_evidence_mode": {
            "pass": bool(evidence_ok.all()),
            "expected": "pre_action_departure_v6",
            "violations": _violation_keys(main, ~evidence_ok),
        },
        "positive_passenger_population": {
            "pass": bool(passengers_ok.all()),
            "violations": _violation_keys(main, ~passengers_ok),
        },
        "positive_fleet_population": {
            "pass": bool(fleet_ok.all()),
            "violations": _violation_keys(main, ~fleet_ok),
        },
        "finite_passenger_fleet_and_metrics": {
            "pass": bool(finite_all.all()),
            "columns": finite_columns,
            "violations_by_column": {
                column: _violation_keys(main, ~mask)
                for column, mask in finite_masks.items()
                if not mask.all()
            },
        },
    }
    return all(bool(value["pass"]) for key, value in checks.items()
               if key != "rollout_count"), checks


def _reference_no_harm(row: pd.Series) -> tuple[bool, dict[str, object]]:
    checks = {}
    passed = True
    for metric, (direction, margin) in REFERENCE_NO_HARM_LIMITS.items():
        if direction == "min":
            observed = float(row[_delta_column(metric, "low")])
            ok = observed >= -float(margin)
            rule = "candidate_minus_main_ci_low >= -margin"
        else:
            observed = float(row[_delta_column(metric, "high")])
            ok = observed <= float(margin)
            rule = "candidate_minus_main_ci_high <= margin"
        checks[metric] = {
            "observed": observed,
            "margin": float(margin),
            "rule": rule,
            "pass": bool(ok),
        }
        passed = passed and bool(ok)
    return bool(passed), checks


def _candidate_no_harm(row: pd.Series) -> tuple[bool, dict[str, object]]:
    checks = {}
    passed = True
    for metric, (direction, margin) in REFERENCE_NO_HARM_LIMITS.items():
        if direction == "min":
            observed = float(row[_delta_column(metric, "high")])
            ok = observed <= float(margin)
            rule = "candidate_minus_main_ci_high <= margin"
        else:
            observed = float(row[_delta_column(metric, "low")])
            ok = observed >= -float(margin)
            rule = "candidate_minus_main_ci_low >= -margin"
        checks[metric] = {
            "observed": observed,
            "margin": float(margin),
            "rule": rule,
            "pass": bool(ok),
        }
        passed = passed and bool(ok)
    return bool(passed), checks


def _evidence_decision(
    paired: pd.DataFrame,
    invariant_pass: bool,
    invariants: dict[str, object],
) -> dict[str, object]:
    _validate_decision_numbers(paired)
    rows = {str(row["candidate"]): row for _, row in paired.iterrows()}

    frequency_checks = {}
    for name in sorted(FREQUENCY_CONTROLS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        low = float(row[_delta_column(PRIMARY, "low")])
        high = float(row[_delta_column(PRIMARY, "high")])
        no_harm, no_harm_checks = _reference_no_harm(row)
        frequency_checks[name] = {
            "candidate_minus_main_mean_min": mean,
            "ci_low": low,
            "ci_high": high,
            "main_advantage_pass": bool(
                mean >= FREQUENCY_MIN_ADVANTAGE_MIN and low > 0.0 and no_harm),
            "control_superior": bool(high < 0.0),
            "main_no_harm_pass": no_harm,
            "main_no_harm_checks": no_harm_checks,
        }

    allocation_checks = {}
    for name in sorted(ALLOCATION_CONTROLS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        low = float(row[_delta_column(PRIMARY, "low")])
        high = float(row[_delta_column(PRIMARY, "high")])
        no_harm, no_harm_checks = _reference_no_harm(row)
        allocation_checks[name] = {
            "candidate_minus_main_mean_min": mean,
            "ci_low": low,
            "ci_high": high,
            "allocation_advantage_pass": bool(
                mean >= ALLOCATION_MIN_ADVANTAGE_MIN and low > 0.0 and no_harm),
            "control_superior": bool(high < 0.0),
            "main_no_harm_pass": no_harm,
            "main_no_harm_checks": no_harm_checks,
        }

    mechanism_checks = {}
    for name in sorted(MECHANISM_ABLATIONS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        low = float(row[_delta_column(PRIMARY, "low")])
        high = float(row[_delta_column(PRIMARY, "high")])
        mechanism_checks[name] = {
            "ablation_minus_main_mean_min": mean,
            "ci_low": low,
            "ci_high": high,
            "performance_support": bool(
                mean >= MECHANISM_MIN_EFFECT_MIN and low > 0.0),
            "ablation_superior": bool(high < 0.0),
        }

    simple_row = rows[SIMPLE_CONFIG]
    simple_no_harm, simple_no_harm_checks = _candidate_no_harm(simple_row)
    simple_high = float(simple_row[_delta_column(PRIMARY, "high")])
    simple_check = {
        "candidate_minus_main_mean_min": float(
            simple_row[f"delta_{PRIMARY}_mean"]),
        "ci_low": float(simple_row[_delta_column(PRIMARY, "low")]),
        "ci_high": simple_high,
        "candidate_no_harm_pass": simple_no_harm,
        "candidate_no_harm_checks": simple_no_harm_checks,
        "noninferiority_pass": bool(
            simple_high <= SIMPLE_NONINFERIORITY_MARGIN_MIN and simple_no_harm),
    }

    frequency_supported = all(
        check["main_advantage_pass"] for check in frequency_checks.values())
    allocation_supported = all(
        check["allocation_advantage_pass"] for check in allocation_checks.values())
    mechanism_supported = all(
        check["performance_support"] for check in mechanism_checks.values())
    any_control_superior = any(
        check["control_superior"] for check in frequency_checks.values()) or any(
        check["control_superior"] for check in allocation_checks.values()) or any(
        check["ablation_superior"] for check in mechanism_checks.values())

    eligible_configs = []
    if (invariant_pass and not any_control_superior and frequency_supported
            and allocation_supported and mechanism_supported):
        eligible_configs.append(REFERENCE)
        if simple_check["noninferiority_pass"]:
            eligible_configs.append(SIMPLE_CONFIG)

    result = {
        "main_invariant_pass": invariant_pass,
        "main_invariants": invariants,
        "frequency_checks": frequency_checks,
        "allocation_checks": allocation_checks,
        "mechanism_checks": mechanism_checks,
        "simple_optimizer_check": simple_check,
        "eligible_configs": eligible_configs,
    }
    if not invariant_pass:
        result.update({
            "status": "implementation_contract_failed",
            "candidate": None,
            "reason": "one or more V6 main rollouts violated the contract",
        })
    elif any_control_superior:
        result.update({
            "status": "structural_redesign_required",
            "candidate": None,
            "reason": "a locked control or ablation is superior",
        })
    elif not frequency_supported:
        result.update({
            "status": "frequency_evidence_inconclusive",
            "candidate": None,
            "reason": "the locked frequency gate did not pass",
        })
    elif not allocation_supported:
        result.update({
            "status": "allocation_evidence_inconclusive",
            "candidate": None,
            "reason": "the locked layer-allocation gate did not pass",
        })
    elif not mechanism_supported:
        result.update({
            "status": "mechanism_evidence_inconclusive",
            "candidate": None,
            "reason": "the locked mechanism gate did not pass",
        })
    elif simple_check["noninferiority_pass"]:
        result.update({
            "status": "frequency_supported_simple_optimizer_candidate",
            "candidate": SIMPLE_CONFIG,
            "reason": "all gates passed and the simpler optimizer is non-inferior",
        })
    else:
        result.update({
            "status": "frequency_supported_main_retained",
            "candidate": REFERENCE,
            "reason": "all V6 gates passed and the main optimizer is retained",
        })
    return result


def _validate_development_manifest(
    path: Path,
) -> tuple[dict[str, object], list[int], list[int]]:
    payload = _load_json(path, "development decision manifest")
    if payload.get("decision_contract") != DECISION_CONTRACT:
        raise ValueError("development manifest uses another decision contract")
    if payload.get("decision_contract_sha256") != DECISION_CONTRACT_SHA256:
        raise ValueError("development manifest contract hash is invalid")
    if payload.get("stage") != "development":
        raise ValueError("development manifest does not record development stage")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("development manifest does not record locked V6")
    if payload.get("reference") != REFERENCE:
        raise ValueError("development manifest reference is not locked V6 main")
    if payload.get("primary_metric") != PRIMARY:
        raise ValueError("development manifest primary metric is not locked")
    if "selected_config" in payload:
        raise ValueError("development manifest illegally contains selected_config")
    candidate = payload.get("candidate_config")
    if candidate not in CONFIGS:
        raise ValueError("development manifest has no eligible locked candidate")
    expected_status = {
        REFERENCE: "frequency_supported_main_retained",
        SIMPLE_CONFIG: "frequency_supported_simple_optimizer_candidate",
    }.get(str(candidate))
    if expected_status is None or payload.get("status") != expected_status:
        raise ValueError(
            "development candidate is inconsistent with its decision status")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("development manifest lacks decision evidence")
    eligible = evidence.get("eligible_configs")
    if not isinstance(eligible, list) or candidate not in eligible:
        raise ValueError(
            "development candidate is inconsistent with its eligibility evidence")
    seed_sets = payload.get("seed_sets")
    if not isinstance(seed_sets, dict):
        raise ValueError("development manifest lacks seed sets")
    train_seeds = _seed_list(seed_sets.get("train"), "development train seeds")
    eval_seeds = _seed_list(seed_sets.get("eval"), "development eval seeds")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("development train and evaluation seeds overlap")
    matrix_binding = payload.get("matrix_manifest")
    if not isinstance(matrix_binding, dict) or len(str(
            matrix_binding.get("sha256", ""))) != 64:
        raise ValueError("development manifest lacks its matrix hash")
    artifact_bindings = payload.get("input_artifacts")
    if not isinstance(artifact_bindings, dict):
        raise ValueError("development manifest lacks artifact bindings")
    for name, primary_key in ARTIFACT_PRIMARY_KEYS.items():
        binding = artifact_bindings.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"development manifest lacks binding for {name}")
        if len(str(binding.get("sha256", ""))) != 64:
            raise ValueError(f"development artifact hash is invalid for {name}")
        if binding.get("primary_key") != list(primary_key):
            raise ValueError(f"development artifact key is invalid for {name}")
    return payload, train_seeds, eval_seeds


def decide(
    matrix_manifest: Path | str,
    *,
    stage: str,
    development_manifest: Path | str | None = None,
) -> dict[str, object]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    if stage == "development" and development_manifest is not None:
        raise ValueError("development stage must not consume a prior decision")
    if stage == "confirmation" and development_manifest is None:
        raise ValueError("confirmation requires --development-manifest")

    manifest_path = Path(matrix_manifest).resolve()
    manifest_sha = sha256_file(manifest_path)
    manifest = _load_json(manifest_path, "matrix manifest")
    train_seeds, eval_seeds, metrics = _validate_manifest_contract(
        manifest, stage=stage)
    frames, artifact_bindings = _load_artifacts(manifest_path, manifest)
    _validate_table_contracts(
        frames,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
    )
    invariant_pass, invariants = _main_invariants(frames[PER_EVAL_FILE], metrics)
    evidence = _evidence_decision(
        frames[PAIRED_FILE], invariant_pass, invariants)

    result = {
        "decision_contract": DECISION_CONTRACT,
        "decision_contract_sha256": DECISION_CONTRACT_SHA256,
        "protocol": PROTOCOL,
        "stage": stage,
        "matrix_manifest": {
            "sha256": manifest_sha,
            "manifest_version": MATRIX_MANIFEST_VERSION,
        },
        "input_artifacts": artifact_bindings,
        "seed_sets": {
            "train": train_seeds,
            "eval": eval_seeds,
        },
        "reference": REFERENCE,
        "primary_metric": PRIMARY,
        "evidence": {
            key: value for key, value in evidence.items()
            if key not in {"status", "candidate", "reason"}
        },
        "status": evidence["status"],
        "reason": evidence["reason"],
    }

    if stage == "development":
        result["candidate_config"] = evidence["candidate"]
        return result

    development_path = Path(development_manifest).resolve()  # type: ignore[arg-type]
    development, dev_train, dev_eval = _validate_development_manifest(
        development_path)
    current_seeds = set(train_seeds) | set(eval_seeds)
    development_seeds = set(dev_train) | set(dev_eval)
    overlap = sorted(current_seeds & development_seeds)
    if overlap:
        raise ValueError(
            f"confirmation seeds overlap development seeds: {overlap}")
    development_candidate = str(development["candidate_config"])
    eligible = set(evidence["eligible_configs"])
    selected = development_candidate if development_candidate in eligible else None
    if selected is not None:
        result["status"] = "confirmation_supported"
        result["reason"] = "independent confirmation replicated the candidate gates"
    elif evidence["candidate"] is not None:
        result["status"] = "confirmation_candidate_not_replicated"
        result["reason"] = "the development candidate did not pass confirmation"
    result["selected_config"] = selected
    result["development_decision"] = {
        "sha256": sha256_file(development_path),
        "decision_contract_sha256": development["decision_contract_sha256"],
        "matrix_manifest_sha256": development["matrix_manifest"]["sha256"],
        "candidate_config": development_candidate,
        "seed_sets": {"train": dev_train, "eval": dev_eval},
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--stage", required=True, choices=("development", "confirmation"))
    parser.add_argument("--development-manifest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = decide(
        args.matrix_manifest,
        stage=args.stage,
        development_manifest=args.development_manifest,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    config_key = (
        "candidate_config" if args.stage == "development" else "selected_config")
    print(json.dumps({
        "stage": args.stage,
        "status": result["status"],
        config_key: result[config_key],
        "out": str(out),
    }))


if __name__ == "__main__":
    main()
