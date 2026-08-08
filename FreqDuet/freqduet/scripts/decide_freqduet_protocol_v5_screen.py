#!/usr/bin/env python3
"""Apply the preregistered internal decision rules for the v5 dev screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "F_freqduet_protocol_v5_main_hiro"
PRIMARY = "restricted_total_journey_horizon_min"
FREQUENCY_CONTROLS = {
    "F_freqduet_protocol_v5_nofreq_hiro",
    "F_freqduet_protocol_v5_rawhistory_hiro",
}
ALLOCATION_CONTROLS = {
    "F_freqduet_protocol_v5_allfreq_hiro",
    "F_freqduet_protocol_v5_upperonly_hiro",
    "F_freqduet_protocol_v5_loweronly_hiro",
}
MECHANISM_ABLATIONS = {
    "F_freqduet_protocol_v5_nobudget_hiro",
    "F_freqduet_protocol_v5_noguard_hiro",
    "F_freqduet_protocol_v5_noloadcost_hiro",
    "F_freqduet_protocol_v5_waitonlycredit_hiro",
}
SIMPLE_CONFIG = "F_freqduet_protocol_v5_csac_hiro"
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


def _delta_column(metric: str, suffix: str) -> str:
    return f"delta_{metric}_ci_{suffix}"


def _required_delta_columns() -> set[str]:
    columns = {
        "candidate", "reference", f"delta_{PRIMARY}_mean",
        _delta_column(PRIMARY, "low"), _delta_column(PRIMARY, "high"),
    }
    for metric in REFERENCE_NO_HARM_LIMITS:
        columns.add(_delta_column(metric, "low"))
        columns.add(_delta_column(metric, "high"))
    return columns


def _reference_no_harm(row: pd.Series):
    """Check that the reference is not harmed relative to the candidate."""
    checks = {}
    passed = True
    for metric, (direction, margin) in REFERENCE_NO_HARM_LIMITS.items():
        if direction == "min":
            observed = float(row[_delta_column(metric, "low")])
            ok = observed >= -float(margin)
            rule = "candidate_minus_reference_ci_low >= -margin"
        else:
            observed = float(row[_delta_column(metric, "high")])
            ok = observed <= float(margin)
            rule = "candidate_minus_reference_ci_high <= margin"
        checks[metric] = {
            "observed": observed,
            "margin": float(margin),
            "rule": rule,
            "pass": bool(ok),
        }
        passed = passed and bool(ok)
    return bool(passed), checks


def _candidate_no_harm(row: pd.Series):
    """Check that a replacement candidate is not harmed vs the reference."""
    checks = {}
    passed = True
    for metric, (direction, margin) in REFERENCE_NO_HARM_LIMITS.items():
        if direction == "min":
            observed = float(row[_delta_column(metric, "high")])
            ok = observed <= float(margin)
            rule = "candidate_minus_reference_ci_high <= margin"
        else:
            observed = float(row[_delta_column(metric, "low")])
            ok = observed >= -float(margin)
            rule = "candidate_minus_reference_ci_low >= -margin"
        checks[metric] = {
            "observed": observed,
            "margin": float(margin),
            "rule": rule,
            "pass": bool(ok),
        }
        passed = passed and bool(ok)
    return bool(passed), checks


def _main_invariants(summary: pd.DataFrame, reference: str):
    required = {
        "config",
        f"{PRIMARY}_mean",
        "lower_causal_guard_enabled_mean",
        "upper_plan_projected_delta_sum_abs_mean_s_mean",
        "upper_interval_onboard_cost_sum_mean",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing v5 invariant columns: {missing}")
    rows = summary[summary["config"].astype(str).eq(str(reference))]
    if len(rows) != 1:
        raise ValueError("summary must contain exactly one reference row")
    row = rows.iloc[0]
    values = {
        "primary_finite": bool(np.isfinite(float(row[f"{PRIMARY}_mean"]))),
        "causal_guard_enabled": (
            abs(float(row["lower_causal_guard_enabled_mean"]) - 1.0) <= 1e-9),
        "headway_budget_conserved": (
            float(row[
                "upper_plan_projected_delta_sum_abs_mean_s_mean"]) <= 1e-5),
        "onboard_credit_observed": (
            float(row["upper_interval_onboard_cost_sum_mean"]) > 0.0),
    }
    return all(values.values()), values


def decide(
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    reference: str = REFERENCE,
) -> dict[str, object]:
    missing = sorted(_required_delta_columns() - set(paired.columns))
    if missing:
        raise ValueError(f"paired-delta table is missing columns: {missing}")
    if paired["candidate"].duplicated().any():
        raise ValueError("paired-delta table contains duplicate candidates")
    if set(paired["reference"].astype(str)) != {str(reference)}:
        raise ValueError("paired-delta table does not use the locked reference")
    numeric = sorted(_required_delta_columns() - {"candidate", "reference"})
    if not np.isfinite(paired[numeric].astype(float).to_numpy()).all():
        raise ValueError("paired-delta decision columns must be finite")

    invariant_pass, invariants = _main_invariants(summary, reference)
    rows = {str(row["candidate"]): row for _, row in paired.iterrows()}
    missing_configs = sorted(
        (FREQUENCY_CONTROLS | ALLOCATION_CONTROLS
         | MECHANISM_ABLATIONS | {SIMPLE_CONFIG})
        - set(rows))
    if missing_configs:
        raise ValueError(f"screen is missing locked configs: {missing_configs}")

    frequency_checks = {}
    for name in sorted(FREQUENCY_CONTROLS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        ci_low = float(row[_delta_column(PRIMARY, "low")])
        ci_high = float(row[_delta_column(PRIMARY, "high")])
        no_harm, no_harm_checks = _reference_no_harm(row)
        frequency_checks[name] = {
            "candidate_minus_main_mean_min": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "main_advantage_pass": bool(
                mean >= FREQUENCY_MIN_ADVANTAGE_MIN
                and ci_low > 0.0
                and no_harm),
            "control_superior": bool(ci_high < 0.0),
            "main_no_harm_pass": no_harm,
            "main_no_harm_checks": no_harm_checks,
        }

    allocation_checks = {}
    for name in sorted(ALLOCATION_CONTROLS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        ci_low = float(row[_delta_column(PRIMARY, "low")])
        ci_high = float(row[_delta_column(PRIMARY, "high")])
        no_harm, no_harm_checks = _reference_no_harm(row)
        allocation_checks[name] = {
            "candidate_minus_main_mean_min": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "allocation_advantage_pass": bool(
                mean >= ALLOCATION_MIN_ADVANTAGE_MIN
                and ci_low > 0.0
                and no_harm),
            "allocation_control_superior": bool(ci_high < 0.0),
            "main_no_harm_pass": no_harm,
            "main_no_harm_checks": no_harm_checks,
        }

    mechanism_checks = {}
    for name in sorted(MECHANISM_ABLATIONS):
        row = rows[name]
        mean = float(row[f"delta_{PRIMARY}_mean"])
        ci_low = float(row[_delta_column(PRIMARY, "low")])
        ci_high = float(row[_delta_column(PRIMARY, "high")])
        mechanism_checks[name] = {
            "ablation_minus_main_mean_min": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "performance_support": bool(
                mean >= MECHANISM_MIN_EFFECT_MIN and ci_low > 0.0),
            "ablation_superior": bool(ci_high < 0.0),
        }

    simple_row = rows[SIMPLE_CONFIG]
    simple_no_harm, simple_no_harm_checks = _candidate_no_harm(simple_row)
    simple_ci_high = float(simple_row[_delta_column(PRIMARY, "high")])
    simple_check = {
        "candidate_minus_main_mean_min": float(
            simple_row[f"delta_{PRIMARY}_mean"]),
        "ci_low": float(simple_row[_delta_column(PRIMARY, "low")]),
        "ci_high": simple_ci_high,
        "candidate_no_harm_pass": simple_no_harm,
        "candidate_no_harm_checks": simple_no_harm_checks,
        "noninferiority_pass": bool(
            simple_ci_high <= SIMPLE_NONINFERIORITY_MARGIN_MIN
            and simple_no_harm),
    }

    any_control_superior = any(
        check["control_superior"] for check in frequency_checks.values())
    any_allocation_control_superior = any(
        check["allocation_control_superior"]
        for check in allocation_checks.values())
    frequency_supported = all(
        check["main_advantage_pass"] for check in frequency_checks.values())
    allocation_supported = all(
        check["allocation_advantage_pass"]
        for check in allocation_checks.values())
    result = {
        "decision_contract": "freqduet-protocol-v5-dev-screen-v1",
        "reference": reference,
        "primary_metric": PRIMARY,
        "frequency_min_advantage_min": FREQUENCY_MIN_ADVANTAGE_MIN,
        "allocation_min_advantage_min": ALLOCATION_MIN_ADVANTAGE_MIN,
        "mechanism_min_effect_min": MECHANISM_MIN_EFFECT_MIN,
        "simple_noninferiority_margin_min": (
            SIMPLE_NONINFERIORITY_MARGIN_MIN),
        "main_invariant_pass": invariant_pass,
        "main_invariants": invariants,
        "frequency_checks": frequency_checks,
        "allocation_checks": allocation_checks,
        "mechanism_checks": mechanism_checks,
        "simple_optimizer_check": simple_check,
    }
    if not invariant_pass:
        result.update({
            "status": "implementation_contract_failed",
            "selected_config": None,
            "reason": "the v5 main did not satisfy its executable invariants",
        })
    elif any_control_superior or any_allocation_control_superior:
        result.update({
            "status": "structural_redesign_required",
            "selected_config": None,
            "reason": (
                "a locked frequency or layer-allocation control is superior"),
        })
    elif not frequency_supported:
        result.update({
            "status": "frequency_evidence_inconclusive",
            "selected_config": None,
            "reason": (
                "the harmonic split did not clear both locked frequency "
                "advantage and no-harm gates"),
        })
    elif not allocation_supported:
        result.update({
            "status": "allocation_evidence_inconclusive",
            "selected_config": None,
            "reason": (
                "frequency features passed, but LF/HF layer allocation did "
                "not clear all locked contribution gates"),
        })
    elif simple_check["noninferiority_pass"]:
        result.update({
            "status": "frequency_supported_simple_optimizer_candidate",
            "selected_config": SIMPLE_CONFIG,
            "reason": (
                "frequency evidence passed and the simpler optimizer was "
                "non-inferior"),
        })
    else:
        result.update({
            "status": "frequency_supported_main_retained",
            "selected_config": reference,
            "reason": "frequency evidence passed; the v5 main is retained",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default=REFERENCE)
    args = parser.parse_args()
    result = decide(
        pd.read_csv(args.paired_deltas),
        pd.read_csv(args.summary),
        reference=args.reference,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "selected_config": result["selected_config"],
        "out": str(out),
    }))


if __name__ == "__main__":
    main()
