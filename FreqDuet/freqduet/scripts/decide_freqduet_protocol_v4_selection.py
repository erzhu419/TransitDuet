#!/usr/bin/env python3
"""Apply the locked protocol-v4 selection and no-harm gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = "service_cost_restricted"
PRIMARY_SUPERIORITY_MARGIN = -0.01
PRIMARY_SIMPLICITY_MARGIN = 0.01
NO_HARM_LIMITS = {
    "passenger_unserved_rate": ("high", 0.005),
    "trip_launch_rate": ("low", -0.005),
    "trip_completion_rate": ("low", -0.005),
    "headway_cv": ("high", 0.02),
    "fleet_overshoot": ("high", 0.0),
    "restricted_in_vehicle_horizon_min": ("high", 0.5),
    "restricted_total_journey_horizon_min": ("high", 0.5),
    "fleet_denied_trips": ("high", 1.0),
    "fleet_readiness_delay_mean_s": ("high", 15.0),
}


def _column(metric: str, suffix: str) -> str:
    return f"delta_{metric}_ci_{suffix}"


def _required_columns() -> set[str]:
    columns = {
        "candidate",
        "reference",
        f"delta_{PRIMARY}_mean",
        _column(PRIMARY, "high"),
    }
    for metric, (side, _) in NO_HARM_LIMITS.items():
        columns.add(_column(metric, side))
    return columns


def _no_harm(row: pd.Series) -> tuple[bool, dict[str, dict[str, object]]]:
    checks = {}
    passed = True
    for metric, (side, limit) in NO_HARM_LIMITS.items():
        value = float(row[_column(metric, side)])
        ok = value <= limit if side == "high" else value >= limit
        checks[metric] = {
            "ci_side": side,
            "observed": value,
            "limit": float(limit),
            "pass": bool(ok),
        }
        passed = passed and bool(ok)
    return passed, checks


def decide(
    frame: pd.DataFrame,
    *,
    reference: str,
    simple_config: str,
    frequency_failure_configs: set[str],
) -> dict[str, object]:
    missing = sorted(_required_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"paired-delta table is missing columns: {missing}")
    if frame["candidate"].duplicated().any():
        raise ValueError("paired-delta table contains duplicate candidates")
    if set(frame["reference"].astype(str)) != {str(reference)}:
        raise ValueError("paired-delta table does not use the locked reference")
    numeric_columns = sorted(_required_columns() - {"candidate", "reference"})
    if not np.isfinite(frame[numeric_columns].astype(float).to_numpy()).all():
        raise ValueError("paired-delta decision columns must all be finite")

    rows = []
    for _, row in frame.iterrows():
        no_harm, checks = _no_harm(row)
        mean = float(row[f"delta_{PRIMARY}_mean"])
        ci_high = float(row[_column(PRIMARY, "high")])
        superior = (
            mean <= PRIMARY_SUPERIORITY_MARGIN and ci_high < 0.0 and no_harm
        )
        simple_noninferior = (
            str(row["candidate"]) == str(simple_config)
            and ci_high <= PRIMARY_SIMPLICITY_MARGIN
            and no_harm
        )
        rows.append({
            "candidate": str(row["candidate"]),
            "primary_delta_mean": mean,
            "primary_ci_high": ci_high,
            "no_harm_pass": bool(no_harm),
            "no_harm_checks": checks,
            "superiority_pass": bool(superior),
            "simplicity_noninferiority_pass": bool(simple_noninferior),
        })

    mean_ranking = sorted(
        [(reference, 0.0)]
        + [(item["candidate"], item["primary_delta_mean"]) for item in rows],
        key=lambda item: (item[1], item[0]),
    )
    best_name, best_delta = mean_ranking[0]
    result = {
        "decision_contract": "freqduet-protocol-v4-selection-v1",
        "reference": reference,
        "primary_metric": PRIMARY,
        "primary_superiority_margin": PRIMARY_SUPERIORITY_MARGIN,
        "simplicity_noninferiority_margin": PRIMARY_SIMPLICITY_MARGIN,
        "mean_ranking": [
            {"config": name, "delta": float(delta)}
            for name, delta in mean_ranking
        ],
        "candidate_checks": rows,
    }
    if best_name in frequency_failure_configs and best_delta < 0.0:
        result.update({
            "status": "frequency_claim_failed",
            "selected_config": None,
            "reason": (
                "a locked no-frequency/raw-history control has the lowest "
                "mean primary endpoint"
            ),
        })
        return result

    superior = [item for item in rows if item["superiority_pass"]]
    simple = [
        item for item in rows if item["simplicity_noninferiority_pass"]]
    structural = [
        item for item in superior if item["candidate"] != simple_config]
    if structural and simple:
        result.update({
            "status": "factorial_followup_required",
            "selected_config": None,
            "reason": (
                "a mechanism change is superior while the simpler optimizer "
                "is non-inferior; their combination must be tested"
            ),
        })
    elif structural:
        selected = min(
            structural,
            key=lambda item: (item["primary_delta_mean"], item["candidate"]),
        )
        result.update({
            "status": "single_axis_candidate_selected",
            "selected_config": selected["candidate"],
            "reason": "candidate passed superiority and every no-harm gate",
        })
    elif simple:
        result.update({
            "status": "simpler_optimizer_selected",
            "selected_config": simple_config,
            "reason": "standard constrained SAC passed locked non-inferiority",
        })
    else:
        result.update({
            "status": "reference_retained",
            "selected_config": reference,
            "reason": "no candidate passed a locked replacement rule",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-deltas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--reference", default="F_freqduet_protocol_v4_main_hiro")
    parser.add_argument(
        "--simple-config", default="F_freqduet_protocol_v4_csac_hiro")
    parser.add_argument(
        "--frequency-failure-configs",
        default=(
            "F_freqduet_protocol_v4_nofreq_hiro,"
            "F_freqduet_protocol_v4_rawhistory_hiro"
        ),
    )
    args = parser.parse_args()
    result = decide(
        pd.read_csv(args.paired_deltas),
        reference=args.reference,
        simple_config=args.simple_config,
        frequency_failure_configs={
            value.strip() for value in args.frequency_failure_configs.split(",")
            if value.strip()
        },
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
