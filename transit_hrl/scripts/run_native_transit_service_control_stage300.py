#!/usr/bin/env python3
"""Freeze a deployable service-only control on the old fit/calibration cohort."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_protected_stage297 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_control_stage300_v1"


def directory(run): return ROOT/"results"/run


def contract():
    return {"source": source.PARENT, "roles": ["fit", "calibration"],
        "hypothesis_informed_by_Stage298": True, "Stage298_E32_used_for_calibration": False,
        "family": "preserve_all_Stage297_base_risk_joint_candidates_plus_service_fit_quantiles1to16_and_zero",
        "action": "base_eligible_and_S8_paired_service_mean_le_cutoff_else_own_raw; no_batch_budget_or_fleet_score",
        "calibration": "same_Stage297_prefix_max_t_and_MC_Bonferroni_waiting_floor_over_entire_expanded_family; same_cost_upper_objective",
        "selection": "eligible_base_or_service_only; minimize_max_cost_CI95_upper_then_mean_then_index",
        "confirmation_gate": "selected_service_nontrivial_subset_and_cost_point_better_than_base; no_claim_from_development",
        "frozen_joint": "Stage298_locked_policy_unchanged; do_not_reselect_joint",
        "next_confirmation": "only_if_admitted; genuinely_new_prefixes_and_futures; Stage298_retired_from_tuning",
        "new_native_episodes": 0, "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def family(fleet, service, eligible):
    original = source.rule.family(fleet, service, eligible)
    cutoffs = np.unique(np.r_[0., np.quantile(service[eligible], source.rule.QUANTILES)])
    return original+[{"family": "service", "threshold": float(t)} for t in cutoffs]


def choices(raw, proposal, eligible, fleet, service, candidates):
    selected = []
    for candidate in candidates:
        if candidate["family"] == "service":
            selected.append(np.where(eligible & (service <= candidate["threshold"]), proposal, raw))
        else:
            selected.append(source.rule.choices(raw, proposal, eligible, fleet, service, [candidate])[:, 0])
    return np.stack(selected, axis=1)


def select_service(result):
    statistics = result["candidate_statistics"]
    eligible = [s for s in statistics if s["family"] in ("base", "service") and s["selection_eligible"]]
    selected = min(eligible, key=lambda s: (s["cost_upper_objective"], s["mean"]["cost"], s["index"])) if eligible else None
    base = statistics[0]
    admitted = bool(selected is not None and selected["family"] == "service"
        and 0 < selected["changed_profile_states"] < base["changed_profile_states"]
        and selected["mean"]["cost"] < base["mean"]["cost"])
    return selected, admitted


def run_calibration(run):
    parent = source.parent
    model = parent.read(parent.directory(source.PARENT)/"fit_raw/model_before_calibration.json")
    _, _, _, eligible, fleet, service = source.source_inputs("fit", model)
    candidates = family(fleet, service, eligible)
    write_json(directory(run)/"freeze_raw/family.json", {"contract": contract(), "candidates": candidates,
        "defined_before_calibration_targets": True})
    print("SERVICE_ONLY_FAMILY_FROZEN_BEFORE_CALIBRATION_TARGETS", flush=True)
    rows, raw, proposal, keep, fleet, service = source.source_inputs("calibration", model)
    selected = choices(raw, proposal, keep, fleet, service, candidates)
    targets, coverage = parent.load_panel(source.PARENT, "calibration", "target")
    actual = source.rule.values(targets, selected, raw)
    counts = np.sum(selected != raw[:, None, :], axis=(0, 2))
    result = source.rule.calibrate(actual, parent.native.cube(rows), candidates, counts)
    service_policy, admitted = select_service(result)
    write_json(directory(run)/"calibration_raw/statistics.json", result["candidate_statistics"])
    frozen = {"contract": contract(), "selection": service_policy, "fresh_confirmation_admitted": admitted}
    write_json(directory(run)/"frozen_raw/selected.json", frozen)
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, **frozen,
        "source_fit_prefixes": 80, "development_calibration_prefixes": 60,
        "family_candidates": len(candidates), "source_target_coverage": coverage,
        "base": result["base"], "prefix_retention_max_t_critical": result["prefix_retention_max_t_critical"],
        "confirmation_performed": False, "parent_files_modified": False,
        "interpretation": "development_control_only; hypothesis_informed_by_Stage298; needs_new_prefix_confirmation"})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    if source.parent.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract():
        p.error("Changed service-only control registration")
    run_calibration(a.run_name)


if __name__ == "__main__": main()
