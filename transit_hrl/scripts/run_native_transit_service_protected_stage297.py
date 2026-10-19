#!/usr/bin/env python3
"""Select service-protected policies using only Stage296 fit/calibration caches."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_paired_fleet_stage296 as parent
from freq_hrl.domains.transit import native_service_protected_calibration as rule
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_protected_stage297_v1"
PARENT = "native_transit_paired_fleet_stage296_20261018_r1"


def directory(run): return ROOT/"results"/run


def contract():
    return {"source": PARENT, "source_roles": ["fit", "calibration"],
        "critic": "unchanged_Stage296_model_before_calibration; no_fitting",
        "proposal": "unchanged_P8_proposal_S8_zero_event_screen; retain_or_own_raw_only",
        "scores": {"risk": "frozen_fleet_critic", "joint": "same_fleet_critic_plus_S8_paired_service_mean"},
        "family": "fit_eligible_score_quantiles1to16_over16_plus_zero; shared_all_base_candidate; frozen_before_calibration_T16",
        "calibration": "60_source_calibration_prefixes_T16; development_selection_not_confirmation",
        "retention": "W_candidate_minus0.5W_base_upper_le0; stratified_prefix_max_t5000_seed297_upper97.5pct_over_both_families; MC_Bonferroni97.5pct; both_T8_points_le0",
        "admissibility": "base_wait_both_CIs_negative_both_T8_negative; candidate_cost_negative_pooled_and_both_T8; service_mean_negative",
        "selection": "minimum_max_of_prefix_and_conditional_MC_cost_CI95_upper; ties_cost_mean_then_index; same_rule_both_families",
        "new_confirmation_gate": "joint_selection_eligible_and_nontrivial_subset_of_base_and_cost_point_better_than_base",
        "next_test_cohort": "fresh120_balanced_prefixes; Stage296_test_replica12to23_reserved; no_Stage296_test_or_E32_reuse",
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0, "new_native_episodes": 0}


def source_inputs(role, model):
    if role not in ("fit", "calibration"):
        raise ValueError("Stage297 only opens source fit/calibration roles")
    rows = parent.load_rows(PARENT, role)
    raw, _, logits = parent.load_controls(PARENT, role)
    first, _ = parent.load_panel(PARENT, role, "proposal")
    proposal = parent.native.proposals(rows, raw, logits, first)
    screen, _ = parent.load_panel(PARENT, role, "screen")
    _, keep = parent.native.rule.screen(proposal, raw, screen)
    x = parent.learning.features(rows, raw, proposal, first, screen)
    fleet = parent.learning.predict(model, x)
    service = parent.native.rule.paired_values(screen, proposal, raw)[..., 2].mean(axis=2)
    return rows, raw, proposal, keep, fleet, service


def run_calibration(run):
    model = parent.read(parent.directory(PARENT)/"fit_raw/model_before_calibration.json")
    _, _, _, eligible, fleet, service = source_inputs("fit", model)
    candidates = rule.family(fleet, service, eligible)
    write_json(directory(run)/"freeze_raw/family.json", {"contract": contract(), "candidates": candidates,
        "source_model": PARENT+"/fit_raw/model_before_calibration.json",
        "defined_before_calibration_targets": True})
    print("SERVICE_PROTECTED_FAMILY_FROZEN_BEFORE_CALIBRATION_TARGETS", flush=True)
    rows, raw, proposal, keep, fleet, service = source_inputs("calibration", model)
    selected = rule.choices(raw, proposal, keep, fleet, service, candidates)
    write_json(directory(run)/"calibration_raw/choices.json", {"choices": selected.tolist()})
    targets, coverage = parent.load_panel(PARENT, "calibration", "target")
    actual = rule.values(targets, selected, raw)
    counts = np.sum(selected != raw[:, None, :], axis=(0, 2))
    result = rule.calibrate(actual, parent.native.cube(rows), candidates, counts)
    write_json(directory(run)/"calibration_raw/candidate_statistics.json", result.pop("candidate_statistics"))
    write_json(directory(run)/"frozen_raw/selected.json", {"contract": contract(),
        "selection": result["selection"], "fresh_confirmation_admitted": result["fresh_confirmation_admitted"]})
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(),
        "source_fit_prefixes": 80, "development_calibration_prefixes": 60, "states": len(rows),
        "fixed_profiles": len(raw), "family_candidates": len(candidates), "source_target_coverage": coverage,
        "Stage296_test_or_E32_read": False, "Stage294_E32_read": False,
        "parent_files_modified": False, "untouched_confirmation_performed": False, **result})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    if parent.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract():
        p.error("Changed Stage297 registration")
    run_calibration(a.run_name)


if __name__ == "__main__": main()
