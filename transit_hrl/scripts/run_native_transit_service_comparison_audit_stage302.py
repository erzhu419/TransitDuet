#!/usr/bin/env python3
"""Paired inference for frozen Stage301 joint and service-only outcomes."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_comparison_stage301 as source
from scripts.run_native_transit_service_audit_stage299 import paired_statistics
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_comparison_audit_stage302_v1"
SOURCE = "native_transit_service_comparison_stage301_20261020_r1"
COMPARATORS = ("raw", "zero_abstain", "matched_thin", "service_rank_matched", "stage247", "service_only")


def directory(run): return ROOT/"results"/run


def contract():
    return {"source": SOURCE, "arms": list(source.ARMS),
        "scope": "post_confirmation_paired_inference; no_policy_selection_or_new_claim_gate",
        "retention": "paired_W_controller_minus0.5W_base_for_joint_and_service_only",
        "inference": "average16profiles_first;400prefix_stratified_bootstrap5000_seed301;paired_MC95;all32_and_both16",
        "registered_primary": "copy_Stage301_result_without_redefinition",
        "groups": "root_regime_means_and_leave_one_prefix_ranges_are_descriptive",
        "new_native_episodes": 0, "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def run_audit(run):
    parent = source.directory(SOURCE)
    official = source.read(parent/"report/result.json")
    if not official["coverage_qualified"] or official["contract"] != source.contract(SOURCE):
        raise ValueError("Qualified frozen Stage301 report required")
    plan, sealed = (source.read(parent/p) for p in ("seal/result.json", "sealed_raw/test.json"))
    rows = [r for key in source.keys() for r in source.read(source.prefix_path(SOURCE, key)/"rows.json")]
    panels = np.zeros((len(rows), 32, 6, 4))
    for i in plan["active_indices"]:
        for chunk in range(4):
            for record in source.read(parent/f"evaluation/state{i}/chunk{chunk}/labels.json"):
                for action, value in record["values"].items():
                    panels[i, record["draw"]-32, int(action)] = value
    actual = source.parent.learning.effects(sealed, panels)
    cube = source.parent.native.cube(rows)
    joint = paired_statistics(actual, cube, arms=source.ARMS, comparators=COMPARATORS, seed=301)
    service = paired_statistics(actual, cube, arms=source.ARMS, primary="service_only",
        comparators=("raw", "zero_abstain"), seed=301)
    for result in (joint, service):
        for pair, statistics in result["all32"].items():
            if pair == "half_base_wait_margin": continue
            for field in source.parent.learning.FIELDS:
                if abs(statistics["mean"][field]-official["comparisons"][pair]["mean"][field]) > 1e-12:
                    raise ValueError("Audit differs from the frozen Stage301 paired means")
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(),
        "source_coverage_qualified": True, "test_prefixes": 400, "fixed_profiles": 16,
        "source_registered_primary_passed": official["critic_incremental_claim_supported"],
        "source_joint_transfer_supported": official["joint_transfer_supported"],
        "source_point_retention": official["wait_retention_vs_zero_abstain"],
        "joint": joint, "service_only": service})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    if source.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract():
        p.error("Changed Stage302 inference specification")
    run_audit(a.run_name)


if __name__ == "__main__": main()
