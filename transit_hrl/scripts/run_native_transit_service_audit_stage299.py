#!/usr/bin/env python3
"""Paired inference on frozen Stage298 outcomes; no policy selection or fitting."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_confirmation_stage298 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_audit_stage299_v1"
SOURCE = "native_transit_service_confirmation_stage298_20261019_r1"
FIELDS = source.parent.learning.FIELDS


def directory(run): return ROOT/"results"/run


def contract():
    return {"source": SOURCE, "arms": list(source.ARMS),
        "scope": "post_confirmation_inference_only; no_policy_selection_or_new_claim_gate",
        "retention": "paired_W_joint_minus0.5W_base; negative_means_at_least_half_base_wait_gain",
        "inference": "average_shared_profiles_first; stratified_prefix_bootstrap5000_seed299; paired_MC_normal95; all32_and_both16",
        "groups": "fixed_root_regime_means_and_leave_one_prefix_sensitivity_are_descriptive",
        "registered_primary": "copy_Stage298_result_without_redefinition",
        "new_native_episodes": 0, "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def paired_statistics(actual, cube, *, arms=source.ARMS,
        comparators=("raw", "zero_abstain", "matched_thin", "service_rank_matched", "stage247", "legacy_risk"),
        primary="joint_budgeted", seed=299):
    weights, names = [], []
    for arm in comparators:
        w = np.zeros(len(arms)); w[arms.index(primary)] = 1.; w[arms.index(arm)] -= 1.
        weights.append(w); names.append(primary+"_vs_"+arm)
    w = np.zeros(len(arms)); w[arms.index(primary)], w[arms.index("zero_abstain")] = 1., -.5
    weights.append(w); names.append("half_base_wait_margin")
    values = np.einsum("sdaf,ka->sdkf", actual, np.asarray(weights))
    block = values[cube]
    g, r, j, d = block.shape[:4]
    rng = np.random.default_rng(seed)
    indices = [rng.integers(r, size=(5000, r)) for _ in range(g)]
    output = {}
    for label, ds in (("all32", slice(None)), ("first16", slice(0, 16)), ("last16", slice(16, 32))):
        current = values[:, ds]
        means = block[:, :, :, ds].mean(axis=(2, 3))
        point = means.mean(axis=(0, 1))
        samples = sum(means[k, indices[k]].mean(axis=1)/g for k in range(g))
        ci = np.quantile(samples, [.025, .975], axis=0)
        se = np.sqrt(current.var(axis=1, ddof=1).sum(axis=0)/(current.shape[1]*len(actual)**2))
        mc = np.stack((point-1.959963984540054*se, point+1.959963984540054*se))
        output[label] = {name: {"mean": dict(zip(FIELDS, map(float, point[k]))),
            "prefix_CI95": {f: ci[:, k, n].tolist() for n, f in enumerate(FIELDS)},
            "conditional_MC_CI95": {f: mc[:, k, n].tolist() for n, f in enumerate(FIELDS)}}
            for k, name in enumerate(names)}
    prefix = block.mean(axis=(2, 3))
    group_means = prefix.mean(axis=1)
    leave_one = group_means.mean(axis=0)+(group_means[:, None]-prefix)/(g*(r-1))
    leave_one = leave_one.reshape(g*r, len(names), len(FIELDS))
    output["descriptive_groups"] = [{"root": root, "scenario": scenario,
        "effects": {name: dict(zip(FIELDS, map(float, prefix[k].mean(axis=0)[n]))) for n, name in enumerate(names)}}
        for k, (root, scenario) in enumerate((root, scenario) for root in source.ROOTS for scenario in source.SCENARIOS)]
    output["leave_one_prefix_mean_range"] = {name: {f: [float(leave_one[:, k, n].min()), float(leave_one[:, k, n].max())]
        for n, f in enumerate(FIELDS)} for k, name in enumerate(names)}
    output["retention_margin_both_CIs_negative"] = {label: all(output[label]["half_base_wait_margin"][ci]["wait_min"][1] < 0
        for ci in ("prefix_CI95", "conditional_MC_CI95")) for label in ("all32", "first16", "last16")}
    return output


def run_audit(run):
    parent = source.directory(SOURCE)
    official = source.read(parent/"report/result.json")
    if not official["coverage_qualified"] or official["contract"] != source.contract(SOURCE):
        raise ValueError("A qualified frozen Stage298 result is required")
    plan, sealed = (source.read(parent/p) for p in ("seal/result.json", "sealed_raw/test.json"))
    rows = [r for key in source.keys() for r in source.read(source.prefix_path(SOURCE, key)/"rows.json")]
    panels = np.zeros((len(rows), 32, 6, 4))
    # Stage298's official report has already qualified these frozen receipts.
    for i in plan["active_indices"]:
        for chunk in range(4):
            for record in source.read(parent/f"evaluation/state{i}/chunk{chunk}/labels.json"):
                for action, value in record["values"].items():
                    panels[i, record["draw"]-32, int(action)] = value
    actual = source.parent.learning.effects(sealed, panels)
    result = paired_statistics(actual, source.parent.native.cube(rows))
    for pair, statistics in result["all32"].items():
        if pair == "half_base_wait_margin": continue
        for field in FIELDS:
            if abs(statistics["mean"][field]-official["comparisons"][pair]["mean"][field]) > 1e-12:
                raise ValueError("Audit differs from the qualified Stage298 paired means")
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(),
        "source_coverage_qualified": True, "test_prefixes": 400, "fixed_profiles": 16,
        "source_registered_primary_passed": official["joint_transfer_supported"],
        "source_point_retention": official["wait_retention_vs_zero_abstain"], **result})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    a = p.parse_args()
    if source.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract():
        p.error("Changed Stage299 inference specification")
    run_audit(a.run_name)


if __name__ == "__main__": main()
