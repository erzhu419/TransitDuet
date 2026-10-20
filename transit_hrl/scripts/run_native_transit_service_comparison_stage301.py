#!/usr/bin/env python3
"""Prospective fresh-prefix test of frozen joint versus service-only control."""

import argparse
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_confirmation_stage298 as previous
from scripts import run_native_transit_service_control_stage300 as control
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_comparison_stage301_v1"
SOURCE = "native_transit_service_confirmation_stage298_20261019_r1"
CONTROL = "native_transit_service_control_stage300_20261020_r1"
PHASES = ("freeze", "capture", "menus", "seal", "evaluate", "report")
ROOTS, SCENARIOS, PROBES, BANKS = previous.ROOTS, previous.SCENARIOS, previous.PROBES, previous.BANKS
ARMS = ("raw", "stage247", "zero_abstain", "service_only", "joint_budgeted", "service_rank_matched", "matched_thin")
parent = previous.parent
directory, prefix_path, sample_path, receipt_path = previous.directory, previous.prefix_path, previous.sample_path, previous.receipt_path
scene_seed, future_seed, read = previous.scene_seed, previous.future_seed, previous.read


def keys(role="test"):
    if role != "test": raise ValueError("Stage301 registers only fresh test prefixes")
    return [(role, root, scenario, replica) for root in ROOTS for scenario in SCENARIOS for replica in range(52, 92)]


def contract(run):
    return {"joint_source": SOURCE, "service_source": CONTROL,
        "locked_policy": read(directory(run)/"inputs/locked_policy.json"),
        "cohort": "400_fresh_prefixes_test_replicas52to91; balanced40_per_root_regime",
        "banks": {b: list(d) for b, d in BANKS.items()}, "arms": list(ARMS),
        "native_qualification": "reuse_Stage298_qualified_native_path; new_arm_is_subset_of_previously_supported_base_actions",
        "proposal": "same_P8_proposal_and_S8_zero_event_base; same_frozen_raw_critic_lower",
        "actions": "Stage298_joint_unchanged; Stage300_service_only_cutoff; all_actions_weights_unions_sealed_before_E32",
        "matching": "same_joint_count_service_ranking_and_expected_thinning_per_profile_root_regime_probe",
        "inference": "average16profiles_and6probes;400prefix_stratified_bootstrap5000_seed301;paired_MC95;both_E16",
        "primary": "joint_vs_service_only_cost_both_CIs_upper_lt0_and_both_E16_cost_differences_lt0;joint_vs_raw_original_joint_transfer_gate",
        "secondary": "service_only_vs_raw_and_joint_vs_matched_controls; separate_claims",
        "freeze": "old_cohorts_retired; no_new_calibration_or_refitting",
        "budget": {"captures": 400, "menus": 4800, "menu_native_episodes": 268800, "max_E32_native_episodes": 537600},
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def protocol(run):
    return SimpleNamespace(PROBES=PROBES, BANKS=BANKS, keys=keys, contract=partial(contract, run),
        prefix_path=prefix_path, sample_path=sample_path, scene_seed=scene_seed, future_seed=future_seed)


def freeze(run):
    locked = read(directory(run)/"inputs/locked_policy.json")
    calibrated = read(directory(CONTROL)/"report/result.json")
    old_report = read(directory(SOURCE)/"report/result.json")
    qualification = read(directory(SOURCE)/"pilot-report/result.json")
    chosen = calibrated["selection"]
    if (not calibrated["fresh_confirmation_admitted"] or not old_report["joint_transfer_supported"]
            or not qualification["software_qualified"]
            or locked["joint"] != read(directory(SOURCE)/"inputs/locked_policy.json")["joint"]
            or locked["service"] != {k: chosen[k] for k in ("family", "threshold", "index")}):
        raise ValueError("Prospective comparison requires both unchanged frozen policies")
    old = parent.keys()+parent.keys("pilot")+previous.keys()+previous.keys("pilot")
    fresh = keys()
    old_scenes = {scene_seed(k) for k in old}
    scenes = {scene_seed(k) for k in fresh}
    before = {future_seed(k, p, d) for k in old for p in PROBES for d in range(64)}
    after = {future_seed(k, p, d) for k in fresh for p in PROBES for d in range(64)}
    if len(scenes) != 400 or scenes & old_scenes or after & before or len(after) != 400*6*64 or max(after) >= 2**31:
        raise ValueError("Fresh comparison seed namespaces overlap or are incomplete")
    write_json(directory(run)/"freeze/result.json", {"contract": contract(run), "software_qualified": True,
        "fresh_prefixes": 400, "native_qualification_reused": True, "new_pilot_episodes": 0,
        "policies_locked_before_fresh_scenes": True, "E32_labels_read": False})


def seal_proposal(run, rows, raw, old, proposal, first, screen):
    sealed = previous._seal_proposal(run, rows, raw, old, proposal, first, screen)
    _, keep = parent.native.rule.screen(proposal, raw, screen)
    service = parent.native.rule.paired_values(screen, proposal, raw)[..., 2].mean(axis=2)
    cutoff = read(directory(run)/"inputs/locked_policy.json")["service"]["threshold"]
    choices = np.asarray(sealed["choices"])
    choices[:, 3] = np.where(keep & (service <= cutoff), proposal, raw)
    return {**sealed, "arms": list(ARMS), "choices": choices.tolist()}


def seal(run):
    rows = [r for k in keys() for r in read(prefix_path(run, k)/"rows.json")]
    controls = [read(prefix_path(run, k)/"controls.json") for k in keys()]
    raw, old, logits = (np.concatenate([np.asarray(c[n]) for c in controls], axis=1) for n in ("raw", "stage247", "logits"))
    first, _ = parent.load_panel(run, "test", "proposal", protocol=protocol(run))
    proposal = parent.native.proposals(rows, raw, logits, first)
    screen, _ = parent.load_panel(run, "test", "screen", protocol=protocol(run))
    sealed = seal_proposal(run, rows, raw, old, proposal, first, screen)
    unions = parent.learning.action_unions(sealed, [r["original_choice"] for r in rows])
    sealed["states"] = [{"key": ["test", r["root"], r["scenario"], r["replica"]],
        "probe": r["probe_index"], "candidates": unions[i]} for i, r in enumerate(rows)]
    write_json(directory(run)/"sealed_raw/test.json", sealed)
    active = [i for i, u in enumerate(unions) if u]
    write_json(directory(run)/"seal/result.json", {"contract": contract(run), "actions_and_weights_frozen": True,
        "E32_labels_read": False, "active_indices": active, "evaluation_tasks": 4*len(active),
        "evaluation_native_episodes": 32*sum(1+len(u) for u in unions if u),
        "changed_profile_states": {a: int((np.asarray(sealed["choices"])[:, n] != raw).sum()) for n, a in enumerate(ARMS[:-1])}})


def report(run):
    plan, sealed = read(directory(run)/"seal/result.json"), read(directory(run)/"sealed_raw/test.json")
    rows = [r for k in keys() for r in read(prefix_path(run, k)/"rows.json")]
    panels, episodes, seeds = np.zeros((len(rows), 32, 6, 4)), 0, set()
    for i in plan["active_indices"]:
        state = sealed["states"][i]
        for chunk in range(4):
            path = directory(run)/f"evaluation/state{i}/chunk{chunk}"
            receipt, labels = read(path/"receipt.json"), read(path/"labels.json")
            draws = list(BANKS["evaluation"][chunk*8:chunk*8+8])
            if (receipt["contract"] != contract(run) or not receipt["software_qualified"] or receipt["key"] != state["key"]
                    or receipt["probe"] != state["probe"] or receipt["candidates"] != state["candidates"]
                    or receipt["draws"] != draws or receipt["state_futures"] != 8 or receipt["exact_identities"] != 8
                    or receipt["native_episodes"] != 8*(1+len(state["candidates"])) or [r["draw"] for r in labels] != draws):
                raise ValueError("Incomplete frozen service-comparison receipt")
            episodes += receipt["native_episodes"]
            original = rows[i]["original_choice"]
            for record in labels:
                seed = future_seed(tuple(state["key"]), state["probe"], record["draw"])
                if record["future_seed"] != seed or seed in seeds or sorted(map(int, record["values"])) != state["candidates"]:
                    raise ValueError("Wrong future or incomplete frozen actions")
                seeds.add(seed)
                values = np.asarray([record["values"][str(a)] for a in state["candidates"]])
                if (not np.isfinite(values).all() or np.any(values[state["candidates"].index(original)] != 0)
                        or np.max(abs(values[:, 0]-values[:, 1]-values[:, 2])) > 1e-12):
                    raise ValueError("Invalid paired costs or original replay")
                for action, value in record["values"].items(): panels[i, record["draw"]-32, int(action)] = value
    if episodes != plan["evaluation_native_episodes"] or len(seeds) != len(plan["active_indices"])*32:
        raise ValueError("Frozen comparison physical budget differs")
    result = parent.learning.summarize(sealed, panels, parent.native.cube(rows), seed=301)
    pair = result["comparisons"]["joint_budgeted_vs_service_only"]
    both_halves = all(result["halves"][h]["joint_budgeted"]["cost"] < result["halves"][h]["service_only"]["cost"] for h in ("first16", "last16"))
    incremental = all(pair[c]["cost"][1] < 0 for c in ("prefix_CI95", "conditional_MC_CI95"))
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(run),
        "coverage_qualified": True, "test_prefixes": 400, "evaluation_native_episodes": episodes, "state_futures": len(seeds),
        "critic_incremental_claim_supported": bool(incremental and both_halves and result["joint_transfer_supported"]), **result})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=PHASES, required=True); p.add_argument("--role", choices=("test",))
    p.add_argument("--root", type=int, choices=ROOTS); p.add_argument("--scenario", choices=SCENARIOS); p.add_argument("--replica", type=int)
    p.add_argument("--probe", type=int, choices=PROBES); p.add_argument("--bank", choices=("proposal", "screen"))
    p.add_argument("--state-index", type=int); p.add_argument("--chunk", type=int, default=0)
    a = p.parse_args(); run = a.run_name
    if read(directory(run)/"inputs/preregistration.json")["contract"] != contract(run): p.error("Changed comparison registration")
    if a.phase != "freeze" and not read(directory(run)/"freeze/result.json")["software_qualified"]: p.error("Freeze first")
    if a.phase in ("capture", "menus"):
        key = (a.role, a.root, a.scenario, a.replica)
        if key not in keys(): p.error("Unregistered fresh prefix")
        if a.phase == "capture": receipt = parent.capture(run, key, protocol=protocol(run))
        else:
            if a.bank is None or a.probe is None or a.chunk != 0: p.error("Unregistered menu")
            _, receipt = parent.rollout(run, key, a.probe, BANKS[a.bank], list(range(6)), sample_path(run, key, a.probe, a.bank), protocol=protocol(run))
        write_json(receipt_path(run, a.phase, key, a.probe, a.bank)/"receipt.json", receipt)
    elif a.phase == "evaluate":
        plan = read(directory(run)/"seal/result.json")
        if not plan["actions_and_weights_frozen"] or a.state_index not in plan["active_indices"] or a.chunk not in range(4): p.error("Unregistered evaluation")
        state = read(directory(run)/"sealed_raw/test.json")["states"][a.state_index]
        _, receipt = parent.rollout(run, tuple(state["key"]), state["probe"], BANKS["evaluation"][a.chunk*8:a.chunk*8+8],
            state["candidates"], directory(run)/f"evaluation/state{a.state_index}/chunk{a.chunk}", protocol=protocol(run))
        write_json(receipt_path(run, "evaluate", index=a.state_index, chunk=a.chunk)/"receipt.json", receipt)
    else: {"freeze": freeze, "seal": seal, "report": report}[a.phase](run)


if __name__ == "__main__": main()
