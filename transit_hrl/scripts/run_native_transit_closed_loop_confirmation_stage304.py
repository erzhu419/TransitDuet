#!/usr/bin/env python3
"""Frozen two-decision native controller evaluation on independent full scenes."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_closed_loop_preflight_stage303 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_closed_loop_confirmation_stage304_v1"
AUDIT = "native_transit_closed_loop_preflight_audit_stage303_20261021_r1"
ARMS = ("raw", "zero_abstain", "service_only", "joint_budgeted")
REPLICAS = tuple(range(32))
FIELDS = ("cost", "fleet", "service", "wait_min")
PHASES = ("freeze", "evaluate", "report")


def directory(run): return ROOT/"results"/run


def scene_seed(root, scenario, replica):
    return 2080000000+source.source.ROOTS.index(root)*100000+source.source.SCENARIOS.index(scenario)*1000+replica


def future_seed(root, scenario, replica, probe, draw):
    return 2030000000+source.source.ROOTS.index(root)*1000000+source.source.SCENARIOS.index(scenario)*100000+replica*1000+source.PROBES.index(probe)*100+draw


def contract():
    return {"qualification_source": AUDIT, "locked_policy": source.contract()["locked_policy"],
        "scope": "two_native_upper_decisions_fixed_lower;profile0;not_end_to_end_HRL",
        "arms": list(ARMS), "profile": 0, "probes": list(source.PROBES),
        "cohort": "320_fresh_full_native_scenes;32_per_root_regime;paired4controllers;no_pilot_scenes",
        "actual_world": "2080000000+root_index100000+regime_index1000+replica",
        "inner_future": "2030000000+root_index1000000+regime_index100000+replica1000+probe_ordinal100+draw",
        "banks": {k: list(v) for k, v in source.BANKS.items()},
        "freeze": "freeze_algorithm_cutoffs_seeds_and_inference_before_new_scenes;actions_use_current_on_policy_state",
        "history": "own_executed_prior_actions;new_native_context_and_independent_P8S8_at_second_probe",
        "primary": "service_only_vs_raw_cost_wait_service_both_CI_upper_lt0;cost_wait_both_seed_halves_negative;half_base_wait_margin_both_CIs_negative_pooled_and_halves",
        "secondary": "joint_vs_raw;zero_abstain_vs_raw;joint_vs_service_only_cost_both_CIs_upper_lt0_and_both_halves_negative_with_joint_transfer",
        "inference": "320_independent_episode_units;stratified_bootstrap5000_seed304_and_paired_stratified_normal95;first16_last16_scenes_each_group",
        "CI_boundary": "both_intervals_measure_full_episode_sampling;neither_is_old_fixed_prefix_conditional_MC",
        "budget": {"evaluation_tasks": 1280, "paired_scene_units": 320, "actual_controller_episodes": 1280,
            "native_episodes_including_inner_and_identity": 219520},
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0}


def context(run, replica):
    return SimpleNamespace(PROTOCOL=PROTOCOL, REPLICA=replica, PROBES=source.PROBES, BANKS=source.BANKS,
        directory=lambda _: directory(run)/"replicas"/str(replica), contract=contract,
        scene_seed=lambda r, s: scene_seed(r, s, replica),
        future_seed=lambda r, s, p, d: future_seed(r, s, replica, p, d))


def cell_path(run, root, scenario, replica, arm):
    return context(run, replica).directory(run)/"cells"/str(root)/scenario/arm


def freeze(run):
    audit = source.source.read(directory(AUDIT)/"report/result.json")
    receipt = source.source.read(directory(AUDIT)/"execution_receipt.json")
    if not audit["all_pilots_qualified"] or receipt["scheduler"]["status"] != "done" or receipt["scheduler"]["exit_code"] != 0:
        raise ValueError("Qualified native preflight audit required")
    old = source.parent.keys()+source.parent.keys("pilot")+source.source.previous.keys()+source.source.keys()
    scenes = {scene_seed(r, s, n) for r in source.source.ROOTS for s in source.source.SCENARIOS for n in REPLICAS}
    inner = {future_seed(r, s, n, p, d) for r in source.source.ROOTS for s in source.source.SCENARIOS
        for n in REPLICAS for p in source.PROBES for d in range(16)}
    reserved_scenes = {source.source.scene_seed(k) for k in old}|{source.scene_seed(r, s) for r in source.source.ROOTS for s in source.source.SCENARIOS}
    reserved_inner = {source.source.future_seed(k, p, d) for k in old for p in source.source.PROBES for d in range(64)}
    reserved_inner.update(source.future_seed(r, s, p, d) for r in source.source.ROOTS for s in source.source.SCENARIOS for p in source.PROBES for d in range(16))
    if len(scenes) != 320 or len(inner) != 10240 or scenes & reserved_scenes or inner & reserved_inner or scenes & inner or max(inner|scenes) >= 2**31:
        raise ValueError("Formal closed-loop seed namespace overlap")
    write_json(directory(run)/"freeze/result.json", {"contract": contract(), "software_qualified": True,
        "scene_units": 320, "distinct_inner_seeds": 10240, "policies_frozen_before_formal_scenes": True,
        "pilot_metrics_used_for_selection": False, "new_native_episodes": 0})


def summarize(values):
    pairs = [(a, "raw") for a in ARMS[1:]]+[("joint_budgeted", "service_only"),
        ("joint_budgeted", "zero_abstain"), ("service_only", "zero_abstain")]
    output = {}
    for label, ds in (("all32_scenes", slice(None)), ("first16_scenes", slice(0, 16)), ("last16_scenes", slice(16, 32))):
        current = values[:, ds]
        g, n = current.shape[:2]
        weights, names = [], []
        for after, before in pairs:
            w = np.zeros(len(ARMS)); w[ARMS.index(after)], w[ARMS.index(before)] = 1., -1.
            weights.append(w); names.append(after+"_vs_"+before)
        for arm in ("service_only", "joint_budgeted"):
            w = np.zeros(len(ARMS)); w[ARMS.index(arm)], w[ARMS.index("zero_abstain")] = 1., -.5
            weights.append(w); names.append(arm+"_half_base_wait_margin")
        effects = np.einsum("grvf,kv->grkf", current, np.asarray(weights))
        point = effects.mean(axis=(0, 1)); rng = np.random.default_rng(304)
        samples = sum(effects[i, rng.integers(n, size=(5000, n))].mean(axis=1)/g for i in range(g))
        ci = np.quantile(samples, [.025, .975], axis=0)
        se = np.sqrt(effects.var(axis=1, ddof=1).sum(axis=0)/(g*g*n))
        normal = np.stack((point-1.959963984540054*se, point+1.959963984540054*se))
        output[label] = {name: {"mean": dict(zip(FIELDS, map(float, point[k]))),
            "episode_bootstrap_CI95": {f: ci[:, k, j].tolist() for j, f in enumerate(FIELDS)},
            "paired_normal_CI95": {f: normal[:, k, j].tolist() for j, f in enumerate(FIELDS)}}
            for k, name in enumerate(names)}
    def passed(part, pair, fields):
        return all(output[part][pair][ci][f][1] < 0 for ci in ("episode_bootstrap_CI95", "paired_normal_CI95") for f in fields)
    transfer = {}
    retention = {}
    for arm in ("service_only", "joint_budgeted"):
        retention[arm] = {}
        for part in output:
            current = values[:, {"all32_scenes": slice(None), "first16_scenes": slice(0, 16), "last16_scenes": slice(16, 32)}[part]]
            base = current[:, :, 1, 3].mean()
            retention[arm][part] = float(current[:, :, ARMS.index(arm), 3].mean()/base) if base < 0 else None
        transfer[arm] = (passed("all32_scenes", arm+"_vs_raw", ("cost", "service", "wait_min"))
            and all(output[h][arm+"_vs_raw"]["mean"][f] < 0 for h in ("first16_scenes", "last16_scenes") for f in ("cost", "wait_min"))
            and all(v is not None and v >= .5 for v in retention[arm].values())
            and all(passed(h, arm+"_half_base_wait_margin", ("wait_min",)) for h in output))
    return {"comparisons": output, "wait_retention_vs_base": retention,
        "service_closed_loop_supported": bool(transfer["service_only"]), "joint_closed_loop_supported": bool(transfer["joint_budgeted"]),
        "critic_incremental_cost_supported": bool(transfer["joint_budgeted"] and passed("all32_scenes", "joint_budgeted_vs_service_only", ("cost",))
            and all(output[h]["joint_budgeted_vs_service_only"]["mean"]["cost"] < 0 for h in ("first16_scenes", "last16_scenes")))}


def report(run):
    values = np.zeros((10, 32, len(ARMS), 4)); episodes = 0
    for g, (root, scenario) in enumerate((r, s) for r in source.source.ROOTS for s in source.source.SCENARIOS):
        for replica in REPLICAS:
            first_row, second_by_prior = None, {}
            for a, arm in enumerate(ARMS):
                r = source.source.read(cell_path(run, root, scenario, replica, arm)/"result.json")
                expected = 5 if arm == "raw" else 227
                if (r["protocol"] != PROTOCOL or r["contract"] != contract() or not r["software_qualified"] or r["performance_claim"]
                        or r["root"] != root or r["scenario"] != scenario or r["arm"] != arm or r["profile"] != 0
                        or r["native_episodes"] != expected or r["ledger_qualified_episodes"] != expected
                        or r["prefix_state_rng_matches"] != (1 if arm == "raw" else 224)
                        or r["exact_identity_replays"] != (1 if arm == "raw" else 32)
                        or [d["probe"] for d in r["decisions"]] != list(source.PROBES)):
                    raise ValueError("Invalid formal closed-loop receipt")
                history = {}
                for d in r["decisions"]:
                    row = d["row"]
                    if d["history_before"] != history or row["replica"] != replica or row["scene_seed"] != scene_seed(root, scenario, replica):
                        raise ValueError("Formal controller used a different scene or prior actions")
                    history[str(d["probe"])] = d["selected"]
                initial, second = (d["row"] for d in r["decisions"])
                if first_row is not None and initial != first_row:
                    raise ValueError("Paired controllers started from different native states")
                first_row = initial
                prior = r["decisions"][0]["selected"]
                if prior in second_by_prior and second != second_by_prior[prior]:
                    raise ValueError("Same prior action produced mismatched second native states")
                second_by_prior[prior] = second
                m = r["actual_metrics"]
                cost, wait, fleet = m["service_cost_restricted"], m["restricted_wait_horizon_min"], max(m["peak_fleet"]-12, 0)**2/12
                values[g, replica, a] = [cost, fleet, cost-fleet, wait]; episodes += expected
    if episodes != 219520 or not np.isfinite(values).all(): raise ValueError("Formal native budget or metrics differ")
    effects = values-values[:, :, :1]
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "independent_scene_units": 320, "actual_controller_episodes": 1280,
        "native_episodes_including_inner": episodes, "fixed_profiles": 1,
        "descriptive_group_effects": {f"{r}/{s}": {arm: dict(zip(FIELDS, map(float, effects[g, :, a].mean(axis=0)))) for a, arm in enumerate(ARMS)}
            for g, (r, s) in enumerate((r, s) for r in source.source.ROOTS for s in source.source.SCENARIOS)}, **summarize(effects)})


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    p.add_argument("--phase", required=True, choices=PHASES); p.add_argument("--root", type=int, choices=source.source.ROOTS)
    p.add_argument("--scenario", choices=source.source.SCENARIOS); p.add_argument("--replica", type=int, choices=REPLICAS)
    p.add_argument("--arm", choices=ARMS); a = p.parse_args()
    if source.source.read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract(): p.error("Changed formal closed-loop registration")
    if a.phase == "freeze": freeze(a.run_name)
    else:
        if not source.source.read(directory(a.run_name)/"freeze/result.json")["software_qualified"]: p.error("Qualify freeze first")
        if a.phase == "report": report(a.run_name)
        else:
            if None in (a.root, a.scenario, a.replica, a.arm): p.error("Complete scene/controller key required")
            source.run_pilot(a.run_name, a.root, a.scenario, a.arm, protocol=context(a.run_name, a.replica))


if __name__ == "__main__": main()
