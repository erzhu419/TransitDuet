#!/usr/bin/env python3
"""Fresh-prefix confirmation of the already selected Stage297 global cutoff."""

import argparse
from functools import partial
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_paired_fleet_stage296 as parent
from scripts import run_native_transit_service_protected_stage297 as selection
from freq_hrl.experiments.pointmaze_root_response import write_json

PROTOCOL = "native_transit_service_confirmation_stage298_v1"
SOURCE = "native_transit_service_protected_stage297_20261019_r1"
PHASES = ("freeze", "pilot", "pilot-report", "capture", "menus", "seal", "evaluate", "report")
ROOTS, SCENARIOS, PROBES = parent.ROOTS, parent.SCENARIOS, parent.PROBES
BANKS = {k: parent.BANKS[k] for k in ("proposal", "screen", "evaluation")}
ARMS = ("raw", "stage247", "zero_abstain", "legacy_risk", "joint_budgeted", "service_rank_matched", "matched_thin")
PREFIXES_PER_GROUP = 40
directory, prefix_path, sample_path, receipt_path = parent.directory, parent.prefix_path, parent.sample_path, parent.receipt_path
scene_seed, future_seed, read = parent.scene_seed, parent.future_seed, parent.read


def keys(role="test"):
    return [(role, root, scenario, replica) for root in ROOTS for scenario in SCENARIOS
        for replica in (range(12, 12+PREFIXES_PER_GROUP) if role == "test" else (1,))]


def contract(run):
    return {"source_selection": SOURCE, "locked_policy": read(directory(run)/"inputs/locked_policy.json"),
        "model_source": selection.PARENT+"/fit_raw/model_before_calibration.json",
        "legacy_adjustment_source": selection.PARENT+"/fit_raw/calibrated_model.json",
        "cohort": "400_new_test_prefixes_replicas12to51; 10_excluded_pilots_replica1; same_Stage296_seed_formula",
        "cohort_expansion": "Stage297_reserved120_expanded_before_any_new_capture_for_prefix_precision; policy_unchanged",
        "banks": {b: list(d) for b, d in BANKS.items()}, "arms": list(ARMS),
        "proposal": "unchanged_P8_wait_constrained_proposal_S8_zero_abstention",
        "action": "frozen_Stage297_fleet_critic_plus_S8_service_score_le_locked_cutoff; retain_base_or_raw_only",
        "risk_budgeted_control": "Stage297_selected_full_base; represented_by_zero_abstain",
        "matching": "expected_uniform_thinning_per_profile_root_regime_probe_over40_test_prefixes",
        "service_control": "lowest_S8_service_scores_at_same_joint_action_count_per_profile_root_regime_probe; frozen_cohort_budget_control_not_online_policy",
        "seal": "all_actions_weights_and_sparse_union_before_E32; no_new_calibration",
        "inference": "average16profiles_and6probes; 400prefix_stratified_bootstrap5000_seed298; paired_MC95; both_E16",
        "primary": "cost_wait_both_CIs_upper_lt0_both_E16_negative; service_wait_both_CIs_negative; wait_gain_retention_ge50pct_pooled_and_both_E16",
        "secondary": "cost_vs_matched_thin_service_rank_and_base_both_CIs_negative_separate_claims",
        "budget": {"main_captures": 400, "menus": 4800, "menu_episodes": 268800,
            "pilot_episodes": 430, "max_E32_episodes": 537600},
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0,
        "Stage296_test_prefixes_reused": False, "Stage296_E32_labels_used": False}


def protocol(run):
    return SimpleNamespace(PROBES=PROBES, BANKS=BANKS, keys=keys, contract=partial(contract, run),
        prefix_path=prefix_path, sample_path=sample_path, scene_seed=scene_seed, future_seed=future_seed)


def freeze(run):
    locked = read(directory(run)/"inputs/locked_policy.json")
    if locked["joint"]["family"] != "joint" or locked["risk"]["family"] != "base":
        raise ValueError("Stage298 confirms the observed Stage297 joint and full-base risk selections")
    old = parent.keys()+parent.keys("pilot")
    fresh = keys()+keys("pilot")
    old_scenes, fresh_scenes = ({scene_seed(k) for k in cohort} for cohort in (old, fresh))
    old_futures, fresh_futures = ({future_seed(k, p, d) for k in cohort for p in PROBES for d in range(64)} for cohort in (old, fresh))
    if (len(fresh_scenes) != 410 or fresh_scenes & old_scenes or fresh_futures & old_futures
            or len(fresh_futures) != 410*6*64 or max(fresh_futures) >= 2**31):
        raise ValueError("Fresh confirmation seed namespaces overlap or are incomplete")
    write_json(directory(run)/"freeze/result.json", {"contract": contract(run), "software_qualified": True,
        "new_test_prefixes": 400, "excluded_pilots": 10, "seeds_disjoint_from_Stage296": True})


def _seal_proposal(run, rows, raw, old, proposal, first, screen):
    _, keep = parent.native.rule.screen(proposal, raw, screen)
    model = read(parent.directory(selection.PARENT)/"fit_raw/model_before_calibration.json")
    offset = read(parent.directory(selection.PARENT)/"fit_raw/calibrated_model.json")["calibration"]["paired_risk"]["upward_adjustment"]
    x = parent.learning.features(rows, raw, proposal, first, screen)
    fleet = parent.learning.predict(model, x)
    service = parent.native.rule.paired_values(screen, proposal, raw)[..., 2].mean(axis=2)
    threshold = read(directory(run)/"inputs/locked_policy.json")["joint"]["threshold"]
    joint = keep & (fleet+service <= threshold)
    cube = parent.native.cube(rows)
    probability, counts = parent.native.rule.thin_probabilities(cube, keep, joint)
    service_control = matched_service_choices(raw, proposal, keep, joint, service, cube)
    choices = np.stack((raw, old, np.where(keep, proposal, raw),
        np.where(keep & (fleet+offset <= 0), proposal, raw), np.where(joint, proposal, raw), service_control, raw), axis=1)
    return {"arms": list(ARMS), "choices": choices.tolist(), "proposal": proposal.tolist(),
        "matched_thin_probabilities": probability.tolist(), "matching": counts}


def matched_service_choices(raw, proposal, eligible, joint, score, cube):
    selected = raw.copy()
    for p in range(len(raw)):
        for group in cube:
            for indices in group.T:
                pool = indices[eligible[p, indices]]
                count = int(joint[p, indices].sum())
                chosen = pool[np.argsort(score[p, pool], kind="stable")[:count]]
                selected[p, chosen] = proposal[p, chosen]
    return selected


def pilot(run, key):
    ctx = protocol(run)
    capture = parent.capture(run, key, protocol=ctx)
    probe = PROBES[(ROOTS.index(key[1])*len(SCENARIOS)+SCENARIOS.index(key[2])) % len(PROBES)]
    panels, episodes, identities, rss = {}, 1, 0, 0.
    for bank in BANKS:
        records, receipt = parent.rollout(run, key, probe, BANKS[bank][:2], list(range(6)),
            sample_path(run, key, probe, bank), protocol=ctx)
        panels[bank] = np.asarray([[[r["values"][str(a)] for a in range(6)] for r in records]])
        episodes += receipt["native_episodes"]; identities += receipt["exact_identities"]
        rss = max(rss, receipt["peak_rss_MiB"])
    rows = [r for r in read(prefix_path(run, key)/"rows.json") if r["probe_index"] == probe]
    saved = read(prefix_path(run, key)/"controls.json")
    slot = PROBES.index(probe)
    raw, old, logits = (np.asarray(saved[k])[:, [slot]] for k in ("raw", "stage247", "logits"))
    proposal = parent.native.proposals(rows, raw, logits, panels["proposal"])
    sealed = _seal_proposal(run, rows, raw, old, proposal, panels["proposal"], panels["screen"])
    choices = np.asarray(sealed["choices"])
    if not np.all((choices[:, 4] == raw) | (choices[:, 4] == choices[:, 2])):
        raise ValueError("Frozen joint rule changed an action instead of retain/fallback")
    return {"contract": contract(run), "software_qualified": True, "key": list(key), "probe": probe,
        "native_episodes": episodes, "exact_identities": identities, "captured_states": capture["states"],
        "max_peak_rss_MiB": rss, "pilot_labels_excluded": True}


def pilot_report(run):
    receipts = [read(receipt_path(run, "pilot", k)/"receipt.json") for k in keys("pilot")]
    if any(r["contract"] != contract(run) or not r["software_qualified"] or r["native_episodes"] != 43
            or r["exact_identities"] != 6 or not r["pilot_labels_excluded"] for r in receipts):
        raise ValueError("Fresh-cohort native pilots did not qualify")
    write_json(directory(run)/"pilot-report/result.json", {"contract": contract(run), "software_qualified": True,
        "native_episodes": 430, "exact_identities": 60,
        "max_peak_rss_MiB": max(r["max_peak_rss_MiB"] for r in receipts), "pilot_labels_excluded": True})


def seal(run):
    rows = [r for key in keys() for r in read(prefix_path(run, key)/"rows.json")]
    controls = [read(prefix_path(run, key)/"controls.json") for key in keys()]
    raw, old, logits = (np.concatenate([np.asarray(c[name]) for c in controls], axis=1) for name in ("raw", "stage247", "logits"))
    first, _ = parent.load_panel(run, "test", "proposal", protocol=protocol(run))
    proposal = parent.native.proposals(rows, raw, logits, first)
    screen, _ = parent.load_panel(run, "test", "screen", protocol=protocol(run))
    sealed = _seal_proposal(run, rows, raw, old, proposal, first, screen)
    unions = parent.learning.action_unions(sealed, [r["original_choice"] for r in rows])
    sealed["states"] = [{"key": ["test", r["root"], r["scenario"], r["replica"]],
        "probe": r["probe_index"], "candidates": unions[i]} for i, r in enumerate(rows)]
    write_json(directory(run)/"sealed_raw/test.json", sealed)
    active = [i for i, u in enumerate(unions) if u]
    write_json(directory(run)/"seal/result.json", {"contract": contract(run), "software_qualified": True,
        "actions_and_weights_frozen": True, "E32_labels_read": False, "active_indices": active,
        "evaluation_tasks": 4*len(active), "evaluation_native_episodes": 32*sum(1+len(u) for u in unions if u),
        "changed_profile_states": {a: int((np.asarray(sealed["choices"])[:, i] != raw).sum()) for i, a in enumerate(ARMS[:-1])}})


def report(run):
    plan, sealed = read(directory(run)/"seal/result.json"), read(directory(run)/"sealed_raw/test.json")
    rows = [r for key in keys() for r in read(prefix_path(run, key)/"rows.json")]
    panels, episodes, seeds = np.zeros((len(rows), 32, 6, 4)), 0, set()
    for i in plan["active_indices"]:
        state = sealed["states"][i]
        for chunk in range(4):
            path = directory(run)/f"evaluation/state{i}/chunk{chunk}"
            receipt, labels = read(path/"receipt.json"), read(path/"labels.json")
            draws = list(BANKS["evaluation"][chunk*8:chunk*8+8])
            if (receipt["contract"] != contract(run) or not receipt["software_qualified"]
                    or receipt["key"] != state["key"] or receipt["probe"] != state["probe"]
                    or receipt["candidates"] != state["candidates"] or receipt["draws"] != draws
                    or receipt["state_futures"] != 8 or receipt["exact_identities"] != 8
                    or receipt["native_episodes"] != 8*(1+len(state["candidates"]))
                    or [r["draw"] for r in labels] != draws):
                raise ValueError("Incomplete Stage298 frozen evaluation")
            episodes += receipt["native_episodes"]
            for r in labels:
                if r["future_seed"] != future_seed(tuple(state["key"]), state["probe"], r["draw"]) or r["future_seed"] in seeds:
                    raise ValueError("Wrong or repeated fresh evaluation future")
                seeds.add(r["future_seed"])
                if sorted(map(int, r["values"])) != state["candidates"]: raise ValueError("Missing frozen action")
                values = np.asarray([r["values"][str(a)] for a in state["candidates"]])
                if (values.shape != (len(state["candidates"]), 4) or not np.isfinite(values).all()
                        or np.any(values[state["candidates"].index(rows[i]["original_choice"])] != 0)
                        or np.max(abs(values[:, 0]-values[:, 1]-values[:, 2])) > 1e-12):
                    raise ValueError("Invalid fresh paired outcome")
                for a, value in r["values"].items(): panels[i, r["draw"]-32, int(a)] = value
    if episodes != plan["evaluation_native_episodes"] or len(seeds) != len(plan["active_indices"])*32:
        raise ValueError("Evaluation differs from the sealed physical budget")
    result = parent.learning.summarize(sealed, panels, parent.native.cube(rows), seed=298)
    pair = result["comparisons"]["joint_budgeted_vs_zero_abstain"]
    result["cost_improvement_vs_base_supported"] = all(pair[c]["cost"][1] < 0 for c in ("prefix_CI95", "conditional_MC_CI95"))
    service_pair = result["comparisons"]["joint_budgeted_vs_service_rank_matched"]
    result["cost_information_vs_service_rank_supported"] = all(service_pair[c]["cost"][1] < 0 for c in ("prefix_CI95", "conditional_MC_CI95"))
    write_json(directory(run)/"report/result.json", {"protocol": PROTOCOL, "contract": contract(run),
        "coverage_qualified": True, "test_prefixes": 400, "states": len(rows), "fixed_profiles": 16,
        "evaluation_native_episodes": episodes, "state_futures": len(seeds), **result,
        "controller_promoted": False, "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True); p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--role", choices=("test", "pilot")); p.add_argument("--root", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS); p.add_argument("--replica", type=int)
    p.add_argument("--probe", type=int, choices=PROBES); p.add_argument("--bank", choices=("proposal", "screen"))
    p.add_argument("--state-index", type=int); p.add_argument("--chunk", type=int, default=0)
    a = p.parse_args(); ctx = protocol(a.run_name)
    if read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract(a.run_name): p.error("Changed fresh-trial registration")
    if a.phase != "freeze" and not read(directory(a.run_name)/"freeze/result.json")["software_qualified"]: p.error("Freeze first")
    if a.phase not in ("freeze", "pilot", "pilot-report") and not read(directory(a.run_name)/"pilot-report/result.json")["software_qualified"]: p.error("Native pilot qualification first")
    key = (a.role, a.root, a.scenario, a.replica)
    if a.phase in ("pilot", "capture", "menus"):
        if key not in keys(a.role) or (a.phase == "pilot") != (a.role == "pilot"): p.error("Unregistered fresh scene")
        if a.phase == "pilot": receipt = pilot(a.run_name, key)
        elif a.phase == "capture": receipt = parent.capture(a.run_name, key, protocol=ctx)
        else:
            if a.bank not in ("proposal", "screen") or a.probe is None or a.chunk != 0: p.error("Unregistered menu")
            _, receipt = parent.rollout(a.run_name, key, a.probe, BANKS[a.bank], list(range(6)),
                sample_path(a.run_name, key, a.probe, a.bank), protocol=ctx)
        write_json(receipt_path(a.run_name, a.phase, key, a.probe, a.bank)/"receipt.json", receipt)
    elif a.phase == "evaluate":
        plan = read(directory(a.run_name)/"seal/result.json")
        if not plan["actions_and_weights_frozen"] or a.state_index not in plan["active_indices"] or a.chunk not in range(4): p.error("Unregistered frozen evaluation")
        state = read(directory(a.run_name)/"sealed_raw/test.json")["states"][a.state_index]
        _, receipt = parent.rollout(a.run_name, tuple(state["key"]), state["probe"], BANKS["evaluation"][a.chunk*8:a.chunk*8+8],
            state["candidates"], directory(a.run_name)/f"evaluation/state{a.state_index}/chunk{a.chunk}", protocol=ctx)
        write_json(receipt_path(a.run_name, "evaluate", index=a.state_index, chunk=a.chunk)/"receipt.json", receipt)
    else: {"freeze": freeze, "pilot-report": pilot_report, "seal": seal, "report": report}[a.phase](a.run_name)


if __name__ == "__main__": main()
