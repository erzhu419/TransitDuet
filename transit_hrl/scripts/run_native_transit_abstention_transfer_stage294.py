#!/usr/bin/env python3
"""Prospective new-prefix transfer of the frozen Stage291 abstention rule."""

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit import native_abstention_transfer as rule
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_abstention_transfer_stage294_v1"
CACHE_RUN = "native_transit_prefix_coverage_stage288_20261017_r1"
GATE_RUN = "native_transit_fleet_screen_stage290_20261017_r1"
RULE_RUN = "native_transit_zero_abstention_stage291_20261017_r1"
MODEL_RUN = "native_transit_value_context_stage177_frozen_20261011_r1"
ROOTS = (397, 401)
SCENARIOS = ("low_noise", "high_noise", "hour_burst", "persistent_shift", "ood_period")
PROBES = (8, 9, 22, 23, 36, 37)
PHASES = ("freeze", "pilot", "pilot-report", "capture", "menus", "seal", "evaluate", "report")
BANKS = {"proposal": tuple(range(8)), "screen": tuple(range(8, 16)), "evaluation": tuple(range(16, 48))}


def read(path):
    return json.loads(path.read_text())


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"models_source": CACHE_RUN+"/freeze_raw/models.json", "rule_source": RULE_RUN,
        "arms": list(rule.ARMS), "roots": list(ROOTS), "regimes": list(SCENARIOS), "probes": list(PROBES),
        "cohort": "240_genuinely_new_prefixes_24_per_root_regime_all6probes_16_fixed_historical_profiles_not_seeds",
        "proposal": "unchanged_Stage289_P8_min_cost_wait_delta_le0_raw_feasibility_native_fallback_raw_then_original_then_index_ties",
        "screen": "unchanged_Stage291_S8_paired_fleet_mean_le0_and_not_all8_zero_else_raw_no_reselection",
        "matched_control": "exact_expected_uniform_k_of_n_accepted_changes_per_profile_root_regime_probe_over24prefixes; frozen_before_evaluation",
        "banks": {k: list(v) for k, v in BANKS.items()},
        "scene_seed": "2000000000+role_index*500000+root_index*100000+regime_index*1000+replica",
        "future_seed": "2050000000+role_index*10000000+root_index*1000000+regime_index*100000+replica*1000+probe_ordinal*100+draw",
        "role_order": ["new", "pilot"], "freeze": "exact_old360_state_raw_247_proposal_witness_and_matching_replay_before_native_capture; new_P8_proposals_saved_before_S8_read; all_choices_weights_saved_before_E32",
        "simulation": "P8_S8_all6actions; E32_only_frozen_within_profile_disagreements_union_plus_native_identity; archived_prefix_state_rng_shared_future_tape",
        "inference": "average16profiles_then6probes; 240prefix_stratified_bootstrap5000_seed294_frozen_matching_weights; separate_shared_future_MC_normal95",
        "primary": "zero_abstain_vs_raw_cost_wait_both_prefix_and_conditional_MC_CI95_upper_lt0_and_both16future_halves_negative",
        "information": "zero_abstain_vs_matched_thin_cost_and_fleet_both_CI95_upper_lt0_separate_secondary_claims",
        "pilot": "10_separate_prefixes_one_probe_each_P2_S2_E2_all6actions_software_only_no_outcome_selection",
        "budget": {"new_main_prefixes": 240, "main_states": 1440, "menu_tasks": 2880,
            "menu_native_episodes": 161280, "evaluation_futures_per_active_state": 32,
            "max_evaluation_native_episodes": 322560, "pilot_native_episodes": 430,
            "new_fits": 0, "actor_updates": 0},
        "boundary": "finite_MC_rule_transfer_single_intervention_fixed_lower_not_learned_deployable_policy_or_end_to_end_HRL; all5regimes_retained_no_controller_promotion"}


def keys(role="new"):
    return [(role, root, scenario, replica) for root in ROOTS for scenario in SCENARIOS
        for replica in range(24 if role == "new" else 1)]


def scene_seed(key):
    role, root, scenario, replica = key
    return 2000000000+("new", "pilot").index(role)*500000+ROOTS.index(root)*100000+SCENARIOS.index(scenario)*1000+replica


def future_seed(key, probe, draw):
    role, root, scenario, replica = key
    return (2050000000+("new", "pilot").index(role)*10000000+ROOTS.index(root)*1000000
        +SCENARIOS.index(scenario)*100000+replica*1000+PROBES.index(probe)*100+draw)


def prefix_path(run, key):
    return directory(run)/"prefixes"/Path(*map(str, key))


def cube(rows):
    groups = list(dict.fromkeys((r["root"], r["scenario"]) for r in rows))
    replicas = list(dict.fromkeys(r["replica"] for r in rows))
    probes = list(dict.fromkeys(r["probe_index"] for r in rows))
    lookup = {(r["root"], r["scenario"], r["replica"], r["probe_index"]): i for i, r in enumerate(rows)}
    result = np.array([[[lookup[(root, scenario, replica, probe)] for probe in probes]
        for replica in replicas] for root, scenario in groups])
    if result.size != len(rows):
        raise ValueError("Incomplete or duplicate prefix/probe coverage")
    return result


def controls(rows):
    from freq_hrl.domains.transit import native_variance_critic as critic
    batch = critic.joint.binding.learned.neural.batches(rows)
    raw, old, logits = [], [], []
    for frozen in read(directory(CACHE_RUN)/"freeze_raw/models.json"):
        data = critic.causal_data(rows, frozen["reference"], batch=batch)
        raw.append(data["raw_choices"])
        old.append(critic.candidates(data, frozen["model"])[frozen["selection"] or "zero"]["chosen"])
        logits.append(data["raw_logits"])
    return np.asarray(raw), np.asarray(old), np.asarray(logits)


def proposals(rows, raw, logits, panels):
    originals = np.array([r["original_choice"] for r in rows])
    return np.stack([rule.propose(l, originals, r, panels) for r, l in zip(raw, logits)])


def freeze(run):
    rows = [r for root in ROOTS for scenario in SCENARIOS for replica in range(6)
        for r in read(directory(CACHE_RUN)/f"prefixes/train/{root}/{scenario}/{replica}/rows.json")]
    raw, old, logits = controls(rows)
    with np.load(directory(CACHE_RUN)/"fit_raw/train_predictions.npz") as saved:
        cached = saved["chosen"][:, :, :2]
        if cached.shape != (16, 6, 2, 360) or not np.all(cached == np.stack((raw, old), axis=1)[:, None]):
            raise ValueError("Frozen raw or Stage247 inference changed")
    def bank(chunks):
        values = []
        for root in ROOTS:
            for scenario in SCENARIOS:
                for replica in range(6):
                    for probe in PROBES:
                        path = directory(CACHE_RUN)/f"prefixes/train/{root}/{scenario}/{replica}/continuations/probe{probe}"
                        labels = [r for chunk in chunks for r in read(path/f"chunk{chunk}/labels.json")]
                        values.append([[r["values"][str(a)] for a in range(6)] for r in labels])
        return np.asarray(values)
    proposal = proposals(rows, raw, logits, bank((0, 1)))
    legacy = np.asarray(read(directory(GATE_RUN)/"report_raw/primary.json")["choices"])
    if not np.array_equal(proposal, legacy[:, 2]):
        raise ValueError("Frozen P8 proposal replay differs")
    sealed = rule.seal(raw, old, proposal, bank((2, 3)), cube(rows))
    witness = read(directory(RULE_RUN)/"report_raw/primary.json")
    if (not np.array_equal(sealed["choices"], np.asarray(witness["choices"])[:, [0, 1, 3, 4]])
            or sealed["matched_thin_probabilities"] != witness["matched_thin_probabilities"]
            or sealed["matching"] != witness["matching"]):
        raise ValueError("Frozen S8 abstention or expected thinning replay differs")
    scenes = {scene_seed(k) for role in ("new", "pilot") for k in keys(role)}
    reserved = set(read(directory("native_transit_value_expansion_stage187_development_20261012_r2")/"preregistration.json")["reserved_scene_seeds"])
    reserved.update(r["scene_seed"] for r in rows)
    reserved.update(1858000000+role*1000000+ri*100000+si*1000+replica
        for role, n in enumerate((6, 4, 1)) for ri in range(2) for si in range(5) for replica in range(n))
    reserved.update(1687000000+ri*100000+si*1000+replica
        for ri in range(2) for si in range(5) for replica in range(9))
    if len(scenes) != 250 or scenes & reserved:
        raise ValueError("New scenes overlap reserved seeds")
    seeds = [future_seed(k, p, d) for role in ("new", "pilot") for k in keys(role)
        for p in PROBES for d in range(48)]
    if len(seeds) != len(set(seeds)) or max(seeds) >= 2**31:
        raise ValueError("Continuation seed namespace collides or overflows")
    write_json(directory(run)/"freeze/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "cached_states_replayed": 360, "fixed_profiles": len(raw),
        "cached_controls_proposals_abstention_and_matching_exact": True, "scene_seeds": sorted(scenes),
        "new_fits": 0, "actor_updates": 0, "old_confirmation_labels_read": False})


def native_inputs(root):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from scripts import run_native_transit_value_expansion_stage187 as source
    sys.path.insert(0, str(source.native.NATIVE))
    import frequency
    import torch
    frequency.DemandFrequencyTracker = source.native.NativeRoutingTracker
    torch.set_num_threads(1)
    if tuple(source.ROOTS) != ROOTS or tuple(source.SCENARIOS) != SCENARIOS:
        raise ValueError("Changed native root/regime order")
    model = read(directory(MODEL_RUN)/"inputs/models.json")[str(root)]
    source.shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = source.native.load_source(root)
    return source, model, checkpoint


def capture(run, key):
    from freq_hrl.domains.transit.native_candidate_value import causal_snapshot
    from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit, qualify_timetable
    class CaptureAudit(NativeTimetableContextAudit):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.snapshots = {}

        def select_candidate(self, candidates):
            index = len(self.decisions)
            selected = super().select_candidate(candidates)
            if index in PROBES:
                self.snapshots[index] = self.snapshot if index == self.probe_index else causal_snapshot(self.env)
            return selected
    source, model, checkpoint = native_inputs(key[1])
    path = prefix_path(run, key)
    observer, plan = source.shared.episode(key[1], key[2], scene_seed(key), path/"raw/observer",
        checkpoint, model, PROBES[-1], plan_class=CaptureAudit)
    source.qualify_observer(observer)
    rows = []
    for probe in PROBES:
        c = next(c for c in plan.contexts if c["index"] == probe)
        row = {"root": key[1], "scenario": key[2], "replica": key[3], "scene_seed": scene_seed(key), "probe_index": probe,
            **{k: copy.deepcopy(c[k]) for k in ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")},
            "original_choice": c["greedy"], "context": source.labels.training_context(plan.snapshots[probe]),
            "occupancy_context": {k: copy.deepcopy(c[k]) for k in source.context_source.OCCUPANCY_KEYS},
            "timetable_context": copy.deepcopy(c["timetable_context"])}
        qualify_timetable(c, row)
        rows.append(row)
    write_json(path/"rows.json", rows)
    raw, old, logits = controls(rows)
    write_json(path/"controls.json", {"raw": raw.tolist(), "stage247": old.tolist(), "logits": logits.tolist()})
    with (path/"prefix_refs.pkl").open("wb") as f:
        pickle.dump({"prefix": plan.prefix, "snapshots": plan.snapshots}, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(path/"receipt.json", {"contract": contract(), "key": list(key), "software_qualified": True,
        "scene_seed": scene_seed(key), "native_episodes": 1, "states": len(rows), "fixed_profiles": len(raw)})


def rollout(run, key, probe, draws, candidates, output):
    from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
    import torch
    source, model, checkpoint = native_inputs(key[1])
    with (prefix_path(run, key)/"prefix_refs.pkl").open("rb") as f:
        refs = pickle.load(f)
    row = next(r for r in read(prefix_path(run, key)/"rows.json") if r["probe_index"] == probe)
    records = []
    start = time.monotonic()
    for draw in draws:
        reference = SimpleNamespace(prefix=refs["prefix"], snapshot=refs["snapshots"][probe])
        seed = future_seed(key, probe, draw)
        audit = partial(NativeContinuationValueAudit, continuation_seed=seed)
        raw = output/f"raw/draw{draw}"
        torch.set_rng_state(torch.tensor(reference.snapshot["rng"]["torch"], dtype=torch.uint8))
        baseline, plan = source.shared.episode(key[1], key[2], scene_seed(key), raw/"observer", checkpoint,
            model, probe, reference=reference, plan_class=audit)
        source.qualify_observer(baseline)
        branches, values = [], {}
        order = [row["original_choice"], *[c for c in candidates if c != row["original_choice"]]]
        for candidate in order:
            torch.set_rng_state(torch.tensor(plan.snapshot["rng"]["torch"], dtype=torch.uint8))
            branch, branch_plan = source.shared.episode(key[1], key[2], scene_seed(key), raw/f"candidate{candidate}",
                checkpoint, model, probe, candidate=candidate, reference=plan, plan_class=audit)
            source.shared.qualify(branch, baseline, candidate=candidate)
            if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
                raise ValueError("Branch did not consume the registered shared future")
            cost = branch["service_cost_restricted"]-baseline["service_cost_restricted"]
            fleet = max(branch["peak_fleet"]-12, 0)**2/12-max(baseline["peak_fleet"]-12, 0)**2/12
            wait = branch["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"]
            values[str(candidate)] = [cost, fleet, cost-fleet, wait]
            branch["policy_trace"] = source.shared.policy_trace(branch.pop("supply"))
            branches.append(branch)
        baseline["policy_trace"] = source.shared.policy_trace(baseline["supply"])
        source.validate_continuations(baseline, sorted(branches, key=lambda b: b["probe"]["selected"]), candidates=candidates)
        write_json(raw/"paired_rollouts.json", {"observer": baseline, "branches": branches})
        records.append({"draw": draw, "future_seed": seed, "values": values})
        print(f"TRANSFER_DRAW_QUALIFIED {key} probe={probe} draw={draw}", flush=True)
    write_json(output/"labels.json", records)
    receipt = {"contract": contract(), "software_qualified": True, "key": list(key), "probe": probe,
        "draws": list(draws), "candidates": list(candidates), "state_futures": len(draws),
        "native_episodes": len(draws)*(1+len(candidates)), "exact_identities": len(draws),
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.}
    write_json(output/"receipt.json", receipt)
    return records, receipt


def pilot(run, key):
    capture(run, key)
    ordinal = (ROOTS.index(key[1])*len(SCENARIOS)+SCENARIOS.index(key[2])) % 6
    probe = PROBES[ordinal]
    path = prefix_path(run, key)
    rows = [read(path/"rows.json")[ordinal]]
    c = read(path/"controls.json")
    raw, old, logits = (np.asarray(c[k])[:, ordinal:ordinal+1] for k in ("raw", "stage247", "logits"))
    banks, receipts = [], []
    for name in BANKS:
        records, receipt = rollout(run, key, probe, BANKS[name][:2], list(range(6)), path/f"pilot/{name}")
        banks.append(np.array([[[r["values"][str(a)] for a in range(6)] for r in records]]))
        receipts.append(receipt)
    proposal = proposals(rows, raw, logits, banks[0])
    sealed = rule.seal(raw, old, proposal, banks[1], np.array([[[0]]]))
    rule.summarize(sealed, banks[2], np.array([[[0]]]))
    write_json(path/"pilot/receipt.json", {"contract": contract(), "software_qualified": True,
        "key": list(key), "probe": probe, "native_episodes": 1+sum(r["native_episodes"] for r in receipts),
        "exact_identities": sum(r["exact_identities"] for r in receipts), "frozen_profiles": len(raw),
        "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in receipts), "outcomes_used_for_selection": False})


def pilot_report(run):
    receipts = [read(prefix_path(run, k)/"pilot/receipt.json") for k in keys("pilot")]
    if any(r["contract"] != contract() or not r["software_qualified"] or r["exact_identities"] != 6
            or r["frozen_profiles"] != 16 or r["outcomes_used_for_selection"] for r in receipts):
        raise ValueError("Native transfer preflight failed")
    write_json(directory(run)/"pilot-report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "root_regime_groups": len(receipts),
        "native_episodes": sum(r["native_episodes"] for r in receipts),
        "max_peak_rss_MiB": max(r["max_peak_rss_MiB"] for r in receipts), "outcomes_used_for_selection": False})


def menu_path(run, key, probe, bank):
    return prefix_path(run, key)/f"menus/probe{probe}/{bank}"


def load_bank(run, bank):
    output = []
    for key in keys():
        for probe in PROBES:
            path = menu_path(run, key, probe, bank)
            receipt, labels = read(path/"receipt.json"), read(path/"labels.json")
            if (receipt["contract"] != contract() or not receipt["software_qualified"]
                    or receipt["draws"] != list(BANKS[bank]) or receipt["candidates"] != list(range(6))
                    or [r["draw"] for r in labels] != list(BANKS[bank])
                    or any(r["future_seed"] != future_seed(key, probe, r["draw"]) for r in labels)):
                raise ValueError("Incomplete or changed proposal/screen bank")
            output.append([[r["values"][str(a)] for a in range(6)] for r in labels])
    return np.asarray(output)


def seal(run):
    rows = [r for key in keys() for r in read(prefix_path(run, key)/"rows.json")]
    saved = [read(prefix_path(run, k)/"controls.json") for k in keys()]
    raw, old, logits = (np.concatenate([np.asarray(c[k]) for c in saved], axis=1) for k in ("raw", "stage247", "logits"))
    proposal = proposals(rows, raw, logits, load_bank(run, "proposal"))
    write_json(directory(run)/"sealed_raw/proposal.json", {"choices": proposal.tolist()})
    print("TRANSFER_P8_PROPOSALS_FROZEN", flush=True)
    sealed = rule.seal(raw, old, proposal, load_bank(run, "screen"), cube(rows))
    unions = rule.action_unions(sealed, [r["original_choice"] for r in rows])
    sealed["states"] = [{"key": ["new", r["root"], r["scenario"], r["replica"]],
        "probe": r["probe_index"], "candidates": unions[i]} for i, r in enumerate(rows)]
    write_json(directory(run)/"sealed_raw/primary.json", sealed)
    active = [i for i, u in enumerate(unions) if u]
    write_json(directory(run)/"seal/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "prefixes": 240, "states": len(rows), "active_indices": active,
        "evaluation_tasks": len(active)*4, "evaluation_state_futures": len(active)*32,
        "evaluation_native_episodes": 32*sum(1+len(u) for u in unions if u), "matching": sealed["matching"],
        "evaluation_labels_read": False, "new_fits": 0, "actor_updates": 0})


def evaluation_path(run, index, chunk):
    return directory(run)/f"evaluation/state{index}/chunk{chunk}"


def report(run):
    sealed, plan = read(directory(run)/"sealed_raw/primary.json"), read(directory(run)/"seal/result.json")
    panels = np.zeros((1440, 32, 6, 4))
    episodes = 0
    for i in plan["active_indices"]:
        state = sealed["states"][i]
        for chunk in range(4):
            path = evaluation_path(run, i, chunk)
            receipt, labels = read(path/"receipt.json"), read(path/"labels.json")
            draws = BANKS["evaluation"][chunk*8:chunk*8+8]
            if (receipt["contract"] != contract() or not receipt["software_qualified"]
                    or receipt["key"] != state["key"] or receipt["probe"] != state["probe"]
                    or receipt["candidates"] != state["candidates"] or receipt["draws"] != list(draws)
                    or [r["draw"] for r in labels] != list(draws)):
                raise ValueError("Incomplete frozen evaluation")
            episodes += receipt["native_episodes"]
            for r in labels:
                if r["future_seed"] != future_seed(tuple(state["key"]), state["probe"], r["draw"]):
                    raise ValueError("Wrong evaluation future")
                if sorted(map(int, r["values"])) != state["candidates"]:
                    raise ValueError("Missing frozen action in evaluation")
                for action, value in r["values"].items():
                    panels[i, r["draw"]-16, int(action)] = value
    if episodes != plan["evaluation_native_episodes"]:
        raise ValueError("Evaluation physical budget differs from sealed plan")
    rows = [r for k in keys() for r in read(prefix_path(run, k)/"rows.json")]
    result = rule.summarize(sealed, panels, cube(rows))
    result["descriptive_group_means"] = {f"{root}/{scenario}": dict(zip(rule.ARMS,
        [dict(zip(rule.FIELDS, values)) for values in group])) for (root, scenario), group
        in zip(((r, s) for r in ROOTS for s in SCENARIOS), result["descriptive_group_means"])}
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "new_prefixes": 240, "states": 1440, "fixed_profiles": 16,
        "evaluation_native_episodes": episodes, "matching": sealed["matching"], **result,
        "new_fits": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--role", choices=("new", "pilot"), default="new")
    p.add_argument("--root", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int)
    p.add_argument("--probe", type=int, choices=PROBES)
    p.add_argument("--bank", choices=("proposal", "screen"))
    p.add_argument("--state-index", type=int)
    p.add_argument("--chunk", type=int, choices=range(4))
    a = p.parse_args()
    if read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract():
        p.error("Changed prospective transfer contract")
    if a.phase != "freeze" and not read(directory(a.run_name)/"freeze/result.json")["software_qualified"]:
        p.error("Cached inference bridge must pass first")
    if a.phase in ("capture", "menus", "seal", "evaluate", "report") and not read(directory(a.run_name)/"pilot-report/result.json")["software_qualified"]:
        p.error("Native pilot must pass first")
    key = (a.role, a.root, a.scenario, a.replica)
    if a.phase in ("pilot", "capture", "menus") and (key not in keys(a.role) or (a.phase == "pilot") != (a.role == "pilot")):
        p.error("Unregistered prefix")
    if a.phase == "pilot":
        pilot(a.run_name, key)
    elif a.phase == "capture":
        capture(a.run_name, key)
    elif a.phase == "menus":
        rollout(a.run_name, key, a.probe, BANKS[a.bank], list(range(6)), menu_path(a.run_name, key, a.probe, a.bank))
    elif a.phase == "evaluate":
        plan = read(directory(a.run_name)/"seal/result.json")
        if a.state_index not in plan["active_indices"] or a.chunk is None:
            p.error("Unregistered frozen evaluation task")
        state = read(directory(a.run_name)/"sealed_raw/primary.json")["states"][a.state_index]
        rollout(a.run_name, tuple(state["key"]), state["probe"], BANKS["evaluation"][a.chunk*8:a.chunk*8+8],
            state["candidates"], evaluation_path(a.run_name, a.state_index, a.chunk))
    else:
        {"freeze": freeze, "pilot-report": pilot_report, "seal": seal, "report": report}[a.phase](a.run_name)
    print("ABSTENTION_TRANSFER_PHASE_COMPLETE "+a.phase, flush=True)


if __name__ == "__main__":
    main()
