#!/usr/bin/env python3
"""Fresh fit/calibration/test scenes for a frozen-proposal paired fleet critic."""

import argparse
import copy
from functools import partial
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_abstention_transfer_stage294 as native
from freq_hrl.domains.transit import native_paired_fleet_calibration as learning
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_paired_fleet_stage296_v1"
ROOTS, SCENARIOS, PROBES = native.ROOTS, native.SCENARIOS, native.PROBES
ROLES = ("fit", "calibration", "test", "pilot")
COUNTS = {"fit": 8, "calibration": 6, "test": 12, "pilot": 1}
BANKS = {"proposal": tuple(range(8)), "screen": tuple(range(8, 16)),
    "target": tuple(range(16, 32)), "evaluation": tuple(range(32, 64))}
PHASES = ("freeze", "pilot", "pilot-report", "capture", "menus", "targets", "fit", "evaluate", "report")
read = native.read


def contract():
    return {"models_source": native.CACHE_RUN+"/freeze_raw/models.json",
        "cohort": "fresh80_fit60_calibration120_test_prefixes_balanced_over10_root_regimes_all6probes; separate10_software_pilots",
        "roles": COUNTS, "banks": {k: list(v) for k, v in BANKS.items()},
        "scene_seed": "2070000000+role_index*500000+root_index*100000+regime_index*1000+replica",
        "future_seed": "2100000000+role_index*10000000+root_index*1000000+regime_index*100000+replica*1000+probe_ordinal*100+draw",
        "proposal": "unchanged_P8_wait_constrained_proposal_and_S8_zero_event_abstention; no_service_fit_or_action_reranking",
        "risk_fit": "one26feature_standardized_ridge1_unpenalized_intercept_on_fit_role_independent_T16_paired_fleet_mean; changed_proposals_only; duplicates_are_profile_weights",
        "features": list(learning.FEATURE_NAMES),
        "calibration": "independent60prefix_T16_selected_population_ratio_bias_plus1.6448536269514722_stratified_prefix_cluster_SE; upward_adjustment_floored0; no_grid",
        "arms": list(learning.ARMS), "bias_control": "same_calibration_recipe_applied_to_unlearned_S8_fleet_mean",
        "action": "retain_base_zero_abstain_proposal_iff_adjusted_risk_score_le0_else_raw; no_action_search",
        "matching": "frozen_expected_uniform_k_of_n_thinning_of_base_zero_abstain_changes_per_profile_root_regime_probe_over12_test_prefixes",
        "freeze": "fit_model_before_calibration_T16_read; calibrated_model_and_all_test_actions_weights_before_any_test_E32; no_Stage294_E32_access",
        "inference": "average16profiles_and6probes_before120prefix_stratified_bootstrap5000_seed296; shared_future_MC_normal95; both_E16_halves",
        "primary": "paired_risk_vs_raw_cost_wait_both_CI_upper_lt0_both_halves_negative; service_and_wait_both_CI_upper_lt0; retain_at_least_half_zero_abstain_wait_gain_in_all32_and_both16",
        "secondary": "fleet_vs_zero_abstain_and_cost_vs_matched_thin_both_CIs_upper_lt0_separate_claims",
        "budget": {"main_captures": 260, "menu_tasks": 3120, "menu_native_episodes": 174720,
            "target_tasks": 1680, "target_native_episodes": 94080, "pilot_native_episodes": 570,
            "max_evaluation_native_episodes": 161280, "risk_fits": 1, "raw_reference_fits": 0, "actor_updates": 0},
        "boundary": "single_intervention_fixed_lower_finite_MC_proposal_with_learned_risk; score_adjustment_not_actionwise_safety_certificate; not_end_to_end_HRL"}


def directory(run): return ROOT/"results"/run


def keys(role=None):
    return [(r, root, scenario, replica) for r in ((role,) if role else ROLES[:-1])
        for root in ROOTS for scenario in SCENARIOS for replica in range(COUNTS[r])]


def scene_seed(key):
    role, root, scenario, replica = key
    return 2070000000+ROLES.index(role)*500000+ROOTS.index(root)*100000+SCENARIOS.index(scenario)*1000+replica


def future_seed(key, probe, draw):
    role, root, scenario, replica = key
    return 2100000000+ROLES.index(role)*10000000+ROOTS.index(root)*1000000+SCENARIOS.index(scenario)*100000+replica*1000+PROBES.index(probe)*100+draw


def prefix_path(run, key): return directory(run)/"prefixes"/Path(*map(str, key))


def sample_path(run, key, probe, bank, chunk=0):
    return prefix_path(run, key)/f"samples/probe{probe}/{bank}/chunk{chunk}"


def receipt_path(run, phase, key=None, probe=None, bank=None, chunk=0, index=None):
    path = directory(run)/"receipts"/phase
    if key is not None: path /= Path(*map(str, key))
    if probe is not None: path /= f"probe{probe}/{bank}/chunk{chunk}"
    if index is not None: path /= f"state{index}/chunk{chunk}"
    return path


def freeze(run):
    rows = [r for root in ROOTS for scenario in SCENARIOS for replica in range(6)
        for r in read(native.directory(native.CACHE_RUN)/f"prefixes/train/{root}/{scenario}/{replica}/rows.json")]
    raw, old, _ = native.controls(rows)
    with np.load(native.directory(native.CACHE_RUN)/"fit_raw/train_predictions.npz") as saved:
        if not np.all(saved["chosen"][:, :, :2] == np.stack((raw, old), axis=1)[:, None]):
            raise ValueError("Frozen raw/Stage247 inference changed")
    scenes = [scene_seed(k) for r in ROLES for k in keys(r)]
    futures = [future_seed(k, p, d) for r in ROLES for k in keys(r) for p in PROBES for d in range(64)]
    if len(set(scenes)) != 270 or len(set(futures)) != len(futures) or max(futures) >= 2**31:
        raise ValueError("Changed fresh role/seed namespaces")
    write_json(directory(run)/"freeze/result.json", {"contract": contract(), "software_qualified": True,
        "old_control_states_replayed": len(rows), "fixed_profiles": len(raw),
        "Stage294_outcomes_read": False, "new_fits": 0, "actor_updates": 0})


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
    source, model, checkpoint = native.native_inputs(key[1])
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
    raw, old, logits = native.controls(rows)
    write_json(path/"controls.json", {"raw": raw.tolist(), "stage247": old.tolist(), "logits": logits.tolist()})
    with (path/"prefix_refs.pkl").open("wb") as f:
        pickle.dump({"prefix": plan.prefix, "snapshots": plan.snapshots}, f, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {"contract": contract(), "software_qualified": True, "key": list(key),
        "native_episodes": 1, "states": len(rows), "fixed_profiles": len(raw)}
    write_json(path/"receipt.json", receipt)
    return receipt


def rollout(run, key, probe, draws, candidates, output):
    from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
    import torch
    source, model, checkpoint = native.native_inputs(key[1])
    with (prefix_path(run, key)/"prefix_refs.pkl").open("rb") as f: refs = pickle.load(f)
    row = next(r for r in read(prefix_path(run, key)/"rows.json") if r["probe_index"] == probe)
    records, start = [], time.monotonic()
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
        print(f"PAIRED_FLEET_DRAW_QUALIFIED {key} probe={probe} draw={draw}", flush=True)
    write_json(output/"labels.json", records)
    receipt = {"contract": contract(), "software_qualified": True, "key": list(key), "probe": probe,
        "draws": list(draws), "candidates": list(candidates), "state_futures": len(draws),
        "native_episodes": len(draws)*(1+len(candidates)), "exact_identities": len(draws),
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.}
    write_json(output/"receipt.json", receipt)
    return records, receipt


def pilot(run, key):
    captured = capture(run, key)
    probe = PROBES[(ROOTS.index(key[1])*len(SCENARIOS)+SCENARIOS.index(key[2])) % len(PROBES)]
    panels, episodes, identities, max_rss = {}, 1, 0, 0.
    for bank in BANKS:
        labels, receipt = rollout(run, key, probe, BANKS[bank][:2], list(range(6)), sample_path(run, key, probe, bank))
        panels[bank] = np.asarray([[[r["values"][str(a)] for a in range(6)] for r in labels]])
        episodes += receipt["native_episodes"]; identities += receipt["exact_identities"]
        max_rss = max(max_rss, receipt["peak_rss_MiB"])
    rows = [r for r in read(prefix_path(run, key)/"rows.json") if r["probe_index"] == probe]
    saved = read(prefix_path(run, key)/"controls.json")
    raw = np.asarray(saved["raw"])[:, [PROBES.index(probe)]]
    proposal = (raw+1) % 6
    x = learning.features(rows, raw, proposal, panels["proposal"], panels["screen"])
    target = native.rule.paired_values(panels["target"], proposal, raw)[..., 1].mean(axis=2)
    model = learning.fit(x, target, np.ones(raw.shape, bool))
    return {"contract": contract(), "software_qualified": True, "key": list(key), "probe": probe,
        "native_episodes": episodes, "exact_identities": identities, "captured_states": captured["states"],
        "feature_count": x.shape[-1], "software_fit_normal_error": model["normal_equation_error"],
        "max_peak_rss_MiB": max_rss, "pilot_used_for_scientific_fit": False}


def pilot_report(run):
    receipts = [read(receipt_path(run, "pilot", k)/"receipt.json") for k in keys("pilot")]
    if any(r["contract"] != contract() or not r["software_qualified"] or r["native_episodes"] != 57
            or r["exact_identities"] != 8 or r["feature_count"] != 26 for r in receipts):
        raise ValueError("Native paired-risk preflight did not qualify")
    write_json(directory(run)/"pilot-report/result.json", {"contract": contract(), "software_qualified": True,
        "native_episodes": sum(r["native_episodes"] for r in receipts),
        "exact_identities": sum(r["exact_identities"] for r in receipts),
        "max_peak_rss_MiB": max(r["max_peak_rss_MiB"] for r in receipts), "pilot_labels_excluded": True})


def load_rows(run, role):
    return [r for k in keys(role) for r in read(prefix_path(run, k)/"rows.json")]


def load_controls(run, role):
    saved = [read(prefix_path(run, k)/"controls.json") for k in keys(role)]
    return tuple(np.concatenate([np.asarray(s[name]) for s in saved], axis=1) for name in ("raw", "stage247", "logits"))


def load_panel(run, role, bank):
    if role == "test" and bank == "target": raise ValueError("Test targets are not part of this experiment")
    result, episodes, seeds = [], 0, set()
    for key in keys(role):
        rows = read(prefix_path(run, key)/"rows.json")
        for probe in PROBES:
            labels = []
            for chunk in range(len(BANKS[bank])//8):
                path = sample_path(run, key, probe, bank, chunk)
                receipt = read(path/"receipt.json")
                draws = BANKS[bank][chunk*8:chunk*8+8]
                if (receipt["contract"] != contract() or not receipt["software_qualified"]
                        or receipt["key"] != list(key) or receipt["probe"] != probe
                        or receipt["candidates"] != list(range(6)) or receipt["draws"] != list(draws)
                        or receipt["native_episodes"] != 56 or receipt["exact_identities"] != 8 or receipt["state_futures"] != 8):
                    raise ValueError("Incomplete paired-risk sampling receipt")
                episodes += receipt["native_episodes"]
                labels.extend(read(path/"labels.json"))
            if [r["draw"] for r in labels] != list(BANKS[bank]): raise ValueError("Wrong paired-risk draws")
            original = next(r["original_choice"] for r in rows if r["probe_index"] == probe)
            for r in labels:
                if r["future_seed"] != future_seed(key, probe, r["draw"]) or r["future_seed"] in seeds:
                    raise ValueError("Wrong or repeated paired-risk future")
                seeds.add(r["future_seed"])
                if sorted(map(int, r["values"])) != list(range(6)): raise ValueError("Incomplete paired-risk action menu")
            values = np.asarray([[r["values"][str(a)] for a in range(6)] for r in labels])
            if (values.shape != (len(BANKS[bank]), 6, 4) or not np.isfinite(values).all()
                    or np.any(values[:, original] != 0) or np.max(abs(values[..., 0]-values[..., 1]-values[..., 2])) > 1e-12):
                raise ValueError("Invalid paired-risk values, identity or cost components")
            result.append(values)
    return np.asarray(result), {"native_episodes": episodes, "state_futures": len(seeds)}


def inputs(run, role):
    rows, (raw, old, logits) = load_rows(run, role), load_controls(run, role)
    first, _ = load_panel(run, role, "proposal")
    proposal = native.proposals(rows, raw, logits, first)
    write_json(directory(run)/f"fit_raw/{role}_proposal.json", {"choices": proposal.tolist()})
    second, _ = load_panel(run, role, "screen")
    x = learning.features(rows, raw, proposal, first, second)
    _, eligible = native.rule.screen(proposal, raw, second)
    return rows, raw, old, proposal, second, x, eligible


def fit(run):
    rows, raw, _, proposal, _, x, _ = inputs(run, "fit")
    target, _ = load_panel(run, "fit", "target")
    y = native.rule.paired_values(target, proposal, raw)[..., 1].mean(axis=2)
    model = learning.fit(x, y, proposal != raw)
    train_mse = float(np.mean((learning.predict(model, x)[proposal != raw]-y[proposal != raw])**2))
    write_json(directory(run)/"fit_raw/model_before_calibration.json", model)
    print("PAIRED_FLEET_MODEL_FROZEN_BEFORE_CALIBRATION", flush=True)
    cal_rows, cal_raw, _, cal_proposal, cal_screen, cx, eligible = inputs(run, "calibration")
    target, _ = load_panel(run, "calibration", "target")
    cy = native.rule.paired_values(target, cal_proposal, cal_raw)[..., 1].mean(axis=2)
    score = learning.predict(model, cx)
    bias_score = native.rule.paired_values(cal_screen, cal_proposal, cal_raw)[..., 1].mean(axis=2)
    calibration = {"paired_risk": learning.calibrate(score, cy, eligible, native.cube(cal_rows)),
        "bias_calibrated": learning.calibrate(bias_score, cy, eligible, native.cube(cal_rows))}
    package = {"model": model, "calibration": calibration}
    write_json(directory(run)/"fit_raw/calibrated_model.json", package)
    rows, raw, old, proposal, screen, x, _ = inputs(run, "test")
    risk = learning.predict(model, x)+calibration["paired_risk"]["upward_adjustment"]
    bias = native.rule.paired_values(screen, proposal, raw)[..., 1].mean(axis=2)+calibration["bias_calibrated"]["upward_adjustment"]
    sealed = learning.seal(raw, old, proposal, screen, risk, bias, native.cube(rows))
    sealed["adjusted_risk_scores"] = risk.tolist()
    unions = learning.action_unions(sealed, [r["original_choice"] for r in rows])
    sealed["states"] = [{"key": ["test", r["root"], r["scenario"], r["replica"]],
        "probe": r["probe_index"], "candidates": unions[i]} for i, r in enumerate(rows)]
    write_json(directory(run)/"sealed_raw/test.json", sealed)
    active = [i for i, u in enumerate(unions) if u]
    write_json(directory(run)/"fit/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "risk_model_and_choices_frozen": True, "risk_fits": 1,
        "fit_prefixes": 80, "calibration_prefixes": 60, "test_prefixes": 120,
        "test_E32_labels_read": False, "Stage294_E32_labels_read": False,
        "normal_equation_error": model["normal_equation_error"], "calibration": calibration,
        "risk_error_diagnostics": {"fit_T16_in_sample_MSE": train_mse,
            "independent_calibration_T16_precalibration_MSE": float(np.mean((score[eligible]-cy[eligible])**2)),
            "independent_calibration_T16_postcalibration_MSE": float(np.mean((score[eligible]+calibration["paired_risk"]["upward_adjustment"]-cy[eligible])**2)),
            "postcalibration_error_is_not_an_independent_validation": True},
        "active_indices": active, "evaluation_tasks": len(active)*4,
        "evaluation_native_episodes": 32*sum(1+len(u) for u in unions if u), "matching": sealed["matching"],
        "raw_reference_fits": 0, "actor_updates": 0})


def report(run):
    plan, sealed = read(directory(run)/"fit/result.json"), read(directory(run)/"sealed_raw/test.json")
    rows, panels, episodes, seeds = load_rows(run, "test"), np.zeros((720, 32, 6, 4)), 0, set()
    for i in plan["active_indices"]:
        state = sealed["states"][i]
        for chunk in range(4):
            path = directory(run)/f"evaluation/state{i}/chunk{chunk}"
            receipt, labels = read(path/"receipt.json"), read(path/"labels.json")
            draws = BANKS["evaluation"][chunk*8:chunk*8+8]
            if (receipt["contract"] != contract() or not receipt["software_qualified"]
                    or receipt["key"] != state["key"] or receipt["probe"] != state["probe"]
                    or receipt["candidates"] != state["candidates"] or receipt["draws"] != list(draws)
                    or receipt["state_futures"] != 8 or receipt["exact_identities"] != 8
                    or receipt["native_episodes"] != 8*(1+len(state["candidates"]))
                    or [r["draw"] for r in labels] != list(draws)):
                raise ValueError("Incomplete frozen Stage296 evaluation")
            episodes += receipt["native_episodes"]
            for r in labels:
                if r["future_seed"] != future_seed(tuple(state["key"]), state["probe"], r["draw"]) or r["future_seed"] in seeds:
                    raise ValueError("Wrong or repeated Stage296 evaluation future")
                seeds.add(r["future_seed"])
                if sorted(map(int, r["values"])) != state["candidates"]: raise ValueError("Missing frozen evaluation action")
                values = np.asarray([r["values"][str(a)] for a in state["candidates"]])
                if (values.shape != (len(state["candidates"]), 4) or not np.isfinite(values).all()
                        or np.any(values[state["candidates"].index(rows[i]["original_choice"])] != 0)
                        or np.max(abs(values[:, 0]-values[:, 1]-values[:, 2])) > 1e-12):
                    raise ValueError("Invalid Stage296 evaluation values or identity")
                for a, value in r["values"].items(): panels[i, r["draw"]-32, int(a)] = value
    if episodes != plan["evaluation_native_episodes"] or len(seeds) != len(plan["active_indices"])*32:
        raise ValueError("Stage296 evaluation differs from frozen physical budget")
    choices = np.asarray(sealed["choices"])
    eligible = choices[:, 2] != choices[:, 0]
    paired_fleet = native.rule.paired_values(panels, np.asarray(sealed["proposal"]), choices[:, 0])[..., 1]
    score = np.asarray(sealed["adjusted_risk_scores"])
    risk_error = {"base_eligible_profile_state_examples": int(eligible.sum()),
        "test_E32_MSE_at_base_eligible_proposals": float(np.mean((score[eligible]-paired_fleet.mean(axis=2)[eligible])**2)) if np.any(eligible) else None,
        "test_E32_label_mean_variance_at_base_eligible_proposals": float(np.mean(paired_fleet.var(axis=2, ddof=1)[eligible]/32)) if np.any(eligible) else None}
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "test_prefixes": 120, "states": 720, "fixed_profiles": 16,
        "evaluation_native_episodes": episodes, "state_futures": len(seeds),
        "risk_error_diagnostics": risk_error,
        "matching": sealed["matching"], **learning.summarize(sealed, panels, native.cube(rows)),
        "Stage294_E32_labels_read": False, "raw_reference_fits": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True); p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--role", choices=ROLES); p.add_argument("--root", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS); p.add_argument("--replica", type=int)
    p.add_argument("--probe", type=int, choices=PROBES); p.add_argument("--bank", choices=("proposal", "screen", "target"))
    p.add_argument("--chunk", type=int, default=0); p.add_argument("--state-index", type=int)
    a = p.parse_args()
    if read(directory(a.run_name)/"inputs/preregistration.json")["contract"] != contract(): p.error("Changed Stage296 registration")
    if a.phase != "freeze" and not read(directory(a.run_name)/"freeze/result.json")["software_qualified"]: p.error("Freeze first")
    if a.phase not in ("freeze", "pilot", "pilot-report") and not read(directory(a.run_name)/"pilot-report/result.json")["software_qualified"]: p.error("Native preflight first")
    key = (a.role, a.root, a.scenario, a.replica)
    if a.phase in ("pilot", "capture", "menus", "targets"):
        if key not in keys(a.role) or (a.phase == "pilot") != (a.role == "pilot"): p.error("Unregistered scene role")
        if a.phase == "pilot": receipt = pilot(a.run_name, key)
        elif a.phase == "capture": receipt = capture(a.run_name, key)
        else:
            bank = a.bank if a.phase == "menus" else "target"
            if (a.phase == "menus" and (bank not in ("proposal", "screen") or a.chunk != 0)
                    or a.phase == "targets" and (a.role not in ("fit", "calibration") or a.chunk not in (0, 1))): p.error("Unregistered bank/chunk")
            _, receipt = rollout(a.run_name, key, a.probe, BANKS[bank][a.chunk*8:a.chunk*8+8],
                list(range(6)), sample_path(a.run_name, key, a.probe, bank, a.chunk))
        write_json(receipt_path(a.run_name, a.phase, key, a.probe, a.bank, a.chunk)/"receipt.json", receipt)
    elif a.phase == "evaluate":
        plan = read(directory(a.run_name)/"fit/result.json")
        if a.state_index not in plan["active_indices"] or a.chunk not in range(4): p.error("Unregistered evaluation")
        state = read(directory(a.run_name)/"sealed_raw/test.json")["states"][a.state_index]
        _, receipt = rollout(a.run_name, tuple(state["key"]), state["probe"], BANKS["evaluation"][a.chunk*8:a.chunk*8+8],
            state["candidates"], directory(a.run_name)/f"evaluation/state{a.state_index}/chunk{a.chunk}")
        write_json(receipt_path(a.run_name, a.phase, index=a.state_index, chunk=a.chunk)/"receipt.json", receipt)
    else: {"freeze": freeze, "pilot-report": pilot_report, "fit": fit, "report": report}[a.phase](a.run_name)
    print("PAIRED_FLEET_PHASE_COMPLETE "+a.phase, flush=True)


if __name__ == "__main__": main()
