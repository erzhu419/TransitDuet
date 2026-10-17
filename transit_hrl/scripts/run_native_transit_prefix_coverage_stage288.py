#!/usr/bin/env python3
"""Fresh training-prefix coverage with fixed Stage247 critic geometry."""

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
from scripts import run_native_transit_frozen_prefix_stage287 as native
from freq_hrl.domains.transit import native_prefix_coverage as learning
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_prefix_coverage_stage288_v1"
SOURCE_RUN = "native_transit_frozen_prefix_stage287_20261017_r1"
ROOTS, SCENARIOS, PROBES = native.ROOTS, native.SCENARIOS, native.PROBES
ROLES = ("train", "held", "pilot")
PHASES = ("freeze", "capture-pilot", "pilot-collect", "pilot-report", "capture", "collect", "fit", "report")
read = native.read


def contract():
    return {"geometry_source": native.LEGACY_RUN,
        "raw_models_source": SOURCE_RUN+"/freeze_raw/models.json_only_no_confirmation_rows_or_labels",
        "arms": list(learning.VARIANTS), "rotations": 6, "fixed_profiles": 16,
        "dense": "per_rotation10_prefixes_replica_equal_rotation_each6probes_60states",
        "wide": "per_rotation60_prefixes_oneprobe_PROBES[(replica+rotation)%6]_60states",
        "balance": "botharms_same_root_regime_probe_counts_and_same360states_once_across6rotations",
        "training": "60fresh_prefixes_6_per_root_regime_all6probes_16fit_plus8independent_error_futures",
        "held": "40fresh_prefixes_4_per_root_regime_all6probes_16futures_shared_by_all_models",
        "pilot": "10separate_prefixes_oneprobe_covering_all6probe_indices_two_futures_software_only",
        "scene_seed": "1858000000+role_index*1000000+root_index*100000+regime_index*1000+replica",
        "future_seed": "1958000000+role_index*1000000+root_index*100000+regime_index*10000+replica*1000+probe_index*100+draw",
        "fit": "existing_state_GLS_head_sum_only_assigned16futures_precision_shrinkage_recipe_unchanged",
        "frozen": "old_training_TSVD_whitening_scale_selected_rank_alpha_A_RMS_ball_cached_head_difference_raw_gate_anchor_and_lower",
        "zero_profiles": "retain_original_zero_selections_no_fits_no_new_selection",
        "calibration": "precision_and_shrinkage_reestimated_from_each_assigned_training_set_only_coverage_effect_includes_group_calibration",
        "validation": "save_all_models_and_unlabeled_train_held_predictions_before_report_opens_held_labels",
        "inference": "average_fixed_profiles_rotations_and6probes_before_40prefix_stratified_bootstrap5000_seed288_CI95",
        "error": "paired_feasible_margin_MSE_cost_fleet_service_train_fit16_train_independent8_held16_MC_variance_reported_separately",
        "coverage_hypothesis": "wide_minus_dense_held_fleet_margin_MSE_CI_upper_below0",
        "control_advancement": "wide_minus_dense_raw_and_unchanged247_cost_and_wait_CI_upper_below0_no_model_selection",
        "budget": {"main_prefixes": 100, "main_states": 600, "main_chunks": 3120,
            "main_continuation_episodes": 87360, "pilot_episodes": 150, "residual_fits": 96,
            "new_raw_reference_fits": 0, "actor_updates": 0},
        "boundary": "fixed_representation_single_intervention_coverage_test_not_end_to_end_HRL_or_training_seed_CI"}


def directory(run):
    return ROOT/"results"/run


def keys(role=None):
    return [(r, root, scenario, i) for r in ((role,) if role else ("train", "held"))
        for root in ROOTS for scenario in SCENARIOS
        for i in range({"train": 6, "held": 4, "pilot": 1}[r])]


def scene_seed(key):
    role, root, scenario, replica = key
    return 1858000000+ROLES.index(role)*1000000+ROOTS.index(root)*100000+SCENARIOS.index(scenario)*1000+replica


def future_seed(key, probe, draw):
    role, root, scenario, replica = key
    return (1958000000+ROLES.index(role)*1000000+ROOTS.index(root)*100000
        +SCENARIOS.index(scenario)*10000+replica*1000+PROBES.index(probe)*100+draw)


def prefix_path(run, key):
    return directory(run)/"prefixes"/Path(*map(str, key))


def chunks(role):
    return range(6 if role == "train" else 4)


def pilot_probe(key):
    return PROBES[(ROOTS.index(key[1])*len(SCENARIOS)+SCENARIOS.index(key[2])) % 6]


def continuation_path(run, key, probe=None, chunk=None):
    return prefix_path(run, key)/("pilot" if key[0] == "pilot" else f"continuations/probe{probe}/chunk{chunk}")


def freeze(run):
    parent = read(native.directory(SOURCE_RUN)/"freeze/result.json")
    if not parent["software_qualified"] or parent["contract"] != native.contract():
        raise ValueError("Frozen source models did not qualify")
    packages = read(native.directory(SOURCE_RUN)/"freeze_raw/models.json")
    bank, _, registered, _ = native.legacy.training.cohort.source.source.load_data()
    views = native.level.cache.cached.training.cohort.source.matched_views(bank["training_data"], registered)
    models = []
    for package in packages:
        key = tuple(package["key"])
        rows, _ = native.legacy.training.cohort.split_rows(views[key[0]], key[1])
        data = learning.base.base.causal_data(rows, package["reference"])
        ledger = read(native.legacy.raw_directory(native.legacy.result_path(native.legacy.SOURCE_RUN, *key))/"fits.json")["full"]
        selected = package["stage247_selection"]
        level = learning.base.strength.CANDIDATES[selected]["level"] if selected else "zero"
        radius = next(f["radius"] for f in ledger["fits"] if f["objective"] == "pair_gls" and f["level"] == level) if selected else 0.
        frozen = learning.geometry(rows, data, package["stage247_model"], selected, radius)
        models.append({"key": list(key), "reference": package["reference"], **frozen})
    reserved = read(native.directory(native.coverage.noise.COLLECTION_RUN)/"preregistration.json")["reserved_scene_seeds"]
    old = {native.scene_seed(*k) for k in native.prefixes()+native.prefixes(pilot=True)}
    old.update(native.source.scene_seed(r, s, i) for r in ROOTS for s in SCENARIOS for i in range(8))
    scenes = {scene_seed(k) for role in ROLES for k in keys(role)}
    if len(models) != 16 or len(scenes) != 110 or scenes & (old|set(reserved)):
        raise ValueError("Changed fixed profiles or fresh scene coverage")
    write_json(directory(run)/"freeze_raw/models.json", models)
    write_json(directory(run)/"freeze/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "fixed_profiles": len(models), "active_profiles": sum(m["selection"] is not None for m in models),
        "max_reconstruction_error": max(m.get("reconstruction_error", 0.) for m in models),
        "new_fits": 0, "native_episodes": 0})


def capture(run, key):
    if not read(directory(run)/"freeze/result.json")["software_qualified"]:
        raise ValueError("Freeze before capture")
    role, root, scenario, replica = key
    path = prefix_path(run, key)
    model, checkpoint = native.native_inputs(root)
    observer, plan = native.source.shared.episode(root, scenario, scene_seed(key), path/"raw/observer",
        checkpoint, model, PROBES[-1], plan_class=native.NativeFrozenPrefixAudit)
    native.source.qualify_observer(observer)
    rows = [{**r, "replica": replica} for r in native.causal_rows(root, scenario, scene_seed(key), plan)]
    write_json(path/"rows.json", rows)
    with (path/"prefix_refs.pkl").open("wb") as f:
        pickle.dump({"prefix": plan.prefix, "snapshots": plan.snapshots}, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(path/"receipt.json", {"software_qualified": True, "contract": contract(), "key": list(key), "native_episodes": 1})


def collect(run, key, probe=None, chunk=None):
    if key[0] != "pilot" and not read(directory(run)/"pilot-report/result.json")["software_qualified"]:
        raise ValueError("Pilot must qualify before main continuations")
    start = time.monotonic()
    path = prefix_path(run, key)
    rows = read(path/"rows.json")
    with (path/"prefix_refs.pkl").open("rb") as f:
        references = pickle.load(f)
    if key[0] == "pilot":
        probe, draws = pilot_probe(key), range(2)
    else:
        draws = range(4*chunk, 4*chunk+4)
    row = next(r for r in rows if r["probe_index"] == probe)
    output = continuation_path(run, key, probe, chunk)
    model, checkpoint = native.native_inputs(key[1])
    records = []
    for draw in draws:
        reference = SimpleNamespace(prefix=references["prefix"], snapshot=references["snapshots"][probe])
        seed = future_seed(key, probe, draw)
        audit = partial(native.NativeContinuationValueAudit, continuation_seed=seed)
        raw = output/f"raw/draw{draw}"
        native.restore_caller_torch_rng(reference.snapshot)
        baseline, plan = native.source.shared.episode(key[1], key[2], scene_seed(key), raw/"observer", checkpoint,
            model, probe, reference=reference, plan_class=audit)
        native.source.qualify_observer(baseline)
        branches, values = [], {}
        for candidate in [row["original_choice"], *[c for c in range(6) if c != row["original_choice"]]]:
            native.restore_caller_torch_rng(plan.snapshot)
            branch, branch_plan = native.source.shared.episode(key[1], key[2], scene_seed(key), raw/f"candidate{candidate}",
                checkpoint, model, probe, candidate=candidate, reference=plan, plan_class=audit)
            native.source.shared.qualify(branch, baseline, candidate=candidate)
            if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
                raise ValueError("Branch did not consume its registered shared future")
            branch["policy_trace"] = native.source.shared.policy_trace(branch.pop("supply"))
            branches.append(branch)
            values[str(candidate)] = native.paired_values(branch, baseline)
        baseline["policy_trace"] = native.source.shared.policy_trace(baseline["supply"])
        native.source.validate_continuations(baseline, sorted(branches, key=lambda b: b["probe"]["selected"]), candidates=list(range(6)))
        write_json(raw/"paired_rollouts.json", {"observer": baseline, "branches": branches})
        records.append({"draw": draw, "future_seed": seed, "values": values})
        print(f"PREFIX_COVERAGE_DRAW_QUALIFIED {key} probe={probe} draw={draw}", flush=True)
    write_json(output/"labels.json", records)
    write_json(output/"receipt.json", {"software_qualified": True, "contract": contract(), "native_episodes": len(records)*7,
        "state_futures": len(records), "elapsed_s": time.monotonic()-start,
        "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.})


def load_rows(run, role):
    return [r for key in keys(role) for r in read(prefix_path(run, key)/"rows.json")]


def load_panels(run, role):
    values, receipts = [], []
    for key in keys(role):
        for probe in ((pilot_probe(key),) if role == "pilot" else PROBES):
            labels = {}
            for chunk in ((None,) if role == "pilot" else chunks(role)):
                path = continuation_path(run, key, probe, chunk)
                receipt = read(path/"receipt.json")
                if not receipt["software_qualified"] or receipt["contract"] != contract():
                    raise ValueError("Incomplete native training/held collection")
                receipts.append(receipt)
                labels.update({r["draw"]: [r["values"][str(a)] for a in range(6)] for r in read(path/"labels.json")})
            values.append([labels[d] for d in range({"train": 24, "held": 16, "pilot": 2}[role])])
    return np.asarray(values), receipts


def pilot_report(run):
    rows = [next(r for r in read(prefix_path(run, key)/"rows.json") if r["probe_index"] == pilot_probe(key)) for key in keys("pilot")]
    panels, receipts = load_panels(run, "pilot")
    ledgers = []
    for frozen in read(directory(run)/"freeze_raw/models.json"):
        data = learning.base.base.causal_data(rows, frozen["reference"])
        _, ledger = learning.fit(rows, data, panels, frozen)
        if ledger["active"]:
            ledgers.append(ledger)
    write_json(directory(run)/"pilot-report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "cross_worker_prefixes": 10, "state_futures": len(rows)*2,
        "native_episodes": sum(r["native_episodes"] for r in receipts)+10,
        "solver_fits": len(ledgers), "max_KKT": max(l["optimality"] for l in ledgers),
        "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in receipts), "outcomes_used_for_selection": False})


def fit(run):
    rows = {role: load_rows(run, role) for role in ("train", "held")}
    panels, _ = load_panels(run, "train")
    frozen_models = read(directory(run)/"freeze_raw/models.json")
    shapes = {role: (len(frozen_models), 6, len(learning.VARIANTS), len(rr)) for role, rr in rows.items()}
    arrays = {role: {"chosen": np.zeros(shape, dtype=int), "cost": np.zeros((*shape, 6)),
        "components": np.zeros((*shape, 6, 2))} for role, shape in shapes.items()}
    models, ledgers = [], []
    for pi, frozen in enumerate(frozen_models):
        data = {role: learning.base.base.causal_data(rr, frozen["reference"]) for role, rr in rows.items()}
        for rotation in range(6):
            fitted = {"stage247": frozen["model"]}
            for arm in ("dense", "wide"):
                ix = learning.assignment(rows["train"], arm, rotation)
                subset = {k: v[ix] for k, v in data["train"].items()}
                fitted[arm], ledger = learning.fit([rows["train"][i] for i in ix], subset, panels[ix, :16], frozen)
                ledgers.append({"profile": pi, "rotation": rotation, "arm": arm, **ledger})
                models.append({"profile": pi, "rotation": rotation, "arm": arm, "model": fitted[arm]})
            for role, d in data.items():
                for vi, variant in enumerate(learning.VARIANTS):
                    prediction = learning.predict(d, fitted.get(variant, frozen["model"]), None if variant == "raw" else frozen["selection"])
                    for field in arrays[role]:
                        arrays[role][field][pi, rotation, vi] = prediction[field]
        print(f"PREFIX_COVERAGE_PROFILE_FITTED {pi}", flush=True)
    # Held rows contain no labels; all fits and choices are saved before report.
    path = directory(run)/"fit_raw"
    write_json(path/"models.json", models)
    write_json(path/"ledgers.json", ledgers)
    for role, data in arrays.items():
        np.savez_compressed(path/f"{role}_predictions.npz", **data)
    write_json(directory(run)/"fit/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "active_residual_fits": sum(l["active"] for l in ledgers),
        "models_and_choices_frozen": True, "held_labels_read": False, "raw_reference_fits": 0, "actor_updates": 0})


def report(run):
    if not read(directory(run)/"fit/result.json")["models_and_choices_frozen"]:
        raise ValueError("Freeze fitted models and held choices before evaluation")
    frozen = read(directory(run)/"freeze_raw/models.json")
    rows = {r: load_rows(run, r) for r in ("train", "held")}
    panels, execution = {}, []
    for role in rows:
        panels[role], receipts = load_panels(run, role)
        execution.extend(receipts)
    predictions = {}
    for role in rows:
        with np.load(directory(run)/f"fit_raw/{role}_predictions.npz") as a:
            predictions[role] = {k: a[k].copy() for k in a.files}
    held_errors = np.zeros((len(learning.VARIANTS), len(rows["held"]), 3))
    held_noise = np.zeros((len(rows["held"]), 3))
    training = {arm: {scope: [] for scope in ("fit16", "independent8", "independent8_minus_MC_variance")} for arm in ("dense", "wide")}
    for pi, model in enumerate(frozen):
        data = {r: learning.base.base.causal_data(rr, model["reference"]) for r, rr in rows.items()}
        for rotation in range(6):
            for vi, variant in enumerate(learning.VARIANTS):
                pred = {k: v[pi, rotation, vi] for k, v in predictions["held"].items()}
                error, noise = learning.margin_errors(data["held"], panels["held"], pred)
                held_errors[vi] += error/(len(frozen)*6)
                if vi == 0:
                    held_noise += noise/(len(frozen)*6)
                if variant in training:
                    ix = learning.assignment(rows["train"], variant, rotation)
                    subset = {k: v[ix] for k, v in data["train"].items()}
                    pred = {k: v[pi, rotation, vi, ix] for k, v in predictions["train"].items()}
                    for scope, draws in (("fit16", panels["train"][ix, :16]), ("independent8", panels["train"][ix, 16:])):
                        error, noise = learning.margin_errors(subset, draws, pred)
                        training[variant][scope].append(error.mean(axis=0))
                        if scope == "independent8":
                            training[variant]["independent8_minus_MC_variance"].append((error-noise).mean(axis=0))
    effects = learning.paired_controls(predictions["held"]["chosen"], panels["held"])
    records = []
    for i, key in enumerate(keys("held")):
        indices = range(6*i, 6*i+6)
        records.append({"root": key[1], "scenario": key[2], "replica": key[3],
            "mean": effects[list(indices)].mean(axis=(0, 1)).tolist(),
            "margin_MSE": held_errors[:, list(indices)].mean(axis=1).tolist(),
            "margin_MC_variance": held_noise[list(indices)].mean(axis=0).tolist()})
    controls = learning.paired_prefix_summary(records)
    error_fields = ("cost_margin_mse", "fleet_margin_mse", "service_margin_mse")
    errors = learning.paired_prefix_summary([{**r, "mean": r["margin_MSE"]} for r in records], fields=error_fields)
    control_pass = all(controls["wide_vs_"+v]["prefix_CI95"][f][1] < 0 for v in ("dense", "raw", "stage247") for f in ("cost", "wait_min"))
    write_json(directory(run)/"report/prefixes.json", records)
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "held_prefixes": len(records), "controls": controls, "paired_margin_error": errors,
        "training_errors": {arm: {scope: dict(zip(error_fields, map(float, np.mean(values, axis=0)))) for scope, values in scopes.items()} for arm, scopes in training.items()},
        "held_errors": {v: {"margin_MSE": dict(zip(error_fields, map(float, held_errors[i].mean(axis=0)))),
            "minus_MC_variance": dict(zip(error_fields, map(float, (held_errors[i]-held_noise).mean(axis=0))))} for i, v in enumerate(learning.VARIANTS)},
        "fleet_error_coverage_supported": errors["wide_vs_dense"]["prefix_CI95"]["fleet_margin_mse"][1] < 0,
        "joint_control_advancement_passed": control_pass, "controller_promoted": False,
        "execution": {"main_chunks": len(execution), "continuation_native_episodes": sum(r["native_episodes"] for r in execution),
            "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in execution)}, "raw_reference_fits": 0, "actor_updates": 0})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", required=True, choices=PHASES)
    p.add_argument("--role", choices=ROLES)
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int)
    p.add_argument("--probe", type=int, choices=PROBES)
    p.add_argument("--chunk", type=int)
    a = p.parse_args()
    if a.phase in ("freeze", "pilot-report", "fit", "report"):
        {"freeze": freeze, "pilot-report": pilot_report, "fit": fit, "report": report}[a.phase](a.run_name)
    else:
        key = a.role, a.seed, a.scenario, a.replica
        if key not in keys("pilot")+keys() or (a.phase in ("capture-pilot", "pilot-collect")) != (a.role == "pilot"):
            p.error("Unregistered prefix/phase")
        if a.phase in ("capture", "capture-pilot"):
            capture(a.run_name, key)
        else:
            if a.phase == "collect" and (a.probe not in PROBES or a.chunk not in chunks(a.role)):
                p.error("Unregistered continuation")
            collect(a.run_name, key, a.probe, a.chunk)
    print("PREFIX_COVERAGE_PHASE_COMPLETE "+a.phase, flush=True)


if __name__ == "__main__":
    main()
