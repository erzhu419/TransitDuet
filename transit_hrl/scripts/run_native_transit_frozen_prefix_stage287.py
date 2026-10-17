#!/usr/bin/env python3
"""Freeze five existing selectors, then evaluate fresh causal prefix scenes."""

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
from scripts import run_native_transit_service_level_stage285 as level
from scripts import run_native_transit_balanced_critic_stage247 as legacy
from scripts import run_native_transit_mean_coverage_stage195 as coverage
from freq_hrl.domains.transit import native_frozen_prefix as learning
from freq_hrl.domains.transit.native_candidate_value import causal_snapshot
from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit, qualify_timetable
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_frozen_prefix_stage287_v1"
LEVEL_RUN = "native_transit_service_level_stage285_20261017_r1"
LEGACY_RUN = "native_transit_balanced_critic_stage247_development_20261015_r1"
KEYS = tuple(k for k in level.KEYS if k[0].startswith("mixed"))
source = coverage.noise.source
ROOTS, SCENARIOS, PROBES = source.ROOTS, source.SCENARIOS, source.PROBES


def contract():
    return {"level_run": LEVEL_RUN, "legacy_run": LEGACY_RUN,
        "variants": list(learning.VARIANTS), "primary": [list(p) for p in learning.PRIMARY],
        "frozen_profiles": "16_mixed_historical_cells_x2_training_banks_equal_weight_not_independent_seeds",
        "baseline": "unchanged_Stage247_state_gls_pooled_safe_including_both_head_and_scalar_paths",
        "new_heads": "uniform_action_level_observed_level_only_raw_risk_gate_and_reference_frozen_no_contrast",
        "bridge": "10_old_replica1_prefixes_one_per_root_regime_32_fresh_futures_draw1_to32",
        "main": "80_new_prefixes_8_per_root_regime_64_fresh_futures_draw0_to63",
        "pilot": "10_bridge_and10_separate_new_replica8_prefixes_one_probe_draw0_software_only_no_outcome_selection",
        "new_scene_seed": "1687000000+root_index*100000+regime_index*1000+replica",
        "future_seed": "1787000000+role_index*1000000+root_index*100000+regime_index*10000+replica*1000+probe_ordinal*100+draw",
        "probes": list(PROBES), "roots": list(ROOTS), "regimes": list(SCENARIOS),
        "freeze": "exact_cached_19200_choices_replayed_before_new_scene_capture_all_choices_saved_before_future_rollouts",
        "intervention": "one_action_then_unchanged_native_greedy_suffix_fixed_lower_weights_61380_ticks",
        "simulation": "only_within_profile_policy_disagreements_union_plus_native_identity_shared_future_tape_exact_prefix_state_rng",
        "analysis": "mean_fixed_profiles_and6_probes_then_prefix; new_stratified_prefix_bootstrap5000_seed287_95percent; bridge_conditional_MC_only; paired_MC_covariance_separate",
        "success": "both_primary_comparisons_cost_and_wait_prefix_CI_upper_below0_no_candidate_selection",
        "fits": 0, "actor_updates": 0, "storage": "snapshots_models_choices_and_rollouts_server_only_small_receipts_and_scalar_report_pull",
        "boundary": "single_intervention_frozen_lower_new_scene_validation_not_training_seed_CI_or_end_to_end_HRL"}


def read(path):
    return json.loads(path.read_text())


def directory(run):
    return ROOT/"results"/run


def scene_seed(role, root, scenario, replica):
    if role == "bridge":
        return source.scene_seed(root, scenario, 1)
    return 1687000000+ROOTS.index(root)*100000+SCENARIOS.index(scenario)*1000+replica


def future_seed(role, root, scenario, replica, probe, draw):
    return (1787000000+("bridge", "new", "pilot_new").index(role)*1000000
        +ROOTS.index(root)*100000+SCENARIOS.index(scenario)*10000+replica*1000+PROBES.index(probe)*100+draw)


def prefixes(*, pilot=False):
    return [(role, root, scenario, replica) for role in (("bridge", "pilot_new") if pilot else ("bridge", "new"))
        for root in ROOTS for scenario in SCENARIOS
        for replica in ((1,) if role == "bridge" else (8,) if role == "pilot_new" else range(8))]


def prefix_path(run, key):
    role, root, scenario, replica = key
    return directory(run)/"prefixes"/role/str(root)/scenario/str(replica)


def freeze(run):
    data, _, registered, _ = level.cache.cached.training.cohort.source.source.load_data()
    views = level.cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    packages, checks = [], 0
    here = lambda run, key: directory(run)/"report_raw"/("_".join(map(str, key)))
    for key in KEYS:
        reference = level.transfer.geometry.source.read_cell(level.cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
        old_reference = read(legacy.raw_directory(legacy.result_path(legacy.contract()["reference_run"], *key))/"models.json")["reference"]
        if reference != old_reference:
            raise ValueError("Stage247 and level heads do not share the frozen raw reference")
        with np.load(here(level.transfer.geometry.source.BASIS_RUN, key)/"predictions.npz") as a:
            basis = a["basis"].copy()
        old_dir = legacy.raw_directory(legacy.result_path(LEGACY_RUN, *key))
        models = read(here(LEVEL_RUN, key)/"models.json")
        geometries = read(here(level.source.source.SOURCE_RUN, key)/"models.json")
        package = {"key": list(key), "reference": reference, "basis": basis,
            "stage247_model": read(old_dir/"models.json")["full"]["state_gls"],
            "stage247_selection": read(old_dir/"aggregate.json")["selection"]["state_gls"]["pooled_safe"],
            "banks": [{"bank": b, "geometry": next(m for m in geometries if m["bank"] == b and m["removed"] is None),
                "level": next(m for m in models if m["bank"] == b and m["removed"] is None)} for b in (0, 1)]}
        predictions, old_actions = read(here(LEVEL_RUN, key)/"predictions.json"), read(old_dir/"actions.json")
        for scope, rows in zip(("train", "held"), level.cache.cached.training.cohort.split_rows(views[key[0]], key[1])):
            causal = [{k: v for k, v in row.items() if k != "labels"} for row in rows]
            for profile in learning.infer(causal, package, legacy):
                q = next(q for q in predictions if q["scope"] == scope and q["bank"] == profile["bank"] and q["removed"] is None)
                old = next(q for q in old_actions if q["scope"] == scope+"60")["choices"][legacy.PRIMARY]
                expected = {"stage247": old, "raw": q["choices"]["raw_prior"],
                    "uniform": q["choices"]["linear_offset_clip__raw_prior"],
                    **{k: q["choices"][k+"_clip__raw_prior"] for k in ("action_level", "observed_level")}}
                for name in learning.VARIANTS:
                    if not np.array_equal(profile["choices"][name], expected[name]):
                        raise ValueError(f"Frozen cached action mismatch: {key}/{scope}/{profile['bank']}/{name}")
                    checks += len(rows)
        packages.append(package)
        print(f"FROZEN_PREFIX_CACHED_CHOICES_REPLAYED {key}", flush=True)
    if len(packages) != 16 or checks != 19200:
        raise ValueError("Incomplete frozen historical profile coverage")
    reserved = read(directory(coverage.noise.COLLECTION_RUN)/"preregistration.json")["reserved_scene_seeds"]
    old_seeds = {source.scene_seed(r, s, i) for r in ROOTS for s in SCENARIOS for i in range(8)}
    new_seeds = [scene_seed(*k) for k in prefixes() if k[0] == "new"]
    pilot_seeds = [scene_seed(*k) for k in prefixes(pilot=True) if k[0] == "pilot_new"]
    if len(set(new_seeds+pilot_seeds)) != 90 or set(new_seeds+pilot_seeds)&(old_seeds|set(reserved)):
        raise ValueError("New scenes overlap development/reserved scenes or the software pilot")
    write_json(directory(run)/"freeze_raw/models.json", packages)
    receipt = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "cached_choice_checks": checks, "cached_choice_mismatches": 0, "profiles": 32,
        "new_main_prefixes": 80, "new_software_pilot_prefixes": 10, "new_fits": 0, "new_native_episodes": 0}
    write_json(directory(run)/"freeze/result.json", receipt)


class NativeFrozenPrefixAudit(NativeTimetableContextAudit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = {}

    def select_candidate(self, candidates):
        index = len(self.decisions)
        selected = super().select_candidate(candidates)
        if index in PROBES:
            self.snapshots[index] = self.snapshot if index == self.probe_index else causal_snapshot(self.env)
        return selected


def native_inputs(root):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.native.NATIVE))
    import frequency
    import torch
    frequency.DemandFrequencyTracker = source.native.NativeRoutingTracker
    torch.set_num_threads(1)
    model = read(directory(coverage.noise.COLLECTION_RUN)/"inputs/models.json")[str(root)]
    source.shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = source.native.load_source(root)
    return model, checkpoint


def causal_rows(root, scenario, scene, plan):
    rows = []
    for p in PROBES:
        c = next(c for c in plan.contexts if c["index"] == p)
        row = {"root": root, "scenario": scenario, "scene_seed": scene, "probe_index": p,
            **{k: copy.deepcopy(c[k]) for k in ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")},
            "original_choice": c["greedy"], "context": source.labels.training_context(plan.snapshots[p]),
            "occupancy_context": {k: copy.deepcopy(c[k]) for k in source.context_source.OCCUPANCY_KEYS},
            "timetable_context": copy.deepcopy(c["timetable_context"])}
        qualify_timetable(c, row)
        rows.append(row)
    return rows


def capture(run, key):
    frozen = read(directory(run)/"freeze/result.json")
    if frozen["contract"] != contract() or not frozen["software_qualified"]:
        raise ValueError("Freeze and cached action replay must finish before native capture")
    role, root, scenario, replica = key
    path = prefix_path(run, key)
    model, checkpoint = native_inputs(root)
    scene = scene_seed(*key)
    observer, plan = source.shared.episode(root, scenario, scene, path/"raw/observer", checkpoint,
        model, PROBES[-1], plan_class=NativeFrozenPrefixAudit)
    source.qualify_observer(observer)
    rows = causal_rows(root, scenario, scene, plan)
    if role == "bridge":
        bank = read(coverage.source.bank_path(coverage.SOURCE_RUN))["training_data"]
        for row in rows:
            old = next(r for r in bank if (r["root"], r["scene_seed"], r["probe_index"]) == (root, scene, row["probe_index"]))
            if row != {k: v for k, v in old.items() if k != "labels"}:
                raise ValueError("Bridge source replay changed frozen causal features")
    profiles = []
    for package in read(directory(run)/"freeze_raw/models.json"):
        for profile in learning.infer(rows, package, legacy):
            profiles.append({"key": package["key"], **profile})
    unions = learning.active_union(profiles, [r["original_choice"] for r in rows])
    write_json(path/"choices.json", {"rows": rows, "profiles": profiles, "unions": unions})
    path.mkdir(parents=True, exist_ok=True)
    with (path/"prefix_refs.pkl").open("wb") as f:
        pickle.dump({"prefix": plan.prefix, "snapshots": plan.snapshots}, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(path/"observer.json", observer)
    receipt = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key),
        "scene_seed": scene, "software_qualified": True, "causal_states": 6, "profiles": len(profiles),
        "candidate_counts": [len(u) for u in unions], "native_episodes": 1, "training_updates": 0}
    write_json(path/"receipt.json", receipt)
    print(f"FROZEN_PREFIX_CHOICES_SEALED {key} active={sum(bool(u) for u in unions)}", flush=True)
    return receipt


def paired_values(branch, baseline):
    cost = branch["service_cost_restricted"]-baseline["service_cost_restricted"]
    fleet = max(branch["peak_fleet"]-12, 0)**2/12-max(baseline["peak_fleet"]-12, 0)**2/12
    wait = branch["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"]
    return [cost, fleet, cost-fleet, wait]


def restore_caller_torch_rng(snapshot):
    import torch
    # Native construction forks/restores this stream; frozen evaluation does not
    # consume it. Recreate the source caller's state before the whole episode.
    torch.set_rng_state(torch.tensor(snapshot["rng"]["torch"], dtype=torch.uint8))


def replay(run, key, jobs, output):
    path = prefix_path(run, key)
    sealed = read(path/"choices.json")
    with (path/"prefix_refs.pkl").open("rb") as f:
        references = pickle.load(f)
    role, root, scenario, replica = key
    scene = scene_seed(*key)
    model, checkpoint = native_inputs(root)
    episodes, records = 0, []
    for ordinal, draw, candidates in jobs:
        probe = PROBES[ordinal]
        registered = sealed["rows"][ordinal]
        reference = SimpleNamespace(prefix=references["prefix"], snapshot=references["snapshots"][probe])
        seed = future_seed(*key, probe, draw)
        audit = partial(NativeContinuationValueAudit, continuation_seed=seed)
        raw = output/f"probe{probe}_draw{draw}"
        restore_caller_torch_rng(reference.snapshot)
        baseline, plan = source.shared.episode(root, scenario, scene, raw/"observer", checkpoint, model,
            probe, reference=reference, plan_class=audit)
        source.qualify_observer(baseline)
        branches, values = [], {}
        order = [registered["original_choice"], *[c for c in candidates if c != registered["original_choice"]]]
        for c in order:
            restore_caller_torch_rng(plan.snapshot)
            branch, branch_plan = source.shared.episode(root, scenario, scene, raw/f"candidate{c}",
                checkpoint, model, probe, candidate=c, reference=plan, plan_class=audit)
            source.shared.qualify(branch, baseline, candidate=c)
            if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
                raise ValueError("Branch did not consume the frozen continuation tape")
            branch["policy_trace"] = source.shared.policy_trace(branch.pop("supply"))
            branches.append(branch)
            values[str(c)] = paired_values(branch, baseline)
        baseline["policy_trace"] = source.shared.policy_trace(baseline["supply"])
        source.validate_continuations(baseline, sorted(branches, key=lambda b: b["probe"]["selected"]), candidates=sorted(candidates))
        count = 1+len(order)
        episodes += count
        write_json(raw/"paired_rollouts.json", {"observer": baseline, "branches": branches})
        records.append({"probe_ordinal": ordinal, "draw": draw, "future_seed": seed, "values": values})
        print(f"FROZEN_PREFIX_CONTINUATION_QUALIFIED {key} probe={probe} draw={draw} episodes={count}", flush=True)
    write_json(output/"labels.json", records)
    return {"software_qualified": True, "state_futures": len(records), "native_episodes": episodes,
        "exact_identities": len(records), "matched_prefix_state_rng_branches": episodes-len(records), "training_updates": 0}


def pilot(run, key):
    start = time.monotonic()
    capture(run, key)
    sealed = read(prefix_path(run, key)/"choices.json")
    ordinal = next((i for p in (22, *PROBES) for i in [PROBES.index(p)] if sealed["unions"][i]), PROBES.index(22))
    candidates = sealed["unions"][ordinal] or [sealed["rows"][ordinal]["original_choice"]]
    result = replay(run, key, [(ordinal, 0, candidates)], prefix_path(run, key)/"pilot_raw")
    write_json(prefix_path(run, key)/"pilot.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "key": list(key), **result, "native_episodes": result["native_episodes"]+1,
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        "outcomes_used_for_selection": False})


def pilot_report(run):
    rows = [read(prefix_path(run, k)/"pilot.json") for k in prefixes(pilot=True)]
    if any(not r["software_qualified"] or r["contract"] != contract() or r["outcomes_used_for_selection"] for r in rows):
        raise ValueError("Native pilot did not qualify")
    write_json(directory(run)/"pilot/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "prefixes": len(rows), "native_episodes": sum(r["native_episodes"] for r in rows),
        "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in rows), "max_elapsed_s": max(r["elapsed_s"] for r in rows),
        "mean_s_per_episode": sum(r["elapsed_s"] for r in rows)/sum(r["native_episodes"] for r in rows),
        "outcomes_used_for_selection": False})


def replay_pilot(run, key):
    start = time.monotonic()
    sealed = read(prefix_path(run, key)/"choices.json")
    jobs = [(p, 0, [row["original_choice"]]) for p, row in enumerate(sealed["rows"])]
    result = replay(run, key, jobs, prefix_path(run, key)/"replay_pilot_raw")
    write_json(prefix_path(run, key)/"replay_pilot.json", {"protocol": EXPERIMENT_PROTOCOL,
        "contract": contract(), "key": list(key), **result, "outcomes_used_for_selection": False,
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.})


def replay_pilot_report(run):
    rows = [read(prefix_path(run, k)/"replay_pilot.json") for k in prefixes(pilot=True)]
    if any(not r["software_qualified"] or r["contract"] != contract() or r["native_episodes"] != 12
            or r["state_futures"] != 6 or r["exact_identities"] != 6 or r["outcomes_used_for_selection"] for r in rows):
        raise ValueError("Cross-worker replay must qualify all six archived probes")
    write_json(directory(run)/"replay-pilot/result.json", {"protocol": EXPERIMENT_PROTOCOL,
        "contract": contract(), "software_qualified": True, "prefixes": 20, "states": 120,
        "native_episodes": 240, "outcomes_used_for_selection": False,
        "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in rows)})


def chunks(role):
    return [list(range(i, i+4)) for i in (range(1, 33, 4) if role == "bridge" else range(0, 64, 4))]


def seal_main(run):
    if not read(directory(run)/"pilot/result.json")["software_qualified"]:
        raise ValueError("Native pilot must qualify before main collection")
    tasks, episodes, active_states = [], 0, 0
    for key in prefixes():
        receipt = read(prefix_path(run, key)/"receipt.json")
        if not receipt["software_qualified"] or receipt["contract"] != contract():
            raise ValueError("Incomplete main prefix capture")
        counts = receipt["candidate_counts"]
        active_states += sum(bool(n) for n in counts)
        for c, draws in enumerate(chunks(key[0])):
            if any(counts):
                tasks.append({"key": list(key), "chunk": c})
                episodes += len(draws)*sum(1+n for n in counts if n)
    write_json(directory(run)/"main_plan.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "prefixes": 90, "states": 540, "active_states": active_states,
        "continuation_native_episodes": episodes, "tasks": tasks})


def collect(run, key, chunk):
    start = time.monotonic()
    if not read(directory(run)/"replay-pilot/result.json")["software_qualified"]:
        raise ValueError("Cross-worker replay must qualify before main continuation collection")
    sealed = read(prefix_path(run, key)/"choices.json")
    jobs = [(p, draw, candidates) for p, candidates in enumerate(sealed["unions"]) if candidates for draw in chunks(key[0])[chunk]]
    output = prefix_path(run, key)/"continuations"/str(chunk)
    result = replay(run, key, jobs, output)
    write_json(output/"receipt.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "chunk": chunk,
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024., **result})


def report(run):
    records = []
    execution = {"chunks": 0, "native_episodes": 0, "state_futures": 0,
        "exact_identities": 0, "matched_prefix_state_rng_branches": 0,
        "max_peak_rss_MiB": 0., "max_chunk_elapsed_s": 0.}
    for key in prefixes():
        path = prefix_path(run, key)
        sealed, labels = read(path/"choices.json"), {}
        draws = list(range(1, 33)) if key[0] == "bridge" else list(range(64))
        if any(sealed["unions"]):
            for c in range(len(chunks(key[0]))):
                output = path/"continuations"/str(c)
                receipt = read(output/"receipt.json")
                if not receipt["software_qualified"] or receipt["contract"] != contract():
                    raise ValueError("Incomplete main continuation collection")
                execution["chunks"] += 1
                for field in ("native_episodes", "state_futures", "exact_identities", "matched_prefix_state_rng_branches"):
                    execution[field] += receipt[field]
                execution["max_peak_rss_MiB"] = max(execution["max_peak_rss_MiB"], receipt["peak_rss_MiB"])
                execution["max_chunk_elapsed_s"] = max(execution["max_chunk_elapsed_s"], receipt["elapsed_s"])
                for row in read(output/"labels.json"):
                    labels[row["probe_ordinal"], draws.index(row["draw"])] = {int(k): v for k, v in row["values"].items()}
        mean, covariance = learning.prefix_effects(sealed["profiles"], labels, len(draws))
        records.append({"role": key[0], "root": key[1], "scenario": key[2], "replica": key[3],
            "mean": mean.tolist(), "MC_covariance": covariance.tolist(), "profiles": len(sealed["profiles"])})
    write_json(directory(run)/"report/prefixes.json", records)
    output = {role: learning.summarize_prefixes([r for r in records if r["role"] == role], new=role == "new") for role in ("bridge", "new")}
    success = all(output["new"]["contrasts"][a+"_vs_"+b]["prefix_CI95"][f][1] < 0
        for a, b in learning.PRIMARY for f in ("cost", "wait_min"))
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "groups": output, "joint_primary_passed": success, "new_fits": 0, "training_updates": 0,
        "execution": execution})
    write_json(directory(run)/"report/attribution.json", {"scope": "descriptive_only_no_selection_or_additional_tests",
        "groups": {role: learning.describe_prefixes([r for r in records if r["role"] == role]) for role in ("bridge", "new")}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=("freeze", "pilot", "pilot-report", "prefix", "seal-main", "replay-pilot", "replay-pilot-report", "collect", "report"), required=True)
    p.add_argument("--role", choices=("bridge", "new", "pilot_new"))
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int)
    p.add_argument("--chunk", type=int)
    a = p.parse_args()
    if a.phase in ("freeze", "pilot-report", "seal-main", "replay-pilot-report", "report"):
        {"freeze": freeze, "pilot-report": pilot_report, "seal-main": seal_main,
            "replay-pilot-report": replay_pilot_report, "report": report}[a.phase](a.run_name)
    else:
        key = a.role, a.seed, a.scenario, a.replica
        if key not in prefixes(pilot=a.phase in ("pilot", "replay-pilot")):
            p.error("Prefix is not registered for this phase")
        if a.phase == "collect":
            if a.chunk not in range(len(chunks(a.role))):
                p.error("Unregistered continuation chunk")
            collect(a.run_name, key, a.chunk)
        else:
            {"pilot": pilot, "replay-pilot": replay_pilot, "prefix": capture}[a.phase](a.run_name, key)
    print("FROZEN_PREFIX_PHASE_COMPLETE "+a.phase, flush=True)


if __name__ == "__main__":
    main()
