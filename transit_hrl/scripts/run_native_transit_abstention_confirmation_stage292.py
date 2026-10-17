#!/usr/bin/env python3
"""Confirm frozen Stage291 actions using new conditional futures only."""

import argparse
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
from scripts import run_native_transit_zero_abstention_stage291 as prior
from scripts import run_native_transit_prefix_coverage_stage288 as cache
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_abstention_confirmation_stage292_v1"
SOURCE_RUN = "native_transit_zero_abstention_stage291_20261017_r1"
CACHE_RUN = prior.source.base.SOURCE_RUN
ARMS = ("raw", "stage247", "zero_abstain", "matched_thin")
FIELDS = prior.FIELDS
PHASES = ("freeze", "pilot", "pilot-report", "collect", "report")
DRAWS, PILOT_DRAWS = tuple(range(24, 88)), (88, 89)
PAIRS = (("stage247", "raw"), ("zero_abstain", "raw"), ("matched_thin", "raw"),
    ("zero_abstain", "matched_thin"), ("zero_abstain", "stage247"))
read = prior.read


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"source_run": SOURCE_RUN, "cache_run": CACHE_RUN, "query": "primary_only_no_permutation_selection",
        "arms": list(ARMS), "draws": list(DRAWS), "pilot_draws": list(PILOT_DRAWS),
        "freeze": "copy_Stage291_primary_choices_proposal_and_screen_only_thinning_weights_before_new_futures; no_reselection_reweighting_or_fit",
        "scene": "same60_Stage288_training_prefixes_all6probes_no_held_prefix_access_or_new_capture",
        "future_seed": cache.contract()["future_seed"],
        "simulation": "only_within_profile_policy_disagreements_union_plus_native_identity; archived_prefix_state_rng_and_shared_continuation_tape",
        "inference": "fixed_weights_prefix_bootstrap5000_seed292_60prefixes; separate_shared_future_MC_normal95; 16profiles_averaged_before_inference_not_seeds",
        "joint": "zero_abstain_vs_raw_cost_wait_both_prefix_and_conditional_MC_CI95_upper_lt0_and_both32draw_halves_negative",
        "information": "zero_abstain_vs_matched_thin_cost_and_fleet_both_CI95_upper_lt0_reported_separately",
        "pilot": "first_active_state_per_root_regime_or_identity_fallback_two_disjoint_futures; software_only_no_outcome_selection",
        "new_fits": 0, "actor_updates": 0,
        "boundary": "fresh_future_fixed_prefix_single_intervention_confirmation_not_new_prefix_transfer_or_end_to_end_HRL; no_controller_promotion"}


def qualify_parent():
    path = prior.directory(SOURCE_RUN)
    result, receipt = read(path/"report/result.json"), read(path/"completion_receipt.json")
    if (result["contract"] != prior.contract() or not result["stage290_control_replay_exact"]
            or receipt["scheduler"]["status_counts"] != {"done": 1}
            or receipt["scheduler"]["exit_code_counts"] != {"0": 1}):
        raise ValueError("Complete qualified Stage291 first")


def seal(ledger, legacy, rows):
    choices = np.asarray(ledger["choices"], dtype=int)
    proposal = np.asarray(legacy["choices"], dtype=int)[:, 2]
    probability = np.asarray(ledger["matched_thin_probabilities"], dtype=float)
    raw, witness = choices[:, 0], choices[:, 3]
    eligible = (choices[:, 2] == proposal)&(proposal != raw)
    keep = witness != raw
    if (ledger["arms"] != list(prior.ARMS) or ledger["query"] != [0, 1, 2]
            or choices.shape != (16, 6, 360) or np.any(keep&~eligible)
            or not np.array_equal(witness, np.where(keep, proposal, raw))
            or not np.array_equal(choices[:, [0, 1, 2, 5]], np.asarray(legacy["choices"])[:, [0, 1, 4, 5]])):
        raise ValueError("Stage291 primary action ledger changed")
    expected, counts = prior.thin_probabilities(prior.row_cube(rows), eligible, keep)
    if not np.array_equal(expected, probability) or counts != ledger["matching"]:
        raise ValueError("Stage291 frozen matching changed")
    fixed = choices[:, [0, 1, 3, 4]]
    states = []
    for i, row in enumerate(rows):
        union = set()
        for pi in range(16):
            actions = set(map(int, fixed[pi, :3, i]))
            if probability[pi, i] > 0:
                actions.add(int(proposal[pi, i]))
            if len(actions) > 1:
                union.update(actions)
        if union:
            union.add(int(row["original_choice"]))
        states.append({"state_index": i, "key": ["train", row["root"], row["scenario"], row["replica"]],
            "probe": row["probe_index"], "original": row["original_choice"], "candidates": sorted(union)})
    pilots = []
    for root in cache.ROOTS:
        for scenario in cache.SCENARIOS:
            group = [s for s in states if s["key"][1:3] == [root, scenario]]
            pilots.append(next((s["state_index"] for s in group if s["candidates"]), group[0]["state_index"]))
    return {"arms": list(ARMS), "choices": fixed.tolist(), "proposal": proposal.tolist(),
        "matched_thin_probabilities": probability.tolist(), "states": states, "pilot_indices": pilots, "matching": counts}


def freeze(run):
    parent = read(prior.directory(SOURCE_RUN)/"report/result.json")
    if parent["contract"] != prior.contract() or not parent["stage290_control_replay_exact"]:
        raise ValueError("Changed Stage291 source")
    sealed = seal(read(prior.directory(SOURCE_RUN)/"report_raw/primary.json"),
        read(prior.source.directory(prior.SOURCE_RUN)/"report_raw/primary.json"), cache.load_rows(CACHE_RUN, "train"))
    seeds = [cache.future_seed(tuple(s["key"]), s["probe"], d) for s in sealed["states"] for d in (*DRAWS, *PILOT_DRAWS)]
    old = {cache.future_seed(tuple(s["key"]), s["probe"], d) for s in sealed["states"] for d in range(24)}
    if len(set(seeds)) != len(seeds) or set(seeds)&old:
        raise ValueError("New futures overlap cached futures or one another")
    write_json(directory(run)/"sealed_raw/primary.json", sealed)
    active = [s for s in sealed["states"] if s["candidates"]]
    write_json(directory(run)/"freeze/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "training_prefixes": 60, "states": 360,
        "active_indices": [s["state_index"] for s in active], "pilot_indices": sealed["pilot_indices"],
        "collection_tasks": len(active)*16, "state_futures": len(active)*64,
        "native_episodes": 64*sum(1+len(s["candidates"]) for s in active),
        "pilot_native_episodes": 2*sum(1+len(sealed["states"][i]["candidates"] or [sealed["states"][i]["original"]]) for i in sealed["pilot_indices"]),
        "matching": sealed["matching"], "held_prefixes_read": 0, "new_fits": 0})


def receipt_path(run, phase, index, chunk=None):
    return directory(run)/("pilot" if phase == "pilot" else "receipts")/f"state{index}"/("receipt.json" if phase == "pilot" else f"chunk{chunk}/receipt.json")


def collect(run, phase, index, chunk=None):
    start = time.monotonic()
    frozen = read(directory(run)/"freeze/result.json")
    if frozen["contract"] != contract() or not frozen["software_qualified"]:
        raise ValueError("Freeze actions first")
    if phase == "collect" and not read(directory(run)/"pilot-report/result.json")["software_qualified"]:
        raise ValueError("Native preflight must qualify before collection")
    sealed = read(directory(run)/"sealed_raw/primary.json")
    valid = frozen["pilot_indices"] if phase == "pilot" else frozen["active_indices"]
    if index not in valid or (phase == "collect" and chunk not in range(16)):
        raise ValueError("Unregistered continuation task")
    state = sealed["states"][index]
    key, probe = tuple(state["key"]), state["probe"]
    candidates = state["candidates"] or [state["original"]]
    draws = PILOT_DRAWS if phase == "pilot" else DRAWS[4*chunk:4*chunk+4]
    source_path = cache.prefix_path(CACHE_RUN, key)
    with (source_path/"prefix_refs.pkl").open("rb") as f:
        refs = pickle.load(f)
    native = cache.native
    model, checkpoint = native.native_inputs(key[1])
    output = directory(run)/"samples"/("pilot" if phase == "pilot" else "main")/f"state{index}"/f"chunk{chunk}"
    records = []
    for draw in draws:
        reference = SimpleNamespace(prefix=refs["prefix"], snapshot=refs["snapshots"][probe])
        seed = cache.future_seed(key, probe, draw)
        audit = partial(native.NativeContinuationValueAudit, continuation_seed=seed)
        raw = output/f"raw/draw{draw}"
        native.restore_caller_torch_rng(reference.snapshot)
        baseline, plan = native.source.shared.episode(key[1], key[2], cache.scene_seed(key), raw/"observer", checkpoint,
            model, probe, reference=reference, plan_class=audit)
        native.source.qualify_observer(baseline)
        branches, values = [], {}
        for candidate in [state["original"], *[c for c in candidates if c != state["original"]]]:
            native.restore_caller_torch_rng(plan.snapshot)
            branch, branch_plan = native.source.shared.episode(key[1], key[2], cache.scene_seed(key), raw/f"candidate{candidate}",
                checkpoint, model, probe, candidate=candidate, reference=plan, plan_class=audit)
            native.source.shared.qualify(branch, baseline, candidate=candidate)
            if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
                raise ValueError("Branch did not consume its registered shared future")
            branch["policy_trace"] = native.source.shared.policy_trace(branch.pop("supply"))
            branches.append(branch)
            values[str(candidate)] = native.paired_values(branch, baseline)
        baseline["policy_trace"] = native.source.shared.policy_trace(baseline["supply"])
        native.source.validate_continuations(baseline, sorted(branches, key=lambda b: b["probe"]["selected"]), candidates=candidates)
        write_json(raw/"paired_rollouts.json", {"observer": baseline, "branches": branches})
        records.append({"draw": draw, "future_seed": seed, "values": values})
        print(f"ABSTENTION_CONFIRM_DRAW_QUALIFIED state={index} draw={draw}", flush=True)
    write_json(output/"labels.json", records)
    write_json(receipt_path(run, phase, index, chunk), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "state_index": index, "draws": list(draws), "state_futures": len(draws),
        "native_episodes": len(draws)*(1+len(candidates)), "exact_identities": len(draws),
        "matched_prefix_state_rng_branches": len(draws)*len(candidates), "outcomes_used_for_selection": False,
        "elapsed_s": time.monotonic()-start, "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.})


def pilot_report(run):
    frozen = read(directory(run)/"freeze/result.json")
    receipts = [read(receipt_path(run, "pilot", i)) for i in frozen["pilot_indices"]]
    if any(r["contract"] != contract() or not r["software_qualified"] or r["draws"] != list(PILOT_DRAWS)
            or r["exact_identities"] != 2 or r["outcomes_used_for_selection"] for r in receipts):
        raise ValueError("Cross-worker native replay preflight failed")
    write_json(directory(run)/"pilot-report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "root_regime_groups": len(receipts), "state_futures": 2*len(receipts),
        "native_episodes": sum(r["native_episodes"] for r in receipts), "outcomes_used_for_selection": False,
        "max_peak_rss_MiB": max(r["peak_rss_MiB"] for r in receipts), "max_elapsed_s": max(r["elapsed_s"] for r in receipts)})


def effects(sealed, panels):
    choices, proposal = np.asarray(sealed["choices"]), np.asarray(sealed["proposal"])
    probability = np.asarray(sealed["matched_thin_probabilities"])
    actual = cache.learning.paired_controls(choices[:, None], panels)
    paired = prior.source.paired_values(panels, proposal, choices[:, 0])
    actual[:, :, 3] = (paired*probability[:, :, None, None]).mean(axis=0)
    return actual


def summarize(sealed, panels, cube):
    actual = effects(sealed, panels)
    block = actual[cube]
    means = block.mean(axis=(2, 3))
    g, r, j, d = block.shape[:4]
    centered = block-block.mean(axis=3, keepdims=True)
    covariance = np.einsum("grjdvf,grjdwf->vwf", centered, centered)/(j*j*d*(d-1)*(g*r)**2)
    rng = np.random.default_rng(292)
    samples = sum(means[i, rng.choice(r, (5000, r), replace=True)].mean(axis=1)/g for i in range(g))
    comparisons = {}
    for after, before in PAIRS:
        a, b = ARMS.index(after), ARMS.index(before)
        point = (means[:, :, a]-means[:, :, b]).mean(axis=(0, 1))
        se = np.sqrt(np.maximum(covariance[a, a]+covariance[b, b]-2*covariance[a, b], 0.))
        ci = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
        mc = np.stack((point-1.959963984540054*se, point+1.959963984540054*se))
        comparisons[after+"_vs_"+before] = {"mean": dict(zip(FIELDS, map(float, point))),
            "conditional_MC_se": dict(zip(FIELDS, map(float, se))),
            "prefix_CI95": {f: list(map(float, ci[:, k])) for k, f in enumerate(FIELDS)},
            "conditional_MC_CI95": {f: list(map(float, mc[:, k])) for k, f in enumerate(FIELDS)}}
    halves = {name: {arm: dict(zip(FIELDS, map(float, actual[:, ds, vi].mean(axis=(0, 1)))))
        for vi, arm in enumerate(ARMS)} for name, ds in (("first32", slice(0, d//2)), ("last32", slice(d//2, d)))}
    return {"comparisons": comparisons, "halves": halves,
        "fleet_nonzero_state_future_counts": dict(zip(ARMS, map(int, np.count_nonzero(actual[..., 1], axis=(0, 1)))))}


def flags(result):
    c = result["comparisons"]
    passed = lambda contrast, fields: all(c[contrast][interval][f][1] < 0
        for interval in ("prefix_CI95", "conditional_MC_CI95") for f in fields)
    halves = all(result["halves"][h]["zero_abstain"][f] < 0 for h in ("first32", "last32") for f in ("cost", "wait_min"))
    joint = passed("zero_abstain_vs_raw", ("cost", "wait_min"))
    return {"joint_retention_supported": joint and halves, "primary_both_CI_passed": joint,
        "primary_both_halves_passed": halves, "cost_information_supported": passed("zero_abstain_vs_matched_thin", ("cost",)),
        "fleet_information_supported": passed("zero_abstain_vs_matched_thin", ("fleet",))}


def report(run):
    frozen, sealed = read(directory(run)/"freeze/result.json"), read(directory(run)/"sealed_raw/primary.json")
    panels = np.zeros((360, 64, 6, 4))
    episodes = 0
    for i in frozen["active_indices"]:
        state = sealed["states"][i]
        for chunk in range(16):
            receipt = read(receipt_path(run, "collect", i, chunk))
            draws = DRAWS[4*chunk:4*chunk+4]
            if receipt["contract"] != contract() or not receipt["software_qualified"] or receipt["draws"] != list(draws):
                raise ValueError("Incomplete registered future collection")
            records = read(directory(run)/f"samples/main/state{i}/chunk{chunk}/labels.json")
            if [v["draw"] for v in records] != list(draws):
                raise ValueError("Missing or duplicate confirmation draws")
            for record in records:
                if (record["future_seed"] != cache.future_seed(tuple(state["key"]), state["probe"], record["draw"])
                        or sorted(map(int, record["values"])) != state["candidates"]):
                    raise ValueError("Changed paired future or action union")
                for action, value in record["values"].items():
                    panels[i, DRAWS.index(record["draw"]), int(action)] = value
            episodes += receipt["native_episodes"]
    rows = cache.load_rows(CACHE_RUN, "train")
    result = summarize(sealed, panels, prior.row_cube(rows))
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "training_prefixes": 60, "states": 360, "fixed_profiles": 16,
        "evaluation": result, **flags(result), "execution": {"collection_tasks": frozen["collection_tasks"],
            "state_futures": frozen["state_futures"], "native_episodes": episodes},
        "held_prefixes_read": 0, "new_fits": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--state-index", type=int)
    p.add_argument("--chunk", type=int)
    a = p.parse_args()
    if read(directory(a.run_name)/"preregistration.json")["contract"] != contract():
        p.error("Changed registered confirmation")
    if a.phase in ("pilot", "collect"):
        collect(a.run_name, a.phase, a.state_index, a.chunk)
    else:
        {"freeze": freeze, "pilot-report": pilot_report, "report": report}[a.phase](a.run_name)
    print("ABSTENTION_CONFIRM_PHASE_COMPLETE "+a.phase, flush=True)


if __name__ == "__main__":
    main()
