#!/usr/bin/env python3
"""Bridge inference conventions and attribute frozen future-confirmation effects."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_abstention_attribution_stage293_v1"
SOURCE_RUN = "native_transit_abstention_confirmation_stage292_20261017_r2"
SOURCE_PROTOCOL = "native_transit_abstention_confirmation_stage292_v1"
OLD_RUN = "native_transit_zero_abstention_stage291_20261017_r1"
GATE_RUN = "native_transit_fleet_screen_stage290_20261017_r1"
CACHE_RUN = "native_transit_prefix_coverage_stage288_20261017_r1"
PAIRS = (("zero_abstain", "raw"), ("zero_abstain", "matched_thin"))
FIELDS = ("cost", "fleet", "service", "wait_min")
ARMS = ("raw", "stage247", "zero_abstain", "matched_thin")


def read(path):
    return json.loads(path.read_text())


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"source_run": SOURCE_RUN, "old_bank": "Stage291_primary_E8_draw16_to23",
        "new_bank": "Stage292_frozen_E64_draw24_to87", "contrasts": [list(p) for p in PAIRS],
        "bridge": "same_frozen_choices_and_weights; fixed_vs_screen_only_rematched_thinning_bootstrap; identical5000_prefix_resamples_seed292_for_all4cells",
        "replay": "new64_fixed_inference_matches_registered_Stage292; old8_points_match_Stage291; alternate_CIs_do_not_replace_primary_criteria",
        "attribution": "60prefixes_root_regime_groups_signs_extreme_contributions_and_equal_stratum_leave_one_prefix_points; descriptive_only_no_pruning_or_policy_selection",
        "variance": "ddof1_stratified_prefix_mean_variance_minus_shared_future_MC_variance; signed_latent_estimate_and_nonnegative_SE_separately_not_summed_twice",
        "budget": {"analysis_tasks": 1, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "postconfirmation_diagnosis_on_known_training_prefixes_not_new_confirmation_or_new_prefix_transfer; no_threshold_or_head_search_no_controller_promotion"}


def qualify_parent():
    path = directory(SOURCE_RUN)
    result, receipt = read(path/"report/result.json"), read(path/"completion_receipt.json")
    if (result["protocol"] != SOURCE_PROTOCOL or result["contract"] != read(path/"preregistration.json")["contract"] or not result["coverage_qualified"]
            or receipt["scheduler"]["collection_status_counts"] != {"done": 2672}
            or receipt["scheduler"]["collection_exit_code_counts"] != {"0": 2672}
            or receipt["scheduler"]["report"]["status"] != "done" or receipt["scheduler"]["report"]["exit_code"] != 0):
        raise ValueError("Complete qualified Stage292 first")


def load_confirmation():
    path = directory(SOURCE_RUN)
    plan, sealed = read(path/"freeze/result.json"), read(path/"sealed_raw/primary.json")
    panels = np.zeros((360, 64, 6, 4))
    # Stage292 already qualified every receipt, draw, seed and action union.
    for i in plan["active_indices"]:
        for chunk in range(16):
            for r in read(path/f"samples/main/state{i}/chunk{chunk}/labels.json"):
                for action, value in r["values"].items():
                    panels[i, r["draw"]-24, int(action)] = value
    return sealed, panels


def load_old(sealed):
    panels = []
    for state in sealed["states"]:
        path = directory(CACHE_RUN)/"prefixes"/Path(*map(str, state["key"]))/f"continuations/probe{state['probe']}"
        labels = {r["draw"]: r["values"] for chunk in (4, 5) for r in read(path/f"chunk{chunk}/labels.json")}
        panels.append([[labels[d][str(a)] for a in range(6)] for d in range(16, 24)])
    return np.asarray(panels)


def rows_and_cube(sealed):
    rows = [{"root": s["key"][1], "scenario": s["key"][2], "replica": s["key"][3], "probe_index": s["probe"]} for s in sealed["states"]]
    groups = list(dict.fromkeys((r["root"], r["scenario"]) for r in rows))
    replicas = list(dict.fromkeys(r["replica"] for r in rows))
    probes = list(dict.fromkeys(r["probe_index"] for r in rows))
    lookup = {(r["root"], r["scenario"], r["replica"], r["probe_index"]): i for i, r in enumerate(rows)}
    cube = np.asarray([[[lookup[(root, scenario, replica, probe)] for probe in probes] for replica in replicas] for root, scenario in groups])
    return rows, cube


def paired_values(panels, choices, raw):
    return np.stack([panels[np.arange(len(c)), :, c]-panels[np.arange(len(c)), :, r] for c, r in zip(choices, raw)])


def effects(sealed, panels):
    choices, proposal = np.asarray(sealed["choices"]), np.asarray(sealed["proposal"])
    actual = np.stack([paired_values(panels, choices[:, v], choices[:, 0]).mean(axis=0) for v in range(len(ARMS))], axis=2)
    actual[:, :, 3] = (paired_values(panels, proposal, choices[:, 0])*np.asarray(sealed["matched_thin_probabilities"])[:, :, None, None]).mean(axis=0)
    return actual


def bridge(sealed, panels, cube, eligible, keep):
    actual = effects(sealed, panels)
    block = actual[cube]
    means = block.mean(axis=(2, 3))
    proposal = np.asarray(sealed["proposal"])
    paired = paired_values(panels, proposal, np.asarray(sealed["choices"])[:, 0]).mean(axis=2)[:, cube]
    pool, witness = eligible[:, cube], keep[:, cube]
    groups, replicas, probes = cube.shape
    profiles = eligible.shape[0]
    fixed = np.zeros((5000, len(ARMS), len(FIELDS)))
    rematched_thin = np.zeros((5000, len(FIELDS)))
    rng = np.random.default_rng(292)
    for g in range(groups):
        ix = rng.choice(replicas, (5000, replicas), replace=True)
        fixed += means[g, ix].mean(axis=1)/groups
        for pi in range(profiles):
            a, w = pool[pi, g, ix], witness[pi, g, ix]
            n, k = a.sum(axis=1), w.sum(axis=1)
            probability = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
            sums = (paired[pi, g, ix]*a[..., None]).sum(axis=1)
            rematched_thin += (sums*probability[..., None]).sum(axis=1)/(profiles*replicas*probes*groups)
    rematched = fixed.copy()
    rematched[:, ARMS.index("matched_thin")] = rematched_thin
    output = {}
    for after, before in PAIRS:
        a, b = ARMS.index(after), ARMS.index(before)
        point = (means[:, :, a]-means[:, :, b]).mean(axis=(0, 1))
        row = {"mean": dict(zip(FIELDS, map(float, point)))}
        for name, samples in (("fixed", fixed), ("rematched", rematched)):
            ci = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
            row[name+"_prefix_CI95"] = {f: list(map(float, ci[:, j])) for j, f in enumerate(FIELDS)}
        output[after+"_vs_"+before] = row
    return output, actual


def attribution(actual, cube, rows):
    block = actual[cube]
    groups, replicas, probes, draws = block.shape[:4]
    keys = [[rows[int(cube[g, r, 0])][k] for k in ("root", "scenario", "replica")]
        for g in range(groups) for r in range(replicas)]
    output, records = {}, []
    for after, before in PAIRS:
        name = after+"_vs_"+before
        delta = block[..., ARMS.index(after), :]-block[..., ARMS.index(before), :]
        means = delta.mean(axis=(2, 3))
        observed = means.var(axis=1, ddof=1).sum(axis=0)/(groups**2*replicas)
        mc = delta.var(axis=3, ddof=1).sum(axis=2)/(probes**2*draws)
        conditional = mc.sum(axis=(0, 1))/(groups*replicas)**2
        latent = observed-conditional
        flat = means.reshape(-1, len(FIELDS))
        pooled = flat.mean(axis=0)
        lo = pooled[None, None]+(means.mean(axis=1)[:, None]-means)/(groups*(replicas-1))
        def item(i):
            return {"key": keys[int(i)], "mean": dict(zip(FIELDS, map(float, flat[i]))),
                "contribution_to_pooled_mean": dict(zip(FIELDS, map(float, flat[i]/len(flat))))}
        cost_order = np.argsort(flat[:, 0])
        wait_index = int(np.argmin(flat[:, 3]))
        group_rows = {f"{keys[g*replicas][0]}/{keys[g*replicas][1]}": dict(zip(FIELDS, map(float, means[g].mean(axis=0)))) for g in range(groups)}
        output[name] = {"groups": group_rows,
            "prefix_signs": {f: {"better": int(np.count_nonzero(flat[:, j] < 0)), "worse": int(np.count_nonzero(flat[:, j] > 0)),
                "equal": int(np.count_nonzero(flat[:, j] == 0))} for j, f in enumerate(FIELDS)},
            "cost_extremes": {"better": [item(int(i)) for i in cost_order[:3]], "worse": [item(int(i)) for i in cost_order[-3:]]},
            "largest_wait_gain_prefix": {**item(wait_index), "share_of_net_wait_gain": float(flat[wait_index, 3]/flat[:, 3].sum()) if pooled[3] < 0 else None},
            "leave_one_prefix_cost_range": [float(lo[..., 0].min()), float(lo[..., 0].max())],
            "variance": {f: {"analytic_prefix_SE_ddof1": float(np.sqrt(observed[j])),
                "conditional_MC_SE": float(np.sqrt(conditional[j])), "latent_signed_variance": float(latent[j]),
                "latent_nonnegative_SE": float(np.sqrt(max(latent[j], 0.)))} for j, f in enumerate(FIELDS)}}
        records.append({"contrast": name, "prefixes": [item(i) for i in range(len(flat))]})
    return output, records


def analyze(run):
    if read(directory(run)/"inputs/preregistration.json")["contract"] != contract():
        raise ValueError("Changed registered attribution")
    parent = read(directory(SOURCE_RUN)/"report/result.json")
    if parent["protocol"] != SOURCE_PROTOCOL or parent["contract"] != read(directory(SOURCE_RUN)/"preregistration.json")["contract"] or not parent["coverage_qualified"]:
        raise ValueError("Unqualified confirmation source")
    sealed, new = load_confirmation()
    rows, cube = rows_and_cube(sealed)
    choices, proposal = np.asarray(sealed["choices"]), np.asarray(sealed["proposal"])
    legacy = read(directory(GATE_RUN)/"report_raw/primary.json")
    eligible = (np.asarray(legacy["choices"])[:, 4] == proposal)&(proposal != choices[:, 0])
    keep = choices[:, 2] != choices[:, 0]
    old = load_old(sealed)
    outputs = {}
    for name, panels in (("old8", old), ("new64", new)):
        outputs[name], actual = bridge(sealed, panels, cube, eligible, keep)
        if name == "new64":
            attributed, prefixes = attribution(actual, cube, rows)
    old_published = read(directory(OLD_RUN)/"report/result.json")["queries"]["primary"]["evaluation8"]["comparisons"]
    for name in outputs["new64"]:
        for field in FIELDS:
            if (abs(outputs["old8"][name]["mean"][field]-old_published[name]["mean"][field]) > 1e-13
                    or abs(outputs["new64"][name]["mean"][field]-parent["evaluation"]["comparisons"][name]["mean"][field]) > 1e-13
                    or np.max(abs(np.asarray(outputs["new64"][name]["fixed_prefix_CI95"][field])-
                        parent["evaluation"]["comparisons"][name]["prefix_CI95"][field])) > 1e-13):
                raise ValueError("Bridge changed frozen confirmation statistics")
    write_json(directory(run)/"report_raw/prefixes.json", prefixes)
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "bridge": outputs, "new64_attribution": attributed, "frozen_statistics_replayed": True,
        "primary_conclusion_unchanged": {k: parent[k] for k in ("joint_retention_supported", "cost_information_supported", "fleet_information_supported")},
        "prefixes": 60, "fixed_profiles": 16, "held_prefixes_read": 0,
        "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    analyze(a.run_name)
    print("ABSTENTION_ATTRIBUTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
