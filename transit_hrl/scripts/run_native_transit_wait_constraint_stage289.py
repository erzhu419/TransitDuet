#!/usr/bin/env python3
"""Cached training-prefix cost/wait information probes across disjoint futures."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_coverage_stage288 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_wait_constraint_stage289_v1"
SOURCE_RUN = "native_transit_prefix_coverage_stage288_20261017_r1"
ARMS = ("raw", "stage247", "cost_only", "wait_constrained")
FIELDS = source.learning.FIELDS
FOLDS = {
    "primary": (tuple(range(16)), tuple(range(16, 24))),
    "check_hold_first8": (tuple(range(8, 24)), tuple(range(8))),
    "check_hold_middle8": (tuple(range(8))+tuple(range(16, 24)), tuple(range(8, 16))),
}
PAIRS = (("stage247", "raw"), ("cost_only", "raw"), ("wait_constrained", "raw"),
    ("wait_constrained", "cost_only"), ("wait_constrained", "stage247"))
read = source.read


def contract():
    return {"source_run": SOURCE_RUN, "arms": list(ARMS),
        "cohort": "Stage288_train_only_60prefixes_360states_24futures_16fixed_profiles_no_rotation_replication",
        "folds": {k: {"selection": list(s), "evaluation": list(e)} for k, (s, e) in FOLDS.items()},
        "selection": "min_MC_mean_cost_with_frozen_raw_feasibility_and_native_original_fallback; wait_constrained_MC_mean_wait_delta_vs_own_raw_le0; raw_first_then_original_then_index_ties",
        "controls": "unchanged_Stage288_raw_and_Stage247_saved_train_choices; no_dense_wide_fit_or_selection",
        "freeze": "primary_choices_written_before_reading_futures16_to23; each_query_propose_receives_selection_only; reverse_checks_fixed_not_blinded",
        "inference": "average16profiles_then6probes_before60prefix_stratified_bootstrap5000_seed289; conditional_MC_se_preserves_shared_futures; evaluation_first4_last4_reported",
        "joint_retention": "primary_wait_constrained_vs_raw_cost_and_wait_CI95_upper_lt0_and_both_evaluation_halves_cost_wait_means_lt0",
        "budget": {"analysis_tasks": 1, "cached_train_chunks": 2160, "physical_state_futures": 8640,
            "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "reused_training_development_data_finite_MC_information_probe_not_deployable_or_oracle_bound; no_held_prefix_access_or_tuning_no_default_change"}


def directory(run):
    return ROOT/"results"/run


def qualify_parent():
    path = source.directory(SOURCE_RUN)
    registration, receipt = read(path/"preregistration.json"), read(path/"completion_receipt.json")
    fit = read(path/"fit/result.json")
    if (registration["contract"] != source.contract() or fit["contract"] != source.contract()
            or not fit["models_and_choices_frozen"] or fit["held_labels_read"]
            or receipt["scheduler"]["collection_status_counts"] != {"done": 3120}
            or receipt["scheduler"]["collection_exit_code_counts"] != {"0": 3120}
            or any(receipt["scheduler"][p]["status"] != "done" or receipt["scheduler"][p]["exit_code"] != 0 for p in ("fit", "report"))):
        raise ValueError("Complete qualified Stage288 before Stage289")


def load_draws(indices):
    values = []
    for key in source.keys("train"):
        for probe in source.PROBES:
            labels = {}
            for chunk in sorted({d//4 for d in indices}):
                path = source.continuation_path(SOURCE_RUN, key, probe, chunk)
                receipt = read(path/"receipt.json")
                if not receipt["software_qualified"] or receipt["contract"] != source.contract():
                    raise ValueError("Unqualified cached train continuation")
                labels.update({r["draw"]: r["values"] for r in read(path/"labels.json")})
            values.append([[labels[d][str(a)] for a in range(6)] for d in indices])
    return np.asarray(values)


def propose(data, selection, raw, old):
    mean = selection.mean(axis=1)
    relative = mean-mean[np.arange(len(raw)), raw][:, None]
    eligible = data["raw_logits"] <= 0
    original = data["originals"]
    eligible[np.arange(len(raw)), original] = True
    if not np.all(eligible[np.arange(len(raw)), raw]):
        raise ValueError("Cached raw choice is outside its native feasible menu")
    choices = [raw, old]
    for constrained in (False, True):
        allowed = eligible & (relative[..., 3] <= 0) if constrained else eligible
        costs = np.where(allowed, relative[..., 0], np.inf)
        chosen = costs.argmin(axis=1)
        for preferred in (original, raw):
            tied = costs[np.arange(len(raw)), preferred] == costs[np.arange(len(raw)), chosen]
            chosen[tied] = preferred[tied]
        choices.append(chosen)
    return np.stack(choices)


def points(effects, *, arms=ARMS):
    return {a: dict(zip(FIELDS, map(float, effects[:, :, i].mean(axis=(0, 1))))) for i, a in enumerate(arms)}


def evaluate(rows, choices, draws, *, arms=ARMS, pairs=PAIRS, seed=289):
    effects = source.learning.paired_controls(choices[:, None], draws)
    records = []
    for key in source.keys("train"):
        ix = [i for i, r in enumerate(rows) if (r["root"], r["scenario"], r["replica"]) == key[1:]]
        if len(ix) != 6:
            raise ValueError("Expected six probes per training prefix")
        paired = effects[ix]
        centered = paired-paired.mean(axis=1, keepdims=True)
        covariance = np.einsum("pdvf,pdwf->vwf", centered, centered)/(36*draws.shape[1]*(draws.shape[1]-1))
        records.append({"root": key[1], "scenario": key[2], "replica": key[3],
            "mean": paired.mean(axis=(0, 1)), "MC_covariance": covariance})
    means = np.asarray([r["mean"] for r in records])
    samples = np.zeros((5000, len(arms), len(FIELDS)))
    rng = np.random.default_rng(seed)
    for root in source.ROOTS:
        for scenario in source.SCENARIOS:
            ix = [i for i, r in enumerate(records) if (r["root"], r["scenario"]) == (root, scenario)]
            if len(ix) != 6:
                raise ValueError("Expected six training prefixes per root/regime")
            samples += means[rng.choice(ix, (5000, 6), replace=True)].mean(axis=1)/(len(source.ROOTS)*len(source.SCENARIOS))
    mc = np.asarray([r["MC_covariance"] for r in records]).sum(axis=0)/len(records)**2
    comparisons = {}
    for after, before in pairs:
        a, b = arms.index(after), arms.index(before)
        bounds = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
        variance = mc[a, a]+mc[b, b]-2*mc[a, b]
        comparisons[after+"_vs_"+before] = {
            "mean": dict(zip(FIELDS, map(float, (means[:, a]-means[:, b]).mean(axis=0)))),
            "prefix_CI95": {f: list(map(float, bounds[:, j])) for j, f in enumerate(FIELDS)},
            "conditional_MC_se": dict(zip(FIELDS, map(float, np.sqrt(np.maximum(variance, 0.)))))}
    # Scalar groups are descriptive; the sole primary interval uses all 60 prefixes.
    groups = {f"root{root}": [i for i, r in enumerate(records) if r["root"] == root] for root in source.ROOTS}
    groups.update({s: [i for i, r in enumerate(records) if r["scenario"] == s] for s in source.SCENARIOS})
    return {"comparisons": comparisons, "halves": {h: points(effects[:, ds], arms=arms) for h, ds in (("first4", slice(0, 4)), ("last4", slice(4, 8)))},
        "descriptive_groups": {g: {a: dict(zip(FIELDS, map(float, means[ix, j].mean(axis=0)))) for j, a in enumerate(arms)} for g, ix in groups.items()}}


def analyze(run):
    if read(directory(run)/"preregistration.json")["contract"] != contract():
        raise ValueError("Changed registered Stage289 comparison")
    rows = source.load_rows(SOURCE_RUN, "train")
    frozen = read(source.directory(SOURCE_RUN)/"freeze_raw/models.json")
    with np.load(source.directory(SOURCE_RUN)/"fit_raw/train_predictions.npz") as saved:
        cached = saved["chosen"][:, :, :2].copy()
    if len(frozen) != 16 or len(rows) != 360 or not np.all(cached == cached[:, :1]):
        raise ValueError("Changed fixed profiles, train coverage or cached controls")
    data = [source.learning.base.base.causal_data(rows, m["reference"]) for m in frozen]
    if any(not np.array_equal(d["raw_choices"], cached[i, 0, 0]) for i, d in enumerate(data)):
        raise ValueError("Raw feasible-menu replay differs from cached control")
    raw_path = directory(run)/"report_raw"
    first = load_draws(FOLDS["primary"][0])

    def save_choices(name, selection):
        choices = np.stack([propose(d, selection, cached[i, 0, 0], cached[i, 0, 1]) for i, d in enumerate(data)])
        write_json(raw_path/(name+".json"), {"selection": FOLDS[name][0], "evaluation": FOLDS[name][1],
            "profile_keys": [m["key"] for m in frozen], "arms": list(ARMS), "choices": choices.tolist()})
        return choices

    proposals = {"primary": save_choices("primary", first)}
    print("WAIT_CONSTRAINT_PRIMARY_CHOICES_FROZEN", flush=True)
    # The independent bank is first opened after the primary action ledger exists.
    last = load_draws(FOLDS["primary"][1])
    panels = np.concatenate((first, last), axis=1)
    for name, (selection, _) in list(FOLDS.items())[1:]:
        proposals[name] = save_choices(name, panels[:, selection])
    reports = {}
    for name, (selection, evaluation) in FOLDS.items():
        if set(selection)&set(evaluation):
            raise ValueError("Selection and evaluation futures overlap")
        choices = proposals[name]
        reports[name] = {"selection16": points(source.learning.paired_controls(choices[:, None], panels[:, selection])),
            "independent8": evaluate(rows, choices, panels[:, evaluation]),
            "changed_fraction_vs_raw": {a: float((choices[:, i] != choices[:, 0]).mean()) for i, a in enumerate(ARMS)}}
        print("WAIT_CONSTRAINT_QUERY_COMPLETE "+name, flush=True)
    primary = reports["primary"]["independent8"]
    joint = all(primary["comparisons"]["wait_constrained_vs_raw"]["prefix_CI95"][f][1] < 0 for f in ("cost", "wait_min"))
    halves = all(primary["halves"][h]["wait_constrained"][f] < 0 for h in ("first4", "last4") for f in ("cost", "wait_min"))
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "training_prefixes": 60, "states": 360, "fixed_profiles": 16,
        "queries": reports, "joint_retention_supported": joint and halves,
        "primary_joint_CI_passed": joint, "primary_both_halves_passed": halves,
        "held_prefixes_read": 0, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0,
        "controller_promoted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    analyze(args.run_name)
    print("WAIT_CONSTRAINT_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
