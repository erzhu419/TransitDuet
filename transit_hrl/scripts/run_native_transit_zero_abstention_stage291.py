#!/usr/bin/env python3
"""Zero-observation abstention versus exactly count-matched random thinning."""

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_fleet_screen_stage290 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_zero_abstention_stage291_v1"
SOURCE_RUN = "native_transit_fleet_screen_stage290_20261017_r1"
ARMS = ("raw", "stage247", "independent_screen", "zero_abstain", "matched_thin", "pooled16")
FIELDS, BANKS, QUERIES = source.FIELDS, source.BANKS, source.QUERIES
PAIRS = tuple((a, "raw") for a in ARMS[1:])+tuple(("zero_abstain", b) for b in ("independent_screen", "matched_thin", "pooled16"))+(("matched_thin", "independent_screen"),)
read = source.read


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"source_run": SOURCE_RUN, "cache_run": source.base.SOURCE_RUN, "arms": list(ARMS),
        "queries": source.contract()["queries"],
        "rule": "Stage290_accepted_changed_proposals_only; abstain_if_all8_paired_screen_fleet_differences_zero; retain_nonzero_mean_le0_including_signed_cancellation; otherwise_raw_no_reselection_or_grid",
        "matched_control": "uniform_without_replacement_exact_k_of_n_Stage290_accepted_changes_per_profile_root_regime_probe; k_equals_witness_count; exact_expected_policy_value_no_random_seed_selection",
        "freeze": "reuse_Stage290_frozen_proposals; primary_proposal_ledger_before_screen_read; witness_choices_and_thinning_probabilities_before_evaluation_read; all6_permutations_fixed",
        "inference": "60prefixes_5000stratified_bootstrap_seed291; resample_same6prefixes_for_all_profiles_probes_arms; recompute_screen_only_k_n_in_each_resample; evaluate_expected_thinning; shared_future_conditional_MC_se; no_assignment_variance_claim",
        "primary": "zero_abstain_vs_raw_cost_wait_CI95_upper_lt0_and_both4draw_halves_cost_wait_lt0",
        "information": "zero_abstain_vs_matched_thin_cost_and_fleet_CI95_upper_lt0_reported_separately_from_joint_retention",
        "budget": {"analysis_tasks": 1, "cached_train_chunks": 2160, "physical_state_futures": 8640,
            "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "Stage290_inspired_exploratory_reused_training_cache_not_fresh_confirmation; no_held_rows_models_or_reference_retraining; expected_random_control_not_an_extra_seed_population_no_default_change"}


def qualify_parent():
    path = source.directory(SOURCE_RUN)
    registration, receipt = read(path/"preregistration.json"), read(path/"completion_receipt.json")
    if (registration["contract"] != source.contract() or receipt["scheduler"]["status_counts"] != {"done": 1}
            or receipt["scheduler"]["exit_code_counts"] != {"0": 1} or not receipt["stage289_control_replay_exact"]
            or receipt["held_prefixes_read"] != 0):
        raise ValueError("Complete qualified Stage290 before Stage291")


def row_cube(rows):
    lookup = {(r["root"], r["scenario"], r["replica"], r["probe_index"]): i for i, r in enumerate(rows)}
    native = source.base.source
    return np.asarray([[[lookup[(root, scenario, replica, probe)] for probe in native.PROBES]
        for replica in range(6)] for root in native.ROOTS for scenario in native.SCENARIOS])


def thin_probabilities(cube, accepted, keep):
    pool, retained = accepted[:, cube], keep[:, cube]
    n, k = pool.sum(axis=2), retained.sum(axis=2)
    fraction = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    probabilities = np.zeros_like(accepted, dtype=float)
    probabilities[:, cube] = pool*fraction[:, :, None]
    error = float(np.max(abs(probabilities[:, cube].sum(axis=2)-k)))
    if error > 1e-12:
        raise ValueError("Matched thinning does not preserve screen-only counts")
    return probabilities, {"eligible_changes": int(accepted.sum()), "kept_changes": int(keep.sum()),
        "expected_thin_changes": float(probabilities.sum()), "matched_count_max_abs_difference": error,
        "strata": int(n.size), "randomizable_kept_changes": int(k[(k > 0)&(k < n)].sum())}


def policy(legacy, screening, cube):
    raw, proposal = legacy[:, 0], legacy[:, 2]
    independent, accepted, zero = source.screen(screening, proposal, raw)
    if not np.array_equal(independent, legacy[:, 4]):
        raise ValueError("Stage290 independent-screen replay differs")
    eligible = accepted&(proposal != raw)
    keep = eligible&~zero
    witness = np.where(keep, proposal, raw)
    probability, counts = thin_probabilities(cube, eligible, keep)
    choices = np.stack((raw, legacy[:, 1], independent, witness, raw, legacy[:, 5]), axis=1)
    return choices, probability, eligible, keep, counts


def effects(choices, probability, proposal, draws):
    paired = source.paired_values(draws, proposal, choices[:, 0])
    result = source.base.source.learning.paired_controls(choices[:, None], draws)
    result[:, :, 4] = (paired*probability[:, :, None, None]).mean(axis=0)
    return result, paired


def bootstrap_samples(prefix_means, proposal_mean, cube, eligible, keep, *, count=5000):
    groups, replicas, probes = cube.shape
    profiles = eligible.shape[0]
    pool, witness = eligible[:, cube], keep[:, cube]
    values = proposal_mean[:, cube]
    rng = np.random.default_rng(291)
    samples = np.zeros((count, len(ARMS), len(FIELDS)))
    for g in range(groups):
        ix = rng.choice(replicas, (count, replicas), replace=True)
        block = prefix_means[g, ix].mean(axis=1)
        block[:, 4] = 0.
        for pi in range(profiles):
            a, k = pool[pi, g, ix], witness[pi, g, ix]
            n, kept = a.sum(axis=1), k.sum(axis=1)
            probability = np.divide(kept, n, out=np.zeros_like(kept, dtype=float), where=n > 0)
            sums = (values[pi, g, ix]*a[..., None]).sum(axis=1)
            block[:, 4] += (sums*probability[..., None]).sum(axis=1)/(profiles*replicas*probes)
        samples += block/groups
    return samples


def evaluate(cube, choices, probability, proposal, eligible, keep, draws):
    actual, paired = effects(choices, probability, proposal, draws)
    means = actual[cube].mean(axis=(2, 3))
    samples = bootstrap_samples(means, paired.mean(axis=2), cube, eligible, keep)
    comparisons = {}
    centered = actual[cube]-actual[cube].mean(axis=3, keepdims=True)
    groups, replicas, probes = cube.shape
    d = draws.shape[1]
    mc = np.einsum("grjdvf,grjdwf->vwf", centered, centered)/(probes**2*d*(d-1)*(groups*replicas)**2)
    for after, before in PAIRS:
        a, b = ARMS.index(after), ARMS.index(before)
        bounds = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
        variance = mc[a, a]+mc[b, b]-2*mc[a, b]
        comparisons[after+"_vs_"+before] = {"mean": dict(zip(FIELDS, map(float, (means[:, :, a]-means[:, :, b]).mean(axis=(0, 1))))),
            "prefix_CI95": {f: list(map(float, bounds[:, j])) for j, f in enumerate(FIELDS)},
            "conditional_MC_se": dict(zip(FIELDS, map(float, np.sqrt(np.maximum(variance, 0.)))))}
    return {"comparisons": comparisons, "halves": {h: source.base.points(actual[:, ds], arms=ARMS)
        for h, ds in (("first4", slice(0, 4)), ("last4", slice(4, 8)))}}


def primary_flags(result):
    c = result["comparisons"]
    joint = all(c["zero_abstain_vs_raw"]["prefix_CI95"][f][1] < 0 for f in ("cost", "wait_min"))
    halves = all(result["halves"][h]["zero_abstain"][f] < 0 for h in ("first4", "last4") for f in ("cost", "wait_min"))
    return {"joint_retention_supported": joint and halves, "primary_joint_CI_passed": joint,
        "primary_both_halves_passed": halves,
        "cost_information_supported": c["zero_abstain_vs_matched_thin"]["prefix_CI95"]["cost"][1] < 0,
        "fleet_information_supported": c["zero_abstain_vs_matched_thin"]["prefix_CI95"]["fleet"][1] < 0}


def load_legacy(name):
    record = read(source.directory(SOURCE_RUN)/"report_raw"/(name+".json"))
    choices = np.asarray(record["choices"])
    if record["arms"] != list(source.ARMS) or record["query"] != list(QUERIES[name]) or choices.shape != (16, 6, 360):
        raise ValueError("Changed Stage290 frozen control ledger")
    return choices


def analyze(run):
    if read(directory(run)/"preregistration.json")["contract"] != contract():
        raise ValueError("Changed registered Stage291 comparison")
    rows = source.base.source.load_rows(source.base.SOURCE_RUN, "train")
    cube = row_cube(rows)
    raw_path = directory(run)/"report_raw"
    first = source.base.load_draws(BANKS[0])
    legacy = load_legacy("primary")
    write_json(raw_path/"primary_proposal.json", {"choices": legacy[:, 2].tolist()})
    print("ZERO_ABSTENTION_PRIMARY_PROPOSAL_FROZEN", flush=True)
    second = source.base.load_draws(BANKS[1])
    choices, probability, eligible, keep, counts = policy(legacy, second, cube)

    def save(name):
        write_json(raw_path/(name+".json"), {"query": QUERIES[name], "arms": list(ARMS),
            "choices": choices.tolist(), "matched_thin_probabilities": probability.tolist(), "matching": counts})

    save("primary")
    print("ZERO_ABSTENTION_PRIMARY_POLICIES_FROZEN", flush=True)
    last = source.base.load_draws(BANKS[2])
    panels = np.concatenate((first, second, last), axis=1)
    reports = {}
    for name, (p, s, e) in QUERIES.items():
        if name != "primary":
            legacy = load_legacy(name)
            choices, probability, eligible, keep, counts = policy(legacy, panels[:, BANKS[s]], cube)
            save(name)
        result = evaluate(cube, choices, probability, legacy[:, 2], eligible, keep, panels[:, BANKS[e]])
        screen_effects, _ = effects(choices, probability, legacy[:, 2], panels[:, BANKS[s]])
        reports[name] = {"matching": counts, "screen8": source.base.points(screen_effects, arms=ARMS), "evaluation8": result}
        print("ZERO_ABSTENTION_QUERY_COMPLETE "+name, flush=True)
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "training_prefixes": 60, "states": 360, "fixed_profiles": 16,
        "queries": reports, **primary_flags(reports["primary"]["evaluation8"]), "stage290_control_replay_exact": True,
        "held_prefixes_read": 0, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    analyze(args.run_name)
    print("ZERO_ABSTENTION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
