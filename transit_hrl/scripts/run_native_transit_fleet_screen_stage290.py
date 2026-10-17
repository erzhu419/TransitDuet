#!/usr/bin/env python3
"""Three-way cached future split for fixed-proposal fleet screening."""

import argparse
from itertools import permutations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_wait_constraint_stage289 as base
from freq_hrl.experiments.pointmaze_root_response import write_json

EXPERIMENT_PROTOCOL = "native_transit_fleet_screen_stage290_v1"
SOURCE_RUN = "native_transit_wait_constraint_stage289_20261017_r1"
ARMS = ("raw", "stage247", "proposal8", "shared_screen", "independent_screen", "pooled16")
FIELDS = base.FIELDS
BANKS = tuple(tuple(range(8*b, 8*b+8)) for b in range(3))
QUERIES = {"primary" if (p, s, e) == (0, 1, 2) else f"p{p}s{s}e{e}": (p, s, e) for p, s, e in permutations(range(3))}
PAIRS = tuple((a, "raw") for a in ARMS[1:])+tuple(("independent_screen", b) for b in ("proposal8", "shared_screen", "pooled16", "stage247"))
read = base.read


def directory(run):
    return ROOT/"results"/run


def contract():
    return {"source_run": SOURCE_RUN, "cache_run": base.SOURCE_RUN, "arms": list(ARMS),
        "cohort": base.contract()["cohort"],
        "queries": {n: {role: list(BANKS[b]) for role, b in zip(("proposal", "screen", "evaluation"), q)} for n, q in QUERIES.items()},
        "proposal": "Stage289_wait_constrained_rule_on8_only_raw_feasibility_fallback_ties_unchanged",
        "screen": "accept_frozen_proposal_iff_paired_MC_mean_fleet_delta_vs_own_raw_le0_else_raw_no_new_action_search_no_margin_or_grid",
        "controls": "same8proposal_unscreened_and_same_bank_screen; pooled16_same_total_information_replays_Stage289_choices_exactly; unchanged_raw_Stage247",
        "freeze": "primary_proposal_ledger_before_screen_bank_read; final_primary_choices_before_evaluation_bank_read; other5_permutations_fixed_no_winner_selection",
        "inference": "Stage289_shared_prefix_statistics_60prefixes_5000stratified_bootstrap_seed290_profiles_averaged_first; conditional_MC_se; both_evaluation_halves; primary_groups_only",
        "joint_retention": "primary_independent_screen_vs_raw_cost_wait_CI95_upper_lt0_and_both_halves_cost_wait_means_lt0",
        "fleet_reduction": "primary_independent_screen_vs_same8proposal_fleet_CI95_upper_lt0_separate_from_joint_retention",
        "diagnostics": "changed_proposal_gate_counts_are_profile_state_exposures_not_independent_samples; accepted_zero_observed_fleet_differences_reported_separately_with_evaluation_contributions",
        "budget": {"analysis_tasks": 1, "cached_train_chunks": 2160, "physical_state_futures": 8640,
            "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "boundary": "reused_training_only_finite_MC_information_diagnostic_not_deployable_or_risk_certificate; six_permutations_share_data_no_held_access_no_default_change"}


def qualify_parent():
    path = base.directory(SOURCE_RUN)
    receipt, registration = read(path/"completion_receipt.json"), read(path/"preregistration.json")
    if (registration["contract"] != base.contract() or receipt["scheduler"]["status_counts"] != {"done": 1}
            or receipt["scheduler"]["exit_code_counts"] != {"0": 1} or receipt["held_prefixes_read"] != 0):
        raise ValueError("Complete qualified Stage289 before Stage290")


def paired_values(draws, choices, raw):
    return np.stack([draws[np.arange(len(c)), :, c]-draws[np.arange(len(c)), :, r] for c, r in zip(choices, raw)])


def screen(draws, proposal, raw):
    fleet = paired_values(draws, proposal, raw)[..., 1]
    accepted = fleet.mean(axis=2) <= 0
    return np.where(accepted, proposal, raw), accepted, np.all(fleet == 0, axis=2)


def compose(data, proposal_values, screen_values, pooled_values, cached):
    raw, old = cached[:, 0, 0], cached[:, 0, 1]
    proposal = np.stack([base.propose(d, proposal_values, raw[i], old[i])[3] for i, d in enumerate(data)])
    shared, _, _ = screen(proposal_values, proposal, raw)
    independent, accepted, zero = screen(screen_values, proposal, raw)
    pooled = np.stack([base.propose(d, pooled_values, raw[i], old[i])[3] for i, d in enumerate(data)])
    return np.stack((raw, old, proposal, shared, independent, pooled), axis=1), accepted, zero


def gate_diagnostics(choices, accepted, zero, evaluation):
    proposal, raw = choices[:, 2], choices[:, 0]
    changed = proposal != raw
    actual = paired_values(evaluation, proposal, raw)
    masks = {"accepted": changed&accepted, "rejected": changed&~accepted,
        "accepted_zero_observed_fleet": changed&accepted&zero}
    return {"profile_state_exposures": int(changed.size), "proposed_changes": int(changed.sum()),
        "counts": {k: int(m.sum()) for k, m in masks.items()},
        "evaluation_contributions_vs_raw": {k: dict(zip(FIELDS, map(float, (actual*m[:, :, None, None]).mean(axis=(0, 1, 2))))) for k, m in masks.items()}}


def analyze(run):
    if read(directory(run)/"preregistration.json")["contract"] != contract():
        raise ValueError("Changed registered Stage290 comparison")
    source = base.source
    rows = source.load_rows(base.SOURCE_RUN, "train")
    frozen = read(source.directory(base.SOURCE_RUN)/"freeze_raw/models.json")
    with np.load(source.directory(base.SOURCE_RUN)/"fit_raw/train_predictions.npz") as saved:
        cached = saved["chosen"][:, :, :2].copy()
    if len(frozen) != 16 or len(rows) != 360 or not np.all(cached == cached[:, :1]):
        raise ValueError("Changed fixed train profiles or cached controls")
    data = [source.learning.base.base.causal_data(rows, m["reference"]) for m in frozen]
    if any(not np.array_equal(d["raw_choices"], cached[i, 0, 0]) for i, d in enumerate(data)):
        raise ValueError("Raw feasible-menu replay differs from cached control")
    raw_path = directory(run)/"report_raw"
    proposal_values = base.load_draws(BANKS[0])
    primary_proposal = np.stack([base.propose(d, proposal_values, cached[i, 0, 0], cached[i, 0, 1])[3] for i, d in enumerate(data)])
    write_json(raw_path/"primary_proposal.json", {"proposal_draws": BANKS[0], "choices": primary_proposal.tolist()})
    print("FLEET_SCREEN_PRIMARY_PROPOSAL_FROZEN", flush=True)
    screening_values = base.load_draws(BANKS[1])
    choices, accepted, zero = compose(data, proposal_values, screening_values,
        np.concatenate((proposal_values, screening_values), axis=1), cached)
    if not np.array_equal(primary_proposal, choices[:, 2]):
        raise ValueError("Screening changed the frozen primary proposal")
    write_json(raw_path/"primary.json", {"query": QUERIES["primary"], "arms": list(ARMS), "choices": choices.tolist()})
    print("FLEET_SCREEN_PRIMARY_CHOICES_FROZEN", flush=True)
    evaluation_values = base.load_draws(BANKS[2])
    panels = np.concatenate((proposal_values, screening_values, evaluation_values), axis=1)
    reports = {}
    for name, (p, s, e) in QUERIES.items():
        if name != "primary":
            choices, accepted, zero = compose(data, panels[:, BANKS[p]], panels[:, BANKS[s]],
                panels[:, sorted(BANKS[p]+BANKS[s])], cached)
            write_json(raw_path/(name+".json"), {"query": QUERIES[name], "arms": list(ARMS), "choices": choices.tolist()})
        old_query = {2: "primary", 0: "check_hold_first8", 1: "check_hold_middle8"}[e]
        legacy = np.asarray(read(base.directory(SOURCE_RUN)/"report_raw"/(old_query+".json"))["choices"])
        if not np.array_equal(choices[:, (0, 1, 5)], legacy[:, (0, 1, 3)]):
            raise ValueError("Raw, Stage247 or pooled16 choices differ from Stage289")
        result = base.evaluate(rows, choices, panels[:, BANKS[e]], arms=ARMS, pairs=PAIRS, seed=290)
        if name != "primary":
            result.pop("descriptive_groups")
        reports[name] = {"proposal8": base.points(source.learning.paired_controls(choices[:, None], panels[:, BANKS[p]]), arms=ARMS),
            "screen8": base.points(source.learning.paired_controls(choices[:, None], panels[:, BANKS[s]]), arms=ARMS),
            "evaluation8": result, "gate": gate_diagnostics(choices, accepted, zero, panels[:, BANKS[e]])}
        print("FLEET_SCREEN_QUERY_COMPLETE "+name, flush=True)
    primary = reports["primary"]["evaluation8"]
    joint = all(primary["comparisons"]["independent_screen_vs_raw"]["prefix_CI95"][f][1] < 0 for f in ("cost", "wait_min"))
    halves = all(primary["halves"][h]["independent_screen"][f] < 0 for h in ("first4", "last4") for f in ("cost", "wait_min"))
    write_json(directory(run)/"report/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "training_prefixes": 60, "states": 360, "fixed_profiles": 16, "queries": reports,
        "stage289_control_replay_exact": True, "joint_retention_supported": joint and halves,
        "primary_joint_CI_passed": joint, "primary_both_halves_passed": halves,
        "fleet_reduction_supported": primary["comparisons"]["independent_screen_vs_proposal8"]["prefix_CI95"]["fleet"][1] < 0,
        "held_prefixes_read": 0, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0, "controller_promoted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    analyze(args.run_name)
    print("FLEET_SCREEN_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
