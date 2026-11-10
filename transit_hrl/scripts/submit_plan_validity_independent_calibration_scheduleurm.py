"""Freeze disjoint precheck/calibration/readout cohorts; submit prechecks only."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def literal_seeds(value, seed_context=False):
    if isinstance(value, dict):
        return set().union(*(literal_seeds(v, seed_context or "seed" in k) for k, v in value.items()), set())
    if isinstance(value, list):
        return set().union(*(literal_seeds(v, seed_context) for v in value), set())
    return {value} if seed_context and isinstance(value, int) and not isinstance(value, bool) else set()


def prepare(source, prior, plan, run_name, manifest_path, forbidden):
    registered = list(seed_cells(plan))
    forbidden = set(forbidden) | literal_seeds(source) | literal_seeds(prior)
    for cell in registered:
        forbidden.update(literal_seeds(cell))
        forbidden.update(_training_seed(optimizer_seed=cell["optimizer_seed"], rollout_root=s, iteration=i)
            for s in cell["seed_roles"]["train_seeds"]
            for i in range(2 if cell["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    cells, jobs = [], {phase: [] for phase in ("precheck", "sample", "select", "readout")}
    previous = {(c["task"], c["optimizer_seed"]): c for c in prior["cells"]}
    for index, old in enumerate(source["cells"]):
        cell = deepcopy(old)
        seed = cell["optimizer_seed"]
        cell.update(precheck_seeds=list(range(seed + 61000, seed + 61004)),
            calibration_prefix_seeds=list(range(seed + 62000, seed + 62016)),
            readout_prefix_seeds=list(range(seed + 63000, seed + 63008)),
            fit_directory=old["output_directory"] + "/fit",
            previous_calibration_directory=previous[(cell["task"], seed)]["output_directory"],
            output_directory=f"results/{run_name}/{cell['task']}/{seed}")
        new = cell["precheck_seeds"] + cell["calibration_prefix_seeds"] + cell["readout_prefix_seeds"]
        if len(set(new)) != len(new) or set(new) & forbidden:
            raise ValueError("new cohorts overlap exposed, reserved or effective training seeds")
        forbidden.update(new)
        cells.append(cell)
        for phase, shards in (("precheck", [-1]), ("sample", [0, 1]), ("select", [-1]), ("readout", [-1])):
            for shard in shards:
                directory = ROOT / cell["output_directory"] / phase
                if shard >= 0:
                    directory /= f"shard{shard}"
                command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_independent_calibration.py",
                    "--manifest", str(manifest_path), "--cell-index", str(index), "--phase", phase, "--shard-index", str(shard)]
                job = deepcopy(source["jobs"]["precheck"][index])
                job.update(project="Freq-HRL-IndependentCalibration", description=f"Independent cut {phase} {cell['task']} {seed} shard{shard}",
                    signature=f"Freq-HRL/independent-calibration/{run_name}/{phase}/{cell['task']}/{seed}/shard{shard}",
                    cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                    result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
                    wait_for_files=[str(ROOT / manifest_path)])
                jobs[phase].append(job)
    return {"protocol_version": "independent_prefix_fixed_score_cut_calibration_v1", "run_name": run_name,
        "source_label_run_name": source["run_name"], "source_fullscore_run_name": prior["run_name"],
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "horizon": 1200, "probes_per_path": 3, "calibration_paths_per_root": 16, "readout_paths_per_root": 8,
        "calibration_A_futures": 4, "calibration_B_futures": 4, "readout_E_futures": 8,
        "selection": "both_frozen_heads_full_scores_calibration_A_only_G_max_W_floor_vs_original_numeric_quantile",
        "ties": "maximize_G_then_minimize_changed_A_actions_then_distance_from_original_cut_then_declared_order",
        "common_continuation": "frozen_C0_phase_checks_clock_same_prefix_policy",
        "future_seed_recipe": "SeedSequence([optimizer_seed,new_path_seed,step,replica_index,840])",
        "sampling_recipe": "existing_observe_617_bins_839_optional_clock_three_probes_no_outcome_selection",
        "cohort_exclusion": "all_plan_validity_protocol_literal_seeds_plus_registered_roles_and_effective_training_seeds",
        "per_job_environment_steps": {"precheck": 12000, "sample": 528000, "select": 0, "readout": 528000},
        "precheck_environment_step_upper_bound": 48000, "calibration_environment_steps": 4224000,
        "readout_environment_steps": 2112000, "total_environment_step_upper_bound": 6384000,
        "new_full_model_fits": 0, "ridge_fit_calls": 0,
        "execution_order": "four_native_prechecks_then_eight_calibration_shards_then_four_cut_freezes_then_four_readouts",
        "current_release": "precheck_only_formal_sampling_requires_explicit_next_stage_decision",
        "readout_gate": "all_four_roots_cuts_frozen_before_any_readout_prefix_access",
        "comparisons": ["independent_minus_quantile", "independent_minus_previous_full_calibration", "independent_minus_C0"],
        "statistics": "prefix_paired_t95_and_conditional_MC_SE_E4_halves_separate_roots_both_heads_no_winner_selection",
        "positive_evidence": "per_head_W_G_positive_in_both_E4_halves_and_both_roots_report_CIs_no_cross_root_claim_from_prefix_CIs",
        "episode_followup": "none_single_intervention_only_no_automatic_controller_or_timing_recipe_promotion",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--previous-calibration", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen independent-calibration manifest")
    forbidden = set().union(*(literal_seeds(json.loads(p.read_text())) for p in (ROOT / "protocols").glob("plan_validity*.json")))
    result = prepare(json.loads(args.source_manifest.read_text()), json.loads(args.previous_calibration.read_text()),
        load_plan(args.preregistration), args.run_name, args.manifest.resolve().relative_to(ROOT), forbidden)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
