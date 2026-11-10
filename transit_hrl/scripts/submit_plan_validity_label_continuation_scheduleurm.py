"""Prepare paired label-continuation training and frozen one-step evaluation."""

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


def prepare(source, plan, run_name, manifest_path):
    cells = deepcopy(source["cells"])
    registered = list(seed_cells(plan))
    forbidden = {s for c in registered for role in c["seed_roles"].values() for s in role}
    forbidden.update(c["optimizer_seed"] for c in registered)
    for cell in cells:
        for role in ("fit_prefix_seeds", "held_prefix_seeds", "exposed_held_prefix_seeds", "evaluation_seeds", "previous_evaluation_seeds"):
            forbidden.update(cell[role])
    forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
        for c in registered for s in c["seed_roles"]["train_seeds"]
        for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    jobs = {phase: [] for phase in ("precheck", "sample", "fit", "probe")}
    for index, cell in enumerate(cells):
        cell["revealed_diagnostic_seeds"] = cell["evaluation_seeds"]
        cell["evaluation_seeds"] = list(range(cell["optimizer_seed"] + 50000, cell["optimizer_seed"] + 50008))
        if set(cell["evaluation_seeds"]) & forbidden:
            raise ValueError("new one-step paths overlap exposed/reserved/effective seeds")
        forbidden.update(cell["evaluation_seeds"])
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{cell['optimizer_seed']}"
        for phase, shards in (("precheck", [-1]), ("sample", [0, 1]), ("fit", [-1]), ("probe", [0, 1])):
            for shard in shards:
                directory = ROOT / cell["output_directory"] / phase
                if shard >= 0:
                    directory /= f"shard{shard}"
                cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_label_continuation.py", "--manifest", str(manifest_path),
                    "--cell-index", str(index), "--phase", phase, "--shard-index", str(shard)]
                job = deepcopy(source["jobs"]["precheck"][index])
                job.update(project="Freq-HRL-LabelContinuation", description=f"Paired label {phase} {cell['task']} {cell['optimizer_seed']} shard{shard}",
                    signature=f"Freq-HRL/label-continuation/{run_name}/{phase}/{cell['task']}/{cell['optimizer_seed']}/shard{shard}",
                    cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd), result_dir=str(directory / "public"),
                    local_result_dir=str(directory / "public"), wait_for_files=[str(ROOT / manifest_path)])
                jobs[phase].append(job)
    return {"protocol_version": "paired_candidate_state_label_continuation_v1", "run_name": run_name,
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "prefix_policy": "frozen_phase_checks_clock", "label_continuations": ["original_history", "frozen_phase_checks_clock"],
        "probes_per_path": 3, "shards_per_root": 2, "training_paths_per_shard": 16, "evaluation_paths_per_shard": 4,
        "fitting_futures": 1, "independent_B_futures": 4, "evaluation_A_futures": 4, "evaluation_B_futures": 4,
        "effective_df": 8, "threshold_quantile": 0.75, "feature_count": 41, "fitting_pairs_per_head": 96,
        "threshold_source": "training_A_leave_prefix_out_fixed_quantile_B_never_calibrates",
        "sampling": "617_bins_839_uniform_candidate_optional_clock_not_R_C_intersection",
        "training_future_tag": 1003, "evaluation_future_tag": 1004,
        "future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,step,replica_index,phase_tag])",
        "precheck_environment_step_upper_bound": 76800,
        "training_environment_steps": 11212800, "probe_environment_steps": 4185600,
        "single_step_primary": "both_heads_on_fixed_candidate_prefix_and_candidate_continuation_W_MSE_and_W_tail_G_gains",
        "single_step_secondary": "reference_continuation_train_A_independent_B_prefix_oof_and_fresh_prefix_A_B",
        "episode_followup": "not_scheduled_requires_separate_readout_decision_no_automatic_candidate_promotion",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen label-continuation manifest")
    manifest = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), args.run_name,
        args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(manifest["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
