"""Reconstruct fold scores, freeze A-only cuts, then read training-prefix B."""

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_decision_calibration import calibrate_cut, decision_readout, reconstruct_oof
from freq_hrl.experiments.plan_validity_label_continuation import CONTINUATIONS, HEADS, validate_banks
from freq_hrl.experiments.plan_validity_window_coverage import error, values
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule


def run_cell(cell, manifest, directory, publish):
    with measured_phase("read_training_cache_and_frozen_heads"):
        fit = ROOT / cell["fit_directory"]
        saved = json.loads((fit / "public/result.json").read_text())
        if (saved["status"] != "complete" or saved["run_name"] != manifest["source_run_name"]
            or saved["task"] != cell["task"] or saved["optimizer_seed"] != cell["optimizer_seed"]):
            raise ValueError("frozen head source differs")
        rules = {k: TimingRule(**v) for k, v in json.loads((fit / "private/frozen_rules.json").read_text()).items()}
        banks = {c: [] for c in CONTINUATIONS}
        for shard in (0, 1):
            sample = ROOT / cell["sample_directory"] / f"shard{shard}"
            row = json.loads((sample / "public/result.json").read_text())
            if (row["status"] != "complete" or row["run_name"] != manifest["source_run_name"]
                or row["optimizer_seed"] != cell["optimizer_seed"] or row["sampled_seeds"] != cell["fit_prefix_seeds"][16 * shard:16 * (shard + 1)]):
                raise ValueError("training sample roster differs")
            raw = json.loads((sample / "private/label_banks.json").read_text())
            for c in CONTINUATIONS:
                banks[c].extend(raw[c])
        validate_banks(banks, seeds=cell["fit_prefix_seeds"], probes=3, a=1, b=4)
    with measured_phase("reconstruct_training_A_fold_scores"):
        oof = reconstruct_oof(banks, rules, effective_df=manifest["effective_df"], quantile=manifest["original_quantile"])
        (directory / "private/reconstructed_oof.json").write_text(json.dumps(_json_ready({
            "state_roster": [[r["seed"], r["opportunity_step"]] for r in banks["candidate"]], "scores": oof})) + "\n")
    with measured_phase("select_using_common_C0_A_only"):
        target = banks["candidate"]
        w, g = values(target, "W", "A").mean(axis=1), values(target, "G", "A").mean(axis=1)
        selected = {name: calibrate_cut(oof[name], w, g, rules[name].threshold) for name in HEADS}
        (directory / "private/frozen_cuts.json").write_text(json.dumps(_json_ready(selected), indent=2) + "\n")
    publish({"stage": "both_cuts_frozen_before_B_readout", "selected": selected})
    with measured_phase("independent_future_B_readout"):
        for name in HEADS:
            reproduced = error(oof[name], target, target="W", block="B", independent=True)["mse"]
            expected = saved["training_readout"][name]["cross_target_B"]["candidate"]["prefix_oof"]["mse"]
            if not np.isclose(reproduced, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError("reconstructed OOF scores do not reproduce saved B error")
        readout = decision_readout(target, oof, rules, selected)
    return {"stage": "decision_calibration_complete", "selected": selected, "readout": readout,
        "training_prefixes": len(cell["fit_prefix_seeds"]), "states": len(target),
        "scope": "B_independent_future_noise_on_known_training_prefixes_not_prefix_held_out_policy_validation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve measured calibration attempts")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "status": "running", "new_frozen_full_model_fits": 0, "new_environment_steps": 0,
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage',result['status'])}", flush=True)

    try:
        with ledger.recording("decision_calibration"), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_training_only_inputs"})
            summary = run_cell(cell, manifest, directory, publish)
        counts = ledger.summary()["counts"]
        if (counts.get("ridge_fit_calls", 0) != 2 * len(cell["fit_prefix_seeds"])
            or any(counts.get(k, 0) for k in ("environment_steps", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("calibration exceeded its reconstruction-only budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    cost = ledger.summary()
    result.update(measured_cost=cost, new_environment_steps=cost["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
