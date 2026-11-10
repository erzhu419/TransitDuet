"""Reuse saved fold scores; freeze OOF/full A-only cuts before reading B."""

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
from freq_hrl.experiments.plan_validity_decision_calibration import calibrate_cut, score_source_readout
from freq_hrl.experiments.plan_validity_label_continuation import CONTINUATIONS, HEADS, validate_banks
from freq_hrl.experiments.plan_validity_phase_support import predict
from freq_hrl.experiments.plan_validity_window_coverage import values
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule


def run_cell(cell, manifest, directory, publish):
    with measured_phase("read_saved_training_scores_and_effects"):
        cached = ROOT / cell["calibration_directory"]
        saved = json.loads((cached / "public/result.json").read_text())
        if (saved["status"] != "complete" or saved["run_name"] != manifest["source_run_name"]
            or saved["task"] != cell["task"] or saved["optimizer_seed"] != cell["optimizer_seed"]):
            raise ValueError("saved calibration source differs")
        recovered = json.loads((cached / "private/reconstructed_oof.json").read_text())
        old_cuts = json.loads((cached / "private/frozen_cuts.json").read_text())
        fit = ROOT / cell["fit_directory"]
        rules = {k: TimingRule(**v) for k, v in json.loads((fit / "private/frozen_rules.json").read_text()).items()}
        banks = {c: [] for c in CONTINUATIONS}
        for shard in (0, 1):
            sample = ROOT / cell["sample_directory"] / f"shard{shard}"
            public = json.loads((sample / "public/result.json").read_text())
            if (public["status"] != "complete" or public["run_name"] != manifest["label_run_name"]
                or public["optimizer_seed"] != cell["optimizer_seed"]
                or public["sampled_seeds"] != cell["fit_prefix_seeds"][16 * shard:16 * (shard + 1)]):
                raise ValueError("training sample roster differs")
            raw = json.loads((sample / "private/label_banks.json").read_text())
            for c in CONTINUATIONS:
                banks[c].extend(raw[c])
        validate_banks(banks, seeds=cell["fit_prefix_seeds"], probes=3, a=1, b=4)
        rows = banks["candidate"]
        if recovered["state_roster"] != [[r["seed"], r["opportunity_step"]] for r in rows]:
            raise ValueError("saved OOF state order differs")
        oof = {name: np.asarray(recovered["scores"][name]) for name in HEADS}
    with measured_phase("freeze_both_calibration_sources_using_A_only"):
        w, g = (values(rows, k, "A").mean(axis=1) for k in ("W", "G"))
        selected = {source: {name: calibrate_cut(scores, w, g, rules[name].threshold)
            for name, scores in score_set.items()} for source, score_set in (
                ("oof_scores", oof), ("frozen_full_scores", {name: predict(rule, rows) for name, rule in rules.items()}))}
        for name in HEADS:
            if (not np.isclose(np.quantile(oof[name], manifest["original_quantile"]), rules[name].threshold, rtol=1e-12, atol=1e-12)
                or selected["oof_scores"][name] != old_cuts[name]):
                raise RuntimeError("saved-score OOF calibration does not reproduce frozen selection")
        (directory / "private/frozen_cuts.json").write_text(json.dumps(_json_ready(selected), indent=2) + "\n")
    publish({"stage": "all_source_cuts_frozen_before_B", "selected": selected})
    with measured_phase("paired_B_score_source_readout"):
        readout = score_source_readout(rows, oof, rules, selected)
    return {"stage": "score_source_complete", "selected": selected, "readout": readout,
        "states": len(rows), "training_prefixes": len(cell["fit_prefix_seeds"]),
        "scope": "known_training_prefix_development_B_full_A_is_in_sample_not_confirmation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve measured score-source attempts")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "status": "running", "new_frozen_full_model_fits": 0,
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])}", flush=True)

    try:
        with ledger.recording("score_source"), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            summary = run_cell(cell, manifest, directory, publish)
        counts = ledger.summary()["counts"]
        if any(counts.get(k, 0) for k in ("ridge_fit_calls", "environment_steps", "controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("score-source control exceeded its no-fit/no-simulation budget")
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
