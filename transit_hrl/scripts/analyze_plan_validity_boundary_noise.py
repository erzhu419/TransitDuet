"""Fixed-prediction suffix MC uncertainty and prefix influence; no fitting."""

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
from freq_hrl.experiments.plan_validity_boundary_parent_scale import verify_prior
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_tail import load_cache


def paired_parts(p, q, tail):
    difference = p - q
    square = float(np.mean(p ** 2 - q ** 2))
    align = [-2 * float(np.mean(difference * block.mean(axis=1))) for block in (tail[:, :2], tail[:, 2:])]
    # The same suffix labels score both predictors; their squared-label term cancels.
    per_window_variance = 4 * difference ** 2 * tail[:, 2:].var(axis=1, ddof=1) / 2
    variance_sum = float(per_window_variance.sum())
    return {"squared_prediction_term": square, "A_alignment_term": align[0], "B_alignment_term": align[1],
        "A_delta": square + align[0], "B_delta": square + align[1],
        "B_suffix_mc_se": float(np.sqrt(variance_sum) / len(p)),
        "B_replica_deltas": [square - 2 * float(np.mean(difference * tail[:, k])) for k in (2, 3)],
        "largest_window_share_of_B_mc_variance": float(per_window_variance.max() / variance_sum) if variance_sum > 0 else None}


def comparison(p, q, data):
    p, q = np.asarray(p), np.asarray(q)
    result = paired_parts(p, q, data["tail"])
    prefixes = []
    for seed in np.unique(data["groups"]):
        mask = data["groups"] == seed
        row = paired_parts(p[mask], q[mask], data["tail"][mask])
        excluded = paired_parts(p[~mask], q[~mask], data["tail"][~mask])
        prefixes.append({"seed": int(seed), "A_delta": row["A_delta"], "B_delta": row["B_delta"],
            "B_suffix_mc_se": row["B_suffix_mc_se"], "B_delta_without_prefix": excluded["B_delta"]})
    means = np.asarray([r["B_delta"] for r in prefixes])
    observed = float(means.var(ddof=1))
    suffix = float(np.mean([r["B_suffix_mc_se"] ** 2 for r in prefixes]))
    benefit = float(-means[means < 0].sum())
    best = float(-means.min()) if benefit > 0 else 0.
    result.update(prefix_reports=prefixes, B_improving_prefixes=int(np.sum(means < 0)),
        B_prefix_variance_observed=observed, B_prefix_suffix_mc_variance=suffix,
        B_prefix_dispersion_corrected=observed - suffix,
        largest_beneficial_prefix_share_of_gross_benefit=best / benefit if benefit > 0 else None,
        largest_beneficial_prefix_share_of_net_benefit=best / -float(means.sum()) if means.sum() < 0 else None,
        B_window_halves=[paired_parts(p[m], q[m], data["tail"][m])
            for m in (data["windows"] < 2, data["windows"] >= 2)])
    return result


def analyze_cell(cell, source):
    directory = ROOT / cell["output_directory"]
    saved = json.loads((directory / "public/result.json").read_text())
    if (saved["run_name"], saved["task"], saved["optimizer_seed"], saved["status"], saved["stage"]) != (
        source["run_name"], cell["task"], cell["optimizer_seed"], "complete", "boundary_parent_scale_complete"):
        raise ValueError("fixed parent-scale predictions must be complete and matching")
    previous = json.loads((directory / "private/models_and_predictions.json").read_text())
    rows = load_cache(cell, source)
    data = {"groups": np.asarray([r["seed"] for r in rows for w in r["windows"]]),
        "windows": np.asarray([j for r in rows for j, w in enumerate(r["windows"])]),
        "tail": np.asarray([[d["tail"] for d in w["suffix_draws"]] for r in rows for w in r["windows"]])}
    verify_prior(data, previous, saved)
    panels = {}
    for panel, key in (("training", "training_predictions"), ("prefix_oof", "prefix_oof_predictions")):
        scores = previous[key]
        panels[panel] = {baseline: comparison(scores["boundary_delta"], scores[baseline], data)
            for baseline in ("zero", "constant", "causal_t")}
    return {"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "prefixes": len(np.unique(data["groups"])), "windows": len(data["tail"]),
        "frozen_losses_reproduced": True, "panels": panels}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve completed diagnostic attempts")
    (directory / "public").mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "status": "running", "cells": [],
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish():
        result["measured_cost"] = ledger.summary()
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")

    try:
        with ledger.recording("fixed_boundary_noise"), single_thread_runtime() as profile:
            if profile != source["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen source")
            result["hardware_profile"] = profile
            for cell in source["cells"]:
                with measured_phase("read_cache_and_derive_moments"):
                    result["cells"].append(analyze_cell(cell, source))
                publish()
                print(f"{cell['task']} {cell['optimizer_seed']}: fixed-noise readout complete", flush=True)
        if ledger.summary()["counts"]:
            raise RuntimeError("read-only noise attribution unexpectedly performed counted work")
        result.update(status="complete", stage="fixed_boundary_noise_complete", new_fits=0, new_environment_steps=0,
            scope="posthoc_fixed_model_development_diagnostic_not_new_confirmation",
            uncertainty_scope="B_suffix_only_given_all_A_fits_and_realized_windows_no_CI_or_AB_gap_test",
            influence_scope="readout_deletion_without_refitting_not_new_prefix_or_policy_inference")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    result["peak_process_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    publish()
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
