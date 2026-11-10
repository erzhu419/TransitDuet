"""Post-hoc cross-future decision readout; reuse private probes on the server."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_continuation import CONTEXTS, DECISION_CHOICES, cross_future_decisions
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from scripts.analyze_plan_validity_continuation import summarize


def aggregate_choices(paths):
    output = {}
    for block in ("A_to_B", "B_to_A", "all"):
        contexts = {}
        for context in CONTEXTS:
            choices = {name: {key: paired_mean_interval([
                p["blocks"][block][context]["choices"][name][key] for p in paths])
                for key in ("gain_W", "gain_tail", "gain_G", "changed_from_reference", "choose_now")}
                for name in DECISION_CHOICES}
            contexts[context] = {"choices": choices,
                "W_G_choice_disagreement": float(np.mean([
                    p["blocks"][block][context]["W_G_choice_disagreement"] for p in paths])),
                "paired_choice_contrasts": {name: {key: paired_mean_interval([
                    p["blocks"][block][context]["choices"][left][key]
                    - p["blocks"][block][context]["choices"][right][key] for p in paths])
                    for key in ("gain_W", "gain_G")}
                    for name, left, right in (("score_zero_minus_candidate", "score_zero", "candidate"),
                        ("W_sign_minus_score_zero", "W_sign", "score_zero"),
                        ("W_sign_minus_candidate", "W_sign", "candidate"),
                        ("G_sign_minus_W_sign", "G_sign", "W_sign"))}}
        output[block] = contexts
    return output


def read_cache(manifest, *, root=ROOT):
    source = summarize(manifest, root=root)
    if source["status"] != "complete":
        raise ValueError("cross-future readout requires all eight completed frozen shards")
    roots = []
    for cell in manifest["cells"]:
        probes, public_paths = [], []
        for shard in range(manifest["shards_per_root"]):
            directory = root / cell["output_directory"] / f"run/shard{shard}"
            probes.extend(json.loads((directory / "private/probes.json").read_text()))
            public_paths.extend(json.loads((directory / "public/result.json").read_text())["paths"])
        if len(probes) != len(cell["evaluation_seeds"]) * manifest["probes_per_path"] or len({(p["seed"], p["step"]) for p in probes}) != len(probes):
            raise ValueError("private probe roster is incomplete or duplicated")
        for probe in probes:
            futures = [[r["future_seed"] for r in probe["effects"][c]] for c in CONTEXTS]
            if (len(futures[0]) != manifest["conditional_replicates"] or len(set(futures[0])) != len(futures[0])
                or any(f != futures[0] for f in futures[1:])):
                raise ValueError("conditional future pairing differs across contexts")
        paths = []
        for seed, public in zip(cell["evaluation_seeds"], public_paths):
            selected = [p for p in probes if p["seed"] == seed]
            if len(selected) != manifest["probes_per_path"]:
                raise ValueError("path probe count differs from the frozen manifest")
            blocks = cross_future_decisions(selected)
            for context in CONTEXTS:
                for key in ("W", "G"):
                    if not np.isclose(blocks["all"][context]["choices"]["candidate"]["gain_" + key],
                        public["blocks"]["all"]["contexts"][context]["candidate_action_gain_" + key], rtol=1e-10, atol=1e-10):
                        raise ValueError("cache and saved public frozen candidate gains differ")
            paths.append({"seed": seed, "blocks": blocks})
        roots.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
            "path_count": len(paths), "blocks": aggregate_choices(paths),
            "path_all_gains": [{"seed": p["seed"], "contexts": {c: {n: {
                k: p["blocks"]["all"][c]["choices"][n][k] for k in ("gain_W", "gain_G")}
                for n in DECISION_CHOICES} for c in CONTEXTS}} for p in paths]})
    return {"run_name": manifest["run_name"], "status": "complete", "role": "posthoc_cache_diagnostic_not_policy_selection",
        "selection": "sign_of_four_future_mean_at_zero_evaluated_on_other_four_then_swap",
        "score_zero": "same_frozen_candidate_score_with_zero_cut_not_refitted_or_adopted",
        "reference": "frozen_history_action_on_the_same_prefix_under_the_same_continuation",
        "statistical_scope": "eight_path_descriptive_CIs_conditional_on_each_frozen_root_not_new_trained_roots",
        "W_G_sign_selection_is_finite_sample_information_benchmark_not_oracle": True,
        "root_readouts": roots, "new_fits": 0, "new_environment_steps": 0,
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    result = read_cache(json.loads(args.manifest.read_text()))
    result["analysis_cpu_seconds"] = time.process_time() - start_cpu
    result["analysis_wall_seconds"] = time.perf_counter() - start_wall
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "new_fits", "new_environment_steps", "analysis_cpu_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
