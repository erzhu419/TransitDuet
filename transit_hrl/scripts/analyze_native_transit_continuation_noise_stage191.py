#!/usr/bin/env python3
"""Separate conditional action effects from sampled continuation noise."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_continuation_noise_stage191 as spec
from freq_hrl.domains.transit.native_action_value import choice_from_cost
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

TARGETS = ("actual_cost_delta", "wait_delta_min")


def choice(labels, original):
    return choice_from_cost(np.where(labels[:, 1] <= 0, labels[:, 0], np.inf), original)


def signal(values):
    a, b = np.triu_indices(6, 1)
    gaps = values[:, a]-values[:, b]
    variance = gaps.var(axis=0, ddof=1)
    squared_mean = gaps.mean(axis=0)**2
    noise = float(variance.mean())
    power = float((squared_mean-variance/len(values)).mean())
    return {"draws": len(values), "mean_pairwise_gap_variance": noise,
        "mean_label_pairwise_noise_variance": noise/len(values),
        "observed_pairwise_squared_mean": float(squared_mean.mean()),
        "debiased_pairwise_effect_power": power,
        "effect_to_innovation_noise_ratio": power/noise if noise > 0 else None}


def state_panel(cells):
    cells = sorted(cells, key=lambda c: c["draw"])
    first = cells[0]
    if [c["draw"] for c in cells] != list(spec.DRAWS):
        raise ValueError("Conditional variance needs all eight registered independent draws")
    for cell in cells:
        spec.qualify_cell(cell)
        if any(cell[k] != first[k] for k in ("root", "scenario", "scene_seed", "probe_index",
                "original_choice", "frozen_choices", "existing_labels")):
            raise ValueError("Future draws changed state, original labels or sealed policy choices")
    labels = np.asarray([[[l[k] for k in TARGETS] for l in c["labels"]] for c in cells])
    original = first["original_choice"]
    mean = labels.mean(axis=0)
    full = choice(mean, original)
    old = np.asarray([[l[k] for k in TARGETS] for l in first["existing_labels"]])
    old_choice = choice(old, original)
    halves = labels[::2].mean(axis=0), labels[1::2].mean(axis=0)
    selected = [choice(h, original) for h in halves]
    transferred = np.asarray([halves[1][selected[0]], halves[0][selected[1]]]).mean(axis=0)
    a, b = np.triu_indices(6, 1)
    signs = [np.sign(h[a, 0]-h[b, 0]) for h in halves]
    strict = (signs[0] != 0) & (signs[1] != 0)
    return {**{k: first[k] for k in ("root", "scenario", "scene_seed", "probe_index", "original_choice")},
        "signal": {target: signal(labels[..., i]) for i, target in enumerate(TARGETS)},
        "half_choices": selected, "half_choice_agreement": selected[0] == selected[1],
        "half_strict_rank_pairs": int(strict.sum()),
        "half_concordant_rank_pairs": int(((signs[0] == signs[1]) & strict).sum()),
        "cross_half_selected_cost": float(transferred[0]), "cross_half_selected_wait_min": float(transferred[1]),
        "same_half_selected_cost": float(np.mean([h[s, 0] for h, s in zip(halves, selected)])),
        "existing_label_oracle_choice": old_choice,
        "existing_label_oracle_old_cost": float(old[old_choice, 0]),
        "existing_label_oracle_new_mean_cost": float(mean[old_choice, 0]),
        "existing_label_oracle_new_mean_wait_min": float(mean[old_choice, 1]),
        "full_mean_oracle_choice": full, "full_mean_oracle_cost": float(mean[full, 0]),
        "full_mean_oracle_wait_min": float(mean[full, 1]),
        "frozen_choice_outcomes": {name: {"selected": c, "old_cost": float(old[c, 0]),
            "new_mean_cost": float(mean[c, 0]), "old_wait_min": float(old[c, 1]),
            "new_mean_wait_min": float(mean[c, 1])} for name, c in first["frozen_choices"].items()}}


def summary(panels):
    fields = ("cross_half_selected_cost", "cross_half_selected_wait_min", "same_half_selected_cost",
        "existing_label_oracle_old_cost", "existing_label_oracle_new_mean_cost", "existing_label_oracle_new_mean_wait_min",
        "full_mean_oracle_cost", "full_mean_oracle_wait_min")
    return {"states": len(panels), "prefix_scene_groups": len({(p["root"], p["scene_seed"]) for p in panels}),
        **{"mean_"+k: float(np.mean([p[k] for p in panels])) for k in fields},
        "half_choice_agreements": sum(p["half_choice_agreement"] for p in panels),
        "half_strict_rank_pairs": sum(p["half_strict_rank_pairs"] for p in panels),
        "half_concordant_rank_pairs": sum(p["half_concordant_rank_pairs"] for p in panels),
        "signal": {target: {k: float(np.mean([p["signal"][target][k] for p in panels])) for k in
            ("mean_pairwise_gap_variance", "mean_label_pairwise_noise_variance", "debiased_pairwise_effect_power")}
            for target in TARGETS},
        "frozen_choice_outcomes": {name: {"changed_choices": sum(p["frozen_choice_outcomes"][name]["selected"] != p["original_choice"] for p in panels),
            **{"mean_"+k: float(np.mean([p["frozen_choice_outcomes"][name][k] for p in panels]))
                for k in ("old_cost", "new_mean_cost", "old_wait_min", "new_mean_wait_min")}}
            for name in panels[0]["frozen_choice_outcomes"]}}


def summarize(cells):
    if set(cells) != set(spec.cells()):
        raise ValueError("Missing or mixed registered continuation-noise cells")
    panels = [state_panel([cells[root, scenario, probe, draw] for draw in spec.DRAWS])
        for root in spec.ROOTS for scenario in spec.SCENARIOS for probe in spec.PROBES]
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "native_episodes": 1920, "native_clock_ticks": 1920*61380, "training_updates": 0,
        "original_source_reproductions": 240, "exact_resampled_identities": 240,
        "prefix_state_rng_matches": 1680, "overall": summary(panels),
        "by_root": {str(r): summary([p for p in panels if p["root"] == r]) for r in spec.ROOTS},
        "by_probe": {str(i): summary([p for p in panels if p["probe_index"] == i]) for i in spec.PROBES},
        "state_panels": panels, "third_scene_screen_eligible": False, "native_deployment_eligible": False,
        "oracle_role": "outcome-based_diagnostics_only_not_causal_learned_choices_or_performance_claims"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {k: json.loads((raw_directory(spec.result_path(directory, *k))/"result.json").read_text()) for k in spec.cells()}
    result = summarize(cells)
    write_json(args.output, result)
    print(json.dumps({"overall": result["overall"], "by_root": result["by_root"]}, indent=2))
    print("NATIVE_CONTINUATION_NOISE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
