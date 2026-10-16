#!/usr/bin/env python3
"""Separate residual scale and normalized alignment using Stage275 sums only."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_alignment_localization_stage275 as source
from freq_hrl.domains.transit import native_alignment_scale as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_alignment_scale_stage276_v1"
SOURCE_RUN = "native_transit_alignment_localization_stage275_20261016_r1"
KEYS, VIEWS = source.KEYS, source.VIEWS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arm": source.ARM, "penalty": source.PENALTY,
        "dominant_prefixes": [list(p) for p in learning.DOMINANT_PREFIXES], "dominant_probes": list(learning.DOMINANT_PROBES),
        "selection": "Stage275_mixed_pooled_paired_top3_and_probe8_9_fixed_before_this_analysis; no_reselection_per_view_bank_or_future; metadata_diagnosis_only",
        "moments": "pool_cached_weighted_sums_then_normalize_by_actual_population_units; D=sqrt_E_delta2; R=sqrt_E_raw_residual2; rho=E_delta_times_raw_residual_over_D_R; uncentered_weighted_cosine_not_Pearson_or_average_of_row_cosines",
        "identity": "global_alignment_contribution=2_mass_D_R_rho; exact_excluded_minus_held_product_decomposition_averaged_over_all24_factor_replacement_orders; report_population_mass_D_R_and_rho_effects_separately; not_causal_attribution",
        "groups": "all_and_dominant_other_prefix_and_dominant_other_probe_marginals_plus_disjoint2x2; global_denominators_preserved_for_additivity; empty_or_zero_energy_population_has_undefined_rho_and_no_product_effect_attribution",
        "controls": "reproduce_every_Stage275_primary_component_and_unit_count_all32_cells_views_banks_futures; weighted_marginal_and2x2_contributions_add_to_full_gap; product_reconstruction",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_contexts": 640, "model_probe_rows": 3840,
            "new_fits": 0, "new_model_inferences": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "read_Stage275_scalar_probe_ledgers_only_no_models_predictions_features_or_panels; full_bank_subgroup_report_remote; pull_compact_scalar_report",
        "boundary": "development_selected_groups_and_repeated_contexts_not_new_confirmation_seed_CI_or_policy_gain; residual_RMS_includes_finite16_future_error; historical_raw_reference_conditioning_unchanged; no_gate_or_default_change; Stage253_248_169_closed"}


def read_cells():
    cells = []
    for key in KEYS:
        path = ROOT/"results"/SOURCE_RUN/"report_raw"/("_".join(map(str, key)))/"probes.json"
        rows = json.loads(path.read_text())
        if len(rows) != 120 or len({r["index"] for r in rows}) != 20: raise ValueError("Incomplete saved probe ledger")
        cells.append({"key": list(key), "rows": rows})
    return cells


def report(cells, before):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, replay_error, partition_error, factor_error = {}, 0., 0., 0.
    for group, predicate in groups.items():
        rows = [r for c in cells if predicate(c["key"]) for r in c["rows"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            selected = [r for r in rows if bank is None or r["bank"] == bank]
            result = learning.summarize(selected);output[group][label] = result
            partition_error = max(partition_error, result["partition_max_abs_error"]);factor_error = max(factor_error, result["factorization_max_abs_error"])
            previous = before["groups"][group][label]["totals"]
            for f in learning.source.FUTURES:
                for pair in learning.source.MODES.values():
                    for pop in pair:
                        for field in ("units", *learning.source.source.FIELDS): replay_error = max(replay_error, abs(result["totals"][f][pop][field]-previous[f][pop][field]))
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(len(c["rows"]) for c in cells) != 3840 or replay_error > 1e-10: raise ValueError("Incomplete Stage276 primary component replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"group_metric_max_abs_error": replay_error, "partition_max_abs_error": partition_error, "factorization_max_abs_error": factor_error},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    def compact(group, bank, result):
        if group == "mixed_pooled" or bank == "pooled": return result
        return {**result, "comparisons": {f: {m: {"all": cases["all"]} for m, cases in modes.items()} for f, modes in result["comparisons"].items()}}
    return {**payload, "groups": {g: {b: compact(g, b, s) for b, s in banks.items()} for g, banks in payload["groups"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    cells = read_cells();before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    payload = report(cells, before);path = ROOT/"results"/a.run_name/"report/result.json"
    write_json(raw_directory(path)/"aggregate.json", payload);write_json(path, compact_report(payload));print("ALIGNMENT_SCALE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
