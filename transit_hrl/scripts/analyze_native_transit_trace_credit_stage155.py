#!/usr/bin/env python3
"""Keep corrected return propagation separate from learned dispatch value."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_trace_credit_stage155 as spec
from scripts.analyze_native_transit_dispatch_train_stage151 import matched_means
from scripts.analyze_native_transit_critic_units_stage152 import qualified_critic_diagnostics
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    contract, metrics, means, regimes = matched_means(cells, spec)
    diagnostics = qualified_critic_diagnostics(cells, spec, means)
    for (method, root), cell in cells.items():
        if not (cell["backup_horizon"] == spec.HORIZONS[method] and cell["trace_lambda"] == .9
                and cell["weight_reg_mode"] == "physical_sum"):
            raise ValueError("Unmatched trace-credit learner")
        if method == "retrace8":
            stats = cell["upper_learning_mean"]
            if not (1 <= stats["upper_trace_mass_mean"] <= 8 and 1 <= stats["upper_trace_valid_steps_mean"] <= 8
                    and 0 <= stats["upper_trace_coefficient_mean"] <= .900001
                    and np.isfinite(stats["upper_trace_correction_abs_mean"])
                    and stats["upper_trace_correction_abs_mean"] > 0):
                raise ValueError("Trace backup did not propagate finite corrected credit")

    def delta(a, b, ca, cb):
        return [{"root": root, **{key: means[a, root, ca][key] - means[b, root, cb][key]
            for key in metrics}} for root in spec.ROOTS]

    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "two_root_delayed_credit_development_not_confirmation",
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "worker_preflight_native_steps": sum(c["worker_preflight_native_steps"] for c in cells.values()),
        "retrace_minus_one_step": delta("retrace8", "one_step", "baseline", "baseline"),
        "learned_minus_neutral_upper": {m: delta(m, m, "baseline", "neutral_upper") for m in spec.METHODS},
        "learned_minus_fixed7": {m: delta(m, m, "baseline", "fixed7") for m in spec.METHODS},
        "zero_holding_minus_learned": {m: delta(m, m, "zero_holding", "baseline") for m in spec.METHODS},
        "critic_diagnostics": diagnostics,
        "regime_root_means": [{"method": m, "root": root, "condition": condition, "scenario": scenario, **values}
            for (m, root, condition, scenario), values in regimes.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {(method, root): json.loads((directory / "cells" / method / f"seed_{root}/result.json").read_text())
             for method in spec.METHODS for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items()
                     if key not in {"critic_diagnostics", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()
