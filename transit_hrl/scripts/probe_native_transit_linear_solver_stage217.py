#!/usr/bin/env python3
"""Compare LP formulations on the first fold that raised HiGHS Unknown."""

import argparse
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_linear_floor_stage217 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank, masses, registered, wait_registered = spec.load_data()
    fold = registered["folds"][0]
    _, ridge, glm = spec.load_frozen_fold(fold, registered, wait_registered)
    rows = bank["training_data"]
    batch = spec.wait_source.neural.batches(rows)
    training = spec.previous.masks(rows, fold, {spec.state_key(r) for r in registered["mean_states"]})["training_all"]
    cost = spec.source.base.centered(batch[0][..., :98], rows, ridge)
    wait = spec.wait_source.source.feasibility.design(batch[0], [r["original_choice"] for r in rows], glm["center"], glm["scale"])
    panels = spec.wait_source.source.diagnostic.load_panels()
    probes = []
    for partition in spec.PARTITIONS:
        for name, (d, margins) in spec.stable_constraints(rows, panels, cost, wait, training, partition).items():
            norm = abs(d).max(axis=1)
            norm[norm == 0] = 1.
            u, singular, vt = np.linalg.svd(d, full_matrices=False)
            cutoff = np.finfo(d.dtype).eps*max(d.shape)*singular[0]
            rank = int(np.sum(singular > cutoff))
            outcomes = {}
            for formulation in ("L1", "free", "scaled_free"):
                scaled = formulation == "scaled_free"
                a, b = (d/norm[:, None], margins/norm) if scaled else (d, margins)
                objective = np.ones(2*d.shape[1]) if formulation == "L1" else np.zeros(d.shape[1])
                inequalities = np.c_[a, -a] if formulation == "L1" else a
                method = "highs" if formulation == "L1" else "highs-ds"
                result = linprog(objective, A_ub=-inequalities, b_ub=-b, bounds=(0., None) if formulation == "L1" else (None, None),
                    method=method, options={"threads": 1})
                outcome = {"method": method, "status": int(result.status), "message": result.message, "iterations": int(result.nit)}
                if result.status == 0:
                    beta = result.x[:d.shape[1]]-result.x[d.shape[1]:] if formulation == "L1" else result.x
                    outcome.update(violation=float(np.maximum(margins-d@beta, 0).max()), coefficient_l1=float(abs(beta).sum()))
                outcomes[formulation] = outcome
            result = linprog(np.zeros(rank), A_ub=-u[:, :rank], b_ub=-margins, bounds=(None, None),
                method="highs-ds", options={"threads": 1, "presolve": False})
            outcomes["numerical_basis"] = {"status": int(result.status), "message": result.message,
                "iterations": int(result.nit), "rank": rank, "singular_cutoff": float(cutoff)}
            if result.status == 0:
                beta = vt[:rank].T@(result.x/singular[:rank])
                outcomes["numerical_basis"].update(violation=float(np.maximum(margins-d@beta, 0).max()), coefficient_l1=float(abs(beta).sum()))
            probes.append({"partition": partition, "kind": name, "shape": list(d.shape),
                "zero_strict_rows": int(np.sum((np.max(abs(d), axis=1) == 0) & (margins > 0))),
                "smallest_nonzero_coefficient": float(abs(d[d != 0]).min()), "largest_coefficient": float(abs(d).max()),
                "outcomes": outcomes})
    write_json(args.output, {"protocol": "native_transit_linear_solver_stage217_v1", "fold": fold,
        "scipy_version": scipy.__version__, "linear_programs": 16, "new_native_episodes": 0, "probes": probes})
    print("NATIVE_LINEAR_SOLVER_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
