#!/usr/bin/env python3
"""Identify archived snapshot mismatches before initializing the caller RNG."""

import argparse
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_native_transit_frozen_prefix_stage287 as spec


def diagnose(run, key):
    with (spec.prefix_path(run, key)/"prefix_refs.pkl").open("rb") as f:
        refs = pickle.load(f)
    saved = refs["snapshots"][8]
    constant = all(s["rng"]["torch"] == saved["rng"]["torch"] for s in refs["snapshots"].values())
    model, checkpoint = spec.native_inputs(key[1])
    try:
        spec.source.shared.episode(key[1], key[2], spec.scene_seed(*key), spec.directory(run)/"diagnostic_raw"/key[0],
            checkpoint, model, 8, reference=SimpleNamespace(prefix=refs["prefix"], snapshot=saved),
            plan_class=spec.NativeTimetableContextAudit)
    except RuntimeError as error:
        tb = error.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename.endswith("native_candidate_value.py") and "self" in tb.tb_frame.f_locals:
                actual = tb.tb_frame.f_locals["self"].snapshot
                fields = [k for k in saved if saved[k] != actual[k]]
                rng = [k for k in saved["rng"] if saved["rng"][k] != actual["rng"][k]]
                if fields != ["rng"] or rng != ["torch"] or not constant:
                    raise
                return {"key": list(key), "different_fields": fields, "different_rng": rng,
                    "archived_torch_state_constant_across_six_probes": constant}
            tb = tb.tb_next
        raise
    raise RuntimeError("Expected uninitialized-worker RNG mismatch was not reproduced")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    cases = [diagnose(a.run_name, k) for k in (("bridge", 397, "low_noise", 1), ("new", 397, "low_noise", 0))]
    result = {"cases": cases, "physical_state_changed": False, "uninitialized_caller_torch_mismatch_confirmed": True}
    spec.write_json(spec.directory(a.run_name)/"diagnostic/result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
