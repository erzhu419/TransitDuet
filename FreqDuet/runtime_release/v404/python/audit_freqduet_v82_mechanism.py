#!/usr/bin/env python3
"""Compact descriptive decomposition of the completed V82 mechanism pilot."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v82_local_global_regularity as pilot


def variance_effect(base, candidate):
    mb, mc = base["mean_headway_s"], candidate["mean_headway_s"]
    vb, vc = base["cv_squared"] * mb ** 2, candidate["cv_squared"] * mc ** 2
    variance = (vc - vb) / mb ** 2
    normalization = vc * (1. / mc ** 2 - 1. / mb ** 2)
    np.testing.assert_allclose(variance + normalization,
        candidate["cv_squared"] - base["cv_squared"], atol=1e-12, rtol=1e-12)
    return dict(variance_at_base_mean=float(variance), mean_normalization=float(normalization))


def local_summary(audits):
    result = {}
    total = sum(g["decisions"] for a in audits for g in a["local"].values())
    for name in ("missing_bilateral", "bilateral_active", "bilateral_inactive"):
        entries = [a["local"][name] for a in audits if name in a["local"]]
        if not entries:
            continue
        n = sum(e["decisions"] for e in entries)
        counts = {m: sum(e["observed"][m] for e in entries) for m in pilot.LOCAL_METRICS}
        result[name] = dict(decisions=n, fraction=n / total,
            ready_coverage=counts["completed_ready_gain_delta"] / n,
            departure_coverage=counts["completed_departure_gain_delta"] / n,
            predicted_nonharm_completed_ready_harm_fraction=sum(
                e["predicted_nonharm_but_completed_ready_harm"] for e in entries) / n,
            means={m: sum(e["means"][m] * e["observed"][m] for e in entries if e["observed"][m]) / counts[m]
                   if counts[m] else None for m in pilot.LOCAL_METRICS})
    return result


def run(root, v81_root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(pilot.ITEMS))]
    original = json.loads((root / "summary/status.json").read_text())
    references = {r["original_v81_index"]: json.loads((v81_root / "pairs" / f"p{r['original_v81_index']}.json").read_text())
                  for r in rows}
    modes = {}
    for mode in ("stochastic", "deterministic"):
        selected = [r for r in rows if r["mode"] == mode]
        contrasts = {}
        for contrast, (left, right) in pilot.projection.CONTRASTS.items():
            parts = []
            days = []
            for day in pilot.DAYS:
                entries = [r for r in selected if r["scenario_seed"] == day]
                values = {}
                for metric in pilot.pilot.METRICS:
                    values[metric] = float(np.mean([references[r["original_v81_index"]]["effects"][contrast][metric] for r in entries]))
                days.append(dict(day=day, effects=values))
            for r in selected:
                a, b = r["audits"][left]["arrival"], r["audits"][right]["arrival"]
                parts.append(dict(**variance_effect(b["stop_direction"], a["stop_direction"]),
                    **{f"{p}_{k}": a[p][k] - b[p][k] for p in ("stop_direction", "four_hour_band")
                       for k in ("within_group_cv_squared", "between_group_cv_squared")}))
            contrasts[contrast] = dict(effects={m: float(np.mean([d["effects"][m] for d in days])) for m in pilot.pilot.METRICS},
                cv_decomposition={k: float(np.mean([p[k] for p in parts])) for k in parts[0]}, days=days)
        modes[mode] = dict(local={label: local_summary([r["audits"][label] for r in selected])
                                 for label in ("base", "raw", "projected")}, contrasts=contrasts)
    for seed, by_mode in original["by_seed"].items():
        for mode, by_label in by_mode.items():
            for label, audit in by_label.items():
                mean = np.mean([r["audits"][label]["arrival"]["stop_direction"]["cv_squared"] for r in rows
                                if (r["train_seed"], r["mode"]) == (int(seed), mode)])
                np.testing.assert_allclose(mean, audit["arrival"]["stop_direction"]["cv_squared"], rtol=0, atol=1e-12)
    pilot.pilot.write_json(root / "mechanism_summary/status.json", dict(status="v82_mechanism_summary_complete",
        modes=modes, days=pilot.DAYS, original_cv_means_reproduced=True, new_rollouts=0,
        scope="four consumed demand days; descriptive only, not another gate or causal decomposition",
        main_policy_changed=False))
    print("V82_MECHANISM_SUMMARY_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v82-root", type=Path, required=True)
    parser.add_argument("--v81-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v82_mechanism.py", "-v"], check=True)
    run(args.v82_root, args.v81_root)
