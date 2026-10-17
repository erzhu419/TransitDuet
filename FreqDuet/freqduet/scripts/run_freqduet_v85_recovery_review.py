#!/usr/bin/env python3
"""Separate V85 action-response effects from fixed-action regime differences."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_freqduet_v85_feedback_recovery as recovery


repeated, pilot = recovery.repeated, recovery.pilot
MODES = (*recovery.propagation.MODES, "recover_900s")
METRICS = (*repeated.RESPONSE_METRICS, "journey_min")


def paired_contrasts(values):
    matrices = {mode: np.asarray(x, dtype=float) for mode, x in values.items()}
    responses = {mode: x[:, 1] - x[:, 0] for mode, x in matrices.items()}
    levels = {mode: matrices["recover_900s"] - x for mode, x in matrices.items() if mode != "recover_900s"}
    differences = {mode: responses["recover_900s"] - responses[mode] for mode in levels}
    for mode in levels:
        np.testing.assert_allclose(differences[mode], levels[mode][:, 1] - levels[mode][:, 0], atol=1e-7, rtol=0)
    return (dict(response={mode: repeated.describe(x) for mode, x in responses.items()},
                 fixed_action={mode: {str(a): repeated.describe(x[:, i]) for i, a in enumerate((0, 45))}
                               for mode, x in levels.items()},
                 response_difference={mode: repeated.describe(x) for mode, x in differences.items()}))


def count_stats(stats):
    return dict(prefixes=len(stats), mean=float(np.mean([x["mean"] for x in stats])),
        median_abs_mean=float(np.median([abs(x["mean"]) for x in stats])),
        median_future_sd=float(np.median([x["future_sd"] for x in stats])),
        ci_negative=sum(x["ci95"][1] < 0 for x in stats), ci_positive=sum(x["ci95"][0] > 0 for x in stats),
        mean_above_sd=sum(abs(x["mean"]) > x["future_sd"] for x in stats),
        all_futures_same_sign=sum(x["negative"] == x["n"] or x["positive"] == x["n"] for x in stats))


def review(root, v84_root):
    original = json.loads((root / "summary/status.json").read_text())
    metadata = json.loads((v84_root / "summary/status.json").read_text())["contexts"]
    arms, phase = {}, []
    for ri in range(len(repeated.REFERENCES)):
        for mode in MODES:
            for a in (0, 6):
                index = recovery.ITEMS.index((ri, a)) if mode == "recover_900s" else repeated.ITEMS.index((ri, mode, a))
                base = root if mode == "recover_900s" else v84_root
                row = json.loads((base / "arms" / f"a{index}.json").read_text())
                arms[ri, mode, a] = row
                if mode == "recover_900s" and row["recovery"] is not None:
                    phase.append(row["recovery"])
    contexts = []
    for old in original["contexts"]:
        ci = old["context_index"]
        refs = [ri for ri, (context, _) in enumerate(repeated.REFERENCES) if context == ci]
        effects = {}
        for metric in METRICS:
            values = {mode: [[arms[ri, mode, a]["metrics"][metric] for a in (0, 6)] for ri in refs] for mode in MODES}
            effects[metric] = paired_contrasts(values)
            for mode in MODES:
                np.testing.assert_allclose(effects[metric]["response"][mode]["mean"],
                    old["effects_45_minus_0"][mode][metric]["mean"], atol=1e-10, rtol=0)
        stages = {}
        for stage in ("departure", "arrival_h1", "arrival_h3", "arrival_h5"):
            values = {mode: [[arms[ri, mode, a]["witness"][stage]["loss"] if arms[ri, mode, a]["witness"][stage]
                             else None for a in (0, 6)] for ri in refs] for mode in MODES}
            complete = all(x is not None for matrix in values.values() for row in matrix for x in row)
            stages[stage] = paired_contrasts(values) if complete else None
        contexts.append(dict(context_index=ci, seed=metadata[ci]["seed"], day=metadata[ci]["day"],
            stratum=old["stratum"], effects=effects, local_pair_losses=stages))
    counts = {metric: dict(
        response={mode: count_stats([c["effects"][metric]["response"][mode] for c in contexts]) for mode in MODES},
        fixed_action={mode: {str(a): count_stats([c["effects"][metric]["fixed_action"][mode][str(a)] for c in contexts])
                            for a in (0, 45)} for mode in recovery.propagation.MODES},
        response_difference={mode: count_stats([c["effects"][metric]["response_difference"][mode] for c in contexts])
                             for mode in recovery.propagation.MODES}) for metric in METRICS}
    result = dict(status="v85_recovery_review_complete", counts=counts, contexts=contexts,
        phase=dict(executed_arms=len(phase), recorded_mismatches=sum(p["recorded_mismatches"] for p in phase),
            release_delay_s=[min(p["first_recovered_decision_s"] - p["release_time_s"] for p in phase),
                             max(p["first_recovered_decision_s"] - p["release_time_s"] for p in phase)],
            recorded_decisions=[min(p["recorded_decisions"] for p in phase), max(p["recorded_decisions"] for p in phase)]),
        scope="24 consumed prefixes, eight futures each; conditional descriptive intervals, no pooled independent-prefix test",
        timing_limit="full-day endpoints identify regime sensitivity, not the time when passenger cost accrued",
        actors_updated=False, main_policy_changed=False, controller_fits=0)
    pilot.write_json(root / "review/full.json", result)
    print("V85_REVIEW_COUNTS " + json.dumps(dict(counts=counts, phase=result["phase"])), flush=True)
    for c in contexts:
        compact = {m: {kind: ({mode: [x["mean"], x["ci95"], x["future_sd"], x["negative"], x["positive"]]
                               for mode, x in e[kind].items()} if kind != "fixed_action" else
                             {mode: {a: [x["mean"], x["ci95"]] for a, x in rows.items()} for mode, rows in e[kind].items()})
                       for kind in ("response", "fixed_action", "response_difference")}
                   for m, e in c["effects"].items() if m in repeated.RESPONSE_METRICS}
        print("V85_REVIEW_CONTEXT " + json.dumps(dict(context_index=c["context_index"], seed=c["seed"],
            day=c["day"], stratum=c["stratum"], effects=compact)), flush=True)
    pilot.write_json(root / "review_ready/done.json", dict(status="compact_review_complete"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v84-root", type=Path, required=True)
    args = parser.parse_args()
    review(args.out_root, args.v84_root)


if __name__ == "__main__":
    main()
