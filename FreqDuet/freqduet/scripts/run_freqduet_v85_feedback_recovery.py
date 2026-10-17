#!/usr/bin/env python3
"""Locate lower feedback recovery without fitting or promoting a controller."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v84_propagation_repeats as repeated


pilot, propagation = repeated.pilot, repeated.propagation
ITEMS = list(itertools.product(range(len(repeated.REFERENCES)), (0, 6)))
RECOVERY_S = 900.


def release_due(fired, now, target):
    return fired and now >= target + RECOVERY_S


def signal_counts(contexts, mode, metric):
    effects = [c["effects"][mode]["projected_vs_base"][metric] for c in contexts]
    noise = [c["action_noise"][mode][metric] for c in contexts]
    ratios = [n["paired_sd"][a] / n["independent_sd"][a]
              for c, n in zip(contexts, noise) for a in range(7)
              if a != c["native_index"] and n["independent_sd"][a] > 0]
    return dict(prefixes=len(contexts), alternatives=6 * len(contexts),
        weighted_mean_above_future_sd=sum(abs(s["mean"]) > s["future_sd"] for s in effects),
        weighted_ci_below_zero=sum(s["ci95"][1] < 0 for s in effects),
        weighted_ci_above_zero=sum(s["ci95"][0] > 0 for s in effects),
        action_mean_above_paired_sd=sum(abs(n["mean_deltas"][a]) > n["paired_sd"][a]
            for c, n in zip(contexts, noise) for a in range(7) if a != c["native_index"]),
        paired_to_independent_sd_median=float(np.median(ratios)) if ratios else None,
        old_mean_sign_repeated=sum(c["original_v83_effects"][mode]["projected_vs_base"][metric] * s["mean"] > 0
                                  for c, s in zip(contexts, effects)))


def audit(root, v84_root):
    summary = json.loads((v84_root / "summary/status.json").read_text())
    contexts = summary["contexts"]
    counts = {mode: {m: signal_counts(contexts, mode, m) for m in repeated.RESPONSE_METRICS}
              for mode in propagation.MODES}
    feedback = {m: dict(mean_positive=sum(c["feedback"]["projected_vs_base"][m]["mean"] > 0 for c in contexts),
        ci_positive=sum(c["feedback"]["projected_vs_base"][m]["ci95"][0] > 0 for c in contexts),
        ci_negative=sum(c["feedback"]["projected_vs_base"][m]["ci95"][1] < 0 for c in contexts))
        for m in repeated.RESPONSE_METRICS}
    result = dict(status="v84_signal_audit_complete", signal=counts, feedback_closed_minus_recorded=feedback,
        interval_unit=summary["interval_unit"], inference="descriptive, not multiplicity-adjusted confirmation",
        actors_updated=False, main_policy_changed=False, controller_fits=0)
    pilot.write_json(root / "audit/status.json", result)
    print("V84_SIGNAL_AUDIT " + json.dumps(result), flush=True)


class RecoveryTrace(repeated.FutureTrace):
    def __init__(self, ref, action, uniforms):
        super().__init__(ref, ref["stratum"], "recorded_lower", action, ref["future_seed"], uniforms)
        self.release_time, self.release_observed = self.target["time_s"] + RECOVERY_S, None

    def __call__(self, policy, states, features, theta):
        if release_due(self.fired, self.runner.env.current_time, self.target["time_s"]):
            self.mode = "closed_loop"
            if self.release_observed is None:
                self.release_observed = float(self.runner.env.current_time)
        return super().__call__(policy, states, features, theta)

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        later = self.result["commands"][self.target["ordinal"] + 1:]
        early = [r for r in later if r[0] < self.release_time]
        if any(r[-1] != self.tape[tuple(r[1:4])] for r in early):
            raise AssertionError("lower feedback resumed before the registered release time")
        if self.release_observed is None or not early:
            raise AssertionError("both recorded and recovered phases must actually occur")
        self.result["recovery"] = dict(release_time_s=self.release_time, first_recovered_decision_s=self.release_observed,
            recorded_decisions=len(early), recorded_mismatches=0,
            later_lower_action_changes=sum(r[-1] != self.tape[tuple(r[1:4])] for r in later if r[0] >= self.release_time))


def arm(root, v84_root, index, check_native=False):
    ri, action = ITEMS[index]
    path = root / ("native_checks" if check_native else "arms") / f"a{index}.json"
    if path.exists():
        return
    ref = json.loads((v84_root / "references" / f"r{ri}.json").read_text())
    native = ref["contexts"][ref["stratum"]]["native_index"]
    if check_native:
        action = native
    recovery, alignment = None, None
    if action == native and not check_native:
        metrics, witness = ref["metrics"], ref["witnesses"][ref["stratum"]]
    else:
        uniforms = pilot.event.EventUniforms(ref["sampler"])
        observer = RecoveryTrace(ref, action, uniforms)
        row, events, _, _ = pilot.rollout(root, ref["seed"], ref["day"], "stochastic", ref["sampler"], ref["theta"],
            f"v85_{index}" + ("_native" if check_native else ""),
            probability_rule=observer, trace_observer=observer, event_uniforms=uniforms)
        metrics, witness = {m: row[m] for m in pilot.METRICS}, observer.result["witnesses"][ref["stratum"]]
        recovery = observer.result["recovery"]
        alignment = pilot.event.alignment({tuple(k): v for k, v in ref["events"]}, events)
        if check_native:
            pilot.joint.assert_same_state(ref["metrics"], metrics, "recovered_native_metrics")
            for key in ("commands", "arrivals", "witnesses"):
                pilot.joint.assert_same_state(ref[key], observer.result[key], "recovered_native_" + key)
            pilot.joint.assert_same_state(ref["events"], [[list(k), v] for k, v in events.items()], "recovered_native_events")
    pilot.write_json(path, dict(index=index, reference_index=ri, context_index=ref["context_index"],
        future=ref["future"], action_index=action, metrics=metrics, witness=witness,
        recovery=recovery, alignment=alignment, reused_native=action == native and not check_native,
        forced_native_verified=check_native, exact_prefix=True))
    print("V85_ARM_DONE " + str(index), flush=True)


def aggregate(root, v84_root):
    contexts = []
    for ci, (old_ri, category) in enumerate(propagation.CONTEXTS):
        matrices = {mode: {} for mode in (*propagation.MODES, "recover_900s")}
        for ri, (context, future) in enumerate(repeated.REFERENCES):
            if context != ci:
                continue
            for mode in matrices:
                rows = [json.loads(((root if mode == "recover_900s" else v84_root) / "arms" /
                    f"a{ITEMS.index((ri, a)) if mode == 'recover_900s' else repeated.ITEMS.index((ri, mode, a))}.json").read_text())
                    for a in (0, 6)]
                for m in pilot.METRICS:
                    matrices[mode].setdefault(m, []).append(rows[1]["metrics"][m] - rows[0]["metrics"][m])
        effects = {mode: {m: repeated.describe(x) for m, x in values.items()} for mode, values in matrices.items()}
        differences = {mode: {m: repeated.describe(np.asarray(matrices["recover_900s"][m]) - np.asarray(values[m]))
            for m in repeated.RESPONSE_METRICS} for mode, values in matrices.items() if mode != "recover_900s"}
        contexts.append(dict(context_index=ci, original_reference_index=old_ri, stratum=category,
                             effects_45_minus_0=effects, recovery_minus_control=differences))
    result = dict(status="feedback_recovery_complete", contexts=contexts, labelled_arms=len(ITEMS), recovery_s=RECOVERY_S,
        contrast="45s minus 0s at one fixed prefix; costs through full day; not the V84 probability-weighted law effect",
        interval_unit="eight matched post-prefix futures conditional on each consumed prefix",
        actors_updated=False, main_policy_changed=False, controller_fits=0)
    pilot.write_json(root / "summary/status.json", result)
    print("V85_FEEDBACK_RECOVERY_COMPLETE", flush=True)


def smoke(root, v84_root):
    tests = Path(__file__).resolve().parents[1] / "tests"
    for pattern in ("test_v84_*.py", "test_v85_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-p", pattern, "-v"], check=True)
    audit(root, v84_root)
    refs = [repeated.REFERENCES.index((ci, 0)) for ci in (1, 16)]
    ref = json.loads((v84_root / "references" / f"r{refs[0]}.json").read_text())
    native = ref["contexts"][ref["stratum"]]["native_index"]
    far = 0 if pilot.step.ACTIONS[native] >= 20 else 6
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        pending = [pool.submit(arm, root, v84_root, ITEMS.index((ri, 0)), True) for ri in refs]
        pending.append(pool.submit(arm, root, v84_root, ITEMS.index((refs[0], far))))
        for task in pending:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_native_identity_and_delayed_release_passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v84-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.out_root, args.v84_root)
    elif args.command == "smoke":
        smoke(args.out_root, args.v84_root)
    elif args.command == "run":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(arm, args.out_root, args.v84_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))
    else:
        aggregate(args.out_root, args.v84_root)


if __name__ == "__main__":
    main()
