#!/usr/bin/env python3
"""Repeat fixed V83 prefixes with matched independent future innovations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

import run_freqduet_v68_repeated_continuations as repeated
import run_freqduet_v83_holding_propagation as propagation


pilot, step = propagation.pilot, propagation.step
FUTURES = tuple(range(8))
REFERENCES = list(itertools.product(range(len(propagation.CONTEXTS)), FUTURES))
ITEMS = list(itertools.product(range(len(REFERENCES)), propagation.MODES, range(len(step.ACTIONS))))
RESPONSE_METRICS = ("passenger_person_s", "cv_squared", "readiness_delay_s", "holding_s_per_trip")


def future_seed(context, future):
    return 84000000 + context * 100 + future


def describe(values):
    x = np.asarray(values, dtype=float)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = float(sd / np.sqrt(len(x)))
    radius = float(student_t.ppf(.975, len(x) - 1)) * se
    return dict(mean=mean, future_sd=sd, mean_se=float(se), ci95=[mean - radius, mean + radius],
                negative=int(sum(x < 0)), positive=int(sum(x > 0)), n=len(x))


def action_noise(values, native):
    values = np.asarray(values, dtype=float)
    baseline = values[:, native]
    paired = np.var(values - baseline[:, None], axis=0, ddof=1)
    unpaired = np.var(values, axis=0, ddof=1) + np.var(baseline, ddof=1)
    return dict(mean_deltas=(values - baseline[:, None]).mean(axis=0).tolist(),
                paired_sd=np.sqrt(paired).tolist(), independent_sd=np.sqrt(unpaired).tolist())


class FutureTrace(propagation.PropagationTrace):
    def __init__(self, reference, stratum, mode, action, seed, uniforms):
        super().__init__(reference, stratum, mode, action)
        self.future_seed, self.uniforms, self.forked = seed, uniforms, False

    def __call__(self, policy, states, features, theta):
        before = self.fired
        law = super().__call__(policy, states, features, theta)
        if self.fired and not before and self.future_seed is not None:
            # All current observations are captured first; only later innovations change.
            repeated.fork_future(self.runner, policy, self.future_seed)
            self.uniforms.seed = self.future_seed + 70000000
            self.forked = True
        return law

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        if self.future_seed is not None and not self.forked:
            raise AssertionError("the registered post-prefix fork was not reached")


def replay(root, ref, category, mode, action, seed, label):
    uniforms = pilot.event.EventUniforms(ref["sampler"])
    observer = FutureTrace(ref, category, mode, action, seed, uniforms)
    row, events, _, _ = pilot.rollout(root, ref["seed"], ref["day"], "stochastic", ref["sampler"],
        ref["theta"], label, probability_rule=observer, trace_observer=observer, event_uniforms=uniforms)
    return dict(metrics={m: row[m] for m in pilot.METRICS},
                events=[[list(k), v] for k, v in events.items()], **observer.result)


def reference(root, v83_root, index):
    path = root / "references" / f"r{index}.json"
    if path.exists():
        return
    ci, future = REFERENCES[index]
    ri, category = propagation.CONTEXTS[ci]
    old = json.loads((v83_root / "references" / f"r{ri}.json").read_text())
    native = old["contexts"][category]["native_index"]
    seed = future_seed(ci, future)
    result = replay(root, old, category, "closed_loop", native, seed, f"reference{index}")
    result["contexts"][category]["native_index"] = native
    pilot.write_json(path, dict(index=index, context_index=ci, future=future, future_seed=seed,
        seed=old["seed"], day=old["day"], stratum=category, theta=old["theta"], sampler=old["sampler"], **result))
    print("V84_REFERENCE_DONE " + str(index), flush=True)


def arm(root, index, force_native=False):
    path = root / "arms" / f"a{index}.json"
    if path.exists() and not force_native:
        return
    ri, mode, action = ITEMS[index]
    ref = json.loads((root / "references" / f"r{ri}.json").read_text())
    category = ref["stratum"]
    capture = ref["contexts"][category]
    native = action == capture["native_index"]
    alignment, changes = None, 0
    if native and not force_native:
        result = ref
    else:
        result = replay(root, ref, category, mode, action, ref["future_seed"],
                        f"arm{index}" + ("_forced_native" if force_native else ""))
        alignment = pilot.event.alignment({tuple(k): v for k, v in ref["events"]},
                                         {tuple(k): v for k, v in result["events"]})
        tape = {tuple(r[1:4]): r[-1] for r in ref["commands"]}
        changes = sum(r[-1] != tape[tuple(r[1:4])] for r in result["commands"][capture["ordinal"] + 1:])
        if native:
            for key in ("metrics", "commands", "arrivals", "events", "witnesses"):
                pilot.joint.assert_same_state(ref[key], result[key], "fresh_native_" + key)
    pilot.write_json(path, dict(index=index, reference_index=ri, context_index=ref["context_index"],
        future=ref["future"], mode=mode, action_index=action, action_s=float(step.ACTIONS[action]),
        metrics=result["metrics"], witness=result["witnesses"][category], alignment=alignment,
        later_lower_action_changes=changes, reused_native=native and not force_native,
        forced_native_verified=force_native, exact_prefix=True))
    print("V84_ARM_DONE " + str(index), flush=True)


def aggregate(root, v83_root):
    arms = [json.loads((root / "arms" / f"a{i}.json").read_text()) for i in range(len(ITEMS))]
    refs = [json.loads((root / "references" / f"r{i}.json").read_text()) for i in range(len(REFERENCES))]
    original = json.loads((v83_root / "summary/status.json").read_text())["contexts"]
    lookup = {(r["reference_index"], r["mode"], r["action_index"]): r for r in arms}
    contexts = []
    for ci, (old_ri, category) in enumerate(propagation.CONTEXTS):
        selected = [r for r in refs if r["context_index"] == ci]
        capture = selected[0]["contexts"][category]
        effects, noise, future_effects = {}, {}, {}
        for mode in propagation.MODES:
            values = {}
            for metric in pilot.METRICS:
                values[metric] = [[lookup[r["index"], mode, a]["metrics"][metric]
                                  for a in range(7)] for r in selected]
            for stage in ("departure", "arrival_h1", "arrival_h3", "arrival_h5"):
                values[stage + "_loss"] = [[lookup[r["index"], mode, a]["witness"][stage]["loss"]
                    if lookup[r["index"], mode, a]["witness"][stage] else None for a in range(7)] for r in selected]
            effects[mode], noise[mode], future_effects[mode] = {}, {}, {}
            for label in ("raw", "projected"):
                contrast = label + "_vs_base"
                effects[mode][contrast], future_effects[mode][contrast] = {}, {}
                for metric, matrix in values.items():
                    complete = all(x is not None for row in matrix for x in row)
                    y = [propagation.expected_effect(capture[label], capture["base"], row)
                         for row in matrix] if complete else None
                    future_effects[mode][contrast][metric] = y
                    effects[mode][contrast][metric] = describe(y) if y is not None else None
            noise[mode] = {m: action_noise(values[m], capture["native_index"]) for m in RESPONSE_METRICS}
        feedback = {}
        for contrast in ("raw_vs_base", "projected_vs_base"):
            feedback[contrast] = {m: describe(np.asarray(future_effects["closed_loop"][contrast][m]) -
                np.asarray(future_effects["recorded_lower"][contrast][m])) for m in RESPONSE_METRICS}
        contexts.append(dict(context_index=ci, original_reference_index=old_ri, seed=selected[0]["seed"],
            day=selected[0]["day"], stratum=category, time_s=capture["time_s"], native_index=capture["native_index"],
            original_v83_effects=original[ci]["effects"], effects=effects, feedback=feedback,
            action_noise=noise, future_effects={mode: {m: future_effects[mode]["projected_vs_base"][m]
                for m in RESPONSE_METRICS} for mode in propagation.MODES}))
    groups = {}
    for category in propagation.STRATA:
        selected = [c for c in contexts if c["stratum"] == category]
        groups[category] = {mode: {m: dict(
            mean=float(np.mean([c["effects"][mode]["projected_vs_base"][m]["mean"] for c in selected])),
            original_mean=float(np.mean([c["original_v83_effects"][mode]["projected_vs_base"][m] for c in selected])),
            contexts_negative=sum(c["effects"][mode]["projected_vs_base"][m]["mean"] < 0 for c in selected),
            intervals_below_zero=sum(c["effects"][mode]["projected_vs_base"][m]["ci95"][1] < 0 for c in selected),
            intervals_above_zero=sum(c["effects"][mode]["projected_vs_base"][m]["ci95"][0] > 0 for c in selected),
            contexts=len(selected)) for m in RESPONSE_METRICS} for mode in propagation.MODES}
    pilot.write_json(root / "summary/status.json", dict(status="matched_future_propagation_complete",
        contexts=contexts, by_stratum=groups, labelled_arms=len(arms), references=len(refs), futures_per_prefix=len(FUTURES),
        forced_native_checks=sum(r["forced_native_verified"] for r in arms),
        max_recorded_lower_changes=max(r["later_lower_action_changes"] for r in arms if r["mode"] == "recorded_lower"),
        actors_updated=False, main_policy_changed=False, controller_fits=0,
        interval_unit="eight post-prefix innovations conditional on each fixed prefix; not training-seed inference",
        scope="24 consumed V83 prefixes, fixed day latents and pending arrivals; not fresh-day confirmation",
        full_policy_effects_are_not_additive_single_action_effects=True))
    print("V84_PROPAGATION_REPEATS_COMPLETE", flush=True)


def report(root):
    summary = json.loads((root / "summary/status.json").read_text())
    print("V84_TOTAL " + json.dumps({k: summary[k] for k in (
        "labelled_arms", "references", "futures_per_prefix", "forced_native_checks",
        "max_recorded_lower_changes", "actors_updated", "main_policy_changed", "controller_fits")}))
    print("V84_STRATA " + json.dumps(summary["by_stratum"]))
    for context in summary["contexts"]:
        modes = {}
        for mode in propagation.MODES:
            effects = context["effects"][mode]["projected_vs_base"]
            old = context["original_v83_effects"][mode]["projected_vs_base"]
            metrics = {}
            for m in RESPONSE_METRICS:
                result = effects[m]
                noise = context["action_noise"][mode][m]
                alternatives = [i for i in range(7) if i != context["native_index"]]
                ratios = [noise["paired_sd"][i] / noise["independent_sd"][i] for i in alternatives
                          if noise["independent_sd"][i] > 0]
                metrics[m] = dict(old=old[m], **result,
                    old_sign_repeats=sum(float(x) * old[m] > 0 for x in context["future_effects"][mode][m]),
                    paired_to_independent_sd_median=float(np.median(ratios)) if ratios else None,
                    action_signal_above_sd=sum(abs(noise["mean_deltas"][i]) > noise["paired_sd"][i]
                                               for i in alternatives))
            modes[mode] = dict(metrics=metrics, stages={k: v for k, v in effects.items() if k.endswith("_loss")})
        print("V84_CONTEXT " + json.dumps(dict(context_index=context["context_index"], seed=context["seed"],
            day=context["day"], stratum=context["stratum"], time_s=context["time_s"],
            modes=modes, feedback=context["feedback"]["projected_vs_base"])))


def smoke(root, v83_root):
    tests = Path(__file__).resolve().parents[1] / "tests"
    for pattern in ("test_v83_*.py", "test_v84_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-p", pattern, "-v"], check=True)
    ci = propagation.CONTEXTS.index((0, "bilateral_active"))
    old = json.loads((v83_root / "references/r0.json").read_text())
    category = "bilateral_active"
    native = old["contexts"][category]["native_index"]
    noop = replay(root, old, category, "closed_loop", native, None, "noop")
    for key in ("metrics", "commands", "arrivals", "events", "witnesses"):
        expected = {category: old[key][category]} if key == "witnesses" else old[key]
        pilot.joint.assert_same_state(expected, noop[key], "old_future_noop_" + key)
    second = propagation.CONTEXTS.index((5, "bilateral_active"))
    selected = [REFERENCES.index((ci, 0)), REFERENCES.index((ci, 1)), REFERENCES.index((second, 0))]
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(reference, root, v83_root, i) for i in selected]:
            task.result()
    first, other = [json.loads((root / "references" / f"r{i}.json").read_text()) for i in selected[:2]]
    ordinal = first["contexts"][category]["ordinal"]
    if first["events"][ordinal][1]["uniform"] == other["events"][ordinal][1]["uniform"]:
        raise AssertionError("distinct future seeds reused the same policy uniform")
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        pending = []
        for ri in (selected[0], selected[2]):
            ref = json.loads((root / "references" / f"r{ri}.json").read_text())
            native = ref["contexts"][category]["native_index"]
            far = 0 if step.ACTIONS[native] >= 20 else 6
            pending.extend(pool.submit(arm, root, ITEMS.index((ri, mode, native)), True) for mode in propagation.MODES)
            pending.append(pool.submit(arm, root, ITEMS.index((ri, "recorded_lower", far))))
        for task in pending:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_exact_prefix_fresh_native_tape_and_crn_passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "references", "run", "aggregate", "report"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v83-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v83_root)
    elif args.command in ("references", "run"):
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = ([pool.submit(reference, args.out_root, args.v83_root, i) for i in range(len(REFERENCES))]
                if args.command == "references" else [pool.submit(arm, args.out_root, i) for i in range(args.start, args.end)])
            for task in pending:
                task.result()
        marker = "reference_ready/done.json" if args.command == "references" else f"shards/{args.start}-{args.end}.json"
        pilot.write_json(args.out_root / marker, dict(status="complete"))
    elif args.command == "report":
        report(args.out_root)
    else:
        aggregate(args.out_root, args.v83_root)


if __name__ == "__main__":
    main()
