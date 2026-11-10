#!/usr/bin/env python3
"""Independent balanced-prefix confirmation of frozen-policy action labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v69_event_coupling as event


labels, repeated, joint = event.labels, event.repeated, event.joint
DEMAND_SEEDS = tuple(range(700001, 700013))
TARGETS = labels.TARGETS
REFERENCES = list(itertools.product(range(len(labels.deployment.CASES)), range(len(DEMAND_SEEDS))))
CONTEXTS = list(itertools.product(range(len(REFERENCES)), range(len(TARGETS))))
FUTURES = tuple(range(32))
ITEMS = list(itertools.product(range(len(CONTEXTS)), FUTURES))
SMOKE_ITEMS = (0, len(FUTURES), (len(CONTEXTS) // 2 + 2) * len(FUTURES))


def reference_spec(index):
    case, day = REFERENCES[index]
    seed, actor_scenario = labels.deployment.CASES[case]
    return dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario,
                scenario_seed=DEMAND_SEEDS[day], sampler_seed=7000000 + case * 1000 + day,
                label=f"{labels.deployment.case_name(seed, actor_scenario)}_demand{DEMAND_SEEDS[day]}")


def context_spec(index):
    reference, window = CONTEXTS[index]
    return dict(reference_spec(reference), reference_index=reference, context_index=index,
                target_time_s=TARGETS[window])


def future_seed(context, future):
    return 70000000 + context * 100 + future


def reference_path(root, index):
    return root / "references" / (reference_spec(index)["label"] + ".pt")


def reference(root, v59_root, index):
    path = reference_path(root, index)
    if path.exists():
        return path.stem
    spec = reference_spec(index)
    result = labels.run_spec(root, v59_root, spec, TARGETS, spec["label"] + "_reference")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(spec=spec, **result), path)
    print("V70_REFERENCE_DONE " + json.dumps(spec), flush=True)
    return path.stem


def group_path(root, index):
    context, future = ITEMS[index]
    return root / "groups" / f"c{context}_f{future}.json"


def run_group(root, v59_root, index, check_noop=False):
    path = group_path(root, index)
    if path.exists():
        return path.stem
    context, future = ITEMS[index]
    spec, seed = context_spec(context), future_seed(context, future)
    base = torch.load(reference_path(root, spec["reference_index"]), map_location="cpu", weights_only=False)
    original = base["captures"][spec["target_time_s"]]

    def replay(action, suffix):
        uniforms = event.EventUniforms(seed + 10000000)

        def continuation(runner, policy):
            startup = repeated.fork_future(runner, policy, seed)
            uniforms.install(runner, policy)
            return dict(startup, event_uniform_seed=uniforms.seed)

        result = labels.run_spec(root, v59_root, reference_spec(spec["reference_index"]),
                                 (spec["target_time_s"],), f"c{context}_f{future}_{suffix}", action, continuation)
        labels.same_prefix(original, result["captures"][spec["target_time_s"]])
        if result["summary"]["scenario_seed"] != seed:
            raise AssertionError("V70 continuation changed its predeclared future seed")
        print("V70_ARM_DONE " + json.dumps(dict(context_index=context, future_index=future, arm=suffix)), flush=True)
        return result, uniforms.records

    native, native_records = replay(None, "native")
    native_action = original["actor_action_s"]
    rows = []
    for action in joint.ACTIONS:
        result, records = (native, native_records) if action == native_action else replay(action, f"a{action}")
        joint.assert_same_state(native["captures"][spec["target_time_s"]]["continuation"],
                                result["captures"][spec["target_time_s"]]["continuation"], "common_future_start")
        rows.append(dict(action_s=action, **{k: result["summary"][k] for k in labels.OUTCOMES},
                         conservation_max_error=result["summary"]["stream_audit"]["conservation_max_error"],
                         alignment=None if action == native_action else event.alignment(native_records, records)))
    if check_noop:
        forced, records = replay(native_action, "forced_native")
        for key in ("summary", "holds", "upper_path"):
            joint.assert_same_state(native[key], forced[key], "forced_native_" + key)
        joint.assert_same_state(native_records, records, "forced_native_event_uniforms")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(**spec, future_index=future, future_seed=seed,
                                    native_action_s=native_action, rows=rows, exact_prefix=True,
                                    common_future_start=True, forced_native_verified=check_noop,
                                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V70_GROUP_DONE " + json.dumps(dict(context_index=context, future_index=future, rollouts=7)), flush=True)
    return path.stem


def analyze_context(spec, groups):
    if [g["future_index"] for g in groups] != list(FUTURES):
        raise ValueError("V70 context lacks the complete 32-future panel")
    if any([r["action_s"] for r in g["rows"]] != list(joint.ACTIONS) for g in groups):
        raise ValueError("V70 future lacks the full seven-action grid")
    if len({g["native_action_s"] for g in groups}) != 1:
        raise ValueError("V70 futures do not share the original current actor action")
    native = joint.ACTIONS.index(groups[0]["native_action_s"])
    arrays = {m: np.asarray([[r[m] for r in g["rows"]] for g in groups], dtype=float)
              for m in repeated.RESPONSE_METRICS}
    halves = [repeated.evaluate_selection(arrays, native, np.arange(16), np.arange(16, 32)),
              repeated.evaluate_selection(arrays, native, np.arange(16, 32), np.arange(16))]
    return dict(**spec, native_action_s=groups[0]["native_action_s"], halves=halves,
                cross_future={m: float(np.mean([h[m] for h in halves])) for m in repeated.RESPONSE_METRICS},
                signal_noise={m: repeated.signal_noise(a - a[:, :1]) for m, a in arrays.items()},
                exact_prefixes=all(g["exact_prefix"] for g in groups),
                common_future_start=all(g["common_future_start"] for g in groups),
                conservation_max_error=max(r["conservation_max_error"] for g in groups for r in g["rows"]),
                forced_native_groups=sum(g["forced_native_verified"] for g in groups),
                alignment={k: sum(r["alignment"][k] for g in groups for r in g["rows"] if r["alignment"] is not None)
                           for k in groups[0]["rows"][1 if native == 0 else 0]["alignment"]})


def interval(contexts, getter):
    means = [float(np.mean([getter(c) for c in contexts if c["scenario_seed"] == day])) for day in DEMAND_SEEDS]
    result = labels.expanded.paired_interval(means, 70070)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def signal_summary(contexts, metric):
    signal, noise = [np.asarray([np.mean([c["signal_noise"][metric][stat] for c in contexts if c["scenario_seed"] == day])
                                for day in DEMAND_SEEDS])
                     for stat in ("debiased_signal_squared", "single_future_noise_variance")]
    indices = np.random.default_rng(70071).integers(len(DEMAND_SEEDS), size=(10000, len(DEMAND_SEEDS)))
    ratios = signal[indices].mean(axis=1) / noise[indices].mean(axis=1)
    return dict(corrected_signal_squared=interval(contexts, lambda c: c["signal_noise"][metric]["debiased_signal_squared"]),
                noise_variance=interval(contexts, lambda c: c["signal_noise"][metric]["single_future_noise_variance"]),
                single_future_snr=float(signal.mean() / noise.mean()),
                single_future_snr_ci95=np.quantile(ratios, [.025, .975]).tolist(),
                mean32_snr=float(32 * signal.mean() / noise.mean()),
                mean32_snr_ci95=(32 * np.quantile(ratios, [.025, .975])).tolist(),
                half_sign_agreement=interval(contexts, lambda c: c["signal_noise"][metric]["half_sign_agreement"]),
                positive_signal_days=int((signal > 0).sum()))


def gate_decision(signal, effects):
    return dict(independent_label_gate_passed=all(s["corrected_signal_squared"]["ci95"][0] > 0
                                                 and s["mean32_snr_ci95"][0] > 1 for s in signal.values()),
                joint_opportunity_gate_passed=effects["passenger_person_s"]["ci95"][1] < 0
                    and all(effects[m]["ci95"][1] <= 0 for m in repeated.CONSTRAINT_METRICS))


def aggregate(root):
    contexts = [analyze_context(context_spec(context),
                [json.loads(group_path(root, context * len(FUTURES) + f).read_text()) for f in FUTURES])
                for context in range(len(CONTEXTS))]
    halves = [h for c in contexts for h in c["halves"] if h["changed"]]
    effects = {m: interval(contexts, lambda c, m=m: c["cross_future"][m]) for m in repeated.RESPONSE_METRICS}
    signal = {m: signal_summary(contexts, m) for m in ("passenger_person_s", "cv_squared")}
    result = dict(status="independent_balanced_action_labels_complete", references=len(REFERENCES), contexts=len(contexts),
                  futures_per_context=len(FUTURES), formal_action_rollouts=len(ITEMS) * len(joint.ACTIONS),
                  demand_day_clusters=len(DEMAND_SEEDS), original_training_seeds=2, demand_seeds=DEMAND_SEEDS,
                  target_times_s=TARGETS, exact_prefixes=all(c["exact_prefixes"] for c in contexts),
                  common_future_start=all(c["common_future_start"] for c in contexts),
                  conservation_max_error=max(c["conservation_max_error"] for c in contexts),
                  forced_native_groups=sum(c["forced_native_groups"] for c in contexts),
                  alignment={k: sum(c["alignment"][k] for c in contexts) for k in contexts[0]["alignment"]},
                  cross_future=effects, signal_noise=signal, **gate_decision(signal, effects),
                  changed_half_decisions=len(halves),
                  changed_passenger_improved=sum(h["passenger_person_s"] < 0 for h in halves),
                  changed_passenger_worsened=sum(h["passenger_person_s"] > 0 for h in halves),
                  changed_joint_nonworse=sum(h["passenger_person_s"] < 0 and
                                            all(h[m] <= 0 for m in repeated.CONSTRAINT_METRICS) for h in halves),
                  heldout_constraint_violations={m: sum(h[m] > 0 for h in halves) for m in repeated.CONSTRAINT_METRICS},
                  by_window={str(t): {m: interval([c for c in contexts if c["target_time_s"] == t],
                                                 lambda c, m=m: c["cross_future"][m]) for m in repeated.RESPONSE_METRICS}
                             for t in TARGETS},
                  by_training_seed={str(seed): {m: interval([c for c in contexts if c["train_seed"] == seed],
                                                          lambda c, m=m: c["cross_future"][m]) for m in repeated.RESPONSE_METRICS}
                                    for seed in sorted({c["train_seed"] for c in contexts})},
                  by_day=[dict(scenario_seed=day, **{m: float(np.mean([c["cross_future"][m] for c in contexts
                                                                     if c["scenario_seed"] == day]))
                                                   for m in repeated.RESPONSE_METRICS}) for day in DEMAND_SEEDS],
                  actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False,
                  independent_prefixes=True, control_admitted=False)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (root / "groups/context_audit.json").write_text(json.dumps(contexts, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V70_AGGREGATE " + json.dumps(result), flush=True)


def smoke(root, v59_root, v69_root, workers):
    tests = str(Path(__file__).resolve().parents[1] / "tests")
    for pattern in ("test_v6[7-9]_*.py", "test_v70_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", tests, "-p", pattern, "-v"], check=True)
    uniforms = event.EventUniforms(repeated.future_seed(0, 0) + 10000000)

    def continuation(runner, policy):
        startup = repeated.fork_future(runner, policy, repeated.future_seed(0, 0))
        uniforms.install(runner, policy)
        return startup

    legacy = labels.run_branch(root, v59_root, 0, (TARGETS[0],), "legacy_v69_native", continuation=continuation)
    old = json.loads(event.group_path(v69_root, 0).read_text())
    old_native = next(r for r in old["rows"] if r["action_s"] == old["native_action_s"])
    for metric in labels.OUTCOMES:
        joint.assert_same_state(old_native[metric], legacy["summary"][metric], "V69_native_identity_" + metric)
    references = sorted({CONTEXTS[ITEMS[i][0]][0] for i in SMOKE_ITEMS})
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(reference, itertools.repeat(root), itertools.repeat(v59_root), references))
        list(pool.map(run_group, itertools.repeat(root), itertools.repeat(v59_root), SMOKE_ITEMS, (True,) * len(SMOKE_ITEMS)))
    if not all(json.loads(group_path(root, i).read_text())["forced_native_verified"] for i in SMOKE_ITEMS):
        raise AssertionError("V70 smoke lacks forced-native identities for every target window")
    output = root / "smoke_export"
    output.mkdir(exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(status="independent_prefix_and_legacy_identity_pass",
                                                     rollouts=27, cached_formal_rollouts=23)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "reference-batch", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v69-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v59_root, args.v69_root, args.workers)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        reference_mode = args.command == "reference-batch"
        count, function = (len(REFERENCES), reference) if reference_mode else (len(ITEMS), run_group)
        end = count if args.end is None else args.end
        if not 0 <= args.start < end <= count:
            parser.error("batch range lies outside the predeclared V70 panel")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            completed = list(pool.map(function, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                      range(args.start, end)))
        output = args.out_root / ("reference_exports" if reference_mode else "exports") / f"batch_{args.start}_{end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed=len(completed), start=args.start, end=end)) + "\n")


if __name__ == "__main__":
    main()
