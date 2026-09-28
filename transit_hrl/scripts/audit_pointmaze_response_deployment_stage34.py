#!/usr/bin/env python3
"""Recompute deployed rewards, causal decisions and paid calls on the server."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.experiments import pointmaze_response_deployment as run
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_root_response as source_run
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_temporal_plan import causal_frame, padded_history
from scripts import pointmaze_response_deployment_stage34_spec as spec


def audit(path):
    result = json.loads(path.read_text())
    root, preflight = result["protocol"]["optimizer_seed"], result["protocol"]["preflight"]
    assert result["protocol"]["contract"] == spec.contract()
    args, controller, models, critics, factual, selected = run.load_frozen(root, preflight=preflight)
    cell = result["cells"][0]
    assert cell["controller_selected_iteration"] == selected
    assert cell["budget"] == spec.budget(root, preflight=preflight)
    paths = spec.evaluation_paths(root, preflight=preflight)
    expected = [(seed, method) for seed in paths for method in spec.METHODS]
    assert [(r["seed"], r["method"]) for r in cell["rows"]] == expected
    arrays = dict(np.load(Path(cell["raw_server_directory"]) / "episodes.npz", allow_pickle=False))
    assert list(zip(arrays["seeds"], arrays["methods"])) == expected
    tapes = motion.tapes_for(paths, protocol={"horizon": args.horizon, "task_options": _task_options(args)})
    original = json.loads(Path(cell["source_result"]).read_text())["cells"][0]
    trained = json.loads(Path(original["controller_result"]).read_text())["cells"][0]
    adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
                                    world_low=np.asarray(trained["world_low"]), world_high=np.asarray(trained["world_high"]))
    previews = plan_calls = response_calls = 0
    for index, row in enumerate(cell["rows"]):
        seed, method = row["seed"], row["method"]
        raw = {k: arrays[k][index] for k in ("ise", "reward", "sequence", "physical", "achieved", "measurement",
                                           "target", "position", "subgoal", "action")}
        np.testing.assert_array_equal(raw["measurement"], tapes[seed][:-1])
        distance = np.linalg.norm(raw["position"] - raw["target"], axis=1)
        np.testing.assert_allclose(raw["ise"], distance ** 2 * .01, rtol=0, atol=1e-12)
        np.testing.assert_allclose(raw["reward"], np.exp(-distance), rtol=0, atol=1e-12)
        assert abs(raw["ise"].sum() - row["tracking_squared_error_integral"]) < 1e-12
        assert abs(raw["reward"].sum() - row["episode_return"]) < 1e-10
        assert row["episode_length"] == row["lower_inference_calls"] == args.horizon
        if method == "fixed50":
            shared = set(range(0, args.horizon, 50))
        else:
            shared = {s for s in source_run.schedule_for(args, seed) if s < 100}
            shared.update(range(spec.checks(args.horizon)[-1] + 150, args.horizon, 50))
        execution, calls = set(row["executed_plan_steps"]), []
        delayed = {d["step"] + 100 for d in row["decisions"] if d["renew"] is False}
        frames, check_index, last_plan = deque(maxlen=64), 0, -1
        for step in range(args.horizon):
            subgoal = raw["achieved"][0] if step == 0 else raw["subgoal"][step - 1]
            observation = SimpleNamespace(physical=raw["physical"][step], achieved_goal=raw["achieved"][step],
                                          target=raw["target"][step], task_measurement=raw["measurement"][step])
            previous = np.zeros(2) if step == 0 else raw["action"][step - 1]
            names, frame = causal_frame(observation, subgoal, previous, step=step, last_plan=last_plan, horizon=args.horizon)
            frames.append(frame)
            if step in spec.checks(args.horizon):
                sequence = padded_history(frames)
                assert names == arrays["feature_names"].tolist()
                np.testing.assert_array_equal(sequence, raw["sequence"][check_index])
                decision = row["decisions"][check_index]
                assert decision["step"] == step
                if method in response.METHODS:
                    probe = {"seed": seed, "check_step": step, "sequence": sequence, "feature_names": names}
                    candidate, state = response.proposal(probe, controller, adapter, history_steps=64)
                    previews += 1
                    plan_calls += 1
                    np.testing.assert_array_equal(candidate, decision["candidate_plan"])
                    probe["candidate_plan"] = candidate
                    design = response.design([probe], models, method=method, root=root)
                    rate = float(critics[method].predict_rates(design)[0, -1])
                    response_calls += 1
                    assert abs(rate - decision["settled_rate"]) < 1e-12
                    assert decision["renew"] == (rate > 0)
                    calls.append({"step": step, "kind": "candidate_preview"})
                    if decision["renew"]:
                        np.testing.assert_array_equal(raw["subgoal"][step], candidate)
                elif method != "fixed50":
                    assert decision["renew"] == (method == "always_renew")
                check_index += 1
            if step in shared or step in delayed or (method == "always_renew" and step in spec.checks(args.horizon)):
                history = raw["measurement"][max(0, step - 63):step + 1]
                history = np.vstack((np.repeat(raw["measurement"][0:1], 64 - len(history), axis=0), history))
                state = np.r_[observation.physical, observation.target - observation.achieved_goal, history.ravel()].astype(np.float32)
                candidate = adapter.decode(np.asarray(controller.plan_goal(state, sample=False)["action"], dtype=np.float32),
                                           observation.achieved_goal)
                plan_calls += 1
                np.testing.assert_array_equal(candidate, raw["subgoal"][step])
                kind = "delayed_plan" if step in delayed else ("immediate_plan" if method == "always_renew"
                        and step in spec.checks(args.horizon) else "shared_plan")
                calls.append({"step": step, "kind": kind})
            if step in execution:
                last_plan = step
            else:
                np.testing.assert_array_equal(subgoal, raw["subgoal"][step])
        assert calls == row["upper_calls"]
        assert row["upper_inference_calls"] == len(calls)
        assert row["executed_plan_count"] == len(execution) == len(shared) + (0 if method == "fixed50" else len(spec.checks(args.horizon)))
        discard = len(delayed) if method in response.METHODS else 0
        assert row["discarded_preview_calls"] == discard
        assert row["upper_inference_calls"] == len(execution) + discard
        assert row["candidate_preview_calls"] == sum(c["kind"] == "candidate_preview" for c in calls)
    assert _json_ready(run.summarize(cell["rows"])) == cell["metrics"]
    return {"status": "pass", "optimizer_seed": root, "checked_episodes": len(expected),
            "checked_primitive_steps": sum(r["episode_length"] for r in cell["rows"]),
            "factual_replay": factual, "additional_verification_primitive_steps": args.horizon,
            "additional_verification_policy_calls": plan_calls, "additional_verification_previews": previews,
            "additional_verification_response_calls": response_calls, "linear_solves": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    torch.set_num_threads(1)
    print(json.dumps(audit(parser.parse_args().result), sort_keys=True))
