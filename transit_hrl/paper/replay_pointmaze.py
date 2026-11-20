"""Replay the first path of the frozen PointMaze review-controls cohort."""

import json
from math import isclose
from pathlib import Path

from freq_hrl.experiments.compute_budget import single_thread_runtime
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_reliability import load_controller


DATA = Path(__file__).resolve().parent / "data"


def main():
    fixture = json.loads((DATA / "pointmaze_replay.json").read_text())
    with single_thread_runtime():
        controller, args = load_controller({
            "configuration": fixture["configuration"],
            "controller_checkpoint": DATA / "pointmaze_replay.pt",
        })
        _, row = run_episode(
            controller, task_spec=PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id),
            seed=fixture["seed"], args=args, time_scale=time_contract(args),
            rule=TimingRule(**fixture["rule"]),
        )

    actual = row["tracking_squared_error_integral"]
    print(json.dumps({
        "seed": fixture["seed"],
        "steps": row["episode_length"],
        "planner_calls": row["upper_decision_count"],
        "decision_steps": row["decision_steps"],
        "ise": actual,
        "reference_ise": fixture["reference_ise"],
    }, indent=2))
    if row["decision_steps"] != fixture["decision_steps"]:
        raise RuntimeError("call times differ from the frozen replay")
    if not isclose(actual, fixture["reference_ise"], rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeError("episode ISE differs from the recorded evaluation")


if __name__ == "__main__":
    main()
