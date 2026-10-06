import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_native_upper_step as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def native_task(stack):
    stack.enter_context(patch.object(experiment.joint.source.native.joint, "_make_task",
        side_effect=lambda **kw: Float32Task()))
    stack.enter_context(patch.object(experiment.joint.source.native.joint, "pointmaze_goal_bounds",
        return_value=(-2 * np.ones(2), 2 * np.ones(2))))


def test_gradient_step_has_correct_sign_fixed_radius_and_frozen_lower(source_data):
    models, _, _, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    initial = experiment.joint.weights(trainer)
    states = np.stack([np.linspace(0., .25, 392), np.linspace(.1, -.1, 392)]).astype(np.float32)
    labels = [{"gradients": {p: np.arange(1., 9.) * scale for p, scale in (("A", 1.), ("B", 1.1))}}] * 2
    candidates, learning, cost = experiment.learn_candidates(trainer, states, labels)
    assert learning["actor_gradient_cosine"] > .999999
    assert cost == {"fisher_jvp_batches": 1, "exact_kl_forward_batches": 2}
    for sign, actor in candidates.items():
        assert experiment.joint.parameter_delta(initial["upper_actor"], actor.state_dict()) > 0
        torch.testing.assert_close(actor.log_std, initial["upper_actor"]["log_std"], atol=0, rtol=0)
        np.testing.assert_allclose(learning["geometry"]["exact_kl"][sign], experiment.spec.FISHER_RADIUS, rtol=2e-6)
        mean = actor.distribution(torch.as_tensor(states)).mean.detach().numpy()
        alignment = float(np.mean(mean @ np.arange(1., 9.)))
        assert alignment > 0 if sign == "plus" else alignment < 0
    torch.testing.assert_close(experiment.joint.weights(trainer), initial, atol=0, rtol=0)


def test_cache_replay_recovers_state_and_rejects_different_native_return(source_data):
    models, predictor, cal, args, teachers = source_data
    trainer = experiment.joint.make_trainer(models["50"], teachers["50"], args)
    query = experiment.spec.source.queries(410011)[0]
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)))
        row, audit = experiment.source.intervention_episode(trainer, args=args, query=query, panel="A",
            action_delta=np.zeros(8), period=50, predictor=predictor, envelope=cal["50"]["envelope"])
        job = (experiment.joint.weights(models["50"]), teachers["50"], query, row["suffix_return"],
            50, predictor, cal["50"]["envelope"])
        replay = experiment.replay_query(job)
        np.testing.assert_array_equal(replay["state"], audit["state"])
        assert replay["native_episodes"] == 1 and replay["native_steps"] == 100
        with pytest.raises(AssertionError):
            experiment.replay_query((*job[:3], row["suffix_return"] + .01, *job[4:]))


def test_reduced_run_updates_upper_and_tests_fresh_closed_loop_with_exact_cost(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    source_snapshot = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    cache_path = tmp_path / "cached_native_credit.json"
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(experiment.spec.source, "arguments", return_value=args))
        stack.enter_context(patch.object(experiment.spec.source, "STARTS", (0,)))
        stack.enter_context(patch.object(experiment.spec.source, "SCENARIOS_PER_START", 1))
        stack.enter_context(patch.object(experiment.spec, "EVALUATION_EPISODES", 1))
        stack.enter_context(patch.object(experiment.spec, "source_result", return_value=cache_path))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state",
            side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        groups = {}
        for period in experiment.spec.PERIODS:
            p = str(period)
            trainer = experiment.joint.make_trainer(models[p], teachers[p], args)
            query = experiment.spec.source.queries(410011)[0]
            row, _ = experiment.source.intervention_episode(trainer, args=args, query=query, panel="A",
                action_delta=np.zeros(8), period=period, predictor=predictor, envelope=cal[p]["envelope"])
            # Synthetic labels isolate the update path; replay returns come from the fixture.
            groups[p] = {"queries": [{"query": query, "gradients": {panel: np.arange(1., 9.).tolist()
                for panel in experiment.spec.source.PANELS},
                "coordinate_suffix_returns": {"A": {"zero": row["suffix_return"]}}}]}
        experiment.write_json(cache_path, {"status": "complete", "protocol": experiment.spec.source.PROTOCOL,
            "root": 410011, "epsilon": experiment.spec.source.EPSILON, "cost": experiment.spec.source.budget(),
            "teacher_upper_lower_and_source_frozen": "passed", "groups": groups})
        result = experiment.run(410011, tmp_path / "result.json")
        assert result["cost"] == experiment.spec.budget()
    assert result["cost"]["native_episodes"] == 12 and result["cost"]["native_steps"] == 1200
    assert result["cost"]["upper_candidate_weight_steps"] == 4
    assert len(list(tmp_path.rglob("*.pt"))) == 4 and not list(tmp_path.rglob("*.npz"))
    for p, group in result["groups"].items():
        assert group["effects"]["native_ascent_minus_source_forecast"] == group["effects"]["native_ascent_minus_native_blinded"]
        assert group["source_replay_and_lower_freeze"] == "passed"
        assert group["mean_metrics"]["native_ascent"]["upper_mean_rms"] > 0
        assert group["mean_metrics"]["native_ascent"]["reference_correction_peak"] <= .05 + 1e-8
        experiment.joint.source.native.curves.support.assert_frozen(models[p], source_snapshot[p])


def test_frozen_radius_budgets_and_fresh_scene_rosters():
    spec = experiment.spec
    assert spec.FISHER_RADIUS == .005 ** 2 / (2 * .15 ** 2)
    assert spec.MINIMUM_GAIN == .5 and spec.PERIODS == (50, 100)
    assert spec.budget()["native_episodes"] == 336 and spec.budget()["native_steps"] == 403200
    assert spec.budget()["evaluation_episodes"] == 320
    seen = set()
    for root in spec.ROOTS:
        queries = spec.source.queries(root)
        old = {s for q in queries for s in (q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values())}
        roles = spec.source.source.seed_roles(root, preflight=False)
        old.update(s for batch in roles["training_rounds"] for q in batch for s in (q["scenario_seed"], *q["noise_seeds"]))
        old.update(roles["native_evaluation"])
        fresh = spec.evaluation_seeds(root)
        assert len(fresh) == len(set(fresh)) == 32 and not set(fresh).intersection(old | seen)
        seen.update(fresh)
