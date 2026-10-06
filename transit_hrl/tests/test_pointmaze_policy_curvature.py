import copy
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.rl.native_return_step import fit_native_return_step
from freq_hrl.experiments import pointmaze_policy_curvature as experiment
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


@pytest.mark.parametrize("slope,curvature,scale", [(2., -4., .25), (2., -.25, 1.),
    (-2., -4., 0.), (2., 1., 1.), (-2., 1., 0.), (-1., 2., 1.), (0., 0., 0.)])
def test_native_quadratic_step_matches_bounded_maximum(slope, curvature, scale):
    zero = np.array([100., 120., 140.])
    result = fit_native_return_step(zero, zero + slope + curvature, zero - slope + curvature)
    assert result["scale"] == scale and result["slope"] == slope
    assert result["quadratic_coefficient"] == curvature
    grid = np.linspace(0., 1., 1001)
    assert result["predicted_gain"] >= np.max(slope * grid + curvature * grid ** 2) - 1e-12


def test_actor_interpolation_preserves_variance_and_original_weights(source_data):
    models, _, _, args, teachers = source_data
    initial = experiment.joint.make_trainer(models["50"], teachers["50"], args).upper_actor.state_dict()
    before = copy.deepcopy(initial)
    direction = {k: v.clone() if k == "log_std" else v + .02 for k, v in initial.items()}
    scaled = experiment.scaled_upper(initial, direction, .25)
    for k in initial:
        expected = initial[k] if k == "log_std" else initial[k] + .005
        torch.testing.assert_close(scaled[k], expected, atol=0, rtol=0)
    torch.testing.assert_close(initial, before, atol=0, rtol=0)
    direction["log_std"] += .01
    with pytest.raises(AssertionError):
        experiment.scaled_upper(initial, direction, .25)


def test_reduced_native_run_uses_training_only_fits_and_exact_cost(source_data, tmp_path):
    models, predictor, cal, args, teachers = source_data
    snapshots = {p: copy.deepcopy(m.state_dict()) for p, m in models.items()}
    source_path = tmp_path / "source" / "result.json"
    with ExitStack() as stack:
        native_task(stack)
        stack.enter_context(patch.object(experiment.spec.source.source, "arguments", return_value=args))
        stack.enter_context(patch.object(experiment.spec, "TRAINING_SCENARIOS", 1))
        stack.enter_context(patch.object(experiment.spec, "EVALUATION_EPISODES", 1))
        stack.enter_context(patch.object(experiment.spec, "source_result", return_value=source_path))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state",
            side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        for period in experiment.spec.PERIODS:
            initial = experiment.joint.make_trainer(models[str(period)], teachers[str(period)], args).upper_actor.state_dict()
            for sign, value in (("plus", .00001), ("minus", -.00001)):
                weights = {k: v.clone() if k == "log_std" else v + value for k, v in initial.items()}
                path = source_path.parent / "final_weights" / f"period_{period}_{sign}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": experiment.spec.source.PROTOCOL, "root": 410011, "period": period,
                    "sign": sign, "fisher_radius": experiment.spec.source.FISHER_RADIUS, "weights": weights}, path)
        experiment.write_json(source_path, {"status": "complete", "protocol": experiment.spec.source.PROTOCOL,
            "root": 410011, "cost": experiment.spec.source.budget(),
            "inherited_Stage123_cost": experiment.spec.source.source.budget()})
        real_fit = experiment.fit_panel
        fit_inputs = []
        def record_fit(rows):
            fit_inputs.append([(r["zero"]["seed"], r["zero"]["noise_seed"]) for r in rows])
            return real_fit(rows)
        stack.enter_context(patch.object(experiment, "fit_panel", side_effect=record_fit))
        result = experiment.run(410011, tmp_path / "candidate" / "result.json")
        assert result["cost"] == experiment.spec.budget()
        training_seeds = {r["scenario_seed"] for r in experiment.spec.training_roles(410011)}
        assert len(fit_inputs) == 6 and all(seed in training_seeds for rows in fit_inputs for seed, _ in rows)
    assert result["cost"]["native_episodes"] == 32 and result["cost"]["native_steps"] == 3200
    assert result["cost"]["upper_candidate_weight_steps"] == 8
    assert len(list((tmp_path / "candidate").rglob("*.pt"))) == 2
    for p, group in result["groups"].items():
        assert 0 <= group["training_native_return_fits"]["pooled"]["scale"] <= 1
        assert group["effects"]["curvature_ascent_minus_source_forecast"] == group["effects"]["curvature_ascent_minus_curvature_blinded"]
        assert group["lower_and_critics_frozen"] == "passed"
        experiment.joint.source.native.curves.support.assert_frozen(models[p], snapshots[p])


def test_fixed_budget_rosters_and_unchanged_gain_threshold():
    spec = experiment.spec
    budget = spec.budget()
    assert budget["training_episodes"] == 192 and budget["crossfit_episodes"] == 128
    assert budget["evaluation_episodes"] == 384 and budget["native_episodes"] == 704
    assert budget["native_steps"] == 844800
    assert spec.MINIMUM_GAIN == .5 and spec.PERIODS == (50, 100)
    seen = set()
    for root in spec.ROOTS:
        old = set(spec.source.evaluation_seeds(root))
        queries = spec.source.source.queries(root)
        old.update(s for q in queries for s in (q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values()))
        roles = spec.source.source.source.seed_roles(root, preflight=False)
        old.update(s for batch in roles["training_rounds"] for q in batch for s in (q["scenario_seed"], *q["noise_seeds"]))
        old.update(roles["native_evaluation"])
        train = [s for q in spec.training_roles(root) for s in (q["scenario_seed"], *q["noise_seeds"].values())]
        evaluation = spec.evaluation_seeds(root)
        assert len(train) == len(set(train)) == 48 and len(evaluation) == len(set(evaluation)) == 32
        assert not set(train).intersection(old | seen | set(evaluation))
        assert not set(evaluation).intersection(old | seen)
        seen.update(train + evaluation)
