from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from freq_hrl.experiments import pointmaze_native_selection as experiment
from freq_hrl.experiments import pointmaze_native_selection_replication as replication
from scripts import pointmaze_third_update_replication_stage133_spec as previous_replication
from test_pointmaze_joint_reference import source_data
from test_pointmaze_native_upper_step import native_task
from test_pointmaze_update_isolation import ImmediatePool


@pytest.mark.parametrize("refresh,radial,expected", ((1., .5, "refresh"), (.1, 1., "stale"),
    (-1., -.1, "two_step"), (0., 0., "two_step"), (1., 1., "refresh")))
def test_selection_uses_measured_increment_and_can_decline_update(refresh, radial, expected):
    rows = [{m: {"episode_return": v} for m, v in
        (("two_step", baseline), ("refresh", baseline+refresh), ("stale", baseline+radial))}
        for baseline in (10., 20.)]
    selected = experiment.select_native_candidate(rows)
    assert selected["method"] == expected
    assert selected["mean_incremental_returns"] == pytest.approx({"two_step": 0., "refresh": refresh, "stale": radial})


@pytest.mark.parametrize("spec", (experiment.spec, replication.spec))
def test_native_selection_uses_separate_validation_before_evaluation_and_exact_budget(source_data, tmp_path, spec):
    models, predictor, cal, args, teachers = source_data
    kernel = experiment.kernel
    root = args.optimizer_seed = spec.ROOTS[0]
    cache = tmp_path/"source"/"result.json"
    groups = {}
    for period in spec.PERIODS:
        actor = experiment.joint.make_trainer(models[str(period)], teachers[str(period)], args).upper_actor
        with torch.no_grad():
            actor.net[0].bias.copy_(torch.arange(1., 9.)*.001)
        fit = {"scale": 1.}
        path = cache.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": spec.source.PROTOCOL, "root": root, "period": period,
            "method": "refresh", "fit": fit, "weights": actor.state_dict()}, path)
        groups[str(period)] = {"training_native_return_fits": {"refresh": {"pooled": fit}}}
    kernel.write_json(cache, {"status": "complete", "protocol": spec.source.PROTOCOL,
        "root": root, "cost": spec.source.budget(), "groups": groups})
    fit_seeds, selected_seeds, decisions = [], [], []
    original_fit, original_select = kernel.fit_method, experiment.select_native_candidate
    original_worker = experiment.curvature.worker_group
    def fit(rows, method):
        fit_seeds.extend(r["zero"]["seed"] for r in rows)
        return original_fit(rows, method)
    def select(rows):
        selected_seeds.append({r["two_step"]["seed"] for r in rows})
        decisions.append(original_select(rows))
        return decisions[-1]
    def worker(job):
        if job[3] in spec.evaluation_seeds(root):
            assert len(decisions) == spec.PERIODS.index(job[5])+1
        return original_worker(job)
    with ExitStack() as stack:
        native_task(stack)
        for name in ("LABEL_SCENARIOS", "TRAINING_SCENARIOS", "VALIDATION_SCENARIOS"):
            stack.enter_context(patch.object(spec, name, 1))
        stack.enter_context(patch.object(spec, "EVALUATION_EPISODES", 2))
        stack.enter_context(patch.object(spec, "arguments", return_value=args))
        stack.enter_context(patch.object(spec, "source_result", return_value=cache))
        stack.enter_context(patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, cal)))
        stack.enter_context(patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]))
        stack.enter_context(patch.object(experiment, "ProcessPoolExecutor", ImmediatePool))
        stack.enter_context(patch.object(kernel, "fit_method", side_effect=fit))
        stack.enter_context(patch.object(experiment, "select_native_candidate", side_effect=select))
        stack.enter_context(patch.object(experiment.curvature, "worker_group", side_effect=worker))
        result = experiment.run(root, tmp_path/"candidate"/"result.json", protocol_spec=spec)
        assert result["cost"] == spec.budget()
        assert set(fit_seeds) == {r["scenario_seed"] for r in spec.training_roles(root)}
        assert selected_seeds == [{r["scenario_seed"] for r in spec.validation_roles(root)}]*2
        for i, period in enumerate(spec.PERIODS):
            g = result["groups"][str(period)]
            assert g["selection"] == decisions[i]
            method = g["selection"]["method"]
            assert g["mean_metrics"]["selected"] == g["mean_metrics"][method]
            assert g["validation_roles"] == spec.validation_roles(root)
            assert g["evaluation_seeds"] == spec.evaluation_seeds(root)
            assert g["lower_and_critics_frozen"] == "passed"
            assert g["effects"]["refresh_minus_source_forecast"] == g["effects"]["refresh_minus_refresh_blinded"]
    assert result["cost"]["native_episodes"] == 174 and result["cost"]["native_steps"] == 17400
    assert result["cost"]["selection_episodes"] == 12 and result["cost"]["evaluation_episodes"] == 28
    assert result["cost"]["native_policy_selections"] == 2 and result["cost"]["evaluation_alias_assignments"] == 4
    assert len(list((tmp_path/"candidate").rglob("*.pt"))) == 4


def test_validation_rosters_are_disjoint_from_fit_evaluation_and_previous_steps():
    spec, seen = experiment.spec, set()
    assert spec.budget()["native_episodes"] == 6048 and spec.budget()["native_steps"] == 7257600
    assert spec.budget()["selection_episodes"] == 192 and spec.budget()["evaluation_episodes"] == 448
    for root in spec.ROOTS:
        parts = []
        for roles in (spec.label_roles(root), spec.training_roles(root), spec.validation_roles(root)):
            parts.append({s for r in roles for s in (r["scenario_seed"], *r["noise_seeds"].values())})
        parts.append(set(spec.evaluation_seeds(root)))
        old = set()
        for earlier in (spec.recipe, spec.source, spec.source.source.source):
            old.update(s for r in earlier.label_roles(root)+earlier.training_roles(root)
                for s in (r["scenario_seed"], *r["noise_seeds"].values()))
            old.update(earlier.evaluation_seeds(root))
        old.update(s for r in spec.source.source.training_roles(root)
            for s in (r["scenario_seed"], *r["noise_seeds"].values()))
        old.update(spec.source.source.evaluation_seeds(root))
        for part in parts:
            assert not part & (old|seen)
            seen.update(part)


def test_replication_keeps_recipe_two_step_source_and_fresh_roles():
    spec, seen = replication.spec, set()
    assert spec.source.PROTOCOL == "pointmaze_continuation_replication_stage131_v1"
    assert not set(spec.ROOTS) & set(experiment.spec.ROOTS)
    assert spec.VARIANTS == experiment.spec.VARIANTS and spec.CONTRASTS == experiment.spec.CONTRASTS
    assert spec.FISHER_RADIUS == experiment.spec.FISHER_RADIUS and spec.EPSILON == experiment.spec.EPSILON
    assert spec.budget()["native_episodes"] == 6496 and spec.budget()["native_steps"] == 7795200
    assert spec.budget()["selection_episodes"] == 192 and spec.budget()["evaluation_episodes"] == 896
    assert len(spec.ENDPOINTS) == 10
    for root in spec.ROOTS:
        old = set()
        for earlier in (spec.source, spec.source.source, previous_replication):
            old.update(s for r in earlier.label_roles(root)+earlier.training_roles(root)
                for s in (r["scenario_seed"], *r["noise_seeds"].values()))
            old.update(earlier.evaluation_seeds(root))
        parts = [{s for r in roles for s in (r["scenario_seed"], *r["noise_seeds"].values())}
            for roles in (spec.label_roles(root), spec.training_roles(root), spec.validation_roles(root))]
        parts.append(set(spec.evaluation_seeds(root)))
        for part in parts:
            assert not part & (old|seen)
            seen.update(part)
