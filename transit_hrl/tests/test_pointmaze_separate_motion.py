from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments import pointmaze_state_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from freq_hrl.experiments.pointmaze_plan_validity_branching import _ridge_fit_predict
from scripts import pointmaze_separate_motion_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from test_pointmaze_plan_hold import arguments


def fixture(directory):
    args = arguments()
    roles = response.path_roles(args.optimizer_seed, preflight=True)
    protocol = {"protocol_version":motion.WINDOWED_PROTOCOL_VERSION, "optimizer_seed":args.optimizer_seed,
                "horizon":300, "task_options":{}}
    tapes = motion.tapes_for(roles["fit"]+roles["evaluation"], protocol=protocol)
    data = {}
    for prefix, paths in (("train",roles["fit"]),("query",roles["evaluation"])):
        rows = motion.sample_rows(paths, horizon=300, bounded=False)
        x = np.zeros((len(rows),64,12),dtype=np.float32)
        y = np.zeros((len(rows),10),dtype=np.float32)
        for i,row in enumerate(rows):
            seed, step = row["seed"], row["step"]
            x[i,:,4:10] = tapes[seed][step-63:step+1]
            y[i,4:10] = tapes[seed][step+1]-tapes[seed][step]
        data[prefix+"_x"], data[prefix+"_y"] = x, y
        if prefix == "query":
            data["query_seeds"] = [r["seed"] for r in rows]
            data["query_steps"] = [r["step"] for r in rows]
            data["history_action_teacher_mean"] = np.full_like(y,.5)
            data["current_action_teacher_mean"] = np.full_like(y,.25)
    raw = directory/"source_raw"
    raw.mkdir()
    np.savez_compressed(raw/"state_response.npz", **data)
    source = {"status":"complete", "protocol":{"protocol_version":motion.SOURCE_PROTOCOL,
              "optimizer_seed":args.optimizer_seed}, "cells":[{"controller_selected_iteration":2,
              "state_seed_roles":roles, "raw_server_directory":str(raw),
              "scales":{"target_scale":[1.]*10}}]}
    controller = {"status":"complete", "protocol":protocol, "cells":[{"controller_selected_iteration":2}]}
    args.source_result, args.controller_result = directory/"source.json", directory/"controller.json"
    args.source_result.write_text(json.dumps(source))
    args.controller_result.write_text(json.dumps(controller))
    args.output = directory/"cell"/"result.json"
    return args, tapes, data


class SeparateMotionTest(unittest.TestCase):
    def test_generic_motion_features_use_observed_lags_and_levels_have_time_units(self):
        rng = np.random.default_rng(30)
        x = rng.normal(size=(12,64,3))
        model = CausalMotionForecaster(observed_dim=3, velocity_channels=(0,2),
                                      horizon_steps=(1,5),dt_seconds=.02)
        design = model.features(x)
        np.testing.assert_equal(design[:,:3],x[:,-1])
        for i,lag in enumerate(motion.LAGS):
            np.testing.assert_allclose(design[:,3+2*i:5+2*i],(x[:,-1,[0,2]]-x[:,-1-lag,[0,2]])/(lag*.02))
        with self.assertRaisesRegex(ValueError,"complete observed"):
            model.features(x[:,:50])
        rates = rng.normal(size=(12,2,3))
        model.fit(x,rates)
        predicted = model.predict_rates(x)
        np.testing.assert_allclose(model.predict_levels(x),x[:,-1:]+predicted*np.array([.02,.10])[None,:,None])
        z = model.features(x)
        for i in range(6):
            expected,_ = _ridge_fit_predict(z,rates.reshape(12,6)[:,i],z,alpha=1.)
            np.testing.assert_allclose(predicted.reshape(12,6)[:,i],expected,rtol=1e-12,atol=1e-12)

    def test_current_and_shuffle_preserve_current_observation_and_are_deterministic(self):
        rng = np.random.default_rng(30)
        x = rng.normal(size=(2,64,6))
        rows = [{"seed":i+1,"step":100} for i in range(2)]
        current = motion.observed_view(x,rows,method="current_repeat",root=30)
        shuffled = motion.observed_view(x,rows,method="shuffled_history",root=30)
        np.testing.assert_equal(current[:,-1],x[:,-1])
        np.testing.assert_equal(shuffled[:,-1],x[:,-1])
        np.testing.assert_equal(shuffled,motion.observed_view(x,rows,method="shuffled_history",root=30))
        model = CausalMotionForecaster(observed_dim=6,velocity_channels=(0,1),horizon_steps=motion.HORIZONS,dt_seconds=.01)
        np.testing.assert_equal(model.features(current)[:,6:],0)
        self.assertFalse(np.array_equal(shuffled[:,:-1],x[:,:-1]))
        np.testing.assert_equal(np.sort(shuffled[:,:-1],axis=1),np.sort(x[:,:-1],axis=1))

    def test_future_changes_labels_not_causal_features_and_wrong_prefix_is_rejected(self):
        protocol = {"horizon":300,"task_options":{}}
        tapes = motion.tapes_for([30],protocol=protocol)
        rows = [{"seed":30,"step":100}]
        x,y = motion.labeled_motion(tapes,rows)
        changed = deepcopy(tapes)
        changed[30][101:] += 2.
        other_x,other_y = motion.labeled_motion(changed,rows,cached_history=x)
        np.testing.assert_equal(x,other_x)
        self.assertFalse(np.array_equal(y,other_y))
        changed[30][100] += 1.
        with self.assertRaisesRegex(ValueError,"prefix differs"):
            motion.labeled_motion(changed,rows,cached_history=x)

    def test_forecasts_have_equal_capacity_and_no_query_dependent_fitting(self):
        rng = np.random.default_rng(30)
        train_x = rng.normal(size=(12,64,6))
        train_y = rng.normal(size=(12,5,6))
        train_rows = [{"seed":1,"step":64+5*i} for i in range(12)]
        query_x = rng.normal(size=(3,64,6))
        query_rows = [{"seed":2,"step":64+5*i} for i in range(3)]
        models,predicted = motion.fit_models(train_x,train_y,train_rows,query_x,query_rows,root=30)
        changed,other = motion.fit_models(train_x,train_y,train_rows,query_x+1000,query_rows,root=30)
        for method in motion.METHODS:
            np.testing.assert_equal(models[method].fitted["weights"],changed[method].fitted["weights"])
            self.assertEqual(models[method].fitted["parameter_count"],450)
            self.assertEqual(predicted[method].shape,(3,5,6))
        self.assertFalse(np.array_equal(predicted["history"],other["history"]))
        with self.assertRaisesRegex(ValueError,"paths overlap"):
            motion.fit_models(train_x,train_y,train_rows,query_x,train_rows[:3],root=30)

    def test_external_replacement_cannot_change_physical_prediction(self):
        rng = np.random.default_rng(30)
        mean = rng.normal(size=(3,10)).astype(np.float32)
        rates = rng.normal(size=(3,5,6))
        scale = np.arange(1,11,dtype=float)
        composed = motion.compose_state_mean(mean,rates,scales=scale)
        np.testing.assert_equal(composed[:,:4],mean[:,:4])
        np.testing.assert_allclose(composed[:,4:],rates[:,0]*.01/scale[4:],rtol=1e-7)
        changed = mean.copy()
        changed[:,:4] += 100
        np.testing.assert_equal(motion.compose_state_mean(changed,rates,scales=scale)[:,4:],composed[:,4:])

    def test_frozen_gate_needs_all_planning_controls_and_ties_fail(self):
        truth = np.zeros((2,5,6))
        predicted = {m:np.ones_like(truth)*(0 if m=="history" else 1) for m in
                     (*motion.METHODS,"lag1_extrapolation","zero")}
        metrics = motion.motion_metrics(truth,predicted)
        self.assertTrue(metrics["motion_gate_passed"])
        self.assertEqual(metrics["planning_target_rate_mse"]["current_repeat"],1.)
        predicted["zero"] = np.zeros_like(truth)
        self.assertFalse(motion.motion_metrics(truth,predicted)["planning_gate_passed"])
        predicted["current_repeat"][:,0] = 0
        self.assertFalse(motion.motion_metrics(truth,predicted)["one_step_gate_passed"])

    def test_registered_paths_counts_and_no_inherited_overlap(self):
        all_paths = set()
        for root in (208001,209011,209061):
            args = arguments(root)
            preflight = root==208001
            paths = motion.evaluation_paths(root,preflight=preflight)
            inherited = {"temporal_seed_roles":{str(i):seeds for i,seeds in enumerate(
                [seeds for module in (response,hold,temporal) for seeds in module.path_roles(root,preflight=preflight).values()])}}
            hold.validate_paths(args,{"fit":[],"evaluation":paths},inherited)
            self.assertFalse(all_paths.intersection(paths))
            all_paths.update(paths)
            count = len(motion.sample_rows(paths,horizon=args.horizon))
            self.assertEqual(count,56 if preflight else 1664)
            fit_paths = response.path_roles(root,preflight=preflight)["fit"]
            self.assertEqual(len(motion.sample_rows(fit_paths,horizon=args.horizon)),56 if preflight else 3328)
            points = (len(fit_paths)+2*len(paths))*(args.horizon+1)
            self.assertEqual(points,1806 if preflight else 38432)

    def test_run_cell_matches_source_and_accounts_without_weights_or_environment(self):
        with TemporaryDirectory() as temporary:
            args,_,data = fixture(Path(temporary))
            cell = motion.run_cell(args)
            self.assertEqual((cell["fit_rows"],cell["evaluation_rows"],cell["cached_bridge_rows"]),(56,56,56))
            self.assertEqual((cell["linear_fits"],cell["multi_rhs_linear_solves"],cell["scalar_rhs_count"]),(3,3,90))
            self.assertEqual(cell["generated_exogenous_tape_points"],1806)
            for key in ("new_environment_primitive_steps","controller_updates","physical_model_updates",
                        "controller_reconstruction_primitive_steps"):
                self.assertEqual(cell[key],0)
            self.assertEqual(len(cell["path_metrics"]),2)
            self.assertTrue(all(cell["cached_bridge_metrics"][m]["physical_prediction_unchanged"] for m in motion.METHODS))
            data["train_y"][0,4] += 1
            np.savez_compressed(Path(cell["raw_source_cache"]),**data)
            with self.assertRaisesRegex(ValueError,"labels differ"):
                motion.run_cell(args)

    def test_scheduler_uses_one_cpu_dynamic_pool_and_only_compact_sources(self):
        for preflight in (True,False):
            task = task_specification("unit_separate_motion",spec.roots(preflight=preflight)[0],
                                      preflight=preflight,protocol_spec=spec)
            self.assertEqual((task["cpu"],task["ram_mb"]),(1,1536))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"],[f"node00{i}" for i in range(1,7)])
            self.assertFalse(any("_raw" in p for p in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
