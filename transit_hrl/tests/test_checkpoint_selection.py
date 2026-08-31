import copy
import unittest

import numpy as np
import torch

from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    RobustValidationCheckpointSelector,
    StateAlignedLexicographicCheckpointSelector,
    SMDPPPOConfig,
)
from freq_hrl.rl.training import (
    _apply_closed_loop_actor_guard,
    _closed_loop_guard_accepts,
    _validated_closed_loop_guard_snapshot,
    projection_consistency_schedule_scale,
    train_frequency_separated_ppo,
    train_joint_ppo,
)


class RobustValidationCheckpointSelectorTest(unittest.TestCase):
    @staticmethod
    def _guard_snapshot(
        rank=(0.0,),
        *,
        reward_violations=0,
        frequency_violations=0,
        frequency_merit=None,
        worst_frequency_violation=None,
    ):
        snapshot = {
            "contract": "unit_closed_loop_guard_v1",
            "rank": tuple(rank),
            "path_count": 1,
            "constraint_count": 1,
            "reward_violation_count": int(reward_violations),
            "frequency_violation_count": int(frequency_violations),
        }
        if frequency_merit is not None:
            snapshot["frequency_violation_merit"] = float(
                frequency_merit
            )
            snapshot["worst_frequency_violation"] = float(
                worst_frequency_violation
            )
        return snapshot

    @staticmethod
    def _closed_loop_model(*, restoration_filter=False):
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
            deployment_frequency_groupwise_robust=True,
            deployment_frequency_closed_loop_trust_region=True,
            deployment_frequency_closed_loop_trust_region_backtracks=4,
            deployment_frequency_closed_loop_restoration_filter=bool(
                restoration_filter
            ),
        ))

    @staticmethod
    def _step_actor_optimizer(actor, optimizer):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.sum() for parameter in actor.parameters()).backward()
        optimizer.step()

    def test_closed_loop_actor_guard_backtracks_both_actors_only(self):
        model = self._closed_loop_model()
        self._step_actor_optimizer(model.upper_actor, model.upper_actor_optimizer)
        self._step_actor_optimizer(model.lower_actor, model.lower_actor_optimizer)
        before_state = copy.deepcopy(model.state_dict())
        self._step_actor_optimizer(model.upper_actor, model.upper_actor_optimizer)
        self._step_actor_optimizer(model.lower_actor, model.lower_actor_optimizer)
        with torch.no_grad():
            for parameter in model.upper_actor.parameters():
                parameter.add_(0.5)
            for parameter in model.lower_actor.parameters():
                parameter.add_(0.5)
            for parameter in model.lower_value.parameters():
                parameter.add_(3.0)
        after_state = copy.deepcopy(model.state_dict())
        actor_key = next(iter(before_state["lower_actor"]))
        before_value = before_state["lower_actor"][actor_key].flatten()[0]
        after_value = after_state["lower_actor"][actor_key].flatten()[0]

        def evaluate(policy):
            current = policy.lower_actor.state_dict()[actor_key].flatten()[0]
            fraction = float(
                ((current - before_value) / (after_value - before_value)).item()
            )
            return self._guard_snapshot(
                rank=(0.0,) if fraction <= 0.5 + 1e-6 else (-1.0,),
                frequency_violations=(0 if fraction <= 0.5 + 1e-6 else 1),
            )

        metrics, _ = _apply_closed_loop_actor_guard(
            model,
            before_state=before_state,
            after_state=after_state,
            before_snapshot=self._guard_snapshot(),
            evaluate_fn=evaluate,
            max_backtracks=4,
        )

        self.assertEqual(
            metrics["deployment_frequency_closed_loop_guard_step_fraction"],
            0.5,
        )
        for actor_name in ("upper_actor", "lower_actor"):
            installed = getattr(model, actor_name).state_dict()
            for key in before_state[actor_name]:
                torch.testing.assert_close(
                    installed[key],
                    0.5 * (
                        before_state[actor_name][key]
                        + after_state[actor_name][key]
                    ),
                )
        for key, value in model.lower_value.state_dict().items():
            torch.testing.assert_close(value, after_state["lower_value"][key])
        installed_optimizer = model.lower_actor_optimizer.state_dict()["state"]
        before_optimizer = before_state["lower_actor_optimizer"]["state"]
        self.assertEqual(set(installed_optimizer), set(before_optimizer))
        for parameter_id in before_optimizer:
            self.assertEqual(
                set(installed_optimizer[parameter_id]),
                set(before_optimizer[parameter_id]),
            )
            for key, expected in before_optimizer[parameter_id].items():
                actual = installed_optimizer[parameter_id][key]
                if torch.is_tensor(expected):
                    torch.testing.assert_close(actual, expected)
                else:
                    self.assertEqual(actual, expected)

    def test_closed_loop_actor_guard_rolls_back_reward_violations(self):
        model = self._closed_loop_model()
        before_state = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for parameter in model.upper_actor.parameters():
                parameter.add_(1.0)
            for parameter in model.lower_actor.parameters():
                parameter.add_(1.0)
            for parameter in model.upper_value.parameters():
                parameter.add_(2.0)
        after_state = copy.deepcopy(model.state_dict())
        actor_key = next(iter(before_state["upper_actor"]))
        before_value = before_state["upper_actor"][actor_key].flatten()[0]

        def evaluate(policy):
            current = policy.upper_actor.state_dict()[actor_key].flatten()[0]
            changed = not torch.isclose(current, before_value, atol=1e-8)
            return self._guard_snapshot(
                reward_violations=int(changed),
            )

        metrics, _ = _apply_closed_loop_actor_guard(
            model,
            before_state=before_state,
            after_state=after_state,
            before_snapshot=self._guard_snapshot(),
            evaluate_fn=evaluate,
            max_backtracks=2,
        )
        self.assertEqual(
            metrics["deployment_frequency_closed_loop_guard_step_fraction"],
            0.0,
        )
        self.assertEqual(
            metrics["deployment_frequency_closed_loop_guard_accepted"], 0.0
        )
        for actor_name in ("upper_actor", "lower_actor"):
            installed = getattr(model, actor_name).state_dict()
            for key in before_state[actor_name]:
                torch.testing.assert_close(
                    installed[key], before_state[actor_name][key]
                )
        for key, value in model.upper_value.state_dict().items():
            torch.testing.assert_close(value, after_state["upper_value"][key])

    def test_closed_loop_restoration_backtracks_to_merit_safe_funnel(self):
        model = self._closed_loop_model(restoration_filter=True)
        before_state = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for parameter in model.upper_actor.parameters():
                parameter.add_(1.0)
            for parameter in model.lower_actor.parameters():
                parameter.add_(1.0)
        after_state = copy.deepcopy(model.state_dict())
        actor_key = next(iter(before_state["upper_actor"]))
        before_value = before_state["upper_actor"][actor_key].flatten()[0]
        after_value = after_state["upper_actor"][actor_key].flatten()[0]

        def snapshot(*, count, merit, worst, rank):
            return self._guard_snapshot(
                rank=rank,
                frequency_violations=count,
                frequency_merit=merit,
                worst_frequency_violation=worst,
            )

        baseline = snapshot(
            count=20,
            merit=0.05,
            worst=0.05,
            rank=(-0.05, -0.05, 0.02),
        )

        def evaluate(policy):
            current = policy.upper_actor.state_dict()[actor_key].flatten()[0]
            fraction = float(
                ((current - before_value) / (after_value - before_value)).item()
            )
            if fraction > 0.75:
                return snapshot(
                    count=2,
                    merit=0.04,
                    worst=0.20,
                    rank=(-0.20, -0.04, 0.02),
                )
            if fraction > 1e-8:
                return snapshot(
                    count=4,
                    merit=0.045,
                    worst=0.10,
                    rank=(-0.10, -0.045, 0.02),
                )
            return baseline

        metrics, selected = _apply_closed_loop_actor_guard(
            model,
            before_state=before_state,
            after_state=after_state,
            before_snapshot=baseline,
            evaluate_fn=evaluate,
            max_backtracks=4,
            restoration_filter=True,
            restoration_min_reduction=1e-3,
            restoration_funnel_limit=0.15,
        )

        prefix = "deployment_frequency_closed_loop_guard_"
        self.assertEqual(metrics[f"{prefix}step_fraction"], 0.5)
        self.assertEqual(metrics[f"{prefix}accepted"], 1.0)
        self.assertEqual(metrics[f"{prefix}restoration_phase_before"], "restoration")
        self.assertEqual(metrics[f"{prefix}restoration_merit_after"], 0.045)
        self.assertEqual(selected["frequency_violation_count"], 4)
        trace = metrics[f"{prefix}trial_trace"]
        self.assertEqual([item["fraction"] for item in trace], [1.0, 0.5])
        self.assertIn(
            "restoration_funnel_exceeded",
            trace[0]["rejection_reasons"],
        )
        self.assertTrue(trace[1]["accepted"])

    def test_closed_loop_restoration_switches_to_hard_maintenance(self):
        feasible = self._guard_snapshot(
            rank=(0.0, 0.0, 0.02),
            frequency_merit=0.0,
            worst_frequency_violation=0.0,
        )
        still_feasible = self._guard_snapshot(
            rank=(-1.0, -1.0, 0.001),
            frequency_merit=0.0,
            worst_frequency_violation=0.0,
        )
        newly_infeasible = self._guard_snapshot(
            rank=(-0.1, -0.01, 0.02),
            frequency_violations=1,
            frequency_merit=0.01,
            worst_frequency_violation=0.1,
        )
        self.assertTrue(_closed_loop_guard_accepts(
            still_feasible,
            feasible,
            restoration_filter=True,
            restoration_funnel_limit=0.0,
        ))
        self.assertFalse(_closed_loop_guard_accepts(
            newly_infeasible,
            feasible,
            restoration_filter=True,
            restoration_funnel_limit=0.0,
        ))

        infeasible = self._guard_snapshot(
            rank=(-0.15, -0.04, 0.02),
            frequency_violations=2,
            frequency_merit=0.04,
            worst_frequency_violation=0.15,
        )
        distributed_improvement = self._guard_snapshot(
            rank=(-0.10, -0.03, 0.02),
            frequency_violations=3,
            frequency_merit=0.03,
            worst_frequency_violation=0.10,
        )
        self.assertTrue(_closed_loop_guard_accepts(
            distributed_improvement,
            infeasible,
            restoration_filter=True,
            restoration_min_reduction=1e-3,
            restoration_funnel_limit=0.20,
        ))

    def test_restoration_guard_uses_squared_merit_tolerance(self):
        snapshot = self._guard_snapshot(
            rank=(-2e-10, -4e-20, 0.02),
            frequency_violations=1,
            frequency_merit=4e-20,
            worst_frequency_violation=2e-10,
        )
        validated = _validated_closed_loop_guard_snapshot(
            snapshot,
            restoration_filter=True,
        )
        self.assertEqual(validated["frequency_violation_count"], 1)
        self.assertEqual(validated["frequency_violation_merit"], 4e-20)

        inconsistent = self._guard_snapshot(
            rank=(0.0, 0.0, 0.02),
            frequency_violations=0,
            frequency_merit=1e-8,
            worst_frequency_violation=0.0,
        )
        with self.assertRaisesRegex(ValueError, "frequency-feasible"):
            _validated_closed_loop_guard_snapshot(
                inconsistent,
                restoration_filter=True,
            )

    def test_state_aligned_selector_uses_each_states_own_rank(self):
        selector = StateAlignedLexicographicCheckpointSelector(
            initial_score=10.0,
            initial_rank=(-2.0, -3.0, 10.0),
            rank_names=("negative_max_violation", "negative_l2", "reward"),
            initial_state={"step": -1},
            minimum_eligible_iteration=1,
        )
        first = selector.consider(
            score=100.0,
            rank=(-0.8, -1.2, 100.0),
            state={"step": 0},
            iteration=0,
        )
        second = selector.consider(
            score=5.0,
            rank=(-0.5, -0.9, 5.0),
            state={"step": 1},
            iteration=1,
        )
        third = selector.consider(
            score=200.0,
            rank=(-0.6, -0.1, 200.0),
            state={"step": 2},
            iteration=2,
        )

        self.assertFalse(first["checkpoint_selection_eligible"])
        self.assertTrue(second["checkpoint_selected"])
        self.assertFalse(third["checkpoint_selected"])
        self.assertEqual(selector.best_state, {"step": 1})
        self.assertEqual(selector.selected_iteration, 1)
        metadata = selector.metadata(total_iterations=3)
        self.assertEqual(
            metadata["checkpoint_selection_protocol"],
            "state_aligned_lexicographic_validation_v1",
        )
        self.assertEqual(
            metadata["checkpoint_selected_rank"]["negative_max_violation"],
            -0.5,
        )

    def test_trailing_window_rejects_an_isolated_validation_spike(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
            smoothing_window=3,
            min_delta=0.05,
        )

        first = selector.consider(
            score=0.6, state={"step": 0}, iteration=0
        )
        second = selector.consider(
            score=-0.6, state={"step": 1}, iteration=1
        )
        third = selector.consider(
            score=0.2, state={"step": 2}, iteration=2
        )

        self.assertFalse(first["checkpoint_selection_eligible"])
        self.assertFalse(first["checkpoint_selected"])
        self.assertFalse(second["checkpoint_selected"])
        self.assertTrue(third["checkpoint_selected"])
        self.assertEqual(selector.selected_iteration, 2)
        self.assertEqual(selector.best_state, {"step": 2})

        metadata = selector.metadata(total_iterations=5)
        self.assertEqual(
            metadata["checkpoint_selection_protocol"],
            "trailing_mean_material_improvement_v1",
        )
        self.assertEqual(metadata["checkpoint_smoothing_window"], 3)
        self.assertEqual(metadata["checkpoint_plateau_tail_iterations"], 2)

    def test_window_one_preserves_strict_best_score_selection(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
        )
        selector.consider(score=0.1, state={"step": 0}, iteration=0)
        selector.consider(score=0.05, state={"step": 1}, iteration=1)

        self.assertAlmostEqual(selector.best_score, 0.1)
        self.assertEqual(selector.selected_iteration, 0)
        self.assertEqual(selector.best_state, {"step": 0})
        self.assertEqual(
            selector.metadata(total_iterations=2)[
                "checkpoint_selection_protocol"
            ],
            "disjoint_validation_paths",
        )

    def test_invalid_selection_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                smoothing_window=0,
            )
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                min_delta=-1e-3,
            )
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                minimum_eligible_iteration=-2,
            )

    def test_minimum_iteration_excludes_anchor_and_selects_first_trained_state(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=10.0,
            initial_state={"step": -1},
            minimum_eligible_iteration=2,
        )
        self.assertFalse(
            selector.initial_history_fields()["checkpoint_selection_eligible"]
        )
        for iteration, score in enumerate((9.0, 8.0, 7.0)):
            result = selector.consider(
                score=score,
                state={"step": iteration},
                iteration=iteration,
            )
        self.assertTrue(result["checkpoint_selected"])
        self.assertEqual(selector.selected_iteration, 2)
        self.assertEqual(selector.best_state, {"step": 2})
        metadata = selector.metadata(total_iterations=4)
        self.assertEqual(
            metadata["checkpoint_selection_protocol"],
            "trailing_mean_material_improvement_minimum_iteration_v2",
        )
        self.assertEqual(metadata["checkpoint_minimum_eligible_iteration"], 2)
        self.assertTrue(metadata["checkpoint_has_eligible_selection"])

    def test_smdp_trainer_accepts_a_registered_custom_score_contract(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))
        score_calls = []

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
            }

        def robust_score(rows):
            score_calls.append(len(rows))
            return min(float(row["reward_mean"]) for row in rows)

        payload, heldout, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10, 20],
            eval_seeds=[30],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_score_fn=robust_score,
            checkpoint_score_contract="minimum_validation_reward_v1",
        )

        self.assertEqual(score_calls, [2, 2])
        self.assertEqual(payload["initial_validation_score"], 10.0)
        self.assertEqual(
            payload["checkpoint_score_contract"],
            "minimum_validation_reward_v1",
        )
        self.assertEqual(heldout[0]["reward_mean"], 30.0)

    def test_smdp_trainer_applies_delayed_projection_consistency_schedule(self):
        self.assertEqual(
            [
                projection_consistency_schedule_scale(
                    iteration=iteration,
                    total_iterations=4,
                    schedule="delayed_linear",
                    warmup_fraction=0.5,
                    ramp_fraction=0.5,
                )
                for iteration in range(4)
            ],
            [0.0, 0.0, 0.5, 1.0],
        )
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
            upper_projection_consistency_coef=2.0,
            lower_projection_consistency_coef=4.0,
        ))

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
                upper_projection_target=np.asarray(
                    [0.25], dtype=np.float32
                ),
                lower_projection_target=np.asarray(
                    [-0.25], dtype=np.float32
                ),
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
            }

        payload, _, trained = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10],
            eval_seeds=[30],
            iterations=4,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            projection_consistency_training_schedule="delayed_linear",
            projection_consistency_warmup_fraction=0.5,
            projection_consistency_ramp_fraction=0.5,
        )

        update_history = payload["history"][1:]
        self.assertEqual(
            [
                row["projection_consistency_schedule_scale"]
                for row in update_history
            ],
            [0.0, 0.0, 0.5, 1.0],
        )
        self.assertEqual(
            [
                row["upper_projection_consistency_effective_coef"]
                for row in update_history
            ],
            [0.0, 0.0, 1.0, 2.0],
        )
        self.assertEqual(
            payload["upper_projection_consistency_target_coef"], 2.0
        )
        self.assertEqual(
            payload["lower_projection_consistency_target_coef"], 4.0
        )
        self.assertEqual(
            trained.config.upper_projection_consistency_coef, 2.0
        )
        self.assertEqual(
            payload["projection_consistency_guard_training"]["upper"][
                "active_iteration_count"
            ],
            0.0,
        )

    def test_smdp_trainer_accepts_a_state_aligned_rank_contract(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
                "violation": float(seed) / 100.0,
            }

        payload, _, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10, 20],
            eval_seeds=[30],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_score_fn=lambda rows: min(
                float(row["reward_mean"]) for row in rows
            ),
            checkpoint_score_contract="minimum_validation_reward_v1",
            checkpoint_rank_fn=lambda rows: (
                -max(float(row["violation"]) for row in rows),
                min(float(row["reward_mean"]) for row in rows),
            ),
            checkpoint_rank_names=(
                "negative_max_violation",
                "worst_reward",
            ),
            checkpoint_rank_contract="feasibility_then_reward_v1",
            checkpoint_diagnostics_fn=lambda rows: {
                "worst_seed": max(int(row["reward_mean"]) for row in rows)
            },
            checkpoint_minimum_iteration=0,
        )

        self.assertEqual(
            payload["checkpoint_selection_protocol"],
            "state_aligned_lexicographic_validation_v1",
        )
        self.assertEqual(
            payload["checkpoint_rank_contract"],
            "feasibility_then_reward_v1",
        )
        self.assertEqual(
            payload["checkpoint_rank_names"],
            ["negative_max_violation", "worst_reward"],
        )
        self.assertEqual(payload["selected_checkpoint_iteration"], 0)
        self.assertEqual(
            payload["selected_checkpoint_diagnostics"], {"worst_seed": 20}
        )
        self.assertEqual(
            payload["history"][-1]["checkpoint_selection_diagnostics"],
            {"worst_seed": 20},
        )

    def test_smdp_trainer_can_require_a_trained_checkpoint(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
            }

        payload, _, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10],
            eval_seeds=[20],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_minimum_iteration=0,
        )

        self.assertFalse(
            payload["history"][0]["checkpoint_selection_eligible"]
        )
        self.assertEqual(payload["selected_checkpoint_iteration"], 0)
        self.assertTrue(payload["checkpoint_has_eligible_selection"])

    def test_smdp_trainer_freezes_anchor_state_replay_on_training_paths(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
            lower_deployment_frequency_rms_budget=1.0,
            lower_deployment_frequency_lambda_init=1.0,
            deployment_frequency_groupwise_robust=True,
            deployment_frequency_anchor_state_replay=True,
        ))
        calls = []

        def rollout_fn(_model, seed, train):
            calls.append((int(seed), bool(train)))
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return builder.build(), {"reward_mean": float(seed)}

        payload, _, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10],
            eval_seeds=[30],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            training_seed_fn=lambda root, iteration: (
                int(root) + 100 + int(iteration)
            ),
            deployment_frequency_reference_rollout_fn=(
                lambda replay_model, seed: rollout_fn(
                    replay_model, seed, False
                )[0]
            ),
        )

        self.assertEqual(calls.count((101, False)), 1)
        self.assertEqual(calls.count((101, True)), 1)
        self.assertEqual(
            payload["deployment_frequency_anchor_state_replay_seeds"],
            [101],
        )
        self.assertEqual(
            payload[
                "deployment_frequency_anchor_state_replay_lower_transitions"
            ],
            1,
        )
        self.assertEqual(
            payload["history"][1][
                "lower_deployment_frequency_group_count"
            ],
            2.0,
        )

    def test_smdp_trainer_uses_explicit_independent_reference_paths(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1,
            lower_state_dim=1,
            upper_action_dim=1,
            lower_action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
            lower_deployment_frequency_rms_budget=1.0,
            lower_deployment_frequency_lambda_init=1.0,
            deployment_frequency_groupwise_robust=True,
            deployment_frequency_anchor_state_replay=True,
        ))
        calls = []

        def rollout_fn(_model, seed, train):
            calls.append((int(seed), bool(train)))
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=0.0,
                done=True,
            )
            return builder.build(), {"reward_mean": float(seed)}

        payload, _, _ = train_frequency_separated_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10],
            eval_seeds=[30],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            training_seed_fn=lambda root, iteration: (
                int(root) + 100 + int(iteration)
            ),
            deployment_frequency_reference_rollout_fn=(
                lambda replay_model, seed: rollout_fn(
                    replay_model, seed, False
                )[0]
            ),
            deployment_frequency_reference_seeds=[701, 703],
        )

        self.assertEqual(calls.count((701, False)), 1)
        self.assertEqual(calls.count((703, False)), 1)
        self.assertNotIn((101, False), calls)
        self.assertEqual(
            payload["deployment_frequency_anchor_state_replay_seeds"],
            [701, 703],
        )
        self.assertEqual(
            payload[
                "deployment_frequency_anchor_state_replay_seed_source"
            ],
            "explicit",
        )
        self.assertEqual(
            payload[
                "deployment_frequency_anchor_state_replay_path_count"
            ],
            2,
        )

    def test_smdp_trainer_requires_and_audits_closed_loop_guard(self):
        def make_model():
            return self._closed_loop_model()

        def rollout_fn(_model, seed, train):
            builder = HierarchicalRolloutBuilder(gamma=0.99)
            builder.begin_upper(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
            )
            builder.add_lower(
                state=np.asarray([0.0], dtype=np.float32),
                action=np.asarray([0.0], dtype=np.float32),
                logp=0.0,
                value=0.0,
                reward=1.0,
                done=True,
            )
            return (builder.build() if train else None), {
                "reward_mean": float(seed),
            }

        common = dict(
            train_seeds=[1],
            selection_seeds=[10],
            eval_seeds=[20],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
        )
        with self.assertRaisesRegex(ValueError, "guard evaluation"):
            train_frequency_separated_ppo(model=make_model(), **common)

        calls = []

        def evaluate(_model):
            calls.append(1)
            return self._guard_snapshot()

        payload, _, _ = train_frequency_separated_ppo(
            model=make_model(),
            deployment_frequency_closed_loop_guard_fn=evaluate,
            **common,
        )
        self.assertEqual(len(calls), 3)
        self.assertTrue(
            payload["deployment_frequency_closed_loop_guard_enabled"]
        )
        self.assertEqual(
            payload["deployment_frequency_closed_loop_guard_evaluation_count"],
            3,
        )
        self.assertEqual(
            payload[
                "deployment_frequency_closed_loop_guard_selected_reward_violation_count"
            ],
            0,
        )
        self.assertEqual(
            payload[
                "deployment_frequency_closed_loop_guard_selected_frequency_violation_count"
            ],
            0,
        )
        self.assertEqual(
            payload["history"][1][
                "deployment_frequency_closed_loop_guard_attempted"
            ],
            1.0,
        )

    def test_joint_trainer_uses_the_same_custom_score_contract(self):
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=1,
            action_dim=1,
            hidden_dim=4,
            epochs=1,
            minibatch_size=4,
        ))

        def rollout_fn(_model, seed, train):
            batch = JointTrajectoryBatch(
                state=np.zeros((1, 1), dtype=np.float32),
                action=np.zeros((1, 1), dtype=np.float32),
                reward=np.zeros(1, dtype=np.float32),
                done=np.ones(1, dtype=np.float32),
                old_logp=np.zeros(1, dtype=np.float32),
                old_value=np.zeros(1, dtype=np.float32),
            )
            return (batch if train else None), {"reward_mean": float(seed)}

        payload, _, _ = train_joint_ppo(
            model=model,
            train_seeds=[1],
            selection_seeds=[10, 20],
            eval_seeds=[],
            iterations=1,
            rollout_fn=rollout_fn,
            objective_fn=lambda row: float(row["reward_mean"]),
            checkpoint_score_fn=lambda rows: min(
                float(row["reward_mean"]) for row in rows
            ),
            checkpoint_score_contract="minimum_validation_reward_v1",
        )
        self.assertEqual(payload["initial_validation_score"], 10.0)
        self.assertEqual(
            payload["checkpoint_score_contract"],
            "minimum_validation_reward_v1",
        )


if __name__ == "__main__":
    unittest.main()
