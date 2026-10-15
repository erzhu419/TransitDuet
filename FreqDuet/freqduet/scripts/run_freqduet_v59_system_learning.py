#!/usr/bin/env python3
"""Server-only learning smoke for chronological system credit; no main-policy promotion."""

import argparse
import copy
import csv
from dataclasses import replace
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from freqduet_system_credit import observable_decision
from freqduet_system_learner import (ArrivalMoments, SystemStateEncoder, SystemSAC,
                                    ingest_day, warm_start_actor)
import run_freqduet_v37_joint_lower as joint
from run_freqduet_v58_system_credit import CASES, CreditObserver, case_name, reference_trace


def csv_episode(row):
    """Use evaluate()'s CSV round trip before exact historical row comparison."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    output.seek(0)
    return pd.read_csv(output).iloc[0].to_dict()


def smoke(root, v56_root, case_index, updates):
    _, Runner, load_config, torch = joint.optimized_runtime(root)
    from lower.cost_replay_buffer import CostReplayBuffer
    from lower.resac_lagrangian import RESACLagrangianTrainer

    seed, scenario = CASES[case_index]
    label = case_name(seed, scenario)
    cell = root / "cells" / label
    replay = CostReplayBuffer(capacity=100000, seed=seed)
    reference = reference_trace(v56_root, seed, scenario)
    budget = None
    learner = None
    rows, update_rows = [], []

    for day in range(2):
        observer = CreditObserver()
        states, potentials = [], []
        moments = ArrivalMoments()
        encoder, actor_before = None, None

        class LearningRunner(Runner):
            def _lower_policy_action(self, obs, last_action=0.0, deterministic=False):
                nonlocal encoder, learner, actor_before
                if encoder is None:
                    # reset() constructs the day's protocol and geometry.
                    env = self.env
                    encoder = SystemStateEncoder(self.lower_state_dim, env.max_agent_num,
                                                 len(env.timetables), len(env.stations),
                                                 env.protocol.evaluation_end_time_s)
                    if learner is None:
                        cfg, old = config["lower"], self.lower_trainer
                        if (self.lower_terminal_action_mode != "transition"
                                or self.lower_causal_guard_policy_mask_enabled):
                            raise ValueError("V59 requires the frozen unmasked main action contract")
                        trainer = RESACLagrangianTrainer(
                            state_dim=encoder.dimension, hidden_dim=cfg["hidden_dim"],
                            action_range=cfg["action_range"], ensemble_size=old.ensemble_size,
                            beta=old.beta, beta_ood=old.beta_ood, weight_reg=old.weight_reg,
                            weight_reg_mode=old.weight_reg_mode, lr=cfg["lr"], lambda_lr=cfg["lambda_lr"],
                            gamma=1.0, soft_tau=old.soft_tau, auto_entropy=False, action_bins=joint.ACTIONS,
                            policy_sample_seed=seed + 59000, critic_aggregation=old.critic_aggregation,
                            action_limit_feature_index=old.action_limit_feature_index, device="cpu")
                        trainer.alpha = 1e-5
                        warm_start_actor(old.policy_net, trainer.policy_net, encoder.local_dim)
                        learner = SystemSAC(trainer)
                    actor_before = copy.deepcopy(learner.trainer.policy_net.state_dict())
                # Capture the same current causal evidence that the base runner
                # captures after choosing an action, so no old context is read.
                key = self._system_agent
                self._capture_lower_action_context(key, self._bus_for_agent(key))
                decision = observable_decision(self, obs, key, last_action)
                decision = replace(decision, pending_commands=tuple(observer.stream.pending_commands))
                moments.observe(self.env.headway_events.headways_s)
                encoded = encoder.encode(decision, moments)
                if day == 0:
                    with torch.no_grad():
                        before = self.lower_trainer.policy_net.dist_info(
                            torch.tensor(decision.local_state, dtype=torch.float32)[None, :])[0]
                        after = learner.trainer.policy_net.dist_info(torch.from_numpy(encoded)[None, :])[0]
                    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
                action = learner.trainer.policy_net.get_action(encoded, deterministic=(day == 0))
                if day == 0:
                    original = super()._lower_policy_action(obs, last_action, deterministic=True)
                    np.testing.assert_array_equal(np.asarray(action).reshape(-1), np.asarray(original).reshape(-1))
                states.append(encoded)
                potentials.append(moments.cv_squared)
                return np.asarray([float(np.asarray(action).item())], dtype=np.float32)

            def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
                self._system_agent = key
                action = super()._lower_action_for_agent(obs, key, last_action, deterministic)
                if not self._lower_terminal_action_masked(self._bus_for_agent(key)):
                    observer.stream.decide(observable_decision(self, obs, key, last_action),
                                           float(np.asarray(action).item()))
                return action

        joint.seed_runtime(torch, seed)
        config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
        runner = LearningRunner(joint.runtime_config(config, seed, cell / f"day{day}/logs", 40), device="cpu")
        checkpoints = joint.TRAIN_ROOT / "logs" / f"{joint.CURRENT}_seed{seed}/checkpoints"
        runner.load_checkpoint(checkpoints, ep=joint.BASE_EP, require_deployment_state=True)
        env = runner.env
        upper_before = copy.deepcopy(runner.upper_trainer.policy_net.state_dict())
        observer.attach(runner)
        # The deployment runner stays in frozen-evaluation mode. Only the new
        # learner below receives global replay and optimizer steps.
        episode = runner.run_episode(joint.BASE_EP, training=False, scenario_seed=scenario,
                                     record_diagnostics=False)
        if len(runner.replay_buffer):
            raise AssertionError("legacy per-trip replay was populated")
        for name, value in upper_before.items():
            torch.testing.assert_close(value, runner.upper_trainer.policy_net.state_dict()[name], rtol=0, atol=0)
        if day == 0:
            expected = dict(reference["snapshot"]["episode"])
            # run_episode does not add evaluate()'s seed/checkpoint metadata.
            pd.testing.assert_frame_equal(pd.DataFrame(observer.holds), reference["holds"], check_exact=True)
            differences = {name: [value, expected[name]] for name, value in episode.items()
                           if name in expected and isinstance(value, (float, int)) and value != expected[name]
                           and not name.startswith("wall_")}
            print("V59_NATIVE_CSV_DIFFERENCES " + json.dumps(differences), flush=True)
            joint.check_unchanged(csv_episode(episode), expected)
        if any(hold["action_s"] not in joint.ACTIONS for hold in observer.holds):
            raise AssertionError("learning emitted an action outside the registered seven bins")
        moments.observe(env.headway_events.headways_s)
        raw_cv = float(env.headway_events.summary()["headway_cv"])
        np.testing.assert_allclose(moments.cv_squared, raw_cv ** 2, rtol=0, atol=1e-10)
        if round(raw_cv, 4) != episode["headway_cv"]:
            raise AssertionError("unrounded arrival CV does not match the reported four-decimal metric")
        if day == 0:
            budget = moments.cv_squared
        audit = observer.stream.audit()
        inserted = ingest_day(replay, observer.stream, states, potentials, moments.cv_squared, day)
        for _ in range(updates):
            update_rows.append(learner.update(replay, runner.batch_size))
        lam_before = float(learner.trainer.lambda_param)
        lam_after = learner.dual_step(moments.cv_squared, budget)
        change = max(float((value - actor_before[name]).abs().max())
                     for name, value in learner.trainer.policy_net.state_dict().items())
        if change == 0.0:
            raise AssertionError("actor did not learn from system replay")
        rows.append(dict(day=day, stochastic_actions=bool(day), **inserted,
                         journey_min=float(episode["restricted_total_journey_horizon_min"]),
                         holding_s_per_trip=float(episode["holding_vehicle_seconds_per_launched_trip"]),
                         readiness_delay_s=float(episode["fleet_readiness_delay_mean_s"]),
                         regularity_budget_cv_squared=budget, lambda_before=lam_before, lambda_after=lam_after,
                         actor_parameter_max_change=change, upper_actor_unchanged=True,
                         legacy_replay_size=0, stream_audit=audit))
        print("V59_DAY " + json.dumps(rows[-1]), flush=True)
    cell.mkdir(parents=True, exist_ok=True)
    torch.save(dict(policy=learner.trainer.policy_net.state_dict(),
                    trainer=learner.trainer.training_state_dict(),
                    format="freqduet-system-learning-v59", objective="finite_day_passenger_time_cv_constraint",
                    encoder=encoder.__dict__, steps=learner.updates), cell / "learning_smoke.pt")
    (cell / "updates.json").write_text(json.dumps(update_rows, indent=2) + "\n")
    result = dict(status="learning_smoke_pass", case=label, episodes=rows,
                  state_dim=encoder.dimension, original_state_dim=encoder.local_dim,
                  batch_size=runner.batch_size, updates_per_day=updates, updates=learner.updates,
                  finite_day_discount=1.0, passenger_hours_scale=1000.0, entropy_alpha=1e-5,
                  exact_warm_start_day=True, main_policy_changed=False,
                  stage="engineering_smoke_not_effectiveness_test",
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    export = root / "exports" / label
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V59_SMOKE_DONE " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v56-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)), default=0)
    parser.add_argument("--updates-per-day", type=int, default=32)
    args = parser.parse_args()
    if args.updates_per_day < 1:
        parser.error("learning smoke requires at least one optimizer step")
    smoke(args.out_root, args.v56_root, args.case_index, args.updates_per_day)
