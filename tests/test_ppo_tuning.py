from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.ppo.report import read_csv
from src.ppo.train import Agent, Config, CsvLog, ppo_loss, train
from src.ppo.tuning import (
    ALTERNATIVES,
    build_parser,
    collect,
    finalize,
    initialize,
    load_plan,
    plan_digest,
    run_task,
    task_config,
    validate_plan,
    write_json,
)
from src.ppo.tuning_report import generate_report


class AuditRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_rejects_fractional_boolean_and_nonfinite_parameters(self):
        for config in (
            Config(num_steps=1.5),
            Config(num_envs=True),
            Config(seed=0.5),
            Config(update_epochs=float("inf")),
            Config(seed=True),
            Config(resource_mask=1),
            Config(sync_envs="false"),
            Config(gamma=float("nan")),
            Config(gae_lambda=float("inf")),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_singleton_policy_advantage_is_not_erased(self):
        agent = Agent(17)
        images = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
        with torch.no_grad():
            actions, logs, _, values = agent.get_action_and_value(images)
        loss, metrics = ppo_loss(
            agent,
            images,
            actions,
            logs,
            values,
            torch.tensor([2.0]),
            values,
            Config(ent_coef=0, vf_coef=0),
        )
        self.assertAlmostEqual(metrics["policy_loss"], -2)
        loss.backward()
        self.assertGreater(agent.actor.bias.grad.abs().sum().item(), 0)

    def test_startup_failure_closes_workers_and_propagates_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            vector = Mock()
            with patch(
                "src.ppo.train.gym.vector.SyncVectorEnv", return_value=vector
            ), patch(
                "src.ppo.train.SummaryWriter",
                side_effect=OSError("Disk quota exceeded"),
            ):
                with self.assertRaisesRegex(OSError, "Disk quota"):
                    train(
                        Config(
                            sync_envs=True,
                            device="cpu",
                            output_dir=str(Path(temporary) / "run"),
                        )
                    )
            vector.close.assert_called_once()

    def test_zero_clock_interval_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch("src.ppo.train.time.monotonic", return_value=10.0):
                train(
                    Config(
                        total_timesteps=8,
                        num_envs=1,
                        num_steps=4,
                        minibatch_size=4,
                        max_episode_steps=4,
                        update_epochs=1,
                        sync_envs=True,
                        device="cpu",
                        output_dir=str(output),
                    )
                )
            rows = read_csv(output / "updates.csv")
            self.assertTrue(all(row["sps"] == "" for row in rows))
            self.assertEqual(rows[-1]["global_step"], "8")


def small_args(root, seeds=(1, 2, 3)):
    return build_parser().parse_args(
        [
            "init",
            "--output-dir",
            str(root),
            "--total-timesteps",
            "16",
            "--num-envs",
            "2",
            "--num-steps",
            "4",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "4",
            "--max-episode-steps",
            "4",
            "--sync-envs",
            "--device",
            "cpu",
            "--validation-episodes",
            "2",
            "--confirmation-episodes",
            "2",
            "--seeds",
            *map(str, seeds),
        ]
    )


def fake_train(config, on_checkpoint=None):
    path = Path(config.output_dir)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", asdict(config))
    checkpoint = path / "checkpoint.pt"
    torch.save(
        {"config": asdict(config), "global_step": config.total_timesteps}, checkpoint
    )
    log = CsvLog(
        path / "updates.csv",
        [
            "global_step",
            "mean_episodic_return",
            "approx_kl",
            "clip_fraction",
            "entropy",
            "value_loss",
            "explained_variance",
            "learning_rate",
        ],
    )
    log.write(
        {
            "global_step": config.total_timesteps,
            "mean_episodic_return": 2,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
            "entropy": 1,
            "value_loss": 0.1,
            "explained_variance": 0.8,
            "learning_rate": config.learning_rate,
        }
    )
    log.close()
    if on_checkpoint:
        on_checkpoint()
    return checkpoint


def fake_evaluate(output, *, checkpoint, episodes, seed_start, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = torch.load(checkpoint, weights_only=True)["config"]
    write_json(
        output / "config.json",
        {
            "action_selection": "sampled categorical",
            "resource_mask": True,
            "encoder": "impala",
            "frame_stack": 1,
        },
    )
    # Make gamma the known validation winner, but worse on the untouched test set.
    is_test = "confirmation" in output.parts
    reward = (
        6
        if config["gamma"] == 0.95 and not is_test
        else 1 if config["gamma"] == 0.95 else 3
    )
    log = CsvLog(
        output / "episodes.csv",
        [
            "environment_seed",
            "episodic_return",
            "episodic_length",
            "achievements",
        ],
    )
    for episode in range(episodes):
        log.write(
            {
                "environment_seed": seed_start + episode,
                "episodic_return": reward + config["seed"] * 0.1,
                "episodic_length": 4,
                "achievements": '{"a": 1, "b": 0}',
            }
        )
    log.close()
    return output


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_21_tasks_change_exactly_one_parameter_and_obey_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = build_parser().parse_args(
                ["init", "--output-dir", str(Path(temporary) / "sweep")]
            )
            plan = initialize(args)
            self.assertEqual(len(plan["tasks"]), 21)
            self.assertEqual(plan["base_config"]["total_timesteps"], 200_000)
            base = Config(**plan["base_config"])
            for index in range(21):
                _, config = task_config(plan, index, Path(temporary))
                fields = [
                    field
                    for field in asdict(config)
                    if field not in ("seed", "output_dir")
                    and getattr(config, field) != getattr(base, field)
                ]
                self.assertEqual(len(fields), 0 if index % 7 == 0 else 1)
                self.assertEqual(config.total_timesteps, 200_000)
                self.assertEqual(config.encoder, "impala")
                self.assertTrue(config.resource_mask)
                self.assertEqual(config.frame_stack, 1)
            for invalid in (
                {
                    **plan,
                    "base_config": {**plan["base_config"], "total_timesteps": 200008},
                },
                {**plan, "confirmation_seed_start": plan["validation_seed_start"]},
                {
                    **plan,
                    "candidates": [
                        *plan["candidates"][:-1],
                        {
                            "name": "kl_0015",
                            "changes": {"gamma": 0.95, "target_kl": 0.015},
                        },
                    ],
                },
            ):
                invalid["sha256"] = plan_digest(invalid)
                with self.assertRaises(ValueError):
                    validate_plan(invalid)
            changed = {**plan, "base_config": {**plan["base_config"], "gamma": 0.98}}
            with self.assertRaisesRegex(ValueError, "plan changed"):
                validate_plan(changed)

    def test_all_seed_ranking_frozen_before_confirmation_and_partial_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            plan = initialize(small_args(root))
            with patch("src.ppo.tuning.train", fake_train), patch(
                "src.ppo.tuning.evaluate", fake_evaluate
            ):
                run_task(root, 1)
                partial = collect(root)
                self.assertEqual(partial["ranking"], [])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    finalize(root)
                self.assertFalse((root / "selection.json").exists())
                with self.assertRaises(FileExistsError):
                    run_task(root, 1)
                for index in range(21):
                    if index != 1:
                        run_task(root, index)
                result = finalize(root)
                self.assertEqual(result["selection"]["candidate"], "gamma_095")
                self.assertEqual(len(result["ranking"]), 7)
                baseline = result["confirmation"]["baseline"]
                winner = result["confirmation"]["gamma_095"]
                self.assertGreater(
                    np.mean([row["mean_return"] for row in baseline]),
                    np.mean([row["mean_return"] for row in winner]),
                )
                selection = (root / "selection.json").read_bytes()
                again = finalize(root)
                self.assertEqual((root / "selection.json").read_bytes(), selection)
                self.assertEqual(again["selection"]["candidate"], "gamma_095")
                report = generate_report(root)
                self.assertTrue(report["all_complete"])
                for curve in report["training_curves"].values():
                    self.assertEqual(curve["n_seeds"], 3)
                    self.assertIsNotNone(curve["std"])
                for name in (
                    "report.html",
                    "training_return.png",
                    "validation_return.png",
                    "optimization.png",
                ):
                    self.assertGreater((root / name).stat().st_size, 100)
                validation = read_csv(
                    root / "gamma_095" / "seed_1" / "validation" / "episodes.csv"
                )
                test = read_csv(
                    root / "gamma_095" / "seed_1" / "confirmation" / "episodes.csv"
                )
                self.assertFalse(
                    {row["environment_seed"] for row in validation}
                    & {row["environment_seed"] for row in test}
                )
            path = root / "gamma_095" / "seed_1" / "validation" / "episodes.csv"
            path.write_text(path.read_text().replace("1800000000", "1800000099"))
            with self.assertRaisesRegex(ValueError, "Validation count/maps"):
                collect(root)

    def test_real_impala_task_and_failure_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sweep"
            initialize(small_args(root, seeds=(1,)))
            status = run_task(root, 0)
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["training_steps"], 16)
            state = torch.load(
                root / "baseline" / "seed_1" / "ppo" / "checkpoint.pt",
                weights_only=True,
            )
            self.assertEqual(state["global_step"], 16)
            self.assertEqual(state["config"]["encoder"], "impala")
            self.assertEqual(len(collect(root)["ranking"]), 1)
            with patch(
                "src.ppo.tuning.train",
                side_effect=RuntimeError("deliberate test failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    run_task(root, 1)
            failed = json.loads(
                (root / "gamma_095" / "seed_1" / "status.json").read_text()
            )
            self.assertEqual(failed["status"], "failed or interrupted")
            self.assertIn("deliberate test failure", failed["error"])
            with self.assertRaises(ValueError):
                collect(root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
