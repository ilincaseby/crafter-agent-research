import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.ppo.evaluate import evaluate
from src.ppo.experiment import run_experiment
from src.ppo.report import (
    aggregate_seeds,
    generate_report,
    read_csv,
    rolling_mean,
    summarize_episodes,
)
from src.ppo.train import (
    Agent,
    Config,
    CrafterEnv,
    compute_gae,
    ppo_loss,
    select_device,
    train,
)


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gae_matches_hand_calculation_and_cuts_episode_boundaries(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[0.5], [0.6], [0.7]])
        dones = torch.tensor([[False], [True], [False]])
        advantages, returns = compute_gae(
            rewards, values, dones, torch.tensor([0.8]), 0.9, 0.95
        )
        expected = torch.tensor([[1.04 + 0.9 * 0.95 * 1.4], [1.4], [3.02]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(returns, expected + values)

    def test_time_limit_bootstraps_but_does_not_leak_reset_episode(self):
        # A time-limit reward is corrected with gamma * V(final observation).
        rewards = torch.tensor([[1.0 + 0.9 * 5.0], [100.0]])
        values = torch.tensor([[2.0], [999.0]])
        advantages, returns = compute_gae(
            rewards,
            values,
            torch.tensor([[True], [False]]),
            torch.tensor([888.0]),
            0.9,
            0.95,
        )
        self.assertAlmostEqual(advantages[0].item(), 3.5)
        self.assertAlmostEqual(returns[0].item(), 5.5)

    def test_agent_and_loss_support_singleton_minibatches(self):
        agent = Agent(17)
        observations = torch.zeros((1, 3, 64, 64), dtype=torch.uint8)
        with torch.no_grad():
            actions, log_probs, entropy, values = agent.get_action_and_value(
                observations
            )
        self.assertEqual(actions.shape, (1,))
        self.assertAlmostEqual(entropy.item(), np.log(17), places=5)
        loss, metrics = ppo_loss(
            agent,
            observations,
            actions,
            log_probs,
            values,
            torch.tensor([1.0]),
            values + 1,
            Config(),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["approx_kl"], 0)
        self.assertAlmostEqual(metrics["value_loss"], 0.5)
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in agent.parameters()
                if parameter.grad is not None
            )
        )

    def test_surrogate_clipping_has_expected_sign_and_value(self):
        class FixedAgent:
            def get_action_and_value(self, observations, actions):
                return (
                    actions,
                    torch.log(torch.tensor([1.5, 0.5])),
                    torch.zeros(2),
                    torch.zeros(2),
                )

        loss, metrics = ppo_loss(
            FixedAgent(),
            torch.zeros(2),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([1.0, -1.0]),
            torch.zeros(2),
            Config(ent_coef=0),
        )
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertAlmostEqual(metrics["clip_fraction"], 1.0)

    def test_invalid_budgets_and_hyperparameters_fail_before_training(self):
        for config in (
            Config(total_timesteps=7, num_envs=2),
            Config(num_envs=0),
            Config(gamma=float("nan")),
            Config(learning_rate=-1),
            Config(ent_coef=-1),
            Config(target_kl=0),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                select_device("cuda")


class EnvironmentTests(unittest.TestCase):
    def test_adapter_distinguishes_death_and_timeout_and_preserves_rewards(self):
        observation = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("src.ppo.train.crafter.Env") as factory:
            game = factory.return_value
            game.reset.return_value = observation
            env = CrafterEnv(max_episode_steps=4)
            actual, _ = env.reset(seed=123)
            self.assertEqual(actual.shape, (3, 64, 64))
            self.assertEqual(actual.dtype, np.uint8)
            for discount, expected in ((0, (True, False)), (1, (False, True))):
                game.step.return_value = (
                    observation,
                    -0.1,
                    True,
                    {"discount": discount, "achievements": {"collect_wood": 1}},
                )
                _, reward, terminated, truncated, info = env.step(0)
                self.assertEqual((terminated, truncated), expected)
                self.assertEqual(reward, -0.1)
                self.assertEqual(info["achievements"]["collect_wood"], 1)
            env.reset()
            factory.assert_called_once()
            env.reset(seed=124)
            self.assertEqual(factory.call_count, 2)


class ReportTests(unittest.TestCase):
    def test_score_uses_zero_success_achievements_and_sample_seed_sd(self):
        rows = [
            {
                "episodic_return": "1",
                "episodic_length": "5",
                "achievements": json.dumps({"a": 1, "b": 0}),
            },
            {
                "episodic_return": "3",
                "episodic_length": "7",
                "achievements": json.dumps({"a": 0, "b": 0}),
            },
        ]
        summary = summarize_episodes(rows)
        self.assertEqual(summary["mean_return"], 2)
        self.assertAlmostEqual(summary["crafter_score"], np.sqrt(51) - 1)
        combined = aggregate_seeds([summary, {**summary, "mean_return": 4}])
        self.assertEqual(combined["mean_return"], 3)
        self.assertAlmostEqual(combined["std_across_seed_means"], np.sqrt(2))
        np.testing.assert_allclose(rolling_mean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])

    def test_partial_evaluations_are_not_treated_as_completed_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "seeds": [1],
                        "eval_episodes": 2,
                        "training": {"total_timesteps": 16, "max_episode_steps": 4},
                    }
                )
            )
            directory = root / "seed_1" / "eval_random"
            directory.mkdir(parents=True)
            with (directory / "episodes.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, ["episodic_return", "episodic_length", "achievements"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "episodic_return": 1,
                        "episodic_length": 4,
                        "achievements": json.dumps({"a": 1, "b": 0}),
                    }
                )
            summary = generate_report(root)
            self.assertIsNone(summary["aggregate"]["random"])
            self.assertIn("(partial)", (root / "report.html").read_text())


class SmokeTests(unittest.TestCase):
    def test_real_training_evaluation_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            config = Config(
                total_timesteps=16,
                num_envs=2,
                num_steps=3,
                minibatch_size=5,
                update_epochs=1,
                max_episode_steps=4,
                device="cpu",
                sync_envs=True,
                output_dir=str(output),
            )
            options = vars(argparse.Namespace(**config.__dict__)).copy()
            options.pop("seed")
            with patch("src.ppo.experiment.shutil.which", return_value=None):
                run_experiment(
                    argparse.Namespace(**options, seeds=[1], eval_episodes=2)
                )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["per_seed"]["1"]["training_steps"], 16)
            self.assertEqual(summary["paired_seeds"], 1)
            updates = read_csv(output / "seed_1/ppo/updates.csv")
            self.assertEqual([int(row["global_step"]) for row in updates], [6, 12, 16])
            self.assertTrue(
                all(np.isfinite(float(row["total_loss"])) for row in updates)
            )
            random_rows = read_csv(output / "seed_1/eval_random/episodes.csv")
            ppo_rows = read_csv(output / "seed_1/eval_ppo/episodes.csv")
            self.assertEqual(
                [row["environment_seed"] for row in random_rows],
                [row["environment_seed"] for row in ppo_rows],
            )
            checkpoint = output / "seed_1/ppo/checkpoint.pt"
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["global_step"], 16)
            with self.assertRaises(FileExistsError):
                train(replace(config, output_dir=str(output / "seed_1/ppo")))
            duplicate = Path(temporary) / "repeat_random"
            evaluate(
                duplicate,
                episodes=2,
                seed_start=1_000_000_000,
                max_episode_steps=4,
                device="cpu",
            )
            self.assertEqual(
                (duplicate / "episodes.csv").read_text(),
                (output / "seed_1/eval_random/episodes.csv").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
