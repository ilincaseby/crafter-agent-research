import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.sebi import direct_future_prediction as dfp


class ShortEnvironment:
    def __init__(self):
        self.action_space = type("ActionSpace", (), {"n": 17})()
        self.closed = False

    def reset(self):
        self.steps = 0
        return np.full((32, 32, 3), 100, dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        inventory = {
            name: dfp.constants.items[name]["initial"] for name in dfp.INVENTORY_NAMES
        }
        inventory["wood"] = 1
        inventory["health"] = 0 if done else 9
        achievements = dict.fromkeys(dfp.achievement_list, 0)
        achievements["collect_wood"] = 1
        return (
            np.full((32, 32, 3), 100 + self.steps * 20, dtype=np.uint8),
            -0.9 if done else 1.0,
            done,
            {
                "inventory": inventory,
                "achievements": achievements,
                "discount": float(not done),
            },
        )

    def close(self):
        self.closed = True


class DFPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def observation(self, value=0):
        return torch.full((2, 3, 32, 32), value, dtype=torch.uint8)

    def test_measurement_scales_and_first_unlock_not_repeat_counts(self):
        initial = dfp.initial_measurements(10)
        info = {
            "inventory": {name: 0 for name in dfp.INVENTORY_NAMES},
            "achievements": dict.fromkeys(dfp.achievement_list, 0),
        }
        info["inventory"]["health"] = 9
        info["inventory"]["wood"] = 3
        info["achievements"]["collect_wood"] = 8
        current = dfp.measurements(info, 2.5, 10)
        self.assertAlmostEqual(current[0].item(), 0.25)
        self.assertEqual(current[dfp.MEASUREMENT_NAMES.index("health")].item(), 1)
        self.assertAlmostEqual(
            current[dfp.MEASUREMENT_NAMES.index("wood")].item(), 1 / 3
        )
        self.assertEqual(
            current[dfp.MEASUREMENT_NAMES.index("achievement_collect_wood")].item(), 1
        )
        self.assertEqual(initial[0].item(), 0)
        self.assertEqual(initial[dfp.MEASUREMENT_NAMES.index("health")].item(), 1)
        self.assertEqual(initial[-len(dfp.achievement_list) :].sum().item(), 0)
        with self.assertRaises(ValueError):
            dfp.measurements(info, float("nan"), 10)

    def test_future_targets_use_exact_horizon_offsets_and_selected_action(self):
        replay = dfp.FutureReplay(10, (1, 3), seed=42)
        replay.start_episode(torch.tensor([10.0, 3.0]))
        for index, value in enumerate((11, 13, 16, 20)):
            replay.append(
                self.observation(index), index, torch.tensor([float(value), 3.0])
            )
        self.assertEqual(len(replay), 2)
        self.assertEqual(len(replay.pending), 2)
        torch.testing.assert_close(
            replay.entries[0].targets, torch.tensor([[1.0, 0.0], [6.0, 0.0]])
        )
        torch.testing.assert_close(
            replay.entries[1].targets, torch.tensor([[2.0, 0.0], [9.0, 0.0]])
        )
        self.assertEqual([entry.action for entry in replay.entries], [0, 1])
        self.assertTrue(replay.entries[0].valid.all().item())

    def test_death_is_absorbing_but_time_limit_masks_unknown_future(self):
        for death in (False, True):
            with self.subTest(death=death):
                replay = dfp.FutureReplay(10, (1, 4), seed=42)
                replay.start_episode(torch.tensor([0.0, 1.0]))
                replay.append(self.observation(), 0, torch.tensor([1.0, 1.0]))
                replay.append(
                    self.observation(1),
                    1,
                    torch.tensor([2.0, 0.0]),
                    episode_end=True,
                    terminated=death,
                )
                self.assertEqual(len(replay), 2)
                self.assertFalse(replay.pending)
                self.assertFalse(replay.history)
                self.assertEqual(replay.entries[0].valid.tolist(), [True, death])
                self.assertEqual(replay.entries[1].valid.tolist(), [True, death])
                if death:
                    torch.testing.assert_close(
                        replay.entries[0].targets[1], torch.tensor([2.0, -1.0])
                    )
                    torch.testing.assert_close(
                        replay.entries[1].targets[1], torch.tensor([1.0, -1.0])
                    )
                replay.start_episode(torch.tensor([100.0, 1.0]))
                replay.append(
                    self.observation(2),
                    2,
                    torch.tensor([101.0, 1.0]),
                    episode_end=True,
                    terminated=True,
                )
                self.assertEqual(replay.entries[-1].targets[:, 0].tolist(), [1.0, 1.0])

    def test_short_budget_does_not_fabricate_targets_and_replay_is_bounded(self):
        replay = dfp.FutureReplay(3, (2, 4), seed=42)
        replay.start_episode(torch.zeros(2))
        replay.append(self.observation(), 0, torch.ones(2))
        replay.finish(terminated=False)
        self.assertEqual(len(replay), 0)
        self.assertEqual(replay.discarded, 1)
        for index in range(5):
            replay.start_episode(torch.zeros(2))
            replay.append(
                self.observation(index),
                index,
                torch.ones(2),
                episode_end=True,
                terminated=True,
            )
        self.assertEqual(len(replay), 3)
        self.assertEqual({entry.action for entry in replay.entries}, {2, 3, 4})
        self.assertEqual(len(replay.sample(2)), 2)
        with self.assertRaises(ValueError):
            replay.sample(4)

    def test_loss_only_supervises_executed_actions_and_observed_horizons(self):
        predictions = torch.zeros(2, 3, 2, 3, requires_grad=True)
        actions = torch.tensor([0, 2])
        targets = torch.ones(2, 2, 3)
        valid = torch.tensor([[True, False], [False, True]])
        loss, reward_loss, auxiliary_loss, _ = dfp.prediction_loss(
            predictions,
            actions,
            targets,
            valid,
            0.1,
        )
        self.assertAlmostEqual(loss.item(), 0.55)
        self.assertEqual(reward_loss.item(), 0.5)
        self.assertEqual(auxiliary_loss.item(), 0.5)
        loss.backward()
        expected_nonzero = torch.zeros_like(predictions, dtype=torch.bool)
        expected_nonzero[0, 0, 0] = True
        expected_nonzero[1, 2, 1] = True
        self.assertTrue(torch.equal(predictions.grad != 0, expected_nonzero))
        with self.assertRaises(ValueError):
            dfp.prediction_loss(
                predictions, actions, targets, torch.zeros_like(valid), 0.1
            )

    def test_decisions_ignore_auxiliary_predictions_and_short_horizons(self):
        class Predictor(torch.nn.Module):
            num_actions = 3

            def forward(self, observation):
                result = torch.zeros(1, 3, 2, len(dfp.MEASUREMENT_NAMES))
                result[:, 0, :, 1:] = 1_000_000
                result[:, 0, 0, 0] = 100
                result[:, 1, -1, 0] = 0.2
                result[:, 2, -1, 0] = 0.1
                return result

        model = Predictor()
        self.assertEqual(
            dfp.choose_action(
                model, self.observation(), 0, random.Random(42), self.device
            ),
            1,
        )
        with patch.object(
            model,
            "forward",
            side_effect=AssertionError("Exploration need not run the CNN"),
        ):
            action = dfp.choose_action(
                model, self.observation(), 1, random.Random(42), self.device
            )
        self.assertIn(action, range(3))

    def test_config_and_epsilon_schedule(self):
        config = dfp.DFPConfig(learning_starts=10, epsilon_decay_steps=100)
        self.assertEqual(dfp.epsilon_at(config, 0), 1)
        self.assertAlmostEqual(dfp.epsilon_at(config, 50), 0.525)
        self.assertAlmostEqual(dfp.epsilon_at(config, 1000), 0.05)
        for changes in (
            {"horizons": ()},
            {"horizons": (4, 1)},
            {"horizons": (1, 1)},
            {"horizons": (0, 1)},
            {"replay_capacity": 1},
            {"height": 16},
            {"reward_scale": 0},
            {"epsilon_end": 2},
            {"learning_rate": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dfp.DFPConfig(**changes)
        with patch.dict(os.environ, {"HORIZONS": "1, 3, 8", "FRAME_STACK": "2"}):
            config = dfp.DFPConfig.from_env()
        self.assertEqual(config.horizons, (1, 3, 8))
        self.assertEqual(config.frame_stack, 2)

    def test_small_cnn_output_shape_and_supervised_training_reduces_loss(self):
        config = dfp.DFPConfig(
            frame_stack=2, horizons=(1, 3), height=32, width=32, batch_size=2
        )
        model = dfp.FuturePredictor(config, 17)
        output = model(torch.zeros(2, 2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 17, 2, len(dfp.MEASUREMENT_NAMES)))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        targets = torch.zeros(2, len(dfp.MEASUREMENT_NAMES))
        targets[:, 0] = 0.4
        examples = [
            dfp.FutureExample(
                self.observation(120), action, targets, torch.ones(2, dtype=torch.bool)
            )
            for action in (0, 1)
        ]
        losses = [
            dfp.train_batch(model, optimizer, examples, config, self.device)["loss"]
            for _ in range(10)
        ]
        self.assertLess(losses[-1], losses[0])

    def test_pooling_matches_adaptive_outputs_and_gradients(self):
        pool = dfp.SpatialMeanPool4x4()
        for height, width in ((4, 4), (8, 8), (8, 12), (5, 7), (8, 9)):
            with self.subTest(shape=(height, width)):
                source = torch.randn(
                    2, 3, height, width, dtype=torch.float64, requires_grad=True
                )
                reference = source.detach().clone().requires_grad_(True)
                actual = pool(source)
                expected = torch.nn.functional.adaptive_avg_pool2d(reference, (4, 4))
                torch.testing.assert_close(actual, expected)
                weights = torch.randn_like(actual)
                (actual * weights).sum().backward()
                (expected * weights).sum().backward()
                torch.testing.assert_close(source.grad, reference.grad)
        self.assertEqual(pool.state_dict(), {})
        model = dfp.FuturePredictor(dfp.DFPConfig(), 17)
        self.assertFalse(
            any(
                isinstance(module, torch.nn.AdaptiveAvgPool2d)
                for module in model.modules()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_training_preserves_strict_determinism(self):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        try:
            dfp.set_seed(42)
            for height, width in ((64, 64), (40, 56)):
                with self.subTest(shape=(height, width)):
                    config = dfp.DFPConfig(
                        frame_stack=2,
                        horizons=(1, 3),
                        height=height,
                        width=width,
                        batch_size=2,
                    )
                    model = dfp.FuturePredictor(config, 17).to("cuda")
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=config.learning_rate
                    )
                    examples = [
                        dfp.FutureExample(
                            torch.full((2, 3, height, width), 120, dtype=torch.uint8),
                            action,
                            torch.ones(2, len(dfp.MEASUREMENT_NAMES)),
                            torch.ones(2, dtype=torch.bool),
                        )
                        for action in (0, 1)
                    ]
                    before = model.predictions.weight.detach().clone()
                    metrics = dfp.train_batch(
                        model, optimizer, examples, config, torch.device("cuda")
                    )
                    self.assertTrue(np.isfinite(metrics["loss"]))
                    self.assertFalse(torch.equal(before, model.predictions.weight))
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    self.assertFalse(
                        torch.is_deterministic_algorithms_warn_only_enabled()
                    )
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)
            if previous_workspace is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace

    def test_training_saves_original_rewards_and_exact_budget(self):
        config = dfp.DFPConfig(
            total_steps=7,
            frame_stack=2,
            horizons=(1, 4),
            height=32,
            width=32,
            replay_capacity=8,
            batch_size=2,
            learning_starts=1,
            train_every=1,
            log_every=2,
            checkpoint_every=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            env = ShortEnvironment()
            with (
                patch.dict(os.environ, {"DEVICE": "cpu"}),
                patch.object(dfp.crafter, "Env", return_value=env),
                patch.object(
                    dfp.crafter, "Recorder", side_effect=lambda env, *args, **kw: env
                ),
            ):
                dfp.run(config, Path(directory))
            root = Path(directory)
            records = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["environment_steps"] for row in records], [2, 4, 6, 7]
            )
            self.assertEqual(records[-1]["updates"], 6)
            self.assertEqual(records[-1]["episodes"], 3)
            self.assertAlmostEqual(records[-1]["average_reward"], 0.1)
            self.assertEqual(records[-1]["pending_targets"], 0)
            self.assertTrue(env.closed)
            self.assertFalse((root / "plots/actor_loss.png").exists())
            self.assertEqual(
                (root / "plots/reward_prediction_loss.png").read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            checkpoint = torch.load(
                root / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["environment_steps"], 7)
            with self.assertRaises(FileExistsError):
                dfp.run(config, root)

    def test_cli_real_crafter_and_checkpoint_reload(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                OUTPUT_DIR=directory,
                DEVICE="cpu",
                TOTAL_STEPS="7",
                FRAME_STACK="2",
                HORIZONS="1,3",
                HEIGHT="32",
                WIDTH="32",
                BATCH_SIZE="2",
                REPLAY_CAPACITY="16",
                LEARNING_STARTS="1",
                TRAIN_EVERY="1",
                LOG_EVERY="3",
                CHECKPOINT_EVERY="100",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            result = subprocess.run(
                [sys.executable, str(project / "src/sebi/direct_future_prediction.py")],
                cwd=project,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(directory)
            records = [
                json.loads(line)
                for line in (root / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[-1]["environment_steps"], 7)
            self.assertEqual(records[-1]["updates"], 4)
            metadata = json.loads((root / "config.json").read_text())
            self.assertEqual(
                metadata["decision_input"], "RGB frames only; no numeric measurements"
            )
            checkpoint = torch.load(
                root / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            model = dfp.FuturePredictor(
                dfp.DFPConfig(**checkpoint["config"]), checkpoint["num_actions"]
            )
            model.load_state_dict(checkpoint["model"])
            predictions = model(torch.zeros(1, 2, 3, 32, 32))
            self.assertTrue(torch.isfinite(predictions).all().item())
            self.assertEqual(dfp.action_scores(predictions).shape, (1, 17))


if __name__ == "__main__":
    unittest.main()
