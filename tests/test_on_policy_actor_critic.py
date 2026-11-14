import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.sebi import actor_critic_on_policy as training
from src.sebi import contrastive_actor_critic_on_policy as contrastive
from src.sebi.models import ActorCritic


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)

    def forward(self, observations):
        encoded = self.projection(observations.mean(dim=(-2, -1))).tanh()
        return encoded, (None, None)


def tiny_model():
    return ActorCritic(TinyEncoder(), 1, 4, 17, head_sizes=(8, 4))


class ShortEnvironment:
    def __init__(self, seed=42, size=(32, 32), time_limit=False):
        self.size = size
        self.time_limit = time_limit
        self.action_space = type("ActionSpace", (), {"n": 17})()
        self.episode = -1
        self.closed = False

    def reset(self):
        self.episode += 1
        self.steps = 0
        return self.observation()

    def observation(self):
        image = np.zeros((*self.size[::-1], 3), dtype=np.uint8)
        image[..., self.episode % 3] = 100 + self.steps * 20
        return image

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        achievements = dict.fromkeys(training.achievement_list, 0)
        achievements["collect_wood" if self.episode % 2 == 0 else "collect_sapling"] = 1
        return (
            self.observation(),
            1.0,
            done,
            {
                "discount": float(not done or self.time_limit),
                "achievements": achievements,
            },
        )

    def close(self):
        self.closed = True


class OnPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def test_gae_bootstraps_time_limits_without_crossing_episode_boundaries(self):
        advantages, returns = training.compute_gae(
            torch.tensor([1.0, 2.0, 100.0]),
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([20.0, 7.0, 40.0]),
            torch.ones(3),
            torch.tensor([0.0, 1.0, 0.0]),
            0.9,
            0.8,
        )
        torch.testing.assert_close(advantages, torch.tensor([0.576, -11.7, 106.0]))
        torch.testing.assert_close(returns, torch.tensor([10.576, 8.3, 136.0]))
        advantages, _ = training.compute_gae(
            torch.tensor([1.0]),
            torch.tensor([4.0]),
            torch.tensor([100.0]),
            torch.zeros(1),
            torch.ones(1),
            0.99,
            0.95,
        )
        self.assertEqual(advantages.item(), -3.0)

    def test_collector_handles_death_timeout_and_continuing_rollouts(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, observations):
                return torch.zeros(len(observations), 17), torch.full(
                    (len(observations), 1), 4.0
                )

        config = training.Config(seq_length=2, gamma=1.0, gae_lambda=1.0)
        for time_limit, expected in ((False, [2.0, 1.0, 5.0]), (True, [6.0, 5.0, 5.0])):
            with self.subTest(time_limit=time_limit):
                env = ShortEnvironment(time_limit=time_limit)
                collector = training.RolloutCollector(env, config)
                rollout = collector.collect(ConstantModel(), 3, self.device)
                torch.testing.assert_close(rollout.returns, torch.tensor(expected))
                self.assertEqual(rollout.observations.dtype, torch.uint8)
                self.assertEqual(rollout.episode_lengths, [2])
                self.assertEqual(rollout.episode_rewards, [2.0])
                self.assertEqual(sum(rollout.achievements.values()), 2)
                self.assertEqual(collector.episode_length, 1)
                self.assertFalse(rollout.observations[0, 0].any().item())
                self.assertFalse(rollout.observations[2, 0].any().item())
                collector.collect(ConstantModel(), 1, self.device)
                self.assertEqual(collector.episode_length, 0)

    def test_microbatches_accumulate_one_exact_full_rollout_update(self):
        model = tiny_model()
        full_batch_model = deepcopy(model)
        observations = torch.randint(0, 256, (5, 2, 3, 2, 2), dtype=torch.uint8)
        advantages = torch.tensor([1.0, -1.0, 0.3, 2.0, -0.4])
        rollout = training.Rollout(
            observations,
            torch.tensor([0, 1, 2, 3, 4]),
            torch.ones(5),
            torch.zeros(5),
            advantages,
            advantages,
            [],
            [],
            {},
        )
        config = training.Config(batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        original = deepcopy(model.state_dict())
        forwards = []

        def verify_unchanged(module, inputs):
            forwards.append(1)
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)

        handle = model.register_forward_pre_hook(verify_unchanged)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            metrics = training.update_actor_critic(
                model, optimizer, rollout, config, 0.01, self.device
            )
        handle.remove()
        self.assertEqual(step.call_count, 1)
        self.assertEqual(len(forwards), 3)
        expected = training.update_actor_critic(
            full_batch_model,
            torch.optim.SGD(full_batch_model.parameters(), lr=0.01),
            rollout,
            replace(config, batch_size=5),
            0.01,
            self.device,
        )
        for actual, target in zip(model.parameters(), full_batch_model.parameters()):
            torch.testing.assert_close(actual, target)
        for key in ("loss", "actor_loss", "critic_loss", "entropy"):
            self.assertAlmostEqual(metrics[key], expected[key], places=6)

    def test_entropy_schedule_and_configuration_validation(self):
        config = training.Config(entropy_decay_steps=100)
        self.assertAlmostEqual(training.entropy_coefficient(config, 0), 0.01)
        self.assertAlmostEqual(training.entropy_coefficient(config, 50), 0.0055)
        self.assertAlmostEqual(training.entropy_coefficient(config, 100), 0.001)
        self.assertAlmostEqual(training.entropy_coefficient(config, 1000), 0.001)
        for changes in (
            {"gamma": 1.1},
            {"gae_lambda": -0.1},
            {"learning_rate": float("nan")},
            {"entropy_end": 0.1},
            {"total_steps": 0},
            {"height": 8},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                training.Config(**changes)
        with patch.dict(os.environ, {"TOTAL_STEPS": "99", "GAMMA": "0.95"}):
            actual = training.config_from_env(training.Config)
        self.assertEqual(actual.total_steps, 99)
        self.assertEqual(actual.gamma, 0.95)

    def test_supervised_contrastive_multilabel_pairs_and_gradients(self):
        embeddings = torch.randn(4, 6, requires_grad=True)
        labels = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 1]])
        loss = contrastive.supervised_contrastive_loss(embeddings, labels, 0.2)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertGreater(embeddings.grad.abs().sum().item(), 0)
        self.assertIsNone(
            contrastive.supervised_contrastive_loss(
                embeddings,
                torch.ones(4, 1),
                0.2,
            )
        )
        self.assertIsNone(
            contrastive.supervised_contrastive_loss(
                embeddings,
                torch.eye(4),
                0.2,
            )
        )
        with self.assertRaises(ValueError):
            contrastive.supervised_contrastive_loss(embeddings, labels, 0.0)

    def make_auxiliary(self, model, **kwargs):
        config = contrastive.ContrastiveConfig(
            contrastive_warmup_steps=0,
            contrastive_every=1,
            **kwargs,
        )
        auxiliary = contrastive.ContrastiveTrainer(model, config, seed=7)
        for index in range(4):
            observation = torch.zeros(2, 3, 2, 2, dtype=torch.uint8)
            observation[:, index % 3] = 100 + index * 30
            auxiliary.observe(observation, torch.tensor([index < 2, index >= 2]))
        return auxiliary

    def test_contrastive_changes_only_encoder_and_uses_fresh_reference(self):
        model = tiny_model()
        auxiliary = self.make_auxiliary(model, contrastive_max_kl=10.0)
        before = deepcopy(model.state_dict())
        observations = torch.stack([state for state, _ in auxiliary.replay])
        before_logs, _ = contrastive.probe(model, observations, self.device, 4)
        metrics = auxiliary.update(model, observations, 10, 1, self.device)
        self.assertEqual(metrics["contrastive_updates"], 1)
        self.assertTrue(
            any(
                not torch.equal(value, before[name])
                for name, value in model.state_dict().items()
                if name.startswith("lse_model.")
            )
        )
        for name, value in model.state_dict().items():
            if not name.startswith("lse_model."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        after_logs, _ = contrastive.probe(model, observations, self.device, 4)
        expected_kl = (
            (before_logs.exp() * (before_logs - after_logs)).sum(-1).mean().item()
        )
        self.assertAlmostEqual(
            metrics["contrastive_proposed_kl"], max(expected_kl, 0), places=6
        )
        with torch.no_grad():
            model.lse_model.projection.bias.add_(0.1)
        reference_logs, _ = contrastive.probe(model, observations, self.device, 4)
        metrics = auxiliary.update(model, observations, 20, 2, self.device)
        current_logs, _ = contrastive.probe(model, observations, self.device, 4)
        expected_kl = (
            (reference_logs.exp() * (reference_logs - current_logs))
            .sum(-1)
            .mean()
            .item()
        )
        self.assertAlmostEqual(
            metrics["contrastive_proposed_kl"], max(expected_kl, 0), places=6
        )

    def test_rejected_contrastive_step_restores_encoder_and_optimizer(self):
        model = tiny_model()
        auxiliary = self.make_auxiliary(model)
        observations = torch.stack([state for state, _ in auxiliary.replay])
        self.assertEqual(
            auxiliary.update(model, observations, 10, 1, self.device)[
                "contrastive_updates"
            ],
            1,
        )
        before = deepcopy(model.state_dict())
        optimizer_before = deepcopy(auxiliary.optimizer.state_dict())
        self.assertTrue(optimizer_before["state"])
        logs = torch.full((8, 17), -np.log(17))
        changed_logs = torch.log_softmax(
            torch.tensor([[20.0] + [0.0] * 16] * 8), dim=-1
        )
        with (
            patch.object(
                contrastive,
                "probe",
                side_effect=[
                    (logs, torch.zeros(8)),
                    (changed_logs, torch.ones(8)),
                ],
            ),
            self.assertLogs(contrastive.logger, level="WARNING"),
        ):
            metrics = auxiliary.update(model, observations, 10, 1, self.device)
        self.assertTrue(metrics["contrastive_rejected"])
        self.assertEqual(metrics["contrastive_policy_kl"], 0)
        self.assertGreater(
            metrics["contrastive_proposed_kl"], auxiliary.config.contrastive_max_kl
        )
        torch.testing.assert_close(auxiliary.optimizer.state_dict(), optimizer_before)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def check_recurrent_contrastive_update(self, device):
        config = training.Config(height=32, width=32, hidden_size=8, seq_length=2)
        model = training.build_model(config, 17).to(device)
        model.eval()
        auxiliary = contrastive.ContrastiveTrainer(
            model,
            contrastive.ContrastiveConfig(
                contrastive_warmup_steps=0,
                contrastive_every=1,
                contrastive_max_kl=10.0,
            ),
            seed=7,
        )
        observations = torch.randint(0, 256, (4, 2, 3, 32, 32), dtype=torch.uint8)
        for index, state in enumerate(observations):
            auxiliary.observe(state, torch.tensor([index < 2, index >= 2]))
        forward_modes, backward_modes = [], []

        def record_forward(module, inputs):
            forward_modes.append((module.training, torch.is_grad_enabled()))

        def record_backward(gradient):
            backward_modes.append(model.lse_model.lstm.training)

        forward_hook = model.lse_model.lstm.register_forward_pre_hook(record_forward)
        backward_hook = model.lse_model.lstm.weight_ih_l0.register_hook(record_backward)
        before = deepcopy(model.state_dict())
        try:
            for iteration in (1, 2):
                metrics = auxiliary.update(
                    model, observations, iteration * 10, iteration, device
                )
                self.assertEqual(metrics["contrastive_updates"], 1)
                self.assertTrue(np.isfinite(metrics["contrastive_loss"]))
                self.assertFalse(model.training)
        finally:
            forward_hook.remove()
            backward_hook.remove()
        self.assertEqual(
            forward_modes,
            [(True, True), (False, False), (False, False)] * 2,
        )
        self.assertEqual(backward_modes, [True, True])
        self.assertFalse(
            torch.equal(
                before["lse_model.lstm.weight_ih_l0"],
                model.lse_model.lstm.weight_ih_l0,
            )
        )
        for name, value in model.state_dict().items():
            if not name.startswith("lse_model."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_recurrent_contrastive_trains_encoder_and_evaluates_probes(self):
        self.check_recurrent_contrastive_update(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_recurrent_contrastive_backward(self):
        with torch.backends.cudnn.flags(enabled=True):
            self.check_recurrent_contrastive_update(torch.device("cuda"))

    def test_replay_is_bounded_and_invalid_pairs_are_explicitly_skipped(self):
        model = tiny_model()
        auxiliary = self.make_auxiliary(model, contrastive_capacity=3)
        self.assertEqual(len(auxiliary.replay), 3)
        self.assertTrue(
            all(state.dtype == torch.uint8 for state, _ in auxiliary.replay)
        )
        state = auxiliary.replay[-1][0]
        for _ in range(3):
            auxiliary.observe(state, torch.tensor([True, False]))
        with self.assertLogs(contrastive.logger, level="WARNING"):
            metrics = auxiliary.update(model, state.unsqueeze(0), 10, 1, self.device)
        self.assertEqual(metrics["contrastive_skip_reason"], "insufficient_pairs")
        self.assertEqual(metrics["contrastive_updates"], 0)

    def test_both_entrypoints_save_exact_step_budget_and_isolated_outputs(self):
        for use_contrastive in (False, True):
            with self.subTest(
                contrastive=use_contrastive
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env = ShortEnvironment()
                settings = {
                    "OUTPUT_DIR": directory,
                    "DEVICE": "cpu",
                    "TOTAL_STEPS": "9",
                    "ROLLOUT_STEPS": "4",
                    "BATCH_SIZE": "3",
                    "SEQ_LENGTH": "2",
                    "HEIGHT": "32",
                    "WIDTH": "32",
                    "HIDDEN_SIZE": "4",
                    "CHECKPOINT_EVERY": "100",
                    "CONTRASTIVE_WARMUP_STEPS": "0",
                    "CONTRASTIVE_EVERY": "1",
                    "CONTRASTIVE_MAX_KL": "10",
                }
                with (
                    patch.dict(os.environ, settings),
                    patch.object(training.crafter, "Env", return_value=env),
                    patch.object(
                        training.crafter,
                        "Recorder",
                        side_effect=lambda env, *a, **k: env,
                    ),
                    patch.object(
                        training, "build_model", side_effect=lambda *a: tiny_model()
                    ),
                ):
                    (contrastive.main if use_contrastive else training.main)()
                records = [
                    json.loads(line)
                    for line in (root / "metrics.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [row["environment_steps"] for row in records], [4, 8, 9]
                )
                self.assertEqual(records[-1]["episodes"], 4)
                self.assertEqual(records[-1]["average_reward"], 2)
                self.assertEqual(records[-1]["average_steps"], 2)
                self.assertEqual(
                    sum(sum(row["achievements"].values()) for row in records), 5
                )
                self.assertTrue(env.closed)
                if use_contrastive:
                    self.assertGreater(
                        sum(row["contrastive_updates"] for row in records), 0
                    )
                else:
                    self.assertTrue(
                        all("contrastive_updates" not in row for row in records)
                    )
                saved_config = json.loads((root / "config.json").read_text())
                self.assertEqual(saved_config["training"]["total_steps"], 9)
                checkpoint = torch.load(
                    root / "checkpoint.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(checkpoint["environment_steps"], 9)
                self.assertEqual("contrastive" in checkpoint, use_contrastive)
                restored = tiny_model()
                restored.load_state_dict(checkpoint["model"])
                self.assertTrue(
                    torch.isfinite(restored(torch.zeros(1, 2, 3, 2, 2))[0]).all().item()
                )
                self.assertEqual(
                    (root / "plots/entropy.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
                )
                with self.assertRaises(FileExistsError):
                    training.run(training.Config(total_steps=1), root)

    def test_command_line_variants_run_real_crafter_and_reload_weights(self):
        project = Path(__file__).resolve().parents[1]
        for script in (
            "actor_critic_on_policy.py",
            "contrastive_actor_critic_on_policy.py",
        ):
            with self.subTest(
                script=script
            ), tempfile.TemporaryDirectory() as directory:
                env = dict(
                    os.environ,
                    OUTPUT_DIR=directory,
                    DEVICE="cpu",
                    TOTAL_STEPS="5",
                    ROLLOUT_STEPS="3",
                    BATCH_SIZE="2",
                    SEQ_LENGTH="2",
                    HEIGHT="32",
                    WIDTH="32",
                    HIDDEN_SIZE="8",
                    CONTRASTIVE_WARMUP_STEPS="0",
                    CONTRASTIVE_EVERY="1",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                result = subprocess.run(
                    [sys.executable, str(project / "src/sebi" / script)],
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
                self.assertEqual([row["environment_steps"] for row in records], [3, 5])
                checkpoint = torch.load(
                    root / "checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model = training.build_model(
                    training.Config(**checkpoint["config"]), 17
                )
                model.load_state_dict(checkpoint["model"])
                logits, value, *_ = model(torch.zeros(1, 2, 3, 32, 32))
                self.assertTrue(torch.isfinite(logits).all().item())
                self.assertTrue(torch.isfinite(value).all().item())
                self.assertEqual(
                    (root / "plots/entropy.png").read_bytes()[:8],
                    b"\x89PNG\r\n\x1a\n",
                )


if __name__ == "__main__":
    unittest.main()
