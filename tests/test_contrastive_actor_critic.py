import importlib
import json
import logging
import os
from copy import deepcopy
from functools import reduce
from time import monotonic
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gym
import numpy as np
import torch

training = importlib.import_module("src.sebi.contrastive_actor_critic")
models = importlib.import_module("src.sebi.models")
experiment = importlib.import_module("src.sebi.experiment")


def notebook_namespace():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "src/sebi/main.ipynb").read_text()
    )
    namespace = {
        "torch": torch,
        "F": torch.nn.functional,
        "logging": logging,
        "random": training.random,
        "reduce": reduce,
        "json": json,
        "Path": Path,
        "os": os,
        "monotonic": monotonic,
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith(
            (
                "def contrastive_loss(",
                "def near_loss(",
                "def train_model(",
                "def plot_metrics(",
            )
        ):
            exec(compile(source, "main.ipynb", "exec"), namespace)
    return namespace


class ShortEnvironment:
    def __init__(self, seed, size):
        self.size = size
        self.observation_space = gym.spaces.Box(
            0, 255, (*size[::-1], 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(17)
        self.closed = False

    def reset(self):
        self.steps = 0
        return np.full((*self.size[::-1], 3), 255, dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        achievements = {key: 0 for key in training.achievement_list}
        achievements["collect_wood"] = 1
        achievements["collect_stone"] = 1
        return (
            self.reset_observation(),
            1.0,
            self.steps == 2,
            {"achievements": achievements},
        )

    def reset_observation(self):
        return np.full((*self.size[::-1], 3), 128, dtype=np.uint8)

    def close(self):
        self.closed = True


class ScriptedEnvironment:
    """Legacy Crafter-style env whose frames encode the episode step."""

    def __init__(self, length):
        self.length = length
        self.observation_space = gym.spaces.Box(0, 255, (1, 1, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(17)

    def reset(self):
        self.steps = 0
        return self.observation()

    def observation(self):
        return np.full((1, 1, 3), self.steps + 1, dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        achievements = {key: 0 for key in training.achievement_list}
        achievements["collect_wood"] = int(self.steps >= 2)
        done = self.steps == self.length
        info = {"achievements": achievements, "discount": 0.0 if done else 1.0}
        return self.observation(), 1.0, done, info


class LastFrameModel(torch.nn.Module):
    def forward(self, observations):
        count = observations.shape[0]
        latent = observations[:, -1, 0, 0, 0, None].expand(count, 4) * 255
        return torch.zeros(count, 17), torch.zeros(count, 1), latent


class ContrastiveActorCriticTests(unittest.TestCase):
    def test_vector_collector_builds_windows_and_carries_unfinished_episodes(self):
        env = gym.vector.SyncVectorEnv(
            [
                lambda: training.CrafterGymEnv(ScriptedEnvironment(2)),
                lambda: training.CrafterGymEnv(ScriptedEnvironment(5)),
            ]
        )
        with patch.multiple(training, seq_length=3, height=1, width=1):
            collector = training.ExperienceCollector(
                env, training.get_achievement_dict(), torch.device("cpu")
            )
            episodes, steps = collector.collect(LastFrameModel(), 3, 100)
            self.assertEqual((steps, collector.total_steps), (10, 10))
            self.assertEqual([len(episode[2]) for episode in episodes], [2, 2, 5])
            obs, f_obs, actions, rewards, ach, latent, f_latent, done = episodes[2]
            self.assertEqual(obs.dtype, torch.uint8)
            self.assertEqual(tuple(obs.shape), (5, 3, 3, 1, 1))
            self.assertEqual(obs[:, :, 0, 0, 0].tolist()[:2], [[0, 0, 1], [0, 1, 2]])
            self.assertEqual(f_obs[:, -1, 0, 0, 0].tolist(), [2, 3, 4, 5, 6])
            self.assertTrue(torch.equal(obs[1:], f_obs[:-1]))
            self.assertEqual(latent[:, 0].tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(f_latent[:, 0].tolist(), [2, 3, 4, 5, 6])
            self.assertEqual(done.tolist(), [0, 0, 0, 0, 1])
            self.assertEqual(rewards.tolist(), [1] * 5)
            self.assertEqual(actions.dtype, torch.long)
            wood = training.get_achievement_dict()["collect_wood"]
            self.assertEqual(ach[:, wood].tolist(), [0, 1, 0, 0, 0])
            self.assertEqual(ach.sum().item(), 1)
            # env 0 started a third episode in the previous call; it resumes.
            episodes, steps = collector.collect(LastFrameModel(), 1, 100)
            self.assertEqual((steps, collector.total_steps), (2, 12))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(
                episodes[0][0][:, :, 0, 0, 0].tolist(), [[0, 0, 1], [0, 1, 2]]
            )
            budget, steps = collector.collect(LastFrameModel(), 5, 1)
            self.assertEqual((budget, steps), ([], 2))
        env.close()

    def test_crafter_adapter_splits_death_and_time_limit(self):
        class Legacy(ScriptedEnvironment):
            def step(self, action):
                observation, reward, _, info = super().step(action)
                info["discount"] = discount
                return observation, reward, True, info

        for discount, expected in ((0.0, (True, False)), (1.0, (False, True))):
            env = training.CrafterGymEnv(Legacy(3))
            observation, info = env.reset(seed=1)
            self.assertEqual((observation.shape, info), ((1, 1, 3), {}))
            self.assertEqual(tuple(env.step(np.int64(3))[2:4]), expected)

    def test_standby_instance_prepares_the_next_world_in_background(self):
        class Tagged(ScriptedEnvironment):
            def __init__(self, tag):
                super().__init__(2)
                self.tag, self.resets = tag, 0

            def reset(self):
                self.resets += 1
                return super().reset() + self.tag

        first, second = Tagged(0), Tagged(100)
        env = training.CrafterGymEnv(first, second)
        observations = [env.reset()[0][0, 0, 0]]
        for _ in range(3):
            env.step(0)
            env.step(0)
            observations.append(env.reset()[0][0, 0, 0])
        env.close()
        self.assertEqual(observations, [1, 101, 1, 101])
        self.assertEqual((first.resets, second.resets), (3, 2))

    def test_experiment_evaluates_trains_and_reports_every_seed(self):
        configuration = {
            "height": 16,
            "width": 16,
            "channels_hidden_1": 2,
            "channels_hidden_2": 2,
            "channels_hidden_3": 2,
            "channels_hidden_4": 2,
            "hidden_size_cnn": 4,
            "num_layers_lstm": 1,
            "hidden_size_lstm": 4,
            "hidden_size_input": 4,
            "seq_length": 2,
            "num_exp": 2,
            "num_envs": 2,
            "vector_env": "sync",
            "total_timesteps": 4,
            "warm_up_steps": 0,
            "prob_contrastive": 0.0,
            "game_batch": 2,
        }
        achievements = {key: 0 for key in training.achievement_list}
        reference = {
            "experiment": {"eval_episodes": 3},
            "per_seed": {
                "1": {
                    "ppo": {
                        "episodes": 3,
                        "mean_return": 5.0,
                        "crafter_score": 2.0,
                        "achievement_success_percent": achievements,
                    }
                },
                "2": {"ppo": None},
            },
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(training, **configuration),
            patch.object(training, "get_device", return_value=torch.device("cpu")),
            patch.object(training.crafter, "Env", side_effect=ShortEnvironment),
            patch.object(
                importlib.import_module("src.sebi.evaluate").crafter,
                "Env",
                side_effect=ShortEnvironment,
            ),
        ):
            reference_path = Path(directory) / "ppo_summary.json"
            reference_path.write_text(json.dumps(reference))
            output = Path(directory) / "run"
            experiment.run_experiment(
                output,
                seeds=[1, 2],
                eval_episodes=3,
                eval_envs=2,
                reference=reference_path,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["paired_seeds"], 2)
            self.assertEqual(summary["paired_mean_return_difference"], 0.0)
            for policy in ("cac", "random"):
                aggregate = summary["aggregate"][policy]
                self.assertEqual((aggregate["seeds"], aggregate["mean_return"]), (2, 2))
            self.assertEqual(summary["reference_ppo"]["mean_return"], 5.0)
            seed = summary["per_seed"]["2"]
            self.assertEqual(seed["training_steps"], 4)
            self.assertEqual(seed["training_episodes"], 2)
            self.assertEqual(
                seed["random"]["achievement_success_percent"]["collect_wood"], 100
            )
            rows = (output / "seed_2/eval_cac/episodes.csv").read_text().splitlines()
            self.assertEqual(len(rows), 4)
            self.assertIn("1000000003", rows[1])
            training_log = (output / "seed_1/cac/episodes.csv").read_text().splitlines()
            self.assertEqual(
                training_log[0],
                "global_step,episodic_return,episodic_length,achievements",
            )
            self.assertTrue(training_log[1].startswith("4,2.0,2,"))
            report = (output / "report.html").read_text()
            self.assertIn("PPO (reference run)", report)
            self.assertEqual(report.count("data:image/png;base64"), 3)
            checkpoint = torch.load(
                output / "seed_1/cac/checkpoint.pt", weights_only=True
            )
            self.assertEqual(checkpoint["config"]["seq_length"], 2)

    def test_import_does_not_start_training(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from unittest.mock import patch; "
                "import crafter; "
                "with_patch = patch.object(crafter, 'Env', side_effect=AssertionError); "
                "with_patch.start(); "
                "import src.sebi.contrastive_actor_critic",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actor_advantage_is_detached_and_critic_remains_trainable(self):
        log_probs = torch.tensor([-0.5, -0.2], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0], requires_grad=True)
        training.actor_loss(log_probs, advantages).backward()
        self.assertIsNone(advantages.grad)
        training.critic_loss(advantages).backward()
        self.assertIsNotNone(advantages.grad)

    def test_near_loss_matches_sparse_keys_and_detaches_reference(self):
        current = {3: torch.tensor([[1.0, 0.0]], requires_grad=True)}
        previous = {3: torch.tensor([[0.0, 1.0]], requires_grad=True)}
        loss = training.near_loss(current, previous)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertGreater(current[3].grad.abs().sum().item(), 0)
        self.assertIsNone(previous[3].grad)

    def test_contrastive_requires_pairs_and_backpropagates_when_available(self):
        self.assertIsNone(training.contrastive_loss({0: torch.randn(2, 4)}, 1.1))
        self.assertIsNone(
            training.contrastive_loss({0: torch.randn(1, 4), 3: torch.randn(1, 4)}, 1.1)
        )
        embeddings = {
            0: torch.randn(2, 4, requires_grad=True),
            3: torch.randn(2, 4, requires_grad=True),
        }
        loss = training.contrastive_loss(embeddings, 1.1)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(value.grad is not None for value in embeddings.values()))

    def test_encoder_respects_dilation_and_group_configuration(self):
        encoder = models.LatentStateEncoder(
            32,
            32,
            4,
            num_layers=1,
            hidden_size_lstm=4,
            c1=4,
            c2=4,
            c3=4,
            c4=4,
            hidden_size_cnn=4,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=2,
        )
        output, _ = encoder(torch.randn(2, 3, 4, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 3, 4))
        self.assertEqual(encoder.visual_encoder.net[0].groups, 2)
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 4, 32, 32))

    def test_notebook_contrastive_groups_and_simultaneous_achievements(self):
        namespace = notebook_namespace()

        class RecordingEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 4)
                self.seen = []

            def forward(self, observations):
                self.seen.append(observations[:, 0, 0, 0, 0].tolist())
                return self.projection(observations[:, :, 0, 0, 0, None]), None

        observations = [torch.full((1, 2, 1, 1, 1), float(i)) for i in range(1, 5)]
        for has_achievements, group_size, expected_warnings in (
            (True, 2, 0),
            (True, 1, 4),
            (False, 2, 2),
        ):
            with (
                self.subTest(achievements=has_achievements, group_size=group_size),
                tempfile.TemporaryDirectory() as directory,
            ):
                encoder = RecordingEncoder()
                achievements = [
                    torch.full((1, 2), float(has_achievements)) for _ in range(4)
                ]
                buffers = [
                    observations,
                    observations,
                    [],
                    [torch.ones(1) for _ in range(4)],
                    achievements,
                    [],
                    [],
                    [],
                ]
                namespace["create_experiences"] = lambda *args: buffers
                with patch.object(logging, "warning") as warning:
                    result = namespace["train_model"](
                        a2c_model=encoder,
                        lstm_model=encoder,
                        clone_lstm=deepcopy(encoder),
                        temp_clone=deepcopy(encoder),
                        env=None,
                        steps_to_play=1,
                        games_per_play=4,
                        optim_lr=1e-4,
                        centered=False,
                        alpha=0.99,
                        eps_optim=1e-8,
                        seq_length=2,
                        c=1,
                        h=1,
                        w=1,
                        hidden_size=4,
                        achievement_dict={"wood": 0, "stone": 1},
                        device=torch.device("cpu"),
                        num_actions=17,
                        game_batch_size=2,
                        discount=0.9985,
                        actor_loss_weight=1,
                        critic_loss_weight=0.5,
                        entropy_loss_weight=0.1,
                        prob_contrastive=1,
                        warm_up_steps=0,
                        contrastive_loss_batch_size=group_size,
                        contrastive_loss_temperature=1.1,
                        lr_contrastive_learning=1e-4,
                        weight_decay=0.01,
                        contrastive_sim_weight=1,
                        contrastive_near_loss_weight=0.2,
                        output_dir=directory,
                    )
                self.assertEqual(warning.call_count, expected_warnings)
                if has_achievements:
                    expected = [
                        list(map(float, range(start, min(start + group_size, 5))))
                        for start in range(1, 5, group_size)
                        for _ in range(2)
                    ]
                    self.assertEqual(encoder.seen, expected)
                    self.assertEqual(len(result[6]), 1)
                    self.assertTrue(np.isfinite(result[6][0]))
                else:
                    self.assertEqual(encoder.seen, [])
                    self.assertEqual(result[6], [])
                metrics = json.loads((Path(directory) / "metrics.jsonl").read_text())
                self.assertEqual(metrics["mode"], "contrastive")
                self.assertNotIn("advantages_abs_mean", metrics)
                if has_achievements:
                    training.save_plots(Path(directory))
                    self.assertTrue(
                        (Path(directory) / "plots/contrastive_loss.png").is_file()
                    )
                else:
                    self.assertIsNone(metrics["contrastive_loss"])
        self.assertIsNone(namespace["contrastive_loss"]({0: torch.randn(2, 4)}, 1.1))
        with self.assertRaises(ValueError):
            namespace["contrastive_loss"]({0: torch.randn(2, 4)}, 0)

    def test_notebook_persists_weighted_diagnostics_for_plotting(self):
        namespace = notebook_namespace()
        namespace["actor_loss"] = training.actor_loss
        namespace["critic_loss"] = training.critic_loss

        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, observations):
                count = observations.shape[0]
                return (
                    self.value.expand(count, 17),
                    self.value.expand(count, 1),
                    None,
                    None,
                    None,
                    None,
                )

        model = ConstantModel()
        observations = [torch.zeros(3, 2, 1, 1, 1)]
        namespace["create_experiences"] = lambda *args: [
            observations,
            observations,
            [torch.zeros(3, dtype=torch.long)],
            [torch.tensor([1.0, 1.0, 10.0])],
            [torch.zeros(3, 2)],
            [torch.zeros(3, 4)],
            [torch.zeros(3, 4)],
            [torch.ones(3)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            namespace["train_model"](
                model,
                model,
                deepcopy(model),
                deepcopy(model),
                None,
                steps_to_play=1,
                games_per_play=1,
                optim_lr=0,
                centered=False,
                alpha=0.99,
                eps_optim=1e-8,
                seq_length=2,
                c=1,
                h=1,
                w=1,
                hidden_size=4,
                achievement_dict={"wood": 0, "stone": 1},
                device=torch.device("cpu"),
                num_actions=17,
                game_batch_size=2,
                discount=0,
                actor_loss_weight=1,
                critic_loss_weight=0.5,
                entropy_loss_weight=0.1,
                prob_contrastive=0,
                warm_up_steps=0,
                contrastive_loss_batch_size=1,
                contrastive_loss_temperature=1.1,
                lr_contrastive_learning=1e-4,
                weight_decay=0.01,
                contrastive_sim_weight=1,
                contrastive_near_loss_weight=0.2,
                output_dir=directory,
            )
            metrics = json.loads((Path(directory) / "metrics.jsonl").read_text())
            self.assertEqual(metrics["mode"], "actor_critic")
            self.assertEqual(metrics["advantages_abs_mean"], 4)
            self.assertEqual(metrics["reward_abs_mean"], 4)
            self.assertEqual(metrics["value_abs_mean"], 0)
            training.save_plots(Path(directory))
            self.assertTrue(
                (Path(directory) / "plots/advantages_abs_mean.png").is_file()
            )

    def test_diagnostic_means_weight_partial_batches_by_transition_count(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, observations):
                count = observations.shape[0]
                return self.value.expand(count, 17), self.value.expand(count, 1)

        observations = [torch.zeros(3, 2, 1, 1, 1)]
        buffers = [
            observations,
            observations,
            [torch.zeros(3, dtype=torch.long)],
            [torch.tensor([1.0, 1.0, 10.0])],
            [],
            [],
            [],
            [torch.zeros(3)],
        ]
        model = ConstantModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        with patch.multiple(training, num_exp=1, game_batch=2, discount=0):
            metrics = training.train_actor_critic(model, optimizer, buffers)
        self.assertAlmostEqual(metrics["advantages_abs_mean"], 4.0)
        self.assertAlmostEqual(metrics["reward_abs_mean"], 4.0)
        self.assertEqual(metrics["value_abs_mean"], 0.0)

    def test_plots_preserve_iteration_gaps_between_training_modes(self):
        history = [
            {"iteration": 1, "advantages_abs_mean": 0.2},
            {"iteration": 2, "contrastive_loss": 1.0},
            {"iteration": 3, "advantages_abs_mean": 0.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in history)
            )
            with patch("matplotlib.axes.Axes.plot", autospec=True) as plot:
                training.save_plots(root)
            calls = plot.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1].args[1], [1, 2, 3])
            self.assertTrue(np.isnan(calls[1].args[2][1]))
            self.assertTrue((root / "plots/advantages_abs_mean.png").is_file())
            self.assertTrue((root / "plots/contrastive_loss.png").is_file())
            self.assertFalse((root / "plots/actor_loss.png").exists())
            with (root / "metrics.jsonl").open("a") as metrics_file:
                metrics_file.write(
                    json.dumps({"iteration": 4, "advantages_abs_mean": 0.05}) + "\n"
                )
            with patch("matplotlib.axes.Axes.plot", autospec=True) as plot:
                training.save_plots(root)
            self.assertEqual(plot.call_args_list[1].args[1], [1, 2, 3, 4])
            self.assertEqual(plot.call_args_list[1].args[2][-1], 0.05)

    def test_main_saves_metrics_and_checkpoints_for_both_training_modes(self):
        configuration = {
            "height": 16,
            "width": 16,
            "channels_hidden_1": 2,
            "channels_hidden_2": 2,
            "channels_hidden_3": 2,
            "channels_hidden_4": 2,
            "hidden_size_cnn": 4,
            "num_layers_lstm": 1,
            "hidden_size_lstm": 4,
            "hidden_size_input": 4,
            "seq_length": 2,
            "num_exp": 2,
            "num_envs": 2,
            "vector_env": "sync",
            "total_timesteps": 4,
            "warm_up_steps": 0,
            "contrastive_loss_batch_size": 1,
            "game_batch": 2,
        }
        for probability, mode, group_size in (
            (0.0, "actor_critic", 1),
            (1.0, "contrastive", 1),
            (1.0, "contrastive", 2),
        ):
            with (
                self.subTest(mode=mode, group_size=group_size),
                tempfile.TemporaryDirectory() as directory,
            ):
                environments = []

                def create_environment(seed, size):
                    env = ShortEnvironment(seed, size)
                    environments.append(env)
                    return env

                with (
                    patch.multiple(
                        training, **configuration, prob_contrastive=probability
                    ),
                    patch.object(training, "contrastive_loss_batch_size", group_size),
                    patch.dict(
                        os.environ, {"OUTPUT_DIR": directory, "CHECKPOINT_EVERY": "1"}
                    ),
                    patch.object(
                        training, "get_device", return_value=torch.device("cpu")
                    ),
                    patch.object(
                        training.crafter, "Env", side_effect=create_environment
                    ),
                    patch.object(
                        training.crafter,
                        "Recorder",
                        side_effect=lambda env, *a, **kw: env,
                    ),
                ):
                    training.main()
                records = [
                    json.loads(line)
                    for line in (Path(directory) / "metrics.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(records[0]["mode"], mode)
                self.assertEqual(records[0]["average_steps"], 2)
                self.assertEqual(records[0]["achievements"]["collect_wood"], 2)
                self.assertEqual(records[0]["achievements"]["collect_stone"], 2)
                if mode == "contrastive":
                    self.assertEqual(records[0]["contrastive_updates"], 2 // group_size)
                    self.assertTrue(
                        (Path(directory) / "plots/contrastive_loss.png").is_file()
                    )
                    self.assertNotIn("advantages_abs_mean", records[0])
                else:
                    self.assertGreaterEqual(records[0]["advantages_abs_mean"], 0)
                    self.assertEqual(records[0]["reward_abs_mean"], 1)
                    self.assertTrue(
                        (Path(directory) / "plots/advantages_abs_mean.png").is_file()
                    )
                for key in ("average_reward", "average_steps"):
                    image = Path(directory) / "plots" / f"{key}.png"
                    self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                checkpoint = torch.load(
                    Path(directory) / "checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(checkpoint["iteration"], 1)
                self.assertEqual(checkpoint["timesteps"], 4)
                self.assertEqual(records[0]["timesteps"], 4)
                self.assertEqual(records[0]["episodes"], 2)
                self.assertTrue(checkpoint["model"])
                self.assertTrue(environments[0].closed)


if __name__ == "__main__":
    unittest.main()
