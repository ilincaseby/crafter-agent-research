import argparse
from dataclasses import asdict, replace
import gzip
from itertools import count
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import crafter
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from src.ppo.evaluate import evaluate
from src.ppo.frame_stacking_experiment import (
    build_parser as experiment_parser,
    run_frame_stacking_experiment,
)
from src.ppo.masking_experiment import validate_baseline
from src.ppo.masking_report import (
    OPTIMIZATION_METRICS,
    VARIANTS,
    generate_masking_report,
)
from src.ppo.report import read_csv
from src.ppo.train import (
    Agent,
    Config,
    RGBFrameStack,
    build_parser,
    compute_gae,
    make_env,
    resource_action_mask,
    train,
)


class CountingRGBEnv(gym.Env):
    def __init__(self, initial=10, limit=6, terminate=False):
        self.observation_space = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)
        self.action_space = gym.spaces.Discrete(17)
        self.initial, self.limit, self.terminate = initial, limit, terminate
        self.episode = 0
        self.steps = 0

    def info(self):
        inventory = {
            name: item["initial"] for name, item in crafter.constants.items.items()
        }
        inventory["wood"] = 2 if self.steps else 0
        return {"action_mask": resource_action_mask(inventory)}

    def frame(self):
        return np.full(
            (3, 64, 64), self.initial + self.episode * 20 + self.steps, np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return self.frame(), self.info()

    def step(self, action):
        self.steps += 1
        done = self.steps == self.limit
        info = self.info()
        if done:
            info["achievements"] = {name: 0 for name in crafter.constants.achievements}
        return (
            self.frame(),
            0.0,
            done and self.terminate,
            done and not self.terminate,
            info,
        )


def frame_values(observation):
    return observation[::3, 0, 0].tolist()


class FrameStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_argparse_defaults_and_overrides(self):
        parser = build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(vars(parser.parse_args([])), asdict(Config()))
        help_text = parser.format_help()
        for expected in (
            "default: 1000000",
            "default: 256",
            "default: 0.00025",
            "--frame-stack",
        ):
            self.assertIn(expected, help_text)
        options = parser.parse_args(
            ["--frame-stack", "4", "--resource-mask", "--learning-rate", "0.0001"]
        )
        self.assertEqual(options.frame_stack, 4)
        self.assertTrue(options.resource_mask)
        self.assertEqual(options.learning_rate, 0.0001)
        experiment = experiment_parser().parse_args([])
        self.assertEqual(experiment.frame_stack, 4)
        self.assertTrue(experiment.resource_mask)
        self.assertEqual(experiment.seeds, [1, 2, 3])
        self.assertEqual(experiment.total_timesteps, 1_000_000)
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Config(frame_stack=invalid).validate()

    def test_order_padding_copying_and_one_action_per_step(self):
        game = CountingRGBEnv()
        env = RGBFrameStack(game, 4)
        observation, info = env.reset(seed=1)
        self.assertEqual(observation.shape, (12, 64, 64))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(frame_values(observation), [30, 30, 30, 30])
        initial = observation.copy()
        for step, expected in enumerate(
            (
                [30, 30, 30, 31],
                [30, 30, 31, 32],
                [30, 31, 32, 33],
                [31, 32, 33, 34],
            ),
            start=1,
        ):
            observation, _, _, _, info = env.step(0)
            self.assertEqual(game.steps, step)
            self.assertEqual(frame_values(observation), expected)
            self.assertTrue(info["action_mask"][11])
        self.assertEqual(frame_values(initial), [30, 30, 30, 30])
        reset, info = env.reset()
        self.assertEqual(frame_values(reset), [50, 50, 50, 50])
        self.assertFalse(info["action_mask"][11])

    def test_vector_workers_keep_independent_terminal_and_reset_stacks(self):
        for terminate in (False, True):
            with self.subTest(terminate=terminate):
                envs = gym.vector.SyncVectorEnv(
                    [
                        lambda: RGBFrameStack(
                            CountingRGBEnv(limit=2, terminate=terminate), 4
                        ),
                        lambda: RGBFrameStack(CountingRGBEnv(initial=100, limit=3), 4),
                    ]
                )
                try:
                    envs.reset(seed=[1, 2])
                    envs.step(np.array([0, 0]))
                    images, _, terminated, truncated, info = envs.step(np.array([0, 0]))
                    self.assertEqual(frame_values(images[0]), [50, 50, 50, 50])
                    self.assertEqual(frame_values(images[1]), [120, 120, 121, 122])
                    self.assertEqual(
                        frame_values(info["final_observation"][0]), [30, 30, 31, 32]
                    )
                    self.assertEqual(bool(terminated[0]), terminate)
                    self.assertEqual(bool(truncated[0]), not terminate)
                    self.assertFalse(info["action_mask"][0][11])
                    self.assertTrue(info["final_info"][0]["action_mask"][11])
                finally:
                    envs.close()

    def test_rollout_boundaries_preserve_history_and_timeout_bootstraps_final_stack(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            for terminate in (False, True):
                collected, bootstraps, gae_rewards = [], [], []
                original_action = Agent.get_action_and_value

                def action_call(agent, observations, actions=None, action_masks=None):
                    if actions is None:
                        collected.extend(frame_values(image) for image in observations)
                    return original_action(agent, observations, actions, action_masks)

                def value_call(agent, observations):
                    bootstraps.extend(frame_values(image) for image in observations)
                    return torch.full(
                        (len(observations),), 5.0, device=observations.device
                    )

                def gae_call(rewards, *args):
                    gae_rewards.append(rewards.clone())
                    return compute_gae(rewards, *args)

                config = Config(
                    total_timesteps=6,
                    num_envs=1,
                    num_steps=2,
                    minibatch_size=2,
                    update_epochs=1,
                    max_episode_steps=6,
                    resource_mask=True,
                    frame_stack=4,
                    sync_envs=True,
                    device="cpu",
                    output_dir=str(Path(temporary) / str(terminate)),
                )
                factory = lambda *args: gym.wrappers.RecordEpisodeStatistics(
                    RGBFrameStack(CountingRGBEnv(terminate=terminate), 4)
                )
                with patch("src.ppo.train.make_env", factory), patch.object(
                    Agent, "get_action_and_value", action_call
                ), patch.object(Agent, "get_value", value_call), patch(
                    "src.ppo.train.compute_gae", gae_call
                ):
                    with patch("src.ppo.train.time.monotonic", side_effect=count()):
                        train(config)
                self.assertEqual(collected[2], [30, 30, 31, 32])
                self.assertEqual(collected[4], [31, 32, 33, 34])
                self.assertEqual(bootstraps[-1], [50, 50, 50, 50])
                if terminate:
                    self.assertNotIn([33, 34, 35, 36], bootstraps)
                    self.assertEqual(gae_rewards[-1][-1].item(), 0)
                else:
                    self.assertIn([33, 34, 35, 36], bootstraps)
                    self.assertAlmostEqual(
                        gae_rewards[-1][-1].item(), config.gamma * 5, places=5
                    )

    def test_legacy_and_stacked_checkpoint_evaluation_and_rgb_recordings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(1)
            legacy = root / "legacy.pt"
            old_config = asdict(Config(max_episode_steps=4))
            old_config.pop("frame_stack")
            old_config.pop("resource_mask")
            torch.save(
                {
                    "config": old_config,
                    "num_actions": 17,
                    "model": Agent(17).state_dict(),
                },
                legacy,
            )
            evaluate(
                root / "legacy_eval",
                checkpoint=legacy,
                episodes=1,
                max_episode_steps=4,
                device="cpu",
            )
            metadata = json.loads((root / "legacy_eval" / "config.json").read_text())
            self.assertEqual(metadata["frame_stack"], 1)
            self.assertFalse(metadata["resource_mask"])
            config = Config(
                total_timesteps=16,
                num_envs=2,
                num_steps=3,
                minibatch_size=5,
                update_epochs=1,
                max_episode_steps=4,
                resource_mask=True,
                frame_stack=4,
                sync_envs=False,
                device="cpu",
                output_dir=str(root / "train"),
            )
            checkpoint = train(config)
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["model"]["encoder.0.weight"].shape[1], 12)
            self.assertEqual(state["global_step"], 16)
            self.assertEqual(state["config"]["frame_stack"], 4)
            output = root / "evaluation"
            original_action = Agent.get_action_and_value
            observed = []

            def checked(agent, images, actions=None, action_masks=None):
                self.assertEqual(images.shape[1:], (12, 64, 64))
                result = original_action(agent, images, actions, action_masks)
                self.assertTrue(action_masks.gather(1, result[0][:, None]).all())
                observed.append(images[0].cpu().numpy()[-3:].copy())
                return result

            with patch.object(Agent, "get_action_and_value", checked):
                evaluate(
                    output,
                    checkpoint=checkpoint,
                    episodes=1,
                    max_episode_steps=4,
                    device="cpu",
                    behavior=True,
                    record_episodes=1,
                )
            gif = output / "behavior" / "episode_0001.gif"
            np.testing.assert_array_equal(
                imageio.imread(gif), observed[0].transpose(1, 2, 0)
            )
            with gzip.open(
                output / "behavior" / "episode_0001.jsonl.gz", "rt"
            ) as stream:
                self.assertEqual(len([*stream]), 5)
            self.assertEqual(
                json.loads((output / "config.json").read_text())["frame_stack"], 4
            )


class ThreeWayExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_seeds_three_variants_on_every_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, output = root / "baseline", root / "comparison"
            config = Config(
                total_timesteps=8,
                num_envs=1,
                num_steps=4,
                minibatch_size=4,
                update_epochs=1,
                max_episode_steps=4,
                sync_envs=True,
                device="cpu",
            )
            for variant in ("vanilla", "masked"):
                for seed in (1, 2, 3):
                    train(
                        replace(
                            config,
                            seed=seed,
                            resource_mask=variant == "masked",
                            output_dir=str(baseline / variant / f"seed_{seed}" / "ppo"),
                        )
                    )
            options = asdict(config)
            options.pop("seed")
            options.update(
                frame_stack=4,
                resource_mask=True,
                seeds=[1, 2, 3],
                baseline_dir=str(baseline),
                output_dir=str(output),
                eval_episodes=2,
                record_episodes=0,
                eval_seed_start=1_500_000_000,
            )
            recorded_plots = []
            from src.ppo.masking_report import figure_image

            def inspect_figure(figure, title, **kwargs):
                recorded_plots.append(
                    (
                        title,
                        [
                            {
                                "labels": [line.get_label() for line in axis.lines],
                                "bands": len(axis.collections),
                                "bar_groups": sum(
                                    container.get_label() in VARIANTS.values()
                                    for container in axis.containers
                                ),
                                "errorbar_groups": sum(
                                    type(container).__name__ == "ErrorbarContainer"
                                    for container in axis.containers
                                ),
                                "xticks": [
                                    tick.get_text() for tick in axis.get_xticklabels()
                                ],
                            }
                            for axis in figure.axes
                        ],
                    )
                )
                return figure_image(figure, title, **kwargs)

            with patch(
                "src.ppo.frame_stacking_experiment.generate_masking_report"
            ) as progress_report:
                # Exercise the real complete report once; avoid redrawing it at every tiny phase.
                run_frame_stacking_experiment(argparse.Namespace(**options))
                self.assertTrue(progress_report.called)
            with patch(
                "src.ppo.masking_report.figure_image", side_effect=inspect_figure
            ):
                result = generate_masking_report(output)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["variants"], ["vanilla", "masked", "stacked"])
            for variant in result["variants"]:
                self.assertEqual(result["aggregate"][variant]["seeds"], 3)
                for curves in result["curves"].values():
                    self.assertEqual(curves[variant]["n_seeds"], 3)
                    self.assertIsNotNone(curves[variant]["std"])
                for metric in OPTIMIZATION_METRICS:
                    self.assertEqual(
                        result["optimization"][variant][metric]["n_seeds"], 3
                    )
                for seed in (1, 2, 3):
                    directory = output / variant / f"seed_{seed}"
                    metadata = json.loads(
                        (directory / "evaluation" / "config.json").read_text()
                    )
                    self.assertEqual(
                        metadata["frame_stack"], 4 if variant == "stacked" else 1
                    )
                    self.assertEqual(metadata["resource_mask"], variant != "vanilla")
                    self.assertEqual(
                        read_csv(directory / "ppo" / "updates.csv")[-1]["global_step"],
                        "8",
                    )
                    rows = read_csv(directory / "evaluation" / "episodes.csv")
                    self.assertEqual(
                        [int(row["environment_seed"]) for row in rows],
                        [1_500_000_000 + (seed - 1) * 2 + i for i in range(2)],
                    )
                    if variant != "stacked":
                        for name in (
                            "checkpoint.pt",
                            "config.json",
                            "episodes.csv",
                            "updates.csv",
                        ):
                            self.assertEqual(
                                (directory / "ppo" / name).read_bytes(),
                                (
                                    baseline / variant / f"seed_{seed}" / "ppo" / name
                                ).read_bytes(),
                            )
            for entry in result["paired_comparisons"].values():
                self.assertEqual(entry["seeds"], 3)
            self.assertEqual(len(result["paired_comparisons"]), 3)
            self.assertEqual(len(recorded_plots), 7)
            for _, axes in recorded_plots[:5]:
                for axis in axes:
                    for name in (VARIANTS[variant] for variant in result["variants"]):
                        self.assertTrue(
                            any(
                                label.startswith(name + " (")
                                for label in axis["labels"]
                            )
                        )
                    self.assertEqual(axis["bands"], 3)
            for axis in recorded_plots[5][1]:
                self.assertEqual(
                    axis["xticks"],
                    [VARIANTS[variant] for variant in result["variants"]],
                )
                self.assertEqual(axis["errorbar_groups"], 3)
            self.assertEqual(recorded_plots[6][1][0]["bar_groups"], 3)
            for name in ("training_curve.csv", "evaluation_curve.csv"):
                self.assertEqual(
                    {row["variant"] for row in read_csv(output / name)},
                    set(result["variants"]),
                )
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(set(metadata["baseline_sha256"]), {"vanilla", "masked"})
            with self.assertRaises(ValueError):
                run_frame_stacking_experiment(
                    argparse.Namespace(**{**options, "frame_stack": 1})
                )
            with self.assertRaises(ValueError):
                validate_baseline(
                    baseline / "masked" / "seed_1" / "ppo",
                    replace(config, gamma=0.9),
                    resource_mask=True,
                )
            # Never present evaluations on different map lists as a paired comparison.
            path = output / "stacked" / "seed_1" / "evaluation" / "episodes.csv"
            path.write_text(path.read_text().replace("1500000000", "1500000099"))
            with self.assertRaisesRegex(ValueError, "Unmatched evaluation maps"):
                generate_masking_report(output)

    def test_partial_stacked_results_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "variants": ["vanilla", "masked", "stacked"],
                        "seeds": [1, 2, 3],
                        "eval_episodes": 100,
                        "record_episodes": 0,
                    }
                )
            )
            result = generate_masking_report(root)
            self.assertEqual(set(result["aggregate"]), {"vanilla", "masked", "stacked"})
            self.assertTrue(
                all(value is None for value in result["aggregate"].values())
            )
            self.assertIsNone(result["curves"]["training"]["stacked"])
            self.assertIn(
                "Masked-PPO + Frame Stacking", (root / "report.html").read_text()
            )


if __name__ == "__main__":
    unittest.main()
