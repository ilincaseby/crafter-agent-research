import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.ppo.evaluate import evaluate
from src.ppo.impala import ImpalaEncoder, ResidualBlock
from src.ppo.masking_experiment import validate_baseline
from src.ppo.masking_report import (
    OPTIMIZATION_METRICS,
    VARIANTS,
    figure_image,
    generate_masking_report,
)
from src.ppo.report import read_csv
from src.ppo.resnet_experiment import build_parser, run_resnet_experiment
from src.ppo.train import Agent, Config, ppo_loss, train


class ImpalaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_architecture_and_initialization(self):
        model = ImpalaEncoder()
        images = torch.zeros(2, 3, 64, 64)
        shapes = []
        handles = [
            stage.register_forward_hook(
                lambda module, inputs, output: shapes.append(tuple(output.shape))
            )
            for stage in model.stages
        ]
        try:
            self.assertEqual(model(images).shape, (2, 1024))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes, [(2, 64, 32, 32), (2, 128, 16, 16), (2, 128, 8, 8)])
        self.assertEqual(
            sum(isinstance(layer, ResidualBlock) for layer in model.modules()), 6
        )
        self.assertFalse(
            any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
        )
        norms = [layer for layer in model.modules() if isinstance(layer, nn.GroupNorm)]
        self.assertTrue(norms)
        self.assertTrue(all(norm.num_groups == 1 for norm in norms))
        self.assertEqual(
            sum(isinstance(layer, nn.LayerNorm) for layer in model.modules()), 2
        )
        for stage in model.stages:
            torch.testing.assert_close(
                stage[0][-2].weight.flatten(1).norm(dim=1),
                torch.ones(stage[0][-2].out_channels),
            )
            for block in (stage[3], stage[4]):
                for unit in block.branch:
                    torch.testing.assert_close(
                        unit[1].weight.flatten(1).norm(dim=1),
                        torch.full((unit[1].out_channels,), 6**-0.25),
                    )
        self.assertEqual(ImpalaEncoder(12)(torch.zeros(1, 12, 64, 64)).shape, (1, 1024))

    def test_residual_skip_is_identity_when_branch_is_zero(self):
        block = ResidualBlock(8)
        for layer in block.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.zeros_(layer.weight)
        inputs = torch.randn(2, 8, 8, 8)
        torch.testing.assert_close(block(inputs), inputs)

    def test_masked_ppo_loss_backward_and_unchanged_cli_defaults(self):
        from src.ppo.train import build_parser as training_parser

        self.assertEqual(vars(training_parser().parse_args([])), asdict(Config()))
        self.assertEqual(
            training_parser().parse_args(["--encoder", "impala"]).encoder, "impala"
        )
        defaults = build_parser().parse_args([])
        self.assertEqual(
            (defaults.encoder, defaults.frame_stack, defaults.resource_mask),
            ("impala", 1, True),
        )
        for key in (
            "gamma",
            "gae_lambda",
            "num_steps",
            "num_envs",
            "minibatch_size",
            "update_epochs",
            "learning_rate",
            "clip_coef",
            "vf_coef",
            "ent_coef",
        ):
            self.assertEqual(getattr(defaults, key), getattr(Config(), key))
        with self.assertRaises(ValueError):
            Config(encoder="resnet18").validate()
        with self.assertRaises(ValueError):
            Agent(17, encoder="invalid")
        agent = Agent(17, encoder="impala")
        self.assertEqual(agent.actor.in_features, 1024)
        torch.testing.assert_close(
            agent.actor.weight.norm(dim=1), torch.full((17,), 0.01)
        )
        torch.testing.assert_close(agent.critic.weight.norm(), torch.tensor(1.0))
        observations = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
        masks = torch.ones(2, 17, dtype=torch.bool)
        masks[:, 7:] = False
        with torch.no_grad():
            actions, logs, _, values = agent.get_action_and_value(
                observations, action_masks=masks
            )
        loss, metrics = ppo_loss(
            agent,
            observations,
            actions,
            logs,
            values,
            torch.tensor([-1.0, 1.0]),
            values + 1,
            Config(encoder="impala", resource_mask=True),
            action_masks=masks,
        )
        self.assertAlmostEqual(metrics["approx_kl"], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in agent.parameters()
                if p.grad is not None
            )
        )
        torch.testing.assert_close(
            agent.actor.weight.grad[7:], torch.zeros_like(agent.actor.weight.grad[7:])
        )
        self.assertGreater(sum(p.numel() for p in agent.parameters()), 3_000_000)


class ResnetExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_resnet_training_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config(
                encoder="impala",
                resource_mask=True,
                total_timesteps=16,
                num_envs=2,
                num_steps=3,
                minibatch_size=5,
                update_epochs=1,
                max_episode_steps=4,
                sync_envs=False,
                device="cpu",
                output_dir=str(root / "ppo"),
            )
            checkpoint = train(config)
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(state["config"]["encoder"], "impala")
            self.assertEqual(state["global_step"], 16)
            self.assertEqual(
                [
                    int(row["global_step"])
                    for row in read_csv(root / "ppo" / "updates.csv")
                ],
                [6, 12, 16],
            )
            metadata = json.loads((root / "ppo" / "model.json").read_text())
            self.assertEqual(
                metadata["parameters"],
                sum(p.numel() for p in Agent(17, encoder="impala").parameters()),
            )
            evidence = validate_baseline(
                root / "ppo", config, resource_mask=True, encoder="impala"
            )
            self.assertEqual(len(evidence), 4)
            with self.assertRaises(ValueError):
                validate_baseline(root / "ppo", config, resource_mask=True)
            evaluate(
                root / "evaluation",
                checkpoint=checkpoint,
                episodes=2,
                max_episode_steps=4,
                device="cpu",
                behavior=True,
                record_episodes=1,
            )
            evaluation = json.loads((root / "evaluation" / "config.json").read_text())
            self.assertEqual(
                (evaluation["encoder"], evaluation["frame_stack"]), ("impala", 1)
            )
            self.assertEqual(len(read_csv(root / "evaluation" / "episodes.csv")), 2)
            self.assertTrue(
                (root / "evaluation" / "behavior" / "episode_0001.gif").is_file()
            )

    def test_resnet_report_omits_stacking_but_preserves_recorded_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline, output = root / "baselines", root / "comparison"
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
            for variant in ("vanilla", "masked", "stacked"):
                for seed in (1, 2, 3):
                    train(
                        replace(
                            config,
                            seed=seed,
                            resource_mask=variant != "vanilla",
                            frame_stack=4 if variant == "stacked" else 1,
                            output_dir=str(baseline / variant / f"seed_{seed}" / "ppo"),
                        )
                    )
            options = asdict(config)
            options.pop("seed")
            options.update(
                encoder="impala",
                resource_mask=True,
                seeds=[1, 2, 3],
                baseline_dir=str(baseline),
                output_dir=str(output),
                eval_episodes=2,
                record_episodes=0,
                eval_seed_start=1_700_000_000,
            )
            with patch("src.ppo.frame_stacking_experiment.generate_masking_report"):
                run_resnet_experiment(argparse.Namespace(**options))
            plots = []

            def inspect_figure(figure, title, **kwargs):
                plots.append(
                    [
                        {
                            "labels": [line.get_label() for line in axis.lines],
                            "bands": len(axis.collections),
                            "containers": [
                                type(container).__name__
                                for container in axis.containers
                            ],
                            "xticks": [
                                tick.get_text() for tick in axis.get_xticklabels()
                            ],
                        }
                        for axis in figure.axes
                    ]
                )
                return figure_image(figure, title, **kwargs)

            with patch(
                "src.ppo.masking_report.figure_image", side_effect=inspect_figure
            ):
                result = generate_masking_report(output)
            self.assertEqual(result["status"], "completed")
            displayed = ["vanilla", "masked", "resnet"]
            labels = [VARIANTS[variant] for variant in displayed]
            self.assertEqual(result["variants"], displayed)
            original_variants = ["vanilla", "masked", "stacked", "resnet"]
            self.assertEqual(result["recorded_variants"], original_variants)
            self.assertEqual(len(result["paired_comparisons"]), 3)
            for section in (
                "aggregate",
                "per_seed",
                "optimization",
                "behavior",
                "models",
            ):
                self.assertEqual(set(result[section]), set(displayed))
            for variant in displayed:
                self.assertEqual(result["aggregate"][variant]["seeds"], 3)
                self.assertEqual(len(result["models"][variant]), 3)
                for curves in result["curves"].values():
                    self.assertEqual(curves[variant]["n_seeds"], 3)
                    self.assertIsNotNone(curves[variant]["std"])
                for metric in OPTIMIZATION_METRICS:
                    self.assertEqual(
                        result["optimization"][variant][metric]["n_seeds"], 3
                    )
                for seed in (1, 2, 3):
                    directory = output / variant / f"seed_{seed}"
                    rows = read_csv(directory / "evaluation" / "episodes.csv")
                    self.assertEqual(
                        [int(row["environment_seed"]) for row in rows],
                        [1_700_000_000 + (seed - 1) * 2 + i for i in range(2)],
                    )
                    if variant != "resnet":
                        for name in (
                            "config.json",
                            "checkpoint.pt",
                            "updates.csv",
                            "episodes.csv",
                        ):
                            self.assertEqual(
                                (directory / "ppo" / name).read_bytes(),
                                (
                                    baseline / variant / f"seed_{seed}" / "ppo" / name
                                ).read_bytes(),
                            )
                    else:
                        saved = Config(
                            **json.loads(
                                (directory / "ppo" / "config.json").read_text()
                            )
                        )
                        for key, value in asdict(config).items():
                            if key not in (
                                "seed",
                                "encoder",
                                "resource_mask",
                                "output_dir",
                            ):
                                self.assertEqual(getattr(saved, key), value)
                        self.assertEqual(saved.frame_stack, 1)
                        self.assertEqual(saved.encoder, "impala")
            self.assertEqual(len(plots), 7)
            for axes in plots[:5]:
                for axis in axes:
                    for label in labels:
                        self.assertTrue(
                            any(
                                line.startswith(label + " (") for line in axis["labels"]
                            )
                        )
                    self.assertFalse(
                        any("Frame Stacking" in label for label in axis["labels"])
                    )
                    self.assertEqual(axis["bands"], 3)
            for axis in plots[5]:
                self.assertEqual(axis["containers"].count("ErrorbarContainer"), 3)
                self.assertEqual(axis["xticks"], labels)
            self.assertEqual(plots[6][0]["containers"].count("BarContainer"), 3)
            for entry in result["paired_comparisons"].values():
                self.assertEqual(entry["seeds"], 3)
            for name in ("training_curve.csv", "evaluation_curve.csv"):
                self.assertEqual(
                    {row["variant"] for row in read_csv(output / name)}, set(displayed)
                )
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["variants"], original_variants)
            for seed in (1, 2, 3):
                directory = output / "stacked" / f"seed_{seed}"
                self.assertTrue((directory / "ppo" / "checkpoint.pt").is_file())
                self.assertEqual(
                    len(read_csv(directory / "evaluation" / "episodes.csv")), 2
                )
            report = (
                (output / "report.html")
                .read_text()
                .split("<h2>Configuration and baseline hashes</h2>")[0]
            )
            self.assertNotIn("Frame Stacking", report)
            self.assertNotIn("stacked", report)
            with self.assertRaises(ValueError):
                run_resnet_experiment(
                    argparse.Namespace(**{**options, "frame_stack": 4})
                )
            with self.assertRaises(ValueError):
                run_resnet_experiment(
                    argparse.Namespace(**{**options, "encoder": "cnn"})
                )


if __name__ == "__main__":
    unittest.main()
