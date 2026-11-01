from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.ppo.masking_report import figure_image, generate_masking_report, VARIANTS
from src.ppo.report import read_csv
from src.ppo.train import Config, train
from src.ppo.tuned_experiment import (
    build_parser,
    run_tuned_experiment,
    selection_provenance,
)
from src.ppo.tuning import build_parser as sweep_parser, initialize, write_json


class TunedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        sweep = root / "sweep"
        args = sweep_parser().parse_args(
            [
                "init",
                "--output-dir",
                str(sweep),
                "--total-timesteps",
                "8",
                "--num-envs",
                "1",
                "--num-steps",
                "4",
                "--minibatch-size",
                "4",
                "--max-episode-steps",
                "4",
                "--device",
                "cpu",
                "--sync-envs",
            ]
        )
        plan = initialize(args)
        selection = {
            "plan_sha256": plan["sha256"],
            "candidate": "lambda_065",
            "changes": {"gae_lambda": 0.65},
        }
        write_json(sweep / "selection.json", selection)
        ranking = [
            {
                "name": candidate["name"],
                "changes": candidate["changes"],
                "per_seed": [{"seed": seed} for seed in (1, 2, 3)],
            }
            for candidate in sorted(
                plan["candidates"], key=lambda item: item["name"] != "lambda_065"
            )
        ]
        write_json(
            sweep / "results.json",
            {
                "plan_sha256": plan["sha256"],
                "all_complete": True,
                "selection": selection,
                "ranking": ranking,
            },
        )
        return sweep, plan

    def test_defaults_are_full_budget_selected_policy(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.total_timesteps, 1_000_000)
        self.assertEqual(args.gae_lambda, 0.65)
        self.assertEqual(args.seeds, [1, 2, 3])
        self.assertEqual(
            (args.encoder, args.frame_stack, args.resource_mask), ("impala", 1, True)
        )
        self.assertEqual(args.eval_seed_start, 2_000_000_000)
        for field in (
            "gamma",
            "learning_rate",
            "num_steps",
            "num_envs",
            "update_epochs",
            "minibatch_size",
        ):
            self.assertEqual(getattr(args, field), getattr(Config(), field))

    def test_rejects_modified_selection_incomplete_sweep_or_extra_hyperparameter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, plan = self.fixture(root)
            options = build_parser().parse_args(
                ["--tuning-dir", str(sweep), "--gamma", "0.95"]
            )
            with self.assertRaisesRegex(ValueError, "retain selected configuration"):
                run_tuned_experiment(options)
            path = sweep / "results.json"
            results = json.loads(path.read_text())
            results["all_complete"] = False
            write_json(path, results)
            with self.assertRaisesRegex(ValueError, "completed frozen"):
                selection_provenance(sweep)
            results["all_complete"] = True
            write_json(path, results)
            selection = json.loads((sweep / "selection.json").read_text())
            selection["changes"] = {"gae_lambda": 0.95}
            write_json(sweep / "selection.json", selection)
            with self.assertRaises(ValueError):
                selection_provenance(sweep)

    def test_three_seeds_report_omits_original_resnet_but_keeps_all_evidence(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sweep, plan = self.fixture(root)
            baseline, output = root / "baseline", root / "comparison"
            for variant in ("vanilla", "masked", "resnet"):
                for seed in (1, 2, 3):
                    train(
                        replace(
                            Config(**plan["base_config"]),
                            seed=seed,
                            encoder="impala" if variant == "resnet" else "cnn",
                            resource_mask=variant != "vanilla",
                            output_dir=str(baseline / variant / f"seed_{seed}" / "ppo"),
                        )
                    )
            options = build_parser().parse_args(
                [
                    "--tuning-dir",
                    str(sweep),
                    "--baseline-dir",
                    str(baseline),
                    "--output-dir",
                    str(output),
                    "--total-timesteps",
                    "8",
                    "--num-envs",
                    "1",
                    "--num-steps",
                    "4",
                    "--minibatch-size",
                    "4",
                    "--max-episode-steps",
                    "4",
                    "--device",
                    "cpu",
                    "--sync-envs",
                    "--eval-episodes",
                    "2",
                    "--record-episodes",
                    "0",
                ]
            )
            with patch("src.ppo.frame_stacking_experiment.generate_masking_report"):
                run_tuned_experiment(options)
            recorded = ["vanilla", "masked", "resnet", "tuned"]
            evidence_before = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for variant in recorded
                for path in (output / variant).rglob("*")
                if path.is_file()
            }
            metadata_before = (output / "experiment.json").read_bytes()
            plots = []

            def inspect(figure, title, **kwargs):
                plots.append(
                    [
                        {
                            "labels": [line.get_label() for line in axis.lines],
                            "bands": len(axis.collections),
                            "containers": [
                                type(container).__name__
                                for container in axis.containers
                            ],
                            "ticks": [
                                tick.get_text() for tick in axis.get_xticklabels()
                            ],
                        }
                        for axis in figure.axes
                    ]
                )
                return figure_image(figure, title, **kwargs)

            with patch("src.ppo.masking_report.figure_image", side_effect=inspect):
                result = generate_masking_report(output)
            variants = ["vanilla", "masked", "tuned"]
            self.assertEqual(result["variants"], variants)
            self.assertEqual(result["recorded_variants"], recorded)
            self.assertNotIn("stacked", result["aggregate"])
            self.assertEqual(
                set(result["paired_comparisons"]),
                {"masked_minus_vanilla", "tuned_minus_vanilla", "tuned_minus_masked"},
            )
            for section in (
                "aggregate",
                "per_seed",
                "optimization",
                "behavior",
                "models",
            ):
                self.assertEqual(set(result[section]), set(variants))
            for variant in variants:
                self.assertEqual(result["aggregate"][variant]["seeds"], 3)
                for curves in result["curves"].values():
                    self.assertEqual(set(curves), set(variants))
                    self.assertEqual(curves[variant]["n_seeds"], 3)
                    self.assertIsNotNone(curves[variant]["std"])
            for variant in recorded:
                for seed in (1, 2, 3):
                    directory = output / variant / f"seed_{seed}"
                    state = torch.load(
                        directory / "ppo" / "checkpoint.pt", weights_only=True
                    )
                    self.assertEqual(state["global_step"], 8)
                    self.assertEqual(
                        state["config"]["gae_lambda"],
                        0.65 if variant == "tuned" else 0.95,
                    )
                    if variant != "tuned":
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
                    rows = read_csv(directory / "evaluation" / "episodes.csv")
                    self.assertEqual(
                        [int(row["environment_seed"]) for row in rows],
                        [2_000_000_000 + (seed - 1) * 2 + i for i in range(2)],
                    )
            for axes in plots[:5]:
                for axis in axes:
                    self.assertEqual(axis["bands"], 3)
                    for variant in variants:
                        self.assertTrue(
                            any(
                                label.startswith(VARIANTS[variant] + " (")
                                for label in axis["labels"]
                            )
                        )
            self.assertEqual(len(plots), 7)
            for axis in plots[5]:
                self.assertEqual(axis["containers"].count("ErrorbarContainer"), 3)
                self.assertEqual(
                    axis["ticks"], [VARIANTS[variant] for variant in variants]
                )
            self.assertEqual(plots[6][0]["containers"].count("BarContainer"), 3)
            for name in (
                "training_curve.csv",
                "evaluation_curve.csv",
                "training_length_curve.csv",
                "evaluation_length_curve.csv",
            ):
                self.assertEqual(
                    {row["variant"] for row in read_csv(output / name)}, set(variants)
                )
            self.assertEqual(
                evidence_before,
                {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for variant in recorded
                    for path in (output / variant).rglob("*")
                    if path.is_file()
                },
            )
            self.assertEqual((output / "experiment.json").read_bytes(), metadata_before)
            report = (
                (output / "report.html")
                .read_text()
                .split("<h2>Configuration and baseline hashes</h2>")[0]
            )
            self.assertNotIn("<td>resnet</td>", report)
            self.assertNotIn("resnet/seed_", report)
            self.assertIn(VARIANTS["tuned"] + " was trained from scratch", report)
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["variants"], recorded)
            self.assertEqual(
                metadata["baseline_hyperparameter_overrides"], {"gae_lambda": 0.95}
            )
            self.assertEqual(
                metadata["configuration_selection"]["selection"]["changes"],
                {"gae_lambda": 0.65},
            )


if __name__ == "__main__":
    unittest.main()
