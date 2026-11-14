"""Full-budget confirmation of the frozen short-sweep winner against three methods."""

from argparse import Namespace
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

if __package__:
    from .frame_stacking_experiment import build_parser as comparison_parser
    from .frame_stacking_experiment import run_ablation_experiment
    from .train import Config
    from .tuning import load_plan
else:
    from frame_stacking_experiment import build_parser as comparison_parser
    from frame_stacking_experiment import run_ablation_experiment
    from train import Config
    from tuning import load_plan


def selection_provenance(root):
    root = Path(root)
    plan = load_plan(root)
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    if (
        selection["plan_sha256"] != plan["sha256"]
        or results["plan_sha256"] != plan["sha256"]
        or not results["all_complete"]
        or results["selection"] != selection
        or len(results["ranking"]) != len(plan["candidates"])
    ):
        raise ValueError("Selection does not belong to a completed frozen tuning sweep")
    winner = results["ranking"][0]
    if (
        winner["name"] != selection["candidate"]
        or winner["changes"] != selection["changes"]
    ):
        raise ValueError("Frozen winner differs from the original validation ranking")
    if selection["changes"] != {"gae_lambda": 0.65}:
        raise ValueError(
            "This confirmation is for the selected gae_lambda=0.65 experiment"
        )
    for candidate in results["ranking"]:
        if sorted(row["seed"] for row in candidate["per_seed"]) != sorted(
            plan["seeds"]
        ):
            raise ValueError("Tuning ranking is missing requested training seeds")
    return {
        "selection": selection,
        "sweep_plan": plan,
        "evidence_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("sweep.json", "selection.json", "results.json")
        },
        "training": "Fresh initialization for the full budget; no continuation of a 200k checkpoint",
        "uncertainty": "The short-budget gain was small and did not establish superiority at 1M steps",
    }


def run_tuned_experiment(args):
    options = vars(args).copy()
    provenance = selection_provenance(options.pop("tuning_dir"))
    sweep = provenance["sweep_plan"]
    expected = Config(
        **{
            **sweep["base_config"],
            **provenance["selection"]["changes"],
        }
    )
    # Only the training budget and operational options change when scaling up.
    ignored = {
        "seed",
        "total_timesteps",
        "output_dir",
        "device",
        "sync_envs",
        "torch_threads",
        "checkpoint_interval",
    }
    for name in asdict(expected):
        if name not in ignored and options[name] != getattr(expected, name):
            raise ValueError(
                f"Full-budget policy must retain selected configuration: {name}"
            )
    if options["seeds"] != sweep["seeds"]:
        raise ValueError("Full-budget seeds must match the selected experiment")
    if options["eval_seed_start"] < (
        sweep["confirmation_seed_start"]
        + len(sweep["seeds"]) * sweep["confirmation_episodes"]
    ):
        raise ValueError("Full-budget evaluation must use a fresh held-out seed range")
    return run_ablation_experiment(
        Namespace(**options),
        trained_variant="tuned",
        baseline_specs={
            "vanilla": (False, 1),
            "masked": (True, 1),
            "resnet": (True, 1),
        },
        baseline_encoders={"resnet": "impala"},
        baseline_overrides={"gae_lambda": sweep["base_config"]["gae_lambda"]},
        provenance=provenance,
    )


def build_parser():
    parser = comparison_parser()
    parser.description = __doc__
    parser.set_defaults(
        total_timesteps=1_000_000,
        encoder="impala",
        frame_stack=1,
        resource_mask=True,
        gae_lambda=0.65,
        baseline_dir="outputs/ppo_resnet",
        eval_seed_start=2_000_000_000,
        output_dir="outputs/ppo_tuned_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    parser.add_argument(
        "--tuning-dir",
        default="outputs/ppo_tuning",
        help="Completed short-budget sweep provenance",
    )
    return parser


if __name__ == "__main__":
    run_tuned_experiment(build_parser().parse_args())
