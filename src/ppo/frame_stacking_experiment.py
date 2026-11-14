"""Compare saved PPO and masked PPO with newly trained masked RGB-frame-stack PPO."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil

import torch

if __package__:
    from .evaluate import evaluate
    from .masking_experiment import validate_baseline
    from .masking_report import generate_masking_report
    from .train import Config, add_training_arguments, select_device, train
else:
    from evaluate import evaluate
    from masking_experiment import validate_baseline
    from masking_report import generate_masking_report
    from train import Config, add_training_arguments, select_device, train


def run_ablation_experiment(
    args,
    *,
    trained_variant,
    baseline_specs,
    baseline_encoders=None,
    baseline_overrides=None,
    provenance=None,
):
    """Train one new variant and evaluate it alongside validated saved baselines."""
    options = vars(args).copy()
    seeds = options.pop("seeds")
    output = Path(options.pop("output_dir"))
    baseline = Path(options.pop("baseline_dir"))
    eval_episodes = options.pop("eval_episodes")
    record_episodes = options.pop("record_episodes")
    eval_seed_start = options.pop("eval_seed_start")
    options["resource_mask"] = True
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not 1 <= eval_episodes <= 100_000:
        raise ValueError("eval_episodes must be in [1, 100000]")
    if not 0 <= record_episodes <= eval_episodes:
        raise ValueError("record_episodes must be between zero and eval_episodes")
    if not 0 <= eval_seed_start < 2**31 - len(seeds) * eval_episodes:
        raise ValueError("Evaluation seed groups must fit in [0, 2**31)")
    if options["total_timesteps"] > 1_000_000:
        raise ValueError("This experiment has a maximum of 1M training steps per seed")
    configs = [Config(seed=seed, **options) for seed in seeds]
    evidence = {variant: {} for variant in baseline_specs}
    for config in configs:
        config.validate()
        if config.num_envs > 1000:
            raise ValueError(
                "num_envs must not overlap seed * 1000 training seed groups"
            )
        for variant in evidence:
            mask, stack = baseline_specs[variant]
            reference = replace(config, **(baseline_overrides or {}))
            evidence[variant][str(config.seed)] = validate_baseline(
                baseline / variant / f"seed_{config.seed}" / "ppo",
                reference,
                resource_mask=mask,
                frame_stack=stack,
                encoder=(baseline_encoders or {}).get(variant, "cnn"),
            )
    if max(seeds) * 1000 + options["num_envs"] - 1 >= eval_seed_start:
        raise ValueError("Held-out seed groups must be above all training seeds")
    device = select_device(options["device"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variants": [*baseline_specs, trained_variant],
        "seeds": seeds,
        "training": options,
        "eval_episodes": eval_episodes,
        "record_episodes": record_episodes,
        "eval_seed_start": eval_seed_start,
        "baseline_dir": str(baseline),
        "baseline_sha256": evidence,
        "baseline_hyperparameter_overrides": baseline_overrides or {},
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "comparison": (
            f"Saved training for {', '.join(baseline_specs)} is reused unchanged. "
            f"Only {trained_variant} is trained from scratch at the same budget. "
            "All final policies are evaluated afresh on matching held-out map lists."
        ),
        "frame_order": "oldest to newest; reset repeats initial RGB frame",
        "action_repeat": 1,
    }
    if provenance is not None:
        metadata["configuration_selection"] = provenance
    previous_metadata = baseline / "experiment.json"
    if previous_metadata.exists():
        metadata["baseline_experiment"] = json.loads(
            previous_metadata.read_text(encoding="utf-8")
        )
    for variant, groups in evidence.items():
        for seed, files in groups.items():
            source = baseline / variant / f"seed_{seed}" / "ppo"
            destination = output / variant / f"seed_{seed}" / "ppo"
            destination.mkdir(parents=True)
            for name in files:
                shutil.copy2(source / name, destination / name)
            state = torch.load(
                source / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            (destination / "model.json").write_text(
                json.dumps(
                    {
                        "encoder": state["config"].get("encoder", "cnn"),
                        "frame_stack": state["config"].get("frame_stack", 1),
                        "parameters": sum(
                            tensor.numel() for tensor in state["model"].values()
                        ),
                        "feature_size": state["model"]["actor.weight"].shape[1],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if trained_variant == "resnet":
        metadata["encoder_ablation"] = {
            "channels": [64, 128, 128],
            "residual_blocks_per_stage": 2,
            "dense_sizes": [256, 1024],
            "conv_normalization": "one-group GroupNorm; first image convolution unnormalized",
            "dense_normalization": "LayerNorm before dense layers",
            "encoder_initialization": "fan-in scaled, per-output weight norm",
            "actor_gain": 0.01,
            "critic_gain": 1.0,
            "value_target_normalization": False,
            "reference": "https://arxiv.org/abs/2307.03486",
            "scope": "Encoder/capacity/normalization ablation, not a paper reproduction",
        }

    def update_report():
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "experiment.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "experiment.json")
        generate_masking_report(output)

    update_report()
    try:
        for index, config in enumerate(configs):
            seed = config.seed
            evaluation = {
                "episodes": eval_episodes,
                "seed_start": eval_seed_start + index * eval_episodes,
                "max_episode_steps": config.max_episode_steps,
                "device": str(device),
                "behavior": True,
                "record_episodes": record_episodes,
            }
            for variant in baseline_specs:
                metadata["phase"] = f"Evaluating {variant} seed {seed}"
                update_report()
                directory = output / variant / f"seed_{seed}"
                evaluate(
                    directory / "evaluation",
                    checkpoint=directory / "ppo" / "checkpoint.pt",
                    **evaluation,
                )
                update_report()
            metadata["phase"] = f"Training {trained_variant} seed {seed}"
            update_report()
            config.output_dir = str(output / trained_variant / f"seed_{seed}" / "ppo")
            checkpoint = train(config, on_checkpoint=update_report)
            metadata["phase"] = f"Evaluating {trained_variant} seed {seed}"
            update_report()
            evaluate(
                output / trained_variant / f"seed_{seed}" / "evaluation",
                checkpoint=checkpoint,
                **evaluation,
            )
            update_report()
        metadata["status"] = "completed"
        metadata["phase"] = "Completed"
    except BaseException:
        metadata["status"] = "failed or interrupted"
        update_report()
        raise
    finally:
        update_report()
    print(f"Comparison report: {output / 'report.html'}", flush=True)


def run_frame_stacking_experiment(args):
    if args.frame_stack < 2:
        raise ValueError("The stacking ablation requires frame_stack >= 2")
    if getattr(args, "encoder", "cnn") != "cnn":
        raise ValueError("Use resnet_experiment.py for the encoder ablation")
    return run_ablation_experiment(
        args,
        trained_variant="stacked",
        baseline_specs={"vanilla": (False, 1), "masked": (True, 1)},
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_training_arguments(parser)
    parser.set_defaults(frame_stack=4, resource_mask=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3], help="Training seeds"
    )
    parser.add_argument(
        "--baseline-dir",
        default="outputs/ppo_masking",
        help="Completed comparison containing vanilla and masked training runs",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Held-out episodes per variant/seed",
    )
    parser.add_argument(
        "--record-episodes", type=int, default=10, help="GIF episodes per variant/seed"
    )
    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=1_500_000_000,
        help="First held-out map seed; default differs from previous experiments",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ppo_frame_stacking_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
        help="New comparison output directory",
    )
    return parser


if __name__ == "__main__":
    run_frame_stacking_experiment(build_parser().parse_args())
