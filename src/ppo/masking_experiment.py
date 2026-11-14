"""Compare resource-masked PPO with saved vanilla runs at the same step budget."""

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil

import torch

if __package__:
    from .evaluate import evaluate
    from .masking_report import generate_masking_report
    from .report import read_csv
    from .train import Config, add_training_arguments, select_device, train
else:
    from evaluate import evaluate
    from masking_report import generate_masking_report
    from report import read_csv
    from train import Config, add_training_arguments, select_device, train


def validate_baseline(
    directory, config, *, resource_mask=False, frame_stack=1, encoder="cnn"
):
    checkpoint = directory / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved = Config(
        **json.loads((directory / "config.json").read_text(encoding="utf-8"))
    )
    stored = Config(**state["config"])
    ignored = {
        "output_dir",
        "device",
        "sync_envs",
        "torch_threads",
        "checkpoint_interval",
    }
    expected = Config(
        **{
            **asdict(config),
            "resource_mask": resource_mask,
            "frame_stack": frame_stack,
            "encoder": encoder,
        }
    )
    for field in fields(Config):
        if field.name in ignored:
            continue
        wanted = getattr(expected, field.name)
        if (
            getattr(saved, field.name) != wanted
            or getattr(stored, field.name) != wanted
        ):
            raise ValueError(
                f"Baseline {directory}: incompatible {field.name}; expected {wanted}"
            )
    updates = read_csv(directory / "updates.csv")
    if (
        state["global_step"] != config.total_timesteps
        or not updates
        or int(updates[-1]["global_step"]) != config.total_timesteps
    ):
        raise ValueError(f"Baseline {directory} has not completed the requested budget")
    if not read_csv(directory / "episodes.csv"):
        raise ValueError(f"Baseline {directory} has no training episode evidence")
    evidence = {}
    for name in ("checkpoint.pt", "config.json", "episodes.csv", "updates.csv"):
        evidence[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return evidence


def run_masking_experiment(args):
    options = vars(args).copy()
    seeds = options.pop("seeds")
    output = Path(options.pop("output_dir"))
    baseline = Path(options.pop("baseline_dir"))
    eval_episodes = options.pop("eval_episodes")
    record_episodes = options.pop("record_episodes")
    options["resource_mask"] = True
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not 1 <= eval_episodes <= 100_000:
        raise ValueError("eval_episodes must be in [1, 100000]")
    if not 0 <= record_episodes <= eval_episodes:
        raise ValueError("record_episodes must be between zero and eval_episodes")
    if options["total_timesteps"] > 1_000_000:
        raise ValueError("This experiment has a maximum of 1M training steps per seed")
    configs = [Config(seed=seed, **options) for seed in seeds]
    evidence = {}
    for config in configs:
        config.validate()
        if config.frame_stack != 1:
            raise ValueError(
                "Use frame_stacking_experiment.py for the stacking ablation"
            )
        if config.encoder != "cnn":
            raise ValueError("Use resnet_experiment.py for the encoder ablation")
        if config.num_envs > 1000:
            raise ValueError(
                "num_envs must not overlap seed * 1000 training seed groups"
            )
        # Also validate the checkpoint's and CSV's recorded training configuration.
        evidence[str(config.seed)] = validate_baseline(
            baseline / f"seed_{config.seed}" / "ppo", config
        )
    train_seed_max = max(seeds) * 1000 + options["num_envs"] - 1
    if train_seed_max >= 1_000_000_000:
        raise ValueError("Training seeds must remain below held-out seed groups")
    device = select_device(options["device"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "training": options,
        "eval_episodes": eval_episodes,
        "record_episodes": record_episodes,
        "baseline_dir": str(baseline),
        "baseline_sha256": evidence,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "comparison": "Saved vanilla training; fresh matched final-checkpoint evaluation for both variants",
        "mask_information": "Recipe inputs and exact inventory only; no spatial checks or reward changes",
    }
    previous_metadata = baseline / "experiment.json"
    if previous_metadata.exists():
        metadata["baseline_experiment"] = json.loads(
            previous_metadata.read_text(encoding="utf-8")
        )
    for seed in seeds:
        source = baseline / f"seed_{seed}" / "ppo"
        destination = output / "vanilla" / f"seed_{seed}" / "ppo"
        destination.mkdir(parents=True)
        for name in evidence[str(seed)]:
            shutil.copy2(source / name, destination / name)

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
            seed_start = 1_000_000_000 + index * eval_episodes
            metadata["phase"] = f"Evaluating vanilla seed {seed}"
            update_report()
            evaluate(
                output / "vanilla" / f"seed_{seed}" / "evaluation",
                checkpoint=output
                / "vanilla"
                / f"seed_{seed}"
                / "ppo"
                / "checkpoint.pt",
                episodes=eval_episodes,
                seed_start=seed_start,
                max_episode_steps=config.max_episode_steps,
                device=str(device),
                behavior=True,
                record_episodes=record_episodes,
            )
            metadata["phase"] = f"Training masked seed {seed}"
            update_report()
            config.output_dir = str(output / "masked" / f"seed_{seed}" / "ppo")
            checkpoint = train(config, on_checkpoint=update_report)
            metadata["phase"] = f"Evaluating masked seed {seed}"
            update_report()
            evaluate(
                output / "masked" / f"seed_{seed}" / "evaluation",
                checkpoint=checkpoint,
                episodes=eval_episodes,
                seed_start=seed_start,
                max_episode_steps=config.max_episode_steps,
                device=str(device),
                behavior=True,
                record_episodes=record_episodes,
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
    print(f"Masking comparison: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--baseline-dir", default="outputs/ppo_comparison")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--record-episodes", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="outputs/ppo_masking_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    run_masking_experiment(parser.parse_args())
