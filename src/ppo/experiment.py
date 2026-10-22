"""Train multiple independent PPO seeds and compare with a random baseline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess

import torch

if __package__:
    from .evaluate import evaluate
    from .report import generate_report
    from .train import Config, add_training_arguments, select_device, train
else:
    from evaluate import evaluate
    from report import generate_report
    from train import Config, add_training_arguments, select_device, train


def run_experiment(args):
    options = vars(args).copy()
    seeds = options.pop("seeds")
    output = Path(options.pop("output_dir"))
    eval_episodes = options.pop("eval_episodes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")
    if eval_episodes < 1 or eval_episodes > 100_000:
        raise ValueError("eval_episodes must be in [1, 100000]")
    for seed in seeds:
        Config(seed=seed, **options).validate()
    device = select_device(options["device"])
    output.mkdir(parents=True, exist_ok=False)
    git = shutil.which("git")
    revision = (
        subprocess.run([git, "rev-parse", "HEAD"], text=True, capture_output=True)
        if git is not None
        else None
    )
    git_revision = (
        revision.stdout.strip()
        if revision is not None and revision.returncode == 0
        else None
    )
    if git_revision is None:
        print(
            "Git revision unavailable; use the launcher's project source snapshot.",
            flush=True,
        )
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "eval_episodes": eval_episodes,
        "training": options,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": git_revision,
        "source_of_truth": "FEP project source snapshot; git revision alone excludes uncommitted edits",
        "reference": "https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py",
    }

    def update_report():
        temporary = output / "experiment.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "experiment.json")
        generate_report(output)

    update_report()
    try:
        for index, seed in enumerate(seeds):
            directory = output / f"seed_{seed}"
            # Disjoint from training seeds (seed * 1000 + worker index), and
            # disjoint across evaluation groups. Both policies get identical maps.
            seed_start = 1_000_000_000 + index * eval_episodes
            evaluate(
                directory / "eval_random",
                episodes=eval_episodes,
                seed_start=seed_start,
                max_episode_steps=options["max_episode_steps"],
                device=str(device),
            )
            update_report()
            checkpoint = train(
                Config(seed=seed, output_dir=str(directory / "ppo"), **options),
                on_checkpoint=update_report,
            )
            update_report()
            evaluate(
                directory / "eval_ppo",
                checkpoint=checkpoint,
                episodes=eval_episodes,
                seed_start=seed_start,
                max_episode_steps=options["max_episode_steps"],
                device=str(device),
            )
            update_report()
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed or interrupted"
        update_report()
        raise
    finally:
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        update_report()
    print(f"Report: {output / 'report.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="outputs/ppo_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    run_experiment(parser.parse_args())
