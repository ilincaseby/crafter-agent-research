"""Train contrastive actor-critic seeds and compare them with a random baseline."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess

import torch

if __package__:
    from . import contrastive_actor_critic as cac
    from .evaluate import evaluate
    from .report import generate_report
else:
    import contrastive_actor_critic as cac
    from evaluate import evaluate
    from report import generate_report


def git_revision():
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_experiment(
    output,
    *,
    seeds=(1, 2, 3),
    eval_episodes=100,
    eval_envs=16,
    checkpoint_every=10,
    references=(),
    overrides=None,
):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")
    if eval_episodes < 1 or eval_episodes > 100_000:
        raise ValueError("eval_episodes must be in [1, 100000]")
    overrides = cac.apply_overrides(overrides or {})
    cac.validate_config(checkpoint_every)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference_index = []
    for number, (label, path) in enumerate(references, start=1):
        name = f"reference_{number}.json"
        shutil.copyfile(path, output / name)
        reference_index.append({"label": label, "file": name, "source": str(path)})
    (output / "references.json").write_text(
        json.dumps(reference_index, indent=2) + "\n", encoding="utf-8"
    )
    device = cac.get_device()
    revision = git_revision()
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "eval_episodes": eval_episodes,
        "training": cac.config_snapshot(),
        "overrides": overrides,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": revision,
        "source_of_truth": "FEP project source snapshot; git revision alone excludes uncommitted edits",
        "references": reference_index,
    }
    if revision is None:
        print(
            "Git revision unavailable; use the launcher's project source snapshot.",
            flush=True,
        )

    def update_report():
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "experiment.tmp"
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "experiment.json")
        generate_report(output)

    update_report()
    try:
        for index, seed in enumerate(seeds):
            directory = output / f"seed_{seed}"
            # Same held-out maps as the PPO experiment, disjoint from training
            # seeds (seed + worker and seed + worker + 10000).
            seed_start = 1_000_000_000 + index * eval_episodes
            evaluate(
                directory / "eval_random",
                episodes=eval_episodes,
                seed_start=seed_start,
                num_envs=eval_envs,
                device=device,
            )
            update_report()
            checkpoint = cac.run(
                seed,
                directory / "cac",
                checkpoint_every=checkpoint_every,
                device=device,
                on_checkpoint=update_report,
            )
            update_report()
            evaluate(
                directory / "eval_cac",
                checkpoint=checkpoint,
                episodes=eval_episodes,
                seed_start=seed_start,
                num_envs=eval_envs,
                device=device,
            )
            update_report()
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed or interrupted"
        raise
    finally:
        update_report()
    print(f"Report: {output / 'report.html'}", flush=True)
    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Override TOTAL_TIMESTEPS (default: 1M or the environment variable).",
    )
    parser.add_argument(
        "--warm-up-steps",
        type=int,
        help="Override WARM_UP_STEPS (default: 100k or the environment variable).",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Summary.json of another experiment on the same held-out maps "
        "(repeatable; a bare PATH is labelled PPO).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a hyperparameter, e.g. --set entropy_loss_weight=0.01.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cac_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
    )
    args = parser.parse_args()
    if args.total_timesteps is not None:
        cac.total_timesteps = args.total_timesteps
    if args.warm_up_steps is not None:
        cac.warm_up_steps = args.warm_up_steps
    run_experiment(
        args.output_dir,
        seeds=args.seeds,
        eval_episodes=args.eval_episodes,
        eval_envs=args.eval_envs,
        checkpoint_every=args.checkpoint_every,
        references=[
            tuple(item.split("=", 1)) if "=" in item else ("PPO", item)
            for item in args.reference
        ],
        overrides=dict(item.split("=", 1) for item in args.set),
    )
