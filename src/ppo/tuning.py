"""Controlled <=200k-step Masked-IMPALA PPO sweep with disjoint validation/test maps."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

if __package__:
    from .evaluate import evaluate
    from .report import read_csv, summarize_episodes
    from .train import Config, add_training_arguments, train
else:
    from evaluate import evaluate
    from report import read_csv, summarize_episodes
    from train import Config, add_training_arguments, train


ALTERNATIVES = (
    ("gamma_095", "gamma", 0.95),
    ("lambda_065", "gae_lambda", 0.65),
    ("lr_0003", "learning_rate", 3e-4),
    ("epochs_3", "update_epochs", 3),
    ("rollout_512", "num_steps", 512),
    ("kl_0015", "target_kl", 0.015),
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def plan_digest(plan):
    payload = {key: value for key, value in plan.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_plan(plan):
    if plan.get("sha256") != plan_digest(plan):
        raise ValueError("Sweep plan changed after initialization")
    base = Config(**plan["base_config"])
    base.validate()
    if not 1 <= base.total_timesteps <= 200_000:
        raise ValueError("Every sweep training run must use at most 200000 steps")
    if base.encoder != "impala" or not base.resource_mask or base.frame_stack != 1:
        raise ValueError("Sweep requires single-frame, resource-masked IMPALA PPO")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    for seed in seeds:
        replace(base, seed=seed).validate()
    if base.num_envs > 1000:
        raise ValueError("Worker seeds must not overlap adjacent seed groups")
    count, test_count = plan["validation_episodes"], plan["confirmation_episodes"]
    if (
        type(count) is not int
        or type(test_count) is not int
        or not 1 <= count <= 100_000
        or not 1 <= test_count <= 100_000
    ):
        raise ValueError("Evaluation counts must be integers in [1, 100000]")
    validation = plan["validation_seed_start"]
    test = plan["confirmation_seed_start"]
    if not max(seeds) * 1000 + base.num_envs - 1 < validation:
        raise ValueError("Validation maps must be above all training seed groups")
    if not validation + len(seeds) * count <= test:
        raise ValueError("Confirmation maps overlap validation maps")
    if test + len(seeds) * test_count > 2**31:
        raise ValueError("Evaluation map groups exceed supported seed range")
    candidates = plan["candidates"]
    if [candidate["name"] for candidate in candidates] != [
        "baseline",
        *[row[0] for row in ALTERNATIVES],
    ]:
        raise ValueError("Unexpected sweep candidates/order")
    if candidates[0]["changes"] != {}:
        raise ValueError("Baseline must remain unchanged")
    for candidate, (_, field, value) in zip(candidates[1:], ALTERNATIVES):
        if candidate["changes"] != {field: value}:
            raise ValueError(
                "Every candidate must change exactly its declared parameter"
            )
        config = replace(base, **candidate["changes"])
        config.validate()
        differences = [
            key for key in asdict(base) if getattr(config, key) != getattr(base, key)
        ]
        if differences != [field]:
            raise ValueError(
                "Each alternative must differ from baseline in exactly one field"
            )
    tasks = [
        {"candidate": item["name"], "seed": seed, "seed_index": index}
        for index, seed in enumerate(seeds)
        for item in candidates
    ]
    if plan["tasks"] != tasks:
        raise ValueError("Task list does not cover the planned candidate/seed groups")
    # A one-sample minibatch would not provide useful normalization statistics.
    for candidate in candidates:
        config = replace(base, **candidate["changes"])
        for size in (
            min(config.num_steps * config.num_envs, config.total_timesteps),
            config.total_timesteps % (config.num_steps * config.num_envs),
        ):
            if size and (
                min(size, config.minibatch_size) < 2
                or size % config.minibatch_size == 1
            ):
                raise ValueError(
                    "Sweep configuration would create singleton minibatches"
                )


def initialize(args):
    options = vars(args).copy()
    for name in ("action", "task_index"):
        options.pop(name, None)
    root = Path(options.pop("output_dir"))
    seeds = options.pop("seeds")
    settings = {
        name: options.pop(name)
        for name in (
            "validation_episodes",
            "confirmation_episodes",
            "validation_seed_start",
            "confirmation_seed_start",
        )
    }
    base = Config(**options)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": asdict(base),
        "seeds": seeds,
        **settings,
        "candidates": [{"name": "baseline", "changes": {}}]
        + [
            {"name": name, "changes": {field: value}}
            for name, field, value in ALTERNATIVES
        ],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "method": "One parameter per alternative vs a common baseline; no fine-tuning or extra training",
        "ranking": "Equal-weight mean original episodic return across all complete training seeds on validation maps",
        "schedule": "Existing linear learning-rate decay over the actual <=200k budget; not the first 200k of a 1M schedule",
    }
    plan["tasks"] = [
        {"candidate": item["name"], "seed": seed, "seed_index": index}
        for index, seed in enumerate(seeds)
        for item in plan["candidates"]
    ]
    plan["sha256"] = plan_digest(plan)
    validate_plan(plan)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "sweep.json", plan)
    print(f"Initialized {len(plan['tasks'])} tasks: {root / 'sweep.json'}", flush=True)
    return plan


def load_plan(root):
    plan = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def task_config(plan, index, root):
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("Task index outside the sweep")
    task = plan["tasks"][index]
    candidate = next(
        item for item in plan["candidates"] if item["name"] == task["candidate"]
    )
    config = replace(
        Config(**plan["base_config"]),
        **candidate["changes"],
        seed=task["seed"],
        output_dir=str(root / task["candidate"] / f"seed_{task['seed']}" / "ppo"),
    )
    config.validate()
    return task, config


def run_task(root, index):
    root = Path(root)
    plan = load_plan(root)
    task, config = task_config(plan, index, root)
    directory = Path(config.output_dir).parent
    directory.mkdir(parents=True, exist_ok=False)
    status = {
        "task_index": index,
        **task,
        "plan_sha256": plan["sha256"],
        "config": asdict(config),
        "status": "training",
    }

    def update():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "status.json", status)

    update()
    try:
        checkpoint = train(config, on_checkpoint=update)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["global_step"] != config.total_timesteps:
            raise ValueError("Checkpoint did not meet the exact training budget")
        status["training_steps"] = state["global_step"]
        status["status"] = "validating"
        update()
        evaluate(
            directory / "validation",
            checkpoint=checkpoint,
            episodes=plan["validation_episodes"],
            seed_start=plan["validation_seed_start"]
            + task["seed_index"] * plan["validation_episodes"],
            max_episode_steps=config.max_episode_steps,
            device=config.device,
        )
        status["validation"] = summarize_episodes(
            read_csv(directory / "validation" / "episodes.csv")
        )
        status["status"] = "completed"
        update()
    except BaseException as error:
        status["status"] = "failed or interrupted"
        status["error"] = f"{type(error).__name__}: {error}"
        update()
        raise
    return status


def collect(root, *, require_complete=False):
    root = Path(root)
    plan = load_plan(root)
    groups = {candidate["name"]: [] for candidate in plan["candidates"]}
    progress = []
    for index, task in enumerate(plan["tasks"]):
        directory = root / task["candidate"] / f"seed_{task['seed']}"
        path = directory / "status.json"
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        rows = read_csv(directory / "validation" / "episodes.csv")
        complete = status is not None and status["status"] == "completed"
        if complete:
            _, expected = task_config(plan, index, root)
            # Paths change after download; compare behavior/training fields, not the output path.
            saved = Config(
                **json.loads((directory / "ppo" / "config.json").read_text())
            )
            for name in asdict(saved):
                if name != "output_dir" and getattr(saved, name) != getattr(
                    expected, name
                ):
                    raise ValueError(
                        f"{task['candidate']}/{task['seed']}: config mismatch {name}"
                    )
            if (
                status["plan_sha256"] != plan["sha256"]
                or status["training_steps"] != expected.total_timesteps
            ):
                raise ValueError("Task completed against a different plan/budget")
            updates = read_csv(directory / "ppo" / "updates.csv")
            if (
                not updates
                or int(updates[-1]["global_step"]) != expected.total_timesteps
            ):
                raise ValueError("Completed task has incomplete training CSV")
            maps = [
                plan["validation_seed_start"]
                + task["seed_index"] * plan["validation_episodes"]
                + episode
                for episode in range(plan["validation_episodes"])
            ]
            if [int(row["environment_seed"]) for row in rows] != maps:
                raise ValueError("Validation count/maps do not match plan")
            config = json.loads((directory / "validation" / "config.json").read_text())
            if (
                config["action_selection"] != "sampled categorical"
                or not config["resource_mask"]
                or config["encoder"] != "impala"
                or config["frame_stack"] != 1
            ):
                raise ValueError("Evaluation used a different policy setup")
            groups[task["candidate"]].append(
                {
                    "seed": task["seed"],
                    **summarize_episodes(rows),
                }
            )
        elif require_complete:
            raise ValueError(f"Task {index} is incomplete; refusing to select a winner")
        updates = read_csv(directory / "ppo" / "updates.csv")
        progress.append(
            {
                "task_index": index,
                **task,
                "status": status["status"] if status else "not started",
                "training_steps": int(updates[-1]["global_step"]) if updates else 0,
                "validation_episodes": len(rows),
            }
        )
    ranking = []
    for candidate in plan["candidates"]:
        values = groups[candidate["name"]]
        if len(values) != len(plan["seeds"]):
            continue
        means = [item["mean_return"] for item in values]
        ranking.append(
            {
                **candidate,
                "mean_return": float(np.mean(means)),
                "sample_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                "per_seed": values,
            }
        )
    ranking.sort(
        key=lambda item: (
            -item["mean_return"],
            item["name"] != "baseline",
            item["name"],
        )
    )
    return {
        "plan_sha256": plan["sha256"],
        "ranking": ranking,
        "progress": progress,
        "all_complete": all(item["status"] == "completed" for item in progress),
    }


def finalize(root):
    root = Path(root)
    plan = load_plan(root)
    result = collect(root, require_complete=True)
    path = root / "selection.json"
    winner = result["ranking"][0]
    if path.exists():
        selection = json.loads(path.read_text())
        if (
            selection["plan_sha256"] != plan["sha256"]
            or selection["candidate"] != winner["name"]
        ):
            raise ValueError(
                "Frozen selection does not match the validated plan/ranking"
            )
    else:
        selection = {
            "plan_sha256": plan["sha256"],
            "candidate": winner["name"],
            "changes": winner["changes"],
            "validation_mean_return": winner["mean_return"],
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "scope": f"Best among seven tested configurations after {plan['base_config']['total_timesteps']} steps; not a global optimum or 1M-step claim",
        }
        write_json(path, selection)
    # Confirmation cannot influence selection: the choice above is frozen before test interactions.
    for name in dict.fromkeys(("baseline", selection["candidate"])):
        for index, seed in enumerate(plan["seeds"]):
            directory = root / name / f"seed_{seed}"
            checkpoint = directory / "ppo" / "checkpoint.pt"
            test = directory / "confirmation"
            if not test.exists():
                evaluate(
                    test,
                    checkpoint=checkpoint,
                    episodes=plan["confirmation_episodes"],
                    seed_start=plan["confirmation_seed_start"]
                    + index * plan["confirmation_episodes"],
                    max_episode_steps=plan["base_config"]["max_episode_steps"],
                    device=plan["base_config"]["device"],
                )
            maps = [
                plan["confirmation_seed_start"]
                + index * plan["confirmation_episodes"]
                + offset
                for offset in range(plan["confirmation_episodes"])
            ]
            if [
                int(row["environment_seed"]) for row in read_csv(test / "episodes.csv")
            ] != maps:
                raise ValueError(
                    "Partial/mismatched confirmation; never treat it as completed"
                )
    result["selection"] = selection
    result["confirmation"] = {
        name: [
            {
                "seed": seed,
                **summarize_episodes(
                    read_csv(
                        root / name / f"seed_{seed}" / "confirmation" / "episodes.csv"
                    )
                ),
            }
            for seed in plan["seeds"]
        ]
        for name in dict.fromkeys(("baseline", selection["candidate"]))
    }
    write_json(root / "results.json", result)
    print(
        json.dumps(
            {"selection": selection, "confirmation": result["confirmation"]}, indent=2
        )
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("action", choices=("init", "task", "finalize", "report"))
    add_training_arguments(parser)
    parser.set_defaults(total_timesteps=200_000, encoder="impala", resource_mask=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-dir", default="outputs/ppo_tuning")
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--confirmation-episodes", type=int, default=100)
    parser.add_argument("--validation-seed-start", type=int, default=1_800_000_000)
    parser.add_argument("--confirmation-seed-start", type=int, default=1_900_000_000)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.action == "init":
        initialize(args)
    elif args.action == "task":
        run_task(args.output_dir, args.task_index)
    elif args.action == "finalize":
        finalize(args.output_dir)
        if __package__:
            from .tuning_report import generate_report
        else:
            from tuning_report import generate_report
        generate_report(args.output_dir)
    else:
        if __package__:
            from .tuning_report import generate_report
        else:
            from tuning_report import generate_report
        generate_report(args.output_dir)
