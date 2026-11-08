"""Controlled short-budget hyperparameter sweep for the contrastive actor-critic.

Every candidate changes exactly one hyperparameter of the baseline configuration,
trains from scratch for a short budget on each seed, and is ranked by its mean
return on validation maps that are disjoint from the final held-out maps.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

if __package__:
    from . import contrastive_actor_critic as cac
    from .evaluate import evaluate
    from .report import read_csv, read_jsonl, rolling_mean, summarize_episodes
else:
    import contrastive_actor_critic as cac
    from evaluate import evaluate
    from report import read_csv, read_jsonl, rolling_mean, summarize_episodes

# (name, hyperparameter, value): one change per candidate.
ALTERNATIVES = (
    ("entropy_0.01", "entropy_loss_weight", 0.01),
    ("entropy_0.001", "entropy_loss_weight", 0.001),
    ("lr_1e-4", "actor_critic_lr", 1e-4),
    ("lr_1e-3", "actor_critic_lr", 1e-3),
    ("discount_0.99", "discount", 0.99),
    ("critic_weight_1.0", "critic_loss_weight", 1.0),
    ("game_batch_256", "game_batch", 256),
    ("seq_length_8", "seq_length", 8),
    ("warm_up_0", "warm_up_steps", 0),
)
VALIDATION_SEED_START = 1_800_000_000


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def plan_digest(plan):
    payload = {key: value for key, value in plan.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidate_config(plan, changes):
    """Baseline configuration with one candidate's changes applied."""
    cac.apply_overrides(plan["base_config"])
    cac.apply_overrides(changes)
    return cac.config_snapshot()


def validate_plan(plan):
    if plan.get("sha256") != plan_digest(plan):
        raise ValueError("Sweep plan changed after initialization")
    base = plan["base_config"]
    candidates = plan["candidates"]
    if candidates[0] != {"name": "baseline", "changes": {}}:
        raise ValueError("The first candidate must be the unchanged baseline")
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique")
    for candidate in candidates[1:]:
        (field, value), *rest = candidate["changes"].items()
        if rest or base[field] == value:
            raise ValueError(f"{candidate['name']} must change exactly one value")
    seeds = plan["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and unique")
    if (
        max(seeds) + base["num_envs"] + cac.STANDBY_SEED_OFFSET
        >= plan["validation_seed_start"]
    ):
        raise ValueError("Validation maps must be above all training seeds")
    start = plan["validation_seed_start"]
    if (
        start < 1_100_000_000
        or start + len(seeds) * plan["validation_episodes"] >= 2**31
    ):
        # Final held-out maps start at 1e9 (100 per seed).
        raise ValueError("Validation maps must avoid held-out maps and fit in int32")
    tasks = [
        {"candidate": candidate["name"], "seed": seed, "seed_index": index}
        for index, seed in enumerate(seeds)
        for candidate in candidates
    ]
    if plan["tasks"] != tasks:
        raise ValueError("Task list does not cover every candidate and seed")


def initialize(root, *, seeds, total_timesteps, validation_episodes):
    cac.apply_overrides({"total_timesteps": total_timesteps})
    base = cac.config_snapshot()
    base.pop("channels_hidden")
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": base,
        "seeds": seeds,
        "validation_episodes": validation_episodes,
        "validation_seed_start": VALIDATION_SEED_START,
        "candidates": [{"name": "baseline", "changes": {}}]
        + [
            {"name": name, "changes": {field: value}}
            for name, field, value in ALTERNATIVES
        ],
        "python": platform.python_version(),
        "method": "One hyperparameter changed per candidate versus a common baseline",
        "ranking": "Equal-weight mean episodic return across seeds on validation maps "
        "disjoint from training and from the final held-out maps (1e9+)",
    }
    plan["tasks"] = [
        {"candidate": candidate["name"], "seed": seed, "seed_index": index}
        for index, seed in enumerate(seeds)
        for candidate in plan["candidates"]
    ]
    plan["sha256"] = plan_digest(plan)
    validate_plan(plan)
    for candidate in plan["candidates"]:
        candidate_config(plan, candidate["changes"])
        cac.validate_config(10)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "sweep.json", plan)
    print(f"Initialized {len(plan['tasks'])} tasks: {root / 'sweep.json'}", flush=True)
    return plan


def load_plan(root):
    plan = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def task_directory(root, task):
    return root / task["candidate"] / f"seed_{task['seed']}"


def run_task(root, index, keep_checkpoint=False):
    plan = load_plan(root)
    task = plan["tasks"][index]
    candidate = next(c for c in plan["candidates"] if c["name"] == task["candidate"])
    config = candidate_config(plan, candidate["changes"])
    directory = task_directory(root, task)
    directory.mkdir(parents=True, exist_ok=False)
    status = {
        "task_index": index,
        **task,
        "changes": candidate["changes"],
        "plan_sha256": plan["sha256"],
        "config": config,
        "status": "training",
    }

    def update():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "status.json", status)

    update()
    try:
        checkpoint = cac.run(task["seed"], directory / "cac", on_checkpoint=update)
        # metrics.jsonl has no row for a final collection without completed episodes.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        status["training_steps"] = state["timesteps"]
        del state
        if status["training_steps"] < config["total_timesteps"]:
            raise ValueError("Training stopped before its step budget")
        status["status"] = "validating"
        update()
        evaluate(
            directory / "validation",
            checkpoint=checkpoint,
            episodes=plan["validation_episodes"],
            seed_start=plan["validation_seed_start"]
            + task["seed_index"] * plan["validation_episodes"],
        )
        status["validation"] = summarize_episodes(
            read_csv(directory / "validation" / "episodes.csv")
        )
        if not keep_checkpoint:
            # 0.6 GB each; the FEP home quota cannot hold one per sweep task.
            checkpoint.unlink()
            status["checkpoint"] = "deleted after validation"
        status["status"] = "completed"
        update()
    except BaseException as error:
        status["status"] = "failed or interrupted"
        status["error"] = f"{type(error).__name__}: {error}"
        update()
        raise
    return status


def run_all(root, parallel):
    """Run every unfinished task, `parallel` processes at a time."""
    plan = load_plan(root)
    pending = [
        index
        for index, task in enumerate(plan["tasks"])
        if not (task_directory(root, task) / "status.json").exists()
    ]
    (root / "logs").mkdir(exist_ok=True)
    running, failed = {}, []
    while pending or running:
        while pending and len(running) < parallel:
            index = pending.pop(0)
            log = (root / "logs" / f"task_{index:03d}.log").open("w")
            running[index] = (
                subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "task",
                        "--task-index",
                        str(index),
                        "--output-dir",
                        str(root),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                ),
                log,
            )
            print(f"Started task {index}: {plan['tasks'][index]}", flush=True)
        time.sleep(5)
        for index, (process, log) in list(running.items()):
            if process.poll() is not None:
                log.close()
                del running[index]
                if process.returncode:
                    failed.append(index)
                print(
                    f"Task {index} finished with code {process.returncode}", flush=True
                )
                write_results(root)
    if failed:
        raise RuntimeError(f"Tasks failed: {failed}; see {root / 'logs'}")


def collect(root):
    plan = load_plan(root)
    groups = {candidate["name"]: [] for candidate in plan["candidates"]}
    progress = []
    for task in plan["tasks"]:
        directory = task_directory(root, task)
        path = directory / "status.json"
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        metrics = read_jsonl(directory / "cac" / "metrics.jsonl")
        training = read_csv(directory / "cac" / "episodes.csv")
        if status is not None and status["status"] == "completed":
            if status["plan_sha256"] != plan["sha256"]:
                raise ValueError("Task completed against a different plan")
            rows = read_csv(directory / "validation" / "episodes.csv")
            maps = [
                plan["validation_seed_start"]
                + task["seed_index"] * plan["validation_episodes"]
                + episode
                for episode in range(plan["validation_episodes"])
            ]
            if [int(row["environment_seed"]) for row in rows] != maps:
                raise ValueError("Validation count/maps do not match the plan")
            returns = [float(row["episodic_return"]) for row in training]
            groups[task["candidate"]].append(
                {
                    "seed": task["seed"],
                    **status["validation"],
                    "training_return_last_100": (
                        float(np.mean(returns[-100:])) if returns else None
                    ),
                    "final_entropy": next(
                        (r["entropy"] for r in reversed(metrics) if "entropy" in r),
                        None,
                    ),
                }
            )
        progress.append(
            {
                **task,
                "status": status["status"] if status else "not started",
                "training_steps": metrics[-1]["timesteps"] if metrics else 0,
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
                "mean_crafter_score": float(
                    np.mean([item["crafter_score"] for item in values])
                ),
                "per_seed": values,
            }
        )
    ranking.sort(key=lambda item: (-item["mean_return"], item["name"] != "baseline"))
    baseline = next((r for r in ranking if r["name"] == "baseline"), None)
    for item in ranking:
        item["delta_vs_baseline"] = (
            item["mean_return"] - baseline["mean_return"] if baseline else None
        )
    return plan, {
        "plan_sha256": plan["sha256"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "all_complete": all(item["status"] == "completed" for item in progress),
        "ranking": ranking,
        "progress": progress,
    }


def figure_image(figure, title):
    from matplotlib import pyplot as plt

    stream = io.BytesIO()
    figure.tight_layout()
    figure.savefig(stream, format="png", dpi=130)
    plt.close(figure)
    image = base64.b64encode(stream.getvalue()).decode("ascii")
    return f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{image}">'


def write_results(root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plan, result = collect(root)
    if result["all_complete"]:
        winner = result["ranking"][0]
        result["selection"] = {
            "candidate": winner["name"],
            "changes": winner["changes"],
            "validation_mean_return": winner["mean_return"],
            "delta_vs_baseline": winner["delta_vs_baseline"],
            # Exact ties keep the plan's candidate order, fixed before training.
            "tied_with": [
                item["name"]
                for item in result["ranking"][1:]
                if abs(item["mean_return"] - winner["mean_return"]) < 1e-9
            ],
            "scope": f"Best of {len(plan['candidates'])} tested configurations after "
            f"{plan['base_config']['total_timesteps']:,} steps; not a global optimum "
            "or a 1M-step claim",
        }
        write_json(root / "selection.json", result["selection"])
    write_json(root / "results.json", result)

    images = []
    ranking = result["ranking"]
    if ranking:
        figure, axis = plt.subplots(figsize=(12, 5))
        positions = np.arange(len(ranking))
        axis.bar(
            positions,
            [item["mean_return"] for item in ranking],
            yerr=[item["sample_sd"] or 0 for item in ranking],
            color=[
                "tab:grey" if item["name"] == "baseline" else "tab:blue"
                for item in ranking
            ],
            capsize=4,
        )
        for position, item in zip(positions, ranking):
            axis.scatter(
                [position] * len(item["per_seed"]),
                [seed["mean_return"] for seed in item["per_seed"]],
                color="black",
                s=12,
                zorder=3,
            )
        axis.set(
            xticks=positions,
            xticklabels=[item["name"] for item in ranking],
            ylabel="Validation mean return",
        )
        axis.tick_params(axis="x", rotation=30)
        images.append(figure_image(figure, "Validation return by candidate"))

    figure, axis = plt.subplots(figsize=(12, 5))
    for candidate in plan["candidates"]:
        curves = []
        for task in plan["tasks"]:
            if task["candidate"] != candidate["name"]:
                continue
            rows = read_csv(task_directory(root, task) / "cac" / "episodes.csv")
            if rows:
                curves.append(
                    (
                        [int(row["global_step"]) for row in rows],
                        rolling_mean([float(row["episodic_return"]) for row in rows]),
                    )
                )
        if curves:
            grid = np.linspace(0, plan["base_config"]["total_timesteps"], 100)
            values = np.mean(
                [np.interp(grid, steps, returns) for steps, returns in curves], axis=0
            )
            axis.plot(
                grid,
                values,
                label=candidate["name"],
                linewidth=2.5 if candidate["name"] == "baseline" else 1.2,
                color="black" if candidate["name"] == "baseline" else None,
            )
    axis.set(
        xlabel="Environment steps",
        ylabel="Training return, last 100 episodes (seed mean)",
    )
    if axis.lines:
        axis.legend(ncol=2)
    images.append(figure_image(figure, "Training return by candidate"))

    def number(value, spec=".3f"):
        return "N/A" if value is None else format(value, spec)

    def mean_entropy(item):
        values = [s["final_entropy"] for s in item["per_seed"]]
        return None if None in values else float(np.mean(values))

    def per_seed_text(item):
        return ", ".join(f"{seed['mean_return']:.2f}" for seed in item["per_seed"])

    rows = "".join(
        f"<tr><td>{rank}</td><td>{html.escape(item['name'])}</td>"
        f"<td>{html.escape(json.dumps(item['changes']))}</td>"
        f"<td>{item['mean_return']:.3f}</td>"
        f"<td>{number(item['sample_sd'])}</td>"
        f"<td>{item['delta_vs_baseline']:+.3f}</td>"
        f"<td>{per_seed_text(item)}</td>"
        f"<td>{item['mean_crafter_score']:.2f}</td>"
        f"<td>{number(mean_entropy(item))}</td></tr>"
        for rank, item in enumerate(ranking, start=1)
    )
    progress = ", ".join(
        f"{item['candidate']}/{item['seed']}: {item['status']}"
        for item in result["progress"]
        if item["status"] != "completed"
    )
    selection = result.get("selection")
    tie_note = (
        f" Tied with {', '.join(selection['tied_with'])}; exact ties are broken by "
        "the candidate order fixed in the plan."
        if selection and selection["tied_with"]
        else ""
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Contrastive actor-critic tuning</title>
<style>
body {{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}}
img {{width:100%;height:auto}} table {{border-collapse:collapse;width:100%;margin:20px 0}}
td,th {{border:1px solid #ccd2dd;padding:8px;text-align:left}} pre {{white-space:pre-wrap}}
</style>
<h1>Contrastive actor-critic: one-at-a-time hyperparameter sweep</h1>
<p>Generated {html.escape(result['generated_at'])}.
{'All tasks complete.' if result['all_complete'] else 'Incomplete: ' + html.escape(progress)}</p>
<p>Each candidate changes exactly one hyperparameter of the baseline, trains from
scratch for {plan['base_config']['total_timesteps']:,} environment steps on seeds
{', '.join(map(str, plan['seeds']))}, and is evaluated with sampled actions on
{plan['validation_episodes']} validation maps per seed (environment seeds from
{plan['validation_seed_start']:,}), disjoint from training maps and from the final
held-out maps. Candidates are ranked by the equal-weight mean of per-seed mean
returns. The contrastive warm-up stays at {plan['base_config']['warm_up_steps']:,} steps
except where it is the tested parameter.</p>
{f"<p><strong>Selected: {html.escape(selection['candidate'])}</strong> "
 f"({html.escape(json.dumps(selection['changes']))}), validation mean return "
 f"{selection['validation_mean_return']:.3f} ({selection['delta_vs_baseline']:+.3f} vs baseline). "
 f"{html.escape(selection['scope'])}.{html.escape(tie_note)}</p>" if selection else ""}
<table><tr><th>Rank</th><th>Candidate</th><th>Change</th><th>Mean return</th>
<th>SD across seeds</th><th>Δ vs baseline</th><th>Per-seed returns</th>
<th>Crafter score</th><th>Final entropy</th></tr>{rows}</table>
<p>Three seeds and {plan['validation_episodes']} maps per seed are a small sample:
differences smaller than the seed standard deviation are not reliable. Short-budget
rankings may not transfer to 1M steps.</p>
{''.join(images)}
<h2>Plan</h2><pre>{html.escape(json.dumps(plan, indent=2))}</pre>
</html>"""
    (root / "report.html").write_text(document, encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "task", "run", "report"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cac_tuning"))
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--validation-episodes", type=int, default=50)
    parser.add_argument("--keep-checkpoints", action="store_true")
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args()
    if args.action == "init":
        initialize(
            args.output_dir,
            seeds=args.seeds,
            total_timesteps=args.total_timesteps,
            validation_episodes=args.validation_episodes,
        )
    elif args.action == "task":
        run_task(args.output_dir, args.task_index, args.keep_checkpoints)
    elif args.action == "run":
        if not (args.output_dir / "sweep.json").exists():
            initialize(
                args.output_dir,
                seeds=args.seeds,
                total_timesteps=args.total_timesteps,
                validation_episodes=args.validation_episodes,
            )
        run_all(args.output_dir, args.parallel)
        print(json.dumps(write_results(args.output_dir).get("selection"), indent=2))
    else:
        write_results(args.output_dir)
