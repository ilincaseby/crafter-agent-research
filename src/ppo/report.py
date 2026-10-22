"""Create an offline HTML report and machine-readable summary from raw CSV logs."""

import argparse
import base64
import csv
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize_episodes(rows):
    if not rows:
        return None
    returns = np.array([float(row["episodic_return"]) for row in rows])
    achievements = [json.loads(row["achievements"]) for row in rows]
    names = sorted(achievements[0])
    if any(sorted(episode) != names for episode in achievements):
        raise ValueError("All episodes must record the same achievement keys")
    success = {
        name: 100 * float(np.mean([episode[name] > 0 for episode in achievements]))
        for name in names
    }
    return {
        "episodes": len(rows),
        "mean_return": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)) if len(rows) > 1 else 0.0,
        "mean_length": float(np.mean([float(row["episodic_length"]) for row in rows])),
        "achievement_success_percent": success,
        "crafter_score": float(np.expm1(np.log1p(list(success.values())).mean())),
    }


def aggregate_seeds(summaries):
    if not summaries:
        return None
    returns = np.array([summary["mean_return"] for summary in summaries])
    scores = np.array([summary["crafter_score"] for summary in summaries])
    return {
        "seeds": len(summaries),
        "mean_return": float(returns.mean()),
        "std_across_seed_means": (
            float(returns.std(ddof=1)) if len(returns) > 1 else None
        ),
        "mean_crafter_score": float(scores.mean()),
        "score_std_across_seeds": (
            float(scores.std(ddof=1)) if len(scores) > 1 else None
        ),
    }


def figure_image(figure, title):
    stream = io.BytesIO()
    figure.tight_layout()
    figure.savefig(stream, format="png", dpi=130)
    plt.close(figure)
    image = base64.b64encode(stream.getvalue()).decode("ascii")
    return f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{image}">'


def rolling_mean(values, window=100):
    values = np.asarray(values, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    starts = np.maximum(0, np.arange(len(values)) + 1 - window)
    return (cumulative[1:] - cumulative[starts]) / (np.arange(len(values)) + 1 - starts)


def generate_report(run_dir):
    root = Path(run_dir)
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    seeds = experiment["seeds"]
    result = {
        "status": experiment["status"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "per_seed": {},
        "aggregate": {},
    }
    training, updates, evaluation = {}, {}, {"ppo": [], "random": []}
    table_rows = []
    for seed in seeds:
        directory = root / f"seed_{seed}"
        training[seed] = read_csv(directory / "ppo" / "episodes.csv")
        updates[seed] = read_csv(directory / "ppo" / "updates.csv")
        per_seed = {}
        for policy in ("ppo", "random"):
            rows = read_csv(directory / f"eval_{policy}" / "episodes.csv")
            summary = summarize_episodes(rows)
            per_seed[policy] = summary
            # Only finished evaluations contribute to the cross-seed comparison.
            complete = len(rows) == experiment["eval_episodes"]
            if summary is not None:
                if complete:
                    evaluation[policy].append(summary)
                table_rows.append(
                    f"<tr><td>{seed}</td><td>{policy}</td>"
                    f"<td>{len(rows)}{' (partial)' if not complete else ''}</td>"
                    f"<td>{summary['mean_return']:.3f}</td>"
                    f"<td>{summary['return_std']:.3f}</td>"
                    f"<td>{summary['mean_length']:.1f}</td>"
                    f"<td>{summary['crafter_score']:.3f}</td></tr>"
                )
        completed_steps = int(updates[seed][-1]["global_step"]) if updates[seed] else 0
        per_seed["training_steps"] = completed_steps
        per_seed["training_episodes"] = len(training[seed])
        result["per_seed"][str(seed)] = per_seed

    for policy, summaries in evaluation.items():
        result["aggregate"][policy] = aggregate_seeds(summaries)
    paired = [
        entry["ppo"]["mean_return"] - entry["random"]["mean_return"]
        for entry in result["per_seed"].values()
        if entry["ppo"] is not None
        and entry["random"] is not None
        and entry["ppo"]["episodes"] == experiment["eval_episodes"]
        and entry["random"]["episodes"] == experiment["eval_episodes"]
    ]
    result["paired_mean_return_difference"] = float(np.mean(paired)) if paired else None
    result["paired_seeds"] = len(paired)
    images = []
    figure, axis = plt.subplots(figsize=(10, 4))
    for seed, rows in training.items():
        if rows:
            axis.plot(
                [int(row["global_step"]) for row in rows],
                rolling_mean([float(row["episodic_return"]) for row in rows]),
                label=f"PPO seed {seed}",
            )
    baseline = result["aggregate"]["random"]
    if baseline:
        axis.axhline(
            baseline["mean_return"],
            color="black",
            linestyle="--",
            label="Random held-out evaluation mean",
        )
    axis.set(
        xlabel="Environment steps (all workers)",
        ylabel="Return, last 100 completed episodes",
    )
    if axis.lines:
        axis.legend()
    images.append(figure_image(figure, "Training episodic return"))

    figure, axes = plt.subplots(3, 3, figsize=(14, 10))
    metric_names = (
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "sps",
    )
    for axis, name in zip(axes.flat, metric_names):
        for seed, rows in updates.items():
            present = [row for row in rows if row[name] != ""]
            if present:
                axis.plot(
                    [int(row["global_step"]) for row in present],
                    [float(row[name]) for row in present],
                    label=f"seed {seed}",
                )
        axis.set(title=name.replace("_", " "), xlabel="Environment steps")
        if axis.lines:
            axis.legend()
    images.append(figure_image(figure, "Optimization and throughput"))
    if any(evaluation.values()):
        summaries = next(values for values in evaluation.values() if values)
        names = sorted(summaries[0]["achievement_success_percent"])
        figure, axis = plt.subplots(figsize=(12, 6))
        positions = np.arange(len(names))
        for offset, (policy, values) in enumerate(evaluation.items()):
            if values:
                rates = [
                    np.mean(
                        [value["achievement_success_percent"][name] for value in values]
                    )
                    for name in names
                ]
                axis.bar(
                    positions + (offset - 0.5) * 0.4, rates, width=0.4, label=policy
                )
        axis.set(xticks=positions, xticklabels=names, ylabel="Successful episodes (%)")
        axis.tick_params(axis="x", rotation=70)
        axis.legend()
        images.append(figure_image(figure, "Held-out achievement success"))

    aggregate_rows = []
    for policy, aggregate in result["aggregate"].items():
        if aggregate:
            deviation = aggregate["std_across_seed_means"]
            deviation_text = (
                f"{deviation:.3f}" if deviation is not None else "N/A (one seed)"
            )
            aggregate_rows.append(
                f"<tr><td>{policy}</td><td>{aggregate['seeds']}/{len(seeds)}</td>"
                f"<td>{aggregate['mean_return']:.3f}</td><td>{deviation_text}</td>"
                f"<td>{aggregate['mean_crafter_score']:.3f}</td></tr>"
            )
    progress = ", ".join(
        f"seed {seed}: {entry['training_steps']:,}/{experiment['training']['total_timesteps']:,} steps"
        for seed, entry in result["per_seed"].items()
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Crafter PPO results</title>
<style>
body {{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}}
img {{width:100%;height:auto}} table {{border-collapse:collapse;width:100%;margin:20px 0}}
td,th {{border:1px solid #ccd2dd;padding:8px;text-align:left}} pre {{white-space:pre-wrap}}
</style>
<h1>Crafter: PPO versus uniform random</h1>
<p><strong>Status: {html.escape(experiment['status'])}</strong>.
Generated {html.escape(result['generated_at'])}.</p>
<p>{html.escape(progress)}</p>
<p>Each trained policy and its random baseline use the same held-out environment
seed list, 64x64 RGB observations, unmodified Crafter rewards, 17 actions, and an
episode limit of {experiment['training']['max_episode_steps']:,} steps.
PPO actions are sampled, not chosen greedily. Evaluation never updates weights.</p>
<h2>Held-out comparison (completed evaluations only)</h2>
<table><tr><th>Policy</th><th>Seeds complete</th><th>Mean return</th>
<th>Sample SD across seed means</th><th>Mean Crafter score</th></tr>
{''.join(aggregate_rows)}</table>
<p>Paired mean return difference (PPO minus random):
{result['paired_mean_return_difference']} across {len(paired)} completed pairs.</p>
<p>Each seed receives equal weight. These are standard deviations, not confidence
intervals. Three seeds are a small sample; this experiment does not establish that
Crafter is solved or reproduce a published baseline.</p>
<h2>Per-seed held-out results</h2>
<table><tr><th>Seed</th><th>Policy</th><th>Episodes</th><th>Mean return</th>
<th>Episode return SD</th><th>Mean length</th><th>Crafter score</th></tr>
{''.join(table_rows)}</table>
<p>Crafter score = exp(mean(log(1 + achievement success percentage))) - 1,
over all recorded achievements, including those never achieved.
Training curves use the last 100 completed episodes; partial episodes are excluded.
Losses are sample-weighted averages across all minibatches and epochs in an update,
not rewards. Loss need not decrease monotonically as the on-policy data changes.
The dashed random line is held-out evaluation, not a random training curve.</p>
{''.join(images)}
<h2>Configuration and provenance</h2>
<pre>{html.escape(json.dumps(experiment, indent=2))}</pre>
<p>Raw evidence: seed_N/ppo/updates.csv, seed_N/ppo/episodes.csv,
seed_N/eval_ppo/episodes.csv, seed_N/eval_random/episodes.csv, TensorBoard event files,
and checkpoint.pt. The FEP run also preserves launch.json, job.slurm, installed
packages, scheduler logs, and the source snapshot.</p>
<p>Algorithm reference: <a href="https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py">
CleanRL PPO Atari</a>; this implementation uses Crafter-specific inputs and
time-limit handling, without Atari reward clipping, frame skipping, or life resets.</p>
</html>"""
    for name, content in (
        ("summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n"),
        ("report.html", document),
    ):
        destination = root / name
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    generate_report(parser.parse_args().run_dir)
