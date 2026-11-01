"""Summarize the controlled short-budget sweep without pooling seeds or leaking test data."""

import html
import json
from pathlib import Path

import numpy as np

if __package__:
    from .masking_report import seed_curve
    from .report import aggregate_seeds, figure_image, plt, read_csv
    from .tuning import collect, load_plan, write_json
else:
    from masking_report import seed_curve
    from report import aggregate_seeds, figure_image, plt, read_csv
    from tuning import collect, load_plan, write_json


def generate_report(root):
    root = Path(root)
    plan = load_plan(root)
    data = collect(root)
    candidates = plan["candidates"]
    colors = plt.get_cmap("tab10")
    images, curves = [], {}
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, candidate in enumerate(candidates):
        series = []
        for seed in plan["seeds"]:
            rows = read_csv(
                root / candidate["name"] / f"seed_{seed}" / "ppo" / "updates.csv"
            )
            present = [row for row in rows if row["mean_episodic_return"] != ""]
            series.append(
                (
                    [int(row["global_step"]) for row in present],
                    [float(row["mean_episodic_return"]) for row in present],
                )
            )
        curve = seed_curve(series)
        curves[candidate["name"]] = curve
        if curve is None:
            continue
        mean = np.array(curve["mean"])
        axis.plot(curve["x"], mean, color=colors(index), label=candidate["name"])
        if curve["std"] is not None:
            sd = np.array(curve["std"])
            axis.fill_between(
                curve["x"], mean - sd, mean + sd, color=colors(index), alpha=0.15
            )
    axis.set(
        xlabel="Training environment steps",
        ylabel="Return, last 100 completed episodes",
    )
    if axis.lines:
        axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    figure.savefig(root / "training_return.png", dpi=150)
    images.append(figure_image(figure, "Training: seed means +/- one sample SD"))
    data["training_curves"] = curves
    figure, axis = plt.subplots(figsize=(11, 5))
    by_name = {item["name"]: item for item in data["ranking"]}
    for index, candidate in enumerate(candidates):
        item = by_name.get(candidate["name"])
        if item:
            axis.errorbar(
                index,
                item["mean_return"],
                yerr=item["sample_sd"],
                fmt="o",
                color=colors(index),
                capsize=6,
            )
    axis.set(
        xticks=range(len(candidates)),
        xticklabels=[item["name"] for item in candidates],
        ylabel="Mean original return on validation maps",
    )
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(root / "validation_return.png", dpi=150)
    images.append(
        figure_image(figure, "Validation at the exact budget: seed mean and sample SD")
    )
    metrics = (
        "approx_kl",
        "clip_fraction",
        "entropy",
        "value_loss",
        "explained_variance",
        "learning_rate",
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, metric in zip(axes.flat, metrics):
        for index, candidate in enumerate(candidates):
            series = []
            for seed in plan["seeds"]:
                rows = read_csv(
                    root / candidate["name"] / f"seed_{seed}" / "ppo" / "updates.csv"
                )
                present = [row for row in rows if row[metric] != ""]
                series.append(
                    (
                        [int(row["global_step"]) for row in present],
                        [float(row[metric]) for row in present],
                    )
                )
            curve = seed_curve(series)
            if curve is not None:
                mean = np.array(curve["mean"])
                axis.plot(
                    curve["x"], mean, color=colors(index), label=candidate["name"]
                )
                if curve["std"] is not None:
                    sd = np.array(curve["std"])
                    axis.fill_between(
                        curve["x"],
                        mean - sd,
                        mean + sd,
                        color=colors(index),
                        alpha=0.12,
                    )
        axis.set(title=metric.replace("_", " "), xlabel="Training steps")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(root / "optimization.png", dpi=150)
    images.append(
        figure_image(figure, "Optimization diagnostics", layout_rect=(0, 0, 1, 0.91))
    )
    results_path = root / "results.json"
    result = json.loads(results_path.read_text()) if results_path.exists() else None
    confirmation = {}
    if result:
        if result["plan_sha256"] != plan["sha256"]:
            raise ValueError("Confirmation results belong to another plan")
        data["selection"] = result["selection"]
        confirmation = {
            name: aggregate_seeds(values)
            for name, values in result["confirmation"].items()
        }
        data["confirmation"] = confirmation
    rows = "".join(
        f"<tr><td>{html.escape(item['name'])}</td><td>{html.escape(json.dumps(item['changes']))}</td>"
        f"<td>{item['mean_return']:.3f}</td><td>{item['sample_sd'] if item['sample_sd'] is not None else 'N/A'}</td></tr>"
        for item in data["ranking"]
    )
    tasks = "".join(
        f"<tr><td>{item['task_index']}</td><td>{item['candidate']}</td><td>{item['seed']}</td>"
        f"<td>{html.escape(item['status'])}</td><td>{item['training_steps']:,}</td><td>{item['validation_episodes']}</td></tr>"
        for item in data["progress"]
    )
    tests = "".join(
        f"<tr><td>{name}</td><td>{value['mean_return']:.3f}</td>"
        f"<td>{value['std_across_seed_means']}</td><td>{value['mean_crafter_score']:.3f}</td></tr>"
        for name, value in confirmation.items()
    )
    status = (
        "completed"
        if result
        else (
            "validation complete; confirmation pending"
            if data["all_complete"]
            else "running or incomplete"
        )
    )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Masked IMPALA PPO hyperparameter sweep</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;margin:16px 0}}
td,th{{border:1px solid #ccd2dd;padding:8px;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>Masked IMPALA PPO: controlled short-budget tuning</h1><p>Status: <b>{status}</b></p>
<p>Every run starts from scratch and uses exactly {plan['base_config']['total_timesteps']:,}
training steps, never more than 200k. Seven configurations share a common baseline;
each alternative changes exactly one hyperparameter. Seeds: {html.escape(str(plan['seeds']))}.
No old 1M-step rewards are mixed into these comparisons.</p>
<p>Linear learning-rate decay uses this short budget consistently across configurations;
these are not truncated 1M-step schedules. One-frame resource masking, IMPALA encoder,
original rewards, and all nonperturbed PPO settings remain unchanged.</p>
<p>Ranking uses only {plan['validation_episodes']} fresh validation episodes per seed.
All requested seeds must finish before a winner can be selected. Seed means receive equal weight;
SD bands/error bars show variability across seeds, not confidence intervals. Incomplete
configurations are shown as progress, never silently promoted as winners.</p>
{''.join(images)}
<h2>Validation ranking (complete configurations only)</h2>
<table><tr><th>Configuration</th><th>Only change vs baseline</th><th>Mean return</th><th>Seed SD</th></tr>{rows}</table>
<h2>Untouched confirmation set</h2>
<p>The winner is frozen in selection.json before any confirmation interactions.
Only the selected configuration and baseline are evaluated on {plan['confirmation_episodes']}
different maps per seed. Those results cannot change the selected hyperparameters.
No extra training is performed. Best means best among tested configurations after this
budget, not a proven global optimum or evidence of superiority at 1M steps.</p>
<table><tr><th>Policy</th><th>Test mean return</th><th>Seed SD</th><th>Crafter score</th></tr>{tests}</table>
<h2>Progress</h2><table><tr><th>Task</th><th>Configuration</th><th>Seed</th><th>Status</th>
<th>Training steps</th><th>Validation episodes</th></tr>{tasks}</table>
<h2>Frozen plan</h2><pre>{html.escape(json.dumps(plan, indent=2))}</pre>
<p>Raw evidence: configuration/seed_N/ppo, validation, and (selected/baseline only)
confirmation directories. Training CSVs, final checkpoints, independent task statuses,
and cluster launch/scheduler logs are preserved.</p></html>"""
    write_json(root / "summary.json", data)
    (root / "report.html").write_text(document, encoding="utf-8")
    return data
