"""Shared PPO ablation plots with seed mean/SD and behavioral evidence."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

if __package__:
    from .behavior import EVENTS
    from .report import (
        aggregate_seeds,
        figure_image,
        plt,
        read_csv,
        rolling_mean,
        summarize_episodes,
    )
else:
    from behavior import EVENTS
    from report import (
        aggregate_seeds,
        figure_image,
        plt,
        read_csv,
        rolling_mean,
        summarize_episodes,
    )


VARIANTS = {
    "vanilla": "PPO",
    "masked": "Masked-PPO",
    "stacked": "Masked-PPO + Frame Stacking",
    "resnet": "Masked-PPO + IMPALA ResNet",
    "tuned": "Masked-PPO + IMPALA ResNet (tuned)",
}
COLORS = {
    "vanilla": "#2468ad",
    "masked": "#cf681b",
    "stacked": "#25854b",
    "resnet": "#8853ac",
    "tuned": "#bc3447",
}
OPTIMIZATION_METRICS = (
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


def seed_curve(series):
    """Align seeds by last observed value on their common support; never extrapolate."""
    if not series or any(not len(x) for x, _ in series):
        return None
    start = max(x[0] for x, _ in series)
    end = min(x[-1] for x, _ in series)
    if start > end:
        return None
    grid = np.unique(np.concatenate([np.asarray(x) for x, _ in series]))
    grid = grid[(grid >= start) & (grid <= end)]
    aligned = np.array(
        [np.asarray(y)[np.searchsorted(x, grid, side="right") - 1] for x, y in series]
    )
    std = aligned.std(axis=0, ddof=1) if len(series) > 1 else None
    return {
        "x": grid.tolist(),
        "mean": aligned.mean(axis=0).tolist(),
        "std": std.tolist() if std is not None else None,
        "variance": (std**2).tolist() if std is not None else None,
        "n_seeds": len(series),
    }


def plot_curve(axis, variant, curve):
    if curve is None:
        return
    mean = np.array(curve["mean"])
    axis.plot(
        curve["x"],
        mean,
        label=f"{VARIANTS[variant]} (n={curve['n_seeds']})",
        color=COLORS[variant],
    )
    if curve["std"] is not None:
        sd = np.array(curve["std"])
        axis.fill_between(
            curve["x"], mean - sd, mean + sd, color=COLORS[variant], alpha=0.2
        )


def generate_masking_report(run_dir):
    root = Path(run_dir)
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    seeds, expected = experiment["seeds"], experiment["eval_episodes"]
    recorded_variants = experiment.get("variants", ["vanilla", "masked"])
    if (
        not recorded_variants
        or len(set(recorded_variants)) != len(recorded_variants)
        or any(variant not in VARIANTS for variant in recorded_variants)
    ):
        raise ValueError("Report variants must be unique registered PPO approaches")
    # Hide superseded approaches without changing the recorded experiment.
    variants = [
        variant
        for variant in recorded_variants
        if not (
            variant == "stacked"
            and ("resnet" in recorded_variants or "tuned" in recorded_variants)
            or variant == "resnet"
            and "tuned" in recorded_variants
        )
    ]
    result = {
        "status": experiment["status"],
        "variants": variants,
        "recorded_variants": recorded_variants,
        "per_seed": {},
        "aggregate": {},
        "curves": {
            "training": {},
            "evaluation": {},
            "training_length": {},
            "evaluation_length": {},
        },
        "optimization": {},
        "behavior": {},
        "models": {},
    }
    tables, media, progress = [], [], []
    completed = {variant: {} for variant in variants}
    evaluation_maps = {variant: {} for variant in variants}
    for variant in variants:
        training_series, evaluation_series = [], []
        training_lengths, evaluation_lengths = [], []
        optimization = {name: [] for name in OPTIMIZATION_METRICS}
        behavior_summaries = []
        result["per_seed"][variant] = {}
        result["models"][variant] = {}
        for seed in seeds:
            directory = root / variant / f"seed_{seed}"
            model_path = directory / "ppo" / "model.json"
            if model_path.exists():
                result["models"][variant][str(seed)] = json.loads(
                    model_path.read_text(encoding="utf-8")
                )
            updates = read_csv(directory / "ppo" / "updates.csv")
            for name in OPTIMIZATION_METRICS:
                present = [row for row in updates if row.get(name, "") != ""]
                optimization[name].append(
                    (
                        [int(row["global_step"]) for row in present],
                        [float(row[name]) for row in present],
                    )
                )
            training_episodes = read_csv(directory / "ppo" / "episodes.csv")
            training_lengths.append(
                (
                    [int(row["global_step"]) for row in training_episodes],
                    rolling_mean(
                        [float(row["episodic_length"]) for row in training_episodes]
                    ),
                )
            )
            valid = [row for row in updates if row["mean_episodic_return"] != ""]
            training_series.append(
                (
                    [int(row["global_step"]) for row in valid],
                    [float(row["mean_episodic_return"]) for row in valid],
                )
            )
            rows = read_csv(directory / "evaluation" / "episodes.csv")
            summary = summarize_episodes(rows)
            result["per_seed"][variant][str(seed)] = summary
            evaluation_series.append(
                (
                    np.arange(1, len(rows) + 1),
                    rolling_mean(
                        [float(row["episodic_return"]) for row in rows], window=10
                    ),
                )
            )
            evaluation_lengths.append(
                (
                    np.arange(1, len(rows) + 1),
                    rolling_mean(
                        [float(row["episodic_length"]) for row in rows], window=10
                    ),
                )
            )
            steps = int(updates[-1]["global_step"]) if updates else 0
            progress.append(
                f"{variant} seed {seed}: {steps:,} steps; {len(rows)}/{expected} eval episodes"
            )
            if summary:
                complete = len(rows) == expected
                if complete:
                    completed[variant][seed] = summary
                    evaluation_maps[variant][seed] = [
                        int(row["environment_seed"]) for row in rows
                    ]
                tables.append(
                    f"<tr><td>{variant}</td><td>{seed}</td>"
                    f"<td>{len(rows)}{' (partial)' if not complete else ''}</td>"
                    f"<td>{summary['mean_return']:.3f}</td>"
                    f"<td>{summary['mean_length']:.1f}</td>"
                    f"<td>{summary['crafter_score']:.3f}</td></tr>"
                )
            for entry in read_csv(
                directory / "evaluation" / "behavior" / "episodes.csv"
            ):
                evidence = json.loads(entry["summary"])
                behavior_summaries.append(evidence)
                if entry["video"] or any(evidence["candidate_counts"].values()):
                    relative = directory.relative_to(root) / "evaluation" / "behavior"
                    trace = html.escape(
                        (relative / entry["trace"]).as_posix(), quote=True
                    )
                    video = (
                        f'<a href="{html.escape((relative / entry["video"]).as_posix(), quote=True)}">GIF</a>'
                        if entry["video"]
                        else "not recorded"
                    )
                    candidates = (
                        ", ".join(
                            f"{event['event']}@{event['step']}"
                            for event in evidence["candidate_steps"]
                        )
                        or "none"
                    )
                    media.append(
                        f"<tr><td>{variant}</td><td>{seed}</td><td>{entry['episode']}</td>"
                        f"<td>{entry['environment_seed']}</td><td>{html.escape(candidates)}</td>"
                        f'<td>{video}; <a href="{trace}">trace</a></td></tr>'
                    )
        result["curves"]["training"][variant] = seed_curve(training_series)
        result["curves"]["evaluation"][variant] = seed_curve(evaluation_series)
        result["curves"]["training_length"][variant] = seed_curve(training_lengths)
        result["curves"]["evaluation_length"][variant] = seed_curve(evaluation_lengths)
        result["optimization"][variant] = {
            name: seed_curve(series) for name, series in optimization.items()
        }
        result["aggregate"][variant] = aggregate_seeds(
            list(completed[variant].values())
        )
        result["behavior"][variant] = {
            "episodes_observed": len(behavior_summaries),
            "events": {
                name: {
                    "count": sum(
                        item["candidate_counts"][name] for item in behavior_summaries
                    ),
                    "episodes": sum(
                        item["candidate_counts"][name] > 0
                        for item in behavior_summaries
                    ),
                }
                for name in EVENTS
            },
            "arrow_threat_steps": sum(
                item["arrow_threat_steps"] for item in behavior_summaries
            ),
            "skeleton_nearby_steps": sum(
                item["skeleton_nearby_steps"] for item in behavior_summaries
            ),
        }
    result["paired_comparisons"] = {}
    for index, variant in enumerate(variants):
        for baseline in variants[:index]:
            paired_seeds = sorted(
                completed[baseline].keys() & completed[variant].keys()
            )
            for seed in paired_seeds:
                if evaluation_maps[baseline][seed] != evaluation_maps[variant][seed]:
                    raise ValueError(
                        f"Unmatched evaluation maps for {baseline}/{variant} seed {seed}"
                    )
            differences = [
                completed[variant][seed]["mean_return"]
                - completed[baseline][seed]["mean_return"]
                for seed in paired_seeds
            ]
            result["paired_comparisons"][f"{variant}_minus_{baseline}"] = {
                "seeds": len(paired_seeds),
                "mean_return_difference": (
                    float(np.mean(differences)) if differences else None
                ),
                "std_across_seed_differences": (
                    float(np.std(differences, ddof=1)) if len(differences) > 1 else None
                ),
            }
    original_pair = result["paired_comparisons"].get("masked_minus_vanilla", {})
    result["paired_seeds"] = original_pair.get("seeds", 0)
    result["paired_mean_return_difference"] = original_pair.get(
        "mean_return_difference"
    )
    images = []
    for name, xlabel, ylabel in (
        (
            "training",
            "Training environment steps (all workers)",
            "Return, last 100 completed episodes",
        ),
        (
            "evaluation",
            "Held-out episode index (no learning)",
            "Return, last 10 held-out episodes",
        ),
        (
            "training_length",
            "Training environment steps (all workers)",
            "Episode length, last 100 completed episodes",
        ),
        (
            "evaluation_length",
            "Held-out episode index (no learning)",
            "Episode length, last 10 held-out episodes",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 4))
        for variant in variants:
            plot_curve(axis, variant, result["curves"][name][variant])
        axis.set(xlabel=xlabel, ylabel=ylabel)
        if axis.lines:
            axis.legend()
        figure.tight_layout()
        plot_name = name if name.endswith("_length") else f"{name}_return"
        figure.savefig(root / f"{plot_name}.png", dpi=150)
        images.append(
            figure_image(
                figure,
                f"{name.replace('_', ' ').title()}: seed mean and +/- 1 sample SD",
            )
        )
        with (root / f"{name}_curve.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["variant", "x", "mean", "sample_sd", "sample_variance", "n_seeds"]
            )
            for variant in variants:
                curve = result["curves"][name][variant]
                if curve is None:
                    continue
                for index, x in enumerate(curve["x"]):
                    writer.writerow(
                        [
                            variant,
                            x,
                            curve["mean"][index],
                            curve["std"][index] if curve["std"] is not None else "",
                            (
                                curve["variance"][index]
                                if curve["variance"] is not None
                                else ""
                            ),
                            curve["n_seeds"],
                        ]
                    )
    aggregates = []
    figure, axes = plt.subplots(3, 3, figsize=(14, 10))
    for axis, name in zip(axes.flat, OPTIMIZATION_METRICS):
        for variant in variants:
            plot_curve(axis, variant, result["optimization"][variant][name])
        axis.set(title=name.replace("_", " "), xlabel="Training environment steps")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(3, len(variants)))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(root / "optimization.png", dpi=150, bbox_inches="tight")
    images.append(
        figure_image(
            figure,
            "Optimization and throughput: seed mean and +/- 1 sample SD",
            layout_rect=(0, 0, 1, 0.91),
        )
    )
    figure, (axis, score_axis) = plt.subplots(1, 2, figsize=(12, 4))
    for index, variant in enumerate(variants):
        aggregate = result["aggregate"][variant]
        if aggregate:
            sd = aggregate["std_across_seed_means"]
            axis.errorbar(
                index,
                aggregate["mean_return"],
                yerr=sd,
                fmt="o",
                capsize=6,
                color=COLORS[variant],
            )
            score_axis.errorbar(
                index,
                aggregate["mean_crafter_score"],
                yerr=aggregate["score_std_across_seeds"],
                fmt="o",
                capsize=6,
                color=COLORS[variant],
            )
            aggregates.append(
                f"<tr><td>{variant}</td><td>{aggregate['seeds']}/{len(seeds)}</td>"
                f"<td>{aggregate['mean_return']:.3f}</td>"
                f"<td>{f'{sd:.3f}' if sd is not None else 'N/A (one seed)'}</td>"
                f"<td>{aggregate['mean_crafter_score']:.3f}</td></tr>"
            )
    axis.set(
        xticks=range(len(variants)),
        xticklabels=[VARIANTS[variant] for variant in variants],
        ylabel="Final held-out mean return",
    )
    score_axis.set(
        xticks=range(len(variants)),
        xticklabels=[VARIANTS[variant] for variant in variants],
        ylabel="Final held-out Crafter score",
    )
    for panel in (axis, score_axis):
        panel.tick_params(axis="x", labelsize=8, rotation=15)
    figure.tight_layout()
    figure.savefig(root / "final_return.png", dpi=150)
    images.append(
        figure_image(figure, "Final evaluation: mean and sample SD across seed means")
    )
    summaries = [
        summary for variant in completed.values() for summary in variant.values()
    ]
    if summaries:
        names = sorted(summaries[0]["achievement_success_percent"])
        figure, axis = plt.subplots(figsize=(12, 6))
        positions = np.arange(len(names))
        width = 0.8 / len(variants)
        for offset, variant in enumerate(variants):
            values = list(completed[variant].values())
            if values:
                matrix = np.array(
                    [
                        [item["achievement_success_percent"][name] for name in names]
                        for item in values
                    ]
                )
                axis.bar(
                    positions + (offset - (len(variants) - 1) / 2) * width,
                    matrix.mean(axis=0),
                    width=width,
                    label=VARIANTS[variant],
                    color=COLORS[variant],
                    yerr=matrix.std(axis=0, ddof=1) if len(values) > 1 else None,
                )
        axis.set(
            xticks=positions,
            xticklabels=names,
            ylabel="Successful held-out episodes (%)",
        )
        axis.tick_params(axis="x", rotation=75)
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / "achievements.png", dpi=150)
        images.append(
            figure_image(figure, "Achievement success: seed mean and sample SD")
        )
    behavior_rows = []
    for name, description in EVENTS.items():
        cells = []
        for variant in variants:
            evidence = result["behavior"][variant]
            event = evidence["events"][name]
            cells.append(
                f"<td>{event['episodes']}/{evidence['episodes_observed']} episodes; {event['count']} events</td>"
            )
        behavior_rows.append(
            f"<tr><td>{html.escape(description)}</td>{''.join(cells)}</tr>"
        )
    title = " versus ".join(VARIANTS[variant] for variant in variants)
    behavior_headings = "".join(
        f"<th>{html.escape(VARIANTS[variant])}</th>" for variant in variants
    )
    paired_text = "".join(
        f"<li>{html.escape(name)}: {entry['mean_return_difference']} across {entry['seeds']} completed pairs</li>"
        for name, entry in result["paired_comparisons"].items()
    )
    comparison = experiment.get(
        "comparison",
        "Vanilla training is reused; both final checkpoints are evaluated afresh.",
    )
    if variants != recorded_variants:
        resnet_label = VARIANTS["tuned" if "tuned" in variants else "resnet"]
        comparison = (
            "PPO and Masked-PPO training is reused unchanged. "
            f"{resnet_label} was trained from scratch at the same budget. "
            "All displayed final policies were evaluated on matching held-out map lists. "
            "Omitted approaches remain in the original experiment data and provenance."
        )
    stacking_note = (
        "<p>Frame stacking concatenates consecutive RGB observations oldest to newest, "
        "without action repeat or extra environment steps. Only the first convolution "
        "gains input channels. Histories reset per episode; masking always uses current "
        "inventory. A larger input layer is a capacity difference, not solely a memory "
        "change.</p>"
        if "stacked" in variants
        else ""
    )
    architecture_rows = "".join(
        f"<tr><td>{html.escape(VARIANTS[variant])}</td><td>{seed}</td>"
        f"<td>{model['parameters']:,}</td><td>{model['feature_size']}</td>"
        f"<td>{html.escape(model['encoder'])}</td><td>{model['frame_stack']}</td></tr>"
        for variant, models in result["models"].items()
        for seed, model in models.items()
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;margin:20px 0}}
td,th{{border:1px solid #ccd2dd;padding:8px;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(title)}</h1>
<p><b>Status: {html.escape(experiment['status'])}</b>. {html.escape(experiment.get('phase', ''))}</p>
<p>{html.escape('; '.join(progress))}</p>
<p>Same training budget, original rewards, and matched held-out maps.
{html.escape(comparison)}
Masking reads exact inventory and recipe inputs; this adds structured domain information.
It does not check tables, terrain, or targets and does not script a strategy.</p>
{stacking_note}
<p>IMPALA ResNet, when enabled, uses 64/128/128 channels, two residual blocks per
stage, GroupNorm in the convolutional encoder, and LayerNorm before 256/1024 dense
layers. It retains the current PPO optimizer, clipping, GAE, actor/critic gains,
and raw value targets. No pretrained weights or achievement distillation are used.
The comparison changes encoder capacity and normalization together, not residual
connections in isolation. Throughput measurements from reused training runs are
hardware-dependent, not a controlled speed benchmark.</p>
<p>For a tuned policy, only the changes recorded in configuration_selection differ
from the original ResNet hyperparameters. Saved baselines keep their original
configuration; baseline_hyperparameter_overrides records their reference values.
The tuned policy is initialized afresh for the full budget, not continued from
the short-budget sweep. Fresh evaluation maps do not influence configuration
selection.</p>
<h2>Model sizes</h2>
<table><tr><th>Variant</th><th>Seed</th><th>Parameters</th><th>Feature size</th>
<th>Encoder</th><th>Frames</th></tr>{architecture_rows}</table>
<p>Each curve equally weights the configured seeds. Haze is +/- 1 sample standard deviation
across seed curves, not a confidence interval or pooled episode variability.
Training aligns last-100-episode returns at common environment steps using the last observed value;
no extrapolation beyond shared support. Curves wait until every seed has data.
Evaluation shows last-10-episode returns versus episode index, not learning or checkpoint evaluation
during training. Final statistics use complete evaluations only. Partial progress is not a final result.</p>
{''.join(images)}
<h2>Final held-out comparison</h2>
<table><tr><th>Variant</th><th>Seeds complete</th><th>Mean return</th><th>Seed mean SD</th><th>Crafter score</th></tr>
{''.join(aggregates)}</table>
<p>Paired return differences on matching map lists:</p><ul>{paired_text}</ul>
<p>Three seeds provide limited evidence.</p>
<table><tr><th>Variant</th><th>Seed</th><th>Episodes</th><th>Mean return</th><th>Mean length</th><th>Crafter score</th></tr>
{''.join(tables)}</table>
<h2>Behavior investigation: candidates, not demonstrated strategies</h2>
<p>All evaluation episodes have read-only state/action traces; only the first
{experiment['record_episodes']} episodes per variant/seed have GIFs. GIFs reuse policy images
every two steps through step 2000, avoiding extra rendering that would advance Crafter's RNG.
Trace step numbers identify exact events. Global terrain/object state is analysis-only, never policy input.
Adjacent mining does not establish tunnel systems; traversing a constructed path tile does not establish crossing
a lake (stone placed on water must first be mined into a walkable path);
a sideways escape does not establish intentional dodging; nearby mining followed by a kill
does not establish surprise or flanking. Inspect trajectories, repeated instances, and opportunities
before drawing conclusions. A zero count is not proof the behavior is impossible.</p>
<table><tr><th>Observable candidate</th>{behavior_headings}</tr>{''.join(behavior_rows)}</table>
<p>Arrow-threat and skeleton-nearby exposure counts are in summary.json.
Natural maps may provide few opportunities. These are exploratory observations, not controlled causal tests.</p>
<table><tr><th>Variant</th><th>Seed</th><th>Episode</th><th>Map seed</th><th>Candidate@step</th><th>Evidence</th></tr>
{''.join(media)}</table>
<p>Keep the behavior directories alongside this HTML for linked GIFs and compressed JSONL traces.
Each trace begins with terrain_xy[x][y] and material IDs, followed by terrain deltas, player
inventory/position/facing, arrow and skeleton IDs/positions, actions, and raw rewards.</p>
<h2>Configuration and baseline hashes</h2>
<details><summary>Full recorded experiment provenance (including omitted approaches)</summary>
<pre>{html.escape(json.dumps(experiment, indent=2))}</pre></details></html>"""
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
    generate_masking_report(parser.parse_args().run_dir)
