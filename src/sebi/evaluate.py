"""Evaluate a contrastive actor-critic checkpoint or a uniform-random policy on held-out maps."""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import crafter
import gym
import numpy as np
import torch

if __package__:
    from . import contrastive_actor_critic as cac
else:
    import contrastive_actor_critic as cac

FIELDS = [
    "episode",
    "environment_seed",
    "episodic_return",
    "episodic_length",
    "achievements",
]


def make_eval_env(seed, size):
    # Same maps as the PPO evaluation: the first episode of crafter.Env(seed).
    return cac.CrafterGymEnv(crafter.Env(seed=seed, size=size))


def evaluate(
    output,
    *,
    checkpoint=None,
    episodes=100,
    seed_start=1_000_000_000,
    num_envs=16,
    device=None,
):
    if episodes < 1 or num_envs < 1 or seed_start < 0:
        raise ValueError(
            "episodes and num_envs must be positive; seed_start must be nonnegative"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = device or cac.get_device()
    config = cac.config_snapshot()
    model = None
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        config = state["config"]
        model, _ = cac.build_model(config, device)
        model.load_state_dict(state["model"])
        model.eval()
    size = (config["width"], config["height"])
    shape = (config["seq_length"], 3, config["height"], config["width"])
    torch.manual_seed(seed_start)
    rng = np.random.default_rng(seed_start)
    metadata = {
        "policy": "contrastive_actor_critic" if model is not None else "random",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "episodes": episodes,
        "seed_start": seed_start,
        "device": str(device),
        "action_selection": (
            "sampled categorical" if model is not None else "uniform random"
        ),
        "seq_length": config["seq_length"],
    }
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        log = csv.DictWriter(stream, FIELDS)
        log.writeheader()
        stream.flush()
        for first in range(0, episodes, num_envs):
            seeds = [
                seed_start + episode
                for episode in range(first, min(first + num_envs, episodes))
            ]
            env_fns = [partial(make_eval_env, seed, size) for seed in seeds]
            env = (
                gym.vector.SyncVectorEnv(env_fns)
                if cac.vector_env == "sync"
                else gym.vector.AsyncVectorEnv(env_fns, context=cac.vector_env_context)
            )
            try:
                results = play_batch(env, model, rng, shape, device)
            finally:
                env.close()
            for offset, (seed, result) in enumerate(zip(seeds, results)):
                log.writerow(
                    {"episode": first + offset + 1, "environment_seed": seed, **result}
                )
            stream.flush()
            print(
                f"{metadata['policy']} evaluation episodes={first + len(seeds)}/{episodes} "
                f"mean_return={np.mean([r['episodic_return'] for r in results]):.2f}",
                flush=True,
            )
    return output


@torch.no_grad()
def play_batch(env, model, rng, shape, device):
    """Play the first episode of every environment in the vector."""
    count = env.num_envs
    windows = torch.zeros(count, *shape, dtype=torch.uint8, device=device)

    def push(frames):
        frames = torch.as_tensor(np.asarray(frames), device=device).permute(0, 3, 1, 2)
        return torch.cat([windows[:, 1:], frames[:, None]], dim=1)

    observations, _ = env.reset()
    windows = push(observations)
    returns = np.zeros(count)
    lengths = np.zeros(count, dtype=int)
    results = [None] * count
    while any(result is None for result in results):
        if model is None:
            actions = rng.integers(env.single_action_space.n, size=count)
        else:
            logits, *_ = model(cac.as_model_input(windows))
            actions = torch.distributions.Categorical(logits=logits).sample()
            actions = actions.cpu().numpy()
        observations, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        frames = np.array(observations, dtype=np.uint8, copy=True)
        for index in np.flatnonzero(dones):
            frames[index] = infos["final_observation"][index]
        windows = push(frames)
        for index in range(count):
            if results[index] is not None:
                continue
            returns[index] += rewards[index]
            lengths[index] += 1
            if dones[index]:
                achievements = infos["final_info"][index]["achievements"]
                results[index] = {
                    "episodic_return": float(returns[index]),
                    "episodic_length": int(lengths[index]),
                    "achievements": json.dumps(
                        {name: int(value) for name, value in achievements.items()}
                    ),
                }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Omit to evaluate the random agent."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()
    evaluate(
        args.output_dir,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        seed_start=args.seed_start,
        num_envs=args.num_envs,
    )
