"""Evaluate a trained PPO checkpoint or a uniform-random policy on held-out maps."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .behavior import BehaviorRecorder
    from .train import Agent, CsvLog, make_env, seed_everything, select_device
else:
    from behavior import BehaviorRecorder
    from train import Agent, CsvLog, make_env, seed_everything, select_device


def evaluate(
    output,
    *,
    checkpoint=None,
    episodes=100,
    seed_start=1_000_000,
    max_episode_steps=10_000,
    device="auto",
    behavior=False,
    record_episodes=0,
):
    if episodes < 1 or max_episode_steps < 1 or seed_start < 0:
        raise ValueError(
            "episodes and max_episode_steps must be positive; seed_start must be nonnegative"
        )
    if record_episodes < 0 or record_episodes > episodes:
        raise ValueError("record_episodes must be between zero and episodes")
    if record_episodes and not behavior:
        raise ValueError("Recordings require behavior=True")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = select_device(device)
    seed_everything(seed_start)
    agent = None
    resource_mask = False
    frame_stack = 1
    encoder = "cnn"
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["config"]["max_episode_steps"] != max_episode_steps:
            raise ValueError(
                "Evaluation horizon must match the checkpoint's training horizon"
            )
        frame_stack = state["config"].get("frame_stack", 1)
        encoder = state["config"].get("encoder", "cnn")
        agent = Agent(state["num_actions"], frame_stack, encoder).to(device)
        agent.load_state_dict(state["model"])
        agent.eval()
        resource_mask = state["config"].get("resource_mask", False)
    env = make_env(max_episode_steps, resource_mask, frame_stack)
    metadata = {
        "policy": "ppo" if agent is not None else "random",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "episodes": episodes,
        "seed_start": seed_start,
        "max_episode_steps": max_episode_steps,
        "device": str(device),
        "action_selection": (
            "sampled categorical" if agent is not None else "uniform random"
        ),
        "resource_mask": resource_mask,
        "frame_stack": frame_stack,
        "encoder": encoder,
        "behavior": behavior,
        "record_episodes": record_episodes,
    }
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    log = CsvLog(
        output / "episodes.csv",
        [
            "episode",
            "environment_seed",
            "episodic_return",
            "episodic_length",
            "achievements",
        ],
    )
    behavior_log = None
    if behavior:
        (output / "behavior").mkdir()
        behavior_log = CsvLog(
            output / "behavior" / "episodes.csv",
            ["episode", "environment_seed", "trace", "video", "summary"],
        )
    recorder = None
    try:
        for episode in range(episodes):
            seed = seed_start + episode
            observation, info = env.reset(seed=seed)
            env.action_space.seed(seed)
            torch.manual_seed(seed)
            episode_return, length = 0.0, 0
            if behavior:
                recorder = BehaviorRecorder(
                    output / "behavior" / f"episode_{episode + 1:04d}",
                    env.unwrapped.game,
                    observation[-3:],
                    record_video=episode < record_episodes,
                )
            done = False
            while not done:
                if agent is None:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(
                            torch.as_tensor(observation[None], device=device),
                            action_masks=(
                                torch.as_tensor(
                                    info["action_mask"][None], device=device
                                )
                                if resource_mask
                                else None
                            ),
                        )
                    action = action.item()
                observation, reward, terminated, truncated, info = env.step(action)
                if recorder is not None:
                    recorder.observe(action, reward, observation[-3:])
                episode_return += reward
                length += 1
                done = terminated or truncated
            if recorder is not None:
                recorder.close()
                behavior_log.write(
                    {
                        "episode": episode + 1,
                        "environment_seed": seed,
                        "trace": f"episode_{episode + 1:04d}.jsonl.gz",
                        "video": (
                            f"episode_{episode + 1:04d}.gif"
                            if episode < record_episodes
                            else ""
                        ),
                        "summary": json.dumps(recorder.summary()),
                    }
                )
                recorder = None
            log.write(
                {
                    "episode": episode + 1,
                    "environment_seed": seed,
                    "episodic_return": episode_return,
                    "episodic_length": length,
                    "achievements": json.dumps(info["achievements"]),
                }
            )
            if episode == 0 or (episode + 1) % 10 == 0 or episode + 1 == episodes:
                print(
                    f"{metadata['policy']} evaluation episode={episode + 1}/{episodes} "
                    f"return={episode_return:.2f} length={length}",
                    flush=True,
                )
    finally:
        if recorder is not None:
            recorder.close()
        if behavior_log is not None:
            behavior_log.close()
        env.close()
        log.close()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Omit to evaluate the random agent."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--behavior", action="store_true")
    parser.add_argument("--record-episodes", type=int, default=0)
    args = parser.parse_args()
    evaluate(
        args.output_dir,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_episode_steps=args.max_episode_steps,
        device=args.device,
        behavior=args.behavior,
        record_episodes=args.record_episodes,
    )
