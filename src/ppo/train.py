"""Readable pixel-based PPO for Crafter, following CleanRL's PPO organization.

Read train() in order: collect a rollout, compute advantages, then update the
policy several times using the same rollout. See README.md for the equations.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import random
import time

import crafter
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    seed: int = 1
    total_timesteps: int = 1_000_000
    num_envs: int = 8
    num_steps: int = 256
    minibatch_size: int = 256
    update_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    max_episode_steps: int = 10_000
    device: str = "auto"
    sync_envs: bool = False
    torch_threads: int = 1
    checkpoint_interval: int = 50
    output_dir: str = "outputs/ppo"

    def validate(self):
        for name in (
            "total_timesteps",
            "num_envs",
            "num_steps",
            "minibatch_size",
            "update_epochs",
            "max_episode_steps",
            "torch_threads",
            "checkpoint_interval",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or self.seed >= 2**31 // 1000:
            raise ValueError("seed must be in [0, 2**31 // 1000)")
        if self.total_timesteps % self.num_envs:
            raise ValueError("total_timesteps must be divisible by num_envs")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("learning_rate", "max_grad_norm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be in (0, 1)")
        for name in ("ent_coef", "vf_coef"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.target_kl is not None and (
            not np.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive")


class CrafterEnv(gym.Env):
    """Translate Crafter's legacy API to Gymnasium without Atari wrappers."""

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=10_000):
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(3, 64, 64), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(len(crafter.constants.actions))
        self.game = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.game is None:
            self.game = crafter.Env(
                seed=seed,
                size=(64, 64),
                reward=True,
                length=self.max_episode_steps,
            )
        return self.game.reset().transpose(2, 0, 1).copy(), {}

    def step(self, action):
        observation, reward, done, info = self.game.step(int(action))
        # Crafter's discount distinguishes death from the episode time limit.
        terminated = bool(done and info["discount"] == 0)
        truncated = bool(done and not terminated)
        episode_info = {"achievements": info["achievements"]} if done else {}
        return (
            observation.transpose(2, 0, 1).copy(),
            float(reward),
            terminated,
            truncated,
            episode_info,
        )


def make_env(max_episode_steps):
    return gym.wrappers.RecordEpisodeStatistics(CrafterEnv(max_episode_steps))


def initialize(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Agent(nn.Module):
    """One shared visual encoder, with separate policy and value heads."""

    def __init__(self, num_actions):
        super().__init__()
        self.encoder = nn.Sequential(
            initialize(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            initialize(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            initialize(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            initialize(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = initialize(nn.Linear(512, num_actions), gain=0.01)
        self.critic = initialize(nn.Linear(512, 1), gain=1.0)

    def get_value(self, observations):
        features = self.encoder(observations.float() / 255.0)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(self, observations, actions=None):
        features = self.encoder(observations.float() / 255.0)
        distribution = Categorical(logits=self.actor(features))
        if actions is None:
            actions = distribution.sample()
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            self.critic(features).squeeze(-1),
        )


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """dones[t] describes the transition AFTER action t, not before it."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for step in reversed(range(len(rewards))):
        following_value = next_value if step == len(rewards) - 1 else values[step + 1]
        nonterminal = (~dones[step]).float()
        delta = rewards[step] + gamma * following_value * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def ppo_loss(
    agent, observations, actions, old_log_probs, old_values, advantages, returns, config
):
    _, log_probs, entropy, values = agent.get_action_and_value(observations, actions)
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    # Minimize the negative of PPO's pessimistic, clipped surrogate objective.
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef)
    policy_loss = torch.maximum(unclipped, clipped).mean()

    clipped_values = old_values + (values - old_values).clamp(
        -config.clip_coef, config.clip_coef
    )
    value_loss = (
        0.5
        * torch.maximum(
            (values - returns).square(), (clipped_values - returns).square()
        ).mean()
    )
    entropy = entropy.mean()
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    with torch.no_grad():
        metrics = {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
            "clip_fraction": ((ratio - 1).abs() > config.clip_coef)
            .float()
            .mean()
            .item(),
        }
    return loss, metrics


class CsvLog:
    def __init__(self, path, fields):
        self.file = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is visible to PyTorch")
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(
        "cuda" if name != "cpu" and torch.cuda.is_available() else "cpu"
    )


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, agent, optimizer, config, global_step):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": global_step,
            "num_actions": int(agent.actor.out_features),
        },
        temporary,
    )
    temporary.replace(path)


def train(config, on_checkpoint=None):
    config.validate()
    device = select_device(config.device)
    seed_everything(config.seed, config.torch_threads)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    factories = [
        partial(make_env, config.max_episode_steps) for _ in range(config.num_envs)
    ]
    vector_class = (
        gym.vector.SyncVectorEnv if config.sync_envs else gym.vector.AsyncVectorEnv
    )
    envs = vector_class(factories, **({} if config.sync_envs else {"context": "spawn"}))
    writer = SummaryWriter(str(output / "tensorboard"))
    episode_log = CsvLog(
        output / "episodes.csv",
        [
            "global_step",
            "episode",
            "episodic_return",
            "episodic_length",
            "achievements",
        ],
    )
    update_log = CsvLog(
        output / "updates.csv",
        [
            "global_step",
            "update",
            "learning_rate",
            "total_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "grad_norm",
            "explained_variance",
            "sps",
            "elapsed_seconds",
            "completed_episodes",
            "mean_episodic_return",
        ],
    )
    try:
        agent = Agent(envs.single_action_space.n).to(device)
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=config.learning_rate, eps=1e-5
        )
        observation, _ = envs.reset(
            seed=[config.seed * 1000 + index for index in range(config.num_envs)]
        )
        observation = torch.as_tensor(observation, device=device)
        global_step, update = 0, 0
        episode_returns = []
        start_time = time.monotonic()
        print(
            f"Training seed={config.seed} device={device} output={output}", flush=True
        )

        while global_step < config.total_timesteps:
            update += 1
            rollout_steps = min(
                config.num_steps,
                (config.total_timesteps - global_step) // config.num_envs,
            )
            shape = (rollout_steps, config.num_envs)
            # Pixels stay uint8: a full float32 rollout would use four times the memory.
            observations = torch.empty(
                (*shape, 3, 64, 64), dtype=torch.uint8, device=device
            )
            actions = torch.empty(shape, dtype=torch.long, device=device)
            old_log_probs = torch.empty(shape, device=device)
            rewards = torch.empty(shape, device=device)
            dones = torch.empty(shape, dtype=torch.bool, device=device)
            values = torch.empty(shape, device=device)
            learning_rate = config.learning_rate * (
                1 - global_step / config.total_timesteps
            )
            optimizer.param_groups[0]["lr"] = learning_rate

            # 1. Collect experience using a frozen copy of the current policy.
            for step in range(rollout_steps):
                observations[step] = observation
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(observation)
                actions[step], old_log_probs[step], values[step] = (
                    action,
                    log_prob,
                    value,
                )
                next_observation, reward, terminated, truncated, infos = envs.step(
                    action.cpu().numpy()
                )
                global_step += config.num_envs
                rewards[step] = torch.as_tensor(reward, device=device)
                dones[step] = torch.as_tensor(terminated | truncated, device=device)

                # Gymnasium 0.29 autoresets. Bootstrap a time limit from its FINAL
                # observation, never the reset observation; death has zero bootstrap.
                timeout_indices = np.flatnonzero(truncated & ~terminated)
                if len(timeout_indices):
                    final_observations = np.stack(
                        [infos["final_observation"][i] for i in timeout_indices]
                    )
                    with torch.no_grad():
                        timeout_values = agent.get_value(
                            torch.as_tensor(final_observations, device=device)
                        )
                    rewards[step, timeout_indices] += config.gamma * timeout_values
                for info in infos.get("final_info", []):
                    if info is None:
                        continue
                    episode_return = float(info["episode"]["r"].item())
                    episode_length = int(info["episode"]["l"].item())
                    episode_returns.append(episode_return)
                    episode_log.write(
                        {
                            "global_step": global_step,
                            "episode": len(episode_returns),
                            "episodic_return": episode_return,
                            "episodic_length": episode_length,
                            "achievements": json.dumps(info["achievements"]),
                        }
                    )
                    writer.add_scalar(
                        "charts/episodic_return", episode_return, global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episode_length, global_step
                    )
                observation = torch.as_tensor(next_observation, device=device)

            # 2. Estimate advantages and value targets; never backpropagate through them.
            with torch.no_grad():
                advantages, returns = compute_gae(
                    rewards,
                    values,
                    dones,
                    agent.get_value(observation),
                    config.gamma,
                    config.gae_lambda,
                )
            batch_observations = observations.flatten(0, 1)
            batch_actions = actions.flatten()
            batch_log_probs = old_log_probs.flatten()
            batch_values = values.flatten()
            batch_advantages = advantages.flatten()
            batch_returns = returns.flatten()
            batch_size = len(batch_actions)

            # 3. Reuse this rollout for several shuffled minibatch optimization passes.
            metric_sums = {}
            samples = 0
            for epoch in range(config.update_epochs):
                indices = torch.randperm(batch_size, device=device)
                epoch_kl, epoch_samples = 0.0, 0
                for start in range(0, batch_size, config.minibatch_size):
                    minibatch = indices[start : start + config.minibatch_size]
                    loss, metrics = ppo_loss(
                        agent,
                        batch_observations[minibatch],
                        batch_actions[minibatch],
                        batch_log_probs[minibatch],
                        batch_values[minibatch],
                        batch_advantages[minibatch],
                        batch_returns[minibatch],
                        config,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Nonfinite PPO loss at step {global_step}"
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        config.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    metrics["grad_norm"] = grad_norm.item()
                    count = len(minibatch)
                    for name, number in metrics.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + number * count
                    samples += count
                    epoch_kl += metrics["approx_kl"] * count
                    epoch_samples += count
                if (
                    config.target_kl is not None
                    and epoch_kl / epoch_samples > config.target_kl
                ):
                    break

            elapsed = time.monotonic() - start_time
            variance = batch_returns.var(unbiased=False).item()
            explained_variance = (
                1 - (batch_returns - batch_values).var(unbiased=False).item() / variance
                if variance > 0
                else None
            )
            metrics = {name: value / samples for name, value in metric_sums.items()}
            metrics.update(
                {
                    "global_step": global_step,
                    "update": update,
                    "learning_rate": learning_rate,
                    "explained_variance": explained_variance,
                    "sps": global_step / elapsed,
                    "elapsed_seconds": elapsed,
                    "completed_episodes": len(episode_returns),
                    "mean_episodic_return": (
                        float(np.mean(episode_returns[-100:]))
                        if episode_returns
                        else None
                    ),
                }
            )
            update_log.write(metrics)
            for name, value in metrics.items():
                if value is not None:
                    writer.add_scalar(name, value, global_step)
            writer.flush()
            if update == 1 or update % 10 == 0 or global_step == config.total_timesteps:
                print(
                    f"seed={config.seed} step={global_step}/{config.total_timesteps} "
                    f"return100={metrics['mean_episodic_return']} "
                    f"loss={metrics['total_loss']:.4f} SPS={metrics['sps']:.1f}",
                    flush=True,
                )
            if (
                update % config.checkpoint_interval == 0
                or global_step == config.total_timesteps
            ):
                save_checkpoint(
                    output / "checkpoint.pt", agent, optimizer, config, global_step
                )
                if on_checkpoint is not None:
                    on_checkpoint()
        return output / "checkpoint.pt"
    finally:
        envs.close()
        writer.close()
        episode_log.close()
        update_log.close()


def add_training_arguments(parser):
    defaults = Config()
    for name in (
        "total_timesteps",
        "num_envs",
        "num_steps",
        "minibatch_size",
        "update_epochs",
        "max_episode_steps",
        "torch_threads",
        "checkpoint_interval",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--sync-envs", action="store_true", help="Run environments in the main process."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/ppo")
    train(Config(**vars(parser.parse_args())))
