"""On-policy actor-critic with GAE; also the shared core of the contrastive run."""

import json
import logging
import math
import os
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import monotonic

import crafter
import numpy as np
import torch

if __package__:
    from .contrastive_actor_critic import (
        achievement_list,
        get_device,
        save_plots,
        set_seed,
    )
    from .models import ActorCritic, LatentStateEncoder
else:
    from contrastive_actor_critic import (
        achievement_list,
        get_device,
        save_plots,
        set_seed,
    )
    from models import ActorCritic, LatentStateEncoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    total_steps: int = 1_000_000
    rollout_steps: int = 128
    batch_size: int = 16
    seq_length: int = 20
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    critic_weight: float = 0.5
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    entropy_decay_steps: int = 500_000
    max_grad_norm: float = 0.5
    height: int = 64
    width: int = 64
    hidden_size: int = 128
    checkpoint_every: int = 100
    seed: int = 42

    def __post_init__(self):
        for name in (
            "total_steps",
            "rollout_steps",
            "batch_size",
            "seq_length",
            "entropy_decay_steps",
            "hidden_size",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if min(self.height, self.width) < 32:
            raise ValueError(
                "height and width must be at least 32 for Crafter rendering"
            )
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0 <= self.entropy_end <= self.entropy_start:
            raise ValueError("entropy coefficients must satisfy 0 <= end <= start")
        if min(self.learning_rate, self.max_grad_norm) <= 0 or self.critic_weight < 0:
            raise ValueError("Invalid learning rate, gradient norm or critic weight")


def config_from_env(config_type):
    defaults = config_type()
    return config_type(
        **{
            field.name: type(getattr(defaults, field.name))(
                os.environ.get(field.name.upper(), getattr(defaults, field.name))
            )
            for field in fields(defaults)
        }
    )


def entropy_coefficient(config, environment_steps):
    fraction = min(max(environment_steps / config.entropy_decay_steps, 0.0), 1.0)
    return config.entropy_start + fraction * (config.entropy_end - config.entropy_start)


def compute_gae(rewards, values, next_values, discounts, boundaries, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    following = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in reversed(range(len(rewards))):
        delta = (
            rewards[index]
            + gamma * discounts[index] * next_values[index]
            - values[index]
        )
        # Time limits bootstrap, but never propagate GAE into the reset episode.
        following = delta + gamma * gae_lambda * (1 - boundaries[index]) * following
        advantages[index] = following
    return advantages, advantages + values


def encode_observation(observation):
    array = np.asarray(observation)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB Crafter observations")
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def model_input(observations, device):
    return observations.to(device=device, dtype=torch.float32) / 255.0


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_rewards: list
    episode_lengths: list
    achievements: dict


class RolloutCollector:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.frames = deque(maxlen=config.seq_length)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.obtained = set()
        self.reset()

    def reset(self):
        observation = encode_observation(self.env.reset())
        self.frames.clear()
        self.frames.extend(
            torch.zeros_like(observation) for _ in range(self.config.seq_length - 1)
        )
        self.frames.append(observation)
        self.obtained.clear()
        self.episode_reward = 0.0
        self.episode_length = 0

    @torch.no_grad()
    def collect(self, model, count, device, auxiliary=None):
        model.eval()
        observations, actions, rewards, values, discounts, boundaries = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        boundary_values = {}
        episode_rewards, episode_lengths = [], []
        achievements = dict.fromkeys(achievement_list, 0)
        for index in range(count):
            state = torch.stack(tuple(self.frames))
            logits, value, *_ = model(model_input(state.unsqueeze(0), device))
            action = torch.distributions.Categorical(logits=logits).sample().item()
            next_observation, reward, done, info = self.env.step(action)
            continuation = float(info["discount"])
            if continuation not in (0.0, 1.0):
                raise ValueError("Crafter continuation discount must be 0 or 1")
            self.frames.append(encode_observation(next_observation))
            observations.append(state)
            actions.append(action)
            rewards.append(float(reward))
            values.append(value.item())
            discounts.append(continuation)
            boundaries.append(float(done))
            unlocked = {
                name
                for name, amount in info["achievements"].items()
                if amount > 0 and name not in self.obtained
            }
            self.obtained.update(unlocked)
            labels = torch.tensor([name in unlocked for name in achievement_list])
            for name in unlocked:
                achievements[name] += 1
            if auxiliary is not None and unlocked:
                auxiliary.observe(state, labels)
            self.episode_reward += float(reward)
            self.episode_length += 1
            if done:
                boundary_values[index] = (
                    self.bootstrap(model, device) if continuation else 0.0
                )
                episode_rewards.append(self.episode_reward)
                episode_lengths.append(self.episode_length)
                self.reset()
        final_value = 0.0 if boundaries[-1] else self.bootstrap(model, device)
        next_values = values[1:] + [final_value]
        for index, value in boundary_values.items():
            next_values[index] = value
        rewards, values, next_values, discounts, boundaries = (
            torch.tensor(items, dtype=torch.float32)
            for items in (rewards, values, next_values, discounts, boundaries)
        )
        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            discounts,
            boundaries,
            self.config.gamma,
            self.config.gae_lambda,
        )
        return Rollout(
            torch.stack(observations),
            torch.tensor(actions),
            rewards,
            values,
            advantages,
            returns,
            episode_rewards,
            episode_lengths,
            achievements,
        )

    def bootstrap(self, model, device):
        state = torch.stack(tuple(self.frames)).unsqueeze(0)
        _, value, *_ = model(model_input(state, device))
        return value.item()


def update_actor_critic(model, optimizer, rollout, config, coefficient, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = len(rollout.actions)
    raw_advantages = rollout.advantages.detach()
    # A constant/singleton advantage must not be erased by centering.
    std = raw_advantages.std(unbiased=False)
    advantages = (
        (raw_advantages - raw_advantages.mean()) / std.clamp_min(1e-8)
        if count > 1 and std.item() > 1e-8
        else raw_advantages
    )
    metrics = dict.fromkeys(("loss", "actor_loss", "critic_loss", "entropy"), 0.0)
    for start in range(0, count, config.batch_size):
        stop = min(start + config.batch_size, count)
        logits, values, *_ = model(
            model_input(rollout.observations[start:stop], device)
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actor_loss = -(
            distribution.log_prob(rollout.actions[start:stop].to(device))
            * advantages[start:stop].to(device)
        ).mean()
        critic_loss = (
            (values.squeeze(-1) - rollout.returns[start:stop].detach().to(device))
            .square()
            .mean()
        )
        entropy = distribution.entropy().mean()
        loss = actor_loss + config.critic_weight * critic_loss - coefficient * entropy
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite on-policy loss")
        weight = (stop - start) / count
        (loss * weight).backward()
        for key, value in zip(metrics, (loss, actor_loss, critic_loss, entropy)):
            metrics[key] += value.item() * weight
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
        error_if_nonfinite=True,
    )
    # Exactly one parameter update: every microbatch used the collection policy.
    optimizer.step()
    variance = rollout.returns.var(unbiased=False).item()
    metrics.update(
        advantages_abs_mean=raw_advantages.abs().mean().item(),
        value_abs_mean=rollout.values.abs().mean().item(),
        reward_abs_mean=rollout.rewards.abs().mean().item(),
        entropy_coefficient=coefficient,
        entropy_fraction=metrics["entropy"] / math.log(model.num_actions),
        grad_norm=grad_norm.item(),
        explained_variance=(
            1 - (rollout.returns - rollout.values).var(unbiased=False).item() / variance
            if variance > 1e-8
            else None
        ),
    )
    return metrics


def build_model(config, num_actions):
    encoder = LatentStateEncoder(
        config.height,
        config.width,
        3,
        num_layers=1,
        hidden_size_lstm=config.hidden_size,
        c1=16,
        c2=32,
        c3=64,
        c4=64,
        hidden_size_cnn=config.hidden_size,
    )
    return ActorCritic(
        encoder, 1, config.hidden_size, num_actions, head_sizes=(256, 128)
    )


def output_path(name):
    return (
        Path(
            os.getenv(
                "OUTPUT_DIR",
                str(Path(__file__).resolve().parents[2] / "logs" / name),
            )
        )
        .expanduser()
        .resolve()
    )


def save_run(
    output_dir, model, optimizer, auxiliary, config, iteration, environment_steps
):
    checkpoint = {
        "model": model.state_dict(),
        "actor_critic_optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
        "environment_steps": environment_steps,
        "variant": "contrastive" if auxiliary is not None else "baseline",
    }
    if auxiliary is not None:
        checkpoint["contrastive"] = auxiliary.state_dict()
    temporary = output_dir / "checkpoint.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / "checkpoint.pt")
    save_plots(
        output_dir,
        extra_labels={
            "advantages_abs_mean": "Mean absolute GAE advantage",
            "entropy_coefficient": "Entropy coefficient",
            "entropy_fraction": "Entropy / log(action count)",
            "explained_variance": "Critic explained variance (before update)",
            "contrastive_policy_kl": "Accepted contrastive policy KL",
            "contrastive_proposed_kl": "Proposed contrastive policy KL",
            "contrastive_value_shift": "Accepted contrastive value absolute shift",
        },
        x_key="environment_steps",
        x_label="Environment steps",
    )


def run(config, output_dir, auxiliary_factory=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Use an empty OUTPUT_DIR to avoid mixing runs: {output_dir}"
        )
    set_seed(config.seed)
    device = (
        torch.device(os.environ["DEVICE"]) if "DEVICE" in os.environ else get_device()
    )
    env = crafter.Env(seed=config.seed, size=(config.width, config.height))
    try:
        env = crafter.Recorder(
            env,
            str(output_dir / "episodes"),
            save_stats=True,
            save_video=False,
            save_episode=False,
        )
        model = build_model(config, env.action_space.n).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        auxiliary = auxiliary_factory(model) if auxiliary_factory is not None else None
        configuration = {
            "variant": "contrastive" if auxiliary is not None else "baseline",
            "training": asdict(config),
            "contrastive": asdict(auxiliary.config) if auxiliary is not None else None,
            "device": str(device),
            "num_actions": env.action_space.n,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
        (output_dir / "config.json").write_text(
            json.dumps(configuration, indent=2) + "\n"
        )
        logger.info("Starting run: %s", configuration)
        collector = RolloutCollector(env, config)
        recent_rewards, recent_lengths = deque(maxlen=100), deque(maxlen=100)
        environment_steps = iteration = episodes = 0
        with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
            while environment_steps < config.total_steps:
                started = monotonic()
                count = min(
                    config.rollout_steps, config.total_steps - environment_steps
                )
                rollout = collector.collect(model, count, device, auxiliary)
                coefficient = entropy_coefficient(config, environment_steps)
                metrics = update_actor_critic(
                    model, optimizer, rollout, config, coefficient, device
                )
                environment_steps += count
                iteration += 1
                episodes += len(rollout.episode_rewards)
                recent_rewards.extend(rollout.episode_rewards)
                recent_lengths.extend(rollout.episode_lengths)
                if auxiliary is not None:
                    metrics.update(
                        auxiliary.update(
                            model,
                            rollout.observations,
                            environment_steps,
                            iteration,
                            device,
                        )
                    )
                metrics.update(
                    iteration=iteration,
                    environment_steps=environment_steps,
                    episodes=episodes,
                    completed_episodes=len(rollout.episode_rewards),
                    average_reward=(
                        sum(recent_rewards) / len(recent_rewards)
                        if recent_rewards
                        else None
                    ),
                    average_steps=(
                        sum(recent_lengths) / len(recent_lengths)
                        if recent_lengths
                        else None
                    ),
                    achievements=rollout.achievements,
                    seconds=monotonic() - started,
                )
                record = json.dumps(metrics, allow_nan=False)
                metrics_file.write(record + "\n")
                metrics_file.flush()
                logger.info("%s", record)
                del rollout
                if (
                    iteration % config.checkpoint_every == 0
                    or environment_steps == config.total_steps
                ):
                    save_run(
                        output_dir,
                        model,
                        optimizer,
                        auxiliary,
                        config,
                        iteration,
                        environment_steps,
                    )
    finally:
        env.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(config_from_env(Config), output_path("actor-critic-on-policy"))


if __name__ == "__main__":
    main()
