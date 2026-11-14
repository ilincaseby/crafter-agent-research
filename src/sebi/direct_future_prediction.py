"""Pixel-only DFP: predict future measurements, act on original reward only."""

import json
import logging
import math
import os
import random
from collections import deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import monotonic

import crafter
from crafter import constants
import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .actor_critic_on_policy import encode_observation, model_input, output_path
    from .contrastive_actor_critic import (
        achievement_list,
        get_device,
        save_plots,
        set_seed,
    )
else:
    from actor_critic_on_policy import encode_observation, model_input, output_path
    from contrastive_actor_critic import (
        achievement_list,
        get_device,
        save_plots,
        set_seed,
    )

logger = logging.getLogger(__name__)
INVENTORY_NAMES = tuple(constants.items)
MEASUREMENT_NAMES = (
    "episode_reward",
    *INVENTORY_NAMES,
    *(f"achievement_{name}" for name in achievement_list),
)


@dataclass(frozen=True)
class DFPConfig:
    total_steps: int = 1_000_000
    frame_stack: int = 4
    horizons: tuple[int, ...] = (4, 16, 64, 128)
    height: int = 64
    width: int = 64
    replay_capacity: int = 4096
    batch_size: int = 32
    learning_starts: int = 1024
    train_every: int = 4
    learning_rate: float = 3e-4
    auxiliary_loss_weight: float = 0.1
    reward_scale: float = 10.0
    max_grad_norm: float = 0.5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    log_every: int = 1000
    checkpoint_every: int = 10_000
    seed: int = 42

    def __post_init__(self):
        for name in (
            "total_steps",
            "frame_stack",
            "replay_capacity",
            "batch_size",
            "train_every",
            "epsilon_decay_steps",
            "log_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not self.horizons
            or any(not isinstance(value, int) or value < 1 for value in self.horizons)
            or tuple(sorted(set(self.horizons))) != tuple(self.horizons)
        ):
            raise ValueError("horizons must be strictly increasing positive integers")
        if min(self.height, self.width) < 32:
            raise ValueError(
                "height and width must be at least 32 for Crafter rendering"
            )
        if self.replay_capacity < self.batch_size or self.learning_starts < 0:
            raise ValueError(
                "Replay must fit a batch; learning_starts must be nonnegative"
            )
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Epsilon must satisfy 0 <= end <= start <= 1")
        if min(self.learning_rate, self.reward_scale, self.max_grad_norm) <= 0:
            raise ValueError(
                "Learning rate, reward scale and gradient limit must be positive"
            )
        if self.auxiliary_loss_weight < 0:
            raise ValueError("auxiliary_loss_weight must be nonnegative")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field in fields(defaults):
            default = getattr(defaults, field.name)
            raw = os.getenv(field.name.upper())
            if raw is None:
                values[field.name] = default
            elif field.name == "horizons":
                values[field.name] = tuple(
                    int(value.strip()) for value in raw.split(",")
                )
            else:
                values[field.name] = type(default)(raw)
        return cls(**values)


def measurements(info, episode_reward, reward_scale):
    values = [episode_reward / reward_scale]
    values.extend(
        info["inventory"][name] / constants.items[name]["max"]
        for name in INVENTORY_NAMES
    )
    values.extend(float(info["achievements"][name] > 0) for name in achievement_list)
    result = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(result).all().item():
        raise ValueError("Non-finite Crafter measurements")
    return result


def initial_measurements(reward_scale):
    # Crafter 1.8.3 resets Player inventory to these public constants.
    return measurements(
        {
            "inventory": {
                name: constants.items[name]["initial"] for name in INVENTORY_NAMES
            },
            "achievements": dict.fromkeys(achievement_list, 0),
        },
        0.0,
        reward_scale,
    )


class SpatialMeanPool4x4(nn.Module):
    """Adaptive 4x4 means without the nondeterministic CUDA pooling backward."""

    def forward(self, features):
        batch, channels, height, width = features.shape
        if height % 4 == 0 and width % 4 == 0:
            return features.reshape(
                batch, channels, 4, height // 4, 4, width // 4
            ).mean(dim=(3, 5))
        # Match adaptive pooling's overlapping floor/ceil bins at uneven sizes.
        cells = []
        for row in range(4):
            top, bottom = row * height // 4, ((row + 1) * height + 3) // 4
            for column in range(4):
                left, right = column * width // 4, ((column + 1) * width + 3) // 4
                cells.append(features[:, :, top:bottom, left:right].mean(dim=(-2, -1)))
        return torch.stack(cells, dim=-1).reshape(batch, channels, 4, 4)


class FuturePredictor(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_horizons = len(config.horizons)
        self.num_measurements = len(MEASUREMENT_NAMES)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * config.frame_stack, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            SpatialMeanPool4x4(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
        )
        self.predictions = nn.Linear(
            128,
            num_actions * self.num_horizons * self.num_measurements,
        )

    def forward(self, observations):
        batch, frames, channels, height, width = observations.shape
        features = self.encoder(
            observations.reshape(batch, frames * channels, height, width)
        )
        return self.predictions(features).reshape(
            batch,
            self.num_actions,
            self.num_horizons,
            self.num_measurements,
        )


def epsilon_at(config, environment_steps):
    if environment_steps < config.learning_starts:
        return 1.0
    progress = min(max(environment_steps / config.epsilon_decay_steps, 0.0), 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def action_scores(predictions):
    # Only original cumulative environment reward, at the longest horizon.
    return predictions[:, :, -1, 0]


@torch.no_grad()
def choose_action(model, observation, epsilon, rng, device):
    if rng.random() < epsilon:
        return rng.randrange(model.num_actions)
    model.eval()
    predictions = model(model_input(observation.unsqueeze(0), device))
    scores = action_scores(predictions)[0]
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("Non-finite DFP action scores")
    best = torch.where(scores == scores.max())[0].tolist()
    return rng.choice(best)


@dataclass
class FutureExample:
    observation: torch.Tensor
    action: int
    targets: torch.Tensor
    valid: torch.Tensor


class FutureReplay:
    def __init__(self, capacity, horizons, seed):
        self.capacity = capacity
        self.horizons = tuple(horizons)
        self.entries = []
        self.position = 0
        self.pending = deque()
        self.history = deque()
        self.rng = random.Random(seed)
        self.discarded = 0

    def __len__(self):
        return len(self.entries)

    def start_episode(self, initial):
        if self.pending:
            raise RuntimeError(
                "Finish pending future targets before resetting the episode"
            )
        self.history.clear()
        self.history.append(initial.detach().cpu().clone())

    def append(
        self,
        observation,
        action,
        next_measurements,
        episode_end=False,
        terminated=False,
    ):
        if not self.history:
            raise RuntimeError("Start an episode before adding future targets")
        if terminated and not episode_end:
            raise ValueError("A true termination must also end the episode")
        self.pending.append((observation.detach().cpu().clone(), action))
        self.history.append(next_measurements.detach().cpu().clone())
        if len(self.pending) >= self.horizons[-1]:
            self._emit_oldest(terminated=False)
        if episode_end:
            self.finish(terminated)

    def _emit_oldest(self, terminated):
        observation, action = self.pending.popleft()
        history = list(self.history)
        available = len(history) - 1
        targets, valid = [], []
        for horizon in self.horizons:
            if horizon <= available:
                targets.append(history[horizon] - history[0])
                valid.append(True)
            elif terminated:
                # Death is absorbing: no more rewards or measurements change.
                targets.append(history[-1] - history[0])
                valid.append(True)
            else:
                targets.append(torch.zeros_like(history[0]))
                valid.append(False)
        self.history.popleft()
        if not any(valid):
            self.discarded += 1
            return
        example = FutureExample(
            observation, action, torch.stack(targets), torch.tensor(valid)
        )
        if len(self.entries) < self.capacity:
            self.entries.append(example)
        else:
            self.entries[self.position] = example
        self.position = (self.position + 1) % self.capacity

    def finish(self, terminated=False):
        while self.pending:
            self._emit_oldest(terminated)
        self.history.clear()

    def sample(self, batch_size):
        if len(self.entries) < batch_size:
            raise ValueError("Not enough mature future targets for this batch")
        return self.rng.sample(self.entries, batch_size)


def prediction_loss(predictions, actions, targets, valid, auxiliary_weight):
    if not valid.any().item():
        raise ValueError("Future-prediction batch contains no observed horizons")
    selected = predictions[torch.arange(len(actions), device=actions.device), actions]
    errors = F.smooth_l1_loss(selected, targets, reduction="none")
    reward_loss = errors[:, :, 0][valid].mean()
    auxiliary_loss = errors[:, :, 1:][valid].mean()
    return (
        reward_loss + auxiliary_weight * auxiliary_loss,
        reward_loss,
        auxiliary_loss,
        selected,
    )


def train_batch(model, optimizer, examples, config, device):
    observations = torch.stack([example.observation for example in examples])
    actions = torch.tensor([example.action for example in examples], device=device)
    targets = torch.stack([example.targets for example in examples]).to(device)
    valid = torch.stack([example.valid for example in examples]).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(model_input(observations, device))
    loss, reward_loss, auxiliary_loss, selected = prediction_loss(
        predictions,
        actions,
        targets,
        valid,
        config.auxiliary_loss_weight,
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite DFP prediction loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.max_grad_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    with torch.no_grad():
        reward_mae = (selected[:, :, 0] - targets[:, :, 0]).abs()[valid].mean()
    return {
        "loss": loss.item(),
        "reward_prediction_loss": reward_loss.item(),
        "auxiliary_prediction_loss": auxiliary_loss.item(),
        "reward_prediction_mae": reward_mae.item() * config.reward_scale,
        "grad_norm": grad_norm.item(),
    }


def reset_episode(env, config, replay):
    observation = encode_observation(env.reset())
    frames = deque(maxlen=config.frame_stack)
    frames.extend(torch.zeros_like(observation) for _ in range(config.frame_stack - 1))
    frames.append(observation)
    replay.start_episode(initial_measurements(config.reward_scale))
    return frames


def save_run(output_dir, model, optimizer, config, environment_steps, updates):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "environment_steps": environment_steps,
        "updates": updates,
        "num_actions": model.num_actions,
        "measurement_names": MEASUREMENT_NAMES,
        "variant": "dfp_pixels_original_reward",
    }
    temporary = output_dir / "checkpoint.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output_dir / "checkpoint.pt")
    save_plots(
        output_dir,
        extra_labels={
            "loss": "DFP supervised prediction loss",
            "reward_prediction_loss": "Normalized reward prediction Huber loss",
            "auxiliary_prediction_loss": "Auxiliary measurement Huber loss",
            "reward_prediction_mae": "Reward prediction MAE (environment reward units)",
            "epsilon": "Random action probability",
            "steps_per_second": "Cumulative environment steps per second",
            "replay_size": "Mature future-prediction examples",
        },
        x_key="environment_steps",
        x_label="Environment steps",
    )


def run(config, output_dir):
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
        model = FuturePredictor(config, env.action_space.n).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        replay = FutureReplay(config.replay_capacity, config.horizons, config.seed + 1)
        action_rng = random.Random(config.seed + 2)
        metadata = {
            "variant": "dfp_pixels_original_reward",
            "training": asdict(config),
            "decision_input": "RGB frames only; no numeric measurements",
            "action_objective": "Predicted original reward sum at the longest horizon",
            "measurement_names": MEASUREMENT_NAMES,
            "measurement_scales": [
                config.reward_scale,
                *(constants.items[name]["max"] for name in INVENTORY_NAMES),
                *([1.0] * len(achievement_list)),
            ],
            "num_actions": model.num_actions,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
        }
        (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info("Starting DFP: %s", metadata)
        frames = reset_episode(env, config, replay)
        episode_reward = episode_length = episodes = updates = window_updates = 0
        recent_rewards, recent_lengths = deque(maxlen=100), deque(maxlen=100)
        obtained = set()
        achievements = dict.fromkeys(achievement_list, 0)
        metric_sums = {}
        started = monotonic()
        with (output_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
            for step in range(1, config.total_steps + 1):
                state = torch.stack(tuple(frames))
                epsilon = epsilon_at(config, step - 1)
                action = choose_action(model, state, epsilon, action_rng, device)
                observation, reward, done, info = env.step(action)
                continuation = float(info["discount"])
                if continuation not in (0.0, 1.0) or (not done and continuation == 0):
                    raise ValueError("Invalid Crafter termination/discount combination")
                episode_reward += float(reward)
                episode_length += 1
                replay.append(
                    state,
                    action,
                    measurements(info, episode_reward, config.reward_scale),
                    episode_end=bool(done),
                    terminated=bool(done and continuation == 0),
                )
                frames.append(encode_observation(observation))
                for name, amount in info["achievements"].items():
                    if amount > 0 and name not in obtained:
                        achievements[name] += 1
                        obtained.add(name)
                if done:
                    episodes += 1
                    recent_rewards.append(episode_reward)
                    recent_lengths.append(episode_length)
                    if step < config.total_steps:
                        frames = reset_episode(env, config, replay)
                        episode_reward = episode_length = 0
                        obtained.clear()
                elif step == config.total_steps:
                    # Ending the budget is censoring, not a fictional death.
                    replay.finish(terminated=False)
                if (
                    step >= config.learning_starts
                    and step % config.train_every == 0
                    and len(replay) >= config.batch_size
                ):
                    metrics = train_batch(
                        model,
                        optimizer,
                        replay.sample(config.batch_size),
                        config,
                        device,
                    )
                    for name, value in metrics.items():
                        metric_sums[name] = metric_sums.get(name, 0.0) + value
                    updates += 1
                    window_updates += 1
                checkpoint_due = (
                    step % config.checkpoint_every == 0 or step == config.total_steps
                )
                if step % config.log_every == 0 or checkpoint_due:
                    record = {
                        name: total / window_updates
                        for name, total in metric_sums.items()
                    }
                    record.update(
                        environment_steps=step,
                        updates=updates,
                        window_updates=window_updates,
                        loss=record.get("loss"),
                        epsilon=epsilon,
                        episodes=episodes,
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
                        replay_size=len(replay),
                        pending_targets=len(replay.pending),
                        discarded_without_horizon=replay.discarded,
                        achievements=achievements,
                        seconds=monotonic() - started,
                        steps_per_second=step / (monotonic() - started),
                    )
                    line = json.dumps(record, allow_nan=False)
                    metrics_file.write(line + "\n")
                    metrics_file.flush()
                    logger.info("%s", line)
                    metric_sums.clear()
                    window_updates = 0
                    achievements = dict.fromkeys(achievement_list, 0)
                if checkpoint_due:
                    save_run(output_dir, model, optimizer, config, step, updates)
        if not updates:
            logger.warning(
                "Run finished without gradient updates: check budget, learning_starts, batch size and horizons"
            )
        if replay.discarded:
            logger.info(
                "Discarded %s censored transitions without any observed horizon",
                replay.discarded,
            )
    finally:
        env.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(DFPConfig.from_env(), output_path("direct-future-prediction"))


if __name__ == "__main__":
    main()
