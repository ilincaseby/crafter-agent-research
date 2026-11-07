import crafter
import gym
import torch
import random
import torch.nn.functional as F
import numpy as np
import os
import sys
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from time import monotonic
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

if __package__:
    from .models import ActorCritic, LatentStateEncoder
else:
    from models import ActorCritic, LatentStateEncoder

logger = logging.getLogger(__name__)

# CNN arguments
height = int(os.getenv("HEIGHT", 64))
width = int(os.getenv("WIDTH", 64))
channels_in = int(os.getenv("CHANNELS_IN", 3))
channels_hidden_1 = int(os.getenv("CHANNELS_HIDDEN_1", 32))
channels_hidden_2 = int(os.getenv("CHANNELS_HIDDEN_2", 64))
channels_hidden_3 = int(os.getenv("CHANNELS_HIDDEN_3", 128))
channels_hidden_4 = int(os.getenv("CHANNELS_HIDDEN_4", 256))
hidden_size_cnn = int(os.getenv("HIDDEN_SIZE_CNN", 1024))
kernel_size = int(os.getenv("KERNEL_SIZE", 4))
stride = int(os.getenv("STRIDE", 2))
padding = int(os.getenv("PADDING", 1))
dilation = int(os.getenv("DILATION", 1))
groups = int(os.getenv("GROUPS", 1))

# LSTM arguments
num_layers_lstm = int(os.getenv("NUM_LAYERS_LSTM", 2))
hidden_size_lstm = int(os.getenv("HIDDEN_SIZE_LSTM", 1024))

# ActorCritic architecture arguments(and training)
hidden_size_input = int(os.getenv("HIDDEN_SIZE_INPUT", hidden_size_lstm))
num_actions = int(os.getenv("NUM_ACTIONS", 17))
actor_critic_lr = float(os.getenv("ACTOR_CRITIC_LR", 0.0003))
# "adam" or "rmsprop"; RMSprop's zero-initialized square average makes its
# first updates ~10x the learning rate, which can collapse the policy at once.
actor_critic_optimizer = os.getenv("ACTOR_CRITIC_OPTIMIZER", "adam").strip().lower()
# RMSprop-only settings
centered = os.getenv("CENTERED", "false").strip().lower() in ("true", "1", "yes", "on")
alpha = float(os.getenv("ALPHA", 0.99))
eps_optim = float(os.getenv("EPS_OPTIM", 1e-8))
game_batch = int(os.getenv("GAME_BATCH", 64))
actor_loss_weight = float(os.getenv("ACTOR_LOSS_WEIGHT", 1.0))
critic_loss_weight = float(os.getenv("CRITIC_LOSS_WEIGHT", 0.5))
entropy_loss_weight = float(os.getenv("ENTROPY_LOSS_WEIGHT", 0.1))

# discount for greediness
discount_greediness = float(os.getenv("DISCOUNT_GREEDINESS", 0.9978))

# after how many steps should update
# steps_updating = 2000
steps_updating = int(os.getenv("STEPS_UPDATING", 1))

# completed episodes collected every iteration
num_exp = int(os.getenv("NUM_EXP", 16))

# parallel Crafter environments; VECTOR_ENV is "async" (subprocesses) or "sync"
num_envs = int(os.getenv("NUM_ENVS", num_exp))
vector_env = os.getenv("VECTOR_ENV", "async").strip().lower()
# multiprocessing start method for AsyncVectorEnv; empty uses the platform default
vector_env_context = os.getenv("VECTOR_ENV_CONTEXT", "") or None

# total environment timesteps (Crafter's standard 1M-step budget)
total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", 1_000_000))

# contrastive loss arguments
# environment timesteps collected before contrastive updates may start
warm_up_steps = int(os.getenv("WARM_UP_STEPS", 100_000))
prob_contrastive = float(os.getenv("PROB_CONTRASTIVE", 0.2))
contrastive_loss_batch_size = int(os.getenv("CONTRASTIVE_LOSS_BATCH_SIZE", 64))
contrastive_loss_temperature = float(os.getenv("CONTRASTIVE_LOSS_TEMPERATURE", 1.1))
lr_contrastive_learning = float(os.getenv("LR_CONTRASTIVE_LEARNING", 1e-4))
weight_decay = float(os.getenv("WEIGHT_DECAY", 1e-2))
contrastive_sim_weight = float(os.getenv("CONTRASTIVE_SIM_WEIGHT", 1.0))
contrastive_near_loss_weight = float(os.getenv("CONTRASTIVE_NEAR_LOSS_WEIGHT", 0.2))

# crafter
seq_length = int(os.getenv("SEQ_LENGTH", 20))

# RL
discount = float(os.getenv("DISCOUNT", 0.9985))

achievement_list = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]


def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Seed offset of each worker's standby Crafter instance.
STANDBY_SEED_OFFSET = 10_000
# World generation is pure Python; with the default 5 ms GIL switch interval
# it delays the worker's steps, and the slowest worker gates every vector step.
WORKER_SWITCH_INTERVAL = 2e-5


class CrafterGymEnv(gym.Env):
    """Adapts Crafter's legacy reset/step API to the one gym.vector expects.

    Crafter generates a new world on every reset, which takes about a second,
    while a step takes about a millisecond. With a standby instance, the next
    world is generated in a background thread while the current episode runs,
    so automatic resets do not stall every environment in the vector.
    Episodes alternate deterministically between the two instances.
    """

    metadata = {"render_modes": []}

    def __init__(self, env, standby=None):
        self.env = env
        self.standby = standby
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._executor = None
        self._pending = None
        if standby is not None:
            sys.setswitchinterval(WORKER_SWITCH_INTERVAL)
            self._executor = ThreadPoolExecutor(1)

    def reset(self, *, seed=None, options=None):
        # Crafter is seeded at construction and reseeds every episode itself.
        if self._pending is None:
            observation = self.env.reset()
        else:
            self.env, self.standby = self.standby, self.env
            observation = self._pending.result()
        if self._executor is not None:
            self._pending = self._executor.submit(self.standby.reset)
        return observation, {}

    def step(self, action):
        observation, reward, done, info = self.env.step(int(action))
        terminated = bool(done) and info.get("discount", 0.0) == 0
        return observation, reward, terminated, bool(done) and not terminated, info

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        for env in (self.env, self.standby):
            close = getattr(env, "close", None)
            if close is not None:
                close()


def make_env(seed):
    return CrafterGymEnv(
        crafter.Env(seed=seed, size=(width, height)),
        crafter.Env(seed=seed + STANDBY_SEED_OFFSET, size=(width, height)),
    )


def create_vector_env(seed):
    env_fns = [partial(make_env, seed + index) for index in range(num_envs)]
    if vector_env == "sync":
        return gym.vector.SyncVectorEnv(env_fns)
    return gym.vector.AsyncVectorEnv(env_fns, context=vector_env_context)


def get_achievement_dict():
    achievement_dict = {
        ach: index for ach, index in zip(achievement_list, range(len(achievement_list)))
    }
    return achievement_dict


def as_model_input(observations):
    if observations.dtype == torch.uint8:
        return observations.float().div_(255.0)
    return observations


class ExperienceCollector:
    """Plays a vector of Crafter environments and returns completed episodes.

    Unfinished episodes are carried over to the next collection, so no
    environment steps are wasted and long episodes are not under-sampled.
    """

    def __init__(self, env, achievement_dict, device):
        self.env = env
        self.num_envs = env.num_envs
        self.achievement_dict = achievement_dict
        self.device = device
        self.total_steps = 0
        # Per-episode training statistics, drained by the training loop.
        self.finished = []
        self.windows = torch.zeros(
            self.num_envs,
            seq_length,
            channels_in,
            height,
            width,
            dtype=torch.uint8,
            device=device,
        )
        self.episodes = [None] * self.num_envs
        observations, _ = env.reset()
        for index in range(self.num_envs):
            self._start_episode(index, observations[index])

    def _frames_to_tensor(self, frames):
        frames = torch.as_tensor(np.asarray(frames), device=self.device)
        return frames.permute(0, 3, 1, 2)

    def _start_episode(self, index, observation):
        self.windows[index].zero_()
        self.windows[index, -1] = self._frames_to_tensor(observation[None])[0]
        self.episodes[index] = {
            "frames": [np.array(observation, dtype=np.uint8, copy=True)],
            "actions": [],
            "rewards": [],
            "achievements": [],
            "latents": [],
            "unlocked": set(),
        }

    def _new_achievements(self, episode, achievements):
        unlocked = np.zeros(len(self.achievement_dict), dtype=np.float32)
        for name, count in achievements.items():
            if count > 0 and name not in episode["unlocked"]:
                unlocked[self.achievement_dict[name]] = 1.0
                episode["unlocked"].add(name)
        return unlocked

    def _finish_episode(self, episode, final_latent):
        device = self.device
        frames = self._frames_to_tensor(episode["frames"])
        padding = torch.zeros(
            seq_length - 1, *frames.shape[1:], dtype=torch.uint8, device=device
        )
        # Zero-copy sliding windows: windows[t] holds frames t-seq_length+1..t.
        windows = (
            torch.cat([padding, frames]).unfold(0, seq_length, 1).permute(0, 4, 1, 2, 3)
        )
        steps = len(episode["actions"])
        latent_obs = torch.stack(episode["latents"])
        done = torch.zeros(steps, device=device)
        done[-1] = 1.0
        return (
            windows[:-1],
            windows[1:],
            torch.tensor(episode["actions"], dtype=torch.long, device=device),
            torch.tensor(episode["rewards"], dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(episode["achievements"]), device=device),
            latent_obs,
            torch.cat([latent_obs[1:], final_latent[None]]),
            done,
        )

    @torch.no_grad()
    def collect(self, model, num_episodes, max_steps):
        completed = []
        steps = 0
        while len(completed) < num_episodes and steps < max_steps:
            logits, _, latents, *_ = model(as_model_input(self.windows))
            actions = torch.distributions.Categorical(logits=logits).sample()
            actions_np = actions.cpu().numpy()
            observations, rewards, terminated, truncated, infos = self.env.step(
                actions_np
            )
            steps += self.num_envs
            dones = np.logical_or(terminated, truncated)
            next_frames = np.array(observations, dtype=np.uint8, copy=True)
            done_indices = np.flatnonzero(dones)
            for index in done_indices:
                next_frames[index] = infos["final_observation"][index]
            self.windows = torch.cat(
                [self.windows[:, 1:], self._frames_to_tensor(next_frames)[:, None]],
                dim=1,
            )
            final_latents = {}
            if done_indices.size:
                selected = torch.as_tensor(done_indices, device=self.device)
                _, _, done_latents, *_ = model(as_model_input(self.windows[selected]))
                final_latents = dict(zip(done_indices.tolist(), done_latents))
            for index in range(self.num_envs):
                episode = self.episodes[index]
                if dones[index]:
                    achievements = infos["final_info"][index]["achievements"]
                else:
                    achievements = infos["achievements"][index]
                episode["actions"].append(int(actions_np[index]))
                episode["rewards"].append(float(rewards[index]))
                episode["achievements"].append(
                    self._new_achievements(episode, achievements)
                )
                episode["latents"].append(latents[index].clone())
                episode["frames"].append(next_frames[index].copy())
                if dones[index]:
                    completed.append(
                        self._finish_episode(episode, final_latents[index])
                    )
                    self.finished.append(
                        {
                            "global_step": self.total_steps + steps,
                            "episodic_return": float(sum(episode["rewards"])),
                            "episodic_length": len(episode["actions"]),
                            "achievements": json.dumps(
                                {
                                    name: int(count)
                                    for name, count in achievements.items()
                                }
                            ),
                        }
                    )
                    self._start_episode(index, observations[index])
        self.total_steps += steps
        return completed, steps


def config_snapshot():
    """Hyperparameters of this run, stored with checkpoints and reports."""
    return {
        "height": height,
        "width": width,
        "channels_in": channels_in,
        "channels_hidden": [
            channels_hidden_1,
            channels_hidden_2,
            channels_hidden_3,
            channels_hidden_4,
        ],
        "hidden_size_cnn": hidden_size_cnn,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "num_layers_lstm": num_layers_lstm,
        "hidden_size_lstm": hidden_size_lstm,
        "num_actions": num_actions,
        "seq_length": seq_length,
        "total_timesteps": total_timesteps,
        "num_exp": num_exp,
        "num_envs": num_envs,
        "vector_env": vector_env,
        "actor_critic_optimizer": actor_critic_optimizer,
        "actor_critic_lr": actor_critic_lr,
        "centered": centered,
        "alpha": alpha,
        "eps_optim": eps_optim,
        "game_batch": game_batch,
        "actor_loss_weight": actor_loss_weight,
        "critic_loss_weight": critic_loss_weight,
        "entropy_loss_weight": entropy_loss_weight,
        "discount": discount,
        "warm_up_steps": warm_up_steps,
        "prob_contrastive": prob_contrastive,
        "contrastive_loss_batch_size": contrastive_loss_batch_size,
        "contrastive_loss_temperature": contrastive_loss_temperature,
        "lr_contrastive_learning": lr_contrastive_learning,
        "weight_decay": weight_decay,
        "contrastive_sim_weight": contrastive_sim_weight,
        "contrastive_near_loss_weight": contrastive_near_loss_weight,
    }


def build_model(config, device):
    c1, c2, c3, c4 = config["channels_hidden"]
    lstm_model = LatentStateEncoder(
        config["height"],
        config["width"],
        config["channels_in"],
        config["num_layers_lstm"],
        config["hidden_size_lstm"],
        c1,
        c2,
        c3,
        c4,
        config["hidden_size_cnn"],
        config["kernel_size"],
        config["stride"],
        config["padding"],
        config["dilation"],
        config["groups"],
    ).to(device)
    model = ActorCritic(
        lstm_model,
        config["num_layers_lstm"],
        config["hidden_size_lstm"],
        config["num_actions"],
    ).to(device)
    return model, lstm_model


def get_device():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def create_experiences(model, collector, max_steps):
    episodes, steps = collector.collect(model, num_exp, max_steps)
    buffers = [list(tensors) for tensors in zip(*episodes)] or [[] for _ in range(8)]
    return buffers, steps


def actor_loss(log_probs, advantages):
    return -(log_probs * advantages.detach()).mean()


def critic_loss(advantages):
    return advantages.square().mean()


def contrastive_loss(ach_episodes, temperature):
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    items = [(key, value) for key, value in ach_episodes.items() if value.shape[0] > 0]
    if len(items) < 2:
        return None
    z = F.normalize(torch.cat([value for _, value in items]), dim=1)
    labels = torch.cat(
        [torch.full((value.shape[0],), key, device=z.device) for key, value in items]
    )
    logits = z @ z.T / temperature
    upper = torch.ones_like(logits, dtype=torch.bool).triu(diagonal=1)
    same = labels[:, None] == labels[None, :]
    pos = logits[upper & same]
    neg = logits[upper & ~same]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    return torch.logsumexp(neg, dim=0) - pos.mean()


def near_loss(ach_tensor, old_ach_tensor):
    if not ach_tensor or ach_tensor.keys() != old_ach_tensor.keys():
        raise ValueError("Near loss requires matching, nonempty achievement embeddings")
    keys = sorted(ach_tensor)
    new_emb = torch.cat([ach_tensor[key] for key in keys])
    old_emb = torch.cat([old_ach_tensor[key] for key in keys]).detach()
    return 1 - F.cosine_similarity(new_emb, old_emb, dim=-1).mean()


def train_contrastive(lstm_model, clone_lstm, optimizer, buffers, achievement_dict):
    losses = []
    games = len(buffers[0])
    for start in range(0, games, contrastive_loss_batch_size):
        stop = min(start + contrastive_loss_batch_size, games)
        ach_tensor, old_ach_tensor = {}, {}
        before_update = deepcopy(lstm_model.state_dict())
        optimizer.zero_grad(set_to_none=True)
        for key in achievement_dict.values():
            examples = []
            for game in range(start, stop):
                achievements = buffers[4][game]
                # A transition may unlock multiple achievements.
                indices = torch.where(achievements[:, key] > 0)[0]
                if indices.numel():
                    examples.append(buffers[0][game][indices])
            if not examples:
                continue
            observations = as_model_input(torch.cat(examples))
            output, _ = lstm_model(observations)
            ach_tensor[key] = output[:, -1, :]
            with torch.no_grad():
                old_output, _ = clone_lstm(observations)
                old_ach_tensor[key] = old_output[:, -1, :]
        if not ach_tensor:
            logger.warning(
                "Skipping contrastive games %s:%s: no achievements", start, stop
            )
            continue
        c_loss = contrastive_loss(ach_tensor, contrastive_loss_temperature)
        loss = contrastive_near_loss_weight * near_loss(ach_tensor, old_ach_tensor)
        if c_loss is None:
            logger.warning(
                "Games %s:%s: insufficient contrastive pairs; near loss only",
                start,
                stop,
            )
        else:
            loss = loss + contrastive_sim_weight * c_loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite contrastive loss")
        loss.backward()
        optimizer.step()
        clone_lstm.load_state_dict(before_update)
        losses.append(loss.item())
    return losses


def train_actor_critic(model, optimizer, buffers):
    metrics = {"loss": [], "actor_loss": [], "critic_loss": [], "entropy": []}
    diagnostic_sums = {
        "advantages_abs_mean": 0.0,
        "value_abs_mean": 0.0,
        "reward_abs_mean": 0.0,
    }
    transition_count = 0
    for game in range(len(buffers[0])):
        length = buffers[0][game].shape[0]
        indices = torch.randperm(length, device=buffers[0][game].device)
        for start in range(0, length, game_batch):
            batch = indices[start : start + game_batch]
            optimizer.zero_grad(set_to_none=True)
            logits, value, *_ = model(as_model_input(buffers[0][game][batch]))
            value = value.squeeze(-1)
            with torch.no_grad():
                _, next_value, *_ = model(as_model_input(buffers[1][game][batch]))
                target = buffers[3][game][batch] + discount * (
                    1 - buffers[7][game][batch]
                ) * next_value.squeeze(-1)
            advantages = target - value
            distribution = torch.distributions.Categorical(logits=logits)
            a_loss = actor_loss(
                distribution.log_prob(buffers[2][game][batch]), advantages
            )
            c_loss = critic_loss(advantages)
            entropy = distribution.entropy().mean()
            loss = (
                actor_loss_weight * a_loss
                + critic_loss_weight * c_loss
                - entropy_loss_weight * entropy
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite actor-critic loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            diagnostic_sums["advantages_abs_mean"] += (
                advantages.detach().abs().sum().item()
            )
            diagnostic_sums["value_abs_mean"] += value.detach().abs().sum().item()
            diagnostic_sums["reward_abs_mean"] += (
                buffers[3][game][batch].abs().sum().item()
            )
            transition_count += batch.numel()
            for key, metric_value in zip(metrics, (loss, a_loss, c_loss, entropy)):
                metrics[key].append(metric_value.item())
    result = {key: sum(values) / len(values) for key, values in metrics.items()}
    result.update(
        {key: total / transition_count for key, total in diagnostic_sums.items()}
    )
    return result


def save_plots(output_dir):
    with (output_dir / "metrics.jsonl").open(encoding="utf-8") as metrics_file:
        history = [json.loads(line) for line in metrics_file]
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "loss": "Combined actor-critic loss",
        "actor_loss": "Actor loss",
        "critic_loss": "Critic loss",
        "entropy": "Policy entropy",
        "contrastive_loss": "Contrastive loss",
        "average_reward": "Average episode reward",
        "average_steps": "Average episode length",
        "advantages_abs_mean": "Mean absolute TD advantage",
        "value_abs_mean": "Mean absolute predicted value",
        "reward_abs_mean": "Mean absolute transition reward",
    }
    for key, label in labels.items():
        values = [record.get(key) for record in history]
        if not any(value is not None for value in values):
            continue
        figure = Figure(figsize=(8, 4), tight_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [record["iteration"] for record in history],
            [float("nan") if value is None else value for value in values],
            marker=".",
        )
        axis.set(xlabel="Training iteration", ylabel=label, title=label)
        axis.grid(alpha=0.3)
        figure.savefig(plot_dir / f"{key}.png")
        figure.clear()


def save_checkpoint(
    path, model, clone_lstm, optimizer, contrastive_optimizer, iteration, timesteps
):
    checkpoint = {
        "iteration": iteration,
        "timesteps": timesteps,
        "config": config_snapshot(),
        "model": model.state_dict(),
        "previous_encoder": clone_lstm.state_dict(),
        "actor_critic_optimizer": optimizer.state_dict(),
        "contrastive_optimizer": contrastive_optimizer.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train_model(
    model,
    lstm_model,
    clone_lstm,
    env,
    achievement_dict,
    device,
    output_dir,
    checkpoint_every,
    on_checkpoint=None,
):
    if actor_critic_optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=actor_critic_lr,
            centered=centered,
            alpha=alpha,
            eps=eps_optim,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=actor_critic_lr, eps=eps_optim
        )
    contrastive_optimizer = torch.optim.AdamW(
        lstm_model.parameters(),
        lr=lr_contrastive_learning,
        weight_decay=weight_decay,
        eps=eps_optim,
    )
    collector = ExperienceCollector(env, achievement_dict, device)
    iteration = 0
    episodes_path = output_dir / "episodes.csv"
    new_log = not episodes_path.exists()
    with (
        (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file,
        episodes_path.open("a", newline="", encoding="utf-8") as episodes_file,
    ):
        episode_log = csv.DictWriter(
            episodes_file,
            ["global_step", "episodic_return", "episodic_length", "achievements"],
        )
        if new_log:
            episode_log.writeheader()
        while collector.total_steps < total_timesteps:
            started = monotonic()
            buffers, steps = create_experiences(
                model, collector, total_timesteps - collector.total_steps
            )
            episode_log.writerows(collector.finished)
            episodes_file.flush()
            collector.finished.clear()
            episodes = len(buffers[0])
            if episodes == 0:
                logger.info(
                    "Timestep budget reached with %s steps in unfinished episodes",
                    steps,
                )
                break
            iteration += 1
            metrics = {
                "iteration": iteration,
                "timesteps": collector.total_steps,
                "episodes": episodes,
                "average_reward": sum(rewards.sum().item() for rewards in buffers[3])
                / episodes,
                "average_steps": sum(rewards.numel() for rewards in buffers[3])
                / episodes,
                "achievements": {
                    key: sum(ach[:, index].sum().item() for ach in buffers[4])
                    for key, index in achievement_dict.items()
                },
            }
            if (
                collector.total_steps >= warm_up_steps
                and random.random() < prob_contrastive
            ):
                losses = train_contrastive(
                    lstm_model,
                    clone_lstm,
                    contrastive_optimizer,
                    buffers,
                    achievement_dict,
                )
                metrics["mode"] = "contrastive"
                metrics["contrastive_updates"] = len(losses)
                metrics["contrastive_loss"] = (
                    sum(losses) / len(losses) if losses else None
                )
            else:
                metrics["mode"] = "actor_critic"
                metrics.update(train_actor_critic(model, optimizer, buffers))
            del buffers
            metrics["seconds"] = monotonic() - started
            metrics["steps"] = steps
            metrics["sps"] = steps / metrics["seconds"]
            record = json.dumps(metrics, allow_nan=False)
            metrics_file.write(record + "\n")
            metrics_file.flush()
            logger.info("%s", record)
            if (
                iteration % checkpoint_every == 0
                and collector.total_steps < total_timesteps
            ):
                save_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    clone_lstm,
                    optimizer,
                    contrastive_optimizer,
                    iteration,
                    collector.total_steps,
                )
                save_plots(output_dir)
                if on_checkpoint is not None:
                    on_checkpoint()
    save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        clone_lstm,
        optimizer,
        contrastive_optimizer,
        iteration,
        collector.total_steps,
    )
    if iteration:
        save_plots(output_dir)
    return output_dir / "checkpoint.pt"


def validate_config(checkpoint_every):
    for name, value in {
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_EXP": num_exp,
        "NUM_ENVS": num_envs,
        "GAME_BATCH": game_batch,
        "SEQ_LENGTH": seq_length,
        "CONTRASTIVE_LOSS_BATCH_SIZE": contrastive_loss_batch_size,
        "CHECKPOINT_EVERY": checkpoint_every,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if vector_env not in ("async", "sync"):
        raise ValueError("VECTOR_ENV must be 'async' or 'sync'")
    if actor_critic_optimizer not in ("adam", "rmsprop"):
        raise ValueError("ACTOR_CRITIC_OPTIMIZER must be 'adam' or 'rmsprop'")
    if not 0 <= prob_contrastive <= 1 or not 0 <= discount <= 1:
        raise ValueError("PROB_CONTRASTIVE and DISCOUNT must be between 0 and 1")
    if warm_up_steps < 0 or contrastive_loss_temperature <= 0:
        raise ValueError("WARM_UP_STEPS must be nonnegative and temperature positive")
    if channels_in != 3 or hidden_size_input != hidden_size_lstm:
        raise ValueError(
            "Crafter requires CHANNELS_IN=3 and HIDDEN_SIZE_INPUT=HIDDEN_SIZE_LSTM"
        )


def run(seed, output_dir, checkpoint_every=10, device=None, on_checkpoint=None):
    """Train one seed and return the path of its final checkpoint."""
    validate_config(checkpoint_every)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = device or get_device()
    logger.info(
        "Training on %s with %s %s environments for %s timesteps; output: %s",
        device,
        num_envs,
        vector_env,
        total_timesteps,
        output_dir,
    )
    # Start environment workers before CUDA initializes so forking is safe.
    env = create_vector_env(seed)
    try:
        if env.single_action_space.n != num_actions:
            raise ValueError("NUM_ACTIONS must match the Crafter action space")
        model, lstm_model = build_model(config_snapshot(), device)
        clone_lstm = deepcopy(lstm_model).requires_grad_(False)
        return train_model(
            model,
            lstm_model,
            clone_lstm,
            env,
            get_achievement_dict(),
            device,
            output_dir,
            checkpoint_every,
            on_checkpoint,
        )
    finally:
        env.close()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(
        seed=int(os.getenv("SEED", 42)),
        output_dir=os.getenv(
            "OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "sebi")
        ),
        checkpoint_every=int(os.getenv("CHECKPOINT_EVERY", 10)),
    )


if __name__ == "__main__":
    main()
