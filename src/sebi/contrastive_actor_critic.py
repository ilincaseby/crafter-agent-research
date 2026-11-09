import crafter
import torch
import random
import torch.nn.functional as F
import numpy as np
import os
import json
import logging
from copy import deepcopy
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

# experiences every iteration
num_exp = int(os.getenv("NUM_EXP", 16))

# total iterations
iterations = int(os.getenv("ITERATIONS", 5000))

# contrastive loss arguments
warm_up_steps = int(os.getenv("WARM_UP_STEPS", 512))
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


def take_action(model, state, num_actions):
    logits, value, last_lstm_out, lstm_out, h_n, c_n = model(state)
    dist = torch.distributions.Categorical(logits=logits[0])
    action = dist.sample()
    return action.item()


def create_env(seed=42, log_dir=None):
    env = crafter.Env(seed=seed, size=(width, height))
    env = crafter.Recorder(
        env,
        str(log_dir or Path(__file__).resolve().parents[2] / "logs" / "sebi"),
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    return env


def get_achievement_dict():
    achievement_dict = {
        ach: index for ach, index in zip(achievement_list, range(len(achievement_list)))
    }
    return achievement_dict


def play_and_store(
    model, env, seq_length, c, h, w, hidden_size, achievement_dict, device, num_actions
):
    with torch.no_grad():
        er_buffer_obs = torch.empty(0, seq_length, c, h, w).to(device)
        er_buffer_f_obs = torch.empty(0, seq_length, c, h, w).to(device)
        er_buffer_actions = torch.empty(0).to(device)
        er_buffer_reward = torch.empty(0).to(device)
        er_buffer_achievement = torch.empty(0, len(achievement_dict)).to(device)
        er_buffer_latent_space_obs = torch.empty(0, hidden_size).to(device)
        er_buffer_latent_space_f_obs = torch.empty(0, hidden_size).to(device)
        obs = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs = obs.permute(2, 0, 1) / 255.0
        tensor_obs = torch.empty(0, c, h, w).to(device)
        for i in range(seq_length + 1):
            obs_zero = torch.zeros(1, c, h, w).to(device)
            tensor_obs = torch.cat([tensor_obs, obs_zero], dim=0)
        tensor_obs = torch.cat([tensor_obs, obs.unsqueeze(0)], dim=0)
        done = False
        past_info = {k: 0 for k, _ in achievement_dict.items()}
        while done is not True:
            action = take_action(
                model, tensor_obs[-seq_length:, :, :, :].unsqueeze(0), num_actions
            )
            obs, reward, done, info = env.step(action)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs = obs.permute(2, 0, 1) / 255.0
            tensor_obs = torch.cat([tensor_obs, obs.unsqueeze(0)], dim=0)
            er_buffer_obs = torch.cat(
                [
                    er_buffer_obs,
                    tensor_obs[-(seq_length + 1) : -1, :, :, :].unsqueeze(0),
                ],
                dim=0,
            )
            er_buffer_f_obs = torch.cat(
                [er_buffer_f_obs, tensor_obs[-seq_length:, :, :, :].unsqueeze(0)], dim=0
            )
            er_buffer_actions = torch.cat(
                [er_buffer_actions, torch.tensor([action], device=device)], dim=0
            )
            er_buffer_reward = torch.cat(
                [er_buffer_reward, torch.tensor([reward]).to(device)], dim=0
            )
            achievement_tensor = torch.zeros(len(achievement_dict)).to(device)
            for ach, obtained in info["achievements"].items():
                if obtained > 0 and past_info[ach] == 0:
                    achievement_tensor[achievement_dict[ach]] = 1.0
                    past_info[ach] = 1.0
            er_buffer_achievement = torch.cat(
                [er_buffer_achievement, achievement_tensor.unsqueeze(0)], dim=0
            )
            _, _, last_lstm_output_obs, _, _, _ = model(
                tensor_obs[-(seq_length + 1) : -1, :, :, :].unsqueeze(0)
            )
            _, _, last_lstm_output_f_obs, _, _, _ = model(
                tensor_obs[-seq_length:, :, :, :].unsqueeze(0)
            )
            er_buffer_latent_space_obs = torch.cat(
                [er_buffer_latent_space_obs, last_lstm_output_obs], dim=0
            )
            er_buffer_latent_space_f_obs = torch.cat(
                [er_buffer_latent_space_f_obs, last_lstm_output_f_obs], dim=0
            )
    er_buffer_done = torch.zeros(er_buffer_reward.shape[0]).to(device)
    er_buffer_done[-1] = 1.0
    return (
        er_buffer_obs,
        er_buffer_f_obs,
        er_buffer_actions.long(),
        er_buffer_reward,
        er_buffer_achievement,
        er_buffer_latent_space_obs,
        er_buffer_latent_space_f_obs,
        er_buffer_done,
    )


def get_device():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def create_experiences(model, env, achievement_dict, device):
    buffers = [[] for _ in range(8)]
    for _ in range(num_exp):
        episode = play_and_store(
            model,
            env,
            seq_length,
            channels_in,
            height,
            width,
            hidden_size_lstm,
            achievement_dict,
            device,
            num_actions,
        )
        for buffer, tensor in zip(buffers, episode):
            buffer.append(tensor)
    return buffers


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
    for start in range(0, num_exp, contrastive_loss_batch_size):
        stop = min(start + contrastive_loss_batch_size, num_exp)
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
            observations = torch.cat(examples)
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
    for game in range(num_exp):
        length = buffers[0][game].shape[0]
        indices = torch.randperm(length, device=buffers[0][game].device)
        for start in range(0, length, game_batch):
            batch = indices[start : start + game_batch]
            optimizer.zero_grad(set_to_none=True)
            logits, value, *_ = model(buffers[0][game][batch])
            value = value.squeeze(-1)
            with torch.no_grad():
                _, next_value, *_ = model(buffers[1][game][batch])
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


def save_plots(
    output_dir, extra_labels=None, x_key="iteration", x_label="Training iteration"
):
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
    labels.update(extra_labels or {})
    for key, label in labels.items():
        values = [record.get(key) for record in history]
        if not any(value is not None for value in values):
            continue
        figure = Figure(figsize=(8, 4), tight_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [record[x_key] for record in history],
            [float("nan") if value is None else value for value in values],
            marker=".",
        )
        axis.set(xlabel=x_label, ylabel=label, title=label)
        axis.grid(alpha=0.3)
        figure.savefig(plot_dir / f"{key}.png")
        figure.clear()


def save_checkpoint(
    path, model, clone_lstm, optimizer, contrastive_optimizer, iteration
):
    checkpoint = {
        "iteration": iteration,
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
):
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=actor_critic_lr,
        centered=centered,
        alpha=alpha,
        eps=eps_optim,
    )
    contrastive_optimizer = torch.optim.AdamW(
        lstm_model.parameters(),
        lr=lr_contrastive_learning,
        weight_decay=weight_decay,
        eps=eps_optim,
    )
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics_file:
        for iteration in range(iterations):
            started = monotonic()
            buffers = create_experiences(model, env, achievement_dict, device)
            metrics = {
                "iteration": iteration + 1,
                "average_reward": sum(rewards.sum().item() for rewards in buffers[3])
                / num_exp,
                "average_steps": sum(rewards.numel() for rewards in buffers[3])
                / num_exp,
                "achievements": {
                    key: sum(ach[:, index].sum().item() for ach in buffers[4])
                    for key, index in achievement_dict.items()
                },
            }
            if iteration >= warm_up_steps and random.random() < prob_contrastive:
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
            record = json.dumps(metrics, allow_nan=False)
            metrics_file.write(record + "\n")
            metrics_file.flush()
            logger.info("%s", record)
            if (iteration + 1) % checkpoint_every == 0 or iteration + 1 == iterations:
                save_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    clone_lstm,
                    optimizer,
                    contrastive_optimizer,
                    iteration + 1,
                )
                if iteration + 1 < iterations:
                    save_plots(output_dir)
    save_plots(output_dir)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    checkpoint_every = int(os.getenv("CHECKPOINT_EVERY", 10))
    seed = int(os.getenv("SEED", 42))
    for name, value in {
        "ITERATIONS": iterations,
        "NUM_EXP": num_exp,
        "GAME_BATCH": game_batch,
        "SEQ_LENGTH": seq_length,
        "CONTRASTIVE_LOSS_BATCH_SIZE": contrastive_loss_batch_size,
        "CHECKPOINT_EVERY": checkpoint_every,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= prob_contrastive <= 1 or not 0 <= discount <= 1:
        raise ValueError("PROB_CONTRASTIVE and DISCOUNT must be between 0 and 1")
    if warm_up_steps < 0 or contrastive_loss_temperature <= 0:
        raise ValueError("WARM_UP_STEPS must be nonnegative and temperature positive")
    if channels_in != 3 or hidden_size_input != hidden_size_lstm:
        raise ValueError(
            "Crafter requires CHANNELS_IN=3 and HIDDEN_SIZE_INPUT=HIDDEN_SIZE_LSTM"
        )
    output_dir = (
        Path(
            os.getenv(
                "OUTPUT_DIR",
                str(Path(__file__).resolve().parents[2] / "logs" / "sebi"),
            )
        )
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = get_device()
    logger.info("Training on %s; output directory: %s", device, output_dir)
    lstm_model = LatentStateEncoder(
        height,
        width,
        channels_in,
        num_layers_lstm,
        hidden_size_lstm,
        channels_hidden_1,
        channels_hidden_2,
        channels_hidden_3,
        channels_hidden_4,
        hidden_size_cnn,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    ).to(device)
    model = ActorCritic(lstm_model, num_layers_lstm, hidden_size_input, num_actions).to(
        device
    )
    clone_lstm = deepcopy(lstm_model).requires_grad_(False)
    env = create_env(seed, output_dir / "episodes")
    try:
        if env.action_space.n != num_actions:
            raise ValueError("NUM_ACTIONS must match the Crafter action space")
        train_model(
            model,
            lstm_model,
            clone_lstm,
            env,
            get_achievement_dict(),
            device,
            output_dir,
            checkpoint_every,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
