import gym
import crafter
import torch.nn as nn
import matplotlib as plt
import torch
import random
from collections import deque
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from cnn_encoder import CNNEncoder
from rnn_q_net import RNNDeepQNet

seed = int(os.getenv('SEED', 42))
save_stats = os.getenv('SAVE_STATS', 'true').lower() in ('true', '1', 'yes')
save_video = os.getenv('SAVE_VIDEO', 'false').lower() in ('true', '1', 'yes')
save_episode = os.getenv('SAVE_EPISODE', 'false').lower() in ('true', '1', 'yes')
height = int(os.getenv('HEIGHT', 64))
weight = int(os.getenv('WEIGHT', 64))
channels_in = int(os.getenv('CHANNELS_IN', 3))
channels_hidden_1 = int(os.getenv('CHANNELS_HIDDEN_1', 256))
channels_hidden_2 = int(os.getenv('CHANNELS_HIDDEN_2', 128))
hidden_size = int(os.getenv('HIDDEN_SIZE', 128))
kernel_size = int(os.getenv('KERNEL_SIZE', 3))
stride = int(os.getenv('STRIDE', 1))
padding = int(os.getenv('PADDING', 0))
dilation = int(os.getenv('DILATION', 1))
groups = int(os.getenv('GROUPS', 1))
hidden_size_inside_1 = int(os.getenv('HIDDEN_SIZE_INSIDE_1', 128))
hidden_size_inside_2 = int(os.getenv('HIDDEN_SIZE_INSIDE_2', 256))
num_actions = int(os.getenv('NUM_ACTIONS', 17))
num_games_played = int(os.getenv('NUM_GAMES_PLAYED', 1000))
eps_greedy = float(os.getenv('EPS_GREEDY', 1.0))
factor_decay_eps = float(os.getenv('FACTOR_DECAY_EPS', 0.9985))
discount_factor = float(os.getenv('DISCOUNT_FACTOR', 0.99))
steps_updating = int(os.getenv('STEPS_UPDATING', 400))
batch_train_size = int(os.getenv('BATCH_TRAIN_SIZE', 64))
seq_length = int(os.getenv('SEQ_LENGTH', 3))
lr = float(os.getenv('LR', 2.5e-4))
centered = os.getenv('CENTERED', 'true').lower() in ('true', '1', 'yes')
alpha = float(os.getenv('ALPHA', 0.95))
eps = float(os.getenv('EPS', 0.01))
log_path = os.getenv('LOG_PATH', '../logs/')
save_path = os.getenv('SAVE_PATH', '../models/')

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = 1
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_env():
    env = crafter.Env(seed=seed)
    env = crafter.Recorder(
        env, '../logs',
        save_stats=save_stats,
        save_video=save_video,
        save_episode=save_episode,
    )
    return env

def loss_bellman(target, pred):
    diff = target - pred
    return diff.mean()

def eps_greedy(model, state, num_actions, eps=0.3):
    sample_from_model = random.choices([0, 1], weights=[eps, 1 - eps])[0]
    return torch.argmax(model(state)).item() if sample_from_model == 1 else random.choice(range(num_actions))

def play_and_store(model, env, er_buffer_obs, er_buffer_future_obs, er_buffer_action, er_buffer_reward, er_done, num_actions, seq_length, eps=0.3):
    obs = env.reset()
    done = False
    stack_frames = deque(maxlen=seq_length)
    for _ in range(seq_length):
        obs_empty = torch.empty((channels_in, height, weight), dtype=torch.float32).to(device)
        stack_frames.append(obs_empty)
    obs_tensor = torch.permute(torch.tensor(obs).to(torch.float32), (2, 0, 1)).to(device)
    stack_frames.append(obs_tensor)
    while done is not True:
        obs_for_model = torch.stack(list(stack_frames)).to(device)
        action = eps_greedy(model, obs_for_model, num_actions, eps)
        obs, reward, done, info = env.step(action)
        stack_frames.append(torch.permute(torch.tensor(obs).to(torch.float32), (2, 0, 1)).to(device))
        future_obs = torch.stack(list(stack_frames)).to(device)
        er_buffer_obs = torch.cat([er_buffer_obs, obs_for_model.unsqueeze(0).to(device)], dim = 0)
        er_buffer_future_obs = torch.cat([er_buffer_future_obs, future_obs.unsqueeze(0).to(device)], dim = 0)
        er_buffer_action = torch.cat([er_buffer_action, torch.tensor([action]).to(device)], dim = 0)
        er_buffer_reward = torch.cat([er_buffer_reward, torch.tensor([reward]).to(device)], dim = 0)
        er_done = torch.cat([er_done, torch.tensor([1 if done is True else 0]).to(device)])
        
    return er_buffer_obs, er_buffer_future_obs, er_buffer_action, er_buffer_reward, er_done

def q_learning(
    env,
    q_network,
    target_q_network,
    num_actions,
    num_games_played,
    eps,
    factor_decay_eps,
    discount_factor,
    steps_updating,
    device
):
    updates = 0
    for _ in range(num_games_played):
        # Init experience replay buffer
        er_buffer_obs = torch.empty(0, seq_length, channels_in, height, weight).to(device)
        er_buffer_future_obs = torch.empty(0, seq_length, channels_in, height, weight).to(device)
        er_buffer_action = torch.empty(0).to(device)
        er_buffer_reward = torch.empty(0).to(device)
        er_done = torch.empty(0).to(device)

        er_buffer_obs, er_buffer_future_obs, er_buffer_action, er_buffer_reward, er_done = play_and_store(q_network, env, er_buffer_obs, er_buffer_future_obs, er_buffer_action, er_buffer_reward, er_done, num_actions, seq_length, eps=eps)
        len_steps = er_buffer_action.shape[0]
        indices = torch.randperm(len_steps)

        random_er_buffer_obs = er_buffer_obs[indices]
        random_er_buffer_future_obs = er_buffer_future_obs[indices]
        random_er_buffer_action = er_buffer_action[indices]
        random_er_buffer_reward = er_buffer_reward[indices]
        random_er_done = er_done[indices]
        
        optimizer = torch.optim.RMSprop(
            q_network.parameters(),
            lr=lr,
            centered=centered,
            alpha=alpha,
            eps=eps
        )

        print(len_steps)
        for i in range(0, len_steps, batch_train_size):
            optimizer.zero_grad()
            target_q_network.eval()
            pred = torch.max(q_network(random_er_buffer_obs[i:min(i + batch_train_size, len_steps)]), dim=1).values
            max_next_state_q = torch.max(target_q_network(random_er_buffer_future_obs[i:min(i + batch_train_size, len_steps)]), dim = 1).values
            next_state_q = discount_factor * (torch.tensor(1) - random_er_done[i:min(i + batch_train_size, len_steps)]) * max_next_state_q
            target = random_er_buffer_reward[i:min(i + batch_train_size, len_steps)] + next_state_q
            loss = loss_bellman(target, pred)
            loss.backward()
            optimizer.step()
            updates += 1
            if updates % steps_updating == 0:
                target_q_network.load_state_dict(q_network.state_dict())
    eps = eps * factor_decay_eps
    return q_network, target_q_network

if __name__ == '__main__':
    set_seed(seed)
    env = create_env()
    q_network = RNNDeepQNet(height, weight, channels_in, channels_hidden_1,
                        channels_hidden_2, hidden_size, kernel_size,
                        hidden_size_inside_1, hidden_size_inside_2, num_actions, device).to(device)

    target_q_network = RNNDeepQNet(height, weight, channels_in, channels_hidden_1,
                            channels_hidden_2, hidden_size, kernel_size,
                            hidden_size_inside_1, hidden_size_inside_2, num_actions).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    q_network, target_q_learning = q_learning(env, q_network, target_q_network, num_actions, num_games_played, eps, factor_decay_eps, discount_factor, steps_updating, device)
    torch.save(q_network.state_dict(), save_path + 'rnn_q_network')