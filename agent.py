# agent.py
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from config import GRID_SIZE, ACTIONS, LR, BATCH_SIZE, GAMMA
from model import DuelingDQN
from replay import ReplayBuffer, Transition
import random

class Agent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        obs_dim = 4 + 3 * (GRID_SIZE ** 2)
        self.policy_net = DuelingDQN(obs_dim, len(ACTIONS)).to(device)
        self.target_net = DuelingDQN(obs_dim, len(ACTIONS)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()
        self.frame_idx = 0

    def _safe_grid_position(self, normalized_pos):
        """Convert normalized position to grid coordinates with clamping"""
        scale = GRID_SIZE - 1
        x = int(normalized_pos[0] * scale + 0.5)
        y = int(normalized_pos[1] * scale + 0.5)
        return (
            max(0, min(GRID_SIZE - 1, x)),
            max(0, min(GRID_SIZE - 1, y))
        )

    def _get_valid_mask(self, states):
        batch_size = states.shape[0]
        obstacle_maps = states[:, 4:4 + GRID_SIZE ** 2].view(batch_size, GRID_SIZE, GRID_SIZE)
        cat_positions = states[:, :2] * (GRID_SIZE - 1)
        cat_x = cat_positions[:, 0].long().clamp(0, GRID_SIZE - 1)
        cat_y = cat_positions[:, 1].long().clamp(0, GRID_SIZE - 1)

        masks = torch.zeros((batch_size, len(ACTIONS)), dtype=torch.bool, device=self.device)

        for action_idx, (dx, dy) in enumerate(ACTIONS):
            new_x = (cat_x + dx).clamp(0, GRID_SIZE - 1)
            new_y = (cat_y + dy).clamp(0, GRID_SIZE - 1)
            valid = obstacle_maps[torch.arange(batch_size), new_y, new_x] < 0.5
            masks[:, action_idx] = valid

        return masks

    def select_action(self, state, eps=0.0):
        if random.random() < eps:
            valid_actions = [i for i, valid in enumerate(self.env.get_valid_actions()) if valid]
            action = random.choice(valid_actions) if valid_actions else 0
        else:
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.policy_net(state_t)
                valid = torch.BoolTensor(self.env.get_valid_actions()).to(self.device)
                q[0, ~valid] = -float('inf')
                action = q.argmax().item()
        return action

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample and convert to tensors
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q values
        q_values = self.policy_net(states)
        current_q = q_values.gather(1, actions)

        with torch.no_grad():
            next_mask = self._get_valid_mask(next_states)

            # Double DQN
            policy_q = self.policy_net(next_states)
            policy_q[~next_mask] = torch.finfo(torch.float32).min
            next_actions = policy_q.argmax(dim=1)

            next_q_values = self.target_net(next_states)
            next_q_values[~next_mask] = torch.finfo(torch.float32).min

            next_q = next_q_values.gather(1, next_actions.unsqueeze(1))
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # Loss and optimization
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()