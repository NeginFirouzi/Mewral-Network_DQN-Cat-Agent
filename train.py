# train.py
import numpy as np
import torch
from utils import set_seed
from env import GridEnv
from agent import Agent
from config import (
    EPISODES, MAX_STEPS, TARGET_UPDATE_FREQ,
    EPS_START, EPS_END, EPS_DECAY
)

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridEnv()
    agent = Agent(env, device)

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            agent.frame_idx += 1
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-agent.frame_idx / EPS_DECAY)

            action = agent.select_action(state, eps)

            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.optimize()

            # Update target network
            if agent.frame_idx % TARGET_UPDATE_FREQ == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            total_reward += reward
            if done:
                break

        print(f"Ep {ep + 1}/{EPISODES} | Reward: {total_reward:6.1f} | ε: {eps:.3f}")

    torch.save(agent.policy_net.state_dict(), "dqn_cat_model.pth")

if __name__ == "__main__":
    main()
