# DQN Cat Agent

A simple grid‑world RL environment where I tried to train a cat agent to navigate to its food while avoiding water and lava hazards.

## Project Structure

```text
├── config.py         # Hyperparameters, environment settings, and asset paths
├── utils.py          # Utility function for reproducibility and seeding 
├── env.py            # Grid environment implementation (states, actions, rewards)
├── model.py          # Neural network architecture (Dueling DQN)
├── replay.py         # Experience replay buffer implementation
├── agent.py          # Agent class (policy_net, target_net, optimize)
├── train.py          # Main training loop 
├── play.py           # PyGame visualization for trained agent
├── images/           # .png/.jpg assets for rendering (cat, food, water, lava, etc.)
└── dqn_cat_model.pth # Saved PyTorch model weights after training
```

<p align="center">
  <img src="Gif.gif" alt="Description of GIF" width="500"/>
</p>
