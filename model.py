# model.py
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.adv_stream   = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.adv_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)
