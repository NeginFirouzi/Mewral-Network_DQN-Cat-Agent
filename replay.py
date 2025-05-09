# replay.py
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)