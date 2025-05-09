# env.py
import random
import numpy as np
from config import GRID_SIZE, ACTIONS, RANDOMIZE_MAP, OBSTACLES, WATER, LAVA, NUM_OBSTACLES, NUM_WATER, NUM_LAVA


class GridEnv:
    def __init__(self):
        self.obstacles = set()
        self.water = set()
        self.lava = set()
        self.obstacle_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.water_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.lava_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.visited = set()
        self.reset()

    def _random_positions(self, count, existing):
        """Generate non-overlapping random positions"""
        positions = set()
        while len(positions) < count:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            pos = (x, y)
            if pos not in existing and pos not in positions:
                positions.add(pos)
        return positions

    def _update_maps(self):
        self.obstacle_map.fill(0)
        self.water_map.fill(0)
        self.lava_map.fill(0)
        for x, y in self.obstacles:
            self.obstacle_map[y, x] = 1.0
        for x, y in self.water:
            self.water_map[y, x] = 1.0
        for x, y in self.lava:
            self.lava_map[y, x] = 1.0

    def reset(self):
        self.obstacles.clear()
        self.water.clear()
        self.lava.clear()

        if RANDOMIZE_MAP:
            # Generate new positions ensuring no overlaps
            occupied = set()
            self.obstacles = self._random_positions(NUM_OBSTACLES, occupied)
            occupied.update(self.obstacles)
            self.water = self._random_positions(NUM_WATER, occupied)
            occupied.update(self.water)
            self.lava = self._random_positions(NUM_LAVA, occupied)
        else:
            # Use static configurations
            self.obstacles = set(OBSTACLES)
            self.water = set(WATER)
            self.lava = set(LAVA)

        self._update_maps()
        occupied = self.obstacles | self.water | self.lava
        free = [(x, y) for x in range(GRID_SIZE)
                for y in range(GRID_SIZE) if (x, y) not in occupied]
        free = [p for p in free if any(self.get_valid_actions(p))]
        # Ensure valid start positions
        self.cat = random.choice(free)
        remaining = [p for p in free if p != self.cat]
        self.food = random.choice(remaining) if remaining else self.cat
        self.visited = {self.cat}
        return self._get_obs()

    def _get_obs(self):
        scale = GRID_SIZE - 1
        return np.concatenate([
            [self.cat[0] / scale, self.cat[1] / scale,
             self.food[0] / scale, self.food[1] / scale],
            self.obstacle_map.ravel(),
            self.water_map.ravel(),
            self.lava_map.ravel()
        ])

    def get_valid_actions(self, pos=None):
        if pos is None:
            pos = self.cat
        x, y = pos
        valid = []
        for dx, dy in ACTIONS:
            nx, ny = x + dx, y + dy
            valid.append(
                (0 <= nx < GRID_SIZE) and
                (0 <= ny < GRID_SIZE) and
                (nx, ny) not in self.obstacles
            )
        return valid

    def step(self, action_idx):
        valid = self.get_valid_actions()
        if not valid[action_idx]:
            return self._get_obs(), -10.0, False

        dx, dy = ACTIONS[action_idx]
        old_pos = self.cat
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        self.cat = new_pos

        # Compute Manhattan distances
        old_dist = abs(old_pos[0] - self.food[0]) + abs(old_pos[1] - self.food[1])
        new_dist = abs(new_pos[0] - self.food[0]) + abs(new_pos[1] - self.food[1])

        # Difference-based shaping
        shaping = 5 * (old_dist - new_dist)

        # Small step cost to favor shorter paths
        step_cost = -0.5

        novelty_reward = 0.0
        if self.cat not in self.visited:
            novelty_reward = 0.1
            self.visited.add(self.cat)

        # Aggregate reward
        reward = step_cost + shaping + novelty_reward

        # Terminal conditions
        done = False
        if self.cat == self.food:
            reward += 20.0
            done = True
        elif self.cat in self.lava:
            reward += -20.0
            done = True
        elif self.cat in self.water:
            reward += -20.0
            done = True

        return self._get_obs(), reward, done
