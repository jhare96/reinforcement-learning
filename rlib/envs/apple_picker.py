import copy
import random
from pathlib import Path

import gymnasium as gym
import numpy as np


def draw_circle(grid, x, y, r):
    left = int(np.floor(x - r * 2))
    right = int(np.ceil(x + r * 2))
    bottom = int(np.floor(y - r * 2))
    top = int(np.ceil(y + r * 2))
    for i in range(left, right):
        for j in range(bottom, top):
            deltaX = x - i
            deltaY = y - j
            distance = np.sqrt(deltaX**2 + deltaY**2)
            color = 1 if distance <= r**2 else 0  # np.clip(r - distance, 0, 1)
            try:
                grid[i, j, 2] = color
                print('i,j', (i, j))
            except IndexError:
                pass
    return grid


class ApplePicker(gym.Env):
    def __init__(self, num_objects=20, default_reward=0):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.grid = (np.load(Path(__file__).parent / 'grid.npy') * 255).astype(np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.grid.shape, dtype=np.uint8
        )
        self.state = self.grid.copy()
        self.item_locs: dict[int, dict[int, int]] = {}
        self.nobjects = num_objects
        self.agent_loc: tuple[int, int] | list[int] = [0, 0]
        self.reset()
        self.def_reward = default_reward

    def rand_loc(self):
        while True:
            x = np.random.randint(0, high=self.grid.shape[0])
            y = np.random.randint(0, high=self.grid.shape[1])
            if self.grid[x, y, 2] == 0:
                return x, y

    def generate_random_locs(self, num_objects):
        objects = 0
        while objects < num_objects:
            x, y = self.rand_loc()
            if x in self.item_locs:
                if y in self.item_locs:
                    pass
                else:
                    self.item_locs[x][y] = 1
                    objects += 1
            else:
                self.item_locs[x] = {y: 1}
                objects += 1

    def step(self, action):
        if action > 3:
            raise ValueError('action size if 4')
        if action == 0:  # up
            new_loc = max(self.agent_loc[0] - 1, 0), self.agent_loc[1]
        if action == 1:  # down
            new_loc = min(self.agent_loc[0] + 1, self.grid.shape[0] - 1), self.agent_loc[1]
        if action == 2:  # left
            new_loc = self.agent_loc[0], max(self.agent_loc[1] - 1, 0)
        if action == 3:  # right
            new_loc = self.agent_loc[0], min(self.agent_loc[1] + 1, self.grid.shape[1] - 1)

        x, y = new_loc
        reward = self.def_reward
        self.agent_loc = new_loc if self.grid[x, y, 1] == 0 else self.agent_loc
        try:
            x, y = self.agent_loc
            del self.item_locs[x][y]
            self.item_locs = {k: v for k, v in self.item_locs.items() if len(v) > 0}
            reward = 1
        except KeyError:
            pass

        self.update_state()
        info: dict = {}
        terminated = len(self.item_locs) == 0
        truncated = False
        return self.state, reward, terminated, truncated, info

    def update_state(self):
        self.state = self.grid.copy()
        x, y = self.agent_loc
        self.state[x, y, 0] = 255
        for row, cols in self.item_locs.items():
            for col in list(cols.keys()):
                self.state[row, col, 1] = 255

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.item_locs = {}
        self.agent_loc = self.rand_loc()
        self.generate_random_locs(self.nobjects)
        self.update_state()
        return self.state, {}


class ApplePickerDeterministic(ApplePicker):
    def __init__(self, num_objects=20, default_reward=0):
        gym.Env.__init__(self)
        self.action_space = gym.spaces.Discrete(4)
        self.grid = (np.load(Path(__file__).parent / 'grid.npy') * 255).astype(np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.grid.shape, dtype=np.uint8
        )
        self.state = self.grid.copy()
        self.item_locs = {}
        self.nobjects = num_objects
        self.start_loc = self.rand_loc()
        self.agent_loc = self.start_loc
        self.generate_random_locs(self.nobjects)
        self.def_reward = default_reward
        self.item_locs_master = copy.deepcopy(self.item_locs)
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Deliberately don't call super().reset() — we want a deterministic start.
        self.item_locs = copy.deepcopy(self.item_locs_master)
        self.agent_loc = self.start_loc
        self.update_state()
        return self.state, {}


if __name__ == '__main__':
    env = ApplePicker(num_objects=100)
    obs, _info = env.reset()
    for _i in range(1000):
        obs, reward, terminated, truncated, info = env.step(random.choice([0, 1, 2, 3]))
        if reward > 0:
            print('reward', reward)
        if terminated or truncated:
            print('done')
            break
