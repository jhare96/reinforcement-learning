import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rlib.envs import make as make_env, wrap as wrap_env
from rlib.envs.base import RLVecEnv

sns.set()

ep_rewards: list[float] = []


def run_episodes(env, number_episodes: int, max_steps: int) -> None:
    rl_env = wrap_env(env)
    for _episode in range(number_episodes):
        _obs, _info = rl_env.reset()
        ep_score = 0.0
        for _t in range(max_steps):
            action = rl_env.action_space.sample()
            _obs, r, terminated, truncated, _info = rl_env.step(action)
            ep_score += r
            if RLVecEnv.merge_done(terminated, truncated):
                ep_rewards.append(ep_score)
                break


def main() -> None:
    env_id = 'MountainCar-v0'
    envs = [make_env(env_id) for _ in range(64)]
    num_eps = int(1e6) // 64
    max_steps = 1000

    threads = [
        threading.Thread(target=run_episodes, args=(envs[i], num_eps, max_steps))
        for i in range(len(envs))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    rewards = np.array(ep_rewards)
    avg_reward_line = np.ones_like(rewards) * np.mean(rewards)
    filename = 'experiments/random/' + env_id + '/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    np.save(filename + str(num_eps * len(envs)) + 'random.npy', rewards)
    plt.plot(rewards)
    plt.plot(avg_reward_line, '--', color='0.5')
    plt.show()


if __name__ == "__main__":
    main()
