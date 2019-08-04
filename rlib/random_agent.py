import numpy as np 
import gym 
import matplotlib.pyplot as plt
import seaborn as sns
import os, time
import threading
sns.set()

ep_rewards = []

def run_episodes(env, number_episodes, max_steps):
    for episode in range(number_episodes):
        obs = env.reset()
        ep_score = 0
        for t in range(max_steps):
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            ep_score += r

            if done:
                ep_rewards.append(ep_score)
                break
def main():
    
    env_id = 'MountainCar-v0'
    envs = [gym.make(env_id) for i in range(64)]
    num_eps = int(1e6) // 64
    max_steps = 1000

    ep_rewards = []
    threads = [threading.Thread(target=run_episodes, args=(envs[i], num_eps, max_steps)) for i in range(len(envs))]

    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()
    

    ep_rewards = np.array(ep_rewards)
    avg_reward_line = np.ones_like(ep_rewards) * np.mean(ep_rewards)
    filename = 'experiments/random/' + env_id + '/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    np.save(filename + str(num_eps * len(envs)) + 'random.npy', ep_rewards)
    plt.plot(ep_rewards)
    plt.plot(avg_reward_line, '--', color='0.5')
    plt.show()


if __name__ == "__main__":
    main()