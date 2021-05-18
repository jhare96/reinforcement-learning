import gym 
import numpy as np
from PIL import Image
from collections import deque

# Code was inspired from or modified from OpenAI baselines https://github.com/openai/baselines/tree/master/baselines/common


def AtariValidate(env):
    env = FireResetEnv(env)
    env = NoopResetEnv(env, max_op=3000)
    env = StackEnv(env)
    return env

class RescaleEnv(gym.Wrapper):
    def __init__(self, env, size):
        gym.Wrapper.__init__(self, env)
        self.size = size
    
    def preprocess(self, frame):
        frame = np.array(Image.fromarray(frame).resize([self.size,self.size]))
        frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame[:,:,np.newaxis]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.preprocess(obs)


class AtariRescale42x42(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def preprocess(self,frame):
        frame = np.array(Image.fromarray(frame).resize([84,110]))[110-84:,0:84,:]
        frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        frame = np.array(Image.fromarray(frame).resize([42,42])).astype(dtype=np.uint8)
        return frame[:,:,np.newaxis]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.preprocess(obs)

class AtariRescaleEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def preprocess(self,frame):
        frame = np.array(Image.fromarray(frame).resize([84,110]))[110-84:,0:84,:]
        frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame[:,:,np.newaxis]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.preprocess(obs)

class AtariRescaleColour(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def preprocess(self,frame):
        frame = np.array(Image.fromarray(frame).resize([84,110]))[110-84:,0:84,:]
        return frame

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.preprocess(obs)


class DummyEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    def step(self, action):
        return self.env.step(action)
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, max_op=7):
        gym.Wrapper.__init__(self, env)
        self.max_op = max_op

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        noops = np.random.randint(0, self.max_op)
        for i in range(noops):
            obs, reward, done, info = self.env.step(0)
        return obs
    
    def step(self, action):
        return self.env.step(action)

class ClipRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.clip(reward, -1, 1)
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NoRewardEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, 0, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self,env)
        self.lives = 0
        self.end_of_episode = True
    

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.end_of_episode = done 
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            done = True
        self.lives = lives 
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        if self.end_of_episode:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        return obs

class TimeLimitEnv(gym.Wrapper):
    def __init__(self, env, time_limit):
        gym.Wrapper.__init__(self,env)
        self._time_limit=time_limit
        self._step = 0
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step > self._time_limit:
            done = True
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        self._step = 0
        return self.env.reset(**kwargs)



class StackEnv(gym.Wrapper):
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self,env)
        #self._stacked_frames = np.array(np.zeros([84,84,k]))
        self._stacked_frames = deque([], maxlen=k)
        self.k = k

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.stack_frames(obs)
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.stack_frames(obs, True)

    
    def stack_frames(self,frame,reset=False):
        if reset:
            for i in range(self.k):
                self._stacked_frames.append(frame)
        else:
            self._stacked_frames.append(frame)
        return np.concatenate(self._stacked_frames,axis=2)
    
    # def stack_frames(self,frame,reset=False):
    #     frame = self.preprocess(frame)
    #     if reset:
    #         for i in range(self.k):
    #             self._stacked_frames[:,:,i] = frame
    #     else:
    #         self._stacked_frames = np.roll(self._stacked_frames, -1, axis=-1)
    #         self._stacked_frames[:,:,-1] = frame
    #     return self._stacked_frames

class DummyBatchEnv(object):
    def __init__(self, env_constructor, env_id, num_envs, make_args={}, **env_args):
        self.envs = self.envs = [env_constructor(gym.make(env_id, **make_args),**env_args) for i in range(num_envs)]

    def __len__(self):
        return len(self.envs)
    
    # def __getattr__(self, name):
    #     return [getattr(self.envs[i], name) for i in range(len(self.envs))]

    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def step(self,actions):
        results = [env.step(action) for env, action in zip(self.envs,actions)]
        obs, rewards, done, info = zip(*results)
        np.stack(obs), np.stack(rewards), np.stack(done), info
        return np.stack(obs), np.stack(rewards), np.stack(done), info
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)
    
    def close(self):
        for env in self.envs:
            env.close()

class AutoResetEnv(gym.Wrapper):
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self,env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()
        return obs, reward, done, info

class ChannelsFirstEnv(gym.Wrapper):
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self,env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.transpose(2, 0, 1), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs.transpose(2, 0, 1)

def apple_pickgame(env, auto_reset=False, max_steps=1000, torch=False):
    if auto_reset:
        env = AutoResetEnv(env)
    if max_steps is not None:
        env = TimeLimitEnv(env, time_limit=max_steps)
    if torch:
        env = ChannelsFirstEnv(env)
    return env


def AtariEnv(env, k=4, rescale=84, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None, channels_first=True):
    # Wrapper function for Determinsitic Atari env 
    # assert 'Deterministic' in env.spec.id
    if reset:
        env = FireResetEnv(env)
    if Noop:
        if 'NoFrameskip' in env.spec.id :
            max_op = 30
        else:
            max_op = 7
        env = NoopResetEnv(env,max_op)
    
    if clip_reward:
        env = ClipRewardEnv(env)

    if episodic:
        env = EpisodicLifeEnv(env)

    if rescale == 42:
        env = AtariRescale42x42(env)
    elif rescale == 84:
        env = AtariRescaleEnv(env)
    else:
        raise ValueError('84 or 42 are valid rescale sizes')

    if k > 1:
        env = StackEnv(env,k)
    
    if time_limit is not None:
        env = TimeLimitEnv(env, time_limit)

    if channels_first:
        return ChannelsFirstEnv(env)
    
    return env