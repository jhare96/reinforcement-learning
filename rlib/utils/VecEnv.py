import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import multiprocessing as mp
import threading
import time
from PIL import Image
#from A2C import ActorCritic
#from networks import*
import scipy.misc
import skimage
import tensorflow as tf
#from Qvalue import Qvalue
from itertools import chain
import matplotlib.pyplot as plt
from collections import deque
#import line_profiler
#profile = line_profiler.LineProfiler()

def MarioEnv(env, rescale=84, k=4, clip_reward=True, no_reward=False):
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    if clip_reward:
        env = ClipRewardEnv(env)
    if no_reward:
        env = NoRewardEnv(env)
    if rescale is not None:
        env = RescaleEnv(env, rescale)
    if k > 1:
        env = StackEnv(env, k)
    return env 

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

# class AtariEnv(gym.Wrapper):
#     def __init__(self, env, k=4):
#         gym.Wrapper.__init__(self,env)
#         self._stacked_frames = deque([], maxlen=k)
#         self.k = k
#         self.lives = 0 
#         self.end_of_episode = True
    
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action) 
#         self.end_of_episode = done
#         lives = info['ale.lives']
#         if lives < self.lives and lives > 0:
#             done = True
#         self.lives = lives
#         obs = self.stack_frames(obs)
#         return obs, reward, done, info
    
#     def reset(self, **kwargs):
#         if self.end_of_episode:
#             obs = self.env.reset(**kwargs)
#             obs, _, done, _ = self.env.step(1)
#             if done:
#                 self.env.reset(**kwargs)
#             reset_ = True
#         else:
#             obs, _, _, _ = self.env.step(0)
#             reset_ =  False
#         self.lives = self.env.unwrapped.ale.lives()
#         return self.stack_frames(obs, reset_)

#     def preprocess(self,frame):
#         frame = np.array(Image.fromarray(frame).resize([84,110]))[110-84:,0:84,:]
#         frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
#         return frame
    
#     def stack_frames(self,frame,reset=False):
#         frame = self.preprocess(frame)
#         if reset:
#             for i in range(self.k):
#                 self._stacked_frames.append(frame)
#         else:
#             self._stacked_frames.append(frame)
#         return np.stack(self._stacked_frames,axis=2)

def AtariEnv(env, k=4, rescale=84, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None):
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

    return env

class Env(object):
    def __init__(self,env,worker_id=0): #, Wrappers=None, **wrapper_args):
        #self.env_id = env_id
        #env = gym.make(env_id)
        self.parent, self.child = mp.Pipe()
        self.worker = Env.Worker(worker_id,env,self.child)
        self.worker.daemon = True
        self.worker.start()
        self.open = True        
    
    def __del__(self):
        self.close()
        self.parent.close()
        self.child.close()
    
    def __getattr__(self, name):
        attribute = self._send_step('getattr', name)
        return attribute()
        
    def _send_step(self,cmd,action):
        self.parent.send((cmd,action))
        return self._recieve
    
    def _recieve(self,):
        return self.parent.recv()

    def step(self,action,blocking=True):
        #if self.open:
        results = self._send_step('step', action)
        # if blocking:
        #     return results()
        # else:
        return results 
    
    def reset(self):
        #if self.open:
        results = self._send_step('reset', None)
        return results()
    
    def close(self):
        if self.open:
            self.open = False
            results = self._send_step('close', None)
            self.worker.join()
    
    def render(self):
        #if self.open:
        self._send_step('render', None)
    
    class Worker(mp.Process):
        def __init__(self, worker_id, env, connection):
            import gym
            np.random.seed()
            mp.Process.__init__(self)
            self.env = env #gym.make(env_id)
            self.worker_id = worker_id
            self.connection = connection
        
        def _step(self):
            try:
                while True:
                    cmd, a = self.connection.recv()
                    if cmd == 'step':
                        obs, r, done, info = self.env.step(a)
                        if done:
                            obs = self.env.reset()
                        self.connection.send((obs,r,done,info))
                    elif cmd == 'render':
                        self.env.render()
                        #self.connection.send((1))
                    elif cmd == 'reset':
                        obs = self.env.reset()
                        self.connection.send(obs)
                    elif cmd == 'getattr':
                        self.connection.send(getattr(self.env, a))
                    elif cmd == 'close':
                        self.env.close()
                        #self.connection.send((1))
                        break
            except KeyboardInterrupt:
                print("closing worker")
            finally:
                self.env.close()
                #self.connection.close()


        def run(self,):
            self._step()




class BatchEnv(object):
    def __init__(self, env_constructor, env_id, num_envs, blocking=False, **env_args):
        #self.envs = [Env(env_constructor(gym.make(env_id),**env_args),worker_id=i) for i in range(num_envs)]
        self.envs = []
        for i in range(num_envs):
            env = gym.make(env_id)
            self.envs.append(Env(env_constructor(env, **env_args)))
        #self.envs = [env_constructor(env_id=env_id,**env_args, worker_id=i) for i in range(num_envs)]
        self.blocking = blocking

    def __len__(self):
        return len(self.envs)
    
    def __getattr__(self, name):
        return getattr(self.envs[0], name)

    def step(self,actions):
        if self.blocking: # wait for each process to return results before starting the next
            results = [env.step(action,True) for env, action in zip(self.envs,actions)]
        else:
            results = [env.step(action,False) for env, action in zip(self.envs,actions)] # apply steps async
            results = [result() for result in results] # collect results
            
        obs, rewards, done, info = zip(*results)
        np.stack(obs), np.stack(rewards), np.stack(done), info
        return np.stack(obs), np.stack(rewards), np.stack(done), info
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.stack(obs)
    
    def close(self):
        for env in self.envs:
            env.close()



def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

class ChunkEnv(object):
    def __init__(self, env_id, num_workers, num_chunks):
        self.num_workers = num_workers
        self.num_chunks = num_chunks
        self.env_id = env_id

        self.workers = []
        self.parents = []
        for i in range(num_workers):
            parent, child = mp.Pipe()
            worker = ChunkWorker(env_id,num_chunks,child)
            self.parents.append(parent)
            self.workers.append(worker)

        try:
            for worker in self.workers:
                worker.start()

        except KeyboardInterrupt:
            self.close()
            exit()
            #for w in self.workers:
                #w.env.close()
                #w.terminate()
                #exit()

        
    def _send_step(self,cmd,actions):
        #actions_list = []
        #for i in range(0,self.num_workers*self.num_chunks, self.num_chunks):
            #actions_list.append(actions[i:i+self.num_chunks])
        #print("actions list", actions_list)
        #for i in range(len(self.parents)):
            #self.parents[i].send((cmd,actions_list[i]))
        for parent, action_chunk in zip(self.parents,chunks(actions, self.num_chunks)):
            parent.send((cmd,action_chunk))
        return self._recieve
    
    def _recieve(self,):
        return [parent.recv() for parent in self.parents]

    def step(self,actions,blocking=True):
        results = self._send_step('step', actions)
        if blocking:
            results = list(chain.from_iterable(results()))
            obs, rewards, dones, infos = zip(*results)
            return np.stack(obs), np.stack(rewards), np.stack(dones), infos
        else:
            return results
    
    def reset(self):
        results = self._send_step('reset',np.zeros((self.num_chunks*self.num_workers)))
        results = list(chain.from_iterable(results()))
        return np.stack(results)
    
    def close(self):
        results = self._send_step('close',np.zeros((self.num_chunks*self.num_workers)))
        for worker in self.workers:
            worker.join()

class ChunkWorker(mp.Process):
    def __init__(self, env_id, num_chunks, connection, render=False):
        mp.Process.__init__(self)
        self.envs = [gym.make(env_id) for i in range(num_chunks)]
        self.connection = connection
        self.render = render
        
    def run(self):
        while True:
            cmd, actions = self.connection.recv()
            if cmd == 'step':
                results = []
                for a, env in zip(actions,self.envs):
                    obs, r, done, info = env.step(a)
                    if done:
                        obs = env.reset()
                    if self.render:
                        self.env.render()
                    results.append((obs,r,done,info))
                self.connection.send(results)
            elif cmd == 'reset':
                results = []
                for a, env in zip(actions,self.envs):
                    obs = env.reset()
                    results.append(obs)
                self.connection.send(results)
            elif cmd == 'close':
                for env in self.envs:
                    env.close()
                self.connection.send((1))
                break



if __name__ == "__main__":
    env_id = 'MontezumaRevenge-v4'
    env = AtariEnv(gym.make(env_id), k=4, time_limit=4500, episodic=True)
    obs = env.reset()
    for i in range(4700):
        action = env.action_space.sample()
        obs, reward, done, info  = env.step(0)
        if done:
            print(i)
            scipy.misc.imshow(obs[:,:,-1])
            env.reset()
            #plt.show()

    