import numpy as np 
import gym
import time, copy
import scipy.misc
from collections import deque
#from DoubleDQN import ReplayMemory as RM

class FrameBuffer(object):
    def __init__(self, size, width, height, stack, Atari = True):
        self._idx = 0
        self._replay_length = size
        self._stack_size = stack
        self._Atari = Atari
        self._frames = np.empty((size,width,height), dtype=np.uint8)
        self._blank_frame = np.zeros((width,height))
        self._stacked_frames = deque([self._blank_frame for i in range(self._stack_size)], maxlen=self._stack_size)
    
    def preprocess_frame(self,frame):
        frame = scipy.misc.imresize(frame, [110,84,3])[110-84:,0:84,:]
        frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
        return frame

    def addFrame(self,frame):
        self._frames[self._idx] = self.preprocess_frame(frame)
        self._idx = (self._idx +1) % self._replay_length

    def stack_frames(self,frame,reset=False):
        self.addFrame(frame)
        if reset:
            for _ in range(self._stack_size):
                self._stacked_frames.append(self._frames[self._idx-1])
        else:
            self._stacked_frames.append(self._frames[self._idx-1])
        
        return copy.copy(self._stacked_frames)


class NumpyReplayMemory(object):
    def __init__(self, replaysize, shape):
        self._idx = 0
        self._full_flag = False
        self._replay_length = replaysize
        self._states = np.zeros((replaysize,*shape), dtype=np.uint8)
        self._actions = np.zeros((replaysize), dtype=np.int)
        self._rewards = np.zeros((replaysize), dtype=np.int)
        self._next_states = np.zeros((replaysize,*shape), dtype=np.uint8)
        self._dones = np.zeros((replaysize), dtype=np.int)
        #self._stacked_frames = deque([np.zeros((width,height), dtype=np.uint8) for i in range(stack)], maxlen=stack)
    
    def addMemory(self,state,action,reward,next_state,done):
        self._states[self._idx] = state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._next_states[self._idx] = next_state
        self._dones[self._idx] = done
        if self._idx + 1 >= self._replay_length:
            self._idx = 0
            self._full_flag = True
        else:
            self._idx += 1
    
    def __len__(self):
        if self._full_flag == False:
            return self._idx
        else:
            return self._replay_length
    
    
    def sample(self,batch_size):
        if self._full_flag == False:
            idxs = np.random.choice(self._idx, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self._replay_length, size=batch_size, replace=False)
        
        states = self._states[idxs]
        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        next_states = self._next_states[idxs]
        dones = self._dones[idxs]

        return states, actions, rewards, next_states, dones, idxs

class replayMemory(object):
    def __init__(self,replay_length,pixels=True):
        self._replay_length = replay_length
        self._pixels = pixels
        self._memory = []
        self._idx = 0
    
    def addMemory(self,state,action,reward,next_state,final_state):
        if len(self._memory) < self._replay_length:
            self._memory.append((state,action,reward,next_state,final_state))
        else:
            self._memory[self._idx] = (state,action,reward,next_state,final_state)
        self._idx = (self._idx +1) % self._replay_length
    
    def getlen(self):
        return len(self._memory)
    
    def resetMemory(self):
        self._memory = []
        self._idx = 0
    
    def sample(self, batch_size):
        idxs = np.random.choice(np.arange(len(self._memory)), size=batch_size, replace=False)
        sample = [self._memory[i] for i in idxs ]
        
        if self._pixels: #stack images to get k previous states
            states = np.stack([np.stack(sample[i][0],axis=2) for i in range(len(sample))],axis=0)
            next_states = np.stack([np.stack(sample[i][3],axis=2) for i in range(len(sample))],axis=0)
        else:
            states = np.stack([sample[i][0]for i in range(len(sample))],axis=0)
            next_states = np.stack([sample[i][3] for i in range(len(sample))],axis=0)
    
        actions = np.array([sample[i][1]for i in range(len(sample))])
        rewards = np.array([sample[i][2]for i in range(len(sample))])
        final_state = np.array([sample[i][4]for i in range(len(sample))])
        
        return (states,actions,rewards,next_states,final_state)



def stack_frames(frame,stacked_frames,reset=False):
    # Preprocess frame
    frame = preprocess_frame(frame)
    
    if reset:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.uint8) for i in range(4)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        for i in range(4):
            stacked_frames.append(frame)
    
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
    
    return stacked_state, stacked_frames

def preprocess_frame(frame):
    frame = scipy.misc.imresize(frame, [110,84,3])[110-84:,0:84,:]
    frame = np.dot(frame[...,:3], np.array([0.299, 0.587, 0.114])).astype(dtype=np.uint8)
    return frame

def main():
    env = gym.make('SpaceInvaders-v0')
    replay =NumpyReplayMemory(100000,84,84,4)
    #framebuffer = FrameBuffer(100000,84,84,4)

    obs = env.reset()
    #state = framebuffer.stack_frames(obs,reset=True)
    state = replay.stack_frames(obs,reset=True)
    print("state shape", state.shape)
    avg_time = 0 
    for t in range(int(1e7)):
        start = time.time()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        #next_state = framebuffer.stack_frames(obs,reset=False)
        next_state = replay.stack_frames(obs,reset=False)
        print("next state shape", next_state.shape)
        replay.addMemory(state,action,reward,next_state,done)
        state = next_state
        if done:
            obs = env.reset()
            #state = framebuffer.stack_frames(obs,reset=True)
            state = replay.stack_frames(obs,reset=True)
            
        
        if t > 32 :
            
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_final_states = replay.sample(32)
            end = time.time()
            #if t % 100 == 0:
                #print("next_state ", t)
                #print("batch_actions shape", batch_actions.shape)
                #print("next_state shape", len(batch_next_states), ",", batch_states[0].shape)
                #for i in range(4):
                    #scipy.misc.imshow(batch_next_states[-1,:,:,i])
            
            avg_time += (end-start)
        if t % 10000 == 0:
            print("time taken for 10000 steps", avg_time)
            avg_time = 0

if __name__ == "__main__":
    main()