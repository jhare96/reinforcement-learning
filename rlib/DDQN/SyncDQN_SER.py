import tensorflow as tf 
import numpy as np 
import gym
import threading
import time, datetime
from collections import OrderedDict

from rlib.networks.networks import*
from rlib.utils.VecEnv import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.DDQN.SyncDQN import DQN
from rlib.utils.utils import one_hot, fold_batch, unfold_batch, log_uniform
#from rlib.utils.ReplayMemory import NumpyReplayMemory


main_lock = threading.Lock()

def save_hyperparameters(filename, **kwargs):
    handle = open(filename, "w")
    for key, value in kwargs.items():
        handle.write("{} = {}\n" .format(key, value))
    handle.close()

class SequentialReplayMemory(object):
    def __init__(self, replaysize, shape):
        num_actors = shape[0]
        self._idx = 0
        self._full_flag = False
        self._replay_length = replaysize
        self._states = np.zeros((replaysize,*shape), dtype=np.uint8)
        self._actions = np.zeros((replaysize,num_actors), dtype=np.int)
        self._rewards = np.zeros((replaysize,num_actors), dtype=np.int)
        #self._next_states = np.zeros((replaysize,*shape), dtype=np.uint8)
        self._dones = np.zeros((replaysize,num_actors), dtype=np.int)
        #self._stacked_frames = deque([np.zeros((width,height), dtype=np.uint8) for i in range(stack)], maxlen=stack)
    
    def addMemory(self,state,action,reward,done):
        self._states[self._idx] = state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        #self._next_states[self._idx] = next_state
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
    
    def get_size(self):
        return self._replay_length
    
    def sample(self, batch_length):
        if self._full_flag == False:
            idx = np.random.choice(self._idx - batch_length -1, replace=False)
        else:
            idx = np.random.choice(self._replay_length - batch_length - 1, replace=False)
        
        states = self._states[idx:idx+batch_length]
        actions = self._actions[idx:idx+batch_length]
        rewards = self._rewards[idx:idx+batch_length]
        dones = self._dones[idx:idx+batch_length]
        next_states = self._states[idx+batch_length+1]

        return states, actions, rewards, dones, next_states


class SyncDDQN(SyncMultiEnvTrainer):
    def __init__(self, envs, model, target_model, val_envs, action_size, log_dir='logs/', model_dir='models/',
                     train_mode='nstep', return_type='nstep', total_steps=1000000, nsteps=5, gamma=0.99, lambda_=0.95,
                     validate_freq=1e6, save_freq=0, render_freq=0, update_target_freq=10000, num_val_episodes=50, log_scalars=True, gpu_growth=True,
                     epsilon_start=1, epsilon_final=0.01, epsilon_steps = 1e6, epsilon_test=0.01, replay_length=1e6):

        
        super().__init__(envs=envs, model=model, val_envs=val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, return_type=return_type, total_steps=total_steps,
                nsteps=nsteps, gamma=gamma, lambda_=lambda_, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq,
                update_target_freq=update_target_freq, num_val_episodes=num_val_episodes, log_scalars=log_scalars, gpu_growth=gpu_growth)


        self.replay = SequentialReplayMemory(int(replay_length)//self.num_envs, self.env.reset().shape)
        
        self.target_model = target_model
        self.epsilon = np.array([epsilon_start], dtype=np.float64)
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        schedule = self.linear_schedule(self.epsilon , epsilon_final, epsilon_steps//self.num_envs)
        self.epsilon_test = np.array(epsilon_test, dtype=np.float64)

        self.action_size = action_size
        self.runner = SyncDDQN.Runner(self.model, self.target_model, self.epsilon, schedule, self.env, self.num_envs,
                                        self.nsteps, self.action_size, self.replay)
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]
        
        self.target_model.set_session(self.sess)

        hyper_paras = {'learning_rate':self.model.lr, 'learning_rate_final':self.model.lr_final, 'lr_decay_steps':self.model.decay_steps , 'grad_clip':self.model.grad_clip,
         'nsteps':self.nsteps, 'num_workers':self.num_envs, 'return type':self.return_type, 'total_steps':self.total_steps, 'gamma':gamma, 'lambda':lambda_,
         'epsilon_start':self.epsilon, 'epsilon_final':self.epsilon_final, 'epsilon_steps':self.epsilon_steps, 'update_freq':update_target_freq}
        
        hyper_paras = OrderedDict(hyper_paras)

        if self.log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        self.populate_memory()
    
    def get_action(self, state):
        if np.random.uniform() < self.epsilon_test:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.forward(state))
        return action

    def update_target(self):
        self.sess.run(self.update_weights)

    
    def populate_memory(self):
        for i in range(self.replay.get_size()//10):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            states, rewards, dones, infos = self.env.step(rand_actions)
            self.replay.addMemory(states, rand_actions, rewards, dones)
        print('finished initial memory population')

    
    def local_attr(self, attr):
        attr['update_target_freq':self.target_freq]
        return attr
    
    class Runner(object):
        def __init__(self, Q, TargetQ, epsilon, epsilon_schedule, env, num_envs, num_steps, action_size, replay):
            self.Q = Q
            self.TargetQ = TargetQ
            self.epsilon = epsilon
            self.schedule = epsilon_schedule
            self.env = env
            self.num_envs = num_envs
            self.num_steps = num_steps
            self.action_size = action_size
            self.states = self.env.reset()
            self.replay = replay
        
        def sample_replay(self):
            states, actions, rewards, dones, next_states = self.replay.sample(self.num_steps)
            TargetQsa = unfold_batch(self.TargetQ.forward(fold_batch(states)), self.num_steps, self.num_envs) # Q(s,a; theta-1)
            values = np.sum(TargetQsa * one_hot(actions, self.action_size), axis=-1) # Q(s, argmax_a Q(s,a; theta); theta-1)
            
            last_actions = np.argmax(self.Q.forward(next_states), axis=1)
            last_TargetQsa = self.TargetQ.forward(next_states) # Q(s,a; theta-1)
            last_values = np.sum(last_TargetQsa * one_hot(last_actions, self.action_size), axis=-1) # Q(s, argmax_a Q(s,a; theta); theta-1)
            return states, actions, rewards, dones, 0, values, last_values
        
        def run_(self):
            rollout = []
            for t in range(self.num_steps):
                Qsa = self.Q.forward(self.states)
                actions = np.argmax(Qsa, axis=1)
                random = np.random.uniform(size=(self.num_envs))
                random_actions = np.random.randint(self.action_size, size=(self.num_envs))
                actions = np.where(random < self.epsilon, random_actions, actions)
                next_states, rewards, dones, infos = self.env.step(actions)
                self.replay.addMemory(self.states, actions, rewards, dones)
                self.states = next_states
                self.schedule.step()
                #print('epsilon', self.epsilon)
            
        def run(self,):
            self.run_()
            return self.sample_replay()
            
    
    class linear_schedule(object):
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._epsilon_final = epsilon_final
            self._step = (epsilon - epsilon_final) / num_steps
            self._num_steps = num_steps
        
        def step(self,):
            if self._counter < self._num_steps :
                self._epsilon -= self._step
                self._counter += 1
            else:
                self._epsilon[:] = self._epsilon_final
        
        def get_epsilon(self,):
            return self._epsilon


def stackFireReset(env):
    return StackEnv(FireResetEnv(env))


def main(env_id,lr,ep_final):

    num_envs = 32
    nsteps = 5

    time.sleep(np.random.uniform(1,30))

    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/SyncDDQN_SER/' + env_id + '/n-step/RMSprop/' + current_time
    model_dir = "models/SyncDDQN_SER/" + env_id + '/' + current_time

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(16)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False , k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)

    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape

    env.close()
    print('action space', action_size)

      
    Q = DQN(mlp, input_shape=input_size, action_size=action_size, name='Q', lr=lr, lr_final=lr, grad_clip=0.5, decay_steps=50e6)
    TargetQ = DQN(mlp, input_shape=input_size, action_size=action_size, name='QTarget', lr=lr, lr_final=lr, grad_clip=0.5, decay_steps=50e6)  

    

    DDQN = SyncDDQN(envs=envs,
                    model=Q,
                    target_model=TargetQ,
                    model_dir = model_dir,
                    log_dir = train_log_dir,
                    val_envs=val_envs,
                    action_size=action_size,
                    train_mode='nstep',
                    return_type='nstep',
                    total_steps=2e6,
                    nsteps=nsteps,
                    gamma=0.99,
                    lambda_=0.95,
                    save_freq=0,
                    render_freq=0,
                    validate_freq=4e4,
                    num_val_episodes=50,
                    update_target_freq=10000,
                    epsilon_start=1,
                    epsilon_final=0.1,
                    epsilon_steps=1e6,
                    epsilon_test=0.01,
                    replay_length=5e5,
                    log_scalars=False)
    
    DDQN.train()
    del DDQN
    tf.reset_default_graph()

if __name__ == "__main__":
    #env_id_list = [ 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4',]# 'MontezumaRevengeDeterministic-v4', ]
    #env_id_list = ['MontezumaRevengeDeterministic-v4']
    env_id_list = ['MountainCar-v0',]# 'CartPole-v1', 'Acrobot-v1', ]
    for i in range(1):
        ep_final = 0.1#np.random.choice([0.1,0.01])
        lr = 1e-4#log_uniform(5e-5,1e-2)
        for env_id in env_id_list:
            main(env_id,lr,ep_final)
   # 
