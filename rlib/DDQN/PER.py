import tensorflow as tf 
import numpy as np 
import gym
from Qvalue import mlp_layer, conv_layer
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
from SyncDQN import DQN
from networks import*
from collections import deque
from ReplayMemory import NumpyReplayMemory
import time

class NumpyPER(object):
    def __init__(self, size, input_shape):
        self._idx = 0
        self._full_flag = False
        self._replay_length = size
        self._states = np.zeros((size, *input_shape), dtype=np.uint8)
        self._actions = np.zeros((size), dtype=np.int)
        self._rewards = np.zeros((size), dtype=np.int)
        self._next_states = np.zeros((size, *input_shape), dtype=np.uint8)
        self._dones = np.zeros((size), dtype=np.int)
        self._priority = np.zeros((size))
        self.alpha = 0.6
        self.beta = 0.4
        #self._stacked_frames = deque([np.zeros((width,height), dtype=np.uint8) for i in range(stack)], maxlen=stack)
    
    def addMemory(self, state, action, reward, next_state, done, priority):
        self._states[self._idx] = state
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._next_states[self._idx] = next_state
        self._dones[self._idx] = done
        self._priority[self._idx] = priority
        if self._idx + 1 >= self._replay_length:
            self._idx = 0
            self._full_flag = True
        else:
            self._idx += 1
    
    def set_priority(self, idx, priority):
        self._priority[self._idx] = priority
    
    def get_pmax(self):
        return np.max(self._priority)
    
    
    def __len__(self):
        if self._full_flag == False:
            return self._idx
        else:
            return self._replay_length
    
    
    def sample(self,batch_size):
        if self._full_flag == False:
            length = self._idx
        else:
            length = self._replay_length

        prob_dense = self._priority[:length]
        prob = np.power(prob_dense, self.alpha) / np.sum(np.power(prob_dense, self.alpha))
        
        IS_weights = np.power(length * prob, -self.beta) 
        IS_weights /= np.max(IS_weights)

        idxs = np.random.choice(length, size=batch_size, p=prob, replace=False)
        
        states = self._states[idxs]
        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        next_states = self._next_states[idxs]
        dones = self._dones[idxs]

        return states, actions, rewards, next_states, dones, IS_weights, idxs


class DQN(object):
    def __init__(self, model, input_shape, action_size, name, learning_rate=0.00025, grad_clip = 0.5, decay_steps=50e6, learning_rate_final=0, **model_args):
        self.learning_rate = learning_rate
        self.learning_rate_final = learning_rate_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.name = name 
        self.action_size = action_size
        self.sess = None

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)

        with tf.variable_scope(name):
            with tf.variable_scope('encoder_network'):
                self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
                dense = model(self.state, **model_args)
    
            with tf.variable_scope("State_Action"):
                self.Qsa = mlp_layer(dense, action_size, activation=None)
                self.R = tf.placeholder("float", shape=[None])
                self.ISweights = tf.placeholder("float", shape=[None])
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                actions_onehot = tf.one_hot(self.actions, action_size)
                self.Qvalue = tf.reduce_sum(tf.multiply(self.Qsa, actions_onehot), axis = 1)
                self.loss = tf.reduce_mean(tf.multiply(self.ISweights , tf.square(self.R - self.Qvalue)))
        
        
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(learning_rate, global_step, decay_steps, end_learning_rate=learning_rate_final, power=1.0, cycle=False, name=None)
            optimiser = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, epsilon=1e-5)
            #self.train_op = optimiser.minimize(self.loss, global_step=global_step)
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    
    def forward(self, state):
        return self.sess.run(self.Qsa, feed_dict = {self.state: state})

    def backprop(self, states, R, weights, actions):
        _,l = self.sess.run([self.train_op,self.loss], feed_dict = {self.state:states, self.R:R, self.ISweights:weights, self.actions:actions})
        return l
    
    def set_session(self, sess):
        self.sess = sess

class SyncDDQN(SyncMultiEnvTrainer):
    def __init__(self, envs, model, target_model, file_loc, val_envs, action_size,
                     train_mode='nstep', total_steps=1000000, nsteps=5,
                     validate_freq=0, save_freq=0, render_freq=0, update_target_freq=10000,
                     epsilon_start=1, epsilon_final=0.01, epsilon_steps = 1e6, epsilon_test=0.01):

        
        super().__init__(envs=envs, model=model, file_loc=file_loc, val_envs=val_envs, train_mode=train_mode, total_steps=total_steps,
                         nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq, update_target_freq=update_target_freq)
        
        print('validate freq', self.validate_freq)
        print('target freq', self.target_freq)
        
        
        self.target_model = target_model
        self.target_model.set_session(self.sess)
        self.epsilon = np.array([epsilon_start], dtype=np.float64)
        self.schedule = self.linear_schedule(self.epsilon , epsilon_final, epsilon_steps//self.num_envs)
        self.test_epsilon = np.array([epsilon_test] , dtype=np.float64)
        self.action_size = action_size
        #self.runner = SyncDDQN.Runner(self.model, self.target_model, self.epsilon, schedule, self.env, self.num_envs, self.nsteps, self.action_size, self.sess)
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='QTarget'), tf.trainable_variables('Q'))]
        
        self.states = self.env.reset()
        #self.replay = deque([], maxlen=int(5e5))
        #self.replay.append([np.zeros_like(self.states[0]),0,0, np.zeros_like(self.states[0])])
        input_shape = self.env.reset().shape[1:]
        self.replay = NumpyPER(int(1e5), input_shape)
        self.replay.addMemory(np.zeros_like(self.states[0]), 0, 0, np.zeros_like(self.states[0]), True, priority=1)
        
        # self.priority = deque([], maxlen=int(5e5))
        # self.priority.append(1)
        #self.replay.set_priority(idx=0, priority=1)
        self.batch_size = 1024
    
    def get_action(self, state):
        if np.random.uniform() < self.test_epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(self.model.forward(state))
        return action

    def update_target(self):
        self.sess.run(self.update_weights)
    
    def one_hot(self,x,n_labels):
        return np.eye(n_labels)[x]
    
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments
        '''
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        
        # main loop
        start = time.time()
        for update in range(1,num_updates+1):
            batch = []
            for t in range(self.nsteps):
                actions = np.argmax(self.model.forward(self.states), axis=1)
                random = np.random.uniform(size=(self.num_envs))
                random_actions = np.random.randint(self.action_size, size=(self.num_envs))
                actions = np.where(random < self.epsilon, random_actions, actions)
                next_states, rewards, dones, infos = self.env.step(actions)
                batch.append((self.states, actions, rewards, next_states, dones, infos))
                self.states = next_states
                self.schedule.step()
            
            
            

            states, actions, rewards, next_states, dones, infos, = zip(*batch)
            states, actions, rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(dones)

            action_values = self.target_model.forward(states[-1]) # Q(s,a; theta-1)
            actions_one_hot = self.one_hot(actions[-1],self.action_size)
            values = np.sum(action_values * actions_one_hot, axis=1) # Q(s, argmax_a Q(s,a; theta); theta-1)
            
            T = len(rewards)
            
            # Calculate R for advantage A = R - V 
            R = np.zeros((T,self.num_envs))
            R[-1] = values * (1-dones[-1])
            
            for i in reversed(range(T-1)):
                # restart score if done as wrapped env continues after end of episode
                R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
                
            
            
            # add each experience in multi-env rollout to replay
            states, actions, R, next_states = self.fold_batch(states), self.fold_batch(actions), self.fold_batch(R), self.fold_batch(next_states)
            #pmax = np.max(np.array(self.priority))
            pmax = self.replay.get_pmax()
            for j in range(states.shape[0]):
                #self.replay.addMemory([states[j],actions[j],R[j],next_states[j]])
                self.replay.addMemory(states[j],actions[j],R[j],next_states[j],False,pmax)
                #self.priority.append(pmax)
            
            if update > 10:
                
                sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones, IS_weights, idxs = self.replay.sample(self.batch_size)
                
                # TD_target = sample_rewards + 0.99 * np.max(self.target_model.forward(sample_next_states), axis=-1)
                
                Qsa = self.model.forward(sample_states)
                actions_one_hot = self.one_hot(sample_actions, self.action_size)
                Qvalues = np.sum(Qsa * actions_one_hot, axis=-1)

                #TD_error = TD_target - Qvalues
                
                weights = np.array(IS_weights[idxs])
                #print('weights', weights.mean())
                
                
                l = self.model.backprop(sample_states, sample_rewards, weights, sample_actions)
                
                
                for i in range(self.batch_size):
                    #self.priority[idxs[i]] = np.abs(TD_target[i] - Qvalues[i])
                    self.replay.set_priority(idxs[i], np.abs(sample_rewards[i]- Qvalues[i]))
                
                #print('update', update)
                #print('TD sh, upape', np.abs(TD_target[i] - Qvalues[i]).shape)
                #sample = [self.replay[i] for i in idxs ]
                #states = np.array([sample[i][0]for i in range(len(sample))])
                #actions = np.array([sample[i][1]for i in range(len(sample))])
                #rewards_n = np.array([sample[i][2]for i in range(len(sample))])
     
            if self.render_freq > 0 and update % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and update % self.validate_freq == 0:
                self.validation_summary(update,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  update % self.save_freq == 0: 
                s += 1
                self.save_model(s)
                print('saved model')
            
            if self.target_freq > 0 and update % self.target_freq == 0: # update target network (for value based learning e.g. DQN)
                self.update_target()
        
        self.env.close()

    
    class Runner(object):
        def __init__(self, Q, TargetQ, epsilon, epsilon_schedule, env, num_envs, num_steps, action_size, sess):
            self.Q = Q
            self.TargetQ = TargetQ
            self.epsilon = epsilon
            self.schedule = epsilon_schedule
            self.env = env
            self.num_envs = num_envs
            self.num_steps = num_steps
            self.action_size = action_size
            self.sess = sess
            
            self.states = self.env.reset()
            
        def one_hot(self,x,n_labels):
            return np.eye(n_labels)[x]
        
    
    class linear_schedule(object):
        def __init__(self, epsilon, epsilon_final, num_steps=1000000):
            self._counter = 0
            self._epsilon = epsilon
            self._step = (epsilon_final - epsilon) / num_steps
            self._num_steps = num_steps
        
        def step(self,):
            if self._counter < self._num_steps :
                self._epsilon += self._step
            self._counter += 1
        
        def get_epsilon(self,):
            return self._epsilon




def main(env_id):
    num_envs = 32
    nsteps = 5

    train_log_dir = 'logs/PER/' + env_id +'/'
    model_dir = "models/PER/" + env_id + '/'

    env = gym.make(env_id)
    action_size = env.action_space.n # get number of discrete actions
    input_size = len(env.reset())
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
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
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False , k=4, reset=reset, episodic=True, clip_reward=True)

    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape

    env.close()
    print('action space', action_size)

    
    
    
    dqn_cnn_args = {'input_shape':[84,84,4], 'action_size':action_size,  'learning_rate':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(num_envs*nsteps),
                    'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
   
    dqn_mlp_args = {'input_shape':input_size, 'dense_size':64, 'action_size':action_size, 'learning_rate':1e-3, 'grad_clip':0.5, 'decay_steps':1, 'learning_rate_final':1e-3}

      
    Q = DQN(mlp, **dqn_mlp_args, name='Q')
    TargetQ = DQN(mlp, **dqn_mlp_args, name='QTarget') 
    
    

    DDQN = SyncDDQN(envs = envs,
                    model = Q,
                    target_model = TargetQ,
                    file_loc = [model_dir, train_log_dir],
                    val_envs = val_envs,
                    action_size = action_size,
                    train_mode ='nstep',
                    total_steps = 5e6,
                    nsteps = nsteps,
                    validate_freq = 1e5,
                    save_freq = 0,
                    render_freq = 0,
                    update_target_freq = 10000,
                    epsilon_start = 1,
                    epsilon_final = 0.01,
                    epsilon_steps = 2e6,
                    epsilon_test = 0.01)
    

    DDQN.train()

    del DDQN

    tf.reset_default_graph()

if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'PongDeterministic-v4',  'SeaquestDeterministic-v4', 'BreakoutDeterministic-v4']
    env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1']
    for env_id in env_id_list:
        main(env_id)