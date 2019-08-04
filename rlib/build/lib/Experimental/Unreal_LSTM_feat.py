import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from utils import fold_batch
from collections import deque
from A2C import ActorCritic
from ActorCritic import ActorCritic_LSTM
from networks import*
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
from OneNetCuriosity import Curiosity_onenet

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

        

class Unreal_LSTM(object):
    def __init__(self,  policy_model, input_shape, action_size, cell_size, num_envs, RP=1, FC=1, VR=1, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.RP, self.FC, self.VR = RP, FC, VR
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)
        
        #with tf.variable_scope('ActorCritic', reuse=True):
            #self.replay_AC = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, **policy_args)
            #enc_state = self.replay_AC.lstm_output
        
        
        with tf.variable_scope('feature_control'):
            state_size = self.train_policy.lstm_output.get_shape()[1].value
            f1 = mlp_layer(self.train_policy.lstm_output, 256, activation=tf.nn.relu, name='feature_1')
            self.Qaux = tf.reshape(mlp_layer(f1, state_size*action_size, activation=None, name='Qaux_layer'), shape=[-1, state_size, action_size], name='Q_auxiliary')

            self.Qaux_actions = tf.placeholder(tf.int32, shape=[None])
            Qaux_actions_onehot = tf.one_hot(self.Qaux_actions, action_size)
            self.Qaux_target = tf.placeholder(tf.float32, shape=[None, state_size])
            Qaux_action = tf.reduce_sum(self.Qaux * tf.reshape(Qaux_actions_onehot, [-1, 1, action_size]), axis=2)
            feature_control_loss =  0.5 * tf.reduce_mean(tf.square(self.Qaux_target - Qaux_action))

        
        with tf.variable_scope('value_replay'):
            #self.replay_R = tf.placeholder(dtype=tf.float32, shape=[None])
            replay_loss = 0.5 * tf.reduce_mean(tf.square(self.train_policy.R - self.train_policy.V))
        
        self.reward_state = tf.placeholder(tf.float32, shape=[3,*input_shape], name='reward_state')
        with tf.variable_scope('ActorCritic/encoder_network', reuse=True):
            reward_enc = policy_model(self.reward_state)
        with tf.variable_scope('reward_model'):
            self.reward_target = tf.placeholder(tf.float32, shape=[None], name='reward_target')
            pred_reward = mlp_layer(tf.concat(reward_enc, axis=0), 1, activation=None)
            reward_loss = 0.5 * tf.reduce_mean(tf.square(self.reward_target - pred_reward))
        
        

        self.on_policy_loss = self.train_policy.loss
        self.auxiliary_loss = RP * reward_loss +  VR * replay_loss + FC * feature_control_loss  
        #self.loss = self.on_policy_loss + self.auxiliary_loss 
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)

        weights = self.train_policy.weights
        grads = tf.gradients(self.on_policy_loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_policy_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        grads = tf.gradients(self.auxiliary_loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_Aux_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

    def forward(self, state, hidden, validate=False):
        if validate:
            return self.validate_policy.forward(state, hidden)
        else: 
            return self.train_policy.forward(fold_batch(state), hidden)

    def forward_all(self, state, hidden):
        mask = np.zeros((1, state.shape[0]))
        feed_dict = {self.train_policy.state:state, self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1], self.train_policy.mask:mask}
        return self.sess.run([self.train_policy.policy_distrib, self.train_policy.V, self.train_policy.hidden_out, self.Qaux], feed_dict=feed_dict)
    
    def A2Cbackprop(self, states, R, actions, hidden, dones):
        feed_dict = {self.train_policy.state:states, self.train_policy.actions:actions, self.train_policy.R:R,
        self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
        self.train_policy.mask:dones}
        _, l = self.sess.run([self.train_policy_op, self.on_policy_loss], feed_dict=feed_dict)
        return l

    def Auxbackprop(self, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones):
        feed_dict = {self.Qaux_target:Qaux_target, self.Qaux_actions:Qaux_actions,
                    self.reward_state:reward_states, self.reward_target:rewards,
                    self.train_policy.state:replay_states, self.train_policy.R:replay_R,
                    self.train_policy.hidden_in[0]:replay_hidden[0], self.train_policy.hidden_in[1]:replay_hidden[1],
                    self.train_policy.mask:replay_dones}
        
        _, l = self.sess.run([self.train_Aux_op, self.auxiliary_loss], feed_dict=feed_dict)
        return l
    
    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)

    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class Auxiliary_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes)
        
        self.replay = deque([], maxlen=2000)
        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay, self.gamma)
    
    def populate_memory(self):
        for t in range(2):
            self.runner.run()
    
    def auxiliary_target(self, rewards, values, dones):
        T = len(rewards)
        #print('values shape', values.shape)
        R = np.zeros((T,*values.shape))
        dones = np.stack([dones for i in range(values.shape[1])], axis=-1)
        rewards = np.stack([rewards for i in range(values.shape[1])], axis=-1)
        #print('R shape', R.shape)
        #print('stack_shape', dones.shape)
        R[-1] = values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[-1])
        
        return R

    def sample_replay(self):
        sample_start = np.random.randint(0, len(self.replay) -21)
        worker = np.random.randint(0,self.num_envs) # sample from one of n workers
        #if self.replay[sample_start][4][worker] == True:
            #sample_start += 1
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
        replay_states = np.stack([replay_sample[i][0] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2] for i in range(len(replay_sample))])
        replay_hiddens = np.stack([replay_sample[i][3] for i in range(len(replay_sample))])
        replay_Qauxs = np.stack([replay_sample[i][4] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][5] for i in range(len(replay_sample))])

        #print('replay_Qauxs', replay_Qauxs.shape)
        #print('replay_dones', replay_dones.shape)
        _, replay_values, *_ = self.model.forward_all(replay_states[-1], replay_hiddens[-1])
        replay_R = self.multistep_target(replay_rewards, replay_values, replay_dones)
        #Qaux_actions = self.fold_batch(np.argmax(Qauxs, axis=-1))
        Qaux_target = self.auxiliary_target(replay_rewards, np.max(replay_Qauxs[-1], axis=-1), replay_dones)
        
        return self.fold_batch(replay_states), self.fold_batch(replay_actions), self.fold_batch(replay_R), self.fold_batch(Qaux_target), replay_hiddens, replay_dones
    
    def sample_reward(self):
        worker = np.random.randint(0,self.num_envs) # sample from one of n workers
        replay_rewards = np.array([self.replay[i][2][worker] for i in range(len(self.replay))])[3:]
        #print('replay_rewards', replay_rewards.shape)
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        #print('zerpo', zero_idxs.shape)
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        replay_states = np.stack([self.replay[i][0][worker] for i in range(idx-3,idx)])
        replay_reward = self.replay[idx][2][worker]
        return replay_states, replay_rewards
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments

            num_steps - number of Total training steps across all environemnts
            nsteps - number of steps performed in environment before updating ActorCritic
        '''
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        self.populate_memory()
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, hidden_batch, Qauxs, dones, infos, values = self.runner.run()

            R = self.multistep_target(rewards, values, dones, clip=False)
            # stack all states,  actions and Rs across all workers into a single batch
            states, actions, rewards, R = self.fold_batch(states), self.fold_batch(actions), self.fold_batch(rewards), self.fold_batch(R)
            A2Closs = self.model.A2Cbackprop(states, R, actions, hidden_batch[0], dones)
            
            #sample replay
            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_hiddens, replay_dones = self.sample_replay()
            #print('replay_R', replay_R.shape)
            #print('replay_states', replay_states.shape)
            #print('replay_rewards', replay_rewards.shape)
            #print('Qaux shape', Qaux_target.shape)
        
            Auxloss = self.model.Auxbackprop(reward_states, rewards, Qaux_target,
                     replay_actions, replay_states, replay_R, replay_hiddens[0], replay_dones)

            l = A2Closs + Auxloss
            if self.render_freq > 0 and t % (self.validate_freq * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % self.validate_freq == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % self.save_freq == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + self.current_time + '/' + str(s) + ".ckpt") )
                print('saved model')
            
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value, hidden = self.model.forward(state[np.newaxis], hidden, validate=True)
                #print('policy', policy, 'value', value)
                action = np.random.choice(policy.shape[1], p=policy[0])
                next_state, reward, done, info = env.step(action)
                state = next_state

                episode_score.append(reward)
                
                if render:
                    with self.lock:
                        env.render()

                if done or t == max_steps -1:
                    tot_reward = np.sum(episode_score)
                    with self.lock:
                        self.validate_rewards.append(tot_reward)
                    
                    break
        if render:
            with self.lock:
                env.close()

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay, gamma):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.prev_hidden = self.model.get_initial_hidden(len(self.env))

        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values, hidden, Qaux = self.model.forward_all(self.states, self.prev_hidden)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states.copy(), actions, rewards, self.prev_hidden, Qaux, dones, infos))
                self.replay.append((self.states, actions, rewards, self.prev_hidden, Qaux, dones, infos)) # add to replay memory
                self.states = next_states
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode
            
            states,  actions,  rewards, hidden_batch, Qaux, dones, infos = zip(*rollout)
            states,  actions,  rewards, Qaux, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(Qaux), np.stack(dones)
            return states, actions, rewards, hidden_batch , Qaux, dones, infos, values
            

def main(env_id, Atari=True):


    config = tf.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.Session(config=config)

    print('gpu aviabliable', tf.test.is_gpu_available())

    num_envs = 32
    nsteps = 20

    env = gym.make(env_id)
    #action_size = env.action_space.n
    #input_size = env.reset().shape[0]
    
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(1)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=True, clip_reward=True)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Auxiliary/' + env_id + '/'
    model_dir = "models/Auxiliary/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = Unreal_LSTM(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      cell_size = 256,
                      num_envs = num_envs,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=40.0,
                      policy_args={})

    

    auxiliary = Auxiliary_Trainer(envs = envs,
                                  model = model,
                                  file_loc = [model_dir, train_log_dir],
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  total_steps = 5e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e5,
                                  save_freq = 0,
                                  render_freq = 3,
                                  num_val_episodes = 3)

    
    
    hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':num_envs,
                  'total_steps':auxiliary.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5}
    
    filename = train_log_dir + auxiliary.current_time + '/hyperparameters.txt'
    auxiliary.save_hyperparameters(filename, **hyper_paras)

    auxiliary.train()

    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = [ 'FreewayDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'PongDeterministic-v4', 'MontezumaRevengeDeterministic-v4']
    #env_id_list = ['MountainCar-v0','CartPole-v1']
    for env_id in env_id_list:
        main(env_id)
    