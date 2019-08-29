import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
import scipy
from rlib.utils.utils import fold_batch, one_hot
from collections import deque
from rlib.A2C.A2C import ActorCritic
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*


def concat_action_reward(actions, rewards, num_classes):
    concat = one_hot(actions, num_classes)
    concat[:,-1] = rewards   
    return concat  

def AtariEnv_(env, k=4, episodic=True, reset=True, clip_reward=True, Noop=True, time_limit=None):
    # Wrapper function for Determinsitic Atari env 
    # assert 'Deterministic' in env.spec.id
    if reset:
        env = FireResetEnv(env)
    if Noop:
        max_op = 7
        env = NoopResetEnv(env,max_op)
    
    if clip_reward:
        env = ClipRewardEnv(env)

    if episodic:
        env = EpisodicLifeEnv(env)

    env = AtariRescaleColour(env)
    
    if k > 1:
        env = StackEnv(env,k)
    if time_limit is not None:
        env = TimeLimitEnv(env, time_limit)
    return env



class ActorCritic_LSTM(object):
    def __init__(self, model_head, input_shape, action_size, num_envs, cell_size, entropy_coeff=0.001, value_coeff=0.5,
                 lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size
        self.num_envs = num_envs
        self.sess = None
        print('action_size', action_size)

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='time_batch_state') # [time*batch, *input_shape]

        with tf.variable_scope('encoder_network'):
            self.dense = model_head(self.state, **model_head_args)
            dense_size = self.dense.get_shape()[1].value
            unfolded_state = tf.reshape(self.dense, shape=[-1, num_envs, dense_size], name='unfolded_state')
        
        with tf.variable_scope('lstm'):
            self.action_reward = tf.placeholder(tf.float32, shape=[None, num_envs, action_size+1], name='last_action_reward') # [action_t-1, reward_t-1]
            lstm_input = tf.concat([unfolded_state, self.action_reward], axis=2)
            print('lstm input ', lstm_input.get_shape().as_list())
            #self.mask = tf.placeholder(tf.float32, shape=[None, None])
            #self.lstm_output, self.hidden_in, self.hidden_out = lstm(unfolded_state, cell_size=cell_size, fold_output=True, time_major=True)
            self.lstm_output, self.hidden_in, self.hidden_out, self.mask = lstm_masked(lstm_input, cell_size=cell_size, batch_size=num_envs, fold_output=True, time_major=True)

        with tf.variable_scope('critic'):
            self.V = tf.reshape( mlp_layer(self.lstm_output, 1, name='state_value', activation=None), shape=[-1])

        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.lstm_output, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R = tf.placeholder(dtype=tf.float32, shape=[None], name='R')
            Advantage = self.R - self.V
            value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.loss =  policy_loss + value_coeff * value_loss - entropy_coeff * entropy

        if opt:
            global_step = tf.Variable(0, trainable=False)
            lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

            
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)
    
    def get_initial_hidden(self, batch_size):
        return np.zeros((batch_size, self.cell_size)), np.zeros((batch_size, self.cell_size))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * np.stack([idxs for i in range(self.cell_size)], axis=1)

    def forward(self, state, hidden, action_reward):
        mask = np.zeros((1, self.num_envs)) # state = [time, batch, ...]
        feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:mask, self.action_reward:action_reward}
        policy, value, hidden = self.sess.run([self.policy_distrib, self.V, self.hidden_out], feed_dict = feed_dict)
        return policy, value, hidden

    def backprop(self, state, y, a, hidden, dones):
        feed_dict = {self.state : state, self.R : R, self.actions: a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:dones, self.action_reward:action_reward}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess


class UnrealA2C(object):
    def __init__(self,  policy_model, input_shape, action_size, cell_size, num_envs, RP=1.0, PC=1.0, VR=1.0, entropy_coeff=0.001, value_coeff=0.5, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.RP, self.PC, self.VR = RP, PC, VR
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.entropy_coeff, self.value_coeff = entropy_coeff, value_coeff
        self.grad_clip = grad_clip
        self.action_size = action_size
        print('action_size', action_size)

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff, lr=lr, lr_final=lr, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff, lr=lr, lr_final=lr, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)
            self.replay_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, entropy_coeff=entropy_coeff, value_coeff=value_coeff, lr=lr, lr_final=lr, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)

        with tf.variable_scope('pixel_control', reuse=tf.AUTO_REUSE):
            self.Qaux_batch = self._build_pixel(self.train_policy.lstm_output)
            self.Qaux = self._build_pixel(self.replay_policy.lstm_output)
            
            self.Qaux_target = tf.placeholder("float", [None, 21, 21]) # temporal difference target for Q_aux
            self.Qaux_actions = tf.placeholder(tf.int32, [None])
            one_hot_actions = tf.one_hot(self.Qaux_actions, action_size)
            pixel_action = tf.reshape(one_hot_actions, shape=[-1,1,1, action_size], name='pixel_action')
            Q_aux_action = tf.reduce_sum(self.Qaux * pixel_action, axis=3)
            pixel_loss = 0.5 * tf.reduce_mean(tf.square(self.Qaux_target - Q_aux_action)) # l2 loss for Q_aux over all pixels and batch

        
        with tf.variable_scope('value_replay'):
            #self.replay_R = tf.placeholder(dtype=tf.float32, shape=[None])
            replay_loss = 0.5 * tf.reduce_mean(tf.square(self.replay_policy.R - self.replay_policy.V))
        
        self.reward_state = tf.placeholder(tf.float32, shape=[3, *input_shape], name='reward_state')
        with tf.variable_scope('ActorCritic/encoder_network', reuse=True):
            reward_enc = policy_model(self.reward_state)
        with tf.variable_scope('reward_model'):
            self.reward_target = tf.placeholder(tf.float32, shape=[1, 3], name='reward_target')
            concat_states = tf.reshape(reward_enc, shape=[1,-1])
            r1 = mlp_layer(concat_states, 128, activation=tf.nn.relu, name='reward_hidden')
            print('rl shape', r1.get_shape().as_list())
            pred_reward = mlp_layer(r1, 3, activation=None, name='pred_reward')
            print('pred reward shape', pred_reward.get_shape().as_list())
            #reward_loss = 0.5 * tf.reduce_mean(tf.square(self.reward_target - pred_reward)) #mse
            reward_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred_reward, onehot_labels=self.reward_target))  # cross entropy over caterogical reward 
            print('reward loss ', reward_loss)
            
        
        

        self.on_policy_loss = self.train_policy.loss
        self.auxiliary_loss = PC * pixel_loss +  RP * reward_loss +  VR * replay_loss 
        self.loss = self.on_policy_loss + self.auxiliary_loss 
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        #self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        self.optimiser = tf.train.AdamOptimizer(lr)

        # weights = self.train_policy.weights
        # grads = tf.gradients(self.on_policy_loss, weights)
        # grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        # grads_vars = list(zip(grads, weights))
        # self.train_policy_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)
    
    def _build_pixel(self, input):
        # ignoring cropping from paper hence deconvoluting to size 21x21 feature map (as 84x84 / 4 == 21x21)
        feat_map = mlp_layer(input, 32*8*8, activation=tf.nn.relu, name='feat_map_flat')
        feat_map = tf.reshape(feat_map, shape=[-1,8,8,32], name='feature_map')
        batch_size = tf.shape(feat_map)[0]
        deconv1 = conv_transpose_layer(feat_map, output_shape=[batch_size,10,10,32], kernel_size=[3,3], strides=[1,1], padding='VALID', activation=tf.nn.relu)
        deconv_advantage = conv2d_transpose(deconv1, output_shape=[batch_size,21,21,self.action_size],
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_adv')
        deconv_value = conv2d_transpose(deconv1, output_shape=[batch_size,21,21,1],
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=None, name='deconv_value')

        # Auxillary Q value calculated via dueling network 
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        Qaux = tf.nn.relu(deconv_value + deconv_advantage - tf.reduce_mean(deconv_advantage, axis=3, keep_dims=True))
        print('Qaux', Qaux.get_shape().as_list())
        return Qaux

    def forward(self, state, hidden, action_reward, validate=False):
        if validate:
            return self.validate_policy.forward(state, hidden, action_reward)
        else: 
            return self.train_policy.forward(fold_batch(state), hidden, action_reward)

    def forward_all(self, state, hidden, action_reward):
        mask = np.zeros((1, state.shape[0]))
        feed_dict = {self.train_policy.state:state, self.train_policy.hidden_in[0]:hidden[0],
         self.train_policy.hidden_in[1]:hidden[1], self.train_policy.mask:mask,
         self.train_policy.action_reward:action_reward}
        return self.sess.run([self.train_policy.policy_distrib, self.train_policy.V, self.train_policy.hidden_out, self.Qaux_batch], feed_dict=feed_dict)
    
    def get_pixel_control(self, state, hidden, action_reward):
        mask = np.zeros((1, state.shape[0]))
        feed_dict = {self.replay_policy.state:state, self.replay_policy.hidden_in[0]:hidden[0],
         self.replay_policy.hidden_in[1]:hidden[1], self.replay_policy.mask:mask,
         self.replay_policy.action_reward:action_reward}
        return self.sess.run(self.Qaux, feed_dict=feed_dict)
    
    # def A2Cbackprop(self, states, R, actions, hidden, dones, action_reward):
    #     feed_dict = {self.train_policy.state:states, self.train_policy.actions:actions, self.train_policy.R:R,
    #     self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
    #     self.train_policy.mask:dones, self.train_policy.action_reward:action_reward}
    #     _, l = self.sess.run([self.train_policy_op, self.on_policy_loss], feed_dict=feed_dict)
    #     return l

    # def Auxbackprop(self, reward_states, rewards, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones, action_reward):
    #     feed_dict = {self.Qaux_target:Qaux_target, self.Qaux_actions:Qaux_actions,
    #                 self.reward_state:reward_states, self.reward_target:rewards,
    #                 self.train_policy.state:replay_states, self.train_policy.R:replay_R, self.train_policy.actions:Qaux_actions,
    #                 self.train_policy.hidden_in[0]:replay_hidden[0], self.train_policy.hidden_in[1]:replay_hidden[1],
    #                 self.train_policy.mask:replay_dones, self.train_policy.action_reward:action_reward}
        
    #     _, l = self.sess.run([self.train_Aux_op, self.auxiliary_loss], feed_dict=feed_dict)
    #     return l

    def backprop(self, states, R, actions, hidden, dones, action_reward, 
                    reward_states, reward, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_dones, replay_action_reward):
        feed_dict = {self.train_policy.state:states, self.train_policy.actions:actions, self.train_policy.R:R,
            self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
            self.train_policy.mask:dones, self.train_policy.action_reward:action_reward,
            self.Qaux_target:Qaux_target, self.Qaux_actions:Qaux_actions,
            self.reward_state:reward_states, self.reward_target:reward,
            self.replay_policy.state:replay_states, self.replay_policy.R:replay_R,
            self.replay_policy.hidden_in[0]:replay_hidden[0], self.replay_policy.hidden_in[1]:replay_hidden[1],
            self.replay_policy.mask:replay_dones, self.replay_policy.action_reward:replay_action_reward}
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)

    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class Unreal_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, validate_freq=1000000, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        
        self.replay = deque([], maxlen=2000)
        self.runner = self.Runner(self.model, self.env, self.nsteps, self.replay)

        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':model.entropy_coeff, 'value_coefficient':0.5}
        
        if log_scalars:
            filename = log_dir + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def populate_memory(self):
        for t in range(2000//self.nsteps):
            self.runner.run()
    
    def auxiliary_target(self, prev_state, states, values, dones):
        T = len(states)
        #print('values shape', values.shape)
        R = np.zeros((T,*values.shape))
        dones = dones[:,np.newaxis,np.newaxis]
        pixel_rewards = np.zeros_like(R)
        pixel_rewards[0] = np.abs((states[0]/255) - (prev_state/255)).reshape(-1,21,4,21,4,3).mean(axis=(2,4,5))
        for i in range(1,T):
            pixel_rewards[i] = np.abs((states[i]/255) - (states[i-1]/255)).reshape(-1,21,4,21,4,3).mean(axis=(2,4,5))
        #print('pixel reward, max', pixel_rewards.max(), 'mean', pixel_rewards.mean())
        
        R[-1] = values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = pixel_rewards[i] + 0.99 * R[i+1] * (1-dones[-1])
        
        return R

    def sample_replay(self):
        sample_start = np.random.randint(0, len(self.replay) -self.nsteps -2)
        worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        if self.replay[sample_start][6][worker] == True:
            sample_start += 2
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
            if self.replay[sample_start][6][worker] == True:
                break
                
        replay_states = np.stack([replay_sample[i][0][worker] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1][worker] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2][worker] for i in range(len(replay_sample))])
        replay_hiddens = np.stack([replay_sample[i][3]for i in range(len(replay_sample))])
        replay_actsrews = np.stack([replay_sample[i][4][worker] for i in range(len(replay_sample))])
        replay_Qauxs = np.stack([replay_sample[i][5][worker] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][6][worker] for i in range(len(replay_sample))])
        #print('replay_hiddens dones shape', replay_dones.shape)
        
        next_state = self.replay[sample_start+self.nsteps][0][worker][np.newaxis] # get state 
        _, replay_values, *_ = self.model.forward(next_state, replay_hiddens[-1,:,worker].reshape(2,1,-1), replay_actsrews[-1][np.newaxis,np.newaxis], validate=True)
        replay_R = self.nstep_return(replay_rewards, replay_values, replay_dones)

        prev_states = self.replay[sample_start-1][0][worker]
        Qaux_value = self.model.get_pixel_control(next_state, replay_hiddens[-1,:,worker].reshape(2,1,-1), replay_actsrews[-1][np.newaxis,np.newaxis])[0]
        #print('Qaux_value shape', Qaux_value.shape)
        Qaux_target = self.auxiliary_target(prev_states, replay_states, np.max(Qaux_value, axis=-1), replay_dones)
        #print('Qaux target shape', Qaux_target.shape)
        
        return replay_states, replay_actions, replay_R, Qaux_target, \
                     replay_hiddens[:,:,worker][:,:,np.newaxis], replay_actsrews[:,np.newaxis], replay_dones[:,np.newaxis]
    
    def sample_reward(self):
       # worker = np.random.randint(0,self.num_envs) # randomly sample from one of n workers
        
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])[3:]
        worker = np.argmax(np.sum(replay_rewards, axis=0)) # sample experience from best worker
        #replay_states = np.array([self.replay[i][0][worker] for i in range(len(self.replay))])[3:]
        #print('replay_rewards', replay_rewards.shape)
        #print('replay_states', replay_states.shape)
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            #print('nonzero')
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        reward_states = np.stack([self.replay[i][0][worker] for i in range(idx-3,idx)])
        #reward_states = np.stack([replay_states[i] for i in range(idx-3,idx)])
        #sign = int(np.sign(self.replay[idx][2][worker]))
        sign = int(np.sign(replay_rewards[idx,worker]))
        reward = np.zeros((1,3))
        reward[0,sign] = 1 # catergorical [zero, positive, negative]
    
        return reward_states, reward
    
    def _train_nstep(self):
        batch_size = (self.num_envs * self.nsteps)
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        #self.validate(self.val_envs[0], 1, 1000)
        self.populate_memory()
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, hidden_batch, prev_acts_rewards, Qauxs, dones, infos, last_values = self.runner.run()

            R = self.nstep_return(rewards, last_values, dones, clip=False)
            # stack all states, actions and Rs across all workers into a single batch
            states, actions, rewards, R = fold_batch(states), fold_batch(actions), fold_batch(rewards), fold_batch(R)
            
            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_hiddens, replay_actsrews, replay_dones = self.sample_replay()
            
            l = self.model.backprop(states, R, actions, hidden_batch[0], dones, prev_acts_rewards,
                reward_states, sample_rewards, Qaux_target, replay_actions, replay_states, replay_R, replay_hiddens[0], replay_dones, replay_actsrews)
            
            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq //batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + '/' + str(s) + ".ckpt") )
                print('saved model')


    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps, replay):
            super().__init__(model, env, num_steps)
            self.replay = replay
            self.action_size = self.model.action_size
            self.prev_hidden = self.model.get_initial_hidden(len(self.env))
            zeros = np.zeros((len(self.env)), dtype=np.int32)
            self.prev_actions_rewards = concat_action_reward(zeros, zeros, self.action_size+1) # start with action 0 and reward 0 

        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values, hidden, Qaux = self.model.forward_all(self.states, self.prev_hidden, self.prev_actions_rewards[np.newaxis])
                #Qaux = self.model.get_pixel_control(self.states, self.prev_hidden, self.prev_actions_rewards[np.newaxis])
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)

                rollout.append((self.states, actions, rewards, self.prev_hidden, self.prev_actions_rewards, Qaux, dones, infos))
                self.replay.append((self.states, actions, rewards, self.prev_hidden, self.prev_actions_rewards, Qaux, dones, infos)) # add to replay memory
                self.states = next_states
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode
                self.prev_actions_rewards = concat_action_reward(actions , rewards, self.action_size+1)
            
            states, actions, rewards, hidden_batch, prev_actions_rewards, Qaux, dones, infos = zip(*rollout)
            states, actions, rewards, prev_actions_rewards, Qaux, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(prev_actions_rewards), np.stack(Qaux), np.stack(dones)
            _, last_values, _ = self.model.forward(self.states[np.newaxis], self.prev_hidden, self.prev_actions_rewards[np.newaxis])
            return states, actions, rewards, hidden_batch, prev_actions_rewards, Qaux, dones, infos, last_values

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
            prev_actrew = concat_action_reward(np.zeros((1),dtype=np.int32), np.zeros((1),dtype=np.int32), self.model.action_size+1)
            #print(' prev_action size', prev_actrew.shape)
            for t in range(max_steps):
                policy, value, hidden = self.model.forward(state[np.newaxis], hidden, prev_actrew[np.newaxis], validate=True)
                #print('policy', policy, 'value', value)
                action = np.random.choice(policy.shape[1], p=policy[0])
                next_state, reward, done, info = env.step(action)
                state = next_state
                prev_actrew = concat_action_reward(np.array(action)[np.newaxis], np.array(reward)[np.newaxis], self.model.action_size+1)
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

def main(env_id, Atari=True):
    print('gpu aviabliable', tf.test.is_gpu_available())

    num_envs = 32
    nsteps = 20

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
        
        val_envs = [AtariEnv_(gym.make(env_id), k=1, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv_, env_id, num_envs, blocking=False, k=1, reset=reset, episodic=True, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Unreal/' + env_id + '/'
    model_dir = "models/Unreal/" + env_id + '/'



    model = UnrealA2C(nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      cell_size = 256,
                      num_envs = num_envs,
                      PC=0.01,
                      entropy_coeff=0.001,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={})

    

    auxiliary = Unreal_Trainer(envs = envs,
                                  model = model,
                                  model_dir=model_dir,
                                  log_dir=train_log_dir,
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  total_steps = 50e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e6,
                                  save_freq = 0,
                                  render_freq = 0,
                                  num_val_episodes = 50,
                                  log_scalars = True)

    
    
    

    auxiliary.train()

    del auxiliary

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['PrivateEyeDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'PongDeterministic-v4' ]
    #env_id_list = ['MountainCar-v0','CartPole-v1']
    for env_id in env_id_list:
        main(env_id)
    