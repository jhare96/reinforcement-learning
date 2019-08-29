import tensorflow as tf
import numpy as np
import threading, multiprocessing
import queue
import gym
import sys

from rlib.utils.VecEnv import*
from rlib.utils.utils import one_hot, fold_batch
from rlib.networks.networks import*

main_lock = threading.Lock()

def concat_action_reward(actions, rewards, num_classes):
    concat = one_hot(actions, num_classes)
    concat[:,-1] = rewards   
    return concat  

def AtariEnv_(env, k=4, episodic=True, reset=True, clip_reward=True, Noop=True):
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
    return env


class ActorCritic_LSTM(object):
    def __init__(self, model_head, input_shape, action_size, num_envs, cell_size,
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
            self.lstm_output, self.hidden_in, self.hidden_out = lstm(unfolded_state, cell_size=cell_size, fold_output=True, time_major=True)

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

        self.loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy

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
        feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.action_reward:action_reward}
        policy, value, hidden = self.sess.run([self.policy_distrib, self.V, self.hidden_out], feed_dict = feed_dict)
        return policy, value, hidden

    def backprop(self, state, y, a, hidden, dones):
        feed_dict = {self.state : state, self.R : R, self.actions: a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.action_reward:action_reward}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess


class Unreal_LSTM(object):
    def __init__(self,  policy_model, input_shape, action_size, cell_size, RP=1.0, PC=1.0, VR=1.0, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.RP, self.PC, self.VR = RP, PC, VR
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        print('action_size', action_size)

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)
            self.replay_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)

        with tf.variable_scope('pixel_control', reuse=tf.AUTO_REUSE):
            #self.Qaux_batch = self._build_pixel(self.train_policy.lstm_output)
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
            reward_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=pred_reward, onehot_labels=self.reward_target))  # cross entropy over caterogical reward ]
            print('reward loss ', reward_loss)
            
        
        

        self.on_policy_loss = self.train_policy.loss
        self.auxiliary_loss = PC * pixel_loss + RP * reward_loss +  VR * replay_loss 
        self.loss = self.on_policy_loss + self.auxiliary_loss 
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)

        # weights = self.train_policy.weights
        # grads = tf.gradients(self.on_policy_loss, weights)
        # grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        # grads_vars = list(zip(grads, weights))
        # self.train_policy_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

        self.weights = weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
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
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=tf.nn.relu, name='deconv_adv')
        deconv_value = conv2d_transpose(deconv1, output_shape=[batch_size,21,21,1],
                kernel_size=[3,3], strides=[2,2], padding='VALID', activation=tf.nn.relu, name='deconv_value')

        # Auxillary Q value calculated via dueling network 
        # Z. Wang, N. de Freitas, and M. Lanctot. Dueling Network Architectures for Deep ReinforcementLearning. https://arxiv.org/pdf/1511.06581.pdf
        Qaux = deconv_value + deconv_advantage - tf.reduce_mean(deconv_advantage, axis=3, keep_dims=True)
        print('Qaux', Qaux.get_shape().as_list())
        return Qaux

    def forward(self, state, hidden, action_reward):
        return self.train_policy.forward(state, hidden, action_reward)

    def forward_all(self, state, hidden, action_reward):
    
        feed_dict = {self.train_policy.state:state, self.train_policy.hidden_in[0]:hidden[0],
         self.train_policy.hidden_in[1]:hidden[1],
         self.train_policy.action_reward:action_reward}
        return self.sess.run([self.train_policy.policy_distrib, self.train_policy.V, self.train_policy.hidden_out], feed_dict=feed_dict)
    
    def get_pixel_control(self, state, hidden, action_reward):
        
        feed_dict = {self.replay_policy.state:state, self.replay_policy.hidden_in[0]:hidden[0],
         self.replay_policy.hidden_in[1]:hidden[1], 
         self.replay_policy.action_reward:action_reward}
        return self.sess.run(self.Qaux, feed_dict=feed_dict)


    def backprop(self, states, R, actions, hidden, action_reward, 
                    reward_states, reward, Qaux_target, Qaux_actions, replay_states, replay_R, replay_hidden, replay_action_reward):
        feed_dict = {self.train_policy.state:states, self.train_policy.actions:actions, self.train_policy.R:R,
            self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
             self.train_policy.action_reward:action_reward,
            self.Qaux_target:Qaux_target, self.Qaux_actions:Qaux_actions,
            self.reward_state:reward_states, self.reward_target:reward,
            self.replay_policy.state:replay_states, self.replay_policy.R:replay_R,
            self.replay_policy.hidden_in[0]:replay_hidden[0], self.replay_policy.hidden_in[1]:replay_hidden[1],
            self.replay_policy.action_reward:replay_action_reward}
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)

    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)


class Worker(threading.Thread):
    def __init__(self, ID, sess, T, episode, device, queue, lock, env_constr, model_constr, daemon, env_args={}, model_args={}, nsteps=20, max_rollouts=5e6):
        threading.Thread.__init__(self, daemon=daemon)
        self.sess = sess
        self.queue = queue
        self.lock = lock
        self.nsteps = nsteps
        self.max_rollouts = int(max_rollouts)
        self.env = env_constr(**env_args)
        self.action_size = self.env.action_space.n
        if 'gpu' in device.lower():
            with tf.variable_scope('worker_' + str(ID)):
                self.model = model_constr(**model_args)
        else:
            with tf.device(device):
                with tf.variable_scope('worker_' + str(ID)):
                    self.model = model_constr(**model_args)
        
        self.model.set_session(self.sess)
        self.workerID = ID
        self.T = T
        self.episode = episode
        # self.update_local = [tf.assign(new, old) for (new, old) in 
        #     zip(tf.trainable_variables(scope='worker_' + str(self.workerID) ), tf.trainable_variables('master'))]
        
        network_weights =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='master')
        self.replay = deque([], maxlen=2000)
        self.update_local = [v1.assign(v2) for v1, v2 in zip(self.model.weights, network_weights)]
    
    def run(self):
        self.train()
    
    def populate_memory(self):
        state = self.env.reset()
        prev_hidden = self.model.get_initial_hidden(1)
        prev_action_reward = np.zeros((1, self.action_size+1))
        for t in range(200):
            #print('state', state.dtype, 'h', prev_hidden.dtype)
            policy, value, hidden = self.model.forward(state[np.newaxis], prev_hidden, prev_action_reward[np.newaxis])
            Qaux = self.model.get_pixel_control(state[np.newaxis], prev_hidden, prev_action_reward[np.newaxis])
            action = np.random.choice(policy.shape[1], p=policy[0])
            next_state, reward, done, info = self.env.step(action)()
            
            self.replay.append((state, action, reward, prev_hidden, prev_action_reward, Qaux, done)) # add to replay memory
            states_ = next_state
            prev_hidden = hidden
            prev_action_reward = np.zeros((1, self.action_size+1))
            prev_action_reward[0,action] = 1
            prev_action_reward[0,-1] = reward


            if done:
                state = self.env.reset()
                prev_hidden = self.model.get_initial_hidden(1)

    def multistep_target(self, rewards, values, dones, clip=False):
        if clip:
            rewards = np.clip(rewards, -1, 1)

        T = len(rewards)
        
        # Calculate R for advantage A = R - V 
        R = np.zeros((T))
        R[-1] = values * (1-dones[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])
        
        return R
    
    def auxiliary_target(self, prev_state, states, values, dones):
        T = len(states)
        #print('states shape', states.shape)
        R = np.zeros((T,*values.shape[1:]))
        #print('R shape', R.shape)
        dones_= np.zeros_like(R)
        pixel_rewards = np.zeros_like(R)
        pixel_rewards[0] = np.abs((states[0]/256.0) - (prev_state/256.0)).reshape(21,4,21,4,3).mean(axis=(1,3,4))
        for i in range(1, T):
            dones_[i] = dones[i]
            pixel_rewards[i] = np.abs((states[i]/256.0) - (states[i-1]/256.0)).reshape(21,4,21,4,3).mean(axis=(1,3,4))
        #dones = np.stack([dones for i in range(values.shape[1:])], axis=-1)
        #rewards = np.stack([rewards for i in range(values.shape[1:])], axis=-1)
        #print('R shape', R.shape)
        #print('stack_shape', dones.shape)
        R[-1] = values * (1-dones_[-1])
        
        for i in reversed(range(T-1)):
            # restart score if done as BatchEnv automatically resets after end of episode
            R[i] = pixel_rewards[i] + 0.99 * R[i+1] * (1-dones_[-1])
        
        return R
    
    def sample_replay(self):
        sample_start = np.random.randint(0, len(self.replay) -21)
        if self.replay[sample_start][6] == True:
            sample_start += 2
        replay_sample = []
        for i in range(sample_start, sample_start+self.nsteps):
            replay_sample.append(self.replay[i])
        replay_states = np.stack([replay_sample[i][0] for i in range(len(replay_sample))])
        replay_actions = np.stack([replay_sample[i][1] for i in range(len(replay_sample))])
        replay_rewards = np.stack([replay_sample[i][2] for i in range(len(replay_sample))])
        replay_hiddens = np.stack([replay_sample[i][3] for i in range(len(replay_sample))])
        replay_actsrews = np.stack([replay_sample[i][4] for i in range(len(replay_sample))])
        replay_Qauxs = np.stack([replay_sample[i][5] for i in range(len(replay_sample))])
        replay_dones = np.stack([replay_sample[i][6] for i in range(len(replay_sample))])

        _, replay_values, *_ = self.model.forward_all(replay_states[-1][np.newaxis], replay_hiddens[-1], replay_actsrews[-1][np.newaxis])
        replay_R = self.multistep_target(replay_rewards, replay_values, replay_dones)
    
        Qaux_target = self.auxiliary_target(self.replay[sample_start-1][0], replay_states, np.max(replay_Qauxs[-1], axis=-1), replay_dones)
        #print('Qaux target shape', Qaux_target.shape)
        
        return replay_states, replay_actions, replay_R, Qaux_target, \
                     replay_hiddens, replay_actsrews, replay_dones
    
    def sample_reward(self):
        replay_rewards = np.array([self.replay[i][2] for i in range(len(self.replay))])[3:]
        #print('replay_rewards', replay_rewards.shape)
        nonzero_idxs = np.where(np.abs(replay_rewards) > 0)[0] # idxs where |reward| > 0 
        zero_idxs = np.where(replay_rewards == 0)[0] # idxs where reward == 0 
        
        #print('zerpo', zero_idxs.shape)
        #print('nonzero_idxs', nonzero_idxs.shape)
        if len(nonzero_idxs) ==0 or len(zero_idxs) == 0: # if nonzero or zero idxs do not exist i.e. all rewards same sign 
            idx = np.random.randint(len(replay_rewards))
        elif np.random.uniform() > 0.5: # sample from zero and nonzero rewards equally
            idx = np.random.choice(nonzero_idxs)
        else:
            idx = np.random.choice(zero_idxs)
        
        
        replay_states = np.stack([self.replay[i][0] for i in range(idx-3,idx)])
        sign = int(np.sign(self.replay[idx][2]))
        replay_reward = np.zeros((1,3))
        replay_reward[0,sign] = 1 # catergorical [zero, positive, negative]
        return replay_states, replay_reward
    
    def train(self):
        prev_hidden = self.model.get_initial_hidden(1)
        env = self.env
        state = env.reset()
        self.populate_memory()
        episode = 0
        epsiode_reward = []
        loss = 0
        frames = 0
        frames_start = time.time()
        while True:
            rollout = []
            start = time.time()
            prev_action_reward = np.zeros((1, self.action_size+1))
            for t in range(self.nsteps):
                #print('state', state.dtype, 'h', prev_hidden.dtype)
                policy, value, hidden = self.model.forward(state[np.newaxis], prev_hidden, prev_action_reward[np.newaxis])
                Qaux = self.model.get_pixel_control(state[np.newaxis], prev_hidden, prev_action_reward[np.newaxis])
                action = np.random.choice(policy.shape[1], p=policy[0])

                next_state, reward, done, info = env.step(action)()
                epsiode_reward.append(reward)
            
                rollout.append((state, action, reward, prev_hidden, prev_action_reward, done))
                self.replay.append((state, action, reward, prev_hidden, prev_action_reward, Qaux, done)) # add to replay memory
                states_ = next_state
                prev_hidden = hidden
                prev_action_reward = np.zeros((1, self.action_size+1))
                prev_action_reward[0,action] = 1
                prev_action_reward[0,-1] = reward

                self.T[:] += 1
                frames += 1
                
                if done:
                    state = env.reset()
                    prev_hidden = self.model.get_initial_hidden(1)
                    time_taken = time.time() - frames_start
                    fps = frames / time_taken
                    if episode % 1 == 0 and episode > 0:
                        print('worker %i, episode %i, total_steps %i, episode reward %f, loss %f, local fps %f' %(self.workerID, episode, self.T, np.sum(epsiode_reward),loss, fps))
                    epsiode_reward = []
                    frames_start = time.time()
                    frames = 0
                    self.episode[:] += 1
                    break
            
            end = time.time()
            #print('nsteps time', end-start)
            states, actions, rewards, hidden_batch, prev_actions_rewards, dones = zip(*rollout)

            states = np.stack(states)
            actions = np.stack(actions)
            rewards = np.stack(rewards)
            #next_states = np.stack(next_states)
            dones = np.stack(dones)
            prev_actions_rewards = np.stack(prev_actions_rewards)
            rewards = np.clip(rewards, -1, 1)
            T = rewards.shape[0]
            
            # Calculate R for advantage A = R - V 
            R = np.zeros((T), dtype=np.float32)
            v = value.reshape(-1)
    
            R[-1] =  v * (1-dones[-1])
            
            for i in reversed(range(T-1)):
                # restart score if done as wrapped env continues after end of episode
                R[i] = rewards[i] + 0.99 * R[i+1] * (1-dones[i])  
            
            
            reward_states, sample_rewards = self.sample_reward()
            replay_states, replay_actions, replay_R, Qaux_target, replay_hiddens, replay_actsrews, replay_dones = self.sample_replay()
            
            l = self.model.backprop(states, R, actions, hidden_batch[0], prev_actions_rewards,
                reward_states, sample_rewards, Qaux_target, replay_actions, replay_states, replay_R, replay_hiddens[0], replay_actsrews)
            #loss = self.model.backprop(states, R, actions, hidden_batch[0])

            
            #print('backprop time', end2-start2)

            start3 = time.time()
            self.sess.run(self.update_local)
            end3 = time.time()
            #print('set weights time', end3-start3)



def actor_constructor(policy_model, **model_args):
    return Unreal_LSTM(policy_model, **model_args)

def env_constructor(env_constr, env_id, **env_args):
    return Env(env_constr(gym.make(env_id), **env_args))

def main():
    config = tf.compat.v1.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.compat.v1.Session(config=config)


    env_id = 'FreewayDeterministic-v4'
    train_log_dir = 'logs/A3C_Thread/' + env_id + '/'

    env = gym.make(env_id)
    action_size = env.action_space.n
    #input_shape = env.reset().shape
    input_shape = [84,84,3]
    env.close()

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.polynomial_decay(1e-3, global_step, 10e6, end_learning_rate=0, power=1.0, cycle=False, name=None)
    #
    # model, optimiser, global_step, input_shape, action_size, dense_size, cell_size, grad_clip=0.5, **model_args)
    # optimiser = tf.train.RMSPropOptimizer(7e-4, decay=0.99, epsilon=1e-3)
    model_args = {'policy_model':nips_cnn, 'action_size':action_size, 'input_shape':input_shape, 'cell_size':256, 'grad_clip':0.5, 'lr':1e-3, 'PC':0.01}
    with tf.device('/cpu:0'):
        with tf.variable_scope('master'):
            master = Unreal_LSTM(**model_args)

    num_workers = 8
    T = np.array([0], dtype=np.int32)
    episode = np.array([0], dtype=np.int32)

    env_args = {'env_constr':AtariEnv_, 'env_id':env_id, 'k':1, 'clip_reward':True, 'reset':False}
    env_constr = env_constructor
    model_constr = actor_constructor

    lock = threading.Lock()
    #self, ID, sess, T, device, queue, lock, env_constr, model_constr, daemon, env_args={}, model_args={}, nsteps=20, max_rollouts=5e6)
    workers = [Worker(i, sess, T, episode, '/cpu:0', None, lock, env_constr, model_constr, True, env_args, model_args) for i in range(num_workers)]


    init = tf.global_variables_initializer()
    sess.run(init)

    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    try:
        for w in workers:
            w.start()
            time.sleep(0)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.env.close()
            w.join()


if __name__ == "__main__":
    main()