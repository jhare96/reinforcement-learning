import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from rlib.A2C.A2C import ActorCritic
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.utils import fold_batch, one_hot, rolling_stats, stack_many, RunningMeanStd

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

class rolling_obs(object):
    def __init__(self, shape=()):
        self.rolling = RunningMeanStd(shape=shape)
    
    def update(self, x):
        if len(x.shape) == 5: # assume image obs 
            return self.rolling.update(fold_batch(x[...,-1:])) #[time,batch,height,width,stack] -> [height, width,1]
        else:
            return self.rolling.update(fold_batch(x)) #[time,batch,*shape] -> [*shape]


# class RunningMeanStd(object):
#     # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
#     def __init__(self, epsilon=1e-4, shape=()):
#         self.mean = np.zeros(shape, 'float64')
#         self.var = np.ones(shape, 'float64')
#         self.count = epsilon

#     def update(self, x):
#         batch_mean = np.mean(x, axis=0)
#         batch_var = np.var(x, axis=0)
#         batch_count = x.shape[0]
#         self.update_from_moments(batch_mean, batch_var, batch_count)

#     def update_from_moments(self, batch_mean, batch_var, batch_count):
#         delta = batch_mean - self.mean
#         tot_count = self.count + batch_count

#         new_mean = self.mean + delta * batch_count / tot_count
#         m_a = self.var * (self.count)
#         m_b = batch_var * (batch_count)
#         M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
#         new_var = M2 / (self.count + batch_count)

#         new_count = batch_count + self.count

#         self.mean = new_mean
#         self.var = new_var
#         self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems     

class Forward_Model(object):
    def __init__(self, forward_model, forward_decoder, input_shape, action_size, lr, cell_size, num_envs, nsteps):
        self.sess = None
        self.num_envs, self.cell_size = num_envs, cell_size

        self.init_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='state')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.optimiser = tf.train.AdamOptimizer(lr)
        with tf.variable_scope('Forward_model'):
            with tf.variable_scope('encoder_model'):
                self.f1 = forward_model(self.init_state)
            
            with tf.variable_scope('latent-space-rnn'):
                lstm_input = mlp_layer(self.f1, cell_size, activation=tf.nn.relu, name='lstm_in')
                self.cell = tf.nn.rnn_cell.LSTMCell(cell_size)
                cell_in = tf.placeholder(tf.float32, shape=[num_envs, cell_size])
                hidden_in = tf.placeholder(tf.float32, shape=[num_envs, cell_size])
                self.hidden = (cell_in, hidden_in)
                state_in = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(cell_in, hidden_in)

                self.num_steps = tf.placeholder(shape=[None], dtype=tf.float32)

                lstm_output, self.hidden_out = one_to_many_rnn(self.cell, lstm_input, state_in, self.num_steps)
                self.lstm_output = tf.reshape(lstm_output, shape=[-1, cell_size])

            with tf.variable_scope('reward_model'):
                self.pred_reward = mlp_layer(tf.concat([self.lstm_output,tf.one_hot(self.action, action_size)], axis=1), 1, activation=None)

            with tf.variable_scope('decoder_model'):
                self.pred_next_state = forward_decoder(self.lstm_output)
            
                
            with tf.variable_scope('forward_losses'):
                self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='next_state')
                self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
                forward_loss = tf.reduce_mean(tf.square(self.pred_next_state - self.next_state))
                reward_loss = tf.reduce_mean(tf.square(self.pred_reward - self.reward))
                self.loss = forward_loss + reward_loss
        self.train_op = self.optimiser.minimize(self.loss)

    def predict_rollout(self, init_state, actions, num_steps):
        hidden = np.zeros((self.num_envs, self.cell_size))
        feed_dict = {self.init_state:init_state, self.hidden[0]:hidden,
                     self.hidden[1]:hidden, self.action:actions,
                     self.num_steps:np.zeros((num_steps))}
        return self.sess.run(self.pred_next_state, feed_dict=feed_dict)

    def encode_state(self, init_state):
        return self.sess.run(self.f1, {self.init_state:init_state})
    
    def predict_next(self, encoded_last_state, hidden, actions, num_steps=1):
        feed_dict = {self.f1:encoded_last_state, self.hidden:hidden, self.action:actions, self.num_steps:np.zeros((num_steps))}
        return self.sess.run([self.pred_next_state, self.pred_reward, self.f1, self.hidden], feed_dict=feed_dict)
    
    def backprop(self, init_state, next_states, actions, rewards, num_steps):
        hidden = np.zeros((self.num_envs, self.cell_size))
        return self.sess.run([self.train_op, self.loss],{self.init_state:init_state, self.action:actions, 
                                                         self.next_state:next_states, self.reward:rewards,
                                                         self.hidden[0]:hidden, self.hidden[1]:hidden,
                                                         self.num_steps:np.zeros((num_steps))})
    def get_initial_hidden(self, batch_size):
        return np.zeros((batch_size, self.cell_size)), np.zeros((batch_size, self.cell_size))
    
    def set_session(self, sess):
        self.sess = sess



class PPO(object):
    def __init__(self, model, input_shape, action_size, value_coeff=1.0, entropy_coeff=0.001, extr_coeff=2.0, intr_coeff=1.0, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5,  name='PPO', optimiser=False, **model_args):
        self.lr, self.lr_final = lr, lr_final
        self.value_coeff, self.entropy_coeff = value_coeff, entropy_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.policy_clip = 0.1
        self.sess = None
        with tf.variable_scope(name):
            with tf.variable_scope('encoder_network'):
                self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
                print('state shape', self.state.get_shape().as_list())
                self.dense = model(self.state, **model_args)
            
            with tf.variable_scope('extr_critic'):
                self.Ve = tf.reshape(mlp_layer(self.dense, 1, name='extr_value', activation=None), shape=[-1])
            
            with tf.variable_scope('intr_critic'):
                self.Vi = tf.reshape(mlp_layer(self.dense, 1, name='intr_value', activation=None), shape=[-1])
            
            with tf.variable_scope("actor"):
                self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution')
                self.actions = tf.placeholder(tf.int32, [None])
                actions_onehot = tf.one_hot(self.actions,action_size)
                
            with tf.variable_scope('losses'):
                self.old_policy = tf.placeholder(dtype=tf.float32, shape=[None, action_size], name='old_policies') + 1e-10
                self.alpha = tf.placeholder(dtype=tf.float32, shape=[], name='alpha')
                self.R_extr = tf.placeholder(dtype=tf.float32, shape=[None])
                self.R_intr = tf.placeholder(dtype=tf.float32, shape=[None])

                extr_value_loss = 0.5 * tf.reduce_mean(tf.square(self.R_extr - self.Ve))
                intr_value_loss = 0.5 * tf.reduce_mean(tf.square(self.R_intr - self.Vi))

                policy_actions = tf.reduce_sum(tf.multiply(self.policy_distrib, actions_onehot), axis=1)
                old_policy_actions = tf.reduce_sum(tf.multiply(self.old_policy, actions_onehot), axis=1)
                
                self.Advantage = tf.placeholder(dtype=tf.float32, shape=[None], name='Adv')

                ratio = policy_actions / old_policy_actions

                policy_loss_unclipped = ratio * -self.Advantage
                policy_loss_clipped = tf.clip_by_value(ratio, 1 - self.policy_clip , 1 + self.policy_clip) * -self.Advantage

                policy_loss = tf.reduce_mean(tf.math.maximum(policy_loss_unclipped, policy_loss_clipped))

                entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -tf.math.log(self.policy_distrib), axis=1))
        
            self.loss =  policy_loss + value_coeff * (extr_value_loss + intr_value_loss) - entropy_coeff * entropy
            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            
            if optimiser:
                global_step = tf.Variable(0, trainable=False)
                lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
                optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
                grads = tf.gradients(self.loss, self.weights)
                grads, _ = tf.clip_by_global_norm(grads, grad_clip)
                grads_vars = list(zip(grads, self.weights))
                self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)

    def forward(self, state):
        return self.sess.run([self.policy_distrib, self.Ve, self.Vi], feed_dict = {self.state:state})

    def get_policy(self, state):
        return self.sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_values(self, state):
        return self.sess.run([self.Ve, self.Vi] , feed_dict = {self.state: state})

    def backprop(self, state, R_extr, R_intr, Adv, a, old_policy, alpha):
        feed_dict = {self.state : state, self.R_extr:R_extr, self.R_intr:R_intr,
                     self.Advantage:Adv, self.actions:a,
                     self.old_policy:old_policy, self.alpha:alpha}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess

def predictor_cnn(x, conv1_size=32 ,conv2_size=64, conv3_size=64, dense_size=512, padding='VALID', activation=tf.nn.leaky_relu, init_scale=np.sqrt(2), trainable=True):
    h1 = conv2d(x,  output_channels=conv1_size, kernel_size=[8,8],  strides=[4,4], padding=padding, kernel_initialiser=tf.orthogonal_initializer(init_scale), activation=activation, dtype=tf.float32, name='conv_1', trainable=trainable)
    h2 = conv2d(h1, output_channels=conv2_size, kernel_size=[4,4],  strides=[2,2], padding=padding, kernel_initialiser=tf.orthogonal_initializer(init_scale), activation=activation, dtype=tf.float32, name='conv_2', trainable=trainable)
    h3 = conv2d(h2, output_channels=conv3_size, kernel_size=[3,3],  strides=[1,1], padding=padding, kernel_initialiser=tf.orthogonal_initializer(init_scale), activation=activation, dtype=tf.float32, name='conv_3', trainable=trainable)
    dense = flatten(h3)
    if trainable:
        dense = mlp_layer(dense, dense_size, activation=tf.nn.relu, weight_initialiser=tf.orthogonal_initializer(init_scale), name='dense1', trainable=trainable)
        dense = mlp_layer(dense, dense_size, activation=tf.nn.relu, weight_initialiser=tf.orthogonal_initializer(init_scale), name='dense2', trainable=trainable)
    dense = mlp_layer(dense, dense_size, activation=None, weight_initialiser=tf.orthogonal_initializer(init_scale), name='pred_state', trainable=trainable)
    return dense

def predictor_mlp(x, num_layers=2, dense_size=64, activation=tf.nn.leaky_relu, init_scale=np.sqrt(2), trainable=True):
    for i in range(num_layers):
        x = mlp_layer(x, dense_size, activation=activation, weight_initialiser=tf.orthogonal_initializer(init_scale), name='dense_' + str(i), trainable=trainable)
    dense = mlp_layer(dense, dense_size, activation=None, weight_initialiser=tf.orthogonal_initializer(init_scale), name='pred_state', trainable=trainable)
    return dense

class RND(object):
    def __init__(self, policy_model, target_model, input_shape, action_size, value_coeff=1.0, intr_coeff=0.5, extr_coeff=1.0, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, RND_args={}):
        self.intr_coeff, self.extr_coeff =  intr_coeff, extr_coeff
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.sess = None
        #self.pred_prob = 1

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        

        with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
            self.policy = PPO(policy_model, input_shape, action_size, value_coeff=value_coeff, intr_coeff=intr_coeff, extr_coeff=extr_coeff, lr=lr, lr_final=lr_final, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)
        
        if len(input_shape) == 3:
            next_state_shape = input_shape[:-1] + (1,)
        else: 
            next_state_shape = input_shape
        self.next_state = tf.placeholder(tf.float32, shape=[None, *next_state_shape], name='next_state')
        self.state_mean = tf.placeholder(tf.float32, shape=[*next_state_shape], name="mean")
        self.state_std = tf.placeholder(tf.float32, shape=[*next_state_shape], name="std")
        norm_next_state = tf.clip_by_value((self.next_state - self.state_mean) / self.state_std, -5, 5)

        with tf.variable_scope('target_model'):
            target_state = target_model(norm_next_state, trainable=False)
        
        with tf.variable_scope('predictor_model'):
            pred_next_state = target_model(norm_next_state, trainable=True)
            self.intr_reward = tf.reduce_mean(tf.square(pred_next_state - tf.stop_gradient(target_state)), axis=-1)
            feat_loss = tf.reduce_mean(self.intr_reward)

        self.loss = self.policy.loss + feat_loss


        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        #self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        self.optimiser = tf.train.AdamOptimizer(lr)
        
        weights = self.policy.weights + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='predictor_model')
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)
    

    def forward(self, state):
        return self.policy.forward(state)
    
    def intrinsic_reward(self, next_state, state_mean, state_std):
        feed_dict={self.next_state:next_state, self.state_mean:state_mean, self.state_std:state_std}
        forward_loss = self.sess.run(self.intr_reward, feed_dict=feed_dict)
        intr_reward = forward_loss
        return intr_reward
    
    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions, old_policy, alpha, state_mean, state_std):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.policy.state:state, self.policy.actions:actions,
                     self.next_state:next_state, self.state_mean:state_mean, self.state_std:state_std,
                     self.policy.R_extr:R_extr, self.policy.R_intr:R_intr, self.policy.Advantage:Adv,
                     self.policy.old_policy:old_policy, self.policy.alpha:alpha}

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)


class MA_RND_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, forward_model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, num_epochs=4, num_minibatches=4, validate_freq=1000000.0,
                 save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        self.runner = self.Runner(self.model, self.env, self.nsteps)
        self.alpha = 1
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.lambda_ = 0.95
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':0.001, 'value_coefficient':0.5, 'intr_coeff':model.intr_coeff,
        'extr_coeff':model.extr_coeff}
        
        self.forward_model = forward_model
        self.forward_model.set_session(self.sess)

        if log_scalars:
            filename = file_loc[1] + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            for t in range(max_steps):
                policy, value_extr, value_intr = self.model.forward(state[np.newaxis])
                #print('policy', policy, 'value_extr', value_extr)
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
    
    def init_state_obs(self, num_steps):
        rollout = []
        states = self.env.reset()
        for i in range(1,num_steps+1):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            #print('rand_actions.shape', rand_actions.shape)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            rollout.append([states, next_states, rand_actions, rewards])
            states = next_states
            if i % self.nsteps == 0:
                mb_states, mb_next_states, mb_actions, mb_rewards = stack_many(zip(*rollout))
                #print('states, next_states, actions, rewards', mb_states.shape, mb_next_states.shape, mb_actions.shape, mb_rewards.shape)
                self.runner.state_mean, self.runner.state_std = self.state_rolling.update(mb_states)
                self.forward_model.backprop(mb_states[0], fold_batch(mb_next_states), fold_batch(mb_actions), fold_batch(mb_rewards), len(mb_states))
                rollout = []
    
    def discount(self, rewards, gamma):
        discounted = 0
        for t in reversed(range(len(rewards))):
            discounted = gamma * discounted  + rewards[t]
        return discounted

    
    def _train_nstep(self):
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        alpha_step = 1/num_updates
        s = 0
        rolling = RunningMeanStd(shape=())
        self.state_rolling = rolling_obs(shape=())
        self.init_state_obs(129)
        #self.runner.state_mean, self.runner.state_std = self.state_rolling.mean, np.sqrt(self.state_rolling.var)
        self.runner.states = self.env.reset()
        forward_filter = RewardForwardFilter(self.gamma)

        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, old_policies, dones = self.runner.run()
            policy, extr_last_values, intr_last_values = self.model.forward(next_states[-1])
            int_rff = np.array([forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))])
            #R_intr_mean, R_intr_std = rolling.update(self.discount(intr_rewards, self.gamma).ravel().mean()) #
            rolling.update(int_rff.ravel())
            R_intr_std = np.sqrt(rolling.var)
            intr_rewards /= R_intr_std
            #print('intr reward', intr_rewards)

            forward_loss = self.forward_model.backprop(states[0], fold_batch(next_states), fold_batch(actions), fold_batch(extr_rewards), self.nsteps)

            Adv_extr = self.GAE(extr_rewards, values_extr, extr_last_values, dones, gamma=0.999, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, values_intr, intr_last_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) # non episodic intr reward signal 
            R_extr = Adv_extr + values_extr
            R_intr = Adv_intr + values_intr
            total_Adv = self.model.extr_coeff * Adv_extr + self.model.intr_coeff * Adv_intr

            #self.runner.state_mean, self.runner.state_std = state_rolling.update(fold_batch(next_states)[:,:,:,-1:]) # update state normalisation statistics 
            self.runner.state_mean, self.runner.state_std = self.state_rolling.update(next_states) # update state normalisation statistics 

            # perform minibatch gradient descent for K epochs 
            l = 0
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                batch_size = self.nsteps//self.num_minibatches
                np.random.shuffle(idxs)
                for batch in range(0,len(states),batch_size):
                    batch_idxs = idxs[batch:batch+batch_size]
                    # stack all states, next_states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Rextr, mb_Rintr, mb_Adv, mb_old_policies = fold_batch(states[batch_idxs]), fold_batch(next_states[batch_idxs]), \
                                                    fold_batch(actions[batch_idxs]), fold_batch(R_extr[batch_idxs]), fold_batch(R_intr[batch_idxs]), \
                                                    fold_batch(total_Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
                
                    mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(batch_size)) < self.pred_prob)][:,:,:,-1:]
                    #mb_nextstates = (mb_nextstates  - self.runner.state_mean[np.newaxis,:,:,np.newaxis]) / self.runner.state_std[np.newaxis,:,:,np.newaxis]
                    mean, std = self.runner.state_mean, self.runner.state_std
                    l += self.model.backprop(mb_states, mb_nextstates, mb_Rextr, mb_Rintr, mb_Adv, mb_actions, mb_old_policies, self.alpha, mean, std)
            
            l /= (self.num_epochs * self.num_minibatches)


            # Imagined future rollout 

            hidden = self.forward_model.get_initial_hidden(self.num_envs)
            obs = next_states[-1]
            encoded_last_state = self.forward_model.encode_state(next_states[-1]) # o_t -> s_t
            actions = [np.random.choice(policy.shape[1], p=policy[i]) for i in range(policy.shape[0])]
            imagined_rollout = []
            with tf.variable_scope('forward_model/latent-space-rnn', reuse=tf.AUTO_REUSE):
                for i in range(self.nsteps):
                    next_obs, extr_rewards, encoded_last_state, hidden = self.forward_model.predict_next(encoded_last_state, hidden, actions) 
                    #print('imagined obs', next_obs.shape)
                    intr_rewards = self.model.intrinsic_reward(next_obs[...,-1:], self.runner.state_mean, self.runner.state_std)
                    policies, extr_values, intr_values = self.model.forward(obs)
                    actions = [np.random.choice(policy.shape[1], p=policy[i]) for i in range(policy.shape[0])]
                    imagined_rollout.append([obs, next_obs, actions, extr_rewards[:,0], intr_rewards, extr_values, intr_values, policies])
                    obs = next_obs
            
            obs, next_obs, actions, extr_rewards, intr_rewards, extr_values, intr_values, old_policies = stack_many(zip(*imagined_rollout))
            #print('imagined obs', obs.shape)
            #print('imagined extr rew', extr_rewards.shape)
            #print('imagined extr_values', extr_values.shape)
            #print('imagined intr_values', intr_values.shape)
            
            intr_rewards /= R_intr_std

            policies, extr_last_values, intr_last_values = self.model.forward(next_obs[-1])
            Adv_extr = self.GAE(extr_rewards, extr_values, extr_last_values, np.zeros_like(dones), gamma=0.999, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, intr_values, intr_last_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) # non episodic intr reward signal 
            R_extr = Adv_extr + values_extr
            R_intr = Adv_intr + values_intr
            total_Adv = self.model.extr_coeff * Adv_extr + self.model.intr_coeff * Adv_intr

            for batch in range(0,len(obs),batch_size):
                batch_idxs = idxs[batch:batch+batch_size]
                # stack all states, next_states, actions and Rs across all workers into a single batch
                mb_states, mb_nextstates, mb_actions, mb_Rextr, mb_Rintr, mb_Adv, mb_old_policies = fold_batch(obs[batch_idxs]), fold_batch(next_obs[batch_idxs]), \
                                                fold_batch(actions[batch_idxs]), fold_batch(R_extr[batch_idxs]), fold_batch(R_intr[batch_idxs]), \
                                                fold_batch(total_Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
            
                mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(batch_size)) < self.pred_prob)][...,-1:]
                #mb_nextstates = (mb_nextstates  - self.runner.state_mean[np.newaxis,:,:,np.newaxis]) / self.runner.state_std[np.newaxis,:,:,np.newaxis]
                mean, std = self.runner.state_mean, self.runner.state_std
                l += self.model.backprop(mb_states, mb_nextstates, mb_Rextr, mb_Rintr, mb_Adv, mb_actions, mb_old_policies, self.alpha, mean, std)

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
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps):
            super().__init__(model, env, num_steps)
            self.state_mean = None
            self.state_std = None
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values_extr, values_intr = self.model.forward(self.states)
                #actions = np.argmax(policies, axis=1)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)

                intr_rewards = self.model.intrinsic_reward(next_states[...,-1:], self.state_mean, self.state_std)
                #print('intr rewards', intr_rewards)
                rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones))
                self.states = next_states

            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones = stack_many(zip(*rollout))
            return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones
    
    


def main(env_id, Atari=True):


    config = tf.ConfigProto() #GPU 
    config.gpu_options.allow_growth=True #GPU
    sess = tf.Session(config=config)

    print('gpu aviabliable', tf.test.is_gpu_available())

    num_envs = 32
    nsteps = 128

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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/RND/' + env_id + '/'
    model_dir = "models/RND/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = RND(nature_cnn,
                predictor_cnn,
                input_shape = input_size,
                action_size = action_size,
                intr_coeff=1.0,
                extr_coeff=2.0,
                value_coeff=0.5,
                lr=1e-4,
                lr_final=1e-4,
                decay_steps=50e6//(num_envs*nsteps),
                grad_clip=0.5,
                policy_args={},
                RND_args={}) #

    forward_model = Forward_Model(nature_cnn, nature_deconv, input_size, action_size, 1e-4, 256, num_envs, nsteps)

    curiosity = MA_RND_Trainer(envs = envs,
                            model = model,
                            forward_model=forward_model,
                            file_loc = [model_dir, train_log_dir],
                            val_envs = val_envs,
                            train_mode = 'nstep',
                            total_steps = 50e6,
                            nsteps = nsteps,
                            num_epochs=4,
                            num_minibatches=4,
                            validate_freq = 1e6,
                            save_freq = 0,
                            render_freq = 0,
                            num_val_episodes = 50,
                            log_scalars=True)
    curiosity.train()
    
    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4','FreewayDeterministic-v4', 'PongDeterministic-v4',]
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1' ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    