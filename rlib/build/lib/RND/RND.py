import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from networks import*
from utils import one_hot, fold_batch
from SyncMultiEnvTrainer import SyncMultiEnvTrainer
from VecEnv import*
from Curiosity import ICM

class ActorCritic_LSTM(object):
    def __init__(self, model_head, input_shape, action_size, num_envs, cell_size,
                 lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.cell_size = cell_size
        self.num_envs = num_envs
        self.sess = None

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='time_batch_state') # [time*batch, *input_shape]

        with tf.variable_scope('encoder_network'):
            self.dense = model_head(self.state, **model_head_args)
            dense_size = self.dense.get_shape()[1].value
            unfolded_state = tf.reshape(self.dense, shape=[-1, num_envs, dense_size], name='unfolded_state')
        
        with tf.variable_scope('lstm'):
            #self.lstm_output, self.hidden_in, self.hidden_out = lstm(unfolded_state, cell_size=cell_size, fold_output=True, time_major=True)
            self.lstm_output, self.hidden_in, self.hidden_out, self.mask = lstm_masked(unfolded_state, cell_size=cell_size, batch_size=num_envs, fold_output=True, time_major=True)

        with tf.variable_scope('extr_critic'):
            self.Ve = tf.reshape( mlp_layer(self.lstm_output, 1, name='state_value_extr', activation=None), shape=[-1])

        with tf.variable_scope('intr_critic'):
            self.Vi = tf.reshape( mlp_layer(self.lstm_output, 1, name='state_value_intr', activation=None), shape=[-1])
        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.lstm_output, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R_extr = tf.placeholder(dtype=tf.float32, shape=[None])
            Advantage_extr = self.R_extr - self.Ve
            extr_value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage_extr))

            self.R_intr = tf.placeholder(dtype=tf.float32, shape=[None])
            Advantage_intr = self.R_intr - self.Vi
            intr_value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage_intr))

            tot_Adv = Advantage_extr + Advantage_intr
            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(tot_Adv))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.loss =  policy_loss + 0.5 * (extr_value_loss + intr_value_loss) - 0.01 * entropy

        if opt:
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

            
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)
    
    def get_initial_hidden(self, batch_size):
        return np.zeros((batch_size, self.cell_size)), np.zeros((batch_size, self.cell_size))
    
    def reset_batch_hidden(self, hidden, idxs):
        return hidden * np.stack([idxs for i in range(self.cell_size)], axis=1)

    def forward(self, state, hidden):
        mask = np.zeros((1, self.num_envs)) # state = [time, batch, ...]
        feed_dict = {self.state:state, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:mask}
        policy, value_extr, value_intr, hidden = self.sess.run([self.policy_distrib, self.Ve, self.Vi, self.hidden_out], feed_dict = feed_dict)
        return policy, value_extr, value_intr, hidden

    def backprop(self, state, R, a, hidden, dones):
        feed_dict = {self.state : state, self.R : R, self.actions: a, self.hidden_in[0]:hidden[0], self.hidden_in[1]:hidden[1], self.mask:dones}
        *_,l = self.sess.run([self.train_op, self.loss], )
        return l
    
    def set_session(self, sess):
        self.sess = sess


class RND(object):
    def __init__(self, policy_model, target_model, input_shape, action_size, num_envs, cell_size, forward_model_scale, policy_importance, reward_scale, prediction_beta, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
        self.reward_scale, self.forward_model_scale, self.policy_importance, self.prediction_beta = reward_scale, forward_model_scale, policy_importance, prediction_beta
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.sess = None
        self.num_envs = num_envs

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape])

        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)
            with tf.variable_scope('encoder_network', reuse=True):
                pred_next_state = policy_model(self.next_state)
            

        with tf.variable_scope('target_model'):
            target_state = target_model(self.next_state)
        
        with tf.variable_scope('predictor_model'):
            self.intr_reward = tf.reduce_sum(tf.square(pred_next_state - target_state), axis=-1)
            feat_loss = tf.reduce_mean(self.intr_reward)

        self.loss = policy_importance * self.train_policy.loss  + feat_loss
        

        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)

    def forward(self, state, hidden, validate=False):
        if validate:
            return self.validate_policy.forward(state, hidden)
        else :
            return self.train_policy.forward(fold_batch(state), hidden)
    
    def get_initial_hidden(self, batch_size):
        return self.train_policy.get_initial_hidden(batch_size)
    
    def reset_batch_hidden(self, hidden, idxs):
        return self.train_policy.reset_batch_hidden(hidden, idxs)
    
    def intrinsic_reward(self, next_state):
        forward_loss = self.sess.run(self.intr_reward, feed_dict={self.next_state:next_state})
        intr_reward = forward_loss * self.prediction_beta 
        return intr_reward
    
    def backprop(self, state, next_state, R_extr, R_intr, actions, hidden, dones):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.train_policy.state:state, self.train_policy.actions:actions, self.next_state:next_state,
                     self.train_policy.R_extr:R_extr, self.train_policy.R_intr:R_intr,
                     self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
                     self.train_policy.mask:dones}

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class Curiosity_LSTM_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes)
        self.runner = self.Runner(self.model, self.env, self.num_envs, self.nsteps)
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments
        '''
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, hidden_batch, dones, infos, extr_values, intr_values = self.runner.run()
            R_extr = self.multistep_target(extr_rewards, extr_values, dones, clip=False)
            R_intr = self.multistep_target(intr_rewards, intr_values, dones, clip=False)
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R_extr, R_intr = fold_batch(states),fold_batch(next_states), fold_batch(actions), fold_batch(R_extr), fold_batch(R_intr)
            l = self.model.backprop(states, next_states, R_extr, R_intr, actions, hidden_batch[0], dones)

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
    
    
    def validate(self,env,num_ep,max_steps,render=False):
        episode_scores = []
        for episode in range(num_ep):
            state = env.reset()
            episode_score = []
            hidden = self.model.get_initial_hidden(1)
            for t in range(max_steps):
                policy, value_extr, value_intr, hidden = self.model.forward(state[np.newaxis], hidden, validate=True)
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
        def __init__(self,model,env,num_envs,num_steps):
            super().__init__(model,env,num_steps)
            self.prev_hidden = self.model.get_initial_hidden(num_envs)
        
        def run(self,):
            memory = []
            for t in range(self.num_steps):
                policies, values_extr, values_intr, hidden = self.model.forward(self.states[np.newaxis], self.prev_hidden)
                #actions = np.argmax(policies, axis=1)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)

                intr_rewards = self.model.intrinsic_reward(next_states) 
                
                memory.append((self.states, next_states, actions, extr_rewards, intr_rewards, self.prev_hidden, dones, infos))
                self.states = next_states
                
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode

            states, next_states, actions, extr_rewards, intr_rewards, hidden_batch, dones, infos = zip(*memory)
            states, next_states, actions, extr_rewards, intr_rewards, dones = np.stack(states), np.stack(next_states), np.stack(actions), np.stack(extr_rewards), np.stack(intr_rewards), np.stack(dones)
            return states, next_states, actions, extr_rewards, intr_rewards, hidden_batch, dones, infos, values_extr, values_intr
            

def main(env_id):

    num_envs = 32
    nsteps = 20

    env = gym.make(env_id)
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(10)]
        envs = BatchEnv(DummyEnv, env_id, num_envs, blocking=False)

    elif 'SuperMarioBros' in env_id:
        print('Mario')
        
        #envs = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        envs = BatchEnv(MarioEnv, env_id, num_envs)
        val_envs = [MarioEnv(gym.make(env_id)) for i in range(16)] 

    else:
        print('Atari')
        if env.unwrapped.get_action_meanings()[1] == 'FIRE':
            reset = True
            print('fire on reset')
        else:
            reset = False
            print('only stack frames')
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=42, episodic=False, reset=reset, clip_reward=False) for i in range(10)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False , rescale=42, k=4, reset=reset, episodic=False, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    
    

    train_log_dir = 'logs/RND/' + env_id +'/'
    model_dir = "models/RND/" + env_id + '/'

    

    ac_cnn_args = {}

    ICM_mlp_args = {}

    ICM_cnn_args = {}
    
    
   
    #ac_mlp_args = {'dense_size':256}

    model = RND(nature_cnn,
                      nature_reservoir,
                      input_shape = input_size,
                      action_size = action_size,
                      num_envs = num_envs,
                      cell_size = 256,
                      forward_model_scale=0.2,
                      policy_importance=1,
                      reward_scale=1,
                      prediction_beta=0.5,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={})

    

    curiosity = Curiosity_LSTM_Trainer(envs = envs,
                                  model = model,
                                  file_loc = [model_dir, train_log_dir],
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  total_steps = 50e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e5,
                                  save_freq = 0,
                                  render_freq = 0,
                                  num_val_episodes = 10)

    print(env_id)
    
    hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':num_envs,
                  'total_steps':curiosity.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5, 'reward_scale':model.reward_scale,
                   'forward_model_scale':model.forward_model_scale, 'policy_importance':model.policy_importance, 'prediction_beta':model.prediction_beta}
    
    filename = train_log_dir + curiosity.current_time + '/hyperparameters.txt'
    curiosity.save_hyperparameters(filename, **hyper_paras)

    curiosity.train()

    del model

    tf.reset_default_graph()

if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'VentureDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1']
    #env_id_list = ['SuperMarioBros-1-1-v0']
    for env_id in env_id_list:
        main(env_id)