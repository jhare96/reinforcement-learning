import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
from rlib.utils.utils import one_hot, fold_batch
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*

from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.Curiosity.Curiosity import ICM

class Curiosity_LSTM(object):
    def __init__(self, policy_model, ICM_model, input_shape, action_size, num_envs, cell_size, forward_model_scale, policy_importance, reward_scale, prediction_beta, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
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
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE):
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, **policy_args)

        with tf.variable_scope('ICM'):
            self.ICM = ICM(ICM_model, input_shape, action_size, **ICM_args)
        
        self.loss = policy_importance * self.train_policy.loss  + reward_scale * ((1-forward_model_scale) * self.ICM.inverse_loss + forward_model_scale * self.ICM.forward_loss ) 
        

        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.AdamOptimizer(lr)#, decay=0.9, epsilon=1e-5)
        
    
        weights = self.train_policy.weights + self.ICM.weights
        #print('weights', weights)
        grads = tf.gradients(self.loss, weights)
        #print('grads', grads)
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
    
    def intrinsic_reward(self, state, action, next_state):
        action = one_hot(action, self.action_size)
        forward_loss = self.sess.run(self.ICM.intr_reward, feed_dict={self.ICM.input1:state, self.ICM.action:action, self.ICM.input2:next_state})
        intr_reward = forward_loss * self.prediction_beta 
        return intr_reward
    
    def backprop(self, state, next_state, R, actions, hidden, dones):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.train_policy.state:state, self.train_policy.actions:actions,
                     self.train_policy.R:R, self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
                     self.train_policy.mask:dones,
                     self.ICM.input1:state, self.ICM.input2:next_state,
                     self.ICM.action:actions_onehot}

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class Curiosity_LSTM_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        self.runner = self.Runner(self.model, self.env, self.num_envs, self.nsteps)
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5, 'reward_scale':model.reward_scale,
                   'forward_model_scale':model.forward_model_scale, 'policy_importance':model.policy_importance, 'prediction_beta':model.prediction_beta}
        
        if self.log_scalars:
            filename = file_loc[1] + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def _train_nstep(self):
        '''
            Episodic training loop for synchronous training over multiple environments
        '''
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, rewards, hidden_batch, dones, infos, values = self.runner.run()
            R = self.multistep_target(rewards, values, dones, clip=False)  
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R = fold_batch(states), fold_batch(next_states), fold_batch(actions), fold_batch(R)
            l = self.model.backprop(states, next_states, R, actions, hidden_batch[0], dones)

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
        def __init__(self,model,env,num_envs,num_steps):
            super().__init__(model,env,num_steps)
            self.prev_hidden = self.model.get_initial_hidden(num_envs)
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values, hidden = self.model.forward(self.states[np.newaxis], self.prev_hidden)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)

                intr_rewards = self.model.intrinsic_reward(self.states, actions, next_states) 
                rewards = extr_rewards + self.model.prediction_beta * intr_rewards
                rollout.append((self.states, next_states, actions, rewards, self.prev_hidden, dones, infos))
                self.states = next_states
                
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode

            states, next_states, actions, rewards, hidden_batch, dones, infos = zip(*rollout)
            states, next_states, actions, rewards, dones = np.stack(states), np.stack(next_states), np.stack(actions), np.stack(rewards), np.stack(dones)
            return states, next_states, actions, rewards, hidden_batch, dones, infos, values
            

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
        
        val_envs = [AtariEnv(gym.make(env_id), k=3, rescale=42, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, rescale=42, blocking=False , k=3, reset=reset, episodic=True, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    
    

    train_log_dir = 'logs/Curiosity_LSTM/' + env_id +'/'
    model_dir = "models/Curiosity_LSTM/" + env_id + '/'

    

    ac_cnn_args = {}

    ICM_mlp_args = {}

    ICM_cnn_args = {}
    
    
   
    #ac_mlp_args = {'dense_size':256}

    model = Curiosity_LSTM(universe_cnn,
                      universe_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      num_envs = num_envs,
                      cell_size = 256,
                      forward_model_scale=0.2,
                      policy_importance=0.1,
                      reward_scale=1.0,
                      prediction_beta=0.1,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={}) #'activation':tf.nn.leaky_relu

    

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
                                  num_val_episodes = 25,
                                  log_scalars=False)

    print(env_id)
    
    curiosity.train()

    del model

    tf.reset_default_graph()

if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1']
    #env_id_list = ['SuperMarioBros-1-1-v0']
    for env_id in env_id_list:
        main(env_id)