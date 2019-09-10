import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time, datetime
import threading
from rlib.A2C.ActorCritic import ActorCritic_LSTM
from rlib.networks.networks import*
from rlib.utils.utils import fold_batch, stack_many
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*


class A2C_LSTM(ActorCritic_LSTM):
    def __init__(self, policy_model, input_shape, action_size, num_envs, cell_size, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}):
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.cell_size = cell_size
        self.sess = None
        self.num_envs = num_envs

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        
        with tf.variable_scope('ActorCritic', reuse=tf.AUTO_REUSE): # due to fixed lstm reshape size, need to create two Actorcritic models for multi and single envs
            self.train_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, num_envs, cell_size, build_optimiser=False, **policy_args)
            self.validate_policy = ActorCritic_LSTM(policy_model, input_shape, action_size, 1, cell_size, build_optimiser=False, **policy_args)
        
        self.loss = self.train_policy.loss
    
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.AdamOptimizer(lr)#, decay=0.9, epsilon=1e-5)     
    
        weights = self.train_policy.weights 
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))
        self.train_op = self.optimiser.apply_gradients(grads_vars)

    def forward(self, state, hidden, validate=False):
        if validate:
            return self.validate_policy.forward(state, hidden)
        else :
            return self.train_policy.forward(state, hidden)
    
    def backprop(self, state, R, actions, hidden, dones):
        feed_dict = {self.train_policy.state:state, self.train_policy.actions:actions,
                     self.train_policy.R:R, self.train_policy.hidden_in[0]:hidden[0], self.train_policy.hidden_in[1]:hidden[1],
                     self.train_policy.mask:dones}

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.train_policy.set_session(sess)
        self.validate_policy.set_session(sess)


class A2C_LSTM_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', return_type='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=20,
                validate_freq=1e6, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        
        super().__init__(envs, model, val_envs, log_dir=log_dir, model_dir=model_dir, train_mode=train_mode, return_type=return_type,
                        total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq, save_freq=save_freq,
                        render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        
        
        self.runner = self.Runner(self.model, self.env, self.num_envs, self.nsteps)
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs,
                  'total_steps':self.total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5}
        
        if self.log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def _train_nstep(self):
        batch_size = (self.num_envs * self.nsteps)
        start = time.time()
        num_updates = self.total_steps // batch_size
        s = 0
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, hidden_batch, dones, infos, values, last_values = self.runner.run()
            
            if self.return_type == 'nstep':
                R = self.nstep_return(rewards, last_values, dones, gamma=self.gamma)
            elif self.return_type == 'GAE':
                R = self.GAE(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_) + values
            elif self.return_type == 'lambda':
                R = self.lambda_return(rewards, values, last_values, dones, gamma=self.gamma, lambda_=self.lambda_)
                
            # stack all states, actions and Rs across all workers into a single batch
            states, actions, R = fold_batch(states), fold_batch(actions), fold_batch(R)
            l = self.model.backprop(states, R, actions, hidden_batch[0], dones)

            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq //batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir  + str(s) + ".ckpt") )
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
                policies, values, hidden = self.model.forward(self.states, self.prev_hidden)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states, actions, rewards, self.prev_hidden, dones, infos))
                self.states = next_states
                
                self.prev_hidden = self.model.reset_batch_hidden(hidden, 1-dones) # reset hidden state at end of episode
                
            states, actions, rewards, hidden_batch, dones, infos = stack_many(zip(*rollout))
            _, last_values, _ = self.model.forward(next_states, self.prev_hidden)
            return states, actions, rewards, hidden_batch, dones, infos, values, last_values
            

def main(env_id):
    num_envs = 32
    nsteps = 20

    env = gym.make(env_id)
    
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=1, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, rescale=84, blocking=False , k=1, reset=reset, episodic=False, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    

    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/A2C_LSTM/' + env_id +'/' + current_time
    model_dir = "models/A2C_LSTM/" + env_id + '/' + current_time
    

    model = A2C_LSTM(nature_cnn,
                    input_shape = input_size,
                    action_size = action_size,
                    num_envs = num_envs,
                    cell_size = 256,
                    lr=1e-3,
                    lr_final=1e-3,
                    decay_steps=50e6//(num_envs*nsteps),
                    grad_clip=0.5,
                    policy_args={})

    

    curiosity = A2C_LSTM_Trainer(envs = envs,
                                  model = model,
                                  model_dir = model_dir,
                                  log_dir = train_log_dir,
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  return_type ='GAE',
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