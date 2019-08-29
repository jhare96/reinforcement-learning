import gym
import numpy as np
import scipy 
import tensorflow as tf
import multiprocessing, threading
from collections import deque
import time, datetime, os
import copy

from rlib.networks.networks import*
from rlib.utils.VecEnv import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fold_batch, stack_many


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class ActorCritic(object):
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=0, decay_steps=80e3, grad_clip=0.5, **model_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.sess = None

        with tf.variable_scope('encoder_network'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
            print('state shape', self.state.get_shape().as_list())
            self.dense = model(self.state, **model_args)
        
        with tf.variable_scope('critic'):
            self.V = tf.reshape(mlp_layer(self.dense, 1, name='state_value', activation=None), shape=[-1])
        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R = tf.placeholder(dtype=tf.float32, shape=[None])
            Advantage = self.R - self.V
            value_loss = 0.5 * tf.reduce_mean(tf.square(Advantage))

            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(Advantage))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

        
        
        self.loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy
        #optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
        optimiser = tf.train.AdamOptimizer(lr)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        grads = tf.gradients(self.loss, self.weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, self.weights))
        self.optimiser = optimiser.apply_gradients(grads_vars)

    def forward(self, state):
        return self.sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})

    def get_policy(self, state):
        return self.sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_value(self, state):
        return self.sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self, state, R, a):
        *_,l = self.sess.run([self.optimiser, self.loss], feed_dict = {self.state:state, self.R:R, self.actions:a})
        return l
    
    def set_session(self, sess):
        self.sess = sess

        
        
#self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, 
                    # validate_freq=1e6, save_freq=0, render_freq=0, update_target_freq = 10000)

class A2C(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', return_type='nstep', log_dir='logs/', model_dir='models/', total_steps=10000, nsteps=5, gamma=0.99, lambda_=0.95,
                 validate_freq=1e6, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True, gpu_growth=True):
        
        super().__init__(envs, model, val_envs, log_dir=log_dir, model_dir=model_dir, train_mode=train_mode, return_type=return_type, total_steps=total_steps, nsteps=nsteps,
         gamma=gamma, lambda_=lambda_, validate_freq=validate_freq, save_freq=save_freq, render_freq=render_freq,
         num_val_episodes=num_val_episodes, log_scalars=log_scalars, gpu_growth=gpu_growth)
        
        self.runner = self.Runner(self.model,self.env,self.nsteps)

        hyperparas = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps , 'grad_clip':model.grad_clip, 'nsteps':nsteps, 'num_workers':self.num_envs,
                  'total_steps':total_steps, 'entropy_coefficient':0.01, 'value_coefficient':0.5 , 'return type':self.return_type}
        
        if log_scalars:
            filename = log_dir + '/' + 'hyperparameters.txt'
            self.save_hyperparameters(filename , **hyperparas)

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self,model,env,num_steps):
            super().__init__(model,env,num_steps)
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states, actions, rewards, values, dones, np.array(infos)))
                self.states = next_states
            
            states, actions, rewards, values, dones, infos = stack_many(zip(*rollout))
            _, last_values = self.model.forward(next_states)
            return states, actions, rewards, dones, infos, values, last_values
    
    def get_action(self, state):
        policy, value = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        return action
    
    
        
    def _train_onestep(self):
        tf_epLoss,tf_epScore,tf_epReward,tf_valReward = self.tf_placeholders
        tf_sum_epLoss,tf_sum_epScore,tf_sum_epReward,tf_sum_valReward = self.tf_summary_scalars
        states = self.env.reset()
        y = np.zeros((self.num_envs))
        num_steps = self.total_steps // self.num_envs
        for t in range(1,num_steps+1):
            policies, values = self.model.forward(states)
            actions = np.array([np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])])
            next_states, rewards, dones, infos = self.env.step(actions)
            rewards = np.clip(rewards, -1, 1)
            y = rewards + self.gamma * self.model.get_value(next_states) * (1-dones)
            
            l = self.model.backprop(states, y, actions)
            states = next_states
            
            
            if t % 1000 == 0:
                render = False
                if t % 10000 == 0:
                    render = True
                score = self.validate(5,2000,render)
                tot_steps = t*self.num_envs
                print("update %i, validation score %f, total steps %i, loss %f" %(t,score,tot_steps,l))
                sumscore, sumloss = self.sess.run([tf_sum_epScore, tf_sum_epLoss], feed_dict = {tf_epScore:score, tf_epLoss:l})
                self.train_writer.add_summary(sumloss, tot_steps)
                self.train_writer.add_summary(sumscore, tot_steps)

       







    




def main(env_id):
    
    num_envs = 32
    nsteps = 20
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')

    train_log_dir = 'logs/A2C/' + env_id +'/n-step/' + current_time 
    model_dir = "models/A2C/" + env_id + '/n-step/' + current_time 
    
    env = gym.make(env_id)
    action_size = env.action_space.n
    input_size = env.reset().shape[0]
    
    
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, k=4, reset=reset, episodic=False, clip_reward=True)
    
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    print('input shape', input_size)

    env.close()
    print('action space', action_size)
    

    
    
    
    ac_cnn_args = {'input_shape':[84,84,4], 'action_size':action_size,  'lr':1e-3, 'grad_clip':0.5, 'decay_steps':50e6/(num_envs*nsteps),
                    'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
   
    ac_mlp_args = {'input_shape':input_size, 'dense_size':64, 'num_layers':2, 'action_size':action_size, 'lr':1e-3, 'grad_clip':0.5, 'decay_steps':5e6//(num_envs*nsteps), 'lr_final':0}

    
    model = ActorCritic(nature_cnn, input_size, action_size, lr=1e-3, lr_final=1e-3, decay_steps=50e6//(num_envs*nsteps), grad_clip=0.5) 
    

    a2c = A2C(envs = envs,
              model = model,
              model_dir = model_dir,
              log_dir = train_log_dir,
              val_envs = val_envs,
              train_mode = 'nstep',
              return_type = 'nstep',
              total_steps = 2e6,
              nsteps = nsteps,
              validate_freq = 1e5,
              save_freq = 0,
              render_freq = 0,
              num_val_episodes = 50,
              log_scalars = False,
              gpu_growth = True)

    a2c.train()

    del a2c

    #tf.reset_default_graph()

    # a2c = A2C.load(A2C, model, envs, val_envs, model_dir + time + '/1.trainer')
    # a2c.train()


if __name__ == "__main__":
    env_id_list = ['SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'PongDeterministic-v4']
    #env_id_list = ['MontezumaRevengeDeterministic-v4']
    #env_id_list = ['CartPole-v1',]# 'MountainCar-v0', 'Acrobot-v1']
    for i in range(1):
        for env_id in env_id_list:
            main(env_id)
