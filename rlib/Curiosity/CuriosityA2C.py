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
from rlib.utils.utils import fold_batch, one_hot, rolling_stats, normalise, stack_many
#from .OneNetCuriosity import Curiosity_onenet

class rolling_obs(object):
    def __init__(self, mean=0):
        self.rolling = rolling_stats(mean)
    
    def update(self, x):
        if len(x.shape) == 4: # assume image obs 
            return self.rolling.update(np.mean(x, axis=(0,-1))) #[time*batch,height,width,stack] -> [height, width]
        else:
            return self.rolling.update(np.mean(x, axis=0)) #[time*batch,*shape] -> [*shape]


class ActorCritic(object):
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, **model_args):
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
            self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution') + 1e-10
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R = tf.placeholder(dtype=tf.float32, shape=[None])
            self.Advantage = tf.placeholder(dtype=tf.float32, shape=[None])
            #Advantage = self.R - self.V
            value_loss = 0.5 * tf.reduce_mean(tf.square(self.R - self.V))

            log_policy = tf.math.log(self.policy_distrib)
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * tf.stop_gradient(self.Advantage))

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        self.loss =  policy_loss + 0.5 * value_loss - 0.01 * entropy
        

    def forward(self, state):
        return self.sess.run([self.policy_distrib, self.V], feed_dict = {self.state:state})

    def get_policy(self, state):
        return self.sess.run(self.policy_distrib, feed_dict = {self.state: state})
    
    def get_value(self, state):
        return self.sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self, state, R, Adv, a):
        *_,l = self.sess.run([self.optimiser, self.loss], feed_dict = {self.state : state, self.R : R, self.Advantage:Adv, self.actions: a})
        return l
    
    def set_session(self, sess):
        self.sess = sess

class ICM(object):
    def __init__(self, model_head, input_size, action_size, **model_head_args):    
            
        print('input_Shape', input_size)
        self.state = tf.placeholder(tf.float32, shape=[None, *input_size], name="state")
        self.next_state = tf.placeholder(tf.float32, shape=[None, *input_size], name="next_state")
        self.state_mean = tf.placeholder(tf.float32, shape=[*input_size], name="mean")
        self.state_std = tf.placeholder(tf.float32, shape=[*input_size], name="std")
        norm_next_state = (self.state - self.state_mean) / self.state_std
        
        self.action = tf.placeholder(tf.int32, shape=[None], name="actions")
        action_onehot = tf.one_hot(self.action, action_size)
       
        with tf.variable_scope('encoder_network'):
            self.phi1 = model_head(norm_next_state,  **model_head_args)
        with tf.variable_scope('encoder_network', reuse=True):
            self.phi2 = model_head(norm_next_state,  **model_head_args)
        

        with tf.variable_scope('Forward_Model'):
            state_size = self.phi1.get_shape()[1].value
            concat = tf.concat([self.phi1, tf.dtypes.cast(action_onehot, tf.float32)], 1, name='state-action-concat')
            f1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='foward_model')
            pred_state = mlp_layer(f1, state_size, activation=None, name='pred_state')
            print('pred_state', pred_state.get_shape().as_list(), 'phi_2', self.phi2.get_shape().as_list())
            self.intr_reward = tf.reduce_mean(tf.square(pred_state - self.phi2), axis=1)# l2 distance metric ‖ˆφ(st+1)−φ(st+1)‖22
            self.forward_loss =  tf.reduce_mean(self.intr_reward) # intr_reward batch loss 
            print('forward_loss', self.forward_loss.get_shape().as_list())
        
        with tf.variable_scope('Inverse_Model'):
            concat = tf.concat([self.phi1, self.phi2], 1, name='state-nextstate-concat')
            i1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='inverse_model')
            pred_action = mlp_layer(i1, action_size, activation=None, name='pred_state')
            self.inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_action, labels=action_onehot)) # batch inverse loss
            print('pred_action', pred_action.get_shape().as_list(), 'inverse_loss', self.inverse_loss.get_shape().as_list())
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        #exit()
        

class Curiosity(object):
    def __init__(self,  policy_model, ICM_model, input_size, action_size, forward_model_scale, policy_importance, reward_scale, intr_coeff, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
        self.reward_scale, self.forward_model_scale, self.policy_importance, self.intr_coeff = reward_scale, forward_model_scale, policy_importance, intr_coeff
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        with tf.variable_scope('ICM'):
            self.ICM = ICM(ICM_model, input_size, action_size, **ICM_args)
        
        with tf.variable_scope('ActorCritic'):
            self.AC = ActorCritic(policy_model, input_size, action_size, lr, lr_final, decay_steps, grad_clip, **policy_args)

        self.loss = policy_importance * self.AC.loss + reward_scale * ((1-forward_model_scale) * self.ICM.inverse_loss + forward_model_scale * self.ICM.forward_loss) 
        
        
        #global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        #self.optimiser = tf.train.AdamOptimizer(lr)
        
        weights = self.AC.weights + self.ICM.weights
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)#, global_step=global_step)

    def forward(self, state):
        return self.AC.forward(state)
    
    def intrinsic_reward(self, state, action, next_state, state_mean, state_std):
        feed_dict={self.ICM.state:state, self.ICM.action:action, self.ICM.next_state:next_state,
                   self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        intr_reward = self.sess.run(self.ICM.intr_reward, feed_dict=feed_dict)
        return intr_reward * self.intr_coeff 
    
    def backprop(self, state, next_state, R, Adv, actions, state_mean, state_std):
        feed_dict = {self.AC.state:state, self.AC.actions:actions, self.AC.R:R, self.AC.Advantage:Adv,
                     self.ICM.state:state, self.ICM.next_state:next_state, self.ICM.action:actions,
                     self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.AC.set_session(sess)


class Curiosity_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes,log_scalars=log_scalars)

        self.state_obs = rolling_obs()
        self.runner = self.Runner(self.model, self.env, self.nsteps)
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':0.01, 'value_coefficient':0.5, 'reward_scale':model.reward_scale,
        'forward_model_scale':model.forward_model_scale, 'policy_importance':model.policy_importance, 'intr_coeff':model.intr_coeff}
    
        if self.log_scalars:
            filename = file_loc[1] + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
        
        self.lambda_ = 0.95
    
    def init_state_obs(self, num_steps):
        states = 0
        for i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        return states / num_steps
    
    
    def _train_nstep(self):
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        self.runner.state_mean, self.runner.state_std = self.state_obs.update(self.init_state_obs(10000//self.num_envs))
        self.runner.states = self.env.reset()
        rolling = rolling_stats()
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            states, next_states, actions, rewards, dones, values = self.runner.run()
            _, last_values = self.model.forward(next_states[-1])

            R_mean, R_std = rolling.update(self.nstep_return(rewards, last_values, dones).ravel().mean(axis=0))
            rewards /= R_std
            #print('rewards', rewards)

            R = self.nstep_return(rewards, last_values, dones)
            Adv = R - values
            #delta = rewards + self.gamma * values[:-1] - values[1:]
            #Adv = self.multistep_target(delta, values[-1], dones, gamma=self.gamma*self.lambda_)
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R, Adv = fold_batch(states), fold_batch(next_states), fold_batch(actions), fold_batch(R), fold_batch(Adv)
            mean, std = np.stack([self.runner.state_mean for i in range(4)], -1), np.stack([self.runner.state_std for i in range(4)], -1)
            
            l = self.model.backprop(states, next_states, R, Adv, actions, mean, std)
            
            #self.runner.state_mean, self.runner.state_std = self.state_obs.update(states)
            
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
                start = time.time()
                policies, values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
                
                mean, std = np.stack([self.state_mean for i in range(4)], -1), np.stack([self.state_std for i in range(4)], -1)
                intr_rewards = self.model.intrinsic_reward(self.states, actions, next_states, mean, std)
                #print('intr_rewards', intr_rewards)
                rewards = extr_rewards + intr_rewards
                #print('rewards', rewards)
                rollout.append((self.states, next_states, actions, rewards, values, dones))
                self.states = next_states
           
            states, next_states, actions, rewards, values, dones = stack_many(zip(*rollout))
            return states, next_states, actions, rewards, dones, values
            

def main(env_id):
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(1)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/Curiosity/' +env_id + '/hyper_unclipped/'
    model_dir = "models/Curiosity/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = Curiosity(nature_cnn,
                      nature_cnn,
                      input_size = input_size,
                      action_size = action_size,
                      forward_model_scale=0.2,
                      policy_importance=1,
                      reward_scale=1.0,
                      intr_coeff=1,
                      lr=1e-3,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={'scale':False})#'activation':tf.nn.leaky_relu

    

    curiosity = Curiosity_Trainer(envs = envs,
                                  model = model,
                                  file_loc = [model_dir, train_log_dir],
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  total_steps = 5e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e5,
                                  save_freq = 0,
                                  render_freq = 1,
                                  num_val_episodes = 5,
                                  log_scalars=False)
    print(env_id)
    curiosity.train()

    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4',  'PongDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    