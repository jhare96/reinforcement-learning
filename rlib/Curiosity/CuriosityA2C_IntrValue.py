import tensorflow as tf
import numpy as np
import scipy
import gym
import os, time
import threading
#from rlib.A2C.A2C import ActorCritic
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.utils import fold_batch, one_hot, stack_many, rolling_stats, RunningMeanStd
#from .OneNetCuriosity import Curiosity_onenet


class rolling_obs(object):
    def __init__(self, shape=()):
        self.rolling = RunningMeanStd(shape=shape)
    
    def update(self, x):
        if len(x.shape) == 5: # assume image obs 
            return self.rolling.update(fold_batch(x[...,-1:])) #[time,batch,height,width,stack] -> [height, width,1]
        else:
            return self.rolling.update(fold_batch(x)) #[time,batch,*shape] -> [*shape]

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

class ActorCritic(object):
    def __init__(self, model_head, input_shape, action_size, intr_coeff=0.5, extr_coeff=1.0, value_coeff=0.5, entropy_coeff=0.01,
                 lr=1e-3, lr_final=1e-6, decay_steps=6e5, grad_clip = 0.5, opt=False, **model_head_args):
        self.lr, self.lr_final = lr, lr_final
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.intr_coeff, self.extr_coeff = intr_coeff, extr_coeff
        self.sess = None

        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='time_batch_state') # [time*batch, *input_shape]

        with tf.variable_scope('encoder_network'):
            self.dense = model_head(self.state, **model_head_args)

        with tf.variable_scope('extr_critic'):
            self.Ve = tf.reshape( mlp_layer(self.dense, 1, name='state_value_extr', activation=None), shape=[-1])

        with tf.variable_scope('intr_critic'):
            self.Vi = tf.reshape( mlp_layer(self.dense, 1, name='state_value_intr', activation=None), shape=[-1])
        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution')
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.R_extr = tf.placeholder(dtype=tf.float32, shape=[None])
            extr_value_loss = 0.5 * tf.reduce_mean(tf.square(self.R_extr - self.Ve))

            self.R_intr = tf.placeholder(dtype=tf.float32, shape=[None])
            intr_value_loss = 0.5 * tf.reduce_mean(tf.square(self.R_intr - self.Vi))

            self.Advantage = tf.placeholder(dtype=tf.float32, shape=[None])
            log_policy = tf.math.log(tf.clip_by_value(self.policy_distrib, 1e-6, 0.999999))
            log_policy_actions = tf.reduce_sum(tf.multiply(log_policy, actions_onehot), axis=1)
            policy_loss =  tf.reduce_mean(-log_policy_actions * self.Advantage)

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy_distrib * -log_policy, axis=1))
    
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)

        self.loss =  policy_loss + value_coeff * (extr_value_loss + intr_value_loss) - entropy_coeff * entropy

        if opt:
            global_step = tf.Variable(0, trainable=False)
            tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

            
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.train_op = optimiser.apply_gradients(grads_vars, global_step=global_step)
    
    def forward(self, state):
        feed_dict = {self.state:state}
        policy, value_extr, value_intr = self.sess.run([self.policy_distrib, self.Ve, self.Vi], feed_dict = feed_dict)
        return policy, value_extr, value_intr

    def backprop(self, state, R, a):
        feed_dict = {self.state : state, self.R : R, self.actions: a}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess


class ICM(object):
    def __init__(self, model_head, input_shape, action_size, reuse=False, **model_head_args):    
            
        print('input_Shape', input_shape)
        self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name="state")
        self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name="next_state")

        if len(input_shape) == 3: # image observation
            next_state_shape = input_shape[:-1] + (1,)
        else: 
            next_state_shape = input_shape

        self.state_mean = tf.placeholder(tf.float32, shape=[*next_state_shape], name="mean")
        self.state_std = tf.placeholder(tf.float32, shape=[*next_state_shape], name="std")
        norm_state = tf.clip_by_value((self.state - self.state_mean) / self.state_std, -5, 5)
        norm_next_state = tf.clip_by_value((self.next_state - self.state_mean) / self.state_std, -5, 5)
        
        self.action = tf.placeholder(tf.int32, shape=[None], name="actions")
        action_onehot = tf.one_hot(self.action, action_size)


        with tf.variable_scope('encoder_network', reuse=reuse):
            self.phi1 = model_head(norm_state,  **model_head_args)
        with tf.variable_scope('encoder_network', reuse=True):
            self.phi2 = model_head(norm_next_state,  **model_head_args)

        with tf.variable_scope('Forward_Model'):
            state_size = self.phi1.get_shape()[1].value
            concat = tf.concat([self.phi1, tf.dtypes.cast(action_onehot, tf.float32)], 1, name='state-action-concat')
            f1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='foward_model')
            pred_state = mlp_layer(f1, state_size, activation=None, name='pred_state')
            print('pred_state', pred_state.get_shape().as_list(), 'phi_2', self.phi2.get_shape().as_list())
            self.intr_reward = tf.reduce_mean(tf.square(pred_state - self.phi2), axis=1)# l2 distance metric ‖ˆφ(st+1)−φ(st+1)‖2
            self.forward_loss =  tf.reduce_mean(self.intr_reward) # intr_reward batch loss 
            print('forward_loss', self.forward_loss.get_shape().as_list())
        
        with tf.variable_scope('Inverse_Model'):
            concat = tf.concat([self.phi1, self.phi2], 1, name='state-nextstate-concat')
            i1 = mlp_layer(concat, state_size, activation=tf.nn.relu, name='inverse_model')
            pred_action = mlp_layer(i1, action_size, activation=None, name='pred_state')
            self.inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_action, labels=action_onehot)) # batch inverse loss
            print('pred_action', pred_action.get_shape().as_list(), 'inverse_loss', self.inverse_loss.get_shape().as_list())
        
        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        

class Curiosity(object):
    def __init__(self,  policy_model, ICM_model, input_size, action_size, forward_model_scale, policy_importance, reward_scale, intr_coeff, extr_coeff=1.0, value_coeff=0.5, entropy_coeff=0.001, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
        self.reward_scale, self.forward_model_scale, self.policy_importance = reward_scale, forward_model_scale, policy_importance
        self.intr_coeff, self.extr_coeff =  intr_coeff, extr_coeff
        self.value_coeff, self.entropy_coeff = value_coeff, entropy_coeff
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_size)
        except TypeError:
            input_size = (input_size,)
        
        with tf.variable_scope('Policy'):
            self.AC = ActorCritic(policy_model, input_size, action_size, intr_coeff=intr_coeff, extr_coeff=extr_coeff,
            value_coeff=value_coeff, entropy_coeff=entropy_coeff, lr=lr, lr_final=lr_final, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)

        with tf.name_scope('ICM'):
            self.ICM = ICM(ICM_model, input_size, action_size, **ICM_args)

        self.loss = policy_importance * self.AC.loss + reward_scale * ((1-forward_model_scale) * self.ICM.inverse_loss + forward_model_scale * self.ICM.forward_loss) 
        
        self.optimiser = tf.train.AdamOptimizer(lr)
        
        weights = self.AC.weights + self.ICM.weights
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)

    def forward(self, state):
        return self.AC.forward(state)
    
    def intrinsic_reward(self, state, action, next_state, state_mean, state_std):
        feed_dict={self.ICM.state:state, self.ICM.action:action, self.ICM.next_state:next_state,
        self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        intr_reward = self.sess.run(self.ICM.intr_reward, feed_dict=feed_dict )
        return intr_reward 
    
    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions, state_mean, state_std):
        feed_dict = {self.AC.state:state, self.AC.actions:actions,
                    self.AC.R_extr:R_extr, self.AC.R_intr:R_intr, self.AC.Advantage:Adv,
                    self.ICM.state:state, self.ICM.next_state:next_state, self.ICM.action:actions,
                    self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.AC.set_session(sess)


class Curiosity_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', return_type='GAE',
    total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes,log_scalars=log_scalars)
        self.runner = self.Runner(self.model, self.env, self.nsteps)
        self.return_type = return_type
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':model.entropy_coeff, 'value_coefficient':model.value_coeff, 'reward_scale':model.reward_scale,
        'forward_model_scale':model.forward_model_scale, 'policy_importance':model.policy_importance, 'intr_coeff':model.intr_coeff,
        'extr_coeff':model.extr_coeff, 'lambda':self.lambda_, 'gamma':self.gamma, 'return type':self.return_type}
    
        if self.log_scalars:
            filename = log_dir + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)

    def init_state_obs(self, num_steps):
        states = []
        for i in range(num_steps):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states.append(next_states)
            if i % self.nsteps == 0 and i > 0:
                self.runner.state_mean, self.runner.state_std = self.state_rolling.update(np.stack(states))
                states = []

    
    def _train_nstep(self):
        start = time.time()
        batch_size = (self.num_envs * self.nsteps)
        num_updates = self.total_steps // batch_size
        s = 0
        rolling = RunningMeanStd()
        self.state_rolling = rolling_obs(shape=())
        self.init_state_obs(128*50)
        self.runner.states = self.env.reset()
        forward_filter = RewardForwardFilter(self.gamma)
        # main loop
        for t in range(1,num_updates+1):
            start2 = time.time()
            states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, infos = self.runner.run()
            policy, last_extr_values, last_intr_values = self.model.forward(next_states[-1])
            print('rollout time', time.time() -start2)

            self.runner.state_mean, self.runner.state_std = self.state_rolling.update(next_states) # update state normalisation statistics 
            
            int_rff = np.array([forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))])
            R_intr_mean, R_intr_std = rolling.update(int_rff.ravel())
            intr_rewards /= R_intr_std
            
            if self.return_type == 'GAE':
                R_extr = self.GAE(extr_rewards, extr_values, last_extr_values, dones, gamma=0.999, lambda_=self.lambda_) + extr_values
                R_intr = self.GAE(intr_rewards, intr_values, last_intr_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) + intr_values
            else:
                R_extr = self.nstep_return(extr_rewards, last_extr_values, dones, gamma=0.999, clip=False)
                R_intr = self.nstep_return(intr_rewards, last_intr_values, np.zeros_like(dones), gamma=0.99, clip=False) # non episodic intr reward signal 
            
            Adv = self.model.extr_coeff * (R_extr - extr_values) + self.model.intr_coeff * (R_intr - intr_values)
                
            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R_extr, R_intr, Adv = fold_batch(states), fold_batch(next_states), fold_batch(actions), fold_batch(R_extr), fold_batch(R_intr), fold_batch(Adv)
            
            start2= time.time()
            l = self.model.backprop(states, next_states, R_extr, R_intr, Adv, actions, self.runner.state_mean, self.runner.state_std)
            print('backprop time', time.time() -start2)
            
            #start= time.time()
            if self.render_freq > 0 and t % ((self.validate_freq // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq //batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq //batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + '/' + str(s) + ".ckpt") )
                print('saved model')
            
            #print('validate time', time.time() -start)
    
    def get_action(self, state):
        policy, *values = self.model.forward(state)
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
                policies, extr_values, intr_values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
                intr_rewards = self.model.intrinsic_reward(self.states, actions, next_states, self.state_mean, self.state_std)
                #print('intr_rewards', self.model.intr_coeff * intr_rewards)
                rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, np.array(infos)))
                self.states = next_states
            
            states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, infos = stack_many(zip(*rollout))
            return states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, infos
            

def main(env_id, Atari=True):
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
        
        val_envs = [AtariEnv(gym.make(env_id), k=4, rescale=84, episodic=False, reset=reset, clip_reward=False) for i in range(16)]
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=False, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/CuriosityA2C_intr_value/' + env_id + '/'
    model_dir = "models/CuriosityA2C_intr_value/" + env_id + '/'


    model = Curiosity(nature_cnn,
                      nature_cnn,
                      input_size = input_size,
                      action_size = action_size,
                      forward_model_scale=0.2,
                      policy_importance = 1.0,
                      extr_coeff=2.0,
                      intr_coeff=1.0,
                      entropy_coeff=0.001,
                      value_coeff=0.5,
                      reward_scale=1.0,
                      lr=1e-4,
                      lr_final=1e-3,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={'scale':False,
                      'weight_initialiser':tf.initializers.orthogonal(np.sqrt(2))}) 

    

    curiosity = Curiosity_Trainer(envs = envs,
                                  model = model,
                                  file_loc = [model_dir, train_log_dir],
                                  val_envs = val_envs,
                                  train_mode = 'nstep',
                                  return_type = 'GAE',
                                  total_steps = 5e6,
                                  nsteps = nsteps,
                                  validate_freq = 1e5,
                                  save_freq = 0,
                                  render_freq = 0,
                                  num_val_episodes = 25,
                                  log_scalars=False)
    print(env_id)
    curiosity.train()

    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['MontezumaRevengeDeterministic-v4', 'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'PongDeterministic-v4', ]
    #env_id_list = ['Acrobot-v1', 'MountainCar-v0', 'CartPole-v1' ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    