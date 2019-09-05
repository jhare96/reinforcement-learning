import tensorflow as tf
import numpy as np
import scipy
import gym
import os
import time, datetime
import threading
from rlib.networks.networks import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.VecEnv import*
from rlib.utils.utils import stack_many, fold_batch, unfold_batch, one_hot, RunningMeanStd
from collections import OrderedDict
import matplotlib.pyplot as plt 
#from .OneNetCuriosity import Curiosity_onenet
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

class rolling_obs(object):
    def __init__(self, shape=(), lastFrame=False):
        self.rolling = RunningMeanStd(shape=shape)
        self.lastFrame = lastFrame
    
    def update(self, x):
        if self.lastFrame: # assume image obs 
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

class ICM(object):
    def __init__(self, model_head, state, input_shape, action_size, **model_head_args):    
            
        print('input_Shape', input_shape)
        #self.state = tf.placeholder(tf.float32, shape=[None, *input_shape], name="state")
        self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name="next_state")

        if len(input_shape) == 3: # image observation
            next_state_shape = input_shape[:-1] + (1,)
        else: 
            next_state_shape = input_shape

        self.state_mean = tf.placeholder(tf.float32, shape=[*next_state_shape], name="mean")
        self.state_std = tf.placeholder(tf.float32, shape=[*next_state_shape], name="std")
        norm_state = tf.clip_by_value((state - self.state_mean) / self.state_std, -5, 5)
        norm_next_state = tf.clip_by_value((self.next_state - self.state_mean) / self.state_std, -5, 5)
        
        self.action = tf.placeholder(tf.int32, shape=[None], name="actions")
        action_onehot = tf.one_hot(self.action, action_size)

        with tf.variable_scope('encoder_network', reuse=False):
            self.phi1 = model_head(norm_state,  **model_head_args)
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

class PPO(object):
    def __init__(self, model, input_shape, action_size, value_coeff=1.0, entropy_coeff=0.001, extr_coeff=2.0, intr_coeff=1.0, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5, optimiser=False, **model_args):
        self.lr, self.lr_final = lr, lr_final
        self.value_coeff, self.entropy_coeff = value_coeff, entropy_coeff
        self.decay_steps = decay_steps
        self.grad_clip = grad_clip
        self.policy_clip = 0.1
        self.sess = None
        with tf.variable_scope('encoder_network'):
            self.state = tf.placeholder(tf.float32, shape=[None, *input_shape])
            print('state shape', self.state.get_shape().as_list())
            self.dense = model(self.state, **model_args)
        
        with tf.variable_scope('extr_critic'):
            self.Ve = tf.reshape(mlp_layer(self.dense, 1, name='extr_value', activation=None), shape=[-1])
        
        with tf.variable_scope('intr_critic'):
            self.Vi = tf.reshape(mlp_layer(self.dense, 1, name='intr_value', activation=None), shape=[-1])
        
        with tf.variable_scope("actor"):
            self.policy_distrib = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution')+ 1e-10
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions,action_size)
            
        with tf.variable_scope('losses'):
            self.old_policy = tf.placeholder(dtype=tf.float32, shape=[None, action_size], name='old_policies') 
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

    def backprop(self, state, R_extr, R_intr, Adv, a, old_policy):
        feed_dict = {self.state : state, self.R_extr:R_extr, self.R_intr:R_intr,
                     self.Advantage:Adv, self.actions:a,
                     self.old_policy:old_policy}
        *_,l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess


class Curiosity(object):
    def __init__(self,  policy_model, ICM_model, input_shape, action_size, forward_model_scale, policy_importance, reward_scale, value_coeff, entropy_coeff, extr_coeff, intr_coeff, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, ICM_args={}):
        self.intr_coeff, self.extr_coeff =  intr_coeff, extr_coeff
        self.value_coeff, self.entropy_coeff = value_coeff, entropy_coeff
        self.reward_scale, self.forward_model_scale, self.policy_importance = reward_scale, forward_model_scale, policy_importance
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_shape = (input_shape,)
        
        with tf.variable_scope('Policy'):
            self.policy = PPO(policy_model,
                              input_shape,
                              action_size,
                              value_coeff=value_coeff,
                              entropy_coeff=entropy_coeff,
                              extr_coeff=extr_coeff,
                              intr_coeff=intr_coeff,
                              **policy_args)
        
        with tf.name_scope('ICM'):
            self.ICM = ICM(ICM_model, self.policy.state, input_shape, action_size, **ICM_args)

        self.loss = policy_importance * self.policy.loss + reward_scale * ((1-forward_model_scale) * self.ICM.inverse_loss + forward_model_scale * self.ICM.forward_loss ) 
        
        self.optimiser = tf.train.AdamOptimizer(lr)
        
        weights = self.policy.weights + self.ICM.weights
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)
    
    def update_old_policy(self):
        self.sess.run(self.update_weights)

    def forward(self, state):
        return self.policy.forward(state)
    
    def get_old_policy(self, state):
        return self.policy_old.forward(state)

    def intrinsic_reward(self, state, action, next_state, state_mean, state_std):
        feed_dict={self.policy.state:state, self.ICM.action:action, self.ICM.next_state:next_state,
                   self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        intr_reward = self.sess.run(self.ICM.intr_reward, feed_dict=feed_dict)
        return intr_reward
    
    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions, old_policy, state_mean, state_std):
        feed_dict = {self.policy.state:state, self.policy.actions:actions,
                    self.policy.R_extr:R_extr, self.policy.R_intr:R_intr, self.policy.Advantage:Adv,
                    self.ICM.next_state:next_state,
                    self.ICM.action:actions, self.policy.old_policy:old_policy,
                    self.ICM.state_mean:state_mean, self.ICM.state_std:state_std}
        _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return l
        
    
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)


class RND_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, num_epochs=4, num_minibatches=4, validate_freq=1000000.0,
                 save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars)
        self.runner = self.Runner(self.model, self.env, self.nsteps)
        self.alpha = 1
        self.pred_prob = 1 / (self.num_envs / 32.0)
        self.lambda_ = 0.95
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches
        hyper_paras = [['learning_rate', model.lr], ['learning_rate_final',model.lr_final], ['lr_decay_steps', model.decay_steps],
         ['grad_clip',model.grad_clip], ['nsteps',self.nsteps], ['num_workers',self.num_envs], ['total_steps',self.total_steps],
          ['entropy_coefficient',model.entropy_coeff], ['value_coefficient',model.value_coeff], ['intr_coeff',model.intr_coeff],
        ['extr_coeff',model.extr_coeff]]
        

        if log_scalars:
            hyper_paras = OrderedDict(hyper_paras)
            filename = log_dir + '/hyperparameters.txt'
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
    
    # def init_state_obs(self, num_steps):
    #     states = []
    #     for i in range(num_steps):
    #         rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
    #         next_states, rewards, dones, infos = self.env.step(rand_actions)
    #         states.append(next_states)
    #         if i % self.nsteps == 0 and i > 0:
    #             self.runner.state_mean, self.runner.state_std = self.state_rolling.update(np.stack(states))
    #             states = []

    def init_state_obs(self, num_steps):
        states = 0
        for i in range(1, num_steps+1):
            rand_actions = np.random.randint(0, self.model.action_size, size=self.num_envs)
            next_states, rewards, dones, infos = self.env.step(rand_actions)
            states += next_states
        mean = states / num_steps
        #var = (states - mean) / num_steps
        self.runner.state_mean, self.runner.state_std = self.state_rolling.update(mean.mean(axis=0)[None,None])
        plt.imshow(mean.mean(axis=0)[...,-1])
        plt.show()
        plt.imshow(self.runner.state_mean[:,:,0])
        plt.show()
    
    def _train_nstep(self):
        batch_size = (self.num_envs * self.nsteps)
        num_updates = self.total_steps //  batch_size
        s = 0
        rolling = RunningMeanStd(shape=())
        self.state_rolling = rolling_obs(shape=(), lastFrame=True)
        self.init_state_obs(128*50)
        self.runner.states = self.env.reset()
        forward_filter = RewardForwardFilter(self.gamma)

        # main loop
        start = time.time()
        for t in range(1,num_updates+1):
            states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, old_policies, dones = self.runner.run()
            policy, extr_last_values, intr_last_values = self.model.forward(next_states[-1])
            int_rff = np.array([forward_filter.update(intr_rewards[i]) for i in range(len(intr_rewards))])
            R_intr_mean, R_intr_std = rolling.update(int_rff.ravel())
            intr_rewards /= R_intr_std


            Adv_extr = self.GAE(extr_rewards, values_extr, extr_last_values, dones, gamma=0.999, lambda_=self.lambda_)
            Adv_intr = self.GAE(intr_rewards, values_intr, intr_last_values, np.zeros_like(dones), gamma=0.99, lambda_=self.lambda_) # non episodic intr reward signal 
            R_extr = Adv_extr + values_extr
            R_intr = Adv_intr + values_intr
            total_Adv = self.model.extr_coeff * Adv_extr + self.model.intr_coeff * Adv_intr

            self.runner.state_mean, self.runner.state_std = self.state_rolling.update(next_states) # update state normalisation statistics 

            # perform minibatch gradient descent for K epochs 
            l = 0
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                mini_batch_size = self.nsteps//self.num_minibatches
                np.random.shuffle(idxs)
                for batch in range(0,len(states), mini_batch_size):
                    batch_idxs = idxs[batch:batch + mini_batch_size]
                    # stack all states, next_states, actions and Rs across all workers into a single batch
                    mb_states, mb_nextstates, mb_actions, mb_Rextr, mb_Rintr, mb_Adv, mb_old_policies = fold_batch(states[batch_idxs]), fold_batch(next_states[batch_idxs]), \
                                                    fold_batch(actions[batch_idxs]), fold_batch(R_extr[batch_idxs]), fold_batch(R_intr[batch_idxs]), \
                                                    fold_batch(total_Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
                
                    #mb_nextstates = mb_nextstates[np.where(np.random.uniform(size=(batch_size)) < self.pred_prob)]
                    
                    mean, std = self.runner.state_mean, self.runner.state_std
                    l += self.model.backprop(mb_states, mb_nextstates, mb_Rextr, mb_Rintr, mb_Adv, mb_actions, mb_old_policies, mean, std)
            
            l /= (self.num_epochs * self.num_minibatches)
        
            if self.render_freq > 0 and t % ((self.validate_freq //batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
     
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size )== 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq  // batch_size) == 0:
                s += 1
                self.saver.save(self.sess, str(self.model_dir + '/' + str(s) + ".ckpt") )
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
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states, next_states, actions, extr_rewards, values_extr, values_intr, policies, dones))
                self.states = next_states

            states, next_states, actions, extr_rewards, values_extr, values_intr, policies, dones = stack_many(zip(*rollout))
            intr_rewards = self.model.intrinsic_reward(fold_batch(states), fold_batch(actions), fold_batch(next_states), self.state_mean, self.state_std)
            intr_rewards = unfold_batch(intr_rewards, self.num_steps, len(self.env))
            return states, next_states, actions, extr_rewards, intr_rewards, values_extr, values_intr, policies, dones
    
            

def main(env_id, Atari=True):
    num_envs = 32
    nsteps = 128

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
    
    
    current_time = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    train_log_dir = 'logs/Curiosity_IntrValue/' + env_id + '/' + current_time
    model_dir = "models/Curiosity_IntrValue/" + env_id + '/' + current_time

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = Curiosity(nature_cnn,
                      nature_cnn,
                      input_shape = input_size,
                      action_size = action_size,
                      forward_model_scale=0.2,
                      policy_importance=1,
                      reward_scale=1,
                      value_coeff=0.5,
                      entropy_coeff=0.001,
                      extr_coeff=2.0,
                      intr_coeff=1.0,
                      lr=1e-4,
                      decay_steps=50e6//(num_envs*nsteps),
                      grad_clip=0.5,
                      policy_args={},
                      ICM_args={'weight_initialiser':tf.initializers.orthogonal(np.sqrt(2)), 'scale':False }) 

    

    curiosity = RND_Trainer(envs = envs,
                            model = model,
                            model_dir = model_dir,
                            log_dir = train_log_dir,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    env_id_list = ['MontezumaRevengeDeterministic-v4']#  'SpaceInvadersDeterministic-v4', 'FreewayDeterministic-v4', 'PongDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', ]
    for i in range(1):
        for env_id in env_id_list:
            main(env_id)
    