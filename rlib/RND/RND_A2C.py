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
from rlib.utils.utils import fold_batch, one_hot, rolling_stats
#from .OneNetCuriosity import Curiosity_onenet

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


class RND(object):
    def __init__(self, policy_model, target_model, input_shape, action_size, policy_importance, reward_scale, intr_coeff=0.5, extr_coeff=1.0, value_coeff=0.5, entropy_coeff=0.001, lr=1e-3, lr_final=1e-3, decay_steps=6e5, grad_clip = 0.5, policy_args ={}, RND_args={}):
        self.reward_scale, self.policy_importance = reward_scale, policy_importance
        self.intr_coeff, self.extr_coeff =  intr_coeff, extr_coeff
        self.value_coeff, self.entropy_coeff = value_coeff, entropy_coeff
        self.lr, self.lr_final, self.decay_steps = lr, lr_final, decay_steps
        self.grad_clip = grad_clip
        self.action_size = action_size
        self.sess = None

        try:
            iterator = iter(input_shape)
        except TypeError:
            input_size = (input_shape,)
        

        with tf.variable_scope('Policy', reuse=tf.AUTO_REUSE):
            self.policy = ActorCritic(policy_model, input_shape, action_size, intr_coeff=intr_coeff, extr_coeff=extr_coeff, lr=lr, lr_final=lr_final, decay_steps=decay_steps, grad_clip=grad_clip, **policy_args)
            
        self.next_state = tf.placeholder(tf.float32, shape=[None, *input_shape], name='next_state')

        with tf.variable_scope('target_model'):
            target_dense = target_model(self.next_state, trainable=False, **RND_args)
            target_state = mlp_layer(target_dense, target_dense.get_shape()[-1].value, activation=None, name='target_state', trainable=False)
        
        with tf.variable_scope('predictor_model'):
            pred_dense = target_model(self.next_state, **RND_args)
            pred_next_state = mlp_layer(pred_dense, target_dense.get_shape()[-1].value, activation=None, name='pred_state')
            self.intr_reward = tf.reduce_mean(tf.square(pred_next_state - target_state), axis=-1)
            feat_loss = tf.reduce_mean(self.intr_reward)

        self.loss = policy_importance * self.policy.loss  + reward_scale * feat_loss
        
        self.update_weights = [tf.assign(new, old) for (new, old) in 
            zip(tf.trainable_variables(scope='Old_Policy'), tf.trainable_variables('Policy'))]


        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)
        self.optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)
        
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = tf.gradients(self.loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
        grads_vars = list(zip(grads, weights))

        self.train_op = self.optimiser.apply_gradients(grads_vars)

    def forward(self, state):
        return self.policy.forward(state)
    
    def intrinsic_reward(self, next_state):
        forward_loss = self.sess.run(self.intr_reward, feed_dict={self.next_state:next_state})
        intr_reward = forward_loss
        return intr_reward
    
    def backprop(self, state, next_state, R_extr, R_intr, Adv, actions):
        actions_onehot = one_hot(actions, self.action_size)
        feed_dict = {self.policy.state:state, self.policy.actions:actions, self.next_state:next_state,
                        self.policy.R_extr:R_extr, self.policy.R_intr:R_intr, self.policy.Advantage:Adv}

        _, l = self.sess.run([self.train_op,self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess
        self.policy.set_session(sess)


class RND_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, file_loc, val_envs, train_mode='nstep', total_steps=1000000, nsteps=5, validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True):
        super().__init__(envs, model, file_loc, val_envs, train_mode=train_mode, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes,log_scalars=log_scalars)
        self.runner = self.Runner(self.model, self.env, self.nsteps)
        
        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
         'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
          'entropy_coefficient':model.entropy_coeff, 'value_coefficient':model.value_coeff, 'reward_scale':model.reward_scale,
        'policy_importance':model.policy_importance, 'intr_coeff':model.intr_coeff,
        'extr_coeff':model.extr_coeff}
    
        if self.log_scalars:
            filename = file_loc[1] + self.current_time + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    def _train_nstep(self):
        
        start = time.time()
        num_updates = self.total_steps // (self.num_envs * self.nsteps)
        s = 0
        rolling = rolling_stats()
        # main loop
        for t in range(1,num_updates+1):
            #start = time.time()
            states, next_states, actions, extr_rewards, intr_rewards, dones, infos, extr_values, intr_values = self.runner.run()
            policy, last_extr_values, last_intr_values = self.model.forward(next_states[-1])
            R_extr = self.nstep_return(extr_rewards, last_extr_values, dones, gamma=0.99, clip=False)
            R_intr = self.nstep_return(intr_rewards, last_intr_values, np.zeros_like(dones), clip=False) # non episodic intr reward signal 

            # R_extr = self.GAE(extr_rewards, extr_values, last_extr_values, dones, gamma=0.999, lambda_=self.lambda_, clip=False) + extr_values
            # R_intr = self.GAE(intr_rewards, intr_values, last_intr_values, dones, gamma=0.99, lambda_=self.lambda_, clip=False) + intr_values
            R_mean, R_std = rolling.update(R_intr.ravel().mean(axis=0))
            self.runner.R_std = R_std
            
            Adv = self.model.intr_coeff * (R_intr - intr_values) + self.model.extr_coeff * (R_extr - extr_values)

            # stack all states, next_states, actions and Rs across all workers into a single batch
            states, next_states, actions, R_extr, R_intr, Adv = fold_batch(states), fold_batch(next_states), fold_batch(actions), fold_batch(R_extr), fold_batch(R_intr), fold_batch(Adv)
            
            
            l = self.model.backprop(states, next_states, R_extr, R_intr, Adv, actions)
            #print('backprop time', time.time() -start)
            
            #start= time.time()
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
            
            #print('validate time', time.time() -start)
    
    def get_action(self, state):
        policy, *values = self.model.forward(state)
        action = int(np.random.choice(policy.shape[1], p=policy[0]))
        #action = np.argmax(policy)
        return action

    class Runner(SyncMultiEnvTrainer.Runner):
        def __init__(self, model, env, num_steps):
            super().__init__(model, env, num_steps)
            self.R_std = 1

        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                start = time.time()
                policies, extr_values, intr_values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, extr_rewards, dones, infos = self.env.step(actions)
                intr_rewards = self.model.intrinsic_reward(next_states)
                #intr_mean, intr_std = self.rolling.update(intr_rewards)
                intr_rewards = intr_rewards / self.R_std
                #print('intr_std', self.R_std.shape)
                #print('intr_rewards', self.model.intr_coeff * intr_rewards)
                rollout.append((self.states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, infos))
                self.states = next_states
            
            states, next_states, actions, extr_rewards, intr_rewards, extr_values, intr_values, dones, infos = zip(*rollout)
            states, next_states, actions, dones = np.stack(states), np.stack(next_states), np.stack(actions), np.stack(dones)
            extr_rewards, intr_rewards, extr_values, intr_values = np.stack(extr_rewards), np.stack(intr_rewards), np.stack(extr_values), np.stack(intr_values)
            return states, next_states, actions, extr_rewards, intr_rewards, dones, infos, extr_values, intr_values
            

def main(env_id, Atari=True):


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
        envs = BatchEnv(AtariEnv, env_id, num_envs, blocking=False, rescale=84, k=4, reset=reset, episodic=True, clip_reward=True, time_limit=4500)
        
    
    env.close()
    action_size = val_envs[0].action_space.n
    input_size = val_envs[0].reset().shape
    
    

    train_log_dir = 'logs/RND_A2C/' + env_id + '/'
    model_dir = "models/RND_A2C/" + env_id + '/'

    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
   
    ac_mlp_args = {'dense_size':64}


    model = RND(nature_cnn,
                nature_cnn,
                input_shape = input_size,
                action_size = action_size,
                policy_importance = 1.0,
                extr_coeff=2.0,
                intr_coeff=1.0,
                entropy_coeff=0.001,
                value_coeff=0.5,
                reward_scale=1.0,
                lr=1e-3,
                lr_final=1e-3,
                decay_steps=50e6//(num_envs*nsteps),
                grad_clip=0.5,
                policy_args={},
                RND_args={'activation':tf.nn.leaky_relu}) #

    

    curiosity = RND_Trainer(envs = envs,
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
    env_id_list = ['FreewayDeterministic-v4', 'MontezumaRevengeDeterministic-v0', 'SpaceInvadersDeterministic-v4', 'PongDeterministic-v4', 'FreewayDeterministic-v4']
    #env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1' ]
    #for i in range(5):
    for env_id in env_id_list:
        main(env_id)
    