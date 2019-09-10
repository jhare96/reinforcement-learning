import tensorflow as tf
import numpy as np
import time, datetime
import gym

from rlib.networks.networks import *
from rlib.utils.VecEnv import*
from rlib.utils.SyncMultiEnvTrainer import SyncMultiEnvTrainer
from rlib.utils.utils import fold_batch

#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

class PPO(object):
    def __init__(self, model, input_shape, action_size, lr=1e-3, lr_final=0, decay_steps=6e5, grad_clip=0.5, value_coeff=1.0, entropy_coeff=0.01, name='PPO', **model_args):
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
            
            with tf.variable_scope('critic'):
                self.V = tf.reshape(mlp_layer(self.dense, 1, name='state_value', activation=None), shape=[-1])
            
            with tf.variable_scope("actor"):
                self.policy = mlp_layer(self.dense, action_size, activation=tf.nn.softmax, name='policy_distribution') + 1e-10
                self.actions = tf.placeholder(tf.int32, [None])
                actions_onehot = tf.one_hot(self.actions,action_size)
                
            with tf.variable_scope('losses'):
                self.old_policy = tf.placeholder(dtype=tf.float32, shape=[None, action_size], name='old_policies')
                self.alpha = tf.placeholder(dtype=tf.float32, shape=[], name='alpha')
                self.R = tf.placeholder(dtype=tf.float32, shape=[None])
                value_loss = 0.5 * tf.reduce_mean(tf.square(self.R - self.V))

                policy = tf.reduce_sum(tf.multiply(self.policy, actions_onehot), axis=1)
                old_policy = tf.reduce_sum(tf.multiply(self.old_policy, actions_onehot), axis=1)
                
                self.Advantage = tf.placeholder(dtype=tf.float32, shape=[None], name='Adv')

                ratio = policy / old_policy

                policy_loss_unclipped = ratio * -self.Advantage
                policy_loss_clipped = tf.clip_by_value(ratio, 1 - (self.policy_clip * self.alpha) , 1 + (self.policy_clip * self.alpha)) * -self.Advantage

                policy_loss = tf.reduce_mean(tf.math.maximum(policy_loss_unclipped, policy_loss_clipped))

                entropy = tf.reduce_mean(tf.reduce_sum(self.policy * -tf.math.log(self.policy), axis=1))
        
            
            #global_step = tf.Variable(0, trainable=False)
            #lr = tf.train.polynomial_decay(lr, global_step, decay_steps, end_learning_rate=lr_final, power=1.0, cycle=False, name=None)

            
            
            self.loss =  policy_loss + value_coeff * value_loss - entropy_coeff * entropy
            #optimiser = tf.train.AdamOptimizer(lr)
            optimiser = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-5)

            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, grad_clip)
            grads_vars = list(zip(grads, self.weights))
            self.optimiser = optimiser.apply_gradients(grads_vars)

    def forward(self, state):
        return self.sess.run([self.policy, self.V], feed_dict = {self.state:state})

    def get_policy(self, state):
        return self.sess.run(self.policy, feed_dict = {self.state: state})
    
    def get_value(self, state):
        return self.sess.run(self.V, feed_dict = {self.state: state})

    def backprop(self, state, R, Adv, a, old_policy, alpha):
        feed_dict = {self.state : state, self.R:R, self.Advantage:Adv, self.actions:a,
                     self.old_policy:old_policy, self.alpha:alpha}
        *_,l = self.sess.run([self.optimiser, self.loss], feed_dict=feed_dict)
        return l
    
    def set_session(self, sess):
        self.sess = sess




class PPO_Trainer(SyncMultiEnvTrainer):
    def __init__(self, envs, model, val_envs, train_mode='nstep', log_dir='logs/', model_dir='models/', total_steps=1000000, nsteps=5, num_epochs=4, num_minibatches=4,
                 validate_freq=1000000.0, save_freq=0, render_freq=0, num_val_episodes=50, log_scalars=True, gpu_growth=True):
        
        super().__init__(envs, model, val_envs, train_mode=train_mode, log_dir=log_dir, model_dir=model_dir, total_steps=total_steps, nsteps=nsteps, validate_freq=validate_freq,
                            save_freq=save_freq, render_freq=render_freq, update_target_freq=0, num_val_episodes=num_val_episodes, log_scalars=log_scalars,
                            gpu_growth=gpu_growth)

        self.runner = self.Runner(self.model, self.env, self.nsteps)
        #self.old_model = old_model
        #self.old_model.set_session(self.sess)
        self.alpha = 1
        self.lambda_ = 0.95
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches

        hyper_paras = {'learning_rate':model.lr, 'learning_rate_final':model.lr_final, 'lr_decay_steps':model.decay_steps,
            'grad_clip':model.grad_clip, 'nsteps':self.nsteps, 'num_workers':self.num_envs, 'total_steps':self.total_steps,
            'entropy_coefficient':0.01, 'value_coefficient':1.0}
        
        if log_scalars:
            filename = log_dir + '/hyperparameters.txt'
            self.save_hyperparameters(filename, **hyper_paras)
    
    
    
    def _train_nstep(self):
        batch_size = self.num_envs * self.nsteps
        num_updates = self.total_steps // batch_size
        alpha_step = 1/num_updates
        s = 0
        mini_batch_size = self.nsteps//self.num_minibatches
        start = time.time()
        # main loop
        for t in range(1,num_updates+1):
            states, actions, rewards, values, last_values, old_policies, dones, infos = self.runner.run()
            Adv = self.GAE(rewards, values, last_values, dones, gamma=0.99, lambda_=self.lambda_)
            R = Adv + values
            l = 0
            idxs = np.arange(len(states))
            for epoch in range(self.num_epochs):
                np.random.shuffle(idxs)
                for batch in range(0,len(states), mini_batch_size):
                    batch_idxs = idxs[batch:batch + mini_batch_size]
                    # stack all states, actions and Rs across all workers into a single batch
                    mb_states, mb_actions, mb_R, mb_Adv, mb_old_policies = fold_batch(states[batch_idxs]), \
                                                    fold_batch(actions[batch_idxs]), fold_batch(R[batch_idxs]), \
                                                    fold_batch(Adv[batch_idxs]), fold_batch(old_policies[batch_idxs])
                    
                    l += self.model.backprop(mb_states, mb_R, mb_Adv, mb_actions, mb_old_policies, self.alpha)
            
            l /= (self.num_epochs*self.num_minibatches)
           
            #self.alpha -= alpha_step
            
            if self.render_freq > 0 and t % ((self.validate_freq  // batch_size) * self.render_freq) == 0:
                render = True
            else:
                render = False
        
            if self.validate_freq > 0 and t % (self.validate_freq // batch_size) == 0:
                self.validation_summary(t,l,start,render)
                start = time.time()
            
            if self.save_freq > 0 and  t % (self.save_freq // batch_size) == 0:
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
        
        def run(self,):
            rollout = []
            for t in range(self.num_steps):
                policies, values = self.model.forward(self.states)
                actions = [np.random.choice(policies.shape[1], p=policies[i]) for i in range(policies.shape[0])]
                next_states, rewards, dones, infos = self.env.step(actions)
                rollout.append((self.states, actions, rewards, values, policies, dones, infos))
                self.states = next_states

            states, actions, rewards, values, policies, dones, infos = zip(*rollout)
            states, actions, rewards, values, policies, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(values), np.stack(policies), np.stack(dones)
            policy, last_values, = self.model.forward(next_states)
            return states, actions, rewards, values, last_values, policies, dones, infos   
    
    


def main(env_id, Atari=True):
    num_envs = 32
    nsteps = 512

    env = gym.make(env_id)
    #action_size = env.action_space.n
    #input_size = env.reset().shape[0]
    
    
    classic_list = ['MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2', 'CartPole-v0', 'CartPole-v1']
    if any(env_id in s for s in classic_list):
        print('Classic Control')
        val_envs = [gym.make(env_id) for i in range(16)]
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
    train_log_dir = 'logs/PPO/' + env_id + '/RMSprop/' + current_time
    model_dir = "models/PPO/" + env_id + '/' + current_time
    

    ac_cnn_args = {'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}

    ICM_mlp_args = { 'input_size':input_size, 'dense_size':4}

    ICM_cnn_args = {'input_size':[84,84,4], 'conv1_size':32, 'conv2_size':64, 'conv3_size':64, 'dense_size':512}
    
    
    
    ac_mlp_args = {'dense_size':64}

    
    model = PPO(mlp,
                input_shape = input_size,
                action_size = action_size,
                lr=1e-3,
                lr_final=1e-3,
                decay_steps=50e6//(num_envs*nsteps),
                grad_clip=0.5,
                value_coeff=0.5,
                entropy_coeff=1.0,
                name='Policy')
                 #'activation':tf.nn.leaky_relu
    

    curiosity = PPO_Trainer(envs = envs,
                            model = model,
                            model_dir = model_dir,
                            log_dir = train_log_dir,
                            val_envs = val_envs,
                            train_mode = 'nstep',
                            total_steps = 2e6,
                            nsteps = nsteps,
                            num_epochs=4,
                            num_minibatches=8,
                            validate_freq = 4e4,
                            save_freq = 0,
                            render_freq = 0,
                            num_val_episodes = 50,
                            log_scalars=False,
                            gpu_growth=False)
    curiosity.train()
    
    del curiosity

    tf.reset_default_graph()


if __name__ == "__main__":
    env_id_list = ['FreewayDeterministic-v4']# 'SpaceInvadersDeterministic-v4',]# , ]
    env_id_list = ['MountainCar-v0', 'Acrobot-v1', 'CartPole-v1', ]
    for i in range(5):
        for env_id in env_id_list:
            main(env_id)
            